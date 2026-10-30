"""Compare the production codec encoder/voice adapter with a separate HF export.

Authoring this file does not run it. The parent owns GPU execution and its budget.
No original HF code is imported by this process; the oracle is JSON+safetensors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CORE = Path(os.environ["MOSS_CORE_ROOT"]).resolve()
COMPARISON_POLICY = {
    "fixed_before_outputs": True,
    "codes": "Identical shape and integer values; no EOS insertion, trimming, alignment, or code selection.",
    "voice_prompt_token": "Identical shape and integer values; normalize reference int64 and production int32 to int64 for comparison only.",
    "floating_features": "Identical float32 dtype and shape; finite values required.",
    "atol": 1e-5,
    "rtol": 1e-5,
    "equal_nan": False,
    "failure_policy": "Record differences and fail; never retune thresholds from observed outputs.",
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def alarm_handler(signum: int, frame: object) -> None:
    raise TimeoutError("Production encoder/feature comparison exceeded its time budget")


def json_scalar(value: int | float) -> int | float | str:
    return value if math.isfinite(value) else str(value)


def tensor_comparison(torch, actual, expected, *, integer: bool) -> dict:
    actual = actual.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    record = {
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "same_shape": actual.shape == expected.shape,
        "actual_finite": bool(torch.isfinite(actual).all().item()),
        "expected_finite": bool(torch.isfinite(expected).all().item()),
        "comparison": (
            "exact integer values" if integer else "torch.allclose with preregistered tolerances"
        ),
    }
    if integer:
        record["accepted_dtypes"] = (
            actual.dtype in (torch.int32, torch.int64) and expected.dtype == torch.int64
        )
    else:
        record["accepted_dtypes"] = actual.dtype == expected.dtype == torch.float32
    if not record["same_shape"]:
        record.update(passed=False, mismatch_count=None, first_mismatches=[])
        return record
    else:
        pass
    left = actual.to(torch.float64)
    right = expected.to(torch.float64)
    errors = (left - right).abs()
    if integer:
        matches = actual.to(torch.int64) == expected.to(torch.int64)
        numeric_pass = bool(matches.all().item())
    else:
        matches = errors <= COMPARISON_POLICY["atol"] + COMPARISON_POLICY["rtol"] * right.abs()
        numeric_pass = record["accepted_dtypes"] and bool(
            torch.allclose(
                actual,
                expected,
                atol=COMPARISON_POLICY["atol"],
                rtol=COMPARISON_POLICY["rtol"],
                equal_nan=False,
            )
        )
    mismatch = (~matches) | ~torch.isfinite(left) | ~torch.isfinite(right)
    indices = mismatch.nonzero().tolist()
    finite_errors = errors[torch.isfinite(errors)]
    record.update(
        passed=record["accepted_dtypes"]
        and record["actual_finite"]
        and record["expected_finite"]
        and numeric_pass,
        exact_values_equal=(
            bool(torch.equal(actual.to(torch.int64), expected.to(torch.int64)))
            if integer
            else bool(torch.equal(actual, expected))
        ),
        mismatch_count=len(indices),
        first_mismatches=[
            {
                "index": index,
                "actual": json_scalar(actual[tuple(index)].item()),
                "expected": json_scalar(expected[tuple(index)].item()),
            }
            for index in indices[:20]
        ],
        max_abs_error=(float(finite_errors.max().item()) if finite_errors.numel() else None),
        mean_abs_error=(float(finite_errors.mean().item()) if finite_errors.numel() else None),
    )
    return record


def run(args, reference: dict, report: dict, output_json: Path, tensor_path: Path) -> None:
    import numpy as np
    import soundfile
    import torch
    from safetensors.torch import load_file, save_file
    from sglang_omni.models.moss_speech.components.codec_adapter import (
        MossSpeechCodecAdapter,
    )
    from sglang_omni.models.moss_speech.components.voice import MEL_PARAMS

    assert reference["status"] == "passed" and reference["voice_features"]["status"] == "passed"
    assert reference["serving_core_modules_imported"] == []
    root = args.codec_path.resolve()
    assert root.name == reference["codec_revision"]
    report["codec_path"] = str(root)
    reference_tensors_path = args.reference_tensors or Path(reference["tensor_artifact"])
    assert digest(reference_tensors_path) == reference["tensor_artifact_sha256"]
    oracle = load_file(str(reference_tensors_path), device="cpu")
    assert all(
        torch.equal(
            oracle[f"{name}.codes"],
            torch.tensor([reference["encoder_cases"][name]["codes"]], dtype=torch.int64),
        )
        for name in ("en", "cn", "voice")
    )
    adapter_source = Path(inspect.getfile(MossSpeechCodecAdapter)).resolve()
    assert adapter_source == CORE / "sglang_omni/models/moss_speech/components/codec_adapter.py"
    source_paths = {
        name: Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name.startswith("sglang_omni.") and getattr(module, "__file__", None) is not None
    }
    assert all(path.is_relative_to(CORE) for path in source_paths.values())
    assert MEL_PARAMS == reference["voice_features"]["feature_parameters"]
    assert digest(root / "flow/campplus.onnx") == reference["voice_features"]["campplus_sha256"]
    report["production_source"] = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in sorted(source_paths.items())
    }
    report["assets"] = {
        name: {"bytes": (root / name).stat().st_size, "sha256": digest(root / name)}
        for name in (
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "flow/campplus.onnx",
        )
    }
    assert report["assets"]["model.safetensors"]["bytes"] == reference["codec_checkpoint"]["bytes"]
    for name in ("config.json", "preprocessor_config.json"):
        assert report["assets"][name]["sha256"] == reference["source_sha256"][name]
    report["feature_parameters"] = MEL_PARAMS
    report["reference_tensor_artifact"] = str(reference_tensors_path.resolve())
    report["reference_tensor_sha256"] = digest(reference_tensors_path)
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    expected_determinism = {
        "seed": args.seed,
        "algorithms_required": True,
        "tf32": False,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    assert expected_determinism == reference["determinism"]
    report["determinism"] = expected_determinism
    properties = torch.cuda.get_device_properties(device)
    report["gpu"] = {
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda,
    }
    waves = {}
    for name in ("en", "cn", "voice"):
        expected_audio = reference["audio_inputs"][name]
        path = Path(os.environ["MOSS_AUDIO_" + name.upper()])
        assert digest(path) == expected_audio["sha256"], name
        array, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(array.copy()).T.contiguous()
        metadata = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "sample_rate": rate,
            "frames": waveform.shape[1],
            "channels": waveform.shape[0],
            "decode": "soundfile float32; production tuple(waveform, sample_rate) API",
        }
        assert all(
            metadata[key] == expected_audio[key]
            for key in ("sha256", "sample_rate", "frames", "channels")
        ), name
        waves[name] = (waveform, rate)
        report["audio_inputs"][name] = metadata
    report["stage"] = "loading-real-production-encoder"
    write_json(output_json, report)
    tensors = {}
    with MossSpeechCodecAdapter(
        root, device=device, dtype=torch.float32, load_encoder=True, load_decoder=False
    ) as adapter:
        assert adapter.encoder_loaded and not adapter.decoder_loaded
        assert {parameter.dtype for parameter in adapter.encoder.parameters()} == {torch.float32}
        report["adapter"] = {
            "class": f"{MossSpeechCodecAdapter.__module__}.{MossSpeechCodecAdapter.__name__}",
            "encoder_loaded": adapter.encoder_loaded,
            "decoder_loaded": adapter.decoder_loaded,
            "weight_dtype": "torch.float32",
            "speaker_session_providers": adapter.speaker_encoder.session.get_providers(),
        }
        for name in ("en", "cn"):
            started = time.perf_counter()
            with torch.inference_mode():
                codes = adapter.encode([waves[name]], batch_size=1)[0]
            torch.cuda.synchronize(device)
            assert codes and all(isinstance(code, int) and 0 <= code < 16384 for code in codes)
            tensors[f"{name}.codes"] = torch.tensor([codes], dtype=torch.int64)
            report["encoder_cases"][name] = {
                "codes": codes,
                "code_count": len(codes),
                "seconds": time.perf_counter() - started,
                "input_sha256": report["audio_inputs"][name]["sha256"],
            }
            report["comparisons"][f"{name}.codes"] = tensor_comparison(
                torch, tensors[f"{name}.codes"], oracle[f"{name}.codes"], integer=True
            )
            report["stage"] = f"encoded-{name}"
            save_file(tensors, str(tensor_path))
            write_json(output_json, report)
        started = time.perf_counter()
        with torch.inference_mode():
            voice = adapter.encode_voice_ref(waves["voice"])
        torch.cuda.synchronize(device)
        tensors.update(
            {
                f"voice.{key}": value.detach().cpu().contiguous()
                for key, value in voice.to_dict().items()
            }
        )
        report["voice_features"] = {
            "meta": voice.meta,
            "seconds": time.perf_counter() - started,
            "full_input_frames": report["audio_inputs"]["voice"]["frames"],
            "full_input_sample_rate": report["audio_inputs"]["voice"]["sample_rate"],
            "input_trimmed": False,
            "encoder_api": "actual MossSpeechCodecAdapter.encode_voice_ref",
            "decoder_loaded": adapter.decoder_loaded,
        }
        assert voice.meta["revision"] == reference["codec_revision"]
        report["voice_features"]["aligned_token_count_match"] = (
            voice.meta["token_len"] == reference["voice_features"]["aligned_prompt_tokens"]
        )
        for key in ("voice.prompt_token", "voice.prompt_feat", "voice.embedding"):
            report["comparisons"][key] = tensor_comparison(
                torch, tensors[key], oracle[key], integer=key == "voice.prompt_token"
            )
    assert adapter.is_closed and not adapter.encoder_loaded and not adapter.decoder_loaded
    report["adapter_closed"] = True
    save_file(
        tensors,
        str(tensor_path),
        metadata={
            "scope": "real production encoder/voice outputs; no waveform or quality claim",
            "comparison_policy": json.dumps(COMPARISON_POLICY, sort_keys=True),
        },
    )
    report["tensors"] = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in tensors.items()
    }
    report["tensor_artifact"] = str(tensor_path.resolve())
    report["tensor_artifact_sha256"] = digest(tensor_path)
    report["reference_code_modules_imported"] = sorted(
        name
        for name in sys.modules
        if name.startswith("transformers_modules.") or name == "audit_pinned_matcha_audio"
    )
    assert not report["reference_code_modules_imported"]
    report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["failed_comparison_count"] = sum(
        not record["passed"] for record in report["comparisons"].values()
    )
    assert report["failed_comparison_count"] == 0, (
        "Independent encoder/voice comparison failed; inspect recorded differences"
    )
    report.update(status="passed", stage="complete-encoder-and-features-comparison-only")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--reference-tensors", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--max-seconds", type=int, default=840)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 840:
        parser.error("max-seconds must be 1..840")
    else:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "core-encoder-features.json"
    tensor_path = args.output_dir / "core-encoder-features.safetensors"
    if output_json.exists() or tensor_path.exists():
        parser.error("Choose a fresh output directory")
    else:
        pass
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    reference = json.loads(args.reference_json.read_text())
    versions = {}
    for name in (
        "torch",
        "torchaudio",
        "transformers",
        "numpy",
        "soundfile",
        "librosa",
        "scipy",
        "safetensors",
        "onnxruntime-gpu",
        "sglang",
        "sglang-omni",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stage": "initializing",
        "comparison_policy": COMPARISON_POLICY,
        "probe_sha256": digest(Path(__file__)),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "argv": sys.argv,
        "package_versions": versions,
        "reference_environment_versions": reference.get("package_versions"),
        "reference_json": str(args.reference_json.resolve()),
        "reference_json_sha256": digest(args.reference_json),
        "codec_revision": reference.get("codec_revision"),
        "core_commit": subprocess.check_output(
            ["git", "-C", str(CORE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "core_worktree_status": subprocess.check_output(
            ["git", "-C", str(CORE), "status", "--porcelain"], text=True
        ),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "audio_inputs": {},
        "encoder_cases": {},
        "voice_features": {},
        "comparisons": {},
        "limitations": [
            "Real encoder codes and voice-conditioning features only; no flow/HiFT decoding or waveform quality claim.",
            "Both processes consume soundfile float32 waveform tuples; no original TorchCodec path-loader claim.",
            "Reference prompt int64 versus production int32 is a predeclared dtype adaptation; integer shapes and values remain exact requirements.",
            "Timing is execution evidence, not a controlled performance benchmark.",
        ],
    }
    write_json(output_json, report)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.max_seconds)
    started = time.perf_counter()
    try:
        run(args, reference, report, output_json, tensor_path)
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["total_seconds"] = time.perf_counter() - started
        write_json(output_json, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "comparison_count": len(report["comparisons"]),
                "failed_comparison_count": report.get("failed_comparison_count"),
                "output": str(output_json),
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
