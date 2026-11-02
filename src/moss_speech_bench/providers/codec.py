"""Run actual production flow/HiFT decode against a separate original HF oracle.

All raw samples and shapes are retained. A failed fixed numerical gate exits 1;
this standalone oracle never changes the strict HTTP C2 repeatability gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import random
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CORE = Path(os.environ["MOSS_CORE_ROOT"]).resolve()
REVISION = "eeec733e4e1dea7da444d332d8e1621ef257414c"
COMPARISON_POLICY = {
    "pcm_atol": 1e-5,
    "pcm_rtol": 1e-5,
    "equal_nan": False,
    "shapes": "exact",
    "dtypes": "exact",
    "code_ledger": "exact",
    "voice_inputs": "exact, complete supplied tensors",
    "cropping_or_padding": "none",
    "strict_http_c2_gate_changed": False,
    "thresholds_fixed_before_outputs": True,
}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timeout(signum: int, frame: object) -> None:
    raise TimeoutError("Production full-codec comparison exceeded its configured budget")


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_hashes(torch, np, device) -> dict:
    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate(), protocol=5)).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=5)).hexdigest(),
        "torch_cpu": tensor_hash(torch.get_rng_state()),
        "torch_cuda": tensor_hash(torch.cuda.get_rng_state(device)),
    }


def seed_case(torch, np, device) -> dict:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return rng_hashes(torch, np, device)


def scalar(value: float | int) -> float | int | str:
    return value if math.isfinite(value) else str(value)


def compare_tensor(torch, actual, expected, *, floating: bool) -> dict:
    actual = actual.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    result = {
        "actual_shape": list(actual.shape),
        "reference_shape": list(expected.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "actual_samples": actual.numel(),
        "reference_samples": expected.numel(),
        "same_shape": actual.shape == expected.shape,
        "same_dtype": actual.dtype == expected.dtype,
        "actual_finite": bool(torch.isfinite(actual).all().item()),
        "reference_finite": bool(torch.isfinite(expected).all().item()),
        "atol": COMPARISON_POLICY["pcm_atol"] if floating else 0,
        "rtol": COMPARISON_POLICY["pcm_rtol"] if floating else 0,
        "output_cropped": False,
        "output_padded": False,
    }
    if not result["same_shape"] or not result["same_dtype"]:
        result.update(
            passed=False,
            mismatched_elements=None,
            first_mismatches=[],
            numeric_comparison="not performed: exact shape/dtype gate failed",
        )
        return result
    else:
        pass
    exact = bool(torch.equal(actual, expected))
    if floating:
        close = torch.isclose(
            actual,
            expected,
            atol=COMPARISON_POLICY["pcm_atol"],
            rtol=COMPARISON_POLICY["pcm_rtol"],
            equal_nan=False,
        )
        numeric_pass = bool(
            torch.allclose(
                actual,
                expected,
                atol=COMPARISON_POLICY["pcm_atol"],
                rtol=COMPARISON_POLICY["pcm_rtol"],
                equal_nan=False,
            )
        )
    else:
        close = actual == expected
        numeric_pass = exact
    mismatch = (~close) | ~torch.isfinite(actual) | ~torch.isfinite(expected)
    positions = mismatch.nonzero().tolist()
    errors = (actual.to(torch.float64) - expected.to(torch.float64)).abs()
    finite_errors = errors[torch.isfinite(errors)]
    result.update(
        passed=numeric_pass and result["actual_finite"] and result["reference_finite"],
        exact_values_equal=exact,
        mismatched_elements=len(positions),
        first_mismatches=[
            {
                "index": position,
                "actual": scalar(actual[tuple(position)].item()),
                "reference": scalar(expected[tuple(position)].item()),
            }
            for position in positions[:20]
        ],
        max_abs_error=(float(finite_errors.max().item()) if finite_errors.numel() else None),
        mean_abs_error=(float(finite_errors.mean().item()) if finite_errors.numel() else None),
        numeric_comparison=("full-shape torch.allclose" if floating else "exact full tensor"),
    )
    return result


def production_sources() -> dict:
    sources = {}
    for name, module in tuple(sys.modules.items()):
        if name == "sglang_omni" or name.startswith("sglang_omni."):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                path = Path(origin).resolve()
                assert path.is_relative_to(CORE), (name, path)
                sources[name] = {"path": str(path), "sha256": digest(path)}
            else:
                assert all(Path(path).resolve().is_relative_to(CORE) for path in module.__path__)
        else:
            pass
    return sources


def forbid_reference_sources() -> list[str]:
    prefixes = ("cosyvoice", "matcha", "Matcha-TTS", "transformers_modules")
    imported = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )
    assert not imported, imported
    return imported


def run(args, reference: dict, report: dict, output_json: Path, tensor_path: Path) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file, save_file
    from sglang_omni.vendor.moss_speech_codec.hf_codec.modeling import (
        OFFLINE_FLOW_CHUNK_SIZE,
        AudioDecoder,
    )

    assert reference["status"] == "passed" and reference["numerical_reference_complete"] is True
    assert reference["serving_core_modules_imported"] == []
    assert reference["codec_revision"] == REVISION
    assert all(
        reference["pre_registered_comparison_policy"][key] == COMPARISON_POLICY[key]
        for key in (
            "pcm_atol",
            "pcm_rtol",
            "shapes",
            "code_ledger",
            "cropping_or_padding",
            "strict_http_c2_gate_changed",
        )
    )
    assert os.environ.get("PYTHONPATH") == str(CORE), (
        "Use only the production checkout as explicit PYTHONPATH"
    )
    source_path = Path(inspect.getfile(AudioDecoder)).resolve()
    assert source_path == CORE / "sglang_omni/vendor/moss_speech_codec/hf_codec/modeling.py"
    root = args.codec_path.resolve()
    assert root.name == REVISION
    assert digest(root / "flow/config.yaml") == reference["source_sha256"]["flow/config.yaml"]
    report["checkpoint_files"] = {
        name: {
            "bytes": (root / "flow" / name).stat().st_size,
            "sha256": digest(root / "flow" / name),
        }
        for name in ("flow.pt", "hift.pt", "campplus.onnx")
    }
    assert report["checkpoint_files"] == reference["checkpoint_files"]
    reference_tensor = args.reference_tensors or Path(reference["tensor_artifact"])
    assert digest(reference_tensor) == reference["tensor_artifact_sha256"]
    oracle = load_file(str(reference_tensor), device="cpu")
    report["reference_tensor_artifact"] = str(reference_tensor.resolve())
    report["reference_tensor_sha256"] = digest(reference_tensor)
    profile_path = Path(os.environ["MOSS_CODE_PROFILE"])
    assert digest(profile_path) == reference["code_profile_sha256"]
    profile = json.loads(profile_path.read_text())
    report["code_profile_json"] = str(profile_path.resolve())
    report["code_profile_sha256"] = digest(profile_path)
    report["production_source"] = production_sources()
    report["reference_source_modules_imported"] = forbid_reference_sources()
    voice = {
        name: oracle["voice." + name].clone()
        for name in ("prompt_token", "prompt_feat", "embedding")
    }
    prompt_count = reference["voice_conditioning"]["full_prompt_tokens"]
    assert voice["prompt_token"].shape == (1, prompt_count)
    assert voice["prompt_feat"].shape == (1, prompt_count * 4, 80)
    assert voice["embedding"].shape == (1, 192)
    assert voice["prompt_token"].dtype == torch.int64
    assert voice["prompt_feat"].dtype == voice["embedding"].dtype == torch.float32
    assert all(bool(torch.isfinite(tensor).all().item()) for tensor in voice.values())
    report["voice_conditioning"] = {
        **reference["voice_conditioning"],
        "actual_input_source": "Exact complete independent HF full-codec safetensors bundle",
        "additional_prompt_trimming": False,
        "actual_shapes": {key: list(value.shape) for key, value in voice.items()},
        "actual_dtypes": {key: str(value.dtype) for key, value in voice.items()},
    }
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    shared_flags = {
        "seed_after_constructor_and_before_each_case": 42,
        "algorithms_required": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    assert all(reference["determinism"][key] == value for key, value in shared_flags.items())
    assert reference["loading"]["sample_rate"] == 24000
    report["determinism"] = shared_flags
    report["reference_determinism"] = reference["determinism"]
    props = torch.cuda.get_device_properties(device)
    report["gpu"] = {
        "device": str(device),
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "cuda_runtime": torch.version.cuda,
    }
    before_constructor = rng_hashes(torch, np, device)
    report["stage"] = "actual-production-audio-decoder-constructor"
    write_json(output_json, report)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    bias_broadcast = os.environ.get("MOSS_CORRECTNESS_PROFILE") == "codec-bias-broadcast"
    decoder_options = {"broadcast_attention_bias": True} if bias_broadcast else {}
    decoder = AudioDecoder(
        root / "flow/flow.pt", root / "flow/hift.pt", device=device, **decoder_options
    ).eval()
    native_owner = None
    if os.environ.get("MOSS_CORRECTNESS_PROFILE") == "direct-lt":
        from sglang_omni.models.moss_speech.components.cfm_direct_lt import (
            FULL_LENGTHS,
            install_direct_lt,
        )

        native_owner = install_direct_lt(
            decoder.flow.decoder.estimator, os.environ["MOSS_DIRECT_LT_LIBRARY"]
        )
    torch.cuda.synchronize(device)
    after_constructor = rng_hashes(torch, np, device)
    assert not decoder.flow.training and not decoder.hift.training
    assert {parameter.dtype for parameter in decoder.parameters()} == {torch.float32}
    report["loading"] = {
        "seconds": time.perf_counter() - started,
        "class": f"{AudioDecoder.__module__}.{AudioDecoder.__name__}",
        "flow_parameter_count": sum(parameter.numel() for parameter in decoder.flow.parameters()),
        "hift_parameter_count": sum(parameter.numel() for parameter in decoder.hift.parameters()),
        "flow_training": decoder.flow.training,
        "hift_training": decoder.hift.training,
        "weight_dtype": "torch.float32",
        "loaded_once": True,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "rng_before_constructor_sha256": before_constructor,
        "rng_after_constructor_sha256": after_constructor,
        "constructor_rng_preserved": before_constructor == after_constructor,
        "constructor_rng_policy": "Production scopes constructor RNG; original YAML/CFM overwrite it. Both are explicitly reseeded42 after construction and before each decode.",
    }
    report["parameter_count_matches_reference"] = {
        key: report["loading"][key] == reference["loading"][key]
        for key in ("flow_parameter_count", "hift_parameter_count")
    }
    report["rng_after_constructor_reseed42_sha256"] = seed_case(torch, np, device)
    report["profile"] = {
        "path": "actual AudioDecoder.decode(audio_token, voice)",
        "dtype": "torch.float32",
        "mode": "offline, streaming=False, finalize=True",
        "offline_flow_chunk_size": OFFLINE_FLOW_CHUNK_SIZE,
        "estimator_static_chunk_size": decoder.flow.decoder.estimator.static_chunk_size,
        "num_decoding_left_chunks": decoder.flow.decoder.estimator.num_decoding_left_chunks,
        "cfm": {
            key: getattr(decoder.flow.decoder, key)
            for key in (
                "sigma_min",
                "t_scheduler",
                "training_cfg_rate",
                "inference_cfg_rate",
            )
        },
        "sample_rate": 24000,
        "voice_prompt_tokens": prompt_count,
        "checkpoint_loads": "one real AudioDecoder construction for all cases",
        "mel_output_observed": False,
    }
    assert OFFLINE_FLOW_CHUNK_SIZE == 5
    noise = decoder.flow.decoder.rand_noise
    noise_before = tensor_hash(noise)
    report["cfm_noise"] = {
        "shape": list(noise.shape),
        "dtype": str(noise.dtype),
        "sha256_before": noise_before,
        "reference_sha256_before": reference["cfm_noise_sha256_before"],
        "same_as_reference": noise_before == reference["cfm_noise_sha256_before"],
    }
    tensors = {"voice." + key: value.clone() for key, value in voice.items()}
    report["voice_input_comparisons"] = {
        key: compare_tensor(torch, tensors["voice." + key], oracle["voice." + key], floating=False)
        for key in voice
    }
    case_order = reference["case_order"]
    assert (
        case_order
        and len(case_order) == len(set(case_order))
        and set(case_order) == set(reference["cases"])
    )
    report["case_order"] = case_order
    for name in case_order:
        reference_case = reference["cases"][name]
        assert reference_case["status"] == "passed"
        count = reference_case["code_count"]
        assert isinstance(count, int) and count > 0 and name == f"codes{count}"
        codes = oracle[name + ".codes"].clone()
        assert codes.dtype == torch.int64 and codes.shape == (1, count)
        assert codes[0].tolist() == reference_case["complete_decode_code_ledger"]
        registered = profile["cases"][name]
        assert codes[0].tolist() == registered["complete_decode_code_ledger"]
        assert reference_case["source_indices"] == registered["source_indices"]
        assert len(reference_case["source_indices"]) == count
        assert (
            hashlib.sha256(
                json.dumps(codes[0].tolist(), separators=(",", ":")).encode()
            ).hexdigest()
            == reference_case["code_ledger_sha256"]
            == registered["code_ledger_sha256"]
        )
        assert (
            hashlib.sha256(
                json.dumps(reference_case["source_indices"], separators=(",", ":")).encode()
            ).hexdigest()
            == reference_case["source_indices_sha256"]
            == registered["source_indices_sha256"]
        )
        assert reference_case["no_eos_appended"] is True
        assert bool(((codes >= 0) & (codes < 16384)).all().item())
        before = seed_case(torch, np, device)
        case = {
            "status": "running",
            "complete_decode_code_ledger": codes[0].tolist(),
            "code_count": count,
            "codes_source": "Exact entire explicit ledger from independent full HF oracle; source provenance retained, no reconstruction or further selection",
            "reference_case_metadata": reference_case,
            "rng_before_case_sha256": before,
            "cpu_rng_matches_reference": before["torch_cpu"]
            == reference_case["cpu_rng_sha256_before"],
            "gpu_rng_matches_reference": before["torch_cuda"]
            == reference_case["gpu_rng_sha256_before"],
        }
        report["cases"][name] = case
        report["stage"] = "actual-production-decode-" + name
        write_json(output_json, report)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            waveform = decoder.decode(codes.to(device), voice)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        raw = waveform.detach().cpu().contiguous().clone()
        tensors.update({name + ".codes": codes.clone(), name + ".pcm": raw})
        comparison = compare_tensor(torch, raw, oracle[name + ".pcm"], floating=True)
        code_comparison = compare_tensor(
            torch, tensors[name + ".codes"], oracle[name + ".codes"], floating=False
        )
        noise_after = tensor_hash(noise)
        unchanged_voice = {
            key: compare_tensor(torch, voice[key], oracle["voice." + key], floating=False)
            for key in voice
        }
        case.update(
            status=(
                "passed"
                if comparison["passed"]
                and code_comparison["passed"]
                and all(value["passed"] for value in unchanged_voice.values())
                and noise_after == noise_before
                and case["cpu_rng_matches_reference"]
                and case["gpu_rng_matches_reference"]
                else "failed"
            ),
            forward_seconds=elapsed,
            raw_pcm_shape=list(raw.shape),
            raw_pcm_samples=raw.numel(),
            raw_pcm_dtype=str(raw.dtype),
            reference_raw_pcm_shape=reference_case["raw_pcm_shape"],
            reference_raw_pcm_samples=reference_case["raw_pcm_samples"],
            code_count_times_1920=count * 1920,
            raw_length_minus_code_budget=raw.numel() - count * 1920,
            output_cropped=False,
            output_padded=False,
            pcm_comparison=comparison,
            code_ledger_comparison=code_comparison,
            voice_inputs_after_decode=unchanged_voice,
            rng_after_case_sha256=rng_hashes(torch, np, device),
            cfm_noise_sha256_after=noise_after,
            cfm_noise_unchanged=noise_after == noise_before,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        )
        exported = {
            key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()
        }
        save_file(
            exported,
            str(tensor_path),
            metadata={
                "codec_revision": REVISION,
                "scope": "actual production AudioDecoder raw PCM; no cropping or padding",
                "seed": "42",
                "comparison_policy": json.dumps(COMPARISON_POLICY, sort_keys=True),
            },
        )
        report["tensor_artifact_sha256"] = digest(tensor_path)
        write_json(output_json, report)
        if raw.ndim == 2 and raw.shape[0] == 1:
            wav_path = args.output_dir / (name + ".float.wav")
            try:
                sf.write(str(wav_path), raw[0].numpy(), 24000, subtype="FLOAT")
                case.update(
                    wav_file=str(wav_path.resolve()),
                    wav_subtype="FLOAT",
                    wav_sha256=digest(wav_path),
                    wav_export_status="passed",
                )
            except Exception as error:
                case.update(
                    wav_export_status="failed",
                    wav_export_error=f"{type(error).__name__}: {error}",
                )
        else:
            case.update(
                wav_file=None,
                wav_export_status="not-applicable: raw shape is not [1,N]",
            )
        write_json(output_json, report)
        print(
            json.dumps(
                {
                    "event": "production-codec-case",
                    "case": name,
                    "status": case["status"],
                    "raw_shape": list(raw.shape),
                    "pcm_gate": comparison["passed"],
                }
            ),
            flush=True,
        )
    report["tensors"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in tensors.items()
    }
    report["production_source"] = production_sources()
    report["reference_source_modules_imported"] = forbid_reference_sources()
    report["failed_case_count"] = sum(
        case["status"] != "passed" for case in report["cases"].values()
    )
    report["numerical_comparison_complete"] = True
    report["status"] = (
        "passed"
        if report["failed_case_count"] == 0
        and report["cfm_noise"]["same_as_reference"]
        and all(report["parameter_count_matches_reference"].values())
        else "failed"
    )
    if native_owner is not None:
        native_owner.close()
        report["direct_lt"] = dict(
            closed=native_owner.closed,
            calls=native_owner.calls,
            installed=native_owner.installed,
            qualified_lengths=list(FULL_LENGTHS),
        )
    if bias_broadcast:
        processors = decoder.bias_broadcast_processors
        report["bias_broadcast"] = dict(
            installed=len(processors),
            calls=[p.broadcast_calls for p in processors],
            head_strides=[p.last_head_stride for p in processors],
            fallback_calls=[p.fallback_calls for p in processors],
        )
        if __package__:
            from .codec_bias_broadcast import validate_qualification
        else:
            from codec_bias_broadcast import validate_qualification
        validate_qualification(report["bias_broadcast"])
    report["stage"] = "complete-production-full-codec-comparison"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-path", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--reference-tensors", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=int, default=840)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 840:
        parser.error("max-seconds must be 1..840")
    else:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "core-full-codec-comparison.json"
    tensor_path = args.output_dir / "core-full-codec-comparison.safetensors"
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
        "scipy",
        "librosa",
        "safetensors",
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
        "numerical_comparison_complete": False,
        "codec_revision": REVISION,
        "codec_path": str(args.codec_path.resolve()),
        "probe_sha256": digest(Path(__file__)),
        "pre_registered_comparison_policy": COMPARISON_POLICY,
        "package_versions": versions,
        "reference_environment_versions": reference.get("package_versions"),
        "python": sys.version,
        "executable": sys.executable,
        "argv": sys.argv,
        "core_commit": subprocess.check_output(
            ["git", "-C", str(CORE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "core_worktree_status": subprocess.check_output(
            ["git", "-C", str(CORE), "status", "--porcelain"], text=True
        ),
        "reference_json": str(args.reference_json.resolve()),
        "reference_json_sha256": digest(args.reference_json),
        "source_profile_sha256": {
            "flow/config.yaml": digest(args.codec_path.resolve() / "flow/config.yaml")
        },
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "case_order": reference.get("case_order", []),
        "cases": {},
        "tensor_artifact": str(tensor_path.resolve()),
        "limitations": [
            "Standalone independent original-versus-production flow/HiFT numerical comparison only; no HTTP concurrency or quality claim.",
            "Strict HTTP C2 exact PCM failure, if observed, remains a separate failed gate and is not relaxed by this tolerance policy.",
            "All raw tensor lengths and values retained, including any 511/512 boundary behavior; no crop, pad, alignment or tolerance retuning.",
            "511/512 cases are preregistered synthetic cyclic code sequences, not model-generated golden completions.",
            "Only tensor-code decoder APIs are exercised; higher-level processor/string last-row handling is not covered.",
            "Production decode returns PCM only; original mel is not claimed to have been compared.",
            "Timings and memory are execution metadata, not a calibrated performance benchmark.",
        ],
    }
    write_json(output_json, report)
    signal.signal(signal.SIGALRM, timeout)
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
        for name in report["case_order"]:
            if name not in report["cases"]:
                report["cases"][name] = {
                    "status": "not-run",
                    "reason": "Earlier probe exception; see top-level error",
                }
            elif report["cases"][name]["status"] == "running":
                report["cases"][name].update(
                    status="failed", error=f"{type(error).__name__}: {error}"
                )
            else:
                pass
    finally:
        signal.alarm(0)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["total_seconds"] = time.perf_counter() - started
        write_json(output_json, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": {name: value["status"] for name, value in report["cases"].items()},
                "report": str(output_json),
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
