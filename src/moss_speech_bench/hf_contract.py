"""Backend-independent identities, frozen HF contracts and numerical assertions."""

import hashlib
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from .providers.allocation import local_devices
from .service import service_config
from .storage import read_json

PROTOCOL = "hf-regression-v1"

GATE_NAMES = {
    "C00": "身份/资产/HF来源",
    "C01": "输入网格 / HF",
    "C02": "完整词表prefill / HF",
    "C03": "cached decode/B1/B2/B4 / HF",
    "C04": "greedy生成 / HF",
    "C05": "参考音色Encoder / HF",
    "C06": "固定codes Decoder / HF",
    "C07": "能力/协议/错误恢复",
    "C08": "请求重放与并发隔离",
}


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    root = (path.parent / config["workspace_root"]).resolve()
    config["workspace_root"] = str(root)
    for field in (
        "core_root",
        "candidate_python",
        "reference_python",
        "model_path",
        "codec_path",
        "voice_wav",
        "hf_source_root",
        "reference",
    ):
        value = root / config[field].replace("{workspace_root}", str(root))
        config[field] = (
            os.path.abspath(value) if field.endswith("_python") else str(value.resolve())
        )
    for asset in config["assets"]:
        asset["path"] = str((root / asset["path"].replace("{workspace_root}", str(root))).resolve())
        if not re.fullmatch("[a-f0-9]{64}", asset["sha256"]):
            raise ValueError("Each asset requires a fixed SHA256")
    asset_ids = [a["id"] for a in config["assets"]]
    if not asset_ids or len(set(asset_ids)) != len(asset_ids):
        raise ValueError("Asset identities must be nonempty and unique")
    if config.get("protocol") != PROTOCOL or config.get("adapter") != "sglang-native-moss-v1":
        raise ValueError("Unknown HF regression protocol/backend adapter")
    if not re.fullmatch("[a-f0-9]{40}", config["core_sha"]):
        raise ValueError("Candidate requires a full commit SHA")
    if config.get("tolerances") != {
        "logits_atol": 0.25,
        "logits_rtol": 0.02,
        "codec_atol": 1e-5,
        "codec_rtol": 1e-5,
    }:
        raise ValueError("HF regression v1 fixes tolerances; changes require a new protocol")
    if config["service"]["tasks"] != ["t2s"]:
        raise ValueError(
            "Native MOSS v1 adapter currently qualifies t2s; other APIs need an adapter"
        )
    config["service"]["reference_audio"] = config["voice_wav"]
    service_config(config["service"], root, False)
    ids = [case["id"] for case in config["cases"]]
    if (
        len(ids) < 4
        or len(ids) != len(set(ids))
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids)
    ):
        raise ValueError("At least four unique, path-safe reference cases required")
    if not {case["language"] for case in config["cases"]} >= {"en", "zh"} or any(
        not c["text"].strip() for c in config["cases"]
    ):
        raise ValueError("Reference requires nonempty English and Chinese inputs")
    sizes = config["batch_sizes"]
    if (
        not sizes
        or sizes != sorted(set(sizes))
        or sizes[0] != 1
        or any(type(n) is not int or n <= 0 for n in sizes)
        or max(sizes) > config["runtime"]["max_running_requests"]
    ):
        raise ValueError(
            "Batch probes require unique positive sizes including one, within declared capacity"
        )
    if (
        not isinstance(config["runtime"]["attention_backend"], str)
        or type(config["runtime"]["disable_cuda_graph"]) is not bool
    ):
        raise ValueError("Declare the actual attention backend and graph policy")
    if not config["teacher_rows"] or not 2 <= config["greedy_steps"] <= 64:
        raise ValueError("Cached teacher-forcing and greedy traces are required")
    if (
        config["codec_code_counts"] != [24, 32, 511, 512]
        or not config["codec_codes"]
        or any(type(c) is not int or not 0 <= c < 16384 for c in config["codec_codes"])
    ):
        raise ValueError("Fixed Decoder boundary vectors are required")
    local_devices(config)
    return config


def oracle_binding(config):
    return {
        name: config[name]
        for name in (
            "protocol",
            "cases",
            "teacher_rows",
            "greedy_steps",
            "sampling",
            "codec_codes",
            "codec_code_counts",
            "tolerances",
        )
    } | {
        "reference_mode": "official-hf-cached-repeat-and-fixed-decoder-noise-v1",
        "system": config["service"]["system"],
        "assets": {a["id"]: a["sha256"] for a in config["assets"]},
    }


def deployment_binding(config):
    """Require the operator's actual deployment to match the probed compute path."""
    return dict(
        runtime=config["runtime"],
        ar_dtype="bfloat16",
        codec_dtype="float32",
        gpu_memory_fraction=config["gpu_memory_fraction"],
        assets={a["id"]: a["sha256"] for a in config["assets"]},
    )


def validate_assets(config):
    pinned = {Path(a["path"]).resolve(): a for a in config["assets"]}
    for asset in config["assets"]:
        assert file_hash(asset["path"]) == asset["sha256"], "Asset changed: " + asset["id"]
    model = Path(config["model_path"])
    index = read_json(model / "model.safetensors.index.json")
    required = [model / name for name in set(index["weight_map"].values())]
    required += [
        model / name
        for name in (
            "modeling_moss_speech.py",
            "configuration_moss_speech.py",
            "processing_moss_speech.py",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
        )
    ]
    codec = Path(config["codec_path"])
    required += [
        codec / name
        for name in (
            "model.safetensors",
            "modeling_moss_speech_codec.py",
            "modeling_whisper.py",
            "flow/flow.pt",
            "flow/hift.pt",
            "flow/campplus.onnx",
            "flow/config.yaml",
        )
    ]
    required += [Path(config["voice_wav"])]
    required += list(Path(config["hf_source_root"]).rglob("*.py"))
    assert all(path.resolve() in pinned for path in required), (
        "Unpinned model, codec, voice or external source"
    )


def reference(config):
    path = Path(config["reference"])
    manifest = read_json(path)
    assert manifest["protocol"] == PROTOCOL and manifest["binding"] == oracle_binding(config), (
        "HF reference contract changed"
    )
    assert set(manifest["files"]) == {
        "ar.json",
        "ar.safetensors",
        "codec.json",
        "codec.safetensors",
        "execution/hf_oracle.py",
        "execution/noise_tape.py",
    }
    for name, expected in manifest["files"].items():
        assert file_hash(path.parent / name) == expected, "HF reference artifact changed: " + name
    for phase in ("ar", "codec"):
        report = read_json(path.parent / (phase + ".json"))
        assert report["status"] == "complete" and report["serving_modules_imported"] == []
        assert report["exporter_sha256"] == manifest["files"]["execution/hf_oracle.py"]
        assert report["noise_helper_sha256"] == manifest["files"]["execution/noise_tape.py"]
    return path.parent


def preflight(config):
    core = config["core_root"]
    assert (
        subprocess.check_output(["git", "-C", core, "rev-parse", "HEAD"], text=True).strip()
        == config["core_sha"]
    )
    assert not subprocess.check_output(
        ["git", "-C", core, "status", "--porcelain"], text=True
    ).strip(), "Candidate worktree must be clean"
    validate_assets(config)
    bundle = reference(config)
    cached = load_file(bundle / "ar.safetensors")
    checks = []
    recompute_diagnostics = []
    for case in config["cases"]:
        for step in range(len(config["teacher_rows"]) + 1):
            for head in ("text", "audio"):
                checks.append(
                    compare(
                        cached[f"teacher/{case['id']}/{step}/{head}"],
                        cached[f"repeat/{case['id']}/{step}/{head}"],
                        0,
                        0,
                        top1=False,
                        id=f"HF-cache-{case['id']}-{step}-{head}",
                    )
                )
                recompute_diagnostics.append(
                    compare(
                        cached[f"teacher/{case['id']}/{step}/{head}"],
                        cached[f"uncached/{case['id']}/{step}/{head}"],
                        0.25,
                        0.02,
                        id=f"HF-recompute-{case['id']}-{step}-{head}",
                    )
                )
    checks.append(dict(id="identity-assets-independent-oracle", status="pass"))
    return dict(
        status="pass" if all(c["status"] == "pass" for c in checks) else "fail",
        checks=checks,
        full_prefix_recompute_diagnostics=recompute_diagnostics,
        identity=dict(
            core_sha=config["core_sha"],
            core_branch=subprocess.check_output(
                ["git", "-C", core, "branch", "--show-current"], text=True
            ).strip(),
            core_dirty=False,
        ),
        reference_sha256=file_hash(config["reference"]),
    )


def compare(actual, expected, atol, rtol, *, top1=False, id):
    shape = actual.shape == expected.shape
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64)) if shape else None
    integer = np.issubdtype(expected.dtype, np.integer)
    same_dtype = actual.dtype == expected.dtype
    close = (
        bool(np.array_equal(actual, expected))
        if integer
        else bool(shape and finite and np.allclose(actual, expected, atol=atol, rtol=rtol))
    )
    argmax = bool(shape and int(actual.argmax()) == int(expected.argmax())) if top1 else None
    return dict(
        id=id,
        status="pass"
        if shape and same_dtype and finite and close and (not top1 or argmax)
        else "fail",
        shape_equal=shape,
        dtype_equal=same_dtype,
        finite=finite,
        allclose=close,
        top1_equal=argmax,
        maximum_absolute_error=float(delta.max()) if delta is not None and delta.size else None,
        outside_tolerance_count=int(np.count_nonzero(delta > (atol + rtol * np.abs(expected))))
        if delta is not None
        else None,
    )
