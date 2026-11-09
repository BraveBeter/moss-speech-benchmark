"""Versioned local Codec-only reconstruction, scoring and paired comparisons."""

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

from .storage import digest, read_json, read_jsonl, source_identity, write_json

PROTOCOL = "moss-codec-only-v1"
BACKENDS = ("hf-native", "sg-fp32", "sg-bf16")


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def git_identity(path):
    return {
        key: subprocess.check_output(["git", "-C", str(path), *command], text=True).strip()
        for key, command in {
            "branch": ["branch", "--show-current"],
            "commit": ["rev-parse", "HEAD"],
            "dirty": ["status", "--porcelain"],
        }.items()
    }


def load_config(path):
    config = read_json(path)
    if config.get("protocol") != PROTOCOL or config.get("backend") not in BACKENDS:
        raise ValueError("Explicit Codec-only protocol and backend are required")
    if config.get("decoder_mode") != "offline":
        raise ValueError("Codec-only v1 uses the native HF offline decode contract")
    if type(config.get("physical_gpu")) is not int or config["physical_gpu"] < 0:
        raise ValueError("An explicitly authorized physical GPU is required")
    if type(config.get("seed")) is not int:
        raise ValueError("A fixed integer seed is required")
    precision = config.get("flow_precision")
    allowed_precision = (
        (
            "ffn_attention",
            "ffn_attention_sdpa",
            "ffn_attention_sdpa_resident",
            "ffn_attention_sdpa_storage",
        )
        if config["backend"] == "sg-bf16"
        else (None,)
    )
    if precision not in allowed_precision:
        raise ValueError("Precision must match the selected Codec backend")
    for key in ["python", "codec_path", "corpus"]:
        if not Path(config[key]).exists():
            raise ValueError("Missing Codec input: " + key)
    if config["backend"] == "hf-native":
        if not Path(config["hf_source_root"]).is_dir():
            raise ValueError("Native HF dependency source root is required")
        identity = dict(kind="hf-native", revision=config["codec_revision"])
    else:
        identity = git_identity(config["model_root"])
        if identity["commit"] != config.get("model_sha") or identity["dirty"]:
            raise ValueError("Codec model checkout differs from the pinned clean SHA")
    assets = config.get("assets", [])
    if not assets or any(file_hash(item["path"]) != item["sha256"] for item in assets):
        raise ValueError("Codec assets are missing or have changed")
    samples = read_jsonl(config["corpus"])
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Codec corpus must contain distinct samples")
    for sample in samples:
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+", sample["id"])
            or sample["language"] not in ("en", "zh")
            or not sample["text"].strip()
        ):
            raise ValueError("Invalid Codec sample identity, language or target text")
        for kind in ("target", "prompt"):
            if file_hash(sample[kind + "_audio"]) != sample[kind + "_sha256"]:
                raise ValueError("Codec corpus audio hash mismatch")
    return config, samples, identity


def run_worker(root, phase, config, python, gpu):
    command = [
        python,
        str(Path(__file__).with_name("codec_worker.py")),
        "--phase",
        phase,
        "--config",
        str(config),
        "--output",
        str(root),
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment["HF_MODULES_CACHE"] = str(root / "execution/hf-modules-cache")
    started = time.time()
    with (root / "execution" / (phase + ".log")).open("w") as log:
        process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    write_json(
        root / "execution" / (phase + "-command.json"),
        dict(
            argv=command,
            physical_gpu=gpu,
            started_at=started,
            finished_at=time.time(),
            returncode=process.returncode,
        ),
    )
    return process.returncode


def infer(config_path, output):
    config, samples, identity = load_config(config_path)
    root = Path(output).resolve()
    if root.exists() and any(path.name != "execution" for path in root.iterdir()):
        raise ValueError("Use a fresh directory for each Codec backend/configuration")
    (root / "execution").mkdir(parents=True, exist_ok=True)
    write_json(root / "execution/config.json", config)
    write_json(root / "samples.json", samples)
    manifest = dict(
        protocol=PROTOCOL,
        status="prepared",
        backend=config["backend"],
        contract=dict(
            decoder_mode="offline",
            seed=config["seed"],
            codec_revision=config["codec_revision"],
            codec_assets_hash=digest(config["assets"]),
            corpus_hash=digest(samples),
            sim_reference="per-sample-prompt",
            ar=False,
        ),
        expected=len(samples),
        source=identity,
        client=source_identity(),
        configuration_hash=digest(config),
    )
    write_json(root / "manifest.json", manifest)
    code = run_worker(
        root, "infer", root / "execution/config.json", config["python"], config["physical_gpu"]
    )
    generation = read_json(root / "generation.json") if (root / "generation.json").exists() else {}
    manifest["status"] = (
        "complete" if code == 0 and generation.get("status") == "complete" else "failed"
    )
    manifest["worker_returncode"] = code
    write_json(root / "manifest.json", manifest)
    render(root)
    return manifest


def score(output, config_path):
    root = Path(output).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["protocol"] != PROTOCOL or manifest["status"] != "complete":
        raise ValueError("Scoring requires a complete Codec reconstruction run")
    if (root / "quality.json").exists():
        raise ValueError("Preserve prior scores; use an explicit new scoring attempt")
    config = read_json(config_path)
    if (
        config.get("asr_en", {}).get("model") != "whisper-large-v3"
        or config.get("asr_zh", {}).get("model") != "paraformer-zh"
    ):
        raise ValueError("Codec-only v1 requires Whisper-large-v3 and Paraformer-zh")
    for sample in read_json(root / "samples.json"):
        for kind in ("target", "prompt"):
            if file_hash(sample[kind + "_audio"]) != sample[kind + "_sha256"]:
                raise ValueError("Frozen corpus changed before scoring")
    write_json(root / "execution/score-config.json", config)
    code = run_worker(
        root,
        "score",
        root / "execution/score-config.json",
        config["python"],
        config["physical_gpu"],
    )
    result = (
        read_json(root / "quality.json")
        if (root / "quality.json").exists()
        else dict(status="failed")
    )
    if code != 0:
        result["status"] = "failed"
    render(root)
    return result


def summarize_quality(records):
    from .quality import aggregate

    return {
        language: aggregate([row for row in records if row["language"] == language])
        for language in ("en", "zh")
    }


def render(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    generation = read_json(root / "generation.json") if (root / "generation.json").exists() else {}
    quality = read_json(root / "quality.json") if (root / "quality.json").exists() else {}
    source = manifest["source"]
    source_label = (
        source.get("branch", "HF native")
        + "@"
        + source.get("commit", source.get("revision", "未记录"))
    )
    client = manifest["client"]
    rows = [
        f"# Codec-only: {manifest['backend']}",
        "",
        f"被测Codec：`{source_label}`，配置`{manifest['backend']}`。",
        f"客户端：`{client.get('branch', '未记录')}@{client.get('commit', '未记录')}`，工作区状态：`{client.get('dirty') or 'clean'}`。",
        "",
        "## 正确性",
        "",
        "| 范围 | 状态 |",
        "|---|---|",
        f"| Encoder→Decoder，AR未执行 | {manifest['status']} |",
        f"| 完整生成 | {sum(row['status'] == 'complete' for row in generation.get('rows', []))}/{manifest['expected']} |",
        "| C00–C08 | 本Codec质量协议未执行，不代替服务资格 |",
        "",
        "## 性能",
        "",
        "| 范围 | 状态 |",
        "|---|---|",
        "| Quick/RTF/TTFA | 未执行；仅Codec质量重建 |",
        "",
        "## 质量",
        "",
        "| 语言 | 指标 | 已评分/计划 | 数值 |",
        "|---|---|---:|---:|",
    ]
    for language, metrics in quality.get("aggregate", {}).items():
        for metric, result in metrics.items():
            rows.append(
                f"| {language} | {metric} | {result['scored']}/{result['expected']} | {result['value']} |"
            )
    if not quality:
        rows.append("| — | WER/CER/SIM | 0 | 未评分 |")
    rows += [
        "",
        "同一配对语音的真实文本用于WER/CER；每条对应prompt用于SIM（cosine×100）。离线模式、样本分母、失败、模型/环境/输入hash均保留；小样本结果不宣称论文全量复现。",
    ]
    (root / "summary.md").write_text("\n".join(rows) + "\n")
    return dict(status=quality.get("status", manifest["status"]))


def compare(runs, output):
    loaded = {
        name: (Path(path), read_json(Path(path) / "manifest.json")) for name, path in runs.items()
    }
    if not loaded:
        raise ValueError("Missing Codec comparison runs")
    contract = next(iter(loaded.values()))[1]["contract"]
    if any(manifest["contract"] != contract for _, manifest in loaded.values()):
        raise ValueError("Codec comparison requires identical corpus, seed and decode mode")
    code_ledgers = []
    scoring_assets = []
    result = dict(protocol=PROTOCOL, status="complete", contract=contract, runs={})
    lines = [
        "# Codec-only 三方比较",
        "",
        "| 后端 | EN WER | ZH CER | EN SIM（0–100） | ZH SIM（0–100） |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (root, manifest) in loaded.items():
        quality = read_json(root / "quality.json")
        scoring_assets.append(quality["scoring_contract"]["assets"])
        generation = read_json(root / "generation.json")
        if manifest["status"] != "complete" or quality["status"] != "complete":
            result["status"] = "partial"
        code_ledgers.append(
            {
                row["id"]: [row.get("codes_sha256"), row.get("prompt_codes_sha256")]
                for row in generation["rows"]
            }
        )
        metrics = quality["aggregate"]
        values = [
            metrics["en"]["wer"]["value"],
            metrics["zh"]["cer"]["value"],
            metrics["en"]["sim"]["value"],
            metrics["zh"]["sim"]["value"],
        ]
        result["runs"][name] = dict(
            path=str(root.resolve()),
            client=manifest["client"],
            source=manifest["source"],
            aggregate=metrics,
        )
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join("未评分" if value is None else f"{value:.6f}" for value in values)
            + " |"
        )
    result["encoder_codes_exact"] = all(ledger == code_ledgers[0] for ledger in code_ledgers)
    if any(assets != scoring_assets[0] for assets in scoring_assets):
        raise ValueError("Codec comparison requires identical scoring assets")
    if not result["encoder_codes_exact"]:
        result["status"] = "partial"
    root = Path(output)
    write_json(root / "analysis/codec-comparison.json", result)
    lines += [
        "",
        f"三方Encoder codes精确一致：{result['encoder_codes_exact']}。WER/CER为比例，SIM为0–100分。实际分母、失败及各次运行身份见JSON和单次摘要。",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return result
