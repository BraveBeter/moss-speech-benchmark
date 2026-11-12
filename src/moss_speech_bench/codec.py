"""Versioned local Codec-only reconstruction, scoring and paired comparisons."""

import os
import re
import subprocess
import time
from pathlib import Path

from .storage import digest, file_hash, read_json, read_jsonl, source_identity, write_json

PROTOCOL = "moss-codec-only-v2-full"
BACKENDS = ("hf-native", "sg-native")
CORPUS_COUNTS = {"en": 1088, "zh": 2020}


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
    path = Path(path).resolve()
    config = read_json(path)
    allowed = {
        "protocol",
        "backend",
        "decoder_mode",
        "physical_gpu",
        "seed",
        "python",
        "codec_path",
        "codec_revision",
        "hf_source_root",
        "model_root",
        "model_sha",
        "corpus",
        "corpus_revision",
        "corpus_sha256",
        "assets",
        "adapter_options",
    }
    if set(config) - allowed:
        raise ValueError("Unknown Codec config fields; subsets and precision injection are removed")
    if config.get("protocol") != PROTOCOL or config.get("backend") not in BACKENDS:
        raise ValueError(
            "Explicit full Codec protocol and hf-native/sg-native backend are required"
        )
    if config.get("decoder_mode") != "offline":
        raise ValueError("Codec uses the native HF offline decode contract")
    if type(config.get("physical_gpu")) is not int or config["physical_gpu"] < 0:
        raise ValueError("An explicitly selected physical GPU is required")
    if type(config.get("seed")) is not int:
        raise ValueError("A fixed integer seed is required")
    for key in ("codec_revision", "corpus_revision", "corpus_sha256"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError("A frozen revision/hash is required: " + key)
    for key in ("python", "codec_path", "corpus", "hf_source_root", "model_root"):
        if key in config:
            config[key] = (
                os.path.abspath(path.parent / config[key])
                if key == "python"
                else str((path.parent / config[key]).resolve())
            )
            if not Path(config[key]).exists():
                raise ValueError("Missing Codec input: " + key)
    options = config.get("adapter_options", {})
    if not isinstance(options, dict) or {"device", "codec_path"} & options.keys():
        raise ValueError("Adapter options cannot override device or Codec assets")
    if config["backend"] == "hf-native":
        if options or not Path(config["hf_source_root"]).is_dir():
            raise ValueError("HF requires its native dependency source and no adapter options")
        identity = dict(kind="hf-native", revision=config["codec_revision"])
    else:
        identity = git_identity(config["model_root"])
        if identity["commit"] != config.get("model_sha") or identity["dirty"]:
            raise ValueError("Codec model checkout differs from the pinned clean SHA")
    assets = config.get("assets", [])
    for item in assets:
        item["path"] = str((path.parent / item["path"]).resolve())
    if not assets or any(file_hash(item["path"]) != item["sha256"] for item in assets):
        raise ValueError("Codec assets are missing or have changed")
    if file_hash(config["corpus"]) != config["corpus_sha256"]:
        raise ValueError("Codec corpus manifest hash mismatch")
    samples = read_jsonl(config["corpus"])
    counts = {lang: sum(s.get("language") == lang for s in samples) for lang in CORPUS_COUNTS}
    if counts != CORPUS_COUNTS or len(samples) != sum(CORPUS_COUNTS.values()):
        raise ValueError("Codec requires the complete Seed-TTS-Eval EN1088 + ZH2020 corpus")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Codec corpus must contain distinct samples")
    for sample in samples:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample["id"]) or not sample["text"].strip():
            raise ValueError("Invalid Codec sample identity or target transcript")
        for kind in ("target", "prompt"):
            audio = (Path(config["corpus"]).parent / sample[kind + "_audio"]).resolve()
            sample[kind + "_audio"] = str(audio)
            if file_hash(audio) != sample[kind + "_sha256"]:
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
            codec_assets_hash=digest(sorted(item["sha256"] for item in config["assets"])),
            corpus_revision=config["corpus_revision"],
            corpus_manifest_hash=config["corpus_sha256"],
            corpus_hash=digest(
                [
                    {k: v for k, v in sample.items() if k not in ("target_audio", "prompt_audio")}
                    for sample in samples
                ]
            ),
            sim_reference="per-sample-prompt",
            ar=False,
        ),
        expected=len(samples),
        corpus_counts=CORPUS_COUNTS,
        source=identity,
        client=source_identity(),
        configuration_hash=digest(config),
    )
    write_json(root / "manifest.json", manifest)
    code = run_worker(
        root, "infer", root / "execution/config.json", config["python"], config["physical_gpu"]
    )
    generation = read_json(root / "generation.json") if (root / "generation.json").exists() else {}
    generated = generation.get("rows", [])
    complete = (
        code == 0
        and generation.get("status") == "complete"
        and len(generated) == len(samples)
        and {r["id"] for r in generated} == {s["id"] for s in samples}
        and all(r["status"] == "complete" for r in generated)
    )
    manifest["status"] = "complete" if complete else "partial"
    manifest["samples_hash"] = digest(samples)
    manifest["generation_hash"] = file_hash(root / "generation.json") if generation else None
    manifest["worker_returncode"] = code
    write_json(root / "manifest.json", manifest)
    render(root)
    return manifest


def score(output, config_path):
    root = Path(output).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["protocol"] != PROTOCOL or manifest["status"] not in ("complete", "partial"):
        raise ValueError("Scoring requires a frozen full-corpus Codec run")
    validate_artifacts(root, manifest)
    if (root / "quality.json").exists():
        raise ValueError("Preserve prior scores; use an explicit new scoring attempt")
    from .quality import configuration

    config, _ = configuration(config_path)
    config["python"] = os.path.abspath(Path(config_path).resolve().parent / config["python"])
    if type(config.get("physical_gpu")) is not int or config["physical_gpu"] < 0:
        raise ValueError("An explicitly selected scoring GPU is required")
    config["device"] = "cuda:0"
    if (
        config.get("asr_en", {}).get("model") != "whisper-large-v3"
        or config.get("asr_zh", {}).get("model") != "paraformer-zh"
    ):
        raise ValueError("Codec requires Whisper-large-v3 and Paraformer-zh")
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
    if (root / "quality.json").exists():
        write_json(root / "quality.json", result)
    render(root)
    return result


def summarize_quality(records):
    from .quality import aggregate

    return {
        language: aggregate([row for row in records if row["language"] == language])
        for language in ("en", "zh")
    }


def validate_artifacts(root, manifest):
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported Codec run; use its original client for historical reports")
    samples = read_json(root / "samples.json")
    if len(samples) != sum(CORPUS_COUNTS.values()) or digest(samples) != manifest["samples_hash"]:
        raise ValueError("Frozen Codec samples changed")
    path = root / "generation.json"
    if manifest.get("generation_hash") != (file_hash(path) if path.exists() else None):
        raise ValueError("Frozen Codec generation changed")
    generation = read_json(path) if path.exists() else {}
    for row in generation.get("rows", []):
        if row.get("wav_path") and file_hash(root / row["wav_path"]) != row["wav_sha256"]:
            raise ValueError("Codec output audio hash mismatch")


def render(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    generation = read_json(root / "generation.json") if (root / "generation.json").exists() else {}
    quality = read_json(root / "quality.json") if (root / "quality.json").exists() else {}
    validate_artifacts(root, manifest)
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
        "## 运行完整性",
        "",
        "| 范围 | 状态 |",
        "|---|---|",
        f"| Encoder→Decoder，AR未执行 | {manifest['status']} |",
        f"| 完整生成 | {sum(row['status'] == 'complete' for row in generation.get('rows', []))}/{manifest['expected']} |",
        "",
        "## 性能",
        "",
        "| 范围 | 状态 |",
        "|---|---|",
        "| 端到端性能 | 未执行；本入口仅评估Codec离线重建质量 |",
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
        "同一配对语音的真实文本用于WER/CER；每条对应prompt用于SIM（cosine×100）。全量EN1088+ZH2020，离线模式、样本分母、失败、模型/环境/输入hash均保留。",
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
        "# Codec-only 比较",
        "",
        "| 后端 | EN WER | ZH CER | EN SIM（0–100） | ZH SIM（0–100） |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (root, manifest) in loaded.items():
        validate_artifacts(root, manifest)
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
    root = Path(output)
    write_json(root / "analysis/codec-comparison.json", result)
    lines += [
        "",
        f"Encoder codes精确一致：{result['encoder_codes_exact']}。WER/CER为比例，SIM为0–100分。实际分母、失败及各次运行身份见JSON和单次摘要。",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return result
