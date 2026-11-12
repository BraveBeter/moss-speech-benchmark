"""Offline scoring of frozen inference artifacts; no generation requests."""

import asyncio
import importlib.metadata
import math
import time
import uuid
from pathlib import Path

from .reporting import _format_value, _load_manifest, collect_cell, dataset_columns
from .storage import digest, file_hash, read_json, read_jsonl, source_identity, write_json


class Unavailable(RuntimeError):
    """An explicitly required scoring dependency or asset is absent."""


class LocalScorers:
    """Lazy optional models. Local assets only; CPU is the default device."""

    def __init__(self, config):
        self.config, self.models = config, {}
        self.transcripts = {}

    def model(self, key):
        if key in self.models:
            return self.models[key]
        spec = self.config.get(key)
        if not spec:
            raise Unavailable(f"{key} not configured")
        try:
            device = self.config.get("device", "cpu")
            if key == "asr_en":
                import whisper

                model = whisper.load_model(spec["path"], device=device)
            elif key == "asr_zh":
                from funasr import AutoModel

                model = AutoModel(model=spec["path"], device=device, disable_update=True)
            elif key == "sim":
                from ._vendor.speaker_similarity import WavLMSpeakerSimilarity

                model = WavLMSpeakerSimilarity(
                    finetune_checkpoint=spec["path"], wavlm_base=spec["wavlm_base"], device=device
                )
            else:
                from ._vendor.utmos_scoring import UTMOSScorer

                model = UTMOSScorer(device, spec["path"])
        except ImportError as exc:
            raise Unavailable(str(exc)) from exc
        self.models[key] = model
        return model

    def transcribe(self, path, lang):
        spec = self.config.get("asr_" + lang, {})
        if spec.get("backend") == "qwen3-asr":
            from .asr import transcription_key

            result = self.transcripts[transcription_key(path, lang, spec)]
            if result["status"] != "scored":
                raise RuntimeError("Qwen3-ASR: " + str(result))
            return result["text"]
        model = self.model("asr_" + lang)
        if lang == "en":
            return model.transcribe(str(path), language="en", fp16=False)["text"]
        return model.generate(input=str(path))[0]["text"]

    def judge(self, sample, answer):
        if not self.config.get("judge"):
            raise Unavailable("Third-party Judge not configured; no EM fallback")
        from .judge import judge_answer

        return judge_answer(
            sample, answer, self.config, self.config["_judge_cache"], self.config.get("_deadline")
        )

    def sim(self, reference, audio):
        # Upstream reports cosine * 100; expose cosine in [-1, 1].
        return self.model("sim").score_batch([str(reference)], [str(audio)])[0] / 100

    def utmos(self, audio):
        return self.model("utmos").score_batch([str(audio)])[0]


def configuration(path):
    config = read_json(path) if path else {}
    if not isinstance(config, dict) or set(config) - {
        "device",
        "asr_en",
        "asr_zh",
        "sim",
        "utmos",
        "judge",
        "python",
        "physical_gpu",
    }:
        raise ValueError("Unsupported scoring configuration fields")
    root = Path(path).resolve().parent if path else Path.cwd()
    assets = {}
    for key in ("asr_en", "asr_zh", "sim", "utmos"):
        if key not in config:
            continue
        for field in ("path", "wavlm_base"):
            if field not in config[key]:
                continue
            asset = (root / config[key][field]).resolve()
            if not asset.exists():
                raise ValueError(f"Missing local scoring asset: {asset}")
            files = sorted(asset.rglob("*")) if asset.is_dir() else [asset]
            assets[key + "." + field] = {
                str(f.relative_to(asset)) if asset.is_dir() else f.name: file_hash(f)
                for f in files
                if f.is_file()
            }
            config[key][field] = str(asset)
    if "judge" in config:
        from urllib.parse import urlsplit

        spec = config["judge"]
        allowed = {"base_url", "model", "revision", "api_key_env", "timeout", "thinking"}
        if not isinstance(spec, dict) or set(spec) - allowed:
            raise ValueError("Judge accepts only endpoint/model/revision/key environment/timeout")
        if "thinking" in spec and spec["thinking"] not in (
            {"type": "disabled"},
            {"type": "enabled"},
        ):
            raise ValueError("Judge thinking must explicitly be enabled or disabled")
        if any(
            not isinstance(spec.get(k), str) or not spec[k].strip()
            for k in ("base_url", "model", "revision")
        ):
            raise ValueError("Judge endpoint, model and revision are required")
        url = urlsplit(spec["base_url"])
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Judge endpoint must not contain embedded credentials or query parameters"
            )
        if (
            not math.isfinite(float(spec.get("timeout", 120)))
            or float(spec.get("timeout", 120)) <= 0
        ):
            raise ValueError("Judge timeout must be positive and finite")
    versions = {}
    for name in ("jiwer", "openai-whisper", "zhon", "funasr", "torch", "torchaudio", "s3prl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return config, dict(
        config=config,
        assets=assets,
        versions=versions,
        code=source_identity()["source_hash"],
        protocol="moss-quality-v3-qa-only",
    )


def _measure(function):
    try:
        value = function()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite score")
        return {"status": "scored", "value": value}
    except (Unavailable, ImportError) as exc:
        return {"status": "unavailable", "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}


def evaluate(sample, row, audio, backend, config):
    metrics = ["accuracy", "utmos"] if sample["task"].endswith("s") else ["accuracy"]
    if not row or not row["is_success"]:
        status = "generation_failed" if row else "not_attempted"
        return {metric: dict(status=status) for metric in metrics}
    transcript = (
        _measure(lambda: backend.transcribe(audio, sample["language"]))
        if sample["task"].endswith("s")
        else None
    )

    def accuracy():
        if transcript and transcript["status"] != "scored":
            raise Unavailable("QA transcript unavailable: " + str(transcript))
        answer = transcript["value"] if transcript else row["text"]
        return backend.judge(sample, answer)

    result = dict(accuracy=_measure(accuracy))
    if "utmos" in metrics:
        result["utmos"] = _measure(lambda: backend.utmos(audio))
        result["transcript"] = transcript
    return result


def aggregate(records):
    summary = {}
    for metric in sorted({m for r in records for m in r["metrics"] if m != "transcript"}):
        rows = [r["metrics"][metric] for r in records]
        values = [r["value"] for r in rows if r["status"] == "scored"]
        item = {
            "expected": len(rows),
            "scored": len(values),
            "statuses": {
                s: sum(r["status"] == s for r in rows)
                for s in (
                    "scored",
                    "failed",
                    "unavailable",
                    "generation_failed",
                    "not_attempted",
                    "budget_exhausted",
                )
            },
        }
        if metric in ("wer", "cer"):
            errors = sum(v["substitutions"] + v["deletions"] + v["insertions"] for v in values)
            words = sum(v["substitutions"] + v["deletions"] + v["hits"] for v in values)
            item.update(
                value=errors / words if words else None,
                mean=sum(v["wer"] for v in values) / len(values) if values else None,
            )
        elif metric == "accuracy":
            correct = sum(v["correct"] for v in values)
            item.update(
                value=correct / len(rows) if values else None,
                evaluated_accuracy=correct / len(values) if values else None,
                invalid=sum(not v["valid"] for v in values),
            )
        else:
            item["value"] = sum(values) / len(values) if values else None
        summary[metric] = item
    return summary


def score(directory, config_path=None, time_budget=None):
    root = Path(directory)
    manifest = _load_manifest(root)
    started = time.monotonic()
    budget = time_budget
    if budget is not None and not 0 < budget < float("inf"):
        raise ValueError("Time budget must be positive and finite")
    invocation = uuid.uuid4().hex
    config, contract = configuration(config_path)
    from .judge_prompts import PROMPTS, UPSTREAM_COMMIT

    contract["judge_templates"] = dict(commit=UPSTREAM_COMMIT, hash=digest(PROMPTS))
    config = dict(
        config,
        _judge_cache=str(root / "quality" / "judge"),
        _deadline=started + budget if budget else None,
    )
    backend = LocalScorers(config)
    selection = [c["id"] for c in manifest["cells"] if c["repeat"] == 1]
    asr_items = []
    for cell in manifest["cells"]:
        if cell["id"] not in selection:
            continue
        rows, _ = collect_cell(root, cell)
        for row in rows:
            spec = config.get("asr_" + row["language"], {})
            if spec.get("backend") == "qwen3-asr" and row["is_success"] and row.get("wav_path"):
                asr_items.append(
                    (
                        root / cell["attempts"][-1] / "outputs" / row["wav_path"],
                        row["language"],
                        spec,
                    )
                )
    if asr_items:
        from .asr import transcribe_many

        backend.transcripts = asyncio.run(
            transcribe_many(
                asr_items, root / "quality" / "asr", started + budget if budget else None
            )
        )
    cells = {}
    for cell in manifest["cells"]:
        if cell["id"] not in selection:
            continue
        rows, _ = collect_cell(root, cell)
        inputs = read_jsonl(root / "inputs" / (cell["dataset"] + ".jsonl"))
        binding = digest(dict(rows=rows, inputs=inputs, attempts=cell["attempts"]))
        cache = root / "quality" / digest(dict(contract=contract, binding=binding))
        cache.mkdir(parents=True, exist_ok=True)
        by_id = {r["sample_id"]: r for r in rows}
        records, record_entries = [], []
        for sample in inputs:
            sample = dict(
                sample,
                source_dataset=manifest["contract"]["datasets"][cell["dataset"]].get(
                    "source_dataset", cell["dataset"]
                ),
            )
            file = cache / (digest(sample["id"]) + ".json")
            if file.exists():
                record = read_json(file)
                checksum = record.pop("_hash", None)
                if checksum != digest(record):
                    raise ValueError("Quality cache hash mismatch")
            else:
                row = by_id.get(sample["id"])
                audio = (
                    root / cell["attempts"][-1] / "outputs" / row["wav_path"]
                    if row and row.get("wav_path")
                    else None
                )
                exhausted = budget is not None and time.monotonic() - started >= budget
                if exhausted and row and row["is_success"]:
                    metrics = {
                        k: dict(status="budget_exhausted")
                        for k in evaluate(sample, None, None, backend, config)
                    }
                    file = cache / "attempts" / invocation / file.name
                else:
                    metrics = evaluate(sample, row, audio, backend, config)
                record = {
                    "sample_id": sample["id"],
                    "finish_reason": row.get("finish_reason") if row else None,
                    "metrics": metrics,
                }
                if any(v["status"] != "scored" for v in metrics.values()):
                    file = cache / "attempts" / invocation / file.name
                write_json(file, dict(record, _hash=digest(record)))
            records.append(record)
            record_entries.append(
                dict(path=str(file.relative_to(root)), hash=digest(read_json(file)))
            )
        cells[cell["id"]] = dict(
            scored=sum(
                all(v["status"] == "scored" for k, v in r["metrics"].items() if k != "transcript")
                for r in records
            ),
            binding=binding,
            metrics=aggregate(records),
            records=record_entries,
        )
    result = dict(
        contract=contract,
        selection=selection,
        cells=cells,
        status="complete",
        time_budget_s=budget,
        elapsed_s=time.monotonic() - started,
    )
    if any(m["scored"] < m["expected"] for c in cells.values() for m in c["metrics"].values()):
        result["status"] = "partial"
    write_json(root / "quality.json", result)
    return result


def read_quality(root):
    """Validate persisted quality evidence without rewriting its report."""
    root = Path(root)
    path = root / "quality.json"
    if not path.exists():
        return None
    quality = read_json(path)
    manifest = _load_manifest(root)
    if (
        not quality["selection"]
        or set(quality["selection"]) != {c["id"] for c in manifest["cells"] if c["repeat"] == 1}
        or set(quality["selection"]) != set(quality["cells"])
    ):
        raise ValueError("Quality selection mismatch")
    for cell in manifest["cells"]:
        if cell["id"] not in quality["selection"]:
            continue
        saved = quality["cells"].get(cell["id"])
        if saved is None:
            raise ValueError("Quality coverage missing cell")
        rows, _ = collect_cell(root, cell)
        inputs = read_jsonl(root / "inputs" / (cell["dataset"] + ".jsonl"))
        if saved["binding"] != digest(dict(rows=rows, inputs=inputs, attempts=cell["attempts"])):
            raise ValueError("Stale quality results; rerun score for current inference artifacts")
        records = []
        for entry in saved["records"]:
            file = (root / entry["path"]).resolve()
            if root.resolve() not in file.parents:
                raise ValueError("Quality artifact outside run directory")
            record = read_json(file)
            if digest(record) != entry["hash"]:
                raise ValueError("Quality artifact hash mismatch")
            checksum = record.pop("_hash", None)
            if checksum != digest(record):
                raise ValueError("Quality cache checksum mismatch")
            records.append(record)
        fully_scored = sum(
            all(v["status"] == "scored" for k, v in r["metrics"].items() if k != "transcript")
            for r in records
        )
        if (
            len(records) != cell["expected"]
            or {r["sample_id"] for r in records} != {r["id"] for r in inputs}
            or aggregate(records) != saved["metrics"]
            or fully_scored != saved["scored"]
        ):
            raise ValueError("Quality summary/coverage mismatch")
    return quality


def quality_report(root):
    root = Path(root)
    quality = read_quality(root)
    if quality:
        (root / "quality.md").write_text(
            "\n".join(quality_markdown(quality, read_json(root / "manifest.json"))) + "\n"
        )
    return quality


def quality_markdown(quality, manifest):
    """One quality table, including an empty table for runs that were not scored."""
    lines = [
        "# Offline quality",
        "",
        f"评分状态：{quality['status'] if quality else '未执行'}。",
        "",
        "| 数据集 | 模态/语言 | 并发 | 轮次 | 指标 | 结果 | 评分/全部 |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    if not quality:
        return lines
    cells = {cell["id"]: cell for cell in manifest["cells"]}
    for name, cell in quality["cells"].items():
        dataset, modality = dataset_columns(manifest, cells[name]["dataset"])
        concurrency = _format_value(cells[name].get("concurrency"))
        repeat = _format_value(cells[name].get("repeat"))
        for metric, value in cell["metrics"].items():
            lines.append(
                f"| {dataset} | {modality} | {concurrency} | {repeat} | {metric} | {_format_value(value['value'])} | {value['scored']}/{value['expected']} |"
            )
    return lines
