"""Offline scoring of frozen inference artifacts; no generation requests."""

import ast
import importlib.metadata
import math
import re
import string
from pathlib import Path
from types import SimpleNamespace

from ._vendor.accuracy import extract_answer_letter
from .reporting import _load_manifest, collect_cell
from .storage import digest, read_json, read_jsonl, source_identity, write_json


class Unavailable(RuntimeError):
    """An explicitly required scoring dependency or asset is absent."""


def normalize_answer(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def aliases(value):
    if isinstance(value, str) and value.startswith("["):
        value = ast.literal_eval(value)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ValueError("Expected nonempty answer string or list of aliases")
    return value


class LocalScorers:
    """Lazy optional models. Local assets only; CPU is the default device."""

    def __init__(self, config):
        self.config, self.models = config, {}

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
        model = self.model("asr_" + lang)
        if lang == "en":
            return model.transcribe(str(path), language="en", fp16=False)["text"]
        return model.generate(input=str(path))[0]["text"]

    def sim(self, reference, audio):
        # Upstream reports cosine * 100; expose cosine in [-1, 1].
        return self.model("sim").score_batch([str(reference)], [str(audio)])[0] / 100

    def utmos(self, audio):
        return self.model("utmos").score_batch([str(audio)])[0]


def _asset_hash(path):
    import hashlib

    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def configuration(path):
    config = read_json(path) if path else {}
    root = Path(path).resolve().parent if path else Path.cwd()
    assets = {}
    for key in ("asr_en", "asr_zh", "sim", "utmos"):
        if key not in config:
            continue
        for field in ("path", "wavlm_base", "reference"):
            if field not in config[key]:
                continue
            asset = (root / config[key][field]).resolve()
            if not asset.exists():
                raise ValueError(f"Missing local scoring asset: {asset}")
            files = sorted(asset.rglob("*")) if asset.is_dir() else [asset]
            assets[key + "." + field] = {
                str(f.relative_to(asset)) if asset.is_dir() else f.name: _asset_hash(f)
                for f in files
                if f.is_file()
            }
            config[key][field] = str(asset)
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
        protocol="moss-quality-v1",
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
    task, lang = sample["task"], sample["language"]
    metrics = ["wer" if lang == "en" else "cer", "sim"] if task == "t2s" else ["accuracy"]
    if task == "s2s":
        metrics = ["accuracy", "utmos"]
    if not row or not row["is_success"]:
        return {m: {"status": "generation_failed" if row else "not_attempted"} for m in metrics}
    transcript = None
    if task.endswith("s"):
        transcript = _measure(lambda: backend.transcribe(audio, lang))
    result = {}
    for metric in metrics:
        if metric in ("wer", "cer"):
            if transcript["status"] != "scored":
                result[metric] = transcript
                continue

            def error_rate():
                from ._vendor.asr_scoring import apply_wer

                scored = apply_wer(
                    SimpleNamespace(target_text=sample["expected"], error="", is_success=False),
                    transcript["value"],
                    lang,
                )
                if not scored.is_success:
                    raise ValueError(scored.error)
                return {
                    k: getattr(scored, k)
                    for k in (
                        "wer",
                        "ref_norm",
                        "hyp_norm",
                        "substitutions",
                        "deletions",
                        "insertions",
                        "hits",
                    )
                }

            result[metric] = _measure(error_rate)
        elif metric == "sim":

            def similarity():
                spec = config.get("sim", {})
                if not spec.get("reference"):
                    raise Unavailable("Fixed-voice SIM requires the actual server voice reference")
                if (
                    not config.get("_voice_sha256")
                    or config.get("_reference_sha256") != config["_voice_sha256"]
                ):
                    raise Unavailable("SIM reference identity does not match deployment voice")
                return backend.sim(spec["reference"], audio)

            result[metric] = _measure(similarity)
        elif metric == "utmos":
            result[metric] = _measure(lambda: backend.utmos(audio))
        else:

            def accuracy():
                if sample.get("expected") is None:
                    raise ValueError("Missing reference answer")
                if task == "t2t":
                    expected = str(sample["expected"]).strip().upper()
                    predicted = extract_answer_letter(row["text"])
                    valid = predicted is not None and predicted < len(config.get("choices", "AB"))
                    return {
                        "correct": valid and chr(65 + predicted) == expected,
                        "valid": valid,
                        "answer": row["text"],
                    }
                if transcript and transcript["status"] != "scored":
                    raise Unavailable("QA transcript unavailable: " + str(transcript))
                answer = transcript["value"] if transcript else row["text"]
                normalized = normalize_answer(answer)
                return {
                    "correct": bool(normalized)
                    and normalized in [normalize_answer(a) for a in aliases(sample["expected"])],
                    "valid": bool(normalized),
                    "answer": answer,
                }

            result[metric] = _measure(accuracy)
    if transcript:
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
                for s in ("scored", "failed", "unavailable", "generation_failed", "not_attempted")
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


def score(directory, config_path=None):
    root = Path(directory)
    manifest = _load_manifest(root)
    config, contract = configuration(config_path)
    backend = LocalScorers(config)
    # Match the fixed reference to the actual deployment, never a SeedTTS clone prompt.
    deployment = manifest["contract"].get("deployment") or {}
    voice = deployment.get("settings", {}).get("voice_reference_sha256")
    reference = contract["assets"].get("sim.reference", {})
    config = dict(
        config, _voice_sha256=voice, _reference_sha256=next(iter(reference.values()), None)
    )
    contract["voice_reference_sha256"] = voice
    selection = config.get("cells")
    if selection is None:
        selected = {}
        for cell in manifest["cells"]:
            if cell["dataset"] not in selected or (
                cell["concurrency"] == 4 and cell["repeat"] == 1
            ):
                selected[cell["dataset"]] = cell["id"]
        selection = list(selected.values())
    known = {c["id"] for c in manifest["cells"]}
    if not selection or len(set(selection)) != len(selection) or not set(selection) <= known:
        raise ValueError("Scoring cells must be nonempty, unique existing cell IDs")
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
        records = []
        for sample in inputs:
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
                record = {
                    "sample_id": sample["id"],
                    "finish_reason": row.get("finish_reason") if row else None,
                    "metrics": evaluate(sample, row, audio, backend, config),
                }
                write_json(file, dict(record, _hash=digest(record)))
            records.append(record)
        cells[cell["id"]] = dict(
            scored=sum(
                all(v["status"] == "scored" for k, v in r["metrics"].items() if k != "transcript")
                for r in records
            ),
            binding=binding,
            metrics=aggregate(records),
            records=[
                dict(path=str(f.relative_to(root)), hash=digest(read_json(f)))
                for f in sorted(cache.glob("*.json"))
            ],
        )
    result = dict(contract=contract, selection=selection, cells=cells, status="complete")
    if any(m["scored"] < m["expected"] for c in cells.values() for m in c["metrics"].values()):
        result["status"] = "partial"
    write_json(root / "quality.json", result)
    return result


def quality_report(root):
    path = root / "quality.json"
    if not path.exists():
        return None
    quality = read_json(path)
    manifest = _load_manifest(root)
    voice = (
        (manifest["contract"].get("deployment") or {})
        .get("settings", {})
        .get("voice_reference_sha256")
    )
    if voice != quality["contract"].get("voice_reference_sha256"):
        raise ValueError("Quality voice identity mismatch")
    if (
        not quality["selection"]
        or not set(quality["selection"]) <= {c["id"] for c in manifest["cells"]}
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
    lines = [
        "# Offline quality",
        "",
        f"Status: {quality['status']}",
        "",
        "| Cell | Metric | Expected | Scored | Value |",
        "|---|---|---:|---:|---:|",
    ]
    for name, cell in quality["cells"].items():
        for metric, value in cell["metrics"].items():
            lines.append(
                f"| {name} | {metric} | {value['expected']} | {value['scored']} | {value['value']} |"
            )
    (root / "quality.md").write_text("\n".join(lines) + "\n")
    return quality
