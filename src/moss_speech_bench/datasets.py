"""Whole-dataset loading and deterministic MOSS request construction."""

import base64
import csv
import hashlib
import re
from pathlib import Path

from ._vendor.dataset import load_seedtts_samples
from .storage import digest, read_json

DAILY = {"seed_en": 1088, "seed_zh": 2020, "storycloze": 1871, "llama_questions": 300}
PERFORMANCE = ("seed_en", "storycloze", "llama_questions")
TASKS = {"t2t", "t2s", "s2t", "s2s"}
PROMPT_VERSION = "moss-chat-v1"


def _records(spec, root):
    source = spec["source"]
    path = (root / source).resolve()
    kind = spec["kind"]
    if kind == "seedtts":
        if not path.is_file():
            if path.suffix == ".lst" or Path(source).is_absolute():
                raise FileNotFoundError(path)
            if not re.fullmatch(r"[0-9a-f]{40}", spec.get("revision") or ""):
                raise ValueError("Remote SeedTTS requires a fixed 40-character commit revision")
        if path.is_file():
            # Upstream loader skips malformed lines; reject them before loading.
            for line in path.read_text().splitlines():
                if line.strip() and len(line.split("|")) < 4:
                    raise ValueError("Malformed SeedTTS meta line")
        samples = load_seedtts_samples(
            str(path) if path.is_file() else source,
            split=spec.get("split", spec["language"]),
            revision=spec.get("revision"),
        )
        return [
            {"id": s.sample_id, "text": s.target_text, "expected": s.target_text} for s in samples
        ]
    if kind == "json":
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError("JSON samples must be a list")
        return rows
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if kind == "storycloze":
        return [
            {
                "id": row["InputStoryid"],
                "text": " ".join(row[f"InputSentence{i}"] for i in range(1, 5))
                + f"\nA: {row['RandomFifthSentenceQuiz1']}\nB: {row['RandomFifthSentenceQuiz2']}"
                + "\nChoose the more coherent ending. Reply with only A or B.",
                "expected": {"1": "A", "2": "B"}[row["AnswerRightEnding"]],
            }
            for row in rows
        ]
    if kind == "openaudio":
        audio_root = (root / spec.get("audio_root", str(path.parent))).resolve()
        return [
            {
                "id": row["audio_filename"],
                "audio": str(audio_root / row["audio_filename"]),
                "expected": row.get(
                    "Answer", row.get("answers", row.get("answer_normalized_aliases"))
                ),
            }
            for row in rows
        ]
    raise ValueError(f"Unknown dataset kind: {kind}")


def prepare_dataset(spec, root, *, model, stream, sampling):
    task = spec["task"]
    if task not in TASKS or spec["language"] not in ("en", "zh"):
        raise ValueError("Invalid task or language")
    records = _records(spec, root)
    expected = spec["expected"]
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected <= 0
        or len(records) != expected
    ):
        raise ValueError(f"{spec['id']}: expected {expected} rows, found {len(records)}")
    prepared, seen = [], set()
    for row in records:
        sample_id = str(row["id"])
        if not sample_id or sample_id in seen:
            raise ValueError(f"Duplicate/empty sample id: {sample_id!r}")
        seen.add(sample_id)
        if task.startswith("s"):
            audio_path = (root / row["audio"]).resolve()
            raw = audio_path.read_bytes()
            if not raw:
                raise ValueError(f"Empty input audio: {audio_path}")
            content = [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(raw).decode(),
                        "format": audio_path.suffix.lstrip("."),
                    },
                }
            ]
        else:
            content = row["text"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Empty text: {sample_id}")
            if task == "t2s":
                instruction = (
                    "Read the following text verbatim. Do not add any other words:\n"
                    if spec["language"] == "en"
                    else "请逐字朗读以下文本，不添加任何其他内容：\n"
                )
                content = instruction + content
        payload = dict(sampling)
        payload.update(
            model=model,
            messages=[{"role": "user", "content": content}],
            modalities=["audio" if task.endswith("s") else "text"],
            stream=stream,
        )
        if task.endswith("s"):
            payload["audio"] = {"format": "pcm" if stream else "wav"}
        prepared.append(
            {
                "id": sample_id,
                "task": task,
                "language": spec["language"],
                "payload": payload,
                "expected": row.get("expected", row.get("text")),
            }
        )
    metadata = {
        k: spec.get(k) for k in ("id", "kind", "task", "language", "revision", "split", "expected")
    }
    metadata.update(samples_hash=digest(prepared), prompt_version=PROMPT_VERSION)
    source = root / spec["source"]
    if source.is_file():
        metadata["source_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return prepared, metadata


def load_workload(path, model, stream, mode):
    path = Path(path).resolve()
    config = read_json(path)
    sampling = config.get("sampling", {})
    allowed = {"seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_tokens"}
    if not isinstance(sampling, dict) or set(sampling) - allowed:
        raise ValueError("Unsupported sampling parameters")
    specs = config["datasets"]
    ids = [s["id"] for s in specs]
    if not specs or len(set(ids)) != len(ids):
        raise ValueError("Dataset IDs must be nonempty and unique")
    if mode in ("daily", "full"):
        expected_ids = set(DAILY) if mode == "daily" else set(PERFORMANCE)
        allowed_ids = [expected_ids] if mode == "daily" else [expected_ids, set(DAILY)]
        if set(ids) not in allowed_ids or any(s["expected"] != DAILY[s["id"]] for s in specs):
            raise ValueError(
                "Daily requires 5279 rows; Full requires all three performance datasets"
            )
        expected_tasks = {
            "seed_en": ("t2s", "en"),
            "seed_zh": ("t2s", "zh"),
            "storycloze": ("t2t", "en"),
            "llama_questions": ("s2s", "en"),
        }
        if any((s["task"], s["language"]) != expected_tasks[s["id"]] for s in specs):
            raise ValueError("Daily/Full dataset task/language mismatch")
    datasets = {}
    for spec in specs:
        if mode == "full" and spec["id"] not in PERFORMANCE:
            continue
        datasets[spec["id"]] = prepare_dataset(
            spec,
            path.parent,
            model=model,
            stream=stream,
            sampling=sampling,
        )
    warmup = None
    if config.get("warmup"):
        warmup = prepare_dataset(
            config["warmup"], path.parent, model=model, stream=stream, sampling=sampling
        )
        payloads = {digest(s["payload"]) for rows, _ in datasets.values() for s in rows}
        if any(digest(s["payload"]) in payloads for s in warmup[0]):
            raise ValueError("Warmup must be disjoint from the formal request cohort")
    return datasets, warmup, sampling
