"""Whole-dataset loading and deterministic MOSS request construction."""

import base64
import csv
import hashlib
import re
from pathlib import Path

from .storage import digest, read_json

QA_DATASETS = {"llama_questions": 300, "trivia_qa": 1000, "web_questions": 1000}
QA_TASKS = ("s2t", "s2s")
SUPPORTED_QA_TASKS = {"t2t", "t2s", "s2t", "s2s"}
QUICK_SAMPLES = 20
QUICK_INFER_BUDGET_S = 2040
QUICK_SCORE_BUDGET_S = 180
SUITE_VERSION = "moss-qa-v6-capabilities"
NATIVE_TEXT_SYSTEM = "You are a helpful assistant. Respond with text outputs."
NATIVE_AUDIO_SYSTEM = "You are a helpful assistant. Respond with spoken outputs."
DEMO_AUDIO_SYSTEM = (
    "You are a helpful voice assistant. Answer the user's questions with spoken responses."
)


def deferred_evaluations(mode):
    if mode != "full":
        return {}
    return {
        "codec_decoder": {
            "status": "deferred",
            "datasets": {"seed_en": 1088, "seed_zh": 2020},
            "reason": "Isolated Codec Decoder protocol is not implemented; no chat fallback.",
        }
    }


TASKS = {"t2t", "t2s", "s2t", "s2s"}
PROMPT_VERSION = "moss-chat-v1"


def _records(spec, root):
    source = spec["source"]
    path = (root / source).resolve()
    kind = spec["kind"]
    if kind == "seedtts":
        raise ValueError("Seed-TTS-Eval is reserved for Codec Decoder evaluation, not implemented")
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
                "question": row.get("question", row.get("Questions")),
                "expected": row.get(
                    "Answer", row.get("answers", row.get("answer_normalized_aliases"))
                ),
            }
            for row in rows
        ]
    raise ValueError(f"Unknown dataset kind: {kind}")


def prepare_dataset(
    spec, root, *, model, stream, sampling, subset_size=None, subset_seed=42, service=None
):
    task = spec["task"]
    if task not in TASKS | {"mixed"} or spec["language"] not in ("en", "zh"):
        raise ValueError("Invalid task or language")
    prompt = spec.get("prompt", {})
    if not isinstance(prompt, dict) or set(prompt) - {"template", "system", "name"}:
        raise ValueError("Invalid prompt configuration")
    if "template" in prompt and (
        not isinstance(prompt["template"], str)
        or prompt["template"].count("{text}") != 1
        or re.findall(r"\{[^{}]*\}", prompt["template"]) != ["{text}"]
    ):
        raise ValueError("Prompt template must contain exactly one {text} field")
    if any(not isinstance(v, str) or not v.strip() for v in prompt.values()):
        raise ValueError("Prompt values must be nonempty strings")
    if task.startswith("s") and "template" in prompt:
        raise ValueError("Speech input accepts a system prompt, not a text template")
    records = _records(spec, root)
    expected = spec["expected"]
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected <= 0
        or len(records) != expected
    ):
        raise ValueError(f"{spec['id']}: expected {expected} rows, found {len(records)}")
    selection = None
    if subset_size is not None:
        if not 0 < subset_size <= len(records):
            raise ValueError("Subset exceeds source population")
        ids = [str(r["id"]) for r in records]
        if len(set(ids)) != len(ids) or any(not i for i in ids):
            raise ValueError("Duplicate/empty sample id in source population")
        records = sorted(records, key=lambda r: digest([spec["id"], subset_seed, str(r["id"])]))[
            :subset_size
        ]
        selection = dict(
            algorithm="sha256-id-v1",
            seed=subset_seed,
            size=subset_size,
            ids=[str(r["id"]) for r in records],
            population_expected=expected,
            population_ids_hash=digest(sorted(ids)),
        )
    prepared, seen = [], set()
    for row in records:
        row_task = row.get("task") if task == "mixed" else task
        if row_task not in TASKS or (task != "mixed" and "task" in row and row["task"] != task):
            raise ValueError("Invalid or conflicting per-row task")
        if row_task.startswith("s") and "template" in prompt:
            raise ValueError("Speech input accepts a system prompt, not a text template")
        sample_id = str(row["id"])
        if not sample_id or sample_id in seen:
            raise ValueError(f"Duplicate/empty sample id: {sample_id!r}")
        seen.add(sample_id)
        if row_task.startswith("s"):
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
            content = row.get("text", row.get("question"))
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Empty text: {sample_id}")
            if "template" in prompt:
                content = prompt["template"].replace("{text}", content)
        payload = dict(sampling)
        payload.update(
            model=model,
            messages=[{"role": "user", "content": content}],
            modalities=["audio" if row_task.endswith("s") else "text"],
            stream=stream,
        )
        if "system" in prompt:
            payload["messages"].insert(0, {"role": "system", "content": prompt["system"]})
        if row_task.endswith("s") and stream:
            payload["audio"] = {"format": "pcm"}
        if service and service["api"] == "speech":
            if row_task != "t2s" or stream:
                raise ValueError("Speech API supports only non-streaming t2s")
            payload = dict(sampling)
            if "max_tokens" in payload:
                payload["max_new_tokens"] = payload.pop("max_tokens")
            payload.update(
                model=model,
                input=content,
                stream=False,
                response_format="wav",
                ref_audio=service["reference_data"],
            )
        prepared.append(
            {
                "id": sample_id,
                "task": row_task,
                "language": spec["language"],
                "payload": payload,
                "expected": row.get("expected", row.get("text")),
                **({"quality_task": spec["quality_task"]} if spec.get("quality_task") else {}),
                **(
                    {"endpoint": "/v1/audio/speech"}
                    if service and service["api"] == "speech"
                    else {}
                ),
                **(
                    {"question": row.get("question", row.get("text"))}
                    if row.get("question") or (spec["id"] in QA_DATASETS and row.get("text"))
                    else {}
                ),
            }
        )
    metadata = {
        k: spec.get(k) for k in ("id", "kind", "task", "language", "revision", "split", "expected")
    }
    metadata.update(samples_hash=digest(prepared), prompt_version=PROMPT_VERSION)
    if prompt:
        metadata.update(prompt=prompt, prompt_hash=digest(prompt))
    if selection:
        metadata.update(expected=len(prepared), selection=selection)
    if service:
        metadata["service"] = {k: v for k, v in service.items() if k != "reference_data"}
    source = root / spec["source"]
    if source.is_file():
        metadata["source_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return prepared, metadata


def load_workload(path, model, stream, mode, subset_size=QUICK_SAMPLES, subset_seed=42, tasks=None):
    path = Path(path).resolve()
    config = read_json(path)
    service = config.get("service")
    if service is not None:
        from .service import service_config

        service = service_config(service, path.parent, stream)
    sampling = config.get("sampling", {})
    allowed = {"seed", "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "max_tokens"}
    if not isinstance(sampling, dict) or set(sampling) - allowed:
        raise ValueError("Unsupported sampling parameters")
    specs = config["datasets"]
    ids = [s["id"] for s in specs]
    if not specs or len(set(ids)) != len(ids):
        raise ValueError("Dataset IDs must be nonempty and unique")
    named = mode in ("quick", "daily", "full")
    task_prompts = config.get("task_prompts", {})
    if (
        not isinstance(task_prompts, dict)
        or set(task_prompts) - SUPPORTED_QA_TASKS
        or any(not isinstance(prompt, dict) for prompt in task_prompts.values())
        or (task_prompts and not named)
    ):
        raise ValueError("task_prompts requires named speech-QA tasks mapped to prompt objects")
    if tasks is not None and not named:
        raise ValueError("Task selection is only supported in named speech-QA suites")
    if mode == "quick" and (type(subset_size) is not int or not 20 <= subset_size <= 50):
        raise ValueError("Quick sample size must be 20..50 questions per dataset")
    if named:
        if set(ids) != set(QA_DATASETS) or any(
            type(s["expected"]) is not int or s["expected"] != QA_DATASETS[s["id"]] for s in specs
        ):
            raise ValueError("Speech-QA suites require LLaMA 300, TriviaQA 1000, WebQuestions 1000")
        if any(s["kind"] not in ("json", "openaudio") for s in specs):
            raise ValueError("Named speech-QA suites require audio datasets")
        if any(s["language"] != "en" or "task" in s for s in specs):
            raise ValueError("Named speech-QA suites require English and tasks, not a single task")
    datasets = {}
    for spec in specs:
        selected_tasks = (
            tasks
            if tasks is not None
            else spec.get("tasks", service["tasks"] if service else QA_TASKS)
        )
        if named and (
            not isinstance(selected_tasks, (list, tuple))
            or not selected_tasks
            or len(set(selected_tasks)) != len(selected_tasks)
            or not set(selected_tasks) <= (SUPPORTED_QA_TASKS if service else set(QA_TASKS))
        ):
            raise ValueError("QA tasks must be unique t2t/t2s/s2t/s2s values")
        if service and not set(selected_tasks) <= set(service["tasks"]):
            raise ValueError("Selected tasks exceed declared service capabilities")
        for task in selected_tasks if named else [spec["task"]]:
            resolved = dict(spec, task=task)
            if named:
                resolved["quality_task"] = "qa"
                official = NATIVE_TEXT_SYSTEM if task.endswith("t") else NATIVE_AUDIO_SYSTEM
                if service and service["api"] == "speech":
                    official = service["system"]
                supplied = task_prompts.get(task, spec.get("prompt", {}))
                if supplied and supplied != {"system": official}:
                    raise ValueError("Named suites require the official default system prompt")
                resolved["prompt"] = {"system": official}
            rows, metadata = prepare_dataset(
                resolved,
                path.parent,
                model=model,
                stream=stream,
                sampling=sampling,
                subset_size=subset_size if mode == "quick" else None,
                subset_seed=subset_seed,
                service=service,
            )
            name = f"{spec['id']}-{task}" if named else spec["id"]
            if named:
                metadata.update(id=name, source_dataset=spec["id"], suite_version=SUITE_VERSION)
            datasets[name] = rows, metadata
    warmup = None
    if config.get("warmup"):
        warmup = prepare_dataset(
            config["warmup"], path.parent, model=model, stream=stream, sampling=sampling
        )
        payloads = {digest(s["payload"]) for rows, _ in datasets.values() for s in rows}
        if any(digest(s["payload"]) in payloads for s in warmup[0]):
            raise ValueError("Warmup must be disjoint from the formal request cohort")
    return datasets, warmup, sampling
