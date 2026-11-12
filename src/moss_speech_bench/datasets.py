"""The three OpenAudioBench QA datasets and reusable, deterministic selections."""

import base64
import csv
from pathlib import Path

from .service import AUDIO_SYSTEM, TEXT_SYSTEM, service_config
from .storage import digest, file_hash, read_json

QA_DATASETS = {"llama_questions": 300, "trivia_qa": 1000, "web_questions": 1000}
TASKS = ("t2t", "t2s", "s2t", "s2s")
SELECTION_VERSION = "openaudio-selection-v1"
SAMPLER = "sha256-id-v1"
SAMPLING_FIELDS = {
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "max_tokens",
}


def select_samples(spec, root, count, seed, frozen=None, *, require_audio=False):
    """Rank stable IDs, then bind both the population and selected input bytes."""
    name = spec["id"]
    if set(spec) - {"id", "source", "audio_root", "revision", "split"}:
        raise ValueError(f"Unknown dataset fields: {name}")
    if not isinstance(spec.get("revision"), str) or not spec["revision"].strip():
        raise ValueError(f"Dataset revision is required: {name}")
    source = (root / spec["source"]).resolve()
    audio_root = (root / spec.get("audio_root", str(source.parent / "audios"))).resolve()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            dict(
                id=row["audio_filename"],
                question=row.get("question", row.get("Questions")),
                expected=row.get(
                    "Answer", row.get("answers", row.get("answer_normalized_aliases"))
                ),
            )
            for row in csv.DictReader(handle)
        ]
    if len(rows) != QA_DATASETS[name]:
        raise ValueError(f"{name}: expected {QA_DATASETS[name]} rows, found {len(rows)}")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or any(
        not r["id"] or not r["question"] or not r["expected"] for r in rows
    ):
        raise ValueError(f"Empty/duplicate IDs, questions or answers: {name}")
    population = dict(
        revision=spec["revision"],
        split=spec.get("split", name),
        expected=len(rows),
        source_sha256=file_hash(source),
        ids_hash=digest(sorted(by_id)),
    )
    if frozen is not None:
        if frozen.get("population") != population:
            raise ValueError(f"Selection population changed: {name}")
        ids = frozen["ids"]
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError(f"Invalid frozen selection IDs: {name}")
        seed = frozen["seed"]
    else:
        count = len(rows) if count is None else count
        if type(count) is not int or not 0 < count <= len(rows):
            raise ValueError(f"Sample count must be 1..{len(rows)}: {name}")
        ids = sorted(by_id, key=lambda key: (digest([name, seed, key]), key))[:count]
    selected, hashes = [], {}
    for sample_id in ids:
        row = by_id[sample_id]
        audio = (audio_root / sample_id).resolve()
        valid_audio = audio_root in audio.parents and audio.is_file() and audio.stat().st_size > 0
        if require_audio and not valid_audio:
            raise ValueError(f"Missing/invalid dataset audio: {sample_id}")
        hashes[sample_id] = dict(
            content=digest(row), audio=file_hash(audio) if valid_audio else None
        )
        selected.append(dict(row, audio=audio))
    selection = dict(algorithm=SAMPLER, seed=seed, population=population, ids=ids, inputs=hashes)
    if frozen is not None and selection != frozen:
        raise ValueError(f"Selected input hash mismatch: {name}")
    return selected, selection


def request(sample, task, model, stream, sampling, service):
    content = sample["question"]
    if task.startswith("s"):
        audio = sample["audio"]
        content = [
            dict(
                type="input_audio",
                input_audio=dict(
                    data=base64.b64encode(audio.read_bytes()).decode(),
                    format=audio.suffix.lstrip("."),
                ),
            )
        ]
    system = TEXT_SYSTEM if task.endswith("t") else AUDIO_SYSTEM
    payload = dict(sampling, model=model, stream=stream)
    if service["api"] == "speech":
        if "max_tokens" in payload:
            payload["max_new_tokens"] = payload.pop("max_tokens")
        payload.update(input=content, response_format="wav", ref_audio=service["reference_data"])
        endpoint = "/v1/audio/speech"
        system = service["system"]
    else:
        payload.update(
            messages=[dict(role="system", content=system), dict(role="user", content=content)],
            modalities=["audio" if task.endswith("s") else "text"],
        )
        if task.endswith("s") and (stream or "reference_data" in service):
            payload["audio"] = dict(format="pcm" if stream else "wav")
            if "reference_data" in service:
                payload["audio"]["ref_audio"] = service["reference_data"]
        endpoint = "/v1/chat/completions"
    return dict(
        id=sample["id"],
        task=task,
        language="en",
        question=sample["question"],
        expected=sample["expected"],
        payload=payload,
        endpoint=endpoint,
    ), system


def load_workload(
    path,
    model,
    stream,
    *,
    datasets=None,
    tasks=None,
    counts=None,
    seed=42,
    selection_path=None,
    sampling_override=None,
):
    path = Path(path).resolve()
    config = read_json(path)
    if set(config) - {"datasets", "service", "sampling"}:
        raise ValueError("E2E config accepts only datasets, service and sampling")
    specs = {s["id"]: s for s in config["datasets"]}
    if not specs or len(specs) != len(config["datasets"]) or not specs.keys() <= QA_DATASETS.keys():
        raise ValueError("Configure unique llama_questions, trivia_qa or web_questions datasets")
    frozen = read_json(selection_path) if selection_path else None
    if frozen is not None and frozen.get("version") != SELECTION_VERSION:
        raise ValueError("Unsupported selection manifest version")
    if frozen is not None and (counts or seed != 42):
        raise ValueError("--selection replaces --samples and --subset-seed")
    names = (
        datasets if datasets is not None else list(frozen["datasets"]) if frozen else list(specs)
    )
    if not names or len(set(names)) != len(names) or not set(names) <= specs.keys():
        raise ValueError("Selected datasets must be unique configured dataset IDs")
    if frozen is not None and set(names) != set(frozen["datasets"]):
        raise ValueError("Selected datasets differ from selection manifest")
    counts = counts or {}
    if not counts.keys() <= set(names):
        raise ValueError("Sample counts refer to an unselected dataset")
    service = service_config(config["service"], path.parent, stream)
    tasks = tasks if tasks is not None else service["tasks"]
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(service["tasks"]):
        raise ValueError("Tasks must be unique and supported by the service")
    sampling = dict(seed=42, temperature=0, max_tokens=512)
    sampling.update(config.get("sampling", {}))
    sampling.update(sampling_override or {})
    if set(sampling) - SAMPLING_FIELDS:
        raise ValueError("Unsupported sampling parameters")
    if type(sampling["max_tokens"]) is not int or sampling["max_tokens"] <= 0:
        raise ValueError("max_tokens must be a positive integer")
    workloads, selection = {}, dict(version=SELECTION_VERSION, datasets={})
    for name in names:
        rows, chosen = select_samples(
            specs[name],
            path.parent,
            counts.get(name),
            seed,
            frozen["datasets"][name] if frozen else None,
            require_audio=any(task.startswith("s") for task in tasks),
        )
        selection["datasets"][name] = chosen
        for task in tasks:
            prepared = [request(row, task, model, stream, sampling, service) for row in rows]
            requests = [row for row, _ in prepared]
            key = f"{name}-{task}"
            metadata = dict(
                id=key,
                source_dataset=name,
                task=task,
                language="en",
                expected=len(rows),
                selection=chosen,
                samples_hash=digest(requests),
                system=prepared[0][1],
                service={k: v for k, v in service.items() if k != "reference_data"},
            )
            workloads[key] = requests, metadata
    return workloads, selection, sampling
