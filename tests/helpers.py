"""Synthetic full-size CSV populations for client tests; no model inference."""

import csv
from pathlib import Path

from moss_speech_bench.cli import parser
from moss_speech_bench.datasets import QA_DATASETS
from moss_speech_bench.storage import digest, write_json


def workload(root, names=("llama_questions",), questions=("Question?",), tasks=("t2s",)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    audio = root / "audio"
    audio.mkdir(exist_ok=True)
    specs = []
    for name in names:
        ids = [f"{name}-{i}.wav" for i in range(QA_DATASETS[name])]
        ranked = sorted(ids, key=lambda key: (digest([name, 42, key]), key))
        selected_questions = {key: questions[i % len(questions)] for i, key in enumerate(ranked)}
        with (root / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("audio_filename", "question", "answers"))
            writer.writeheader()
            for key in ids:
                writer.writerow(
                    dict(audio_filename=key, question=selected_questions[key], answers="Paris")
                )
                (audio / key).write_bytes(b"frozen synthetic input audio")
        specs.append(
            dict(id=name, source=f"{name}.csv", audio_root="audio", revision="test-revision")
        )
    path = root / "e2e.json"
    write_json(
        path,
        dict(
            service=dict(api="chat", streaming=True, tasks=list(tasks)),
            datasets=specs,
            sampling=dict(seed=42, max_tokens=128),
        ),
    )
    return path


def args(config, url, output, *extra):
    return parser().parse_args(
        [
            "e2e",
            "run",
            "--config",
            str(config),
            "--base-url",
            url,
            "--model",
            "test",
            "--output-dir",
            str(output),
            *extra,
        ]
    )
