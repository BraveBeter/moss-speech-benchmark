import csv

import pytest
from helpers import workload

from moss_speech_bench.datasets import load_workload
from moss_speech_bench.storage import read_json, write_json


def load(path, **kwargs):
    return load_workload(path, "test", True, **kwargs)


def test_deterministic_nested_selection_and_reuse_across_modalities(tmp_path):
    path = workload(tmp_path, tasks=("t2t", "t2s", "s2t", "s2s"))
    _, small, _ = load(path, counts={"llama_questions": 20})
    rows, large, _ = load(path, counts={"llama_questions": 50})
    chosen = large["datasets"]["llama_questions"]["ids"]
    assert small["datasets"]["llama_questions"]["ids"] == chosen[:20]
    assert all([r["id"] for r in requests] == chosen for requests, _ in rows.values())
    assert len(set(chosen)) == 50
    frozen = tmp_path / "selection.json"
    write_json(frozen, large)
    assert load(path, selection_path=frozen)[1] == large
    assert (
        load_workload(path, "another-model", False, tasks=["s2t"], selection_path=frozen)[1]
        == large
    )
    assert load(path, counts={"llama_questions": 50}, seed=99)[1] != large


def test_ranking_independent_of_csv_order_but_reuse_detects_changed_source(tmp_path):
    path = workload(tmp_path)
    _, selected, _ = load(path, counts={"llama_questions": 30})
    frozen = tmp_path / "selection.json"
    write_json(frozen, selected)
    source = tmp_path / "llama_questions.csv"
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reversed(rows))
    again = load(path, counts={"llama_questions": 30})[1]
    assert (
        again["datasets"]["llama_questions"]["ids"]
        == selected["datasets"]["llama_questions"]["ids"]
    )
    with pytest.raises(ValueError, match="population changed"):
        load(path, selection_path=frozen)


def test_reuse_detects_audio_change_and_unknown_selection_version(tmp_path):
    path = workload(tmp_path)
    _, selection, _ = load(path, counts={"llama_questions": 1})
    frozen = tmp_path / "selection.json"
    write_json(frozen, selection)
    first = selection["datasets"]["llama_questions"]["ids"][0]
    (tmp_path / "audio" / first).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input hash mismatch"):
        load(path, selection_path=frozen)
    selection["version"] = "unknown"
    write_json(frozen, selection)
    with pytest.raises(ValueError, match="version"):
        load(path, selection_path=frozen)


@pytest.mark.parametrize(
    "counts", [{"llama_questions": 0}, {"llama_questions": 301}, {"web_questions": 20}]
)
def test_invalid_counts(tmp_path, counts):
    with pytest.raises(ValueError):
        load(workload(tmp_path), counts=counts)


def test_selection_and_capability_validation(tmp_path):
    path = workload(tmp_path)
    with pytest.raises(ValueError, match="supported"):
        load(path, tasks=["t2t"])
    with pytest.raises(ValueError, match="unique configured"):
        load(path, datasets=["llama_questions", "llama_questions"])
    spec = read_json(path)
    spec["datasets"][0]["id"] = "storycloze"
    write_json(path, spec)
    with pytest.raises(ValueError, match="Configure unique"):
        load(path)


@pytest.mark.parametrize("task", ["t2t", "t2s", "s2t", "s2s"])
def test_modality_payload_and_system(tmp_path, task):
    path = workload(tmp_path, tasks=(task,))
    rows, _, _ = load(path, counts={"llama_questions": 1})
    sample = next(iter(rows.values()))[0][0]
    payload = sample["payload"]
    assert payload["modalities"] == ["audio" if task.endswith("s") else "text"]
    assert isinstance(payload["messages"][-1]["content"], list) == task.startswith("s")
    assert ("spoken outputs" in payload["messages"][0]["content"]) == task.endswith("s")
    assert "repeat" not in payload["messages"][0]["content"].lower()
    assert sample["expected"] == "Paris" and sample["question"] == "Question?"
    assert ("audio" in payload) == task.endswith("s")


def test_speech_transport_is_nonstreaming_t2s_and_freezes_voice(tmp_path):
    path = workload(tmp_path)
    config = read_json(path)
    (tmp_path / "voice.wav").write_bytes(b"voice")
    config["service"] = dict(
        api="speech", tasks=["t2s"], streaming=False, reference_audio="voice.wav"
    )
    write_json(path, config)
    with pytest.raises(ValueError, match="no-stream"):
        load(path)
    rows, _, _ = load_workload(path, "hf", False, counts={"llama_questions": 1})
    sample = next(iter(rows.values()))[0][0]
    assert sample["endpoint"] == "/v1/audio/speech"
    assert sample["payload"]["input"] == "Question?"
    assert sample["payload"]["max_new_tokens"] == 128
    assert sample["payload"]["ref_audio"].startswith("data:audio/wav;base64,")


def test_text_inputs_can_run_without_recordings_but_speech_inputs_require_them(tmp_path):
    import shutil

    path = workload(tmp_path, tasks=("t2t", "t2s", "s2t"))
    shutil.rmtree(tmp_path / "audio")
    rows, selection, _ = load(path, tasks=["t2t", "t2s"], counts={"llama_questions": 2})
    assert len(rows) == 2
    assert all(
        item["audio"] is None
        for item in selection["datasets"]["llama_questions"]["inputs"].values()
    )
    with pytest.raises(ValueError, match="Missing/invalid dataset audio"):
        load(path, tasks=["s2t"], counts={"llama_questions": 2})


@pytest.mark.parametrize("task", ["t2s", "s2s"])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_audio_reference_is_frozen_and_sent_for_both_transports(tmp_path, task, stream):
    path = workload(tmp_path, tasks=(task,))
    config = read_json(path)
    (tmp_path / "voice.wav").write_bytes(b"fixed speaker")
    config["service"]["reference_audio"] = "voice.wav"
    write_json(path, config)
    rows, _, _ = load_workload(path, "test", stream, counts={"llama_questions": 1})
    requests, metadata = next(iter(rows.values()))
    audio = requests[0]["payload"]["audio"]
    assert audio["format"] == ("pcm" if stream else "wav")
    assert audio["ref_audio"].startswith("data:audio/wav;base64,")
    assert metadata["service"]["reference_sha256"]
    assert "reference_data" not in metadata["service"]
