import base64
import csv

import pytest
from moss_speech_bench.datasets import prepare_dataset
from moss_speech_bench.storage import write_json


def prepare(tmp_path, kind, source, task, **kw):
    spec = dict(id="test", kind=kind, source=source, task=task, language="en", expected=1, **kw)
    return prepare_dataset(spec, tmp_path, model="moss", stream=True, sampling={"seed": 42})


def test_storycloze_and_speech_input_mapping(tmp_path):
    story = {"InputStoryid": "story-1", "AnswerRightEnding": "2"}
    story.update({f"InputSentence{i}": f"Sentence {i}." for i in range(1, 5)})
    story.update(RandomFifthSentenceQuiz1="Wrong ending", RandomFifthSentenceQuiz2="Right ending")
    with (tmp_path / "story.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=story)
        writer.writeheader()
        writer.writerow(story)
    rows, metadata = prepare(tmp_path, "storycloze", "story.csv", "t2t")
    assert rows[0]["expected"] == "B"
    assert "A: Wrong ending\nB: Right ending" in rows[0]["payload"]["messages"][0]["content"]
    assert metadata["source_file_sha256"] and metadata["samples_hash"]
    (tmp_path / "question.wav").write_bytes(b"test audio")
    (tmp_path / "audio.csv").write_text(
        "audio_filename,answer_normalized_aliases\nquestion.wav,Paris\n"
    )
    rows, _ = prepare(tmp_path, "openaudio", "audio.csv", "s2s")
    payload = rows[0]["payload"]
    assert payload["modalities"] == ["audio"]
    audio = payload["messages"][0]["content"][0]["input_audio"]
    assert base64.b64decode(audio["data"]) == b"test audio"
    assert rows[0]["expected"] == "Paris"


def test_seedtts_reads_target_and_rejects_bad_lines(tmp_path):
    meta = tmp_path / "meta.lst"
    meta.write_text("s1|reference text|ref.wav|target text\n")
    rows, _ = prepare(tmp_path, "seedtts", "meta.lst", "t2s")
    assert rows[0]["expected"] == "target text"
    assert rows[0]["payload"]["messages"][0]["content"].endswith("target text")
    assert "ref.wav" not in str(rows[0]["payload"])
    meta.write_text("s1|bad\n")
    with pytest.raises(ValueError, match="Malformed"):
        prepare(tmp_path, "seedtts", "meta.lst", "t2s")


def test_duplicate_ids_and_formal_warmup_overlap_rejected(tmp_path):
    from moss_speech_bench.datasets import load_workload

    write_json(tmp_path / "samples.json", [{"id": "same", "text": "hello"}] * 2)
    spec = dict(
        id="test", kind="json", source="samples.json", task="t2s", language="en", expected=2
    )
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_dataset(spec, tmp_path, model="moss", stream=True, sampling={})
    write_json(tmp_path / "samples.json", [{"id": "same", "text": "hello"}])
    spec["expected"] = 1
    write_json(tmp_path / "config.json", {"datasets": [spec], "warmup": spec})
    with pytest.raises(ValueError, match="disjoint"):
        load_workload(tmp_path / "config.json", "moss", True, "custom")
