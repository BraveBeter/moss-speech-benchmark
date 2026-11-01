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


def test_seedtts_does_not_fall_back_to_chat(tmp_path):
    (tmp_path / "meta.lst").write_text("s1|reference text|ref.wav|target text\n")
    with pytest.raises(ValueError, match="Codec Decoder"):
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


def test_explicit_prompt_preserves_reference_and_identity(tmp_path):
    write_json(tmp_path / "samples.json", [{"id": "x", "text": "目标原文", "expected": "目标原文"}])
    spec = dict(
        id="test", kind="json", source="samples.json", task="t2s", language="zh", expected=1
    )
    default, before = prepare_dataset(spec, tmp_path, model="moss", stream=True, sampling={})
    spec["prompt"] = {
        "name": "tts-system-zh-v1",
        "system": "只用中文朗读，不翻译。",
        "template": "{text}",
    }
    rows, metadata = prepare_dataset(spec, tmp_path, model="moss", stream=True, sampling={})
    assert rows[0]["expected"] == "目标原文"
    assert rows[0]["payload"]["messages"] == [
        {"role": "system", "content": "只用中文朗读，不翻译。"},
        {"role": "user", "content": "目标原文"},
    ]
    assert before["samples_hash"] != metadata["samples_hash"] and metadata["prompt_hash"]
    assert default[0]["payload"]["messages"][0]["content"].startswith("请逐字朗读")
    spec["prompt"]["template"] = "{text} {other}"
    with pytest.raises(ValueError, match="exactly one"):
        prepare_dataset(spec, tmp_path, model="moss", stream=True, sampling={})


def test_speech_prompt_changes_system_only(tmp_path):
    (tmp_path / "audio.wav").write_bytes(b"audio")
    write_json(tmp_path / "samples.json", [{"id": "x", "audio": "audio.wav", "expected": "Moscow"}])
    spec = dict(
        id="test",
        kind="json",
        source="samples.json",
        task="s2s",
        language="en",
        expected=1,
        prompt={"name": "qa-short-v1", "system": "Answer with a short phrase."},
    )
    rows, _ = prepare_dataset(spec, tmp_path, model="moss", stream=False, sampling={})
    assert rows[0]["payload"]["messages"][0]["role"] == "system"
    assert rows[0]["payload"]["messages"][1]["content"][0]["type"] == "input_audio"
    assert rows[0]["expected"] == "Moscow"


def test_reference_pcm_reader_preserves_signed_samples(tmp_path):
    import wave

    import numpy as np

    from moss_speech_bench.providers.hf_http import read_pcm16

    values = np.array([-32768, -1, 0, 1, 32767], dtype="<i2")
    path = tmp_path / "reference.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(values.tobytes())
    samples, rate = read_pcm16(path)
    assert rate == 24000 and samples.shape == (1, 5)
    np.testing.assert_array_equal(samples[0], values.astype(np.float32) / np.float32(32768))


@pytest.mark.parametrize("task", [None, "mixed", "video", "S2T"])
def test_mixed_rows_reject_missing_or_unsupported_task(tmp_path, task):
    write_json(tmp_path / "mixed.json", [dict(id="one", task=task, text="hello")])
    with pytest.raises(ValueError, match="per-row task"):
        prepare(tmp_path, "json", "mixed.json", "mixed")
