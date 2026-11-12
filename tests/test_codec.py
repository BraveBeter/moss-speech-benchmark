import json
import os
import sys

import pytest

from moss_speech_bench import codec
from moss_speech_bench.cli import parser, run
from moss_speech_bench.storage import file_hash, read_json, write_json


def configuration(root):
    target, prompt = root / "target.wav", root / "prompt.wav"
    target.write_bytes(b"target")
    prompt.write_bytes(b"prompt")
    samples = [
        dict(
            id=f"{lang}-{i}",
            language=lang,
            text="Target transcript",
            target_audio="target.wav",
            prompt_audio="prompt.wav",
            target_sha256=file_hash(target),
            prompt_sha256=file_hash(prompt),
        )
        for lang, count in codec.CORPUS_COUNTS.items()
        for i in range(count)
    ]
    corpus = root / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(s) + "\n" for s in samples))
    path = root / "codec.json"
    write_json(
        path,
        dict(
            protocol=codec.PROTOCOL,
            backend="hf-native",
            python=sys.executable,
            physical_gpu=0,
            seed=42,
            decoder_mode="offline",
            codec_path=".",
            codec_revision="fixed-codec",
            hf_source_root=".",
            corpus="corpus.jsonl",
            corpus_revision="fixed-seedtts",
            corpus_sha256=file_hash(corpus),
            assets=[dict(path="target.wav", sha256=file_hash(target))],
        ),
    )
    return path


def test_full_corpus_and_pair_hashes_are_required(tmp_path):
    path = configuration(tmp_path)
    config, samples, _ = codec.load_config(path)
    assert config["python"] == os.path.abspath(sys.executable)
    assert len(samples) == 3108 and samples[0]["prompt_audio"] == str(tmp_path / "prompt.wav")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(corpus.read_text().splitlines()[0] + "\n")
    spec = read_json(path)
    spec["corpus_sha256"] = file_hash(corpus)
    write_json(path, spec)
    with pytest.raises(ValueError, match="complete Seed-TTS-Eval"):
        codec.load_config(path)


@pytest.mark.parametrize(
    "change",
    [
        dict(flow_precision="full_flow_bf16"),
        dict(samples=20),
        dict(backend="sg-bf16"),
        dict(decoder_mode="streaming"),
        dict(physical_gpu=True),
        dict(seed=None),
    ],
)
def test_removed_or_invalid_codec_options_are_rejected(tmp_path, change):
    path = configuration(tmp_path)
    write_json(path, read_json(path) | change)
    with pytest.raises(ValueError):
        codec.load_config(path)


def test_codec_rejects_changed_manifest_audio_and_dirty_source(tmp_path, monkeypatch):
    path = configuration(tmp_path)
    (tmp_path / "prompt.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio hash mismatch"):
        codec.load_config(path)
    path = configuration(tmp_path)
    spec = read_json(path) | dict(backend="sg-native", model_root=".", model_sha="a" * 40)
    write_json(path, spec)
    monkeypatch.setattr(codec, "git_identity", lambda _: dict(commit="a" * 40, dirty=True))
    with pytest.raises(ValueError, match="pinned clean SHA"):
        codec.load_config(path)


def test_codec_worker_failure_keeps_full_plan_and_report(tmp_path, monkeypatch):
    path = configuration(tmp_path)
    monkeypatch.setattr(codec, "run_worker", lambda *args: 1)
    output = tmp_path / "result"
    result = codec.infer(path, output)
    assert result["status"] == "partial" and result["expected"] == 3108
    assert "0/3108" in (output / "summary.md").read_text()
    assert "C00" not in (output / "summary.md").read_text()
    with pytest.raises(ValueError, match="fresh directory"):
        codec.infer(path, output)


async def test_codec_cli_is_separate_from_http_and_has_no_subset(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        codec, "infer", lambda config, root: calls.append((config, root)) or dict(status="complete")
    )
    args = parser().parse_args(
        [
            "codec",
            "run",
            "--config",
            str(tmp_path / "codec.json"),
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    assert await run(args) == {"codec": {"failed": False}}
    assert len(calls) == 1
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["codec", "run", "--config", "codec.json", "--output-dir", "out", "--samples", "20"]
        )


def test_codec_worker_scores_each_reference_and_records_missing_generation(tmp_path, monkeypatch):
    from moss_speech_bench import codec_worker, quality
    from moss_speech_bench._vendor import asr_scoring

    monkeypatch.setattr(asr_scoring, "_get_en_normalizer", lambda: str.lower)
    references = []

    class Scorers:
        def __init__(self, config):
            pass

        def model(self, key):
            assert key == "asr_en"
            return self

        def transcribe(self, path, language=None, **options):
            if language == "en":
                assert options == dict(fp16=False, temperature=0.0, beam_size=5)
                return dict(text="hello")
            assert language == "zh"
            return "你好"

        def sim(self, reference, audio):
            references.append(reference)
            return 0.75

    monkeypatch.setattr(quality, "LocalScorers", Scorers)
    monkeypatch.setattr(quality, "configuration", lambda path: ({}, {}))
    samples, generated = [], []
    for lang, text in [("en", "hello"), ("zh", "你好"), ("en", "missing")]:
        key = lang + text
        samples.append(
            dict(id=key, language=lang, text=text, prompt_audio=key + "-ref.wav", prompt_sha256=key)
        )
        if text == "missing":
            continue
        wav = tmp_path / (key + ".wav")
        wav.write_bytes(lang.encode())
        generated.append(
            dict(id=key, status="complete", wav_path=wav.name, wav_sha256=file_hash(wav))
        )
    write_json(tmp_path / "samples.json", samples)
    write_json(tmp_path / "generation.json", dict(rows=generated))
    result = codec_worker.score(dict(physical_gpu=0), tmp_path)
    assert result["status"] == "partial"
    assert references == ["enhello-ref.wav", "zh你好-ref.wav"]
    assert result["aggregate"]["en"]["wer"]["expected"] == 2
    assert result["aggregate"]["en"]["wer"]["statuses"]["not_attempted"] == 1
    assert result["aggregate"]["zh"]["cer"]["value"] == 0
    assert result["aggregate"]["zh"]["sim"]["value"] == 75


def test_codec_generation_and_wav_integrity_before_reporting(tmp_path, monkeypatch):
    path = configuration(tmp_path)
    output = tmp_path / "result"

    def worker(root, *args):
        wav = root / "audio.wav"
        wav.write_bytes(b"frozen output")
        rows = [
            dict(id=s["id"], status="complete", wav_path="audio.wav", wav_sha256=file_hash(wav))
            for s in read_json(root / "samples.json")
        ]
        write_json(root / "generation.json", dict(status="complete", rows=rows))
        return 0

    monkeypatch.setattr(codec, "run_worker", worker)
    assert codec.infer(path, output)["status"] == "complete"
    (output / "audio.wav").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="audio hash mismatch"):
        codec.render(output)


def test_codec_score_resolves_assets_relative_to_original_config(tmp_path, monkeypatch):
    path = configuration(tmp_path)
    output = tmp_path / "result"
    monkeypatch.setattr(codec, "run_worker", lambda *a: 1)
    codec.infer(path, output)
    score = tmp_path / "scorers.json"
    write_json(
        score,
        dict(
            python=sys.executable,
            physical_gpu=0,
            asr_en=dict(model="whisper-large-v3", path="target.wav"),
            asr_zh=dict(model="paraformer-zh", path="prompt.wav"),
        ),
    )

    def worker(root, phase, config, python, gpu):
        resolved = read_json(config)
        assert resolved["asr_en"]["path"] == str(tmp_path / "target.wav")
        assert resolved["asr_zh"]["path"] == str(tmp_path / "prompt.wav")
        assert python == os.path.abspath(sys.executable)
        write_json(root / "quality.json", dict(status="partial"))
        return 1

    monkeypatch.setattr(codec, "run_worker", worker)
    assert codec.score(output, score)["status"] == "failed"
    assert read_json(output / "quality.json")["status"] == "failed"
    assert codec.render(output)["status"] == "failed"
