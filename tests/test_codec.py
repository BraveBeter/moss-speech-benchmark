"""Codec-only contracts preserve targets, per-sample voices and complete denominators."""

import sys
from pathlib import Path

import pytest

from moss_speech_bench import codec
from moss_speech_bench.cli import parser, run
from moss_speech_bench.storage import read_json, write_json


def configuration(tmp_path):
    target, prompt = tmp_path / "target.wav", tmp_path / "prompt.wav"
    target.write_bytes(b"target recording")
    prompt.write_bytes(b"same-speaker reference")
    sample = dict(
        id="en-01",
        language="en",
        text="Target transcript",
        target_audio=str(target),
        prompt_audio=str(prompt),
        target_sha256=codec.file_hash(target),
        prompt_sha256=codec.file_hash(prompt),
    )
    corpus = tmp_path / "corpus.jsonl"
    import json

    corpus.write_text(json.dumps(sample) + "\n")
    spec = dict(
        protocol=codec.PROTOCOL,
        backend="hf-native",
        python=sys.executable,
        physical_gpu=0,
        seed=42,
        decoder_mode="offline",
        codec_path=str(tmp_path),
        codec_revision="frozen",
        hf_source_root=str(tmp_path),
        corpus=str(corpus),
        assets=[dict(path=str(target), sha256=codec.file_hash(target))],
    )
    path = tmp_path / "config.json"
    write_json(path, spec)
    return path, sample


def test_codec_pairs_use_actual_target_text_and_each_reference(tmp_path):
    path, sample = configuration(tmp_path)
    config, samples, identity = codec.load_config(path)
    assert samples == [sample] and identity["kind"] == "hf-native"
    assert config["decoder_mode"] == "offline"
    Path(sample["prompt_audio"]).write_bytes(b"another voice")
    with pytest.raises(ValueError, match="audio hash mismatch"):
        codec.load_config(path)


@pytest.mark.parametrize("precision", ["ffn_attention", "ffn_attention_sdpa"])
def test_codec_bf16_profiles_require_explicit_clean_source(tmp_path, monkeypatch, precision):
    path, _ = configuration(tmp_path)
    spec = read_json(path) | dict(
        backend="sg-bf16", flow_precision=precision, model_root=str(tmp_path), model_sha="a" * 40
    )
    write_json(path, spec)
    monkeypatch.setattr(codec, "git_identity", lambda _: dict(commit="a" * 40, dirty=False))
    loaded, _, _ = codec.load_config(path)
    assert loaded["flow_precision"] == precision
    monkeypatch.setattr(codec, "git_identity", lambda _: dict(commit="a" * 40, dirty=True))
    with pytest.raises(ValueError, match="pinned clean SHA"):
        codec.load_config(path)


@pytest.mark.parametrize(
    "change",
    [
        dict(protocol="unknown"),
        dict(backend="ar"),
        dict(physical_gpu=True),
        dict(seed=None),
        dict(decoder_mode="streaming"),
        dict(flow_precision="ffn_attention"),
    ],
)
def test_codec_rejects_undeclared_execution_contracts(tmp_path, change):
    path, _ = configuration(tmp_path)
    write_json(path, read_json(path) | change)
    with pytest.raises(ValueError):
        codec.load_config(path)


def test_codec_quality_corpus_micro_average_retains_missing(tmp_path):
    rows = [
        dict(
            language="en",
            metrics=dict(
                wer=dict(
                    status="scored",
                    value=dict(
                        substitutions=0, deletions=1, insertions=0, hits=count, wer=1 / (count + 1)
                    ),
                ),
                sim=dict(status="scored", value=score),
            ),
        )
        for count, score in [(1, 70), (9, 80)]
    ]
    rows += [
        dict(language="en", metrics=dict(wer=dict(status="failed"), sim=dict(status="failed")))
    ]
    result = codec.summarize_quality(rows)["en"]
    assert result["wer"]["value"] == 2 / 12 and result["wer"]["mean"] == 0.3
    assert result["wer"]["scored"] == 2 and result["wer"]["expected"] == 3
    assert result["sim"]["value"] == 75


def test_codec_generation_failure_cannot_become_complete(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    output = tmp_path / "result"
    monkeypatch.setattr(codec, "run_worker", lambda *arguments: 1)
    result = codec.infer(path, output)
    assert result["status"] == "failed" and result["expected"] == 1
    with pytest.raises(ValueError, match="complete"):
        codec.score(output, path)
    with pytest.raises(ValueError, match="fresh directory"):
        codec.infer(path, output)


def test_codec_scorer_model_change_is_rejected_before_worker(tmp_path):
    root = tmp_path / "result"
    write_json(root / "manifest.json", dict(protocol=codec.PROTOCOL, status="complete"))
    config = tmp_path / "score.json"
    write_json(config, dict(asr_en=dict(model="qwen3-asr"), asr_zh=dict(model="paraformer-zh")))
    with pytest.raises(ValueError, match="Whisper-large-v3"):
        codec.score(root, config)


def test_codec_comparison_requires_same_corpus_and_encoder_codes(tmp_path):
    runs = {}
    for name in ["hf", "sg"]:
        root = tmp_path / name
        write_json(
            root / "manifest.json",
            dict(status="complete", contract=dict(corpus_hash="same"), client={}, source={}),
        )
        write_json(
            root / "generation.json",
            dict(rows=[dict(id="en-01", codes_sha256=name, prompt_codes_sha256="same")]),
        )
        write_json(
            root / "quality.json",
            dict(
                status="complete",
                scoring_contract=dict(assets={"scorer": "frozen"}),
                aggregate={
                    "en": dict(wer=dict(value=0), sim=dict(value=80)),
                    "zh": dict(cer=dict(value=0), sim=dict(value=80)),
                },
            ),
        )
        runs[name] = str(root)
    result = codec.compare(runs, tmp_path / "comparison")
    assert result["status"] == "partial" and result["encoder_codes_exact"] is False
    path = tmp_path / "sg/manifest.json"
    changed = read_json(path)
    changed["contract"]["corpus_hash"] = "other"
    write_json(path, changed)
    with pytest.raises(ValueError, match="identical corpus"):
        codec.compare(runs, tmp_path / "other")


async def test_codec_cli_routes_without_chat_service(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    called = []

    def infer(config, root):
        called.append((config, root))
        return dict(status="complete")

    monkeypatch.setattr(codec, "infer", infer)
    args = parser().parse_args(
        ["--phase", "codec-infer", "--config", str(path), "--output-dir", str(tmp_path / "result")]
    )
    assert await run(args) == {"codec": {"failed": False}}
    assert called == [(path, tmp_path / "result")]


def test_codec_worker_scores_each_prompt_and_freezes_english_decoding(tmp_path, monkeypatch):
    from moss_speech_bench import codec_worker, quality
    from moss_speech_bench._vendor import asr_scoring

    monkeypatch.setattr(asr_scoring, "_get_en_normalizer", lambda: str.lower)
    references = []

    class Scorers:
        def __init__(self, settings):
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
    for language, text in [("en", "hello"), ("zh", "你好")]:
        wav = tmp_path / (language + ".wav")
        wav.write_bytes(language.encode())
        samples.append(
            dict(
                id=language,
                language=language,
                text=text,
                prompt_audio=language + "-reference.wav",
                prompt_sha256=language,
            )
        )
        generated.append(
            dict(id=language, status="complete", wav_path=wav.name, wav_sha256=codec.file_hash(wav))
        )
    write_json(tmp_path / "samples.json", samples)
    write_json(tmp_path / "generation.json", dict(rows=generated))
    result = codec_worker.score(dict(physical_gpu=0), tmp_path)
    assert result["status"] == "complete"
    assert references == ["en-reference.wav", "zh-reference.wav"]
    assert result["aggregate"]["en"]["wer"]["value"] == 0
    assert result["aggregate"]["zh"]["cer"]["value"] == 0
    assert result["aggregate"]["en"]["sim"]["value"] == 75
