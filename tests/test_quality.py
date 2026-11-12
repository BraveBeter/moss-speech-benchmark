import pytest
from helpers import args, workload

from moss_speech_bench import quality
from moss_speech_bench.cli import run
from moss_speech_bench.storage import read_json, write_json


async def test_four_modalities_first_round_quality_without_regeneration(
    server, tmp_path, monkeypatch
):
    path = workload(tmp_path, tasks=("t2t", "t2s", "s2t", "s2s"))
    output = tmp_path / "result"
    await run(
        args(
            path,
            server[0],
            output,
            "--samples",
            "llama_questions=2",
            "--repeats",
            "2",
            "--warmup",
            "0",
        )
    )
    before = (output / "performance.json").read_bytes()
    sent = len(server[1]["payloads"])
    calls = []

    class Scorers:
        def __init__(self, config):
            pass

        def transcribe(self, audio, language):
            assert audio.is_file() and language == "en"
            return "Paris"

        def judge(self, sample, answer):
            calls.append(sample["id"])
            return dict(correct=True, valid=True, answer=answer)

        def utmos(self, audio):
            return 4.0

    monkeypatch.setattr(quality, "LocalScorers", Scorers)
    scored = quality.score(output)
    assert scored["status"] == "complete" and len(calls) == 8
    assert len(scored["cells"]) == 4 and all(key.endswith("r1") for key in scored["cells"])
    for name, cell in scored["cells"].items():
        assert cell["metrics"]["accuracy"]["value"] == 1
        assert ("utmos" in cell["metrics"]) == ("t2s" in name or "s2s" in name)
    assert quality.score(output)["cells"] == scored["cells"] and len(calls) == 8
    assert quality.quality_report(output)["cells"] == scored["cells"]
    assert (output / "performance.json").read_bytes() == before and len(
        server[1]["payloads"]
    ) == sent
    frozen_quality = (output / "quality.json").read_bytes()
    changed = read_json(output / "quality.json")
    removed = changed["selection"].pop()
    changed["cells"].pop(removed)
    write_json(output / "quality.json", changed)
    with pytest.raises(ValueError, match="selection mismatch"):
        quality.read_quality(output)
    (output / "quality.json").write_bytes(frozen_quality)
    entry = next(iter(scored["cells"].values()))["records"][0]["path"]
    write_json(output / entry, {"changed": True})
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        quality.read_quality(output)


async def test_missing_and_failed_scores_preserve_denominator_and_retry(
    server, tmp_path, monkeypatch
):
    path = workload(tmp_path, tasks=("t2t",), questions=("Question?", "HTTP_FAIL"))
    output = tmp_path / "result"
    await run(
        args(
            path,
            server[0],
            output,
            "--samples",
            "llama_questions=2",
            "--repeats",
            "1",
            "--warmup",
            "0",
        )
    )
    partial = quality.score(output)
    cell = next(iter(partial["cells"].values()))
    assert partial["status"] == "partial"
    assert cell["metrics"]["accuracy"]["expected"] == 2
    assert cell["metrics"]["accuracy"]["statuses"]["unavailable"] == 1
    monkeypatch.setattr(quality.LocalScorers, "judge", lambda *a: dict(correct=True, valid=True))
    retried = quality.score(output)
    metric = next(iter(retried["cells"].values()))["metrics"]["accuracy"]
    assert retried["status"] == "partial" and metric["scored"] == 1
    assert metric["value"] == 0.5 and metric["evaluated_accuracy"] == 1
    assert list((output / "quality").glob("*/attempts/*/*.json"))


async def test_quality_budget_and_legacy_options_are_explicit(server, tmp_path):
    path = workload(tmp_path, tasks=("t2t",))
    output = tmp_path / "result"
    await run(
        args(
            path,
            server[0],
            output,
            "--samples",
            "llama_questions=1",
            "--repeats",
            "1",
            "--warmup",
            "0",
        )
    )
    with pytest.raises(ValueError, match="positive and finite"):
        quality.score(output, time_budget=float("nan"))
    score = tmp_path / "score.json"
    write_json(score, dict(qa_metric="legacy_em"))
    with pytest.raises(ValueError, match="Unsupported scoring"):
        quality.score(output, score)


def test_codec_error_rates_are_corpus_weighted_and_failures_are_counted():
    rows = [
        dict(
            metrics=dict(
                wer=dict(
                    status="scored",
                    value=dict(
                        substitutions=0, deletions=1, insertions=0, hits=hits, wer=1 / (hits + 1)
                    ),
                )
            )
        )
        for hits in (1, 9)
    ]
    rows.append(dict(metrics=dict(wer=dict(status="failed"))))
    metric = quality.aggregate(rows)["wer"]
    assert metric["expected"] == 3 and metric["scored"] == 2
    assert metric["value"] == 2 / 12 and metric["mean"] == 0.3
