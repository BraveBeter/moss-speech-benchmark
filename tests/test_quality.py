import hashlib

import pytest
from moss_speech_bench.cli import run
from moss_speech_bench.quality import aggregate, evaluate, quality_report, score
from moss_speech_bench.reporting import report
from moss_speech_bench.storage import read_json, write_json
from test_runner import args, workload


async def test_offline_qa_and_report_resume(server, tmp_path):
    url, state = server
    config = workload(tmp_path, ("question", "HTTP_FAIL"))
    workload_config = read_json(config)
    workload_config["datasets"][0]["task"] = "t2t"
    write_json(config, workload_config)
    write_json(
        tmp_path / "samples.json",
        [dict(id="0", text="question", expected="A"), dict(id="1", text="HTTP_FAIL", expected="B")],
    )
    output = tmp_path / "result"
    await run(args(config, url, output))
    count = len(state["payloads"])
    performance = (output / "performance.json").read_bytes()
    result = score(output)
    metric = result["cells"]["en-c4-r1"]["metrics"]["accuracy"]
    assert metric["expected"] == 2 and metric["scored"] == 1
    assert metric["value"] == 0 and metric["invalid"] == 1
    assert metric["statuses"]["generation_failed"] == 1
    assert result["status"] == "partial"
    assert score(output) == result
    report(output)
    assert read_json(output / "summary.json")["scored"] == 1
    assert (output / "performance.json").read_bytes() == performance
    assert len(state["payloads"]) == count
    file = next((output / "quality").glob("*/*.json"))
    data = read_json(file)
    data["sample_id"] = "corrupted"
    write_json(file, data)
    with pytest.raises(ValueError, match="hash mismatch"):
        quality_report(output)
    with pytest.raises(ValueError, match="cache hash mismatch"):
        score(output)


def test_cer_and_sim_use_original_output_and_voice_identity():
    class Backend:
        def transcribe(self, path, lang):
            assert path == "saved.wav" and lang == "zh"
            return "你！"

        def sim(self, reference, path):
            assert reference == "voice.wav" and path == "saved.wav"
            return 0.8

    config = dict(sim={"reference": "voice.wav"}, _voice_sha256="abc", _reference_sha256="abc")
    sample = dict(task="t2s", language="zh", expected="你好")
    metrics = evaluate(sample, {"is_success": True}, "saved.wav", Backend(), config)
    assert metrics["cer"]["value"]["deletions"] == 1
    assert metrics["cer"]["value"]["wer"] == 0.5
    assert metrics["sim"]["value"] == 0.8
    config["_voice_sha256"] = None
    assert (
        evaluate(sample, {"is_success": True}, "saved.wav", Backend(), config)["sim"]["status"]
        == "unavailable"
    )


def test_qa_aliases_and_missing_models():
    class Backend:
        def transcribe(self, path, lang):
            return "The Paris!"

        def utmos(self, path):
            raise ImportError("optional model missing")

    sample = dict(task="s2s", language="en", expected="['Paris', 'City of Paris']")
    values = evaluate(sample, dict(is_success=True), "audio.wav", Backend(), {})
    assert values["accuracy"]["value"]["correct"]
    assert values["utmos"]["status"] == "unavailable"
    assert evaluate(sample, None, None, Backend(), {})["accuracy"]["status"] == "not_attempted"
    sample["expected"] = "__import__('os').system('false')"
    assert not evaluate(sample, dict(is_success=True), "audio.wav", Backend(), {})["accuracy"][
        "value"
    ]["correct"]


def test_corpus_rate_is_not_mean_and_missing_is_not_zero():
    records = [
        dict(
            metrics={
                "cer": dict(
                    status="scored",
                    value=dict(substitutions=0, deletions=1, insertions=0, hits=n, wer=1 / (n + 1)),
                )
            }
        )
        for n in (1, 9)
    ]
    records.append(dict(metrics={"cer": dict(status="unavailable")}))
    summary = aggregate(records)["cer"]
    assert summary["value"] == 2 / 12 and summary["mean"] == 0.3
    assert summary["scored"] == 2 and summary["expected"] == 3
    assert (
        aggregate([dict(metrics={"accuracy": dict(status="unavailable")})])["accuracy"]["value"]
        is None
    )


async def test_score_cli_without_service_and_declared_assets(server, tmp_path):
    url, state = server
    config = workload(tmp_path, ("hello",))
    output = tmp_path / "result"
    await run(args(config, url, output))
    # No scoring models configured: explicit unavailable, not a fake quality pass.
    a = args(config, "http://127.0.0.1:1", output, "--phase", "score")
    result = await run(a)
    assert result["quality"]["failed"]
    assert len(state["payloads"]) == 1
    quality = quality_report(output)
    assert quality["status"] == "partial"
    assert all(v["value"] is None for v in quality["cells"]["en-c4-r1"]["metrics"].values())
    from moss_speech_bench.quality import configuration

    asset = tmp_path / "voice.wav"
    asset.write_bytes(b"voice")
    path = tmp_path / "scoring.json"
    write_json(path, {"sim": {"reference": "voice.wav"}})
    _, contract = configuration(path)
    assert contract["assets"]["sim.reference"]["voice.wav"] == hashlib.sha256(b"voice").hexdigest()


async def test_quality_baseline_and_reference_cell_selection(server, tmp_path):
    url, _ = server
    config = workload(tmp_path, ("choice",))
    spec = read_json(config)
    spec["datasets"][0]["task"] = "t2t"
    write_json(config, spec)
    write_json(tmp_path / "samples.json", [dict(id="0", text="choice", expected="A")])
    deployment = tmp_path / "deployment.json"
    write_json(deployment, {"model_revision": "fixed"})
    for name in ("base", "candidate"):
        a = args(
            config, url, tmp_path / name, "--concurrencies", "1,4", "--deployment", str(deployment)
        )
        await run(a)
        result = score(a.output_dir)
        assert result["selection"] == ["en-c4-r1"]
        assert len(result["cells"]) == 1
    report(tmp_path / "candidate", tmp_path / "base")
    comparison = read_json(tmp_path / "candidate/comparison.json")["quality"]
    assert comparison["compatible"]
    assert comparison["cells"]["en-c4-r1"]["accuracy"]["delta"] == 0
    assert read_json(tmp_path / "candidate/summary.json")["quality_expected"] == 1


async def test_installed_offline_score_entrypoint(server, tmp_path):
    import asyncio
    import os
    import sys

    url, state = server
    config = workload(tmp_path, ("question",))
    data = read_json(config)
    data["datasets"][0]["task"] = "t2t"
    write_json(config, data)
    write_json(tmp_path / "samples.json", [dict(id="0", text="question", expected="A")])
    output = tmp_path / "run"
    await run(args(config, url, output))
    executable = os.environ.get("BENCHMARK_TEST_PYTHON", sys.executable)
    process = await asyncio.create_subprocess_exec(
        executable,
        "-m",
        "moss_speech_bench",
        "--phase",
        "score",
        "--output-dir",
        str(output),
        "--base-url",
        "http://127.0.0.1:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tmp_path,
    )
    _, error = await process.communicate()
    assert process.returncode == 0, error.decode()
    assert quality_report(output)["status"] == "complete"
    assert len(state["payloads"]) == 1
