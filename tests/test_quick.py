import time

import pytest
from test_runner import args, workload

from moss_speech_bench.cli import run
from moss_speech_bench.datasets import QA_DATASETS, load_workload
from moss_speech_bench.quality import quality_report, score
from moss_speech_bench.reporting import report
from moss_speech_bench.storage import read_json, write_json


def daily_workload(root):
    (root / "input.wav").write_bytes(b"fixture")
    specs = []
    for name, count in QA_DATASETS.items():
        rows = [
            dict(id=str(i), text=f"input {i}", expected="A", audio="input.wav")
            for i in range(count)
        ]
        write_json(root / (name + ".json"), rows)
        specs.append(
            dict(
                id=name,
                kind="json",
                source=name + ".json",
                language="en",
                expected=count,
            )
        )
    config = root / "daily.json"
    write_json(config, dict(datasets=specs, sampling=dict(seed=42, max_tokens=512)))
    return config


def test_fixed_selection_and_only_selected_audio_read(tmp_path):
    config = daily_workload(tmp_path)
    original, _, _ = load_workload(config, "test", True, "quick")
    assert all(len(rows) == 20 for rows, _ in original.values())
    for name in QA_DATASETS:
        text, audio = original[name + "-s2t"], original[name + "-s2s"]
        assert text[1]["selection"] == audio[1]["selection"]
        assert text[0][0]["payload"]["modalities"] == ["text"]
        assert audio[0][0]["payload"]["modalities"] == ["audio"]
        assert text[0][0]["payload"]["messages"] == audio[0][0]["payload"]["messages"]
    path = tmp_path / "llama_questions.json"
    rows = read_json(path)
    selected = set(original["llama_questions-s2s"][1]["selection"]["ids"])
    for row in rows:
        if row["id"] not in selected:
            row["audio"] = "missing.wav"
    write_json(path, list(reversed(rows)))
    reordered, _, _ = load_workload(config, "test", True, "quick")
    assert original["llama_questions-s2s"][0] == reordered["llama_questions-s2s"][0]
    with pytest.raises(FileNotFoundError):
        load_workload(config, "test", True, "daily")
    for size in (19, 51):
        with pytest.raises(ValueError, match="20..50"):
            load_workload(config, "test", True, "quick", size)
    # Full population IDs, including unselected rows, must remain unique.
    rows[-1]["id"] = rows[0]["id"]
    write_json(path, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_workload(config, "test", True, "quick")


async def test_quick_inference_cohort_and_resume_contract(server, tmp_path, correctness_stub):
    url, state = server
    config = daily_workload(tmp_path)
    dry = await run(args(config, url, tmp_path / "unused", "--mode", "quick", "--dry-run"))
    assert dry["formal_requests"] == 120 and dry["time_budget_s"] == 2040
    correctness_stub["url"] = url
    a = args(
        config, url, tmp_path / "quick", "--mode", "quick", "--correctness-config", "fixture.json"
    )
    metrics = await run(a)
    assert len(state["payloads"]) == 120 and state["peak"] == 4
    assert all(m["expected"] == m["succeeded"] == 20 for m in metrics.values())
    manifest = read_json(a.output_dir / "manifest.json")
    for name, metadata in manifest["contract"]["datasets"].items():
        assert (
            metadata["selection"]["population_expected"] == QA_DATASETS[metadata["source_dataset"]]
        )
        assert len(metadata["selection"]["ids"]) == 20
    assert "Mode: quick" in (a.output_dir / "summary.md").read_text()
    a.resume = True
    await run(a)
    assert len(state["payloads"]) == 120
    assert score(a.output_dir)["time_budget_s"] == 180
    a.subset_seed = 99
    with pytest.raises(ValueError, match="contract mismatch"):
        await run(a)
    bigger = await run(
        args(
            config,
            url,
            tmp_path / "unused",
            "--mode",
            "quick",
            "--samples-per-dataset",
            "50",
            "--dry-run",
        )
    )
    assert bigger["formal_requests"] == 300
    with pytest.raises(ValueError, match="only supported"):
        await run(args(config, url, tmp_path / "unused", "--samples-per-dataset", "30"))
    with pytest.raises(ValueError, match="exactly one"):
        await run(args(config, url, tmp_path / "unused", "--mode", "quick", "--repeats", "2"))


async def test_inference_budget_retains_incomplete_cohort(server, tmp_path):
    url, _ = server
    config = workload(tmp_path, tuple(str(i) for i in range(30)))
    output = tmp_path / "bounded"
    metrics = await run(args(config, url, output, "--concurrency", "1", "--time-budget", "0.05"))
    assert read_json(output / "manifest.json")["state"] == "budget_exhausted"
    value = next(iter(metrics.values()))
    assert value["expected"] == 30 and value["not_attempted"] > 0
    assert value["throughput_qps"] is None and not value["measurement_complete"]
    report(output)


async def test_scoring_budget_preserves_pending_and_reuses_done(server, tmp_path, monkeypatch):
    from moss_speech_bench import quality

    url, _ = server
    config = workload(tmp_path)
    spec = read_json(config)
    spec["datasets"][0]["task"] = "t2t"
    write_json(config, spec)
    output = tmp_path / "scoring"
    await run(args(config, url, output))
    actual = quality.evaluate
    calls = []

    def slow(*args, **kwargs):
        if args[1] is not None:
            calls.append(args[0]["id"])
            time.sleep(0.04)
        return actual(*args, **kwargs)

    monkeypatch.setattr(quality, "evaluate", slow)
    partial = score(output, time_budget=0.03)
    assert partial["status"] == "partial"
    metric = partial["cells"]["en-c4-r1"]["metrics"]["accuracy"]
    assert metric["statuses"]["budget_exhausted"] > 0
    assert quality_report(output)["status"] == "partial"
    pending = {p: p.read_bytes() for p in (output / "quality").glob("*/attempts/*/*.json")}
    assert pending
    assert score(output, time_budget=2)["status"] == "complete"
    assert len(calls) == 3
    assert all(p.read_bytes() == body for p, body in pending.items())
    assert quality_report(output)["status"] == "complete"


async def test_installed_quick_dry_run(tmp_path):
    import asyncio
    import json
    import os
    import sys

    config = daily_workload(tmp_path)
    python = os.environ.get("BENCHMARK_TEST_PYTHON", sys.executable)
    process = await asyncio.create_subprocess_exec(
        python,
        "-m",
        "moss_speech_bench",
        "--mode",
        "quick",
        "--config",
        str(config),
        "--model",
        "test",
        "--output-dir",
        str(tmp_path / "unused"),
        "--dry-run",
        cwd=tmp_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output, error = await process.communicate()
    assert process.returncode == 0, error.decode()
    data = json.loads(output)
    assert data["mode"] == "quick" and data["formal_requests"] == 120
    assert data["time_budget_s"] == 2040 and len(data["cells"]) == 6
    assert not (tmp_path / "unused").exists()


async def test_named_gate_blocks_http_and_proof_is_required(server, tmp_path, correctness_stub):
    url, state = server
    config = daily_workload(tmp_path)
    correctness_stub.update(url=url, status="fail")
    a = args(
        config, url, tmp_path / "blocked", "--mode", "quick", "--correctness-config", "fixture.json"
    )
    assert (await run(a))["correctness"]["failed"]
    assert not state["payloads"] and not (a.output_dir / "manifest.json").exists()
    blocked_summary = (a.output_dir / "summary.md").read_text()
    assert "被测服务代码：分支 `未记录`" in blocked_summary
    assert "评测客户端代码：分支" in blocked_summary
    correctness_stub["status"] = "pass"
    a.output_dir = tmp_path / "passed"
    await run(a)
    assert len(state["payloads"]) == 120
    saved = read_json(a.output_dir / "summary.json")
    assert saved["correctness_status"] == "pass"
    (a.output_dir / "correctness/correctness.json").write_text("{}")
    with pytest.raises(ValueError, match="proof hash"):
        score(a.output_dir)
    with pytest.raises(ValueError, match="correctness-config"):
        await run(args(config, url, tmp_path / "missing", "--mode", "quick"))


async def test_task_selection_and_quickly_alias(tmp_path):
    config = daily_workload(tmp_path)
    selected = await run(
        args(
            config,
            "http://127.0.0.1:1",
            tmp_path / "unused",
            "--mode",
            "quickly",
            "--tasks",
            "s2s",
            "--dry-run",
        )
    )
    assert selected["mode"] == "quick" and selected["formal_requests"] == 60
    for bad in ["s2s,s2s", "t2s", ""]:
        with pytest.raises(ValueError, match="tasks"):
            await run(
                args(
                    config,
                    "http://127.0.0.1:1",
                    tmp_path / "unused",
                    "--mode",
                    "quick",
                    "--tasks",
                    bad,
                    "--dry-run",
                )
            )


@pytest.mark.parametrize("mode", ["daily", "full"])
async def test_full_and_daily_cannot_bypass_correctness(server, tmp_path, correctness_stub, mode):
    url, state = server
    config = daily_workload(tmp_path)
    correctness_stub.update(url=url, status="fail")
    result = await run(
        args(config, url, tmp_path / mode, "--mode", mode, "--correctness-config", "fixture.json")
    )
    assert result["correctness"]["failed"] and not state["payloads"]


async def test_quick_preserves_outer_execution_records_and_rejects_prior_results(
    server, tmp_path, correctness_stub
):
    url, state = server
    config = daily_workload(tmp_path)
    correctness_stub.update(url=url, status="pass")
    output = tmp_path / "prepared"
    execution = output / "execution"
    execution.mkdir(parents=True)
    record = execution / "identity.json"
    record.write_bytes(b'{"commit":"operator-declared"}\n')
    saved = record.read_bytes()
    arguments = args(config, url, output, "--mode", "quick", "--correctness-config", "fixture.json")
    await run(arguments)
    assert len(state["payloads"]) == 120 and record.read_bytes() == saved
    with pytest.raises(ValueError, match="Output directory is not empty"):
        await run(arguments)
    assert len(state["payloads"]) == 120
    bad = tmp_path / "execution-file"
    bad.mkdir()
    (bad / "execution").write_text("not a directory")
    arguments.output_dir = bad
    with pytest.raises(ValueError, match="Output directory is not empty"):
        await run(arguments)
    assert len(state["payloads"]) == 120


async def test_declared_server_must_match_correctness_candidate(
    server, tmp_path, correctness_stub, monkeypatch
):
    from moss_speech_bench import correctness

    url, state = server
    correctness_stub["url"] = url
    monkeypatch.setattr(correctness, "load_suite", lambda _: dict(base_url=url, core_sha="a" * 40))
    deployment = tmp_path / "deployment.json"
    write_json(deployment, {"server": {"commit": "b" * 40}})
    a = args(
        daily_workload(tmp_path),
        url,
        tmp_path / "run",
        "--mode",
        "quick",
        "--correctness-config",
        "fixture.json",
        "--deployment",
        str(deployment),
    )
    with pytest.raises(ValueError, match="correctness candidate SHA"):
        await run(a)
    assert not state["payloads"] and correctness_stub["calls"] == 0
