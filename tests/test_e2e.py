import asyncio
import sys

import pytest
from helpers import args, workload

from moss_speech_bench.cli import parser, run
from moss_speech_bench.reporting import report
from moss_speech_bench.storage import read_json, write_json


async def test_run_warmup_sweep_repeat_rebuild_and_resume(server, tmp_path):
    url, state = server
    path = workload(tmp_path)
    output = tmp_path / "result"
    a = args(
        path,
        url,
        output,
        "--samples",
        "llama_questions=3",
        "--concurrency",
        "1",
        "2",
        "--repeats",
        "2",
        "--warmup",
        "5",
    )
    metrics = await run(a)
    assert len(metrics) == 4 and len(state["payloads"]) == 32
    assert state["peak"] == 2
    assert all(m["succeeded"] == 3 and m["length_count"] == 3 for m in metrics.values())
    assert read_json(output / "plan.json")["quality_requests"] == 6
    frozen = (output / "performance.json").read_bytes()
    report(output)
    assert (output / "performance.json").read_bytes() == frozen
    a.resume = True
    await run(a)
    assert len(state["payloads"]) == 32
    text = (output / "summary.md").read_text()
    assert "正确性" not in text and "Quick" not in text
    assert "TPS" in text and "C50 (%)" in text and "P95" in text
    assert "first-selected-cyclic" in (output / "manifest.json").read_text()
    audio = next(output.glob("cells/**/outputs/*.wav"))
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        report(output)


async def test_disabled_warmup_failures_unknown_usage_and_buffered_metrics(server, tmp_path):
    url, state = server
    path = workload(tmp_path, questions=("HTTP_FAIL", "NO_USAGE"))
    output = tmp_path / "result"
    metrics = await run(
        args(
            path,
            url,
            output,
            "--samples",
            "llama_questions=2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--no-stream",
        )
    )
    value = next(iter(metrics.values()))
    assert value["attempted"] == 2 and value["failed"] == 1
    assert value["output_throughput"] is None and value["audio_ttfp_p95_s"] is None
    assert value["c50"] is None and value["max_playback_underrun_max_s"] is None
    assert len(state["payloads"]) == 2 and not read_json(output / "warmup.json")
    assert read_json(output / "summary.json")["suite_status"] == "incomplete"


async def test_dry_run_all_data_or_individual_counts_without_output(tmp_path):
    path = workload(
        tmp_path,
        names=("llama_questions", "trivia_qa", "web_questions"),
        tasks=("t2t", "t2s", "s2t", "s2s"),
    )
    output = tmp_path / "unused"
    a = args(path, "http://localhost:1", output, "--dry-run")
    plan = await run(a)
    assert plan["formal_requests"] == 2300 * 4 * 3
    assert plan["warmup_requests"] == 3 * 4 * 3 * 4
    assert plan["quality_requests"] == 2300 * 4
    assert not output.exists()
    a.datasets = ["llama_questions", "web_questions"]
    a.samples = ["llama_questions=30", "web_questions=50"]
    a.tasks, a.repeats, a.warmup = ["t2s", "s2s"], 2, 0
    plan = await run(a)
    assert plan["formal_requests"] == 320 and plan["warmup_requests"] == 0
    a.samples = ["llama_questions=1"]
    assert (await run(a))["formal_requests"] == 4004


@pytest.mark.parametrize(
    "extra",
    [
        ["--concurrency", "0"],
        ["--repeats", "0"],
        ["--warmup", "-1"],
        ["--timeout", "nan"],
        ["--time-budget", "inf"],
        ["--samples", "llama_questions=1", "llama_questions=2"],
    ],
)
async def test_invalid_plan_rejected_before_any_http(server, tmp_path, extra):
    url, state = server
    with pytest.raises(ValueError):
        await run(args(workload(tmp_path), url, tmp_path / "result", *extra))
    assert not state["payloads"]


async def test_warmup_failure_keeps_planned_denominators(server, tmp_path):
    path = workload(tmp_path, questions=("HTTP_FAIL",))
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="Warmup failed"):
        await run(args(path, server[0], output, "--samples", "llama_questions=2", "--repeats", "1"))
    summary = read_json(output / "summary.json")
    assert summary["expected"] == summary["not_attempted"] == 2
    assert summary["attempted"] == 0 and summary["suite_status"] == "incomplete"
    assert next(iter(read_json(output / "warmup.json").values()))["failed"] == 4


async def test_budget_interruption_resume_keeps_previous_attempt(server, tmp_path):
    path = workload(tmp_path)
    output = tmp_path / "result"
    a = args(
        path,
        server[0],
        output,
        "--samples",
        "llama_questions=5",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--time-budget",
        "0.001",
    )
    await run(a)
    manifest = read_json(output / "manifest.json")
    assert manifest["state"] == "budget_exhausted"
    a.resume = True
    # Keep the frozen budget contract; remove the elapsed-budget wrapper for this isolated resume test.
    original = asyncio.wait_for
    from moss_speech_bench import e2e

    async def unlimited(work, timeout):
        return await original(work, timeout=10)

    from unittest.mock import patch

    with patch.object(e2e.asyncio, "wait_for", unlimited):
        await run(a)
    manifest = read_json(output / "manifest.json")
    assert manifest["state"] == "complete" and len(manifest["cells"][0]["attempts"]) == 2
    assert (output / "cells" / manifest["cells"][0]["id"] / "attempt-01").is_dir()


async def test_resume_rejects_contract_and_frozen_input_changes(server, tmp_path):
    path = workload(tmp_path)
    output = tmp_path / "result"
    a = args(
        path, server[0], output, "--samples", "llama_questions=1", "--repeats", "1", "--warmup", "0"
    )
    await run(a)
    a.resume, a.max_tokens = True, 999
    with pytest.raises(ValueError, match="Resume contract"):
        await run(a)
    frozen = next((output / "inputs").glob("*.jsonl"))
    frozen.write_text(frozen.read_text().replace("Question?", "Changed?"))
    with pytest.raises(ValueError, match="Input manifest hash mismatch"):
        report(output)


async def test_baseline_requires_matching_workload_and_records_implementation_difference(
    server, tmp_path
):
    path = workload(tmp_path)
    deployment = tmp_path / "deployment.json"
    write_json(deployment, dict(server=dict(branch="hf", commit="a" * 40, dirty=False)))
    a = args(
        path,
        server[0],
        tmp_path / "hf",
        "--samples",
        "llama_questions=2",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--deployment",
        str(deployment),
    )
    await run(a)
    write_json(deployment, dict(server=dict(branch="sg", commit="b" * 40, dirty=False)))
    a.output_dir = tmp_path / "sg"
    await run(a)
    report(a.output_dir, tmp_path / "hf")
    comparison = read_json(a.output_dir / "comparison.json")
    assert comparison["compatible"] and comparison["cells"]
    assert comparison["gain_metrics"] == ["audio_ttfp_p95_s", "rtf_p95"]
    assert comparison["baseline_identity"] != comparison["current_identity"]
    a.output_dir, a.max_tokens = tmp_path / "other", 256
    await run(a)
    report(a.output_dir, tmp_path / "hf")
    assert not read_json(a.output_dir / "comparison.json")["compatible"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "quick"],
        ["--phase", "correctness"],
        ["custom"],
        ["codec", "run", "--samples", "20"],
    ],
)
def test_removed_interfaces_are_rejected(argv):
    with pytest.raises(SystemExit):
        parser().parse_args(argv)


async def test_cli_failure_and_offline_report_exit_codes(server, tmp_path):
    path = workload(tmp_path, questions=("HTTP_FAIL",))
    output = tmp_path / "result"

    async def invoke(*argv):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "moss_speech_bench",
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    code, _, error = await invoke(
        "e2e",
        "run",
        "--config",
        str(path),
        "--model",
        "test",
        "--base-url",
        server[0],
        "--output-dir",
        str(output),
        "--samples",
        "llama_questions=1",
        "--repeats",
        "1",
        "--warmup",
        "0",
    )
    assert code == 1, error
    code, stdout, error = await invoke("e2e", "report", "--output-dir", str(output))
    assert code == 1 and "Report:" in stdout, error
