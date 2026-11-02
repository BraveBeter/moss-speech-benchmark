import asyncio

import pytest
from test_quick import daily_workload
from test_runner import args, workload

from moss_speech_bench import cli
from moss_speech_bench.datasets import load_workload
from moss_speech_bench.reporting import average_repeats, report
from moss_speech_bench.storage import read_json, read_jsonl


@pytest.mark.parametrize("mode", ["quick", "daily", "full"])
async def test_named_warmup_precedes_every_repeat_and_is_excluded(
    server, tmp_path, correctness_stub, monkeypatch, mode
):
    url, state = server
    correctness_stub["url"] = url
    config = daily_workload(tmp_path)
    datasets, _, sampling = load_workload(config, "test", True, "quick")
    # Limit HTTP traffic here; real population/plan counts are checked separately.
    datasets = {name: (rows[:2], dict(meta, expected=2)) for name, (rows, meta) in datasets.items()}
    from moss_speech_bench.storage import digest

    for rows, meta in datasets.values():
        meta["samples_hash"] = digest(rows)
    monkeypatch.setattr(cli, "load_workload", lambda *a: (datasets, None, sampling))
    order = []
    send = cli.ChatClient.send

    async def recorded(self, session, sample):
        order.append(sample["id"].startswith("warmup-"))
        return await send(self, session, sample)

    monkeypatch.setattr(cli.ChatClient, "send", recorded)
    a = args(
        config,
        url,
        tmp_path / "out",
        "--mode",
        mode,
        "--concurrency",
        "2",
        "--correctness-config",
        "fixture.json",
    )
    await cli.run(a)
    assert order == [True, True, False, False] * 18
    assert len(state["payloads"]) == 72
    assert len(read_jsonl(a.output_dir / "requests.jsonl")) == 36
    averages = read_json(a.output_dir / "performance-averages.json")
    assert len(averages) == 6
    assert all(v["valid_repeats"] == 3 and v["succeeded"] == 6 for v in averages.values())
    assert all(v["succeeded"] == 2 for v in read_json(a.output_dir / "warmup.json").values())
    original = (a.output_dir / "performance-averages.json").read_bytes()
    report(a.output_dir)
    assert (a.output_dir / "performance-averages.json").read_bytes() == original
    a.resume = True
    await cli.run(a)
    assert len(order) == 72
    frozen = next((a.output_dir / "warmup-inputs").glob("*.jsonl"))
    frozen.write_text("[]\n")
    with pytest.raises(ValueError, match="Warmup input manifest hash"):
        report(a.output_dir)


def test_repeat_means_are_not_pooled_and_incomplete_or_failed_rounds_are_not_averaged():
    cells = [dict(id=f"x-c4-r{i}", dataset="x", concurrency=4, repeat=i) for i in (1, 2, 3)]
    manifest = {"cells": cells, "contract": {"repeats": 3}}
    performance = {
        cell["id"]: dict(
            expected=2,
            attempted=2,
            succeeded=2,
            failed=0,
            not_attempted=0,
            length_count=1,
            measurement_complete=True,
            throughput_qps=qps,
            latency_median_s=latency,
        )
        for cell, qps, latency in zip(cells, (1, 2, 9), (1, 4, 10), strict=True)
    }
    result = average_repeats(manifest, performance)["x-c4"]
    assert result["throughput_qps"] == 4 and result["latency_median_s"] == 5
    assert result["expected"] == 6 and result["length_count"] == 3
    assert result["audio_ttfp_mean_s"] is None
    performance[cells[-1]["id"]].update(
        measurement_complete=False, not_attempted=1, attempted=1, succeeded=1
    )
    result = average_repeats(manifest, performance)["x-c4"]
    assert result["valid_repeats"] == 2 and result["throughput_qps"] is None
    assert result["not_attempted"] == 1 and result["expected"] == 6
    performance[cells[-1]["id"]].update(
        measurement_complete=True, not_attempted=0, attempted=2, failed=1
    )
    result = average_repeats(manifest, performance)["x-c4"]
    assert result["completed_repeats"] == 3 and result["valid_repeats"] == 2
    assert result["failed"] == 1 and result["latency_median_s"] is None


async def test_warmup_failure_retains_evidence_and_blocks_formal_requests(server, tmp_path):
    from moss_speech_bench.storage import write_json

    url, state = server
    config = workload(tmp_path, warmup=True)
    write_json(tmp_path / "warmup.json", [{"id": "bad", "text": "HTTP_FAIL"}])
    a = args(config, url, tmp_path / "out")
    with pytest.raises(ValueError, match="Warmup failed"):
        await cli.run(a)
    assert len(state["payloads"]) == 1
    assert not read_jsonl(a.output_dir / "requests.jsonl")
    assert read_json(a.output_dir / "warmup.json")["en-c4-r1"]["failed"] == 1
    assert read_json(a.output_dir / "performance-averages.json")["en-c4"]["not_attempted"] == 3


async def test_cancel_during_warmup_resumes_with_fresh_warmup(server, tmp_path, monkeypatch):
    url, state = server
    config = workload(tmp_path, warmup=True)
    a = args(config, url, tmp_path / "out")
    entered = asyncio.Event()
    send = cli.ChatClient.send

    async def paused(self, session, sample):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(cli.ChatClient, "send", paused)
    task = asyncio.create_task(cli.run(a))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(cli.ChatClient, "send", send)
    a.resume = True
    await cli.run(a)
    assert len(state["payloads"]) == 4
    cell = read_json(a.output_dir / "manifest.json")["cells"][0]
    assert len(cell["attempts"]) == 2
    assert not (a.output_dir / cell["attempts"][0] / "timing.json").exists()
    warm_result = next((a.output_dir / cell["attempts"][-1] / "warmup").glob("*.result.json"))
    warm_result.write_text("{}")
    with pytest.raises((KeyError, ValueError)):
        report(a.output_dir)
