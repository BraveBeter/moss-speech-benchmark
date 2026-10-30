import pytest
from moss_speech_bench.cli import parser, run
from moss_speech_bench.reporting import report, summarize
from moss_speech_bench.storage import read_json, write_json


def workload(tmp_path, texts=("one", "two", "three"), warmup=False):
    write_json(tmp_path / "samples.json", [{"id": str(i), "text": t} for i, t in enumerate(texts)])
    spec = dict(
        id="en", kind="json", source="samples.json", language="en", task="t2s", expected=len(texts)
    )
    config = {"datasets": [spec], "sampling": {"seed": 42, "max_tokens": 128}}
    if warmup:
        write_json(tmp_path / "warmup.json", [{"id": "warm", "text": "Warm up separately"}])
        config["warmup"] = dict(spec, id="warm", source="warmup.json", expected=1)
    write_json(tmp_path / "config.json", config)
    return tmp_path / "config.json"


def args(config, url, output, *extra):
    return parser().parse_args(
        [
            "--config",
            str(config),
            "--base-url",
            url,
            "--model",
            "test",
            "--output-dir",
            str(output),
            *extra,
        ]
    )


async def test_sweep_warmup_rebuild_and_resume(server, tmp_path):
    url, state = server
    config = workload(tmp_path, warmup=True)
    output = tmp_path / "result"
    a = args(config, url, output, "--concurrencies", "1,2", "--repeats", "2")
    metrics = await run(a)
    assert len(metrics) == 4
    assert len(state["payloads"]) == 16  # 12 formal + 4 disjoint warmups
    assert state["peak"] == 2
    for value in metrics.values():
        assert value["expected"] == value["attempted"] == value["succeeded"] == 3
        assert value["length_count"] == 3
        assert value["scored"] == 0
        assert value["audio_throughput_s_per_s"] > 0
    original = (output / "performance.json").read_bytes()
    report(output)
    assert (output / "performance.json").read_bytes() == original
    a.resume = True
    await run(a)
    assert len(state["payloads"]) == 16  # completed measurements are not resent
    # Corruption of a result or its outputs must not be silently reused.
    audio = next(output.glob("cells/**/outputs/*.wav"))
    audio.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        report(output)


async def test_failed_requests_and_unknown_usage(server, tmp_path):
    url, state = server
    config = workload(tmp_path, ("HTTP_FAIL", "NO_USAGE"))
    metrics = await run(args(config, url, tmp_path / "out"))
    value = next(iter(metrics.values()))
    assert value["attempted"] == 2 and value["failed"] == 1
    assert value["failure_rate"] == 0.5
    assert value["output_throughput"] is None
    assert len(state["payloads"]) == 2


async def test_contract_rejects_subset_and_unsupported_score(server, tmp_path):
    url, _ = server
    config = workload(tmp_path)
    for extra, message in [
        (("--mode", "daily"), "5279"),
        (("--phase", "all"), "not implemented"),
        (("--concurrency", "0"), "positive"),
    ]:
        with pytest.raises(ValueError, match=message):
            await run(args(config, url, tmp_path / "out", *extra))
    data = read_json(config)
    data["datasets"][0]["expected"] = 4
    write_json(config, data)
    with pytest.raises(ValueError, match="expected 4 rows, found 3"):
        await run(args(config, url, tmp_path / "out"))


async def test_baseline_contract_and_result_tampering(server, tmp_path):
    url, _ = server
    config = workload(tmp_path)
    metadata = tmp_path / "deployment.json"
    write_json(metadata, {"model_revision": "a", "settings": {"graph": False}})
    a = args(config, url, tmp_path / "base", "--deployment", str(metadata))
    await run(a)
    write_json(metadata, {"model_revision": "a", "settings": {"graph": True}})
    a.output_dir = tmp_path / "candidate"
    await run(a)
    report(a.output_dir, tmp_path / "base")
    assert not read_json(a.output_dir / "comparison.json")["compatible"]
    report(a.output_dir, tmp_path / "base", ["deployment.settings.graph"])
    comparison = read_json(a.output_dir / "comparison.json")
    assert comparison["compatible"] and comparison["cells"]
    result = next(a.output_dir.glob("cells/**/outputs/*.result.json"))
    obj = read_json(result)
    obj["latency_s"] = 999
    write_json(result, obj)
    with pytest.raises(ValueError, match="ledger hash mismatch"):
        report(a.output_dir)


def test_all_failed_metrics_have_defined_denominators():
    value = summarize([{"is_success": False, "latency_s": 2}], 3, 5)
    assert value["expected"] == 3 and value["not_attempted"] == 2
    assert value["failure_rate"] == 1 and value["throughput_qps"] == 0
    assert value["rtf_mean"] is None


async def test_cancelled_run_resume_keeps_attempts(server, tmp_path):
    import asyncio

    url, state = server
    config = workload(tmp_path, tuple(f"text-{i}" for i in range(30)))
    output = tmp_path / "out"
    a = args(config, url, output, "--concurrency", "1")
    task = asyncio.create_task(run(a))
    while len(state["payloads"]) < 3:
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    manifest = read_json(output / "manifest.json")
    assert manifest["state"] == "interrupted"
    partial = next(iter(read_json(output / "performance.json").values()))
    assert partial["not_attempted"] > 0 and partial["failed"] == 1
    assert partial["throughput_qps"] is None
    assert not partial["measurement_complete"]
    old_attempt = output / manifest["cells"][0]["attempts"][0]
    old = {p.name: p.read_bytes() for p in (old_attempt / "outputs").iterdir()}
    count = len(state["payloads"])
    a.resume = True
    await run(a)
    assert len(state["payloads"]) == count + 30  # rerun whole interrupted cell
    assert old == {p.name: p.read_bytes() for p in (old_attempt / "outputs").iterdir()}
    manifest = read_json(output / "manifest.json")
    assert manifest["state"] == "complete" and "error" not in manifest
    assert len(manifest["cells"][0]["attempts"]) == 2
    frozen = output / "inputs/en.jsonl"
    frozen.write_text(frozen.read_text().replace("text-0", "changed"))
    with pytest.raises(ValueError, match="Input manifest hash mismatch"):
        report(output)


async def test_installed_cli_exit_codes_and_offline_report(server, tmp_path):
    import asyncio
    import os
    import sys

    url, _ = server
    config = workload(tmp_path, ("HTTP_FAIL",))
    output = tmp_path / "cli-result"
    executable = os.environ.get("BENCHMARK_TEST_PYTHON", sys.executable)

    async def invoke(*argv):
        process = await asyncio.create_subprocess_exec(
            executable,
            "-m",
            "moss_speech_bench",
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmp_path,
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    code, _, error = await invoke(
        "--config", str(config), "--base-url", url, "--model", "test", "--output-dir", str(output)
    )
    assert code == 1, error  # failed request must not become a successful shell command
    code, stdout, error = await invoke("--phase", "report", "--output-dir", str(output))
    assert code == 1 and "Report:" in stdout, error
    code, _, error = await invoke("--phase", "score", "--output-dir", str(output))
    assert code == 2 and "not implemented" in error


async def test_daily_and_full_counts_without_service(tmp_path):
    from moss_speech_bench.datasets import DAILY

    # Synthetic IDs exercise scheduling only; these are not official dataset results.
    (tmp_path / "input.wav").write_bytes(b"fixture")
    specs = []
    for name, count in DAILY.items():
        task = "s2s" if name == "llama_questions" else "t2t" if name == "storycloze" else "t2s"
        rows = [{"id": str(i), "text": str(i), "audio": "input.wav"} for i in range(count)]
        write_json(tmp_path / f"{name}.json", rows)
        specs.append(
            dict(
                id=name,
                kind="json",
                source=f"{name}.json",
                task=task,
                language="zh" if name == "seed_zh" else "en",
                expected=count,
            )
        )
    path = tmp_path / "all.json"
    write_json(path, {"datasets": specs})
    for mode, expected in (("daily", 5279), ("full", 39108)):
        result = await run(
            args(path, "http://127.0.0.1:1", tmp_path / "unused", "--mode", mode, "--dry-run")
        )
        assert result["formal_requests"] == expected
        assert result["warmup_requests"] == 0
        assert not (tmp_path / "unused").exists()
    # Performance-only Full does not require the unused Chinese quality dataset.
    write_json(path, {"datasets": [s for s in specs if s["id"] != "seed_zh"]})
    result = await run(
        args(path, "http://127.0.0.1:1", tmp_path / "unused", "--mode", "full", "--dry-run")
    )
    assert result["formal_requests"] == 39108


async def test_buffered_audio_uses_server_default_format(server, tmp_path):
    url, state = server
    config = workload(tmp_path)
    metrics = await run(args(config, url, tmp_path / "buffered", "--no-stream"))
    result = next(iter(metrics.values()))
    assert result["succeeded"] == 3 and result["failed"] == 0
    assert result["audio_ttfp_mean_s"] is None
    assert result["audio_duration_mean_s"] == 0.1
    assert all("audio" not in payload for payload in state["payloads"])
