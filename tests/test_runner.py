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
        (("--mode", "daily"), "LLaMA"),
        (("--phase", "all"), "separate infer and score"),
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
    code, _, error = await invoke("--phase", "all", "--output-dir", str(output))
    assert code == 2 and "separate infer and score" in error


async def test_daily_and_full_counts_without_service(tmp_path):
    from moss_speech_bench.datasets import QA_DATASETS

    # Synthetic IDs exercise scheduling only; these are not official dataset results.
    (tmp_path / "input.wav").write_bytes(b"fixture")
    specs = []
    for name, count in QA_DATASETS.items():
        rows = [{"id": str(i), "text": str(i), "audio": "input.wav"} for i in range(count)]
        write_json(tmp_path / f"{name}.json", rows)
        specs.append(
            dict(
                id=name,
                kind="json",
                source=f"{name}.json",
                language="en",
                expected=count,
            )
        )
    path = tmp_path / "all.json"
    write_json(path, {"datasets": specs})
    for mode, expected in (("daily", 4600), ("full", 55200)):
        result = await run(
            args(path, "http://127.0.0.1:1", tmp_path / "unused", "--mode", mode, "--dry-run")
        )
        assert result["formal_requests"] == expected
        assert result["warmup_requests"] == 0
        assert not (tmp_path / "unused").exists()
    assert result["deferred_evaluations"]["codec_decoder"]["status"] == "deferred"
    specs.pop()
    write_json(path, {"datasets": specs})
    with pytest.raises(ValueError, match="WebQuestions"):
        await run(
            args(path, "http://127.0.0.1:1", tmp_path / "unused", "--mode", "full", "--dry-run")
        )


async def test_buffered_audio_uses_server_default_format(server, tmp_path):
    url, state = server
    config = workload(tmp_path)
    metrics = await run(args(config, url, tmp_path / "buffered", "--no-stream"))
    result = next(iter(metrics.values()))
    assert result["succeeded"] == 3 and result["failed"] == 0
    assert result["audio_ttfp_mean_s"] is None
    assert result["audio_duration_mean_s"] == 0.1
    assert all("audio" not in payload for payload in state["payloads"])


async def test_deferred_decoder_is_not_a_full_pass(server, tmp_path):
    from moss_speech_bench.datasets import deferred_evaluations

    url, _ = server
    output = tmp_path / "deferred"
    await run(args(workload(tmp_path), url, output))
    manifest = read_json(output / "manifest.json")
    manifest["contract"]["deferred_evaluations"] = deferred_evaluations("full")
    write_json(output / "manifest.json", manifest)
    report(output)
    summary = read_json(output / "summary.json")
    assert summary["suite_status"] == "incomplete"
    assert summary["deferred_evaluations"]["codec_decoder"]["datasets"] == {
        "seed_en": 1088,
        "seed_zh": 2020,
    }
    assert "Full suite is incomplete" in (output / "summary.md").read_text()
    import asyncio
    import sys

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "moss_speech_bench",
        "--phase",
        "report",
        "--output-dir",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, error = await process.communicate()
    assert process.returncode == 1, error.decode()


async def test_summary_uses_frozen_server_and_client_identities(server, tmp_path, monkeypatch):
    import hashlib

    from moss_speech_bench import cli

    server_sha, client_sha = "a" * 40, "b" * 40
    monkeypatch.setattr(
        cli,
        "source_identity",
        lambda: dict(source_hash="fixed", branch="client-at-infer", commit=client_sha, dirty=""),
    )
    metadata = tmp_path / "deployment.json"
    write_json(metadata, {"server": dict(commit=server_sha, dirty=True)})
    output = tmp_path / "run"
    await run(args(workload(tmp_path), server[0], output, "--deployment", str(metadata)))
    summary = (output / "summary.md").read_text()
    assert f"被测服务代码：分支 `未记录`；commit `{server_sha}`；工作区：有未提交修改。" in summary
    assert (
        f"评测客户端代码：分支 `client-at-infer`；commit `{client_sha}`；工作区：干净。" in summary
    )
    performance = (output / "performance.json").read_bytes()
    frozen_manifest = (output / "manifest.json").read_bytes()
    # Historical supplement is bound to the original manifest; numeric evidence stays frozen.
    identity = output / "execution" / "records" / "code-identity.json"
    supplement = dict(
        manifest_sha256=hashlib.sha256(frozen_manifest).hexdigest(),
        server={"branch": "server-at-infer"},
    )
    write_json(identity, supplement)
    monkeypatch.setattr(cli, "source_identity", lambda: pytest.fail("Report read current Git"))
    report(output)
    assert (
        f"被测服务代码：分支 `server-at-infer`；commit `{server_sha}`"
        in (output / "summary.md").read_text()
    )
    assert (output / "manifest.json").read_bytes() == frozen_manifest
    assert (output / "performance.json").read_bytes() == performance
    rebuilt = (output / "summary.md").read_bytes()
    report(output)
    assert (output / "summary.md").read_bytes() == rebuilt
    supplement["server"]["commit"] = "c" * 40
    write_json(identity, supplement)
    with pytest.raises(ValueError, match="Conflicting saved code identity"):
        report(output)
    supplement["manifest_sha256"] = "invalid"
    write_json(identity, supplement)
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        report(output)


def test_summary_identifies_unknown_wheel_and_detached_checkout():
    from moss_speech_bench.reporting import code_identity_lines

    text = "\n".join(code_identity_lines({"client": {"source_hash": "wheel-source"}}))
    assert "被测服务代码：分支 `未记录`；commit `未记录`；工作区：未记录。" in text
    assert "source_hash `wheel-source`" in text
    text = "\n".join(
        code_identity_lines({"client": dict(branch="", commit="d" * 40, dirty=" M file.py")})
    )
    assert "分支 `detached HEAD`" in text and "工作区：有未提交修改" in text
