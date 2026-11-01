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
    markdown = (output / "summary.md").read_text()
    assert "| audio-s/s | QPS |" in markdown
    assert "[custom-json/en](inputs/en.jsonl)" in markdown
    assert "## 正确性测试" in markdown and "未绑定本run" in markdown
    assert "## 性能测试" in markdown
    assert "## 质量测试" in markdown and "评分状态：未执行" in markdown
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
        code_identity_lines(
            {
                "contract": {
                    "deployment": {
                        "client_build": {
                            "branch": "frozen-wheel",
                            "commit": "e" * 40,
                            "dirty": False,
                        }
                    }
                },
                "client": {"commit": None, "source_hash": "wheel-source"},
            }
        )
    )
    assert "分支 `frozen-wheel`" in text and "commit `" + "e" * 40 + "`" in text
    text = "\n".join(
        code_identity_lines({"client": dict(branch="", commit="d" * 40, dirty=" M file.py")})
    )
    assert "分支 `detached HEAD`" in text and "工作区：有未提交修改" in text


def test_performance_table_displays_recorded_dataset_and_audio_throughput():
    from moss_speech_bench.reporting import performance_table

    audio = summarize(
        [
            {"is_success": True, "audio_duration_s": 2, "latency_s": 1},
            {"is_success": True, "audio_duration_s": 4, "latency_s": 1},
            {"is_success": False, "audio_duration_s": 99, "latency_s": 1},
        ],
        3,
        5,
    )
    assert audio["audio_throughput_s_per_s"] == 1.2  # successful audio seconds / cell wall time
    text = summarize([{"is_success": True, "latency_s": 1}], 1, 2)
    manifest = dict(
        contract={
            "datasets": {
                "opaque": {
                    "kind": "openaudio",
                    "source_dataset": "llama_questions",
                    "task": "s2s",
                    "language": "en",
                },
                "t2t-en": {"kind": "json", "task": "t2t", "language": "en"},
            }
        },
        cells=[
            {"id": "opaque-c4-r1", "dataset": "opaque", "concurrency": 4, "repeat": 1},
            {"id": "t2t-en-c1-r1", "dataset": "t2t-en", "concurrency": 1, "repeat": 1},
        ],
    )
    table = performance_table(
        manifest,
        {"opaque-c4-r1": audio, "t2t-en-c1-r1": text},
    )
    header = [c.strip() for c in table[0].strip("|").split("|")]
    rows = [
        dict(zip(header, [c.strip() for c in line.strip("|").split("|")], strict=True))
        for line in table[2:]
    ]
    assert rows[0]["数据集"] == "[llama_questions](inputs/opaque.jsonl)"
    assert rows[0]["audio-s/s"] == "1.2" and rows[0]["QPS"] == "0.4"
    assert rows[0]["成功/全部"] == "2/3" and rows[0]["并发"] == "4"
    assert rows[0]["模态/语言"] == "S2S / EN"
    assert rows[1]["数据集"] == "[custom-json/t2t-en](inputs/t2t-en.jsonl)"
    assert rows[1]["audio-s/s"] == "N/A" and rows[1]["QPS"] == "0.5"


def test_correctness_summary_validates_binding_and_reports_missing_or_incomplete(tmp_path):
    import hashlib

    from moss_speech_bench.correctness import GATES
    from moss_speech_bench.reporting import correctness_section
    from moss_speech_bench.storage import digest

    path = tmp_path / "correctness/correctness.json"
    suite = dict(
        config={},
        config_hash=digest({}),
        status="pass",
        gates={g: dict(status="pass", checks=[{"status": "pass"}]) for g in GATES},
    )
    write_json(path, suite)
    view = path.with_suffix(".md")
    view.write_text("Keep the historical detailed view unchanged.\n")
    proof = dict(
        status="pass",
        report="correctness/correctness.json",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    lines = "\n".join(correctness_section(tmp_path, {"correctness": proof}))
    assert "状态：pass；通过 9/9 组" in lines
    assert view.read_text() == "Keep the historical detailed view unchanged.\n"
    unbound = "\n".join(correctness_section(tmp_path, {"correctness": "outside_client_scope"}))
    assert "未绑定本run" in unbound and "状态：pass" not in unbound
    suite["gates"]["C04"] = {"status": "unsupported", "checks": []}
    suite["status"] = proof["status"] = "incomplete"
    write_json(path, suite)
    proof["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    lines = "\n".join(correctness_section(tmp_path, {"correctness": proof}))
    assert "状态：incomplete" in lines and "| C04 AR / HF | N/A | unsupported |" in lines
    proof["sha256"] = "invalid"
    with pytest.raises(ValueError, match="hash mismatch"):
        correctness_section(tmp_path, {"correctness": proof})


def test_single_summary_template_keeps_empty_phase_tables(tmp_path):
    from pathlib import Path

    from moss_speech_bench.reporting import summary_markdown

    manifest = {"contract": {"mode": "custom", "datasets": {}}, "state": "complete", "cells": []}
    text = "\n".join(summary_markdown(tmp_path, manifest, {}, None, []))
    headings = ("## 正确性测试", "## 性能测试", "## 质量测试")
    assert [text.index(h) for h in headings] == sorted(text.index(h) for h in headings)
    assert all(text.count(h) == 1 for h in headings)
    correctness, _, quality = text.split("## ")[1:]
    assert len([line for line in correctness.splitlines() if line.startswith("|")]) == 2
    assert len([line for line in quality.splitlines() if line.startswith("|")]) == 2
    assert "| 成功/全部 |" in text and "| 评分/全部 |" in text
    assert "Expected" not in text and "Attempted" not in text and "Succeeded" not in text
    assert "(performance.json)" not in text
    template = (Path(__file__).parents[1] / "SUMMARY_TEMPLATE.md").read_text()
    for line in text.splitlines():
        if line.startswith("| 数据集") or line.startswith("| 检查项"):
            assert line in template


async def test_custom_mixed_cell_overlaps_tasks_and_archives_per_request_identity(server, tmp_path):
    url, state = server
    write_json(
        tmp_path / "mixed.json",
        [
            dict(id="text", task="t2t", text="Text output"),
            dict(id="audio", task="t2s", text="Audio output"),
        ],
    )
    write_json(
        tmp_path / "mixed-config.json",
        dict(
            datasets=[
                dict(
                    id="mixed-pair",
                    kind="json",
                    source="mixed.json",
                    task="mixed",
                    language="en",
                    expected=2,
                )
            ],
            sampling=dict(seed=42, max_tokens=128),
        ),
    )
    output = tmp_path / "mixed-output"
    values = await run(args(tmp_path / "mixed-config.json", url, output, "--concurrency", "2"))
    assert len(values) == 1 and state["peak"] == 2
    metrics = next(iter(values.values()))
    assert metrics["expected"] == metrics["succeeded"] == 2
    from moss_speech_bench.storage import read_jsonl

    rows = read_jsonl(output / "requests.jsonl")
    assert {row["task"] for row in rows} == {"t2t", "t2s"}
    assert {p["modalities"][0] for p in state["payloads"]} == {"text", "audio"}
    frozen = read_jsonl(output / "inputs/mixed-pair.jsonl")
    assert {row["task"] for row in frozen} == {"t2t", "t2s"}
