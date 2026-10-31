import asyncio
import base64
import io
import json
import wave

import pytest
import pytest_asyncio
from aiohttp import web

from moss_speech_bench.correctness import (
    BASE,
    numerical_checks,
    render_correctness,
    run_correctness,
)
from moss_speech_bench.correctness_http import run_http
from moss_speech_bench.storage import read_json, write_json


@pytest_asyncio.fixture
async def correct_server(tmp_path):
    state = {"active": 0, "calls": 0, "corrupt": False, "bad_terminal": False}

    async def handle(request):
        data = await request.json()
        state["calls"] += 1
        if "min_p" in data:
            return web.json_response({"error": "min_p unsupported"}, status=400)
        state["active"] += 1
        try:
            await asyncio.sleep(0.01)
            text = str(data["messages"])
            color = "blue" if "blue." in text else "red" if "red." in text else "water"
            code = 2 if color == "blue" else 1
            if state["corrupt"] and state["active"] > 1:
                code += 1
            audio = data["modalities"] == ["audio"]
            count = data["max_tokens"] if data["stream"] and data["max_tokens"] == 64 else 4
            pcm = (code.to_bytes(2, "little")) * (1920 * count)
            usage = dict(prompt_tokens=5, completion_tokens=count, total_tokens=5 + count)
            finish = "length" if count == 64 else "stop"
            if not data["stream"]:
                message = {"content": "" if audio else color}
                if audio:
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(24000)
                        wav.writeframes(pcm)
                    message["audio"] = {"data": base64.b64encode(buf.getvalue()).decode()}
                return web.json_response(
                    dict(choices=[dict(message=message, finish_reason=finish)], usage=usage)
                )
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            try:
                for i in range(2):
                    delta = (
                        {
                            "audio": {
                                "data": base64.b64encode(
                                    pcm[i * len(pcm) // 2 : (i + 1) * len(pcm) // 2]
                                ).decode()
                            }
                        }
                        if audio
                        else {"content": color}
                    )
                    raw = "data: " + json.dumps(dict(choices=[dict(delta=delta)])) + "\n\n"
                    await response.write(raw.encode())
                    await asyncio.sleep(0.01)
                delta = {"content": "unexpected terminal text"} if state["bad_terminal"] else {}
                await response.write(
                    (
                        "data: "
                        + json.dumps(
                            dict(choices=[dict(delta=delta, finish_reason=finish)], usage=usage)
                        )
                        + "\n\n"
                    ).encode()
                )
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            except (ConnectionResetError, web.HTTPException):
                pass
            return response
        finally:
            state["active"] -= 1

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    config = dict(
        base_url="http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]),
        model="moss",
        timeout=2,
    )
    for lang in ("en", "zh"):
        path = tmp_path / (lang + ".wav")
        path.write_bytes(b"audio fixture")
        config["audio_" + lang] = str(path)
    try:
        yield config, state
    finally:
        await runner.cleanup()


async def test_http_matrix_boundaries_and_exact_c2(correct_server, tmp_path):
    config, state = correct_server
    first = await run_http("C07", config, tmp_path / "C07")
    assert first["status"] == "pass", first
    assert len(first["checks"]) == 20 and first["http_attempts"] == 22
    second = await run_http("C08", config, tmp_path / "C08")
    assert second["status"] == "pass", second
    assert second["http_attempts"] == 8 and state["calls"] == 30
    assert all(r["detail"]["overlap_s"] > 0 for r in second["checks"])
    assert read_json(tmp_path / "C07/attempts.json")[-3]["case"] == "after-disconnect"


async def test_http_detects_terminal_payload_and_one_lsb(correct_server, tmp_path):
    config, state = correct_server
    state["corrupt"] = True
    result = await run_http("C08", config, tmp_path / "corrupt")
    assert result["status"] == "fail"
    assert read_json(tmp_path / "corrupt/audio-comparison.json")["differences"]
    state["corrupt"] = False
    state["bad_terminal"] = True
    result = await run_http("C07", config, tmp_path / "terminal")
    assert result["status"] == "fail"


def test_numerical_requires_all_heads_and_raw_top1():
    cases = {
        str(i): {
            "heads": {
                h: {"comparison": dict(finite=True, allclose=True, top1_equal=True)}
                for h in ("text", "audio")
            }
        }
        for i in range(8)
    }
    report = dict(status="passed", cases=cases)
    assert len(numerical_checks("C04", report, report)) == 32
    cases["0"]["heads"]["audio"]["comparison"]["top1_equal"] = False
    with pytest.raises(AssertionError, match="greedy"):
        numerical_checks("C04", report, report)
    with pytest.raises(AssertionError):
        numerical_checks("C06", dict(status="passed", cases={"codes24": {"status": "passed"}}), {})


async def test_identity_fail_stops_dependents_and_report_hashes(tmp_path):
    config = tmp_path / "config.json"
    write_json(
        config,
        dict(workspace_root=".", profile="eager-base", core_sha=BASE, core_root=str(tmp_path)),
    )
    suite = await run_correctness(config, tmp_path / "output")
    assert suite["status"] == "fail" and suite["gates"]["C01"]["status"] == "not_run"
    assert all(g["status"] == "not_run" for k, g in suite["gates"].items() if k != "C00")
    assert render_correctness(tmp_path / "output")["status"] == "fail"
    with pytest.raises(ValueError, match="fresh"):
        await run_correctness(config, tmp_path / "output")


async def test_diagnostic_subset_never_passes_full_suite(correct_server, tmp_path):
    config, state = correct_server
    config.update(workspace_root=".", profile="eager-base", core_sha=BASE)
    path = tmp_path / "config.json"
    write_json(path, config)
    result = await run_correctness(path, tmp_path / "output", "C08")
    assert result["status"] == "incomplete" and result["gates"]["C08"]["status"] == "pass"
    artifact = tmp_path / "output/C08/attempts.json"
    artifact.write_text("[]")
    with pytest.raises(AssertionError, match="hash"):
        render_correctness(tmp_path / "output")


def test_origin_guard_handles_namespace_and_ignores_unrelated_proxies(tmp_path):
    import types

    from moss_speech_bench.providers.identity import collect_origins

    module = types.ModuleType("sglang_omni")
    module.__path__ = [str(tmp_path / "sglang_omni")]

    class Proxy:
        def __getattr__(self, name):
            raise AssertionError("Unrelated module inspected")

    assert collect_origins(tmp_path, {"sglang_omni": module, "other": Proxy()})
    module.__path__ = ["/outside/source"]
    with pytest.raises(AssertionError):
        collect_origins(tmp_path, {"sglang_omni": module})


def test_provider_entry_imports_candidate_and_rejects_zero_collection(tmp_path):
    import os
    import subprocess
    import sys

    from moss_speech_bench.correctness import PROVIDERS

    core = tmp_path / "core"
    (core / "sglang_omni").mkdir(parents=True)
    (core / "sglang_omni/__init__.py").write_text("")
    output = tmp_path / "result"
    output.mkdir()
    result = subprocess.run(
        [sys.executable, str(PROVIDERS / "entry.py"), str(output), "C01"],
        env=dict(os.environ, MOSS_CORE_ROOT=str(core), PYTHONPATH=str(core)),
        cwd=core,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert read_json(output / "origins.json")["sglang_omni"] == [
        str(core / "sglang_omni/__init__.py")
    ]
    assert read_json(output / "coverage.json")["outcomes"] == {}


async def test_resume_preserves_failures_and_reuses_complete_gates(correct_server, tmp_path):
    config, state = correct_server
    config.update(workspace_root=".", profile="eager-base", core_sha=BASE)
    path = tmp_path / "config.json"
    write_json(path, config)
    state["corrupt"] = True
    failed = await run_correctness(path, tmp_path / "output", "C08")
    assert failed["gates"]["C08"]["status"] == "fail"
    old = (tmp_path / "output/C08/attempts.json").read_bytes()
    state["corrupt"] = False
    completed = await run_correctness(path, tmp_path / "output", "C08", True)
    assert completed["status"] == "incomplete" and completed["gates"]["C08"]["status"] == "pass"
    assert completed["history"][0]["result"]["status"] == "fail"
    assert (tmp_path / "output/C08/attempts.json").read_bytes() == old
    calls = state["calls"]
    await run_correctness(path, tmp_path / "output", "C08", True)
    assert state["calls"] == calls


@pytest.mark.parametrize("kind", ["ar", "encoder", "codec"])
def test_tensor_policy_with_optional_provider_runtime(kind, monkeypatch, tmp_path):
    import importlib
    import os

    if os.environ.get("MOSS_TENSOR_TEST") != "1":
        pytest.skip("Run in explicit Torch provider environment with MOSS_TENSOR_TEST=1")
    import torch

    monkeypatch.setenv("MOSS_CORE_ROOT", str(tmp_path))
    module = importlib.import_module("moss_speech_bench.providers." + kind)
    if kind == "ar":
        reference = torch.arange(16, dtype=torch.float32) * 0.01
        candidate = reference.clone()
        candidate[-2] = candidate[-1] + 0.01
        result = module.compare_logits(
            candidate, reference, dict(absolute_tolerance=0.25, relative_tolerance=0.02)
        )
        assert result["allclose"] and not result["top1_equal"]
    elif kind == "encoder":
        compare = module.tensor_comparison
        assert compare(
            torch,
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int64),
            integer=True,
        )["passed"]
        assert not compare(torch, torch.tensor([1.0]), torch.tensor([1.0, 2.0]), integer=False)[
            "passed"
        ]
        assert not compare(
            torch, torch.tensor([float("nan")]), torch.tensor([float("nan")]), integer=False
        )["passed"]
    else:
        compare = module.compare_tensor
        assert compare(torch, torch.tensor([1.0]), torch.tensor([1.0]), floating=True)["passed"]
        assert not compare(
            torch, torch.tensor([1.0], dtype=torch.float64), torch.tensor([1.0]), floating=True
        )["passed"]
        assert not compare(
            torch, torch.tensor([float("nan")]), torch.tensor([float("nan")]), floating=True
        )["passed"]


async def test_correctness_cli_and_offline_report_in_installed_runtime(correct_server, tmp_path):
    import os
    import sys

    config, state = correct_server
    config.update(workspace_root=".", profile="eager-base", core_sha=BASE)
    path = tmp_path / "config.json"
    write_json(path, config)
    executable = os.environ.get("BENCHMARK_TEST_PYTHON", sys.executable)
    output = tmp_path / "cli-result"

    async def invoke(*args):
        process = await asyncio.create_subprocess_exec(
            executable,
            "-m",
            "moss_speech_bench",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmp_path,
        )
        _, error = await process.communicate()
        return process.returncode, error.decode()

    code, error = await invoke(
        "--phase",
        "correctness",
        "--correctness-config",
        str(path),
        "--dry-run",
        "--output-dir",
        str(output),
    )
    assert code == 2 and "no dry-run" in error and state["calls"] == 0
    code, error = await invoke(
        "--phase",
        "correctness",
        "--correctness-config",
        str(path),
        "--checks",
        "C08",
        "--output-dir",
        str(output),
    )
    assert code == 1, error
    assert read_json(output / "correctness.json")["gates"]["C08"]["status"] == "pass"
    calls = state["calls"]
    code, error = await invoke("--phase", "correctness-report", "--output-dir", str(output))
    assert code == 1 and state["calls"] == calls, error
    (output / "C08/attempts.json").write_text("[]")
    code, error = await invoke("--phase", "correctness-report", "--output-dir", str(output))
    assert code == 2 and "hash mismatch" in error


def test_terminal_trace_preserves_grid_and_rejects_path_traversal(tmp_path, monkeypatch):
    from moss_speech_bench.providers.sg_trace import write_trace

    monkeypatch.setenv("MOSS_TRACE_DIRECTORY", str(tmp_path))
    grid = [[151667, 17], [151645, 16384]]
    write_trace("fixed-request", "ar", {"output_grid": grid})
    saved = json.loads((tmp_path / "fixed-request.ar.json").read_text())
    assert saved == {"request_id": "fixed-request", "output_grid": grid}
    assert grid == [[151667, 17], [151645, 16384]]
    with pytest.raises(ValueError, match="Invalid trace"):
        write_trace("../escape", "ar", {})
    assert not list(tmp_path.glob("*.tmp"))
