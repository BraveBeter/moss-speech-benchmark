"""Regressions for capability-selected QA and independent numerical policy."""

import asyncio
import copy
import io
import os
import sys
import wave
from pathlib import Path

import aiohttp
import numpy as np
import pytest
import pytest_asyncio
from aiohttp import web
from test_quick import daily_workload
from test_runner import args

from moss_speech_bench.cli import run
from moss_speech_bench.client import ChatClient
from moss_speech_bench.datasets import DEMO_AUDIO_SYSTEM, load_workload
from moss_speech_bench.hf_contract import deployment_binding, load_config, oracle_binding
from moss_speech_bench.hf_regression import compare
from moss_speech_bench.quality import evaluate
from moss_speech_bench.reporting import summarize
from moss_speech_bench.storage import read_json, write_json


def wav():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 2400)
    return stream.getvalue()


def speech_workload(root):
    path = daily_workload(root)
    reference = root / "reference.wav"
    reference.write_bytes(wav())
    config = read_json(path)
    config["service"] = dict(
        api="speech",
        tasks=["t2s"],
        streaming=False,
        reference_audio="reference.wav",
        system=DEMO_AUDIO_SYSTEM,
    )
    write_json(path, config)
    return path


@pytest_asyncio.fixture
async def speech_server():
    state = dict(payloads=[], active=0, peak=0)

    async def handle(request):
        payload = await request.json()
        state["payloads"].append(payload)
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            await asyncio.sleep(0.002)
            return web.Response(body=wav(), content_type="audio/wav")
        finally:
            state["active"] -= 1

    app = web.Application()
    app.router.add_post("/v1/audio/speech", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1]), state
    finally:
        await runner.cleanup()


async def test_speech_qa_keeps_three_rounds_warmup_and_no_fabricated_finish(
    speech_server, tmp_path, correctness_stub
):
    """Raw speech answers need the existing QA cohort, not a synthetic stop event."""
    path = speech_workload(tmp_path)
    url, state = speech_server
    correctness_stub["url"] = url
    command = args(
        path,
        url,
        tmp_path / "quick",
        "--mode",
        "quick",
        "--no-stream",
        "--correctness-config",
        "fixture.json",
    )
    metrics = await run(command)
    assert len(state["payloads"]) == 180 + 36 and state["peak"] == 4
    assert len(metrics) == 9 and all(
        m["expected"] == m["succeeded"] == 20 for m in metrics.values()
    )
    assert all(
        "messages" not in p
        and "expected" not in p
        and "max_tokens" not in p
        and p["max_new_tokens"] == 512
        and p["input"].startswith("input ")
        for p in state["payloads"]
    )
    assert all(
        m["unknown_finish_count"] == 20
        and m["ttfa_requests"] == 0
        and m["usage_counted_requests"] == 0
        for m in metrics.values()
    )
    manifest = read_json(command.output_dir / "manifest.json")
    assert manifest["contract"]["repeats"] == 3
    command.resume = True
    await run(command)
    assert len(state["payloads"]) == 216


def test_declared_capabilities_do_not_silently_switch_stream_or_task(tmp_path):
    path = speech_workload(tmp_path)
    with pytest.raises(ValueError, match="no-stream"):
        load_workload(path, "test", True, "quick")
    with pytest.raises(ValueError, match="capabilities"):
        load_workload(path, "test", False, "quick", tasks=["s2s"])
    rows, _, _ = load_workload(path, "test", False, "quick")
    assert all(
        r["payload"]["input"] == r["question"] for cohort, _ in rows.values() for r in cohort
    )


async def test_raw_speech_finish_and_usage_remain_unknown(speech_server, tmp_path):
    url, _ = speech_server
    sample = dict(
        id="question",
        task="t2s",
        language="en",
        endpoint="/v1/audio/speech",
        payload=dict(input="What is two plus two?", stream=False),
    )
    async with aiohttp.ClientSession() as session:
        result = await ChatClient(url, tmp_path).send(session, sample)
    assert result.is_success and result.finish_reason is None and result.usage == {}
    assert result.audio_ttfp_s is None and result.audio_duration_s == 0.1
    metric = summarize([read_json(next(tmp_path.glob("*.result.json")))], 1, 1)
    assert metric["output_tokens_total"] is None and metric["unknown_finish_count"] == 1


def test_t2s_answer_is_semantic_qa_while_custom_reading_retains_wer():
    """A correct spoken answer must not be scored against the question transcript."""

    class Backend:
        def transcribe(self, audio, language):
            return "Paris"

        def judge(self, sample, answer):
            assert sample["question"] == "What is the capital of France?" and answer == "Paris"
            return dict(correct=True, valid=True)

        def utmos(self, audio):
            return 4.0

    sample = dict(
        task="t2s",
        language="en",
        quality_task="qa",
        source_dataset="web_questions",
        question="What is the capital of France?",
        expected="Paris",
    )
    result = evaluate(sample, dict(is_success=True), "answer.wav", Backend(), {})
    assert result["accuracy"]["value"]["correct"] is True and result["utmos"]["value"] == 4.0
    assert "wer" not in result and "sim" not in result
    sample.pop("source_dataset")
    sample.pop("quality_task")
    result = evaluate(sample, None, None, Backend(), {})
    assert set(result) == {"wer", "sim"}


@pytest.mark.parametrize("change", ["error", "top1", "shape", "dtype", "nan"])
def test_numerical_policy_detects_single_value_and_structural_regressions(change):
    expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    actual = expected.copy()
    if change == "error":
        actual[0] = 1.0
    elif change == "top1":
        actual[1] = 0.31
    elif change == "shape":
        actual = actual[:2]
    elif change == "dtype":
        actual = actual.astype(np.float64)
    else:
        actual[0] = np.nan
    assert compare(actual, expected, 0.25, 0.02, top1=True, id="vector")["status"] == "fail"


def test_integer_codes_are_exact_even_when_float_tolerance_would_allow_difference():
    expected = np.array([1, 2, 3], dtype=np.int64)
    assert compare(expected, expected, 1e-5, 1e-5, id="codes")["status"] == "pass"
    assert compare(expected + 1, expected, 10, 10, id="codes")["status"] == "fail"


def regression_config(tmp_path):
    config = read_json(Path(__file__).parents[1] / "examples/hf-regression.json")
    config.update(
        workspace_root=str(tmp_path),
        core_sha="a" * 40,
        assets=[dict(id="voice", path="voice.wav", sha256="1" * 64)],
    )
    (tmp_path / "voice.wav").write_bytes(wav())
    config["voice_wav"] = "voice.wav"
    config["service"]["reference_audio"] = "voice.wav"
    return config


def test_interpreter_symlink_retains_virtual_environment(tmp_path):
    config = regression_config(tmp_path)
    interpreter = tmp_path / "env/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    config["reference_python"] = "env/bin/python"
    path = tmp_path / "config.json"
    write_json(path, config)
    assert load_config(path)["reference_python"] == str(interpreter)
    config["assets"] *= 2
    write_json(path, config)
    with pytest.raises(ValueError, match="unique"):
        load_config(path)


def test_reference_binding_is_independent_of_candidate_but_pins_inputs_and_policy(tmp_path):
    config = regression_config(tmp_path)
    original = oracle_binding(config)
    moved = copy.deepcopy(config)
    moved.update(core_sha="b" * 40, core_root="elsewhere", model_path="moved")
    moved["assets"][0]["path"] = "moved-voice.wav"
    moved["runtime"]["attention_backend"] = "torch_native"
    assert oracle_binding(moved) == original
    assert deployment_binding(moved) != deployment_binding(config)
    for field in ("cases", "sampling", "assets", "tolerances"):
        changed = copy.deepcopy(config)
        changed[field] = []
        assert oracle_binding(changed) != original


async def test_worker_cancellation_terminates_owned_process(tmp_path):
    from moss_speech_bench.hf_regression import worker

    config = regression_config(tmp_path)
    interpreter = tmp_path / "blocking-python"
    pidfile = tmp_path / "worker.pid"
    interpreter.write_text(
        f"#!{sys.executable}\nimport os, time\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(120)\n"
    )
    interpreter.chmod(0o700)
    config.update(reference_python=str(interpreter), worker_timeout=120)
    task = asyncio.create_task(worker(config, "reference", "ar", tmp_path / "worker"))
    for _ in range(100):
        if pidfile.exists():
            break
        await asyncio.sleep(0.01)
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
