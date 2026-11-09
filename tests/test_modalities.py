"""Every declared QA modality has independent input, transport and scoring paths."""

import aiohttp
import pytest
from test_quick import daily_workload
from test_runner import args

from moss_speech_bench.cli import run
from moss_speech_bench.client import ChatClient
from moss_speech_bench.datasets import load_workload
from moss_speech_bench.hf_contract import deployment_binding
from moss_speech_bench.quality import evaluate
from moss_speech_bench.storage import read_json, write_json

TASKS = ["t2s", "s2s", "t2t", "s2t"]


def multimodal_workload(root):
    path = daily_workload(root)
    config = read_json(path)
    config["service"] = dict(api="chat", tasks=TASKS, streaming=True)
    write_json(path, config)
    return path


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("stream", [False, True])
async def test_each_mode_selects_only_its_input_output_and_quick_cells(
    server, tmp_path, task, stream
):
    path = multimodal_workload(tmp_path)
    if task.startswith("t"):
        (tmp_path / "input.wav").unlink()
    datasets, _, _ = load_workload(path, "test", stream, "quick", tasks=[task])
    assert len(datasets) == 3
    sample = next(iter(datasets.values()))[0][0]
    assert all(
        metadata["task"] == task and name.endswith("-" + task)
        for name, (_, metadata) in datasets.items()
    )
    payload = sample["payload"]
    content = payload["messages"][-1]["content"]
    assert isinstance(content, str) == task.startswith("t")
    if task.startswith("s"):
        assert content[0]["type"] == "input_audio"
    assert payload["modalities"] == ["audio" if task.endswith("s") else "text"]
    assert "expected" not in payload and sample["quality_task"] == "qa"
    url, _ = server
    command = args(
        path,
        url,
        tmp_path / "dry",
        "--mode",
        "quick",
        "--tasks",
        task,
        "--dry-run",
        "--stream" if stream else "--no-stream",
    )
    plan = await run(command)
    assert plan["formal_requests"] == 180 and plan["warmup_requests"] == 36
    async with aiohttp.ClientSession() as session:
        result = await ChatClient(url, tmp_path / "transport").send(session, sample)
    assert result.is_success and result.task == task
    assert bool(result.text) == task.endswith("t")
    assert (result.audio_duration_s > 0) == task.endswith("s")


def test_combined_selection_keeps_separate_cells_and_identical_question_ids(tmp_path):
    path = multimodal_workload(tmp_path)
    datasets, _, _ = load_workload(path, "test", True, "quick", tasks=TASKS)
    assert len(datasets) == 12
    for source in ("llama_questions", "trivia_qa", "web_questions"):
        selected = [datasets[source + "-" + task][1]["selection"] for task in TASKS]
        assert all(selection == selected[0] for selection in selected)
    config = read_json(path)
    config["service"]["tasks"] = ["t2s"]
    write_json(path, config)
    with pytest.raises(ValueError, match="capabilities"):
        load_workload(path, "test", True, "quick", tasks=["s2s"])


async def test_t2s_certificate_cannot_admit_t2t_inference(server, tmp_path, monkeypatch):
    from moss_speech_bench import correctness
    from moss_speech_bench.datasets import DEMO_AUDIO_SYSTEM

    path = multimodal_workload(tmp_path)
    url, state = server
    certificate = dict(
        protocol="hf-regression-v1",
        workspace_root=str(tmp_path),
        core_sha="a" * 40,
        base_url=url,
        runtime={},
        gpu_memory_fraction=0.72,
        assets=[],
        service=dict(
            api="speech",
            tasks=["t2s"],
            streaming=False,
            reference_audio="input.wav",
            system=DEMO_AUDIO_SYSTEM,
        ),
    )
    monkeypatch.setattr(correctness, "load_suite", lambda _: certificate)
    deployment = tmp_path / "deployment.json"
    write_json(
        deployment,
        dict(
            server=dict(commit="a" * 40, dirty=False), hf_regression=deployment_binding(certificate)
        ),
    )
    command = args(
        path,
        url,
        tmp_path / "forbidden",
        "--mode",
        "quick",
        "--tasks",
        "t2t",
        "--no-stream",
        "--deployment",
        str(deployment),
        "--correctness-config",
        "fixture.json",
    )
    with pytest.raises(ValueError, match="correctness capability contract"):
        await run(command)
    assert state["payloads"] == []


@pytest.mark.parametrize("task", TASKS)
def test_quality_uses_output_modality_and_semantic_qa(task):
    calls = []

    class Backend:
        def transcribe(self, path, lang):
            calls.append("asr")
            return "Paris"

        def judge(self, sample, answer):
            calls.append("judge")
            assert answer == "Paris" and sample["question"] == "Capital of France?"
            return dict(correct=True, valid=True)

        def utmos(self, path):
            calls.append("utmos")
            return 4.0

    sample = dict(
        task=task,
        language="en",
        quality_task="qa",
        source_dataset="web_questions",
        question="Capital of France?",
        expected="Paris",
    )
    metrics = evaluate(sample, dict(is_success=True, text="Paris"), "frozen.wav", Backend(), {})
    audio_output = task.endswith("s")
    assert calls == (["asr", "judge", "utmos"] if audio_output else ["judge"])
    assert set(metrics) == ({"accuracy", "utmos", "transcript"} if audio_output else {"accuracy"})
    assert metrics["accuracy"]["value"]["correct"]
