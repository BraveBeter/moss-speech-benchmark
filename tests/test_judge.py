import json

import pytest

from moss_speech_bench.judge import judge_answer, parse_verdict
from moss_speech_bench.quality import LocalScorers, evaluate


def test_semantic_judge_cache_binding_and_no_credential_persistence(tmp_path, monkeypatch):
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                dict(
                    model="pinned-judge",
                    choices=[
                        dict(message=dict(content='{"judgment":"correct"}'), finish_reason="stop")
                    ],
                )
            ).encode()

    def send(request, timeout):
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", send)
    monkeypatch.setenv("TEST_JUDGE_KEY", "private-token")
    sample = dict(
        question="What is France's capital?", expected="Paris", source_dataset="trivia_qa"
    )
    config = dict(
        judge=dict(
            model="pinned-judge",
            revision="fixed",
            base_url="https://example.test/v1",
            api_key_env="TEST_JUDGE_KEY",
        )
    )
    answer = "The capital of France is Paris."
    result = judge_answer(sample, answer, config, tmp_path)
    assert result["correct"] and result["answer"] == answer
    assert judge_answer(sample, answer, config, tmp_path) == result and len(requests) == 1
    assert "France" in requests[0]["messages"][0]["content"]
    judge_answer(dict(sample, question="Different question?"), answer, config, tmp_path)
    assert len(requests) == 2
    assert all("private-token" not in f.read_text() for f in tmp_path.glob("*.json"))


@pytest.mark.parametrize(
    "text", ["yes", '{"judgment":"maybe"}', "the score is [Correct] and the score is [Incorrect]"]
)
def test_malformed_judge_never_falls_back_to_matching(text):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_verdict(text, "llama_questions" if "score" in text else "trivia_qa")


def test_missing_judge_is_unavailable_even_for_identical_answer():
    sample = dict(task="s2t", language="en", expected="Paris", question="Capital?")
    result = evaluate(sample, dict(is_success=True, text="Paris"), None, LocalScorers({}), {})
    assert result["accuracy"]["status"] == "unavailable"


async def test_offline_judge_uses_frozen_question_without_new_generation(
    server, tmp_path, monkeypatch
):
    import io

    from test_runner import args, workload

    from moss_speech_bench.cli import run
    from moss_speech_bench.quality import quality_report, score
    from moss_speech_bench.storage import read_json, write_json

    url, state = server
    config = workload(tmp_path, ("question", "HTTP_FAIL"))
    workload_spec = read_json(config)
    workload_spec["datasets"][0].update(id="trivia_qa", task="s2t")
    write_json(config, workload_spec)
    write_json(
        tmp_path / "samples.json",
        [
            dict(id=str(i), audio="input.wav", expected="Paris", question="Capital of France?")
            for i in range(2)
        ],
    )
    (tmp_path / "input.wav").write_bytes(b"fixture")
    output = tmp_path / "run"
    await run(args(config, url, output))
    before = (output / "performance.json").read_bytes()
    count = len(state["payloads"])
    assert all("Capital of France?" not in json.dumps(p) for p in state["payloads"])
    calls = []

    def judge(request, timeout):
        calls.append(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                dict(
                    choices=[
                        dict(message=dict(content='{"judgment":"correct"}'), finish_reason="stop")
                    ]
                )
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", judge)
    score_config = tmp_path / "score.json"
    write_json(
        score_config,
        dict(judge=dict(base_url="https://example.test/v1", model="judge", revision="fixed")),
    )
    result = score(output, score_config)
    assert result["status"] == "complete"
    assert len(calls) == 1 and "Capital of France?" in calls[0]["messages"][0]["content"]
    assert score(output, score_config)["cells"] == result["cells"] and len(calls) == 1
    assert quality_report(output)["cells"] == result["cells"]
    assert (output / "performance.json").read_bytes() == before and len(state["payloads"]) == count
