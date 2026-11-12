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
