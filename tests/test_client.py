import aiohttp
import pytest

from moss_speech_bench.client import ChatClient


def sample(text="hello", task="t2s", stream=True):
    return {
        "id": text,
        "language": "en",
        "task": task,
        "payload": {
            "messages": [{"role": "user", "content": text}],
            "stream": stream,
            "modalities": ["audio" if task.endswith("s") else "text"],
        },
    }


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("task", ["t2t", "t2s"])
async def test_real_http_payloads_and_audio(server, tmp_path, stream, task):
    url, _ = server
    async with aiohttp.ClientSession() as session:
        result = await ChatClient(url, tmp_path).send(session, sample(task=task, stream=stream))
    assert result.is_success
    assert result.completion_tokens == 7
    assert result.latency_s > 0
    if task == "t2s":
        assert result.audio_duration_s == (0.2 if stream else 0.1)
        assert result.rtf == result.latency_s / result.audio_duration_s
        assert bool(result.audio_ttfp_s) == stream
        assert result.audio_chunk_count == (2 if stream else 0)
        assert (tmp_path / result.wav_path).is_file()
    else:
        assert "你好🙂" in result.text
        assert bool(result.text_ttft_s) == stream
        assert result.audio_ttfp_s is None
    assert len(list(tmp_path.glob("*.result.json"))) == 1


@pytest.mark.parametrize(
    "text,error", [("HTTP_FAIL", "503"), ("MISSING_DONE", "missing finish or DONE")]
)
async def test_failures_are_durable(server, tmp_path, text, error):
    url, _ = server
    async with aiohttp.ClientSession() as session:
        result = await ChatClient(url, tmp_path).send(session, sample(text=text))
    assert not result.is_success
    assert error in result.error
    assert list(tmp_path.glob("*.events.jsonl"))
    if text == "MISSING_DONE":
        assert result.audio_duration_s == 0.2  # partial payloads retained


async def test_cancel_during_storage_waits_for_artifacts(server, tmp_path):
    import asyncio
    import threading

    url, _ = server
    client = ChatClient(url, tmp_path)
    entered, release = threading.Event(), threading.Event()
    persist = client._persist

    def slow_persist(*args):
        entered.set()
        assert release.wait(timeout=5)
        persist(*args)

    client._persist = slow_persist
    async with aiohttp.ClientSession() as session:
        task = asyncio.create_task(client.send(session, sample()))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.005)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()  # background I/O must finish before run finalization
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(list(tmp_path.glob("*.result.json"))) == 1
