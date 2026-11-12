import pytest
from aiohttp import web

from moss_speech_bench.asr import transcribe_many


async def test_asr_multipart_fanout_deduplication_cache_and_failure_retry(tmp_path):
    calls = []
    fail = True

    async def transcribe(request):
        fields = await request.post()
        calls.append(dict(fields))
        assert fields["model"] == "test-asr" and fields["language"] == "en"
        assert fields["response_format"] == "json" and fields["file"].file.read() == b"wav"
        if fail:
            return web.json_response(dict(error="temporary"), status=503)
        return web.json_response(dict(text="Paris"))

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"wav")
        spec = dict(
            base_url=f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}",
            model="test-asr",
            revision="fixed",
            concurrency=2,
        )
        items = [(audio, "en", spec)] * 2
        cache = tmp_path / "cache"
        assert next(iter((await transcribe_many(items, cache)).values()))["status"] == "failed"
        fail = False
        result = await transcribe_many(items, cache)
        assert next(iter(result.values()))["text"] == "Paris" and len(calls) == 2
        assert await transcribe_many(items, cache) == result and len(calls) == 2
        file = next(cache.glob("*.json"))
        file.write_text('{"_hash":"changed"}')
        with pytest.raises(ValueError, match="cache hash mismatch"):
            await transcribe_many(items, cache)
    finally:
        await runner.cleanup()
