import asyncio
import base64
import io
import json
import wave

import pytest_asyncio
from aiohttp import web


@pytest_asyncio.fixture
async def server():
    state = {"active": 0, "peak": 0, "payloads": []}

    async def handle(request):
        payload = await request.json()
        state["payloads"].append(payload)
        # MOSS accepts request-level format controls only on streaming requests.
        if payload.get("audio") and not payload["stream"]:
            return web.json_response(
                {"error": "request-level audio configuration unsupported"}, status=400
            )
        text = str(payload["messages"])
        if "HTTP_FAIL" in text:
            return web.json_response({"error": "deliberate failure"}, status=503)
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            await asyncio.sleep(0.01)
            audio = payload["modalities"] == ["audio"]
            pcm = b"\x01\x00" * 2400
            usage = {} if "NO_USAGE" in text else {"completion_tokens": 7, "prompt_tokens": 5}
            if not payload["stream"]:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(pcm)
                return web.json_response(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": "你好🙂" if not audio else "",
                                    "audio": {
                                        "data": base64.b64encode(buffer.getvalue()).decode(),
                                    }
                                    if audio
                                    else None,
                                },
                            }
                        ],
                        "usage": usage,
                    }
                )
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for i in range(2):
                delta = (
                    {"audio": {"data": base64.b64encode(pcm).decode()}}
                    if audio
                    else {"content": "你好🙂"}
                )
                raw = (
                    "data: "
                    + json.dumps({"choices": [{"delta": delta}]}, ensure_ascii=False)
                    + "\r\n\r\n"
                ).encode()
                # Fragment a UTF-8/SSE payload at arbitrary byte positions.
                for part in (raw[:17], raw[17:31], raw[31:]):
                    await response.write(part)
                await asyncio.sleep(0.005)
            terminal = {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": usage}
            await response.write(("data: " + json.dumps(terminal) + "\n\n").encode())
            if "MISSING_DONE" not in text:
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        finally:
            state["active"] -= 1

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
    try:
        yield url, state
    finally:
        await runner.cleanup()


@pytest_asyncio.fixture
async def correctness_stub(monkeypatch):
    """Real orchestration is tested separately; gate before any fake HTTP traffic."""
    from moss_speech_bench import correctness
    from moss_speech_bench.storage import digest, write_json

    state = {"status": "pass", "calls": 0}
    monkeypatch.setattr(correctness, "load_suite", lambda _: {"base_url": state["url"]})

    async def run(_, output, selected=None, resume=False):
        state["calls"] += 1
        config = {"base_url": state["url"]}
        suite = dict(
            status=state["status"],
            config=config,
            config_hash=digest(config),
            gates={g: dict(status="pass", checks=[dict(status="pass")]) for g in correctness.GATES},
        )
        if state["status"] != "pass":
            suite["gates"]["C04"] = dict(status="fail", checks=[])
        write_json(output / "correctness.json", suite)
        return suite

    monkeypatch.setattr(correctness, "run_correctness", run)
    return state
