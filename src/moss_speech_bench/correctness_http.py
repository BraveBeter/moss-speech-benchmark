"""Eager-base HTTP assertions, with no server lifecycle or model imports."""

import asyncio
import base64
import json
import os
import wave
from dataclasses import asdict
from pathlib import Path

import aiohttp
import numpy as np

from .client import ChatClient, sse_events
from .storage import read_json, read_jsonl, write_json

PROMPTS = {
    "en": "Answer briefly: what do people drink when they are thirsty?",
    "zh": "请简短回答：人们口渴时通常喝什么？",
}


def payload(config, lang="en", source="text", target="text", stream=False, limit=256):
    content = (
        PROMPTS[lang]
        if source == "text"
        else [
            dict(
                type="input_audio",
                input_audio=dict(
                    data=base64.b64encode(Path(config["audio_" + lang]).read_bytes()).decode(),
                    format="wav",
                ),
            )
        ]
    )
    data = dict(
        model=config["model"],
        messages=[dict(role="user", content=content)],
        modalities=[target],
        stream=stream,
        temperature=0.0,
        top_k=1,
        seed=42,
        max_tokens=limit,
    )
    if stream:
        data["audio"] = dict(format="pcm")
    return data


def pcm(row, directory):
    if not row.wav_path:
        return b""
    with wave.open(str(directory / row.wav_path), "rb") as wav:
        assert wav.getframerate() == 24000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        return wav.readframes(wav.getnframes())


def check(row, request, directory):
    assert row.is_success, row.error
    assert row.http_status == 200 and row.finish_reason in ("stop", "length")
    usage = row.usage
    assert all(
        type(usage.get(k)) is int for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    budget = request.get("max_tokens")
    if budget is None:
        budget = 40960 - usage["prompt_tokens"]
    assert usage["prompt_tokens"] > 0 and 0 < usage["completion_tokens"] <= budget
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    audio = pcm(row, directory)
    if request["modalities"] == ["text"]:
        assert row.text.strip() and not audio
    else:
        assert audio and np.any(np.frombuffer(audio, dtype="<i2") != 0)
    if request["stream"]:
        events = read_jsonl(directory / (row.request_id + ".events.jsonl"))
        terminals, done, chunks = 0, 0, []
        for record in events:
            assert done == 0, "Payload after DONE"
            if record["data"] == "[DONE]":
                assert terminals == 1
                done += 1
                continue
            event = json.loads(record["data"])
            assert len(event.get("choices", [])) == 1
            choice = event["choices"][0]
            delta = choice.get("delta", {})
            if choice.get("finish_reason") is not None:
                assert not delta.get("content") and not delta.get("audio")
                terminals += 1
            else:
                assert terminals == 0, "Choice after terminal"
            if (delta.get("audio") or {}).get("data"):
                part = base64.b64decode(delta["audio"]["data"], validate=True)
                assert part and len(part) % 2 == 0
                chunks.append(part)
        assert terminals == done == 1
        if audio:
            assert b"".join(chunks) == audio
            samples = len(audio) // 2
            assert samples % 1920 == 0 and 0 < samples // 1920 <= usage["completion_tokens"]
    return audio


async def run_http(gate, config, output):
    output = Path(output)
    key = os.environ.get(config["api_key_env"]) if config.get("api_key_env") else None
    if config.get("api_key_env") and not key:
        raise ValueError("API key environment variable is empty")
    client = ChatClient(config["base_url"], output, api_key=key)
    attempts, results = [], []

    async def send(case, request, language="en", session=None):
        sample = dict(
            id=case,
            task=("s" if any(isinstance(m["content"], list) for m in request["messages"]) else "t")
            + ("s" if request["modalities"] == ["audio"] else "t"),
            language=language,
            payload=request,
        )
        if session is None:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.get("timeout", 300))
            ) as own:
                row = await client.send(own, sample)
        else:
            row = await client.send(session, sample)
        attempts.append(dict(case=case, request=request, result=asdict(row)))
        write_json(output / "attempts.json", attempts)
        check(row, request, output)
        return row

    async def logical(case, function):
        try:
            detail = await function()
            results.append(dict(id=case, status="pass", detail=detail))
        except Exception as exc:
            results.append(dict(id=case, status="fail", error=type(exc).__name__ + ": " + str(exc)))
        write_json(output / "checks.json", results)

    if gate == "C07":
        for lang in ("en", "zh"):
            for source in ("text", "audio"):
                for target in ("text", "audio"):
                    for stream in (False, True):
                        case = f"{lang}-{source}-{target}-{stream}"

                        async def matrix(
                            case=case, lang=lang, source=source, target=target, stream=stream
                        ):
                            await send(case, payload(config, lang, source, target, stream), lang)

                        await logical(case, matrix)

        async def eos():
            request = payload(config, limit=512)
            request["messages"][0]["content"] = "Reply with exactly one word: hello."
            row = await send("natural-eos", request)
            assert row.finish_reason == "stop" and row.usage["completion_tokens"] < 512

        async def tail():
            request = payload(config, target="audio", stream=True, limit=64)
            request["messages"][0]["content"] = (
                "Count aloud from one to one hundred, without skipping any numbers."
            )
            row = await send("length-tail", request)
            assert row.finish_reason == "length" and row.usage["completion_tokens"] == 64
            assert row.chunk_audio_duration_s[-1] > 0

        async def invalid():
            request = payload(config)
            request["min_p"] = 0.2
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.get("timeout", 300))
            ) as session:
                async with session.post(
                    client.url, json=request, headers=client.headers
                ) as response:
                    body = await response.text()
                    attempts.append(
                        dict(
                            case="invalid", request=request, http_status=response.status, body=body
                        )
                    )
                    write_json(output / "attempts.json", attempts)
                    assert response.status == 400 and "min_p" in body
            await send("after-invalid", payload(config))

        async def disconnect():
            request = payload(config, target="audio", stream=True, limit=512)
            events = []
            closed = False
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=config.get("timeout", 300))
                ) as session:
                    async with session.post(
                        client.url, json=request, headers=client.headers
                    ) as response:
                        assert response.status == 200
                        async for raw, timestamp in sse_events(response):
                            events.append(dict(data=raw, timestamp=timestamp))
                            event = json.loads(raw)
                            choice = event["choices"][0]
                            assert choice.get("finish_reason") is None
                            data = (choice.get("delta", {}).get("audio") or {}).get("data")
                            if data:
                                assert base64.b64decode(data, validate=True)
                                closed = True
                                break
            finally:
                attempts.append(
                    dict(
                        case="disconnect",
                        request=request,
                        events=events,
                        client_closed=closed,
                        server_cancel_ack=None,
                    )
                )
                write_json(output / "attempts.json", attempts)
            assert closed
            await send("after-disconnect", payload(config))
            return dict(client_closed=True, server_cancel_ack=None)

        for name, function in [
            ("natural-eos", eos),
            ("length-tail", tail),
            ("disconnect", disconnect),
            ("invalid", invalid),
        ]:
            await logical(name, function)
        if config.get("profile", "").startswith("context-"):
            probes = config["context_probes"]
            assert [p["prompt_tokens"] for p in probes] == [40959, 40960]

            async def context_boundary():
                trials = [
                    (p, status, omitted)
                    for p, status in zip(probes, (200, 400))
                    for omitted in (
                        [False, True] if config["profile"] != "context-window" else [False]
                    )
                ]
                for probe, expected_status, omitted in trials:
                    request = payload(config, limit=1)
                    request["messages"][0]["content"] = probe["text"]
                    if omitted:
                        request.pop("max_tokens")
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=config.get("timeout", 300))
                    ) as session:
                        async with session.post(
                            client.url, json=request, headers=client.headers
                        ) as response:
                            body = await response.json()
                            attempts.append(
                                dict(
                                    case="context-boundary",
                                    request=request,
                                    http_status=response.status,
                                    response=body,
                                    expected_prompt_tokens=probe["prompt_tokens"],
                                )
                            )
                            write_json(output / "attempts.json", attempts)
                            assert response.status == expected_status, body
                            if expected_status == 200:
                                assert body["usage"]["prompt_tokens"] == 40959
                                assert body["usage"]["completion_tokens"] == 1
                                assert body["usage"]["total_tokens"] == 40960
                            else:
                                assert "context" in str(body).lower() or "40960" in str(body)
                await send("after-context-rejection", payload(config))
                return dict(
                    accepted_prompt_tokens=40959,
                    reserved_generation_tokens=1,
                    rejected_prompt_tokens=40960,
                )

            async def generation_boundary(target):
                budget = 1001 if config["profile"] != "context-window" else 1000
                request = payload(config, target=target, stream=True, limit=budget)
                request["messages"][0]["content"] = (
                    "请写一篇不少于两千字的科幻故事：一位宇航员在木星的卫星上发现了一座失落的城市。请直接开始写完整故事，详细描写探索的经过和人物对话，不要写提纲，不要总结任务。"
                )
                if target == "text":
                    request = read_json(config["long_text_request"])
                    request.update(model=config["model"], max_tokens=budget, stream=True)
                    request["audio"] = dict(format="pcm")
                row = await send(
                    f"generation-{budget}-" + target, request, "en" if target == "text" else "zh"
                )
                assert row.usage["completion_tokens"] == budget and row.finish_reason == "length", (
                    row.usage
                )
                return dict(
                    completion_tokens=row.usage["completion_tokens"],
                    finish_reason=row.finish_reason,
                )

            async def over_generation():
                budget = 40960 if config["profile"] != "context-window" else 1001
                request = payload(config, limit=budget)
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=config.get("timeout", 300))
                ) as session:
                    async with session.post(
                        client.url, json=request, headers=client.headers
                    ) as response:
                        body = await response.text()
                        attempts.append(
                            dict(
                                case=f"generation-{budget}",
                                request=request,
                                http_status=response.status,
                                body=body,
                            )
                        )
                        write_json(output / "attempts.json", attempts)
                        assert (
                            response.status == 400
                            and str(40959 if budget == 40960 else 1000) in body
                        ), body
                await send("after-generation-rejection", payload(config))

            async def omitted_eos(target):
                request = payload(config, target=target, stream=True)
                request["messages"][0]["content"] = "Reply with exactly one word: hello."
                if target == "text":
                    request.pop("max_tokens")
                else:
                    request["max_tokens"] = None
                row = await send("omitted-eos-" + target, request)
                assert row.finish_reason == "stop", row.usage
                assert 0 < row.usage["completion_tokens"] < 40960 - row.usage["prompt_tokens"]
                return dict(
                    usage=row.usage,
                    finish_reason=row.finish_reason,
                    budget=request.get("max_tokens"),
                )

            await logical("context-boundary", context_boundary)
            budget = 1001 if config["profile"] != "context-window" else 1000
            await logical(f"generation-{budget}-text", lambda: generation_boundary("text"))
            await logical(f"generation-{budget}-audio", lambda: generation_boundary("audio"))
            await logical(
                "generation-over-context-rejected"
                if config["profile"] != "context-window"
                else "generation-1001-rejected",
                over_generation,
            )
            if config["profile"] != "context-window":
                await logical("omitted-eos-text", lambda: omitted_eos("text"))
                await logical("omitted-eos-audio", lambda: omitted_eos("audio"))
    else:
        for target, limit in [("text", 64), ("audio", 96)]:

            async def concurrent(target=target, limit=limit):
                requests = []
                for color in ("red", "blue"):
                    request = payload(config, target=target, limit=limit)
                    request["messages"][0]["content"] = f"Reply with exactly one word: {color}."
                    requests.append(request)
                serial = [await send(f"{target}-serial-{i}", r) for i, r in enumerate(requests)]

                def signature(row):
                    return row.text, row.finish_reason, row.usage, pcm(row, output)

                assert (serial[0].text, pcm(serial[0], output)) != (
                    serial[1].text,
                    pcm(serial[1], output),
                ), "Serial red/blue are indistinguishable"
                barrier = asyncio.Event()

                async def parallel(i, request):
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=config.get("timeout", 300)),
                        connector=aiohttp.TCPConnector(limit=1),
                    ) as session:
                        await barrier.wait()
                        return await send(f"{target}-concurrent-{i}", request, session=session)

                tasks = [asyncio.create_task(parallel(i, r)) for i, r in enumerate(requests)]
                barrier.set()
                parallel_rows = await asyncio.gather(*tasks)
                overlap = min(r.ended_s for r in parallel_rows) - max(
                    r.started_s for r in parallel_rows
                )
                assert overlap > 0, "Concurrent request intervals did not overlap"
                differences = []
                for i, (a, b) in enumerate(zip(serial, parallel_rows)):
                    ap, bp = pcm(a, output), pcm(b, output)
                    if signature(a) != signature(b):
                        aa, bb = np.frombuffer(ap, dtype="<i2"), np.frombuffer(bp, dtype="<i2")
                        indices = (
                            np.flatnonzero(aa != bb)
                            if aa.shape == bb.shape
                            else np.array([], dtype=int)
                        )
                        differences.append(
                            dict(
                                request=i,
                                same_length=len(ap) == len(bp),
                                different_samples=int(len(indices)),
                                first=indices[:20].tolist(),
                            )
                        )
                write_json(
                    output / (target + "-comparison.json"),
                    dict(overlap_s=overlap, differences=differences),
                )
                assert not differences, differences
                return dict(overlap_s=overlap, post_attempts=4)

            await logical(target, concurrent)
    expected = (
        (
            26
            if config.get("profile", "").startswith("context-")
            and config["profile"] != "context-window"
            else 24
            if config.get("profile") == "context-window"
            else 20
        )
        if gate == "C07"
        else 2
    )
    return dict(
        checks=results,
        http_attempts=len(attempts),
        expected=expected,
        status="pass"
        if len(results) == expected and all(r["status"] == "pass" for r in results)
        else "fail",
    )
