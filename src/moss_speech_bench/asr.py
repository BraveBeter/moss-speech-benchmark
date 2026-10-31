"""Qwen3-ASR multipart protocol and bounded fan-out from Omni's ASR CI."""

import asyncio
import hashlib
import time
from pathlib import Path

import aiohttp

from .storage import digest, read_json, write_json


def transcription_key(path, language, spec):
    return digest(
        dict(
            audio_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            language=language,
            scorer=spec,
            protocol="omni-qwen3-asr-v1",
        )
    )


async def transcribe_many(items, cache, deadline=None):
    """Reuse complete transcripts; persist every attempted or deferred request."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in {digest(item[2]): item[2] for item in items}.values():
        concurrency = spec.get("concurrency", 32)
        if type(concurrency) is not int or concurrency <= 0:
            raise ValueError("ASR concurrency must be positive")
        semaphore = asyncio.Semaphore(concurrency)
        timeout = aiohttp.ClientTimeout(total=spec.get("timeout", 300))
        async with aiohttp.ClientSession(
            timeout=timeout, connector=aiohttp.TCPConnector(limit=concurrency)
        ) as session:

            async def send(path, language, key):
                file = cache / (key + ".json")
                if file.exists():
                    saved = read_json(file)
                    checksum = saved.pop("_hash", None)
                    if checksum != digest(saved):
                        raise ValueError("ASR cache hash mismatch")
                    if saved["status"] != "budget_exhausted":
                        results[key] = saved
                        return
                async with semaphore:
                    result = dict(status="budget_exhausted", audio=str(path), language=language)
                    started = time.monotonic()
                    if deadline is None or started < deadline:
                        form = aiohttp.FormData()
                        form.add_field("model", spec["model"])
                        hint = spec.get("language_hint", language)
                        if hint != "auto":
                            form.add_field("language", hint)
                        form.add_field("response_format", "json")
                        form.add_field(
                            "file",
                            Path(path).read_bytes(),
                            filename=Path(path).name,
                            content_type="audio/wav",
                        )
                        try:
                            remaining = (
                                deadline - time.monotonic()
                                if deadline
                                else spec.get("timeout", 300)
                            )
                            async with session.post(
                                spec["base_url"].rstrip("/") + "/v1/audio/transcriptions",
                                data=form,
                                timeout=aiohttp.ClientTimeout(
                                    total=max(0.001, min(remaining, spec.get("timeout", 300)))
                                ),
                            ) as response:
                                body = await response.json()
                                if response.status != 200 or not isinstance(body.get("text"), str):
                                    raise ValueError(f"ASR HTTP {response.status}: {body}")
                                result.update(status="scored", text=body["text"])
                        except Exception as error:
                            result.update(
                                status="failed", error=type(error).__name__ + ": " + str(error)
                            )
                    result["started_s"] = started
                    result["ended_s"] = time.monotonic()
                    result["elapsed_s"] = result["ended_s"] - started
                    results[key] = result
                    write_json(file, dict(result, _hash=digest(result)))

            unique = {transcription_key(p, lang, s): (p, lang) for p, lang, s in items if s == spec}
            await asyncio.gather(*(send(p, lang, key) for key, (p, lang) in unique.items()))
    return results
