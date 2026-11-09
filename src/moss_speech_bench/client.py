"""MOSS chat adapter; timestamps refer to fully received SSE application events."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

from ._vendor.data import RequestResult
from ._vendor.metrics.playback_continuity import compute_max_playback_underrun_s
from .storage import digest, dumps, write_json


@dataclass
class Observation(RequestResult):
    sample_id: str = ""
    task: str = ""
    language: str = ""
    finish_reason: str | None = None
    http_status: int | None = None
    usage: dict = field(default_factory=dict)
    payload_hash: str = ""
    started_s: float = 0.0
    ended_s: float = 0.0
    audio_arrival_s: list[float] = field(default_factory=list)
    output_hashes: dict = field(default_factory=dict)
    error_type: str | None = None
    endpoint: str = "/v1/chat/completions"


async def sse_events(response):
    """Reassemble UTF-8/SSE independently of TCP reads, including multiline data."""
    buffer = b""
    data = []
    async for block in response.content.iter_any():
        buffer += block
        if len(buffer) > 32 * 1024 * 1024:
            raise ValueError("SSE line exceeds 32 MiB")
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if not line:
                if data:
                    yield b"\n".join(data).decode("utf-8"), time.perf_counter()
                    data.clear()
            elif line.startswith(b"data:"):
                data.append(line[5:].removeprefix(b" "))
    if buffer.strip():
        raise ValueError("Incomplete SSE line at EOF")
    if data:
        raise ValueError("Incomplete SSE event at EOF")


def decode_wav(data):
    with wave.open(io.BytesIO(data), "rb") as audio:
        if audio.getcomptype() != "NONE" or audio.getsampwidth() != 2:
            raise ValueError("Expected uncompressed PCM16 WAV")
        pcm = audio.readframes(audio.getnframes())
        rate, channels = audio.getframerate(), audio.getnchannels()
        if len(pcm) != audio.getnframes() * channels * 2 or not pcm:
            raise ValueError("Empty or truncated WAV")
    return pcm, rate, channels


class ChatClient:
    def __init__(self, url, output_dir, *, sample_rate=24000, api_key=None):
        self.origin = url.rstrip("/")
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.directory = Path(output_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def send(self, session: aiohttp.ClientSession, sample: dict) -> Observation:
        payload = sample["payload"]
        endpoint = sample.get("endpoint", "/v1/chat/completions")
        if endpoint not in ("/v1/chat/completions", "/v1/audio/speech"):
            raise ValueError("Unsupported generation endpoint")
        speech = endpoint == "/v1/audio/speech"
        if speech and (payload.get("stream") or sample["task"] != "t2s"):
            raise ValueError("Speech endpoint requires non-streaming t2s")
        key = digest(sample["id"])[:24]
        result = Observation(
            request_id=key,
            sample_id=sample["id"],
            task=sample["task"],
            language=sample["language"],
            payload_hash=digest(payload),
            endpoint=endpoint,
        )
        events, pcm_parts = [], []
        rate, channels = self.sample_rate, 1
        started = time.perf_counter()
        result.started_s = started
        cancelled = False
        try:
            async with session.post(
                self.origin + endpoint, json=payload, headers=self.headers
            ) as response:
                result.http_status = response.status
                if response.status != 200:
                    body = await response.text()
                    events.append({"t_s": time.perf_counter() - started, "body": body})
                    raise ValueError(f"HTTP {response.status}: {body[:500]}")
                if speech:
                    raw = await response.read()
                    result.ended_s = time.perf_counter()
                    pcm, rate, channels = decode_wav(raw)
                    if rate != self.sample_rate or channels != 1:
                        raise ValueError("Speech response must match the declared mono sample rate")
                    pcm_parts.append(pcm)
                    events.append(
                        dict(
                            t_s=result.ended_s - started,
                            response_sha256=hashlib.sha256(raw).hexdigest(),
                            content_type=response.headers.get("Content-Type"),
                            bytes=len(raw),
                        )
                    )
                elif payload["stream"]:
                    if "text/event-stream" not in response.headers.get("Content-Type", ""):
                        raise ValueError("Expected text/event-stream")
                    done = False
                    async for raw, arrived in sse_events(response):
                        elapsed = arrived - started
                        events.append({"t_s": elapsed, "data": raw})
                        if done:
                            raise ValueError("Event after DONE")
                        if raw == "[DONE]":
                            done = True
                            continue
                        event = json.loads(raw)
                        if event.get("error"):
                            raise ValueError(f"Server error: {event['error']}")
                        if event.get("usage"):
                            result.usage = event["usage"]
                        choices = event.get("choices", [])
                        if not choices:  # OpenAI usage-only event.
                            continue
                        if len(choices) != 1:
                            raise ValueError("Expected one choice")
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        text = delta.get("content") or ""
                        audio = delta.get("audio") or {}
                        if result.finish_reason is not None:
                            raise ValueError("Choice after finish event")
                        if text:
                            result.text += text
                            if result.text_ttft_s is None:
                                result.text_ttft_s = elapsed
                        if audio.get("data"):
                            pcm = base64.b64decode(audio["data"], validate=True)
                            if not pcm or len(pcm) % 2:
                                raise ValueError("Invalid PCM16 event")
                            pcm_parts.append(pcm)
                            result.audio_arrival_s.append(elapsed)
                            result.chunk_audio_duration_s.append(len(pcm) / (rate * 2))
                            if result.audio_ttfp_s is None:
                                result.audio_ttfp_s = elapsed
                                result.first_audio_payload_bytes = len(pcm)
                        if choice.get("finish_reason") is not None:
                            result.finish_reason = choice["finish_reason"]
                    if not done or result.finish_reason is None:
                        raise ValueError("Incomplete stream: missing finish or DONE")
                else:
                    raw = await response.read()
                    result.ended_s = time.perf_counter()
                    obj = json.loads(raw)
                    events.append({"t_s": result.ended_s - started, "body": obj})
                    choices = obj.get("choices", [])
                    if obj.get("error") or len(choices) != 1:
                        raise ValueError("Invalid completion")
                    message = choices[0]["message"]
                    result.text = message.get("content") or ""
                    result.finish_reason = choices[0].get("finish_reason")
                    result.usage = obj.get("usage") or {}
                    if (message.get("audio") or {}).get("data"):
                        pcm, rate, channels = decode_wav(
                            base64.b64decode(message["audio"]["data"], validate=True)
                        )
                        pcm_parts.append(pcm)
                result.ended_s = result.ended_s or time.perf_counter()
            pcm = b"".join(pcm_parts)
            if not speech and result.finish_reason not in ("stop", "length"):
                raise ValueError(f"Unsuccessful finish: {result.finish_reason!r}")
            if sample["task"].endswith("s") and not pcm:
                raise ValueError("Empty audio output")
            if sample["task"].endswith("t") and not result.text.strip():
                raise ValueError("Empty text output")
            result.is_success = True
        except asyncio.CancelledError:
            cancelled = True
            result.error = "Client run cancelled"
            result.error_type = "CancelledError"
        except Exception as exc:
            result.error = str(exc)
            result.error_type = type(exc).__name__
        result.ended_s = result.ended_s or time.perf_counter()
        result.latency_s = result.ended_s - started
        result.engine_time_s = result.latency_s  # Request-time proxy, not engine instrumentation.
        for name in ("prompt_tokens", "completion_tokens"):
            count = result.usage.get(name)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                setattr(result, name, count)
        result.tok_per_s = result.completion_tokens / result.latency_s if result.latency_s else 0
        pcm = b"".join(pcm_parts)
        result.audio_duration_s = len(pcm) / (rate * channels * 2)
        result.rtf = result.latency_s / result.audio_duration_s if pcm else 0
        result.audio_chunk_count = len(result.audio_arrival_s)
        result.inter_chunk_s = [
            b - a for a, b in zip(result.audio_arrival_s, result.audio_arrival_s[1:])
        ]
        result.max_playback_underrun_s = compute_max_playback_underrun_s(
            result.audio_arrival_s,
            result.chunk_audio_duration_s,
        )
        # File I/O must not stall other requests on the event loop.
        persistence = asyncio.create_task(
            asyncio.to_thread(self._persist, result, events, pcm, rate, channels)
        )
        try:
            await asyncio.shield(persistence)
        except asyncio.CancelledError:
            # Do not finalize the run while its worker thread still writes artifacts.
            await persistence
            raise
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _persist(self, result, events, pcm, rate, channels):
        # Storage is outside request latency but inside run wall time.
        key = result.request_id
        event_path = self.directory / f"{key}.events.jsonl"
        event_path.write_text("".join(dumps(event) + "\n" for event in events))
        text_path = self.directory / f"{key}.txt"
        text_path.write_text(result.text)
        paths = [event_path, text_path]
        if pcm:
            audio_path = self.directory / f"{key}.wav"
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm)
            result.wav_path = audio_path.name
            paths.append(audio_path)
        result.output_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        write_json(self.directory / f"{key}.result.json", asdict(result))
