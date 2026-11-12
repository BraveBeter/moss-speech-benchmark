"""Declared HTTP capabilities; deployments and GPU allocation stay outside E2E."""

import base64

from .storage import file_hash

TEXT_SYSTEM = "You are a helpful assistant. Respond with text outputs."
AUDIO_SYSTEM = "You are a helpful assistant. Respond with spoken outputs."


def service_config(spec, root, stream):
    if not isinstance(spec, dict) or set(spec) - {
        "api",
        "tasks",
        "streaming",
        "reference_audio",
        "system",
    }:
        raise ValueError("Unknown service capability fields")
    tasks = spec.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or len(set(tasks)) != len(tasks)
        or not set(tasks) <= {"t2t", "t2s", "s2t", "s2s"}
    ):
        raise ValueError("Service requires unique t2t/t2s/s2t/s2s tasks")
    if spec.get("api") not in ("chat", "speech") or type(spec.get("streaming")) is not bool:
        raise ValueError("Service requires api=chat/speech and a streaming boolean")
    if stream and not spec["streaming"]:
        raise ValueError("Service does not support streaming; use --no-stream")
    result = dict(spec)
    if spec["api"] == "speech":
        if tasks != ["t2s"] or spec["streaming"]:
            raise ValueError("Speech API requires tasks=[t2s] and streaming=false")
        system = spec.get("system", AUDIO_SYSTEM)
        if not isinstance(system, str) or not system.strip():
            raise ValueError("Speech API requires a nonempty declared system prompt")
        audio = (root / spec["reference_audio"]).resolve()
        if not audio.stat().st_size:
            raise ValueError("Empty reference audio")
        result.update(
            system=system,
            reference_audio=str(audio),
            reference_sha256=file_hash(audio),
            reference_data="data:audio/wav;base64," + base64.b64encode(audio.read_bytes()).decode(),
        )
    elif set(spec) & {"system", "reference_audio"}:
        raise ValueError("Chat uses the standard per-modality system prompts")
    return result
