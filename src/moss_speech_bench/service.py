"""Declared HTTP capabilities shared by workloads and HF regression."""

import base64
import hashlib
from pathlib import Path

from .datasets import DEMO_AUDIO_SYSTEM, NATIVE_AUDIO_SYSTEM, SUPPORTED_QA_TASKS


def service_config(spec, root, stream):
    allowed = {"api", "tasks", "streaming", "reference_audio", "system"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Unknown service capability fields")
    tasks = spec.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or len(set(tasks)) != len(tasks)
        or not set(tasks) <= SUPPORTED_QA_TASKS
    ):
        raise ValueError("Service requires explicit unique QA tasks")
    if spec.get("api") not in ("chat", "speech") or type(spec.get("streaming")) is not bool:
        raise ValueError("Service requires api=chat/speech and a streaming boolean")
    if stream and not spec["streaming"]:
        raise ValueError("Service does not support streaming; use --no-stream")
    result = dict(spec)
    if spec["api"] == "speech":
        if tasks != ["t2s"] or spec["streaming"]:
            raise ValueError("Speech transport requires tasks=[t2s] and streaming=false")
        if spec.get("system") not in (DEMO_AUDIO_SYSTEM, NATIVE_AUDIO_SYSTEM):
            raise ValueError("Named speech QA requires an official audio system prompt")
    if spec["api"] == "speech" or "reference_audio" in spec:
        path = (Path(root) / spec["reference_audio"]).resolve()
        raw = path.read_bytes()
        if not raw:
            raise ValueError("Empty reference audio")
        result.update(
            reference_audio=str(path),
            reference_sha256=hashlib.sha256(raw).hexdigest(),
            reference_data="data:audio/wav;base64," + base64.b64encode(raw).decode(),
        )
    return result
