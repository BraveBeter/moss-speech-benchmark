"""Cached OpenAudioBench semantic judging through a configured third-party endpoint."""

import json
import os
import re
import time
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .judge_prompts import PROMPTS, UPSTREAM_COMMIT
from .storage import digest, read_json, write_json


def parse_verdict(text, dataset):
    if dataset == "llama_questions":
        matches = re.findall(r"the score is \[(correct|incorrect)\]", text, re.I)
        if len(matches) != 1:
            raise ValueError("Judge must return one unambiguous verdict")
        return matches[0].lower() == "correct"
    start = text.find("{")
    raw = text[start:] if start >= 0 else text
    value, end = json.JSONDecoder().raw_decode(raw)
    if not isinstance(value, dict) or "{" in raw[end:] or "}" in raw[end:]:
        raise ValueError("Ambiguous Judge response")
    verdict = value.get("judgment", "").lower()
    if verdict not in ("correct", "incorrect"):
        raise ValueError("Invalid Judge judgment")
    return verdict == "correct"


def judge_answer(sample, answer, config, cache, deadline=None):
    spec = config.get("judge")
    if not spec:
        raise ValueError("Third-party judge configuration is required; no EM fallback")
    dataset = sample.get("source_dataset")
    if dataset not in PROMPTS or not sample.get("question"):
        raise ValueError("Judge requires frozen question and source_dataset metadata")
    url = urlsplit(spec["base_url"])
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Invalid Judge base_url")
    if not spec.get("model") or not spec.get("revision"):
        raise ValueError("Judge model and revision must be explicit")
    prompt = PROMPTS[dataset].format(
        question=sample["question"], answer=sample["expected"], prediction=answer
    )
    payload = dict(
        model=spec["model"],
        messages=[dict(role="user", content=prompt)],
        temperature=0,
        max_tokens=1024,
    )
    binding = dict(request=payload, judge=spec, template_commit=UPSTREAM_COMMIT)
    if "thinking" in spec:
        payload["thinking"] = spec["thinking"]
    path = Path(cache) / (digest(binding) + ".json")
    if path.exists():
        saved = read_json(path)
        checksum = saved.pop("_hash")
        if checksum != digest(saved):
            raise ValueError("Judge cache hash mismatch")
        return saved
    timeout = float(spec.get("timeout", 120))
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("Judge scoring budget exhausted")
    headers = {"Content-Type": "application/json"}
    key_name = spec.get("api_key_env", "OPENAI_API_KEY")
    if os.environ.get(key_name):
        headers["Authorization"] = "Bearer " + os.environ[key_name]
    request = urllib.request.Request(
        spec["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    attempt = Path(cache) / "attempts" / (digest(binding) + "-" + uuid.uuid4().hex + ".json")
    evidence = dict(binding=binding, response=body)
    write_json(attempt, dict(evidence, _hash=digest(evidence)))
    text = body["choices"][0]["message"]["content"]
    if body["choices"][0].get("finish_reason") not in ("stop", None):
        raise ValueError("Judge response was truncated or rejected")
    correct = parse_verdict(text, dataset)
    result = dict(
        correct=correct,
        valid=True,
        answer=answer,
        judge_response=text,
        response_id=body.get("id"),
        resolved_model=body.get("model"),
        binding=binding,
    )
    write_json(path, dict(result, _hash=digest(result)))
    return result
