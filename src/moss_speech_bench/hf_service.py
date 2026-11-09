"""Behavioral checks for the declared public generation interface."""

import asyncio
from dataclasses import asdict
from pathlib import Path

import aiohttp

from .client import ChatClient, decode_wav
from .service import service_config
from .storage import write_json


async def run_http(gate, config, output):
    output = Path(output)
    service = service_config(config["service"], config["workspace_root"], False)
    client = ChatClient(config["base_url"], output, sample_rate=24000)
    checks, attempts = [], []

    def sample(case, id):
        return dict(
            id=id,
            task="t2s",
            language=case["language"],
            endpoint="/v1/audio/speech",
            payload=dict(
                model=config["model"],
                input=case["text"],
                ref_audio=service["reference_data"],
                stream=False,
                response_format="wav",
                seed=config["sampling"]["seed"],
                temperature=0,
                top_k=1,
                repetition_penalty=config["sampling"]["repetition_penalty"],
                max_new_tokens=config["http_max_tokens"],
            ),
        )

    async def send(session, case, id):
        request = sample(case, id)
        row = await client.send(session, request)
        attempts.append(dict(id=id, payload=request["payload"], result=asdict(row)))
        write_json(output / "attempts.json", attempts)
        assert row.is_success, row.error
        pcm, rate, channels = decode_wav((output / row.wav_path).read_bytes())
        assert pcm and rate == 24000 and channels == 1
        assert row.finish_reason is None and not row.usage
        assert row.audio_ttfp_s is None, "Buffered transport must not fabricate TTFA"
        return row, pcm

    async def logical(id, function):
        try:
            detail = await function()
            checks.append(dict(id=id, status="pass", detail=detail))
        except Exception as error:
            checks.append(
                dict(id=id, status="fail", reason=type(error).__name__ + ": " + str(error))
            )
        write_json(output / "checks.json", checks)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=config.get("http_timeout", 300)),
        connector=aiohttp.TCPConnector(limit=4),
    ) as session:
        if gate == "C07":
            for case in (
                config["cases"][0],
                next(c for c in config["cases"] if c["language"] == "zh"),
            ):

                async def valid(case=case):
                    row, _ = await send(session, case, "valid-" + case["id"])
                    return dict(
                        audio_seconds=row.audio_duration_s,
                        finish_reason="unreported",
                        usage="unreported",
                    )

                await logical("supported-" + case["id"], valid)
            invalid = [
                ("empty-input", dict(input="")),
                ("zero-budget", dict(max_new_tokens=0)),
                ("unsupported-stream", dict(stream=True)),
                ("missing-reference", dict(ref_audio=None)),
                ("invalid-reference", dict(ref_audio="data:audio/wav;base64,YmFk")),
            ]
            for id, changes in invalid:

                async def reject(id=id, changes=changes):
                    request = sample(config["cases"][0], id)["payload"] | changes
                    async with session.post(
                        config["base_url"].rstrip("/") + "/v1/audio/speech", json=request
                    ) as response:
                        body = await response.text()
                        attempts.append(
                            dict(id=id, payload=request, http_status=response.status, body=body)
                        )
                        write_json(output / "attempts.json", attempts)
                        assert response.status in (400, 422), body
                    return dict(status=response.status)

                await logical(id, reject)

            async def recovery():
                await send(session, config["cases"][0], "after-rejections")

            await logical("recovery-after-rejections", recovery)
        elif gate == "C08":

            async def isolation():
                cases = config["cases"][:4]
                serial = [await send(session, c, "serial-" + c["id"]) for c in cases]
                assert len({pcm for _, pcm in serial}) == 4, (
                    "Different prompts produced indistinguishable serial audio"
                )
                barrier = asyncio.Event()

                async def parallel(case):
                    await barrier.wait()
                    return await send(session, case, "parallel-" + case["id"])

                tasks = [asyncio.create_task(parallel(c)) for c in cases]
                barrier.set()
                concurrent = await asyncio.gather(*tasks)
                overlap = min(row.ended_s for row, _ in concurrent) - max(
                    row.started_s for row, _ in concurrent
                )
                assert overlap > 0, "Request intervals did not overlap"
                for (a, apcm), (b, bpcm) in zip(serial, concurrent, strict=True):
                    assert apcm == bpcm, "Serial/concurrent PCM differs"
                    assert a.usage == b.usage and a.finish_reason == b.finish_reason
                repeated, pcm = await send(session, cases[0], "repeat-after-concurrency")
                assert pcm == serial[0][1]
                return dict(
                    paired=4,
                    pcm_exact=True,
                    overlap_seconds=overlap,
                    engine_batch_size="not inferred from HTTP",
                    finish_reason=repeated.finish_reason,
                )

            await logical("serial-parallel-repeat-isolation", isolation)
        else:
            raise ValueError("Unknown HTTP gate")
    return dict(
        status="pass" if checks and all(c["status"] == "pass" for c in checks) else "fail",
        checks=checks,
        expected=len(checks),
        http_attempts=len(attempts),
    )
