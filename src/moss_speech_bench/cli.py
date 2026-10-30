"""Whole-dataset performance runs, repeat/concurrency cells, and offline reporting."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from ._vendor.runner import BenchmarkRunner, RunConfig
from .client import ChatClient
from .datasets import PERFORMANCE, load_workload
from .reporting import collect_cell, report
from .storage import dumps, read_json, source_identity, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("infer", "report", "score", "all"), default="infer")
    p.add_argument("--mode", choices=("custom", "daily", "full"), default="custom")
    p.add_argument("--config", type=Path, help="Whole-dataset workload JSON")
    p.add_argument("--model")
    p.add_argument("--score-config", type=Path, help="Offline scoring assets/config JSON")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--deployment", type=Path, help="Operator-supplied server metadata JSON")
    p.add_argument("--output-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--concurrency", type=int)
    group.add_argument("--concurrencies", help="Comma-separated concurrency sweep")
    p.add_argument("--repeats", type=int)
    p.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument(
        "--sample-rate", type=int, default=24000, help="Declared streaming PCM16 mono rate"
    )
    p.add_argument("--api-key-env", help="Environment variable containing the API key; never saved")
    p.add_argument("--baseline", type=Path)
    p.add_argument(
        "--allow-difference",
        action="append",
        default=[],
        help="Declared contract field allowed to differ",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete cells; rerun interrupted cells in a new attempt",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate complete inputs and print request counts without HTTP",
    )
    return p


async def run(args):
    if args.phase == "all":
        raise ValueError(
            "Use separate infer and score phases to avoid scorer/service resource overlap"
        )
    if args.phase == "score":
        from .quality import score

        quality = await asyncio.to_thread(score, args.output_dir, args.score_config)
        report(args.output_dir, args.baseline, args.allow_difference)
        return {"quality": {"failed": quality["status"] != "complete"}}
    if args.phase == "report":
        return report(args.output_dir, args.baseline, args.allow_difference)
    if not args.config or not args.model:
        raise ValueError("infer requires --config and --model")
    url = urlsplit(args.base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise ValueError(
            "base-url must be an HTTP(S) origin without credentials, path, query or fragment"
        )
    levels = (
        [int(x) for x in args.concurrencies.split(",")]
        if args.concurrencies
        else [args.concurrency]
        if args.concurrency is not None
        else [1, 4, 8, 16]
        if args.mode == "full"
        else [4]
    )
    repeats = args.repeats if args.repeats is not None else 3 if args.mode == "full" else 1
    if not levels or any(c <= 0 for c in levels) or len(set(levels)) != len(levels):
        raise ValueError("Concurrency levels must be unique positive integers")
    if repeats <= 0 or not 0 < args.timeout < float("inf") or args.sample_rate <= 0:
        raise ValueError("Repeats, timeout and sample rate must be positive and finite")
    if args.mode == "daily" and (len(levels) != 1 or repeats != 1):
        raise ValueError("Daily uses exactly one concurrency and one formal repeat")
    datasets, warmup, sampling = load_workload(args.config, args.model, args.stream, args.mode)
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in datasets):
        raise ValueError("Dataset IDs may contain only letters, digits, underscore and hyphen")
    cells = [
        dict(
            id=f"{name}-c{c}-r{repeat}",
            dataset=name,
            concurrency=c,
            repeat=repeat,
            expected=len(rows),
            attempts=[],
            status="pending",
        )
        for name, (rows, _) in datasets.items()
        if args.mode != "full" or name in PERFORMANCE
        for repeat in range(1, repeats + 1)
        for c in levels
    ]
    if args.dry_run:
        return {
            "scope": "performance_only",
            "cells": cells,
            "formal_requests": sum(c["expected"] for c in cells),
            "warmup_requests": len(warmup[0]) * len(cells) if warmup else 0,
        }
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        raise ValueError("API key environment variable is empty")
    deployment = read_json(args.deployment) if args.deployment else None
    if deployment is not None and not isinstance(deployment, dict):
        raise ValueError("Deployment metadata must be an object")
    identity = source_identity()
    contract = {
        "model": args.model,
        "deployment": deployment,
        "mode": args.mode,
        "datasets": {k: v[1] for k, v in datasets.items()},
        "sampling": sampling,
        "stream": args.stream,
        "sample_rate": args.sample_rate,
        "timeout": args.timeout,
        "levels": levels,
        "repeats": repeats,
        "warmup": warmup[1] if warmup else None,
        "client_source_hash": identity["source_hash"],
        "metric_version": "omni-f96b10-moss-client-v1",
    }
    root = args.output_dir
    if args.resume:
        manifest = read_json(root / "manifest.json")
        if manifest["contract"] != contract:
            raise ValueError("Resume contract mismatch; use a new output directory")
        cells = manifest["cells"]
        if manifest["base_url"] != args.base_url:
            raise ValueError("Resume endpoint mismatch")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Output directory is not empty; use --resume or a fresh directory")
        root.mkdir(parents=True, exist_ok=True)
        (root / "inputs").mkdir()
        for name, (rows, _) in datasets.items():
            (root / "inputs" / f"{name}.jsonl").write_text(
                "".join(dumps(row) + "\n" for row in rows)
            )
        manifest = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "scope": "performance_only",
            "contract": contract,
            "base_url": args.base_url,
            "client": identity,
            "cells": cells,
            "python": platform.python_version(),
            "state": "prepared",
            "dependencies": {
                name: importlib.metadata.version(name) for name in ("aiohttp", "numpy", "tqdm")
            },
            "quality_status": "not_scored",
            "correctness": "outside_client_scope",
        }
    manifest["finished_at"] = None
    write_json(root / "manifest.json", manifest)
    try:
        for cell in cells:
            if cell["status"] == "complete":
                _, summary = collect_cell(root, cell)
                if not summary["measurement_complete"]:
                    raise ValueError("Completed cell has incomplete artifacts")
                continue
            attempt_rel = f"cells/{cell['id']}/attempt-{len(cell['attempts']) + 1:02d}"
            attempt = root / attempt_rel
            attempt.mkdir(parents=True)
            cell["attempts"].append(attempt_rel)
            cell["status"] = "running"
            manifest["state"] = "running"
            write_json(root / "manifest.json", manifest)
            config = RunConfig(
                max_concurrency=cell["concurrency"],
                warmup=0,
                disable_tqdm=True,
                timeout_s=args.timeout,
            )
            if warmup:
                warm_client = ChatClient(
                    args.base_url, attempt / "warmup", sample_rate=args.sample_rate, api_key=key
                )
                warmed = await BenchmarkRunner(config).run(warmup[0], warm_client.send)
                if any(not r.is_success for r in warmed):
                    raise ValueError("Warmup failed; formal cohort not started")
            client = ChatClient(
                args.base_url, attempt / "outputs", sample_rate=args.sample_rate, api_key=key
            )
            runner = BenchmarkRunner(config)
            await runner.run(datasets[cell["dataset"]][0], client.send)
            result_hashes = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((attempt / "outputs").glob("*.result.json"))
            }
            write_json(
                attempt / "timing.json",
                {
                    "complete": True,
                    "wall_clock_s": runner.wall_clock_s,
                    "result_hashes": result_hashes,
                },
            )
            cell["status"] = "complete"
            write_json(root / "manifest.json", manifest)
        manifest["state"] = "complete"
        manifest.pop("error", None)
    except BaseException as exc:
        manifest["state"] = "interrupted"
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "manifest.json", manifest)
        result = report(root, args.baseline, args.allow_difference)
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        result = asyncio.run(run(args))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        return 130
    if args.dry_run:
        print(dumps(result))
    else:
        print(f"Report: {args.output_dir / 'summary.md'}")
    quality_file = args.output_dir / "quality.json"
    if args.phase in ("score", "report") and quality_file.exists():
        if read_json(quality_file)["status"] != "complete":
            return 1
    if isinstance(result, dict) and any(
        isinstance(v, dict) and (v.get("failed") or v.get("not_attempted")) for v in result.values()
    ):
        return 1
    return 0
