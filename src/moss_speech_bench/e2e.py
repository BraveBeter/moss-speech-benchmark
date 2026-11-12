"""Plan and execute explicit dataset × modality × concurrency × repeat cells."""

import asyncio
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .client import ChatClient
from .datasets import load_workload
from .reporting import _load_manifest, collect_cell, report
from .runner import dispatch
from .storage import digest, dumps, file_hash, read_json, source_identity, write_json


def sample_counts(values):
    counts = {}
    for value in values or []:
        name, separator, count = value.partition("=")
        if not separator or name in counts:
            raise ValueError("--samples requires unique DATASET=COUNT entries")
        counts[name] = int(count)
    return counts


def check_fresh_output(root):
    if root.exists() and any(p.name != "execution" or not p.is_dir() for p in root.iterdir()):
        raise ValueError("Output directory is not empty; use --resume or a fresh directory")


def timing(directory, wall):
    write_json(
        directory / "timing.json",
        dict(
            complete=True,
            wall_clock_s=wall,
            result_hashes={p.name: file_hash(p) for p in sorted(directory.glob("*.result.json"))},
        ),
    )


async def infer(args):
    started = time.monotonic()
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
        raise ValueError("base-url must be an HTTP(S) origin without credentials or a path")
    levels = args.concurrency
    if not levels or len(set(levels)) != len(levels) or any(c <= 0 for c in levels):
        raise ValueError("Concurrency levels must be unique positive integers")
    if args.repeats <= 0 or args.sample_rate <= 0 or not 0 < args.timeout < float("inf"):
        raise ValueError("Repeats, sample rate and timeout must be positive and finite")
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("Warmup must be nonnegative; use 0 to disable it")
    if args.time_budget is not None and not 0 < args.time_budget < float("inf"):
        raise ValueError("Time budget must be positive and finite")
    if args.selection and (args.samples or args.subset_seed is not None):
        raise ValueError("--selection replaces --samples and --subset-seed")
    overrides = {
        key: getattr(args, key)
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "seed",
        )
        if getattr(args, key) is not None
    }
    datasets, selection, sampling = load_workload(
        args.config,
        args.model,
        args.stream,
        datasets=args.datasets,
        tasks=args.tasks,
        counts=sample_counts(args.samples),
        seed=42 if args.subset_seed is None else args.subset_seed,
        selection_path=args.selection,
        sampling_override=overrides,
    )
    cells = [
        dict(
            id=f"{name}-c{c}-r{repeat}",
            dataset=name,
            concurrency=c,
            repeat=repeat,
            expected=len(rows),
            warmup_expected=c if args.warmup is None else args.warmup,
            attempts=[],
            status="pending",
        )
        for name, (rows, _) in datasets.items()
        for c in levels
        for repeat in range(1, args.repeats + 1)
    ]
    warmups = {}
    for cell in cells:
        rows = datasets[cell["dataset"]][0]
        if cell["warmup_expected"]:
            warmups[cell["id"]] = [
                dict(rows[i % len(rows)], id=f"warmup-{i}", source_id=rows[i % len(rows)]["id"])
                for i in range(cell["warmup_expected"])
            ]
    plan = dict(
        suite="e2e",
        cells=cells,
        formal_requests=sum(c["expected"] for c in cells),
        warmup_requests=sum(c["warmup_expected"] for c in cells),
        quality_requests=sum(c["expected"] for c in cells if c["repeat"] == 1),
        selection=selection,
    )
    if args.dry_run:
        return plan
    print(
        f"E2E: {plan['formal_requests']} formal + {plan['warmup_requests']} warmup; "
        f"{plan['quality_requests']} first-round quality samples",
        flush=True,
    )
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not key:
        raise ValueError("API key environment variable is empty")
    deployment = read_json(args.deployment) if args.deployment else None
    if deployment is not None and not isinstance(deployment, dict):
        raise ValueError("Deployment metadata must be an object")
    identity = source_identity()
    contract = dict(
        model=args.model,
        deployment=deployment,
        datasets={k: v[1] for k, v in datasets.items()},
        selection_hash=digest(selection),
        sampling=sampling,
        stream=args.stream,
        sample_rate=args.sample_rate,
        timeout=args.timeout,
        levels=levels,
        repeats=args.repeats,
        warmup_policy="first-selected-cyclic-per-cell-v1",
        warmup_inputs={
            name: dict(expected=len(rows), samples_hash=digest(rows))
            for name, rows in warmups.items()
        },
        warmup_count=args.warmup,
        repeat_aggregation="arithmetic-mean-of-repeat-metrics-v1",
        client_source_hash=identity["source_hash"],
        metric_version="omni-f96b10-moss-client-v1",
        time_budget_s=args.time_budget,
    )
    root = Path(args.output_dir)
    if args.resume:
        manifest = _load_manifest(root)
        if manifest["contract"] != contract or manifest["base_url"] != args.base_url:
            raise ValueError("Resume contract/endpoint mismatch; use a new output directory")
        cells = manifest["cells"]
    else:
        check_fresh_output(root)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "selection.json", selection)
        write_json(root / "plan.json", plan)
        for name, (rows, _) in datasets.items():
            path = root / "inputs" / f"{name}.jsonl"
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(dumps(row) + "\n" for row in rows))
        for name, rows in warmups.items():
            path = root / "warmup-inputs" / f"{name}.jsonl"
            path.parent.mkdir(exist_ok=True)
            path.write_text("".join(dumps(row) + "\n" for row in rows))
        manifest = dict(
            schema_version=2,
            suite="e2e",
            started_at=datetime.now(timezone.utc).isoformat(),
            contract=contract,
            base_url=args.base_url,
            client=identity,
            cells=cells,
            python=platform.python_version(),
            state="prepared",
        )
    manifest["finished_at"] = None
    write_json(root / "manifest.json", manifest)

    async def measured(rows, client, concurrency):
        work = dispatch(rows, client, concurrency, args.timeout)
        if args.time_budget is None:
            return await work
        remaining = args.time_budget - (time.monotonic() - started)
        return await asyncio.wait_for(work, timeout=max(0, remaining))

    try:
        for cell in cells:
            if cell["status"] == "complete":
                if not collect_cell(root, cell)[1]["measurement_complete"]:
                    raise ValueError("Completed cell has incomplete artifacts")
                continue
            attempt_rel = f"cells/{cell['id']}/attempt-{len(cell['attempts']) + 1:02d}"
            attempt = root / attempt_rel
            attempt.mkdir(parents=True)
            cell["attempts"].append(attempt_rel)
            cell["status"] = manifest["state"] = "running"
            write_json(root / "manifest.json", manifest)
            if cell["warmup_expected"]:
                client = ChatClient(
                    args.base_url, attempt / "warmup", sample_rate=args.sample_rate, api_key=key
                )
                results, wall = await measured(warmups[cell["id"]], client, cell["concurrency"])
                timing(attempt / "warmup", wall)
                if any(not r.is_success for r in results):
                    raise ValueError("Warmup failed; formal cohort not started")
            client = ChatClient(
                args.base_url, attempt / "outputs", sample_rate=args.sample_rate, api_key=key
            )
            _, wall = await measured(datasets[cell["dataset"]][0], client, cell["concurrency"])
            write_json(
                attempt / "timing.json",
                dict(
                    complete=True,
                    wall_clock_s=wall,
                    result_hashes={
                        p.name: file_hash(p)
                        for p in sorted((attempt / "outputs").glob("*.result.json"))
                    },
                ),
            )
            cell["status"] = "complete"
            write_json(root / "manifest.json", manifest)
        manifest["state"] = "complete"
        manifest.pop("error", None)
    except asyncio.TimeoutError:
        manifest["state"] = "budget_exhausted"
        manifest["error"] = "Inference time budget exhausted; partial results retained"
    except BaseException as exc:
        manifest["state"] = "interrupted"
        manifest["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        for cell in cells:
            if cell["status"] == "running":
                cell["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["infer_invocation_elapsed_s"] = time.monotonic() - started
        write_json(root / "manifest.json", manifest)
        result = report(root)
    return result
