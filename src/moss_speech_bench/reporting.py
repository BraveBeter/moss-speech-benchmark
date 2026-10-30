"""Rebuild performance reports from durable observations; no service required."""

import hashlib
from dataclasses import fields
from pathlib import Path

from ._vendor.metrics.performance import compute_speed_metrics
from .client import Observation
from .storage import digest, dumps, read_json, read_jsonl, write_json

METRICS = (
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "latency_p99_s",
    "audio_ttfp_mean_s",
    "audio_ttfp_median_s",
    "audio_ttfp_p95_s",
    "audio_ttfp_p99_s",
    "text_ttft_mean_s",
    "text_ttft_median_s",
    "text_ttft_p95_s",
    "text_ttft_p99_s",
    "rtf_mean",
    "rtf_median",
    "rtf_p95",
    "rtf_p99",
    "throughput_qps",
    "audio_throughput_s_per_s",
    "output_throughput",
    "c50",
    "c100",
    "c200",
)


def summarize(rows, expected, wall_s):
    observations = [
        Observation(**{f.name: row[f.name] for f in fields(Observation) if f.name in row})
        for row in rows
    ]
    summary = compute_speed_metrics(observations, wall_clock_s=wall_s)
    succeeded = sum(r.is_success for r in observations)
    summary.update(
        expected=expected,
        attempted=len(rows),
        succeeded=succeeded,
        failed=len(rows) - succeeded,
        not_attempted=expected - len(rows),
        scored=0,
        failure_rate=(len(rows) - succeeded) / len(rows) if rows else None,
        coverage=len(rows) / expected if expected else None,
        wall_clock_s=wall_s,
        length_count=sum(r.finish_reason == "length" for r in observations),
        ttfa_requests=sum(r.is_success and r.audio_ttfp_s is not None for r in observations),
        ttft_requests=sum(r.is_success and r.text_ttft_s is not None for r in observations),
        rtf_requests=sum(r.is_success and r.audio_duration_s > 0 for r in observations),
        quality_status="not_scored",
        peak_gpu_memory=None,
        metric_version="omni-f96b10-moss-client-v1",
    )
    summary["throughput_qps"] = succeeded / wall_s if wall_s and wall_s > 0 else None
    for key in METRICS:
        summary.setdefault(key, None)
    # Unknown usage is not zero tokens. Aggregate only if all successful rows expose counts.
    complete_usage = bool(succeeded) and all(
        type(r.usage.get("completion_tokens")) is int and r.usage["completion_tokens"] >= 0
        for r in observations
        if r.is_success
    )
    summary["usage_counted_requests"] = sum(
        "completion_tokens" in r.usage for r in observations if r.is_success
    )
    if not complete_usage:
        for key in (
            "output_throughput",
            "output_tokens_total",
            "output_tokens_mean",
            "output_tok_per_req_s",
        ):
            summary[key] = None
    else:
        summary["output_tokens_total"] = sum(
            r.completion_tokens for r in observations if r.is_success
        )
        summary["output_tokens_mean"] = summary["output_tokens_total"] / succeeded
        summary["output_throughput"] = summary["output_tokens_total"] / wall_s if wall_s else None
    summary["ttfa_event"] = "first nonempty PCM SSE event; may contain silence"
    summary["ttfc_event"] = "N/A: native codec boundaries are not exposed"
    return summary


def collect_cell(root, cell):
    expected = cell["expected"]
    if not cell["attempts"]:
        return [], summarize([], expected, None)
    attempt = root / cell["attempts"][-1]
    rows = [read_json(p) for p in sorted((attempt / "outputs").glob("*.result.json"))]
    if len(rows) > expected or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate/excess request artifacts")
    for row in rows:
        for name, expected_hash in row["output_hashes"].items():
            path = attempt / "outputs" / name
            if (
                path.parent != attempt / "outputs"
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
            ):
                raise ValueError(f"Artifact hash mismatch: {path}")
    timing = read_json(attempt / "timing.json") if (attempt / "timing.json").exists() else {}
    if timing.get("complete"):
        actual = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((attempt / "outputs").glob("*.result.json"))
        }
        if actual != timing.get("result_hashes"):
            raise ValueError("Result ledger hash mismatch")
    inputs = {r["id"]: r for r in read_jsonl(root / "inputs" / f"{cell['dataset']}.jsonl")}
    for row in rows:
        if row["sample_id"] not in inputs or row["payload_hash"] != digest(
            inputs[row["sample_id"]]["payload"]
        ):
            raise ValueError("Request does not match frozen input manifest")
    wall = timing.get("wall_clock_s") if timing.get("complete") else None
    summary = summarize(rows, expected, wall)
    summary["measurement_complete"] = wall is not None and len(rows) == expected
    if wall is None:
        # Never fabricate throughput by summing request times after an interrupted run.
        for key in ("throughput_qps", "audio_throughput_s_per_s", "output_throughput"):
            summary[key] = None
    return rows, summary


def _differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d
            for key in sorted(a.keys() | b.keys())
            for d in _differences(a.get(key), b.get(key), f"{prefix}.{key}".strip("."))
        ]
    return [] if a == b else [prefix]


def _load_manifest(root):
    manifest = read_json(root / "manifest.json")
    for name, metadata in manifest["contract"]["datasets"].items():
        rows = read_jsonl(root / "inputs" / f"{name}.jsonl")
        if len(rows) != metadata["expected"] or digest(rows) != metadata["samples_hash"]:
            raise ValueError(f"Input manifest hash mismatch: {name}")
    return manifest


def report(directory, baseline=None, allow_difference=()):
    root = Path(directory)
    manifest = _load_manifest(root)
    from .quality import quality_report

    quality = quality_report(root)
    performance, ledger = {}, []
    for cell in manifest["cells"]:
        rows, summary = collect_cell(root, cell)
        performance[cell["id"]] = summary
        ledger.extend(dict(row, cell_id=cell["id"], attempt=cell["attempts"][-1]) for row in rows)
    (root / "requests.jsonl").write_text("".join(dumps(row) + "\n" for row in ledger))
    write_json(root / "performance.json", performance)
    write_json(
        root / "summary.json",
        {
            "scope": manifest["scope"],
            "state": manifest["state"],
            "quality_status": "not_scored",
            **{
                key: sum(s[key] for s in performance.values())
                for key in (
                    "expected",
                    "attempted",
                    "succeeded",
                    "failed",
                    "not_attempted",
                    "scored",
                )
            },
        },
    )
    lines = [
        "# MOSS-Speech performance",
        "",
        "Quality: see offline quality section when scored. Model correctness: outside client scope.",
        "",
        "TTFA = first nonempty PCM SSE event (audio_ttfp_*); native TTFC and server GPU peak: N/A.",
        "Each row is one continuous dataset/concurrency/repeat; percentiles are not pooled across repeats.",
        "",
        "| Dataset / C / repeat | Expected | Attempted | Succeeded | Failed | Scored | Length | Latency P50 s | TTFA P95 s | RTF mean | QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "N/A" if x is None else f"{x:.5g}" if isinstance(x, float) else str(x)

    for key, metrics in performance.items():
        columns = [
            quality["cells"].get(key, {}).get("scored", 0)
            if k == "scored" and quality
            else metrics.get(k)
            for k in (
                "expected",
                "attempted",
                "succeeded",
                "failed",
                "scored",
                "length_count",
                "latency_median_s",
                "audio_ttfp_p95_s",
                "rtf_mean",
                "throughput_qps",
            )
        ]
        lines.append("| " + key + " | " + " | ".join(map(fmt, columns)) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    if baseline:
        base_root = Path(baseline)
        base = _load_manifest(base_root)
        differences = _differences(base["contract"], manifest["contract"])
        unexpected = [
            d
            for d in differences
            if not any(d == a or d.startswith(a + ".") for a in allow_difference)
        ]
        verified_metadata = bool(base["contract"].get("deployment")) and bool(
            manifest["contract"].get("deployment")
        )
        compatible = not unexpected and verified_metadata
        comparison = {
            "compatible": compatible,
            "differences": differences,
            "unexpected_differences": unexpected,
            "deployment_provided": verified_metadata,
            "allowed_differences": list(allow_difference),
            "cells": {},
        }
        table = [
            "# Baseline comparison",
            "",
            f"Compatible: {compatible}; deployment metadata is operator supplied.",
            f"Differences: {differences}; unexpected: {unexpected}",
            "",
            "| Cell | Metric | Baseline | Current | Delta | Delta % |",
            "|---|---|---:|---:|---:|---:|",
        ]
        base_cells = {cell["id"]: collect_cell(base_root, cell)[1] for cell in base["cells"]}
        if compatible:
            for key, current in performance.items():
                previous = base_cells.get(key, {})
                if not current.get("measurement_complete") or not previous.get(
                    "measurement_complete"
                ):
                    continue
                changes = {}
                for metric in (*METRICS, "failed", "length_count", "succeeded"):
                    x, y = previous.get(metric), current.get(metric)
                    if x is None or y is None:
                        continue
                    delta = y - x
                    percent = delta / x * 100 if x else None
                    changes[metric] = {
                        "baseline": x,
                        "current": y,
                        "delta": delta,
                        "percent": percent,
                    }
                    table.append(
                        f"| {key} | {metric} | {fmt(x)} | {fmt(y)} | {fmt(delta)} | {fmt(percent)} |"
                    )
                comparison["cells"][key] = changes
        write_json(root / "comparison.json", comparison)
        (root / "comparison.md").write_text("\n".join(table) + "\n")
    if quality:
        summary_path = root / "summary.json"
        combined = read_json(summary_path)
        combined["quality_status"] = quality["status"]
        combined["quality_expected"] = sum(
            next(iter(c["metrics"].values()))["expected"] for c in quality["cells"].values()
        )
        combined["scored"] = sum(c["scored"] for c in quality["cells"].values())
        write_json(summary_path, combined)
        with (root / "summary.md").open("a") as output:
            output.write("\n" + (root / "quality.md").read_text())
        if baseline and (Path(baseline) / "quality.json").exists():
            previous = quality_report(Path(baseline))
            differences = _differences(previous["contract"], quality["contract"])
            comparison = read_json(root / "comparison.json")
            comparison["quality"] = {
                "compatible": comparison["compatible"] and not differences,
                "differences": differences,
                "cells": {},
            }
            if comparison["quality"]["compatible"]:
                for name, cell in quality["cells"].items():
                    changes = {}
                    for metric, current in cell["metrics"].items():
                        old = previous["cells"].get(name, {}).get("metrics", {}).get(metric, {})
                        if (
                            current["scored"] == current["expected"]
                            and old.get("scored") == current["scored"]
                            and current["value"] is not None
                            and old.get("value") is not None
                        ):
                            changes[metric] = {
                                "baseline": old["value"],
                                "current": current["value"],
                                "delta": current["value"] - old["value"],
                            }
                    comparison["quality"]["cells"][name] = changes
            write_json(root / "comparison.json", comparison)
            with (root / "comparison.md").open("a") as output:
                output.write("\nOffline quality comparison: " + dumps(comparison["quality"]) + "\n")
    return performance
