"""Rebuild performance reports from durable observations; no service required."""

import hashlib
from dataclasses import fields
from pathlib import Path
from urllib.parse import quote

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
    if manifest["contract"].get("suite_version"):
        from .correctness import read_correctness

        proof = manifest.get("correctness", {})
        if not isinstance(proof, dict) or proof.get("status") != "pass":
            raise ValueError("Speech-QA suite has no passed correctness proof")
        if proof.get("report") != "correctness/correctness.json":
            raise ValueError("Invalid correctness report path")
        if hashlib.sha256((root / proof["report"]).read_bytes()).hexdigest() != proof["sha256"]:
            raise ValueError("Correctness proof hash mismatch")
        if read_correctness(root / "correctness")["status"] != "pass":
            raise ValueError("Correctness gate is incomplete")
    for name, metadata in manifest["contract"]["datasets"].items():
        rows = read_jsonl(root / "inputs" / f"{name}.jsonl")
        if len(rows) != metadata["expected"] or digest(rows) != metadata["samples_hash"]:
            raise ValueError(f"Input manifest hash mismatch: {name}")
    return manifest


def code_identity_lines(manifest, supplement=None):
    """Display frozen run identities, never the checkout rebuilding the report."""
    supplement = supplement or {}
    deployment = manifest.get("contract", {}).get("deployment") or {}
    identities = {
        "被测服务代码": deployment.get("server") or {},
        "评测客户端代码": manifest.get("client") or {},
    }
    lines = []
    for label, key in (("被测服务代码", "server"), ("评测客户端代码", "client")):
        identity = dict(identities[label])
        for field, value in supplement.get(key, {}).items():
            if identity.get(field) is not None and identity[field] != value:
                raise ValueError(f"Conflicting saved code identity: {key}.{field}")
            identity[field] = value
        if key == "client":
            for field, value in (deployment.get("client_build") or {}).items():
                if identity.get(field) is None:
                    identity[field] = value
        branch = identity.get("branch") or (
            "detached HEAD" if identity.get("branch") == "" else "未记录"
        )
        dirty = identity.get("dirty")
        state = "未记录" if dirty is None else "有未提交修改" if dirty else "干净"
        line = f"{label}：分支 `{branch}`；commit `{identity.get('commit') or '未记录'}`；工作区：{state}。"
        if not identity.get("commit") and identity.get("source_hash"):
            line += f" source_hash `{identity['source_hash']}`。"
        lines.extend([line, ""])
    return lines


def _format_value(value):
    return "N/A" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)


def dataset_columns(manifest, dataset_id):
    """Use frozen dataset metadata for labels, with an explicit local-sample fallback."""
    metadata = manifest["contract"]["datasets"][dataset_id]
    kind = metadata.get("kind") or "unrecorded"
    source = (
        metadata.get("source_dataset")
        or f"{'custom-json' if kind == 'json' else kind}/{dataset_id}"
    )
    label = str(source).replace("|", r"\|").replace("\n", " ")
    dataset = f"[{label}](inputs/{quote(dataset_id, safe='')}.jsonl)"
    modality = (
        f"{(metadata.get('task') or 'N/A').upper()} / {(metadata.get('language') or 'N/A').upper()}"
    )
    return dataset, modality


def performance_table(manifest, performance):
    """Render the one compact performance table used by live and archived summaries."""
    columns = (
        ("latency_median_s", "E2E P50 (s)"),
        ("text_ttft_p95_s", "TTFT P95 (s)"),
        ("audio_ttfp_p95_s", "TTFA P95 (s)"),
        ("rtf_mean", "RTF Mean"),
        ("audio_throughput_s_per_s", "audio-s/s"),
        ("throughput_qps", "QPS"),
    )
    lines = [
        "| 数据集 | 模态/语言 | 并发 | 轮次 | 成功/全部 | "
        + " | ".join(label for _, label in columns)
        + " |",
        "|---|---|---:|---:|---:|" + "---:|" * len(columns),
    ]
    for cell in manifest["cells"]:
        dataset, modality = dataset_columns(manifest, cell["dataset"])
        values = performance[cell["id"]]
        prefix = [
            dataset,
            modality,
            _format_value(cell.get("concurrency")),
            _format_value(cell.get("repeat")),
            f"{values['succeeded']}/{values['expected']}",
        ]
        lines.append(
            "| "
            + " | ".join([*prefix, *(_format_value(values.get(field)) for field, _ in columns)])
            + " |"
        )
    return lines


def correctness_section(root, manifest):
    """Display manifest-bound checks; keep an empty table when no proof is bound."""
    root = Path(root)
    proof = manifest.get("correctness")
    header = ["| 检查项 | 通过/全部 | 状态 |", "|---|---:|---|"]
    if not isinstance(proof, dict):
        lines = [
            "## 正确性测试",
            "",
            "状态：未绑定本run的完整正确性证明。外部检查见执行说明。",
            "",
            *header,
        ]
        if (root / "correctness/correctness.json").exists():
            lines.extend(["", "[未绑定的本地报告](correctness/correctness.json)"])
        return lines
    if proof.get("report") != "correctness/correctness.json":
        raise ValueError("Invalid correctness report path")
    path = root / proof["report"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != proof.get("sha256"):
        raise ValueError("Correctness proof hash mismatch")
    from .correctness import read_correctness

    suite = read_correctness(root / "correctness")
    if suite["status"] != proof.get("status"):
        raise ValueError("Correctness proof status mismatch")
    names = (
        "身份与输入",
        "CPU回归",
        "Processor / HF",
        "CUDA回归",
        "AR / HF",
        "Encoder / HF",
        "Decoder / HF",
        "服务协议",
        "并发隔离",
    )
    passed = sum(gate["status"] == "pass" for gate in suite["gates"].values())
    lines = [
        "## 正确性测试",
        "",
        f"状态：{suite['status']}；通过 {passed}/{len(suite['gates'])} 组。",
        "",
        *header,
    ]
    for gate_id, name in zip(sorted(suite["gates"]), names, strict=True):
        gate = suite["gates"][gate_id]
        checks = gate.get("checks", [])
        count = sum(check["status"] == "pass" for check in checks)
        coverage = f"{count}/{len(checks)}" if checks else "N/A"
        lines.append(f"| {gate_id} {name} | {coverage} | {gate['status']} |")
    target = (
        "correctness/correctness.md"
        if (root / "correctness/correctness.md").exists()
        else proof["report"]
    )
    lines.extend(["", f"[完整正确性报告]({target})；通过/全部表示记录的检查项数。"])
    return lines


def summary_markdown(root, manifest, performance, quality, identity_lines, notes=()):
    """The single summary template for inference, offline rebuilds and archive views."""
    from .quality import quality_markdown

    lines = [
        "# MOSS-Speech Benchmark",
        "",
        *identity_lines,
        f"Mode: {manifest['contract']['mode']}; recorded inference state: {manifest['state']}.",
        "",
        *correctness_section(root, manifest),
        "",
        "## 性能测试",
        "",
        *performance_table(manifest, performance),
        "",
        (
            "成功/全部以计划请求数为分母；失败、未执行和length明细见 [performance.json](performance.json)。"
            if performance
            else "性能阶段未执行，暂无性能记录。"
        ),
        "audio-s/s = 成功请求生成音频总秒数 / 本组墙钟秒数；QPS = 成功请求数 / 本组墙钟秒数。无音频或计时不完整记 N/A。",
        "E2E为完整响应耗时；TTFT为首个非空文本；TTFA为首个非空PCM（可能含静音）。分位数逐组计算。",
        "custom-json表示自定义固定样例，数据集链接指向本run冻结输入；原始测量编号保留在performance.json。",
    ]
    for component, item in manifest["contract"].get("deferred_evaluations", {}).items():
        lines.extend(["", f"Deferred: {component}: {item['reason']} Full suite is incomplete."])
    quality_lines = quality_markdown(quality, manifest)
    quality_lines[0] = "## 质量测试"
    lines.extend(["", *quality_lines])
    if notes:
        lines.extend(["", *notes])
    return lines


def report(directory, baseline=None, allow_difference=()):
    root = Path(directory)
    manifest = _load_manifest(root)
    supplement = None
    identity_path = root / "execution" / "records" / "code-identity.json"
    if identity_path.exists():
        supplement = read_json(identity_path)
        if (
            supplement.get("manifest_sha256")
            != hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Supplemental code identity manifest hash mismatch")
    identity_lines = code_identity_lines(manifest, supplement)
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
            "mode": manifest["contract"]["mode"],
            "state": manifest["state"],
            "quality_status": "not_scored",
            "correctness_status": (
                manifest.get("correctness", {}).get("status")
                if isinstance(manifest.get("correctness"), dict)
                else "outside_client_scope"
            ),
            "deferred_evaluations": manifest["contract"].get("deferred_evaluations", {}),
            "suite_status": "incomplete"
            if manifest["contract"].get("deferred_evaluations")
            else manifest["state"],
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
    lines = summary_markdown(root, manifest, performance, quality, identity_lines)
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
                        f"| {key} | {metric} | {_format_value(x)} | {_format_value(y)} | {_format_value(delta)} | {_format_value(percent)} |"
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
