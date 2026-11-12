"""Rebuild performance reports from durable observations; no service required."""

import hashlib
from dataclasses import fields
from pathlib import Path
from statistics import mean
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
    "max_playback_underrun_max_s",
    "max_playback_underrun_p95_s",
    "throughput_qps",
    "audio_throughput_s_per_s",
    "output_throughput",
    "c50",
    "c100",
    "c200",
)

GAIN_METRICS = {
    "audio_ttfp_p95_s": "TTFA P95 (s)",
    "rtf_p95": "RTF P95",
}


def summarize(rows, expected, wall_s):
    observations = [
        Observation(**{f.name: row[f.name] for f in fields(Observation) if f.name in row})
        for row in rows
    ]
    summary = compute_speed_metrics(observations, wall_clock_s=wall_s)
    underruns = [
        r.max_playback_underrun_s
        for r in observations
        if r.is_success and r.chunk_audio_duration_s and r.max_playback_underrun_s is not None
    ]
    succeeded = sum(r.is_success for r in observations)
    summary.update(
        max_playback_underrun_max_s=round(max(underruns), 4) if underruns else None,
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
        unknown_finish_count=sum(r.is_success and r.finish_reason is None for r in observations),
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


def collect_cell(root, cell, *, warmup=False):
    expected = cell["warmup_expected"] if warmup else cell["expected"]
    if not cell["attempts"]:
        return [], summarize([], expected, None)
    attempt = root / cell["attempts"][-1]
    output = attempt / ("warmup" if warmup else "outputs")
    rows = [read_json(p) for p in sorted(output.glob("*.result.json"))]
    if len(rows) > expected or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate/excess request artifacts")
    for row in rows:
        for name, expected_hash in row["output_hashes"].items():
            path = output / name
            if (
                path.parent != output
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
            ):
                raise ValueError(f"Artifact hash mismatch: {path}")
    timing_path = (output if warmup else attempt) / "timing.json"
    timing = read_json(timing_path) if timing_path.exists() else {}
    if timing.get("complete"):
        actual = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.glob("*.result.json"))
        }
        if actual != timing.get("result_hashes"):
            raise ValueError("Result ledger hash mismatch")
    input_path = (
        root / "warmup-inputs" / f"{cell['id']}.jsonl"
        if warmup
        else root / "inputs" / f"{cell['dataset']}.jsonl"
    )
    inputs = {r["id"]: r for r in read_jsonl(input_path)}
    for row in rows:
        if row["sample_id"] not in inputs or row["payload_hash"] != digest(
            inputs[row["sample_id"]]["payload"]
        ):
            raise ValueError("Request does not match frozen input manifest")
    wall = timing.get("wall_clock_s") if timing.get("complete") else None
    summary = summarize(rows, expected, wall)
    summary["measurement_complete"] = wall is not None and len(rows) == expected
    if not warmup and cell.get("warmup_expected") and summary["measurement_complete"]:
        _, warmed = collect_cell(root, cell, warmup=True)
        if not warmed.get("measurement_complete") or warmed["succeeded"] != warmed["expected"]:
            raise ValueError("Completed cell has no successful warmup")
    if wall is None:
        # Never fabricate throughput by summing request times after an interrupted run.
        for key in ("throughput_qps", "audio_throughput_s_per_s", "output_throughput"):
            summary[key] = None
    return rows, summary


def average_repeats(manifest, performance):
    """Equal-weight means of complete repeats, never pooled requests or partial rounds."""
    groups = {}
    for cell in manifest["cells"]:
        key = f"{cell['dataset']}-c{cell['concurrency']}"
        groups.setdefault(key, []).append(cell)
    result = {}
    repeats = manifest["contract"]["repeats"]
    for key, cells in groups.items():
        values = [performance[cell["id"]] for cell in cells]
        completed = sum(bool(v.get("measurement_complete")) for v in values)
        valid = sum(
            bool(v.get("measurement_complete")) and v["succeeded"] == v["expected"] for v in values
        )
        complete = (
            len(cells) == repeats
            and sorted(c["repeat"] for c in cells) == list(range(1, repeats + 1))
            and valid == repeats
        )
        result[key] = {
            "dataset": cells[0]["dataset"],
            "concurrency": cells[0]["concurrency"],
            "cells": [c["id"] for c in cells],
            "aggregation": "arithmetic-mean-of-repeat-metrics-v1",
            "expected_repeats": repeats,
            "completed_repeats": completed,
            "valid_repeats": valid,
            "measurement_complete": complete,
            **{
                field: sum(v[field] for v in values)
                for field in (
                    "expected",
                    "attempted",
                    "succeeded",
                    "failed",
                    "not_attempted",
                    "length_count",
                )
            },
            **{
                metric: mean(v[metric] for v in values)
                if complete and all(v.get(metric) is not None for v in values)
                else None
                for metric in METRICS
            },
        }
    return result


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
    if manifest.get("schema_version") != 2 or manifest.get("suite") != "e2e":
        raise ValueError(
            "Unsupported run schema; rebuild historical runs with their original client"
        )
    selection = read_json(root / "selection.json")
    if digest(selection) != manifest["contract"]["selection_hash"]:
        raise ValueError("Selection manifest hash mismatch")
    for name, metadata in manifest["contract"]["datasets"].items():
        rows = read_jsonl(root / "inputs" / f"{name}.jsonl")
        chosen = selection["datasets"][metadata["source_dataset"]]
        if metadata["selection"] != chosen or [r["id"] for r in rows] != chosen["ids"]:
            raise ValueError("Dataset differs from frozen selection")
        if len(rows) != metadata["expected"] or digest(rows) != metadata["samples_hash"]:
            raise ValueError(f"Input manifest hash mismatch: {name}")
    for name, metadata in manifest["contract"].get("warmup_inputs", {}).items():
        rows = read_jsonl(root / "warmup-inputs" / f"{name}.jsonl")
        if len(rows) != metadata["expected"] or digest(rows) != metadata["samples_hash"]:
            raise ValueError(f"Warmup input manifest hash mismatch: {name}")
    contract = manifest["contract"]
    expected = {
        f"{name}-c{c}-r{r}"
        for name in contract["datasets"]
        for c in contract["levels"]
        for r in range(1, contract["repeats"] + 1)
    }
    cells = manifest["cells"]
    if len(cells) != len(expected) or {c["id"] for c in cells} != expected:
        raise ValueError("Manifest cell grid differs from the planned workload")
    return manifest


def code_identity_lines(manifest):
    """Display frozen run identities, never the checkout rebuilding the report."""
    deployment = manifest.get("contract", {}).get("deployment") or {}
    identities = {
        "被测服务代码": deployment.get("server") or {},
        "评测客户端代码": manifest.get("client") or {},
    }
    lines = []
    for label, identity in identities.items():
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
    """Label the frozen QA dataset and modality."""
    metadata = manifest["contract"]["datasets"][dataset_id]
    source = metadata["source_dataset"]
    label = str(source).replace("|", r"\|").replace("\n", " ")
    dataset = f"[{label}](inputs/{quote(dataset_id, safe='')}.jsonl)"
    modality = (
        f"{(metadata.get('task') or 'N/A').upper()} / {(metadata.get('language') or 'N/A').upper()}"
    )
    return dataset, modality


def performance_table(manifest, performance, *, averages=False):
    """Render the one compact performance table used by live and archived summaries."""
    columns = (
        ("latency_median_s", "E2E P50 (s)"),
        ("text_ttft_p95_s", "TTFT P95 (s)"),
        *GAIN_METRICS.items(),
        ("audio_ttfp_mean_s", "TTFA Mean (s)"),
        ("rtf_mean", "RTF Mean"),
        ("max_playback_underrun_max_s", "最大欠载 Max (s)"),
        ("max_playback_underrun_p95_s", "最大欠载 P95 (s)"),
        ("c50", "C50 (%)"),
        ("c100", "C100 (%)"),
        ("audio_throughput_s_per_s", "audio-s/s"),
        ("output_throughput", "TPS"),
        ("throughput_qps", "QPS"),
    )
    lines = [
        "| 数据集 | 模态/语言 | 并发 | 轮次 | 成功/全部 | "
        + " | ".join(label + ("（轮均）" if averages else "") for _, label in columns)
        + " |",
        "|---|---|---:|---:|---:|" + "---:|" * len(columns),
    ]
    cells = (
        [
            dict(value, id=key, repeat=f"均值 {value['valid_repeats']}/{value['expected_repeats']}")
            for key, value in performance.items()
        ]
        if averages
        else manifest["cells"]
    )
    for cell in cells:
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


def summary_markdown(root, manifest, performance, quality, identity_lines, notes=()):
    """The summary shared by live runs and offline rebuilds."""
    from .quality import quality_markdown

    lines = [
        "# MOSS-Speech Benchmark",
        "",
        *identity_lines,
        f"E2E inference state: {manifest['state']}; planned repeats: {manifest['contract']['repeats']}.",
        "",
        "## 运行配置",
        "",
        f"并发：{manifest['contract']['levels']}；轮数：{manifest['contract']['repeats']}；预热数/组：{manifest['contract']['warmup_count'] if manifest['contract']['warmup_count'] is not None else '等于并发'}；stream：{manifest['contract']['stream']}。",
        f"生成参数：`{dumps(manifest['contract']['sampling'])}`。",
        f"本次状态：{manifest['state']}；错误：{manifest.get('error') or '无'}。",
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
        "audio-s/s = 成功请求音频总秒数 / 本组墙钟秒数；TPS = 成功请求completion_tokens总数 / 本组墙钟秒数；QPS = 成功请求数 / 本组墙钟秒数。缺少usage或对应观测时记N/A。",
        "E2E为完整响应耗时；TTFT为首个非空文本；TTFA为首个非空PCM（可能含静音）。分位数逐组计算。",
        "RTF Mean与TTFA Mean是本组成功且指标有效请求的算术平均；RTF P95与TTFA P95是这些请求的95分位数。RTF先逐请求计算E2E/实际PCM时长，再求统计量。",
        "收益统计与性能筛选统一使用TTFA P95 (s)和RTF P95；Mean仅保留为诊断信息，缺失P95不回退Mean。",
        "最大欠载是单请求从首块开始立即播放、耗尽已到达音频后等待后续块的最长停顿；Max/P95分别对本组有效成功请求的最大欠载取最大值/95分位数，单位秒，不是块间隔。",
        "C50/C100是最大欠载分别≤50ms/≤100ms的有效成功多块音频请求百分比，范围0–100；失败和缺少多块观测的请求不进分母，非流式/纯文本/单块音频记N/A。",
        "数据集链接指向本run冻结输入；选样身份见 [selection.json](selection.json)。",
    ]
    if manifest["contract"].get("repeat_aggregation") and performance:
        lines.extend(
            [
                "",
                "逐轮均值（相同数据集、任务、并发；每轮等权）：",
                "",
                *performance_table(manifest, average_repeats(manifest, performance), averages=True),
                "",
                "仅全部轮次计时完整且所有请求成功时给出均值；缺轮/失败/超时为 N/A。",
                "P50/P95 均值为各轮分位数的算术平均，不是合并请求的分位数；QPS/audio-s/s 先逐轮计算再平均。",
                "收益采用标为“轮均”的P95列，即各轮P95等权平均；Mean列仅诊断，跨数据集汇总须注明权重。",
                "最大欠载Max/P95及C50/C100轮均也先逐轮计算再等权平均；Max轮均不是全程最坏欠载，C50/C100轮均不是合并请求的通过率；任一轮缺少有效观测则记N/A。",
                "预热在每轮正式计时前完成，不计入性能或质量分母；记录见 [warmup.json](warmup.json)。",
            ]
        )
    quality_lines = quality_markdown(quality, manifest)
    quality_lines[0] = "## 质量测试"
    lines.extend(["", *quality_lines])
    if notes:
        lines.extend(["", *notes])
    return lines


def comparison_contract(manifest):
    """Server implementations may differ; workloads and metric definitions may not."""
    contract = manifest["contract"]
    return {
        key: contract[key]
        for key in (
            "selection_hash",
            "sampling",
            "stream",
            "sample_rate",
            "timeout",
            "levels",
            "repeats",
            "warmup_policy",
            "warmup_count",
            "metric_version",
            "client_source_hash",
        )
    } | {
        "datasets": {
            name: {key: metadata[key] for key in ("source_dataset", "task", "language", "system")}
            | {"reference_sha256": metadata["service"].get("reference_sha256")}
            for name, metadata in contract["datasets"].items()
        }
    }


def report(directory, baseline=None):
    root = Path(directory)
    manifest = _load_manifest(root)
    identity_lines = code_identity_lines(manifest)
    from .quality import quality_report

    quality = quality_report(root)
    performance, ledger = {}, []
    for cell in manifest["cells"]:
        rows, summary = collect_cell(root, cell)
        performance[cell["id"]] = summary
        ledger.extend(dict(row, cell_id=cell["id"], attempt=cell["attempts"][-1]) for row in rows)
    (root / "requests.jsonl").write_text("".join(dumps(row) + "\n" for row in ledger))
    write_json(root / "performance.json", performance)
    if manifest["contract"].get("repeat_aggregation"):
        write_json(root / "performance-averages.json", average_repeats(manifest, performance))
        write_json(
            root / "warmup.json",
            {
                cell["id"]: collect_cell(root, cell, warmup=True)[1]
                for cell in manifest["cells"]
                if cell.get("warmup_expected")
            },
        )
    write_json(
        root / "summary.json",
        {
            "suite": "e2e",
            "state": manifest["state"],
            "quality_status": "not_scored",
            "suite_status": "complete"
            if manifest["state"] == "complete"
            and all(
                s["measurement_complete"] and s["succeeded"] == s["expected"]
                for s in performance.values()
            )
            else "incomplete",
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
        differences = _differences(comparison_contract(base), comparison_contract(manifest))
        verified_metadata = all(
            (m["contract"].get("deployment") or {}).get("server") for m in (base, manifest)
        )
        compatible = not differences and bool(verified_metadata)
        comparison = dict(
            compatible=compatible,
            differences=differences,
            deployment_provided=bool(verified_metadata),
            gain_metrics=list(GAIN_METRICS),
            cells={},
            baseline_identity=base["contract"].get("deployment"),
            current_identity=manifest["contract"].get("deployment"),
        )
        table = [
            "# Baseline comparison",
            "",
            f"Compatible: {compatible}; deployment metadata is operator supplied.",
            f"Workload differences: {differences}",
            "收益仅按TTFA P95 (s)与RTF P95比较；*-mean行为各轮P95等权轮均，Mean及其他指标仅存comparison.json作诊断。缺失或不可比的P95不计算收益，不回退Mean。",
            "",
            "| Cell | Metric | Baseline | Current | Delta | Delta % |",
            "|---|---|---:|---:|---:|---:|",
        ]
        base_cells = {cell["id"]: collect_cell(base_root, cell)[1] for cell in base["cells"]}
        current_cells = dict(performance)
        if manifest["contract"].get("repeat_aggregation") and base["contract"].get(
            "repeat_aggregation"
        ):
            current_cells.update(
                {
                    key + "-mean": value
                    for key, value in average_repeats(manifest, performance).items()
                }
            )
            base_cells.update(
                {key + "-mean": value for key, value in average_repeats(base, base_cells).items()}
            )
        if compatible:
            for key, current in current_cells.items():
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
                    if metric in GAIN_METRICS:
                        table.append(
                            f"| {key} | {GAIN_METRICS[metric]} | {_format_value(x)} | {_format_value(y)} | {_format_value(delta)} | {_format_value(percent)} |"
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
