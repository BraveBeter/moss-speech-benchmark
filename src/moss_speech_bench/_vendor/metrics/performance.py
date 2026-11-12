# SPDX-License-Identifier: Apache-2.0
"""System performance metrics: latency, RTF, throughput, token throughput,
streaming UX.

Metric semantics:

``throughput_qps``
    Completed requests divided by measured benchmark wall-clock seconds.
``output_tokens_total``
    Sum of completion tokens across successful requests with completion tokens.
``output_tokens_mean``
    Mean completion tokens per successful request with completion tokens.
``output_throughput``
    Completion tokens divided by measured benchmark wall-clock seconds.
``output_tok_per_req_s``
    Completion tokens divided by summed per-request engine/request-time seconds.
``output_token_rate``
    Per-request completion tokens divided by that request's engine/request time.
``rtf_mean``
    Mean request elapsed seconds divided by generated output audio duration seconds.
``rtf_p95`` / ``rtf_p99``
    Tail percentiles of per-request RTF.
``audio_throughput_s_per_s``
    Total seconds of generated audio divided by benchmark wall-clock seconds.
    Independent of per-request audio duration; comparable across engines that
    emit different audio lengths for the same input.
``audio_ttfp_mean_s`` (TTFC)
    Mean time-to-first-audio-chunk: client-side wall time from request send
    to first decoded audio chunk arrival. Streaming only.
``audio_ttfp_median_s`` / ``audio_ttfp_p95_s`` / ``audio_ttfp_p99_s``
    Median / tail percentiles of TTFC.
``text_ttft_mean_s`` (TTFT)
    Mean time-to-first-text-token: client-side wall time from request send to
    first non-empty content delta. Streaming only; populated when the model
    emits text deltas (dual-modality text+audio output).
``text_ttft_median_s`` / ``text_ttft_p95_s`` / ``text_ttft_p99_s``
    Median / tail percentiles of TTFT.
``inter_chunk_mean_s`` (ITL)
    Mean inter-arrival latency between successive audio chunks within a
    request. Streaming smoothness metric.
``inter_chunk_p95_s`` / ``inter_chunk_p99_s``
    Tail percentiles of inter-chunk latency (streaming jitter).
``max_playback_underrun_mean_s`` / ``max_playback_underrun_p95_s`` /
``max_playback_underrun_p99_s``
    Maximum client-side playback starvation for streaming audio requests.
``c50`` / ``c100`` / ``c200``
    Percent of multi-chunk streams whose maximum playback underrun is at most
    50 / 100 / 200 ms. Single-chunk streams are reported as N/A.
``playback_continuity_requests`` / ``playback_continuity_na_requests``
    Scored versus N/A requests behind the ``c*`` gates. All the fields above are
    emitted only for workloads that stream audio chunks.
``audio_chunks_mean``
    Mean number of audio chunks observed per successful streaming request.
    For raw PCM streaming, HTTP chunk boundaries are preserved when available
    rather than counting arbitrary client read frames.
``first_audio_payload_bytes_mean``
    Mean size of the first audio payload. For raw PCM streaming this is the
    first HTTP body chunk; for SSE this is the first decoded PCM payload
    extracted from an audio event.
"""

from __future__ import annotations


import numpy as np
from moss_speech_bench._vendor.data import RequestResult
from moss_speech_bench._vendor.metrics.playback_continuity import summarize_playback_continuity


def _compute_token_metrics(
    successes: list[RequestResult],
    *,
    wall_clock_s: float | None,
) -> dict:
    output_token_counts = [
        o.completion_tokens for o in successes if o.completion_tokens > 0
    ]
    total_tokens = sum(output_token_counts)
    total_engine_time = sum(o.engine_time_s for o in successes if o.engine_time_s > 0)

    prompt_token_counts = [o.prompt_tokens for o in successes if o.prompt_tokens > 0]

    token_metrics: dict = {}
    if total_engine_time > 0 and total_tokens > 0:
        token_metrics["output_tok_per_req_s"] = round(
            total_tokens / total_engine_time,
            1,
        )
    if wall_clock_s is not None and wall_clock_s > 0 and total_tokens > 0:
        token_metrics["output_throughput"] = round(total_tokens / wall_clock_s, 1)
    if output_token_counts:
        token_metrics["output_tokens_mean"] = round(
            float(np.mean(output_token_counts)), 0
        )
        token_metrics["output_tokens_total"] = total_tokens
    if prompt_token_counts:
        token_metrics["prompt_tokens_mean"] = round(
            float(np.mean(prompt_token_counts)), 0
        )
        token_metrics["prompt_tokens_total"] = sum(prompt_token_counts)
    return token_metrics


def compute_speed_metrics(
    outputs: list[RequestResult], wall_clock_s: float | None = None
) -> dict:
    """Compute system performance summary from a list of request results."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {
            "total_requests": len(outputs),
            "completed_requests": 0,
            "failed_requests": len(outputs),
        }

    latencies = [o.latency_s for o in successes]
    rtfs = [o.rtf for o in successes if 0 < o.rtf < float("inf")]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]
    ttfps = [
        o.audio_ttfp_s
        for o in successes
        if getattr(o, "audio_ttfp_s", None) is not None
    ]
    text_ttfts = [
        o.text_ttft_s for o in successes if getattr(o, "text_ttft_s", None) is not None
    ]
    inter_chunk_deltas = [
        d for o in successes for d in getattr(o, "inter_chunk_s", []) or []
    ]
    audio_chunk_counts = [
        o.audio_chunk_count for o in successes if getattr(o, "audio_chunk_count", 0) > 0
    ]
    first_payload_bytes = [
        o.first_audio_payload_bytes
        for o in successes
        if getattr(o, "first_audio_payload_bytes", 0) > 0
    ]
    streaming_audio_underruns = [
        getattr(o, "max_playback_underrun_s", None)
        for o in successes
        if getattr(o, "chunk_audio_duration_s", None)
    ]

    if wall_clock_s is not None and wall_clock_s > 0:
        throughput = round(len(successes) / wall_clock_s, 3)
    else:
        total_latency = sum(latencies)
        throughput = (
            round(len(successes) / total_latency, 3) if total_latency > 0 else 0
        )

    metrics_summary: dict = {
        "total_requests": len(outputs),
        "completed_requests": len(successes),
        "failed_requests": len(outputs) - len(successes),
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_s": round(float(np.percentile(latencies, 99)), 3),
        "audio_duration_mean_s": (
            round(float(np.mean(audio_durations)), 3) if audio_durations else 0
        ),
        "rtf_mean": round(float(np.mean(rtfs)), 4) if rtfs else None,
        "rtf_median": round(float(np.median(rtfs)), 4) if rtfs else None,
        "rtf_p95": round(float(np.percentile(rtfs, 95)), 4) if rtfs else None,
        "rtf_p99": round(float(np.percentile(rtfs, 99)), 4) if rtfs else None,
        "throughput_qps": throughput,
        **_compute_token_metrics(successes, wall_clock_s=wall_clock_s),
    }
    if audio_durations and wall_clock_s is not None and wall_clock_s > 0:
        total_audio_s = sum(audio_durations)
        metrics_summary["audio_throughput_s_per_s"] = round(
            total_audio_s / wall_clock_s, 3
        )
    if ttfps:
        metrics_summary["audio_ttfp_mean_s"] = round(float(np.mean(ttfps)), 4)
        metrics_summary["audio_ttfp_median_s"] = round(float(np.median(ttfps)), 4)
        metrics_summary["audio_ttfp_p95_s"] = round(float(np.percentile(ttfps, 95)), 4)
        metrics_summary["audio_ttfp_p99_s"] = round(float(np.percentile(ttfps, 99)), 4)
    if text_ttfts:
        metrics_summary["text_ttft_mean_s"] = round(float(np.mean(text_ttfts)), 4)
        metrics_summary["text_ttft_median_s"] = round(float(np.median(text_ttfts)), 4)
        metrics_summary["text_ttft_p95_s"] = round(
            float(np.percentile(text_ttfts, 95)), 4
        )
        metrics_summary["text_ttft_p99_s"] = round(
            float(np.percentile(text_ttfts, 99)), 4
        )
    if inter_chunk_deltas:
        metrics_summary["inter_chunk_mean_s"] = round(
            float(np.mean(inter_chunk_deltas)), 4
        )
        metrics_summary["inter_chunk_p95_s"] = round(
            float(np.percentile(inter_chunk_deltas, 95)), 4
        )
        metrics_summary["inter_chunk_p99_s"] = round(
            float(np.percentile(inter_chunk_deltas, 99)), 4
        )
    if audio_chunk_counts:
        metrics_summary["audio_chunks_mean"] = round(
            float(np.mean(audio_chunk_counts)), 2
        )
        metrics_summary["audio_chunks_p95"] = round(
            float(np.percentile(audio_chunk_counts, 95)), 2
        )
    if first_payload_bytes:
        metrics_summary["first_audio_payload_bytes_mean"] = round(
            float(np.mean(first_payload_bytes)), 1
        )
        metrics_summary["first_audio_payload_bytes_p95"] = round(
            float(np.percentile(first_payload_bytes, 95)), 1
        )
    if streaming_audio_underruns:
        metrics_summary.update(summarize_playback_continuity(streaming_audio_underruns))
    return metrics_summary
