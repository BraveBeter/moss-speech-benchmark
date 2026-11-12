import pytest

from moss_speech_bench._vendor.metrics.playback_continuity import compute_max_playback_underrun_s
from moss_speech_bench.reporting import average_repeats, summarize


def test_p95_rtf_throughput_continuity_and_missing_usage():
    rows = [
        dict(
            is_success=True,
            latency_s=elapsed,
            audio_duration_s=duration,
            rtf=elapsed / duration,
            audio_ttfp_s=ttfa,
            chunk_audio_duration_s=[duration / 2] * 2,
            max_playback_underrun_s=underrun,
            completion_tokens=tokens,
            usage=dict(completion_tokens=tokens),
        )
        for elapsed, duration, ttfa, underrun, tokens in [
            (1, 2, 0.2, 0.05, 10),
            (1, 4, 0.8, 0.1, 20),
        ]
    ]
    rows.append(dict(is_success=False, latency_s=99, audio_duration_s=99, rtf=99, audio_ttfp_s=99))
    metric = summarize(rows, 4, 5)
    assert metric["rtf_p95"] == 0.4875 and metric["rtf_mean"] == 0.375
    assert metric["audio_ttfp_p95_s"] == 0.77 and metric["audio_ttfp_mean_s"] == 0.5
    assert metric["max_playback_underrun_max_s"] == 0.1
    assert metric["max_playback_underrun_p95_s"] == 0.0975
    assert metric["c50"] == 50 and metric["c100"] == 100
    assert metric["audio_throughput_s_per_s"] == 1.2 and metric["output_throughput"] == 6
    assert metric["expected"] == 4 and metric["not_attempted"] == metric["failed"] == 1
    rows[0]["usage"] = {}
    assert summarize(rows, 4, 5)["output_throughput"] is None
    assert summarize(rows, 4, None)["audio_throughput_s_per_s"] is None


def test_repeat_aggregation_is_equal_weight_p95_and_requires_all_successful_rounds():
    cells = [
        dict(id=f"cell{i}", dataset="llama_questions-t2s", concurrency=4, repeat=i) for i in (1, 2)
    ]
    manifest = dict(cells=cells, contract=dict(repeats=2))
    performance = {
        cell["id"]: dict(
            summarize([dict(is_success=True, latency_s=i, audio_duration_s=1, rtf=i)], 1, i),
            measurement_complete=True,
        )
        for i, cell in enumerate(cells, 1)
    }
    averaged = next(iter(average_repeats(manifest, performance).values()))
    assert averaged["rtf_p95"] == 1.5
    assert averaged["audio_throughput_s_per_s"] == 0.75
    performance["cell2"]["succeeded"] = 0
    averaged = next(iter(average_repeats(manifest, performance).values()))
    assert averaged["rtf_p95"] is None and not averaged["measurement_complete"]


def test_underrun_is_playback_starvation_not_interarrival_time():
    assert compute_max_playback_underrun_s([1, 1.2, 2], [0.5, 0.2, 1]) == pytest.approx(0.3)
    assert compute_max_playback_underrun_s([1], [0.5]) is None


def test_all_failed_and_text_metrics_have_no_audio_observations():
    failed = summarize([dict(is_success=False, latency_s=2)], 3, 5)
    assert failed["not_attempted"] == 2 and failed["failure_rate"] == 1
    assert failed["throughput_qps"] == 0 and failed["rtf_p95"] is None
    text = summarize([dict(is_success=True, latency_s=1, text_ttft_s=0.1)], 1, 2)
    assert text["text_ttft_p95_s"] == 0.1
    assert all(text[key] is None for key in ("rtf_p95", "audio_ttfp_p95_s", "c50", "c100"))
