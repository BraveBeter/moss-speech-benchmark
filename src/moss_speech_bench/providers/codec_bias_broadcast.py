"""Explicit live-bias broadcast qualification; no mask caching or new backend."""

PROFILE = "codec-bias-broadcast"


def validate(config):
    deployment = config.get("deployment", {})
    if deployment.get("broadcast_attention_bias") is not True:
        raise ValueError("Codec bias broadcast requires explicit activation")
    if (
        config.get("context_remaining") is not True
        or config.get("test_suite", "legacy") != "legacy"
    ):
        raise ValueError("Codec bias broadcast requires the remaining-context contract")
    if any(
        deployment.get(k)
        for k in (
            "ar_graph",
            "decode_graph",
            "decode_graph_serial_batch",
            "decode_rope_reuse",
            "projection_graph_batch_size",
            "prefill_graph_lengths",
            "use_host_attention_metadata",
            "mixed_modalities",
            "ar_prefix_cache",
            "cache_sampling_history",
            "stream_decode_fairness",
            "cfm_graph_mel_length",
            "small_first_chunk",
            "stream_startup_schedule",
            "native_norm_library_sha256",
            "direct_lt_library_sha256",
        )
    ) or any(
        config.get(k)
        for k in (
            "native_norm_library",
            "direct_lt_library",
            "ar_optimizations",
            "graph_lengths",
            "projection_graph_batch_size",
        )
    ):
        raise ValueError("Codec bias broadcast does not qualify additional strategies")
    if config.get("service_activation_required", [PROFILE]) != [PROFILE]:
        raise ValueError("Cannot omit Codec bias broadcast serving activation")
    if (
        not isinstance(config.get("service_activation_log"), str)
        or not config["service_activation_log"]
    ):
        raise ValueError("Fresh serving activation log required")
    offset = config.get("service_activation_offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid service activation offset")


def validate_qualification(report):
    assert report["installed"] == 56
    assert report["head_strides"] == [0] * 56
    for key in ("calls", "fallback_calls"):
        values = report[key]
        assert isinstance(values, list) and len(values) == 56
        assert all(type(n) is int and (n > 0 if key == "calls" else n >= 0) for n in values)
