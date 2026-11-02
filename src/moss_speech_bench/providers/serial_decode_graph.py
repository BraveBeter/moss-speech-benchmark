"""Qualify the explicit serial B2/B4 extension, never an untested combination."""

PROFILE = "ar-serial-decode-graph"


def validate(config):
    deployment = config.get("deployment", {})
    if any(
        deployment.get(k) is not True
        for k in ("decode_graph_serial_batch", "decode_graph", "ar_graph")
    ):
        raise ValueError("Serial decode graph requires all explicit activation flags")
    if (
        config.get("context_remaining") is not True
        or config.get("test_suite", "legacy") != "legacy"
    ):
        raise ValueError("Serial decode graph requires the remaining-context contract")
    if any(
        deployment.get(k)
        for k in (
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
        raise ValueError("Serial decode graph does not qualify additional strategies")
    if config.get("service_activation_required", [PROFILE]) != [PROFILE]:
        raise ValueError("Cannot omit serial decode graph serving activation")
    if (
        not isinstance(config.get("service_activation_log"), str)
        or not config["service_activation_log"]
    ):
        raise ValueError("Fresh serving activation log required")
    offset = config.get("service_activation_offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid service activation offset")


def validate_qualification(report):
    assert report["batch_sizes"] == [2, 4]
    assert set(report["executions"]) == {"2", "4"}
    assert all(type(n) is int and n > 0 for n in report["executions"].values())
    assert type(report["captures"]) is int and report["captures"] > 0
    assert report["kv_layers"] == 40
    assert all(
        report[k] is True
        for k in (
            "complete_heads_exact",
            "complete_hidden_exact",
            "no_partial_retry",
            "all_kv_exact",
            "ownership_exact",
            "all_row_rejection_exact",
            "runtime_guards_exact",
            "boundary_fallback_exact",
        )
    )
