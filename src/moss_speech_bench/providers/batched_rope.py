"""Qualification of one default-off B2/B4 decode RoPE strategy, not a combination."""


def validate(config):
    deployment = config.get("deployment", {})
    if deployment.get("decode_rope_reuse") is not True:
        raise ValueError("ar-batched-rope requires enabled decode_rope_reuse")
    if (
        config.get("context_remaining") is not True
        or config.get("test_suite", "legacy") != "legacy"
    ):
        raise ValueError("ar-batched-rope requires the remaining-context contract")
    if any(
        deployment.get(key)
        for key in (
            "ar_graph",
            "decode_graph",
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
        config.get(key)
        for key in (
            "native_norm_library",
            "direct_lt_library",
            "ar_optimizations",
            "graph_lengths",
            "projection_graph_batch_size",
        )
    ):
        raise ValueError("ar-batched-rope does not qualify additional strategies")
    if config.get("service_activation_required", ["ar-batched-rope"]) != ["ar-batched-rope"]:
        raise ValueError("Cannot omit batched RoPE service activation")
    if (
        not isinstance(config.get("service_activation_log"), str)
        or not config["service_activation_log"]
    ):
        raise ValueError("A fresh service activation log is required")
    offset = config.get("service_activation_offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid service activation offset")


def validate_qualification(report, case_names):
    assert report["batch_sizes"] == [2, 4], "Both fixed batch sizes are required"
    assert report["complete_heads_exact"] is True and report["all_kv_exact"] is True
    assert report["kv_layers"] == 40
    assert set(report["executions"]) == {"2", "4"}
    assert set(report["cases"]) == {"2", "4"}
    assert len(case_names) == 8 and len(set(case_names)) == 8
    for batch in ("2", "4"):
        assert type(report["executions"][batch]) is int and report["executions"][batch] > 0
        assert set(report["cases"][batch]) == set(case_names), "Missing fixed HF case"
        for heads in report["cases"][batch].values():
            assert set(heads) == {"text", "audio"}
            for comparison in heads.values():
                assert all(
                    comparison[key] is True for key in ("finite", "allclose", "top1_equal")
                ), "Batched HF head mismatch"
