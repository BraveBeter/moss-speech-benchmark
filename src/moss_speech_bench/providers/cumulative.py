"""Explicit deployment and regression contracts for cumulative optimization stages."""

if __package__:
    from .ar_combination import FEATURES as AR_FEATURES
    from .ar_combination import MARKERS
else:
    from ar_combination import FEATURES as AR_FEATURES
    from ar_combination import MARKERS

MICRO_SUITES = {
    "codec-unready-return": "p4-15",
    "sampling-returns": "p4-16",
    "codec-host-overlap": "p4-17",
    "codec-shared-hift": "p4-18",
}
FLAGS = {
    "decode_graph": "ar-decode-graph",
    "decode_graph_serial_batch": "ar-serial-decode-graph",
    "decode_rope_reuse": "ar-batched-rope",
    "use_host_attention_metadata": "ar-host-metadata",
    "mixed_modalities": "mixed-modalities",
    "ar_prefix_cache": "ar-prefix-cache",
    "cache_sampling_history": "ar-sampling-history",
    "broadcast_attention_bias": "codec-bias-broadcast",
    "stream_decode_fairness": "codec-fairness",
    "small_first_chunk": "context-small-first",
}
FEATURES = tuple(
    dict.fromkeys(
        (
            *AR_FEATURES,
            *FLAGS.values(),
            *MICRO_SUITES,
            "codec-cfm-graph",
            "codec-flow-bf16",
            "codec-sdpa-bf16",
            "codec-resident-bf16",
            "codec-storage-bf16",
            "codec-encoder-bf16",
            "codec-full-flow-bf16",
            "codec-score-release",
            "direct-lt",
            "context-startup-schedule",
        )
    )
)
COMPUTE_AR_FEATURES = (*AR_FEATURES, "ar-serial-decode-graph", "ar-batched-rope")
STREAMING_CODEC_FEATURES = (
    "codec-flow-bf16",
    "codec-sdpa-bf16",
    "codec-resident-bf16",
    "codec-storage-bf16",
    "codec-encoder-bf16",
    "codec-full-flow-bf16",
    "codec-bias-broadcast",
    "codec-cfm-graph",
    "direct-lt",
    "codec-shared-hift",
    "codec-fairness",
    "context-small-first",
    "context-startup-schedule",
    "codec-host-overlap",
    "codec-unready-return",
    "codec-score-release",
)


def validate_codec_qualification(report, selected, reference, precision_contract=None):
    expected = set(selected) & set(STREAMING_CODEC_FEATURES)
    if len(expected) < 2:
        assert report is None
        return
    assert set(report["optimizations"]) == expected
    assert report["reference"] == reference, "Cumulative codec used another reference"
    if {"codec-flow-bf16", "codec-full-flow-bf16"} & expected:
        assert precision_contract == "e13-bf16-approx-v1"
        assert report["precision_contract"] == precision_contract
        assert report["joint_pcm_within_frozen_tolerance"] is False
        assert report["same_precision_pcm_exact"] is True
        assert set(report["cross_precision_statistics"]) == {
            "pcm_max_abs",
            "pcm_relative_l2",
            "pcm_tensor_cosine",
        }
    else:
        assert precision_contract is None
        assert report["joint_pcm_within_frozen_tolerance"] is True
    assert report["code_ledger_exact"] is True
    assert report["request_state_isolation_exact"] is True
    assert report["rng_isolation_exact"] is True
    assert report["cleanup_exact"] is True
    for feature in expected:
        assert type(report["executions"][feature]) is int and report["executions"][feature] > 0
    if "codec-sdpa-bf16" in expected:
        assert report["bf16_sdpa_mask_broadcast"] is True
    if "codec-resident-bf16" in expected:
        assert report["bf16_weight_residency"] is True
        for field in ("resident_weight_bytes", "resident_parameter_copies"):
            assert type(report[field]) is int and report[field] > 0
    if "codec-storage-bf16" in expected:
        assert report["bf16_parameter_storage"] is True
        assert report["fp32_cpu_backup_bytes"] == 2 * report["resident_weight_bytes"]

    if "codec-encoder-bf16" in expected:
        assert report["bf16_flow_encoder"] is True and report["encoder_fp32_boundaries"] is True
        assert (
            report["encoder_bf16_executions"] > 0
            and report["encoder_bf16_convolution_executions"] > 0
        )

    if "codec-full-flow-bf16" in expected:
        assert report["full_flow_bf16"] is True
        assert report["bf16_solver_state_exact_dtype"] is True
        assert report["bf16_sdpa_mask_broadcast"] is True
        for field in ("bf16_solver_steps", "full_flow_bf16_parameter_bytes"):
            assert type(report[field]) is int and report[field] > 0
        assert report["full_flow_cpu_backup_bytes"] == 2 * report["full_flow_bf16_parameter_bytes"]


def validate_ar_qualification(report, selected):
    expected = set(selected) & set(COMPUTE_AR_FEATURES)
    if not expected:
        assert report is None
        return
    assert report["complete_heads_exact"] is True and report["all_kv_exact"] is True
    assert set(report["optimizations"]) == expected
    assert set(report["joint_optimizations"]) == expected
    assert report["joint_complete_heads_exact"] is True
    assert report["joint_all_kv_exact"] is True
    for feature in expected:
        proof = report["optimizations"][feature]
        assert type(proof["executions"]) is int and proof["executions"] > 0, feature
        assert proof["complete_heads_exact"] is True and proof["all_kv_exact"] is True, feature
    if "ar-serial-decode-graph" in expected:
        if __package__:
            from .serial_decode_graph import validate_qualification
        else:
            from serial_decode_graph import validate_qualification
        validate_qualification(report["serial_decode_graph"])


def validate(config):
    if config.get("cumulative_api") not in (None, "integrated-ar", "independent"):
        raise ValueError("Unknown cumulative source API")
    selected = config.get("optimizations")
    if (
        not isinstance(selected, list)
        or len(selected) < 2
        or not all(isinstance(feature, str) and feature in FEATURES for feature in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("Cumulative profile requires explicit unique known optimizations")
    if config.get("context_remaining") is not True or config.get("test_suite") != "eager-bugs":
        raise ValueError("Cumulative profile requires the new eager regression contract")
    deployment = config.get("deployment", {})
    for flag, feature in FLAGS.items():
        if deployment.get(flag, False) is not (feature in selected):
            raise ValueError("Cumulative activation mismatch: " + flag)
        factory = (
            config.get("run_vocoder", {})
            if flag in ("broadcast_attention_bias", "stream_decode_fairness", "small_first_chunk")
            else config.get("run_factory", {})
        )
        option = "prefix_cache" if flag == "ar_prefix_cache" else flag
        if factory.get(option, False) is not (feature in selected):
            raise ValueError("Cumulative serving factory mismatch: " + option)
    if "ar-serial-decode-graph" in selected and "ar-decode-graph" not in selected:
        raise ValueError("Serial decode requires its B1 graph dependency")
    graph_enabled = bool(
        set(selected) & {"ar-decode-graph", "ar-projection-graph", "context-prefill-graph"}
    )
    if deployment.get("ar_graph", False) is not graph_enabled:
        raise ValueError("Cumulative graph deployment mismatch")
    lengths = config.get("graph_lengths", [])
    if "context-prefill-graph" in selected:
        if (
            not isinstance(lengths, list)
            or not 1 <= len(lengths) <= 8
            or len(set(lengths)) != len(lengths)
            or any(type(value) is not int or not 1 <= value <= 512 for value in lengths)
        ):
            raise ValueError("Invalid cumulative prefill lengths")
    elif lengths:
        raise ValueError("Undeclared cumulative prefill lengths")
    if deployment.get("prefill_graph_lengths", []) != lengths:
        raise ValueError("Cumulative prefill deployment mismatch")
    batch = config.get("projection_graph_batch_size")
    if "ar-projection-graph" in selected:
        if type(batch) is not int or not 1 <= batch <= 4:
            raise ValueError("Invalid cumulative projection batch")
    elif batch is not None:
        raise ValueError("Undeclared cumulative projection graph")
    if deployment.get("projection_graph_batch_size") != batch:
        raise ValueError("Cumulative projection deployment mismatch")
    for feature, option, identity in (
        ("native-norm", "native_norm_library", "native_norm_library_sha256"),
        ("direct-lt", "direct_lt_library", "direct_lt_library_sha256"),
    ):
        if bool(config.get(option)) != (feature in selected) or bool(deployment.get(identity)) != (
            feature in selected
        ):
            raise ValueError("Cumulative pinned library mismatch: " + feature)
    length = deployment.get("cfm_graph_mel_length")
    if "codec-cfm-graph" in selected:
        if type(length) is not int or not 100 <= length <= 4000 or length % 100:
            raise ValueError("Invalid cumulative CFM graph length")
    elif length is not None:
        raise ValueError("Undeclared cumulative CFM graph")
    precision = deployment.get("flow_precision")
    if {"codec-flow-bf16", "codec-full-flow-bf16"} & set(selected):
        if (
            config.get("precision_contract") != "e13-bf16-approx-v1"
            or precision
            not in (
                "ffn",
                "attention",
                "ffn_attention",
                "ffn_attention_sdpa",
                "ffn_attention_sdpa_resident",
                "ffn_attention_sdpa_storage",
                "ffn_attention_sdpa_storage_encoder",
                "full_flow_bf16",
            )
            or config.get("run_vocoder", {}).get("flow_precision") != precision
            or "codec-cfm-graph" in selected
            or not {"codec-bias-broadcast", "codec-unready-return"} <= set(selected)
        ):
            raise ValueError("Invalid BF16 approximate-inference contract or activation")
    elif (
        precision is not None
        or config.get("run_vocoder", {}).get("flow_precision") is not None
        or config.get("precision_contract") is not None
    ):
        raise ValueError("Undeclared BF16 Flow precision")
    if ("codec-sdpa-bf16" in selected) != (
        precision
        in (
            "ffn_attention_sdpa",
            "ffn_attention_sdpa_resident",
            "ffn_attention_sdpa_storage",
            "ffn_attention_sdpa_storage_encoder",
        )
    ):
        raise ValueError("BF16 SDPA requires its explicit activation and regression coverage")
    if ("codec-resident-bf16" in selected) != (
        precision
        in (
            "ffn_attention_sdpa_resident",
            "ffn_attention_sdpa_storage",
            "ffn_attention_sdpa_storage_encoder",
        )
    ):
        raise ValueError("BF16 resident weights require explicit activation and ownership proof")
    if ("codec-storage-bf16" in selected) != (
        precision in ("ffn_attention_sdpa_storage", "ffn_attention_sdpa_storage_encoder")
    ):
        raise ValueError("BF16 parameter storage requires explicit activation and CPU backup proof")
    if ("codec-encoder-bf16" in selected) != (precision == "ffn_attention_sdpa_storage_encoder"):
        raise ValueError("Flow encoder BF16 requires explicit activation and boundary proof")

    if ("codec-full-flow-bf16" in selected) != (precision == "full_flow_bf16"):
        raise ValueError("Full Flow BF16 requires its explicit solver and storage proof")
    if precision == "full_flow_bf16" and {
        "codec-flow-bf16",
        "codec-sdpa-bf16",
        "codec-resident-bf16",
        "codec-storage-bf16",
        "codec-encoder-bf16",
    } & set(selected):
        raise ValueError("Full Flow and component BF16 profiles are independent")
    schedule = deployment.get("stream_startup_schedule")
    if "context-startup-schedule" in selected:
        if schedule != "6_8_align25" or "context-small-first" in selected:
            raise ValueError("Invalid or conflicting cumulative startup schedule")
    elif schedule is not None:
        raise ValueError("Undeclared cumulative startup schedule")
    expected = set(selected) & set(MARKERS)
    required = config.get("service_activation_required", sorted(expected))
    if (
        not isinstance(required, list)
        or len(required) != len(set(required))
        or not set(required) <= expected
        or expected - set(required) - {"context-prefill-graph"}
    ):
        raise ValueError("Cumulative profile cannot omit required activation")
    if (
        not isinstance(config.get("service_activation_log"), str)
        or not config["service_activation_log"]
    ):
        raise ValueError("Cumulative profile requires a fresh service log")
    offset = config.get("service_activation_offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid cumulative activation offset")


def regression_nodes(selection, gate, selected, expected_tests, cumulative_api=None):
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(FEATURES):
        raise ValueError("Missing, unknown or duplicate cumulative strategies")
    core = [feature for feature in selected if feature in AR_FEATURES]
    integrated = cumulative_api != "independent" and len(core) >= 2
    if integrated:
        expected = expected_tests(selection, gate, "ar-combination", True, "eager-bugs", core)
    elif core and cumulative_api != "independent":
        expected = expected_tests(selection, gate, core[0], True, "eager-bugs")
    else:
        expected = expected_tests(selection, gate, "context-remaining", True, "eager-bugs")
    replacements = {}
    additions = []
    for feature in selected:
        if feature in core and cumulative_api != "independent":
            continue
        if feature in MICRO_SUITES:
            changes = selection["test_suites"][MICRO_SUITES[feature]].get(gate, {})
            patch, nodes = changes.get("replace", {}), changes.get("add", [])
        else:
            name = gate + "-" + feature
            if name not in selection:
                raise ValueError("Missing cumulative strategy qualification: " + name)
            patch, nodes = selection.get(name + "-replacements", {}), selection[name]
        for old, new in patch.items():
            if old in replacements and replacements[old] != new:
                previous = replacements[old]
                previous_nodes = previous if isinstance(previous, list) else [previous]
                current_nodes = new if isinstance(new, list) else [new]
                if set(current_nodes) <= set(previous_nodes):
                    continue
                if not set(previous_nodes) <= set(current_nodes):
                    if not all(
                        node.startswith(old + "[") and node.endswith("]")
                        for node in previous_nodes + current_nodes
                    ):
                        raise ValueError("Conflicting cumulative regression replacement: " + old)
                    new = list(dict.fromkeys(previous_nodes + current_nodes))
            replacements[old] = new
        additions.extend(nodes)
    for old, new in replacements.items():
        if old not in expected:
            if new in expected:
                continue
            raise ValueError("Cumulative strategy replaces an absent regression: " + old)
        index = expected.index(old)
        expected[index : index + 1] = new if isinstance(new, list) else [new]
    for node in additions:
        replacement = replacements.get(node, node)
        nodes = replacement if isinstance(replacement, list) else [replacement]
        for actual in nodes:
            if actual not in expected:
                expected.append(actual)
    if gate == "C03" and integrated:
        expected += selection["C03-cumulative"]
    elif (
        gate == "C03"
        and cumulative_api == "independent"
        and len(set(selected) & set(COMPUTE_AR_FEATURES)) > 2
    ):
        expected += selection["C03-cumulative-independent"]
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate cumulative correctness nodes")
    return expected
