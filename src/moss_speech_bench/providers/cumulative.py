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
            "codec-score-release",
            "direct-lt",
            "context-startup-schedule",
        )
    )
)
COMPUTE_AR_FEATURES = (*AR_FEATURES, "ar-serial-decode-graph", "ar-batched-rope")
STREAMING_CODEC_FEATURES = (
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


def validate_codec_qualification(report, selected, reference):
    expected = set(selected) & set(STREAMING_CODEC_FEATURES)
    if len(expected) < 2:
        assert report is None
        return
    assert set(report["optimizations"]) == expected
    assert report["reference"] == reference, "Cumulative codec used another reference"
    assert report["joint_pcm_within_frozen_tolerance"] is True
    assert report["code_ledger_exact"] is True
    assert report["request_state_isolation_exact"] is True
    assert report["rng_isolation_exact"] is True
    assert report["cleanup_exact"] is True
    for feature in expected:
        assert type(report["executions"][feature]) is int and report["executions"][feature] > 0


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
                raise ValueError("Conflicting cumulative regression replacement: " + old)
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
        if node not in expected:
            expected.append(node)
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
