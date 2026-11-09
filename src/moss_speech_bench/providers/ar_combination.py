"""Bounded qualification contract for the six existing AR optimization strategies."""

import hashlib
import json
import re
from pathlib import Path

FEATURES = (
    "ar-decode-graph",
    "ar-projection-graph",
    "context-prefill-graph",
    "native-norm",
    "ar-host-metadata",
    "mixed-modalities",
)
MARKERS = {
    "codec-resident-bf16": "MOSS_FLOW_BF16_RESIDENT_ACTIVE",
    "codec-sdpa-bf16": "MOSS_CODEC_SDPA_BF16_ACTIVE",
    "codec-bias-broadcast": "MOSS_CODEC_BIAS_BROADCAST_ACTIVE",
    "ar-serial-decode-graph": "serial batch decode graph replay active",
    "ar-batched-rope": "batched RoPE reuse active",
    "ar-decode-graph": "full decode graph replay active",
    "ar-projection-graph": "projection graph replay active",
    "context-prefill-graph": "full prefill graph replay active",
    "native-norm": "native norm pointwise execution active",
    "ar-host-metadata": "host attention metadata execution active",
    "mixed-modalities": "executed mixed-output AR batch",
}


def features(config):
    if config.get("profile") == "cumulative":
        return config.get("optimizations", [])
    return (
        config.get("ar_optimizations", [])
        if config.get("profile") == "ar-combination"
        else [config.get("profile")]
    )


def validate(config):
    selected = config.get("ar_optimizations")
    if (
        not isinstance(selected, list)
        or len(selected) < 2
        or any(not isinstance(f, str) for f in selected)
    ):
        raise ValueError("AR combination requires at least two explicit optimizations")
    if len(selected) != len(set(selected)) or not set(selected) <= set(FEATURES):
        raise ValueError("Unknown or duplicate AR combination optimization")
    if config.get("context_remaining") is not True or config.get("test_suite", "legacy") not in (
        "legacy",
        "eager-bugs",
    ):
        raise ValueError("AR combination requires the current remaining-context contract")
    deployment = config.get("deployment", {})
    for key, expected in (
        ("decode_graph", "ar-decode-graph" in selected),
        (
            "ar_graph",
            bool(
                set(selected) & {"ar-decode-graph", "ar-projection-graph", "context-prefill-graph"}
            ),
        ),
        ("use_host_attention_metadata", "ar-host-metadata" in selected),
        ("mixed_modalities", "mixed-modalities" in selected),
    ):
        if deployment.get(key, False) is not expected:
            raise ValueError("AR combination activation mismatch: " + key)
    batch = config.get("projection_graph_batch_size")
    if "ar-projection-graph" in selected:
        if type(batch) is not int or not 1 <= batch <= 4:
            raise ValueError("Projection graph requires batch size1..4")
    elif batch is not None:
        raise ValueError("Unqualified projection graph batch")
    if deployment.get("projection_graph_batch_size") != batch:
        raise ValueError("Projection graph deployment batch mismatch")
    lengths = config.get("graph_lengths", [])
    if "context-prefill-graph" in selected:
        if (
            not isinstance(lengths, list)
            or not 1 <= len(lengths) <= 8
            or any(type(n) is not int or not 1 <= n <= 512 for n in lengths)
            or len(set(lengths)) != len(lengths)
        ):
            raise ValueError("Invalid finite prefill graph lengths")
    elif lengths:
        raise ValueError("Unqualified prefill graph lengths")
    if deployment.get("prefill_graph_lengths", []) != lengths:
        raise ValueError("Prefill graph deployment lengths mismatch")
    library = config.get("native_norm_library")
    if bool(library) != ("native-norm" in selected):
        raise ValueError("Native norm requires its explicit pinned library")
    if bool(deployment.get("native_norm_library_sha256")) != bool(library):
        raise ValueError("Native norm deployment library mismatch")
    if any(
        deployment.get(k)
        for k in (
            "ar_prefix_cache",
            "cache_sampling_history",
            "stream_decode_fairness",
            "cfm_graph_mel_length",
            "small_first_chunk",
            "stream_startup_schedule",
        )
    ) or config.get("direct_lt_library"):
        raise ValueError("AR combination does not qualify additional strategies")
    required = config.get("service_activation_required", selected)
    if (
        not isinstance(required, list)
        or len(required) != len(set(required))
        or not set(required) <= set(selected)
        or set(selected) - set(required) - {"context-prefill-graph"}
    ):
        raise ValueError("AR combination cannot omit required service activation")
    if (
        not isinstance(config.get("service_activation_log"), str)
        or not config["service_activation_log"]
    ):
        raise ValueError("AR combination requires a fresh service activation log")
    offset = config.get("service_activation_offset", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid service activation offset")


def validate_qualification(report, selected):
    assert report["complete_heads_exact"] is True and report["all_kv_exact"] is True
    assert set(report["optimizations"]) == set(selected), "Incomplete combination qualification"
    for feature in selected:
        proof = report["optimizations"][feature]
        assert type(proof["executions"]) is int and proof["executions"] > 0, feature
        assert proof["complete_heads_exact"] is True and proof["all_kv_exact"] is True, feature
        if feature in ("ar-decode-graph", "ar-projection-graph", "context-prefill-graph"):
            assert all(
                type(proof[key]) is int and proof[key] > 0 for key in ("captures", "replays")
            ), feature


def validate_activation(config, output, require=True):
    source = Path(config["service_activation_log"])
    offset = config.get("service_activation_offset", 0)
    with source.open("rb") as handle:
        assert source.stat().st_size >= offset, "Service activation log was truncated"
        handle.seek(offset)
        payload = handle.read()
    text = payload.decode("utf-8", errors="replace")
    selected = [feature for feature in features(config) if feature in MARKERS]
    required = config.get("service_activation_required", selected)
    counts = {feature: text.count(MARKERS[feature]) for feature in selected}
    native_graph_replays = sum(
        any(MARKERS[feature] in line for feature in selected if feature.endswith("graph"))
        and all(
            (match := re.search(r"\b" + name + r"=(\d+)\b", line)) and int(match[1]) > 0
            for name in ("native_norm_capture_calls", "native_owners")
        )
        for line in text.splitlines()
    )
    native_eager_requests = text.count("native norm eager request execution active")
    observed = dict(counts)
    if "native-norm" in selected:
        observed["native-norm"] = native_graph_replays + native_eager_requests
    serial_batches = {}
    if "ar-serial-decode-graph" in selected:
        marker = MARKERS["ar-serial-decode-graph"]
        serial_batches = {
            str(batch): sum(
                marker in line and bool(re.search(r"\bbatch=" + str(batch) + r"\b", line))
                for line in text.splitlines()
            )
            for batch in (2, 4)
        }
        observed["ar-serial-decode-graph"] = min(serial_batches.values())
    missing = [feature for feature in required if not observed[feature]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "service-activation.log").write_bytes(payload)
    report = dict(
        status="pass" if not missing else "incomplete",
        missing=missing,
        source=str(source),
        offset=offset,
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        serial_decode_batches=serial_batches,
        native_serving_evidence=dict(
            graph_replays=native_graph_replays, eager_requests=native_eager_requests
        ),
        features={
            feature: dict(
                marker=MARKERS[feature],
                count=counts[feature],
                qualifying_count=observed[feature],
                status="observed" if observed[feature] else "not_observed",
            )
            for feature in selected
        },
    )

    (output / "service-activation.json").write_text(json.dumps(report, indent=2) + "\n")
    if require:
        assert not missing, "Missing actual service activation: " + ", ".join(missing)
    return report


def main():
    import argparse

    from moss_speech_bench.correctness import load_suite

    parser = argparse.ArgumentParser(
        description="Audit actual AR activation after all serving workloads"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    config = load_suite(arguments.config)
    if config["profile"] not in (
        "cumulative",
        "ar-combination",
        "ar-batched-rope",
        "ar-serial-decode-graph",
        "codec-bias-broadcast",
    ):
        parser.error("Activation audit requires an explicit qualified AR configuration")
    validate_activation(config, arguments.output)


if __name__ == "__main__":
    main()
