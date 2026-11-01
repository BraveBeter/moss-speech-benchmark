"""Regression contract for explicitly declared, independently qualified AR combinations."""

import json
from pathlib import Path

import pytest

from moss_speech_bench.correctness import load_suite
from moss_speech_bench.providers.ar_combination import (
    FEATURES,
    MARKERS,
    validate_activation,
    validate_qualification,
)
from moss_speech_bench.providers.selection import selected_nodes


def configuration(tmp_path, features=None, **changes):
    features = list(FEATURES) if features is None else features
    config = dict(
        workspace_root=".",
        core_sha="a" * 40,
        profile="ar-combination",
        context_remaining=True,
        ar_optimizations=features,
        projection_graph_batch_size=4,
        graph_lengths=[34, 41],
        native_norm_library="/pinned/native.so",
        service_activation_log="/fresh/server.log",
        deployment=dict(
            ar_graph=True,
            decode_graph=True,
            projection_graph_batch_size=4,
            prefill_graph_lengths=[34, 41],
            mixed_modalities=True,
            use_host_attention_metadata=True,
            native_norm_library_sha256="b" * 64,
        ),
    )
    config.update(changes)
    path = tmp_path / "combination.json"
    path.write_text(json.dumps(config))
    return path


def test_combination_requires_explicit_exact_feature_deployment(tmp_path):
    assert load_suite(configuration(tmp_path))["ar_optimizations"] == list(FEATURES)
    for features in ([], ["unknown"], ["native-norm"], ["native-norm", "native-norm"]):
        with pytest.raises(ValueError):
            load_suite(configuration(tmp_path, features))
    config = json.loads(configuration(tmp_path).read_text())
    config["deployment"]["use_host_attention_metadata"] = False
    with pytest.raises(ValueError, match="host"):
        load_suite(configuration(tmp_path, **config))


def test_combination_does_not_accept_unqualified_library_or_graph(tmp_path):
    with pytest.raises(ValueError, match="library"):
        load_suite(configuration(tmp_path, native_norm_library=None))
    with pytest.raises(ValueError, match="length"):
        load_suite(configuration(tmp_path, graph_lengths=[513]))
    with pytest.raises(ValueError, match="batch"):
        load_suite(configuration(tmp_path, projection_graph_batch_size=True))


def test_combination_selection_retains_all_strategy_cuda_tests():
    combined = selected_nodes("C03", "ar-combination", True, optimizations=list(FEATURES))
    assert len(combined) == len(set(combined))
    for feature in FEATURES:
        nodes = set(selected_nodes("C03", feature, True))
        base = "tests/unit_test/moss_speech/test_model.py::test_native_cuda_prefill_decode_keep_requests_and_tail_kv_independent"
        if base in nodes:
            nodes.remove(base)
            nodes.add(base + "[False]")
        assert nodes <= set(combined)
    assert any("ar_combination" in node for node in combined)


def test_activation_requires_actual_markers_and_preserves_evidence(tmp_path):
    config = load_suite(configuration(tmp_path))
    path = tmp_path / "server.log"
    config["service_activation_log"] = str(path)
    path.write_text("MOSS full decode graph replay active\n")
    with pytest.raises(AssertionError, match="activation"):
        validate_activation(config, tmp_path / "proof")


def test_combination_cannot_silently_skip_activation(tmp_path):
    with pytest.raises(ValueError, match="activation"):
        load_suite(configuration(tmp_path, service_activation_required=[]))


def test_qualification_rejects_eager_fallback_and_missing_feature():
    report = dict(
        complete_heads_exact=True,
        all_kv_exact=True,
        optimizations={
            feature: dict(
                executions=1, complete_heads_exact=True, all_kv_exact=True, captures=1, replays=1
            )
            for feature in FEATURES
        },
    )
    validate_qualification(report, FEATURES)
    report["optimizations"]["ar-projection-graph"]["replays"] = 0
    with pytest.raises(AssertionError):
        validate_qualification(report, FEATURES)
    report["optimizations"].pop("native-norm")
    with pytest.raises(AssertionError, match="Incomplete"):
        validate_qualification(report, FEATURES)


def test_activation_audit_saves_partial_and_complete_observations(tmp_path):
    config = dict(
        profile="ar-combination",
        ar_optimizations=list(FEATURES),
        service_activation_log=str(tmp_path / "server.log"),
    )
    path = tmp_path / "server.log"
    path.write_text(MARKERS[FEATURES[0]])
    partial = validate_activation(config, tmp_path / "partial", require=False)
    assert partial["status"] == "incomplete"
    assert partial["features"]["ar-projection-graph"]["status"] == "not_observed"
    with pytest.raises(AssertionError):
        validate_activation(config, tmp_path / "failed")
    assert (tmp_path / "failed/service-activation.log").read_bytes() == path.read_bytes()
    path.write_text(
        "\n".join(MARKERS.values())
        + "\n"
        + MARKERS[FEATURES[0]]
        + " native_norm_capture_calls=81 native_owners=81"
    )
    assert validate_activation(config, tmp_path / "passed")["status"] == "pass"
    config["service_activation_offset"] = path.stat().st_size
    with pytest.raises(AssertionError):
        validate_activation(config, tmp_path / "old-markers")


def test_valid_bounded_subset_preserves_structural_regressions(tmp_path):
    config = load_suite(
        configuration(
            tmp_path,
            features=["ar-projection-graph", "ar-host-metadata"],
            projection_graph_batch_size=1,
            graph_lengths=[],
            native_norm_library=None,
            deployment=dict(
                ar_graph=True,
                decode_graph=False,
                projection_graph_batch_size=1,
                prefill_graph_lengths=[],
                use_host_attention_metadata=True,
                mixed_modalities=False,
            ),
        )
    )
    assert config["ar_optimizations"] == ["ar-projection-graph", "ar-host-metadata"]
    nodes = selected_nodes("C03", "ar-combination", True, optimizations=config["ar_optimizations"])
    assert any("[False]" in n and "native_cuda_prefill" in n for n in nodes)
    assert not any("native_norm_cuda" in n for n in nodes)


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        " native_norm_capture_calls=0 native_owners=81",
        " native_norm_capture_calls=81 native_owners=0",
    ],
)
def test_native_startup_or_fallback_capture_does_not_prove_serving(tmp_path, suffix):
    config = dict(
        profile="ar-combination",
        ar_optimizations=["native-norm", "ar-decode-graph"],
        service_activation_log=str(tmp_path / "server.log"),
    )
    Path(config["service_activation_log"]).write_text(
        MARKERS["native-norm"] + "\n" + MARKERS["ar-decode-graph"] + suffix
    )
    with pytest.raises(AssertionError, match="native-norm"):
        validate_activation(config, tmp_path / "proof")


def test_native_actual_eager_request_can_qualify_without_graph(tmp_path):
    config = dict(
        profile="ar-combination",
        ar_optimizations=["native-norm", "ar-host-metadata"],
        service_activation_log=str(tmp_path / "server.log"),
    )
    Path(config["service_activation_log"]).write_text(
        MARKERS["ar-host-metadata"]
        + "\nMOSS native norm eager request execution active requests=['owned']"
    )
    assert validate_activation(config, tmp_path / "proof")["status"] == "pass"
