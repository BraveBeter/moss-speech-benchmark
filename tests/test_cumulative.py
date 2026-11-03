"""Cumulative qualification cannot replace independent or specialized checks."""

import copy
import json

import pytest

from moss_speech_bench.correctness import load_suite
from moss_speech_bench.providers.ar_combination import validate_activation
from moss_speech_bench.providers.cumulative import (
    validate_ar_qualification,
    validate_codec_qualification,
)
from moss_speech_bench.providers.selection import selected_nodes


def configuration(tmp_path, **changes):
    config = dict(
        workspace_root=".",
        core_sha="a" * 40,
        profile="cumulative",
        context_remaining=True,
        test_suite="eager-bugs",
        optimizations=["ar-decode-graph", "ar-serial-decode-graph"],
        deployment=dict(ar_graph=True, decode_graph=True, decode_graph_serial_batch=True),
        run_factory=dict(decode_graph=True, decode_graph_serial_batch=True),
        service_activation_log=str(tmp_path / "server.log"),
    )
    config.update(changes)
    path = tmp_path / "cumulative.json"
    path.write_text(json.dumps(config))
    return path


def test_new_eager_and_all_selected_strategy_regressions_remain(tmp_path):
    config = load_suite(configuration(tmp_path))
    for gate in ("C01", "C03"):
        selected = selected_nodes(gate, "cumulative", True, "eager-bugs", config["optimizations"])
        assert len(selected) == len(set(selected))
        assert set(selected_nodes(gate, "ar-serial-decode-graph", True, "eager-bugs")) <= set(
            selected
        )
    cpu = selected_nodes("C01", "cumulative", True, "eager-bugs", config["optimizations"])
    assert sum("length_end_before_first_audio_code" in node for node in cpu) == 2
    assert not any("collat" in node for node in cpu)


@pytest.mark.parametrize(
    "change",
    [
        {"optimizations": []},
        {"optimizations": ["unknown", "native-norm"]},
        {"optimizations": ["ar-decode-graph", "ar-decode-graph"]},
        {"optimizations": ["ar-serial-decode-graph", "codec-score-release"]},
        {"test_suite": "legacy"},
        {"context_remaining": False},
        {"run_factory": dict(decode_graph=True)},
        {
            "run_factory": dict(
                decode_graph=True, decode_graph_serial_batch=True, mixed_modalities=True
            )
        },
        {"deployment": dict(ar_graph=True, decode_graph=True, decode_graph_serial_batch=1)},
        {"service_activation_required": ["ar-decode-graph"]},
        {"service_activation_offset": -1},
    ],
)
def test_cumulative_rejects_undeclared_or_incomplete_paths(tmp_path, change):
    with pytest.raises(ValueError):
        load_suite(configuration(tmp_path, **change))


def test_codec_addition_keeps_the_full_cuda_bias_checks(tmp_path):
    selected = ["ar-decode-graph", "ar-serial-decode-graph", "codec-bias-broadcast"]
    config = load_suite(
        configuration(
            tmp_path,
            optimizations=selected,
            run_vocoder=dict(broadcast_attention_bias=True),
            deployment=dict(
                ar_graph=True,
                decode_graph=True,
                decode_graph_serial_batch=True,
                broadcast_attention_bias=True,
            ),
        )
    )
    cuda = selected_nodes("C03", "cumulative", True, "eager-bugs", config["optimizations"])
    assert set(selected_nodes("C03", "codec-bias-broadcast", True, "eager-bugs")) <= set(cuda)


def test_sampling_return_suite_preserves_both_cuda_rng_draw_orders():
    selected = ["ar-decode-graph", "ar-serial-decode-graph", "sampling-returns"]
    cpu = selected_nodes("C01", "cumulative", True, "eager-bugs", selected)
    cuda = selected_nodes("C03", "cumulative", True, "eager-bugs", selected)
    assert sum("test_sample_row_uses_text_then_audio_rng_draw_order[" in node for node in cpu) == 4
    assert sum("test_sample_row_uses_text_then_audio_rng_draw_order[" in node for node in cuda) == 4
    assert any("scalar_sampling_fallback" in node for node in cuda)


def test_cumulative_activation_requires_actual_both_serial_batches(tmp_path):
    config = load_suite(configuration(tmp_path))
    path = tmp_path / "server.log"
    path.write_text(
        "MOSS full decode graph replay active\nserial batch decode graph replay active batch=2\n"
    )
    with pytest.raises(AssertionError):
        validate_activation(config, tmp_path / "proof")
    with path.open("a") as output:
        output.write("serial batch decode graph replay active batch=4\n")
    assert validate_activation(config, tmp_path / "proof")["status"] == "pass"


def test_joint_numerical_proof_cannot_be_replaced_by_individual_successes():
    features = ["native-norm", "ar-host-metadata"]
    report = dict(
        complete_heads_exact=True,
        all_kv_exact=True,
        optimizations={
            feature: dict(executions=1, complete_heads_exact=True, all_kv_exact=True)
            for feature in features
        },
        joint_optimizations=features,
        joint_complete_heads_exact=True,
        joint_all_kv_exact=True,
    )
    validate_ar_qualification(report, features)
    for key, value in (("joint_optimizations", features[:1]), ("joint_all_kv_exact", False)):
        changed = copy.deepcopy(report)
        changed[key] = value
        with pytest.raises(AssertionError):
            validate_ar_qualification(changed, features)
    report["optimizations"]["native-norm"]["executions"] = 0
    with pytest.raises(AssertionError):
        validate_ar_qualification(report, features)


def test_codec_joint_proof_keeps_the_same_reference_and_state_invariants():
    selected = ["codec-bias-broadcast", "codec-cfm-graph"]
    reference = {
        "report_sha256": "a" * 64,
        "tensors_sha256": "b" * 64,
        "pcm_atol": 1e-5,
        "pcm_rtol": 1e-5,
    }
    report = dict(
        optimizations=selected,
        reference=reference,
        joint_pcm_within_frozen_tolerance=True,
        code_ledger_exact=True,
        request_state_isolation_exact=True,
        rng_isolation_exact=True,
        cleanup_exact=True,
        executions={feature: 1 for feature in selected},
    )
    validate_codec_qualification(report, selected, reference)
    for field in (
        "joint_pcm_within_frozen_tolerance",
        "code_ledger_exact",
        "rng_isolation_exact",
        "cleanup_exact",
    ):
        changed = dict(report, **{field: False})
        with pytest.raises(AssertionError):
            validate_codec_qualification(changed, selected, reference)
    with pytest.raises(AssertionError):
        validate_codec_qualification(report, selected, dict(reference, report_sha256="c" * 64))
    with pytest.raises(AssertionError):
        validate_codec_qualification(
            dict(report, executions={feature: 0 for feature in selected}), selected, reference
        )


def test_cumulative_ar_combination_keeps_all_specialized_nodes(tmp_path):
    selected = [
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "native-norm",
        "ar-host-metadata",
        "context-prefill-graph",
    ]
    cuda = selected_nodes("C03", "cumulative", True, "eager-bugs", selected)
    assert any("test_cumulative_serial_graph_host_metadata_prefill_exact" in node for node in cuda)
    serial = set(selected_nodes("C03", "ar-serial-decode-graph", True, "eager-bugs"))
    baseline = "tests/unit_test/moss_speech/test_model.py::test_native_cuda_prefill_decode_keep_requests_and_tail_kv_independent"
    serial.remove(baseline)
    serial.add(baseline + "[False]")
    assert serial <= set(cuda)
    assert baseline + "[True]" in cuda
    cpu = selected_nodes("C01", "cumulative", True, "eager-bugs", selected)
    assert sum("length_end_before_first_audio_code" in node for node in cpu) == 2
    assert len(cpu) == len(set(cpu))
