"""Cumulative qualification cannot replace independent or specialized checks."""

import copy
import json
import sys
from pathlib import Path

import pytest

from moss_speech_bench.correctness import load_suite
from moss_speech_bench.providers import selection as selection_module
from moss_speech_bench.providers.ar_combination import main, validate_activation
from moss_speech_bench.providers.cumulative import (
    validate_ar_qualification,
    validate_codec_qualification,
)
from moss_speech_bench.providers.selection import expected_tests, selected_nodes


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


def test_activation_cli_accepts_the_cumulative_profile(tmp_path, monkeypatch):
    config = configuration(tmp_path)
    (tmp_path / "server.log").write_text(
        "MOSS full decode graph replay active\n"
        "serial batch decode graph replay active batch=2\n"
        "serial batch decode graph replay active batch=4\n"
    )
    output = tmp_path / "cli-proof"
    monkeypatch.setattr(
        sys, "argv", ["activation", "--config", str(config), "--output", str(output)]
    )
    main()
    assert json.loads((output / "service-activation.json").read_text())["status"] == "pass"


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


def test_independent_ports_keep_atomic_and_joint_checks_without_integrated_api():
    selected = ["ar-decode-graph", "ar-serial-decode-graph", "native-norm", "codec-bias-broadcast"]
    cuda = selected_nodes("C03", "cumulative", True, "eager-bugs", selected, "independent")
    assert len(cuda) == len(set(cuda))
    for feature in selected:
        assert set(selected_nodes("C03", feature, True, "eager-bugs")) <= set(cuda)
    assert any("test_cumulative_independent_ar_cuda_exact" in node for node in cuda)
    assert not any("test_ar_combination_cuda" in node for node in cuda)
    cpu = selected_nodes("C01", "cumulative", True, "eager-bugs", selected, "independent")
    assert sum("length_end_before_first_audio_code" in node for node in cpu) == 2
    assert any("native_norm" in node for node in cpu)


@pytest.mark.parametrize("reverse", [False, True])
def test_host_and_rope_keep_the_superset_of_shared_factory_cases(reverse):
    strategies = [
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "ar-batched-rope",
        "ar-host-metadata",
    ]
    if reverse:
        strategies.reverse()
    nodes = selected_nodes("C01", "cumulative", True, "eager-bugs", strategies, "independent")
    factory = "tests/unit_test/moss_speech/test_config.py::test_ar_factory_preflights_and_forwards_placement_budget"
    assert nodes.count(factory + "[False]") == 1
    assert nodes.count(factory + "[True]") == 1
    assert len(nodes) == len(set(nodes))
    assert sum("length_end_before_first_audio_code" in node for node in nodes) == 2


def test_incompatible_replacement_destinations_still_fail():
    catalog = json.loads(Path(selection_module.__file__).with_name("selection.json").read_text())
    factory = "tests/unit_test/moss_speech/test_config.py::test_ar_factory_preflights_and_forwards_placement_budget"
    catalog["C01-ar-host-metadata-replacements"][factory] = (
        "tests/unit_test/moss_speech/test_config.py::test_incompatible_factory_contract"
    )
    with pytest.raises(ValueError, match="Conflicting cumulative regression replacement"):
        expected_tests(
            catalog,
            "C01",
            "cumulative",
            True,
            "eager-bugs",
            ["ar-decode-graph", "ar-batched-rope", "ar-host-metadata"],
            "independent",
        )


def test_prefill_full_cuda_list_uses_the_host_baseline_replacement():
    selected = [
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "ar-batched-rope",
        "ar-host-metadata",
        "context-prefill-graph",
    ]
    nodes = selected_nodes("C03", "cumulative", True, "eager-bugs", selected, "independent")
    baseline = "tests/unit_test/moss_speech/test_model.py::test_native_cuda_prefill_decode_keep_requests_and_tail_kv_independent"
    assert baseline not in nodes
    assert nodes.count(baseline + "[False]") == 1
    assert nodes.count(baseline + "[True]") == 1
    for node in selected_nodes("C03", "context-prefill-graph", True, "eager-bugs"):
        assert node == baseline or node in nodes
    cpu = selected_nodes("C01", "cumulative", True, "eager-bugs", selected, "independent")
    assert sum("length_end_before_first_audio_code" in node for node in cpu) == 2


@pytest.mark.parametrize("strategy", ["context-small-first", "context-startup-schedule"])
def test_known_empty_extra_gate_keeps_the_original_cuda_baseline(strategy):
    selected = ["ar-decode-graph", "ar-serial-decode-graph", "codec-bias-broadcast", strategy]
    for gate in ("C01", "C03"):
        nodes = selected_nodes(gate, "cumulative", True, "eager-bugs", selected, "independent")
        assert set(selected_nodes(gate, strategy, True, "eager-bugs")) <= set(nodes)
        assert len(nodes) == len(set(nodes))


@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_parameterizations_of_one_factory_keep_every_fixed_case(reverse):
    strategies = [
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "ar-batched-rope",
        "ar-host-metadata",
        "mixed-modalities",
    ]
    if reverse:
        strategies.reverse()
    nodes = selected_nodes("C01", "cumulative", True, "eager-bugs", strategies, "independent")
    factory = "tests/unit_test/moss_speech/test_config.py::test_ar_factory_preflights_and_forwards_placement_budget"
    for case in ("False", "True", "default", "disabled", "enabled"):
        assert nodes.count(factory + "[" + case + "]") == 1
    assert len(nodes) == len(set(nodes))


@pytest.mark.parametrize(
    "precision",
    [
        "ffn",
        "ffn_attention_sdpa",
        "ffn_attention_sdpa_resident",
        "ffn_attention_sdpa_storage",
        "ffn_attention_sdpa_storage_encoder",
    ],
)
def test_bf16_profile_requires_versioned_contract_and_actual_activation(tmp_path, precision):
    changes = dict(
        optimizations=[
            "ar-decode-graph",
            "ar-serial-decode-graph",
            "codec-bias-broadcast",
            "codec-unready-return",
            "codec-flow-bf16",
        ],
        precision_contract="e13-bf16-approx-v1",
        deployment=dict(
            ar_graph=True,
            decode_graph=True,
            decode_graph_serial_batch=True,
            broadcast_attention_bias=True,
            flow_precision=precision,
        ),
        run_vocoder=dict(broadcast_attention_bias=True, flow_precision=precision),
    )
    if precision in (
        "ffn_attention_sdpa",
        "ffn_attention_sdpa_resident",
        "ffn_attention_sdpa_storage",
        "ffn_attention_sdpa_storage_encoder",
    ):
        changes["optimizations"].append("codec-sdpa-bf16")
    if precision in (
        "ffn_attention_sdpa_resident",
        "ffn_attention_sdpa_storage",
        "ffn_attention_sdpa_storage_encoder",
    ):
        changes["optimizations"].append("codec-resident-bf16")
    if precision in ("ffn_attention_sdpa_storage", "ffn_attention_sdpa_storage_encoder"):
        changes["optimizations"].append("codec-storage-bf16")
    if precision == "ffn_attention_sdpa_storage_encoder":
        changes["optimizations"].append("codec-encoder-bf16")
    load_suite(configuration(tmp_path, **changes))
    for patch in [
        dict(precision_contract=None),
        dict(precision_contract="unknown"),
        dict(run_vocoder=dict(broadcast_attention_bias=True)),
        dict(optimizations=changes["optimizations"] + ["codec-cfm-graph"]),
        dict(
            optimizations=[
                feature for feature in changes["optimizations"] if feature != "codec-resident-bf16"
            ]
        )
        if precision
        in (
            "ffn_attention_sdpa_resident",
            "ffn_attention_sdpa_storage",
            "ffn_attention_sdpa_storage_encoder",
        )
        else dict(optimizations=changes["optimizations"] + ["codec-resident-bf16"]),
        dict(
            optimizations=[
                feature for feature in changes["optimizations"] if feature != "codec-flow-bf16"
            ]
        ),
    ]:
        with pytest.raises(ValueError):
            load_suite(configuration(tmp_path, **(changes | patch)))
    with pytest.raises(ValueError):
        load_suite(configuration(tmp_path, run_vocoder=dict(flow_precision="ffn")))


@pytest.mark.parametrize(
    "sdpa,resident,storage,encoder",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ],
)
def test_bf16_structural_proof_cannot_claim_old_fp32_numeric_pass(sdpa, resident, storage, encoder):
    selected = ["codec-bias-broadcast", "codec-unready-return", "codec-flow-bf16"]
    if sdpa:
        selected.append("codec-sdpa-bf16")
    if resident:
        selected.append("codec-resident-bf16")
    if storage:
        selected.append("codec-storage-bf16")
    if encoder:
        selected.append("codec-encoder-bf16")
    reference = dict(report_sha256="a" * 64, tensors_sha256="b" * 64, pcm_atol=1e-5, pcm_rtol=1e-5)
    report = dict(
        optimizations=selected,
        reference=reference,
        joint_pcm_within_frozen_tolerance=False,
        precision_contract="e13-bf16-approx-v1",
        same_precision_pcm_exact=True,
        cross_precision_statistics=dict(
            pcm_max_abs=0.9, pcm_relative_l2=0.8, pcm_tensor_cosine=0.7
        ),
        code_ledger_exact=True,
        request_state_isolation_exact=True,
        rng_isolation_exact=True,
        cleanup_exact=True,
        executions={feature: 1 for feature in selected},
        bf16_sdpa_mask_broadcast=sdpa,
        bf16_weight_residency=resident,
        resident_weight_bytes=4 if resident else 0,
        resident_parameter_copies=1 if resident else 0,
        bf16_flow_encoder=encoder,
        encoder_fp32_boundaries=encoder,
        encoder_bf16_executions=1 if encoder else 0,
        encoder_bf16_convolution_executions=1 if encoder else 0,
        bf16_parameter_storage=storage,
        fp32_cpu_backup_bytes=8 if storage else 0,
    )
    validate_codec_qualification(report, selected, reference, "e13-bf16-approx-v1")
    with pytest.raises(AssertionError):
        validate_codec_qualification(report, selected, reference)
    for patch in [
        dict(same_precision_pcm_exact=False),
        dict(joint_pcm_within_frozen_tolerance=True),
        dict(precision_contract="unknown"),
        dict(rng_isolation_exact=False),
    ]:
        with pytest.raises(AssertionError):
            validate_codec_qualification(report | patch, selected, reference, "e13-bf16-approx-v1")
    if sdpa:
        for patch in (
            dict(bf16_sdpa_mask_broadcast=False),
            dict(executions=report["executions"] | {"codec-sdpa-bf16": 0}),
        ):
            with pytest.raises(AssertionError):
                validate_codec_qualification(
                    report | patch, selected, reference, "e13-bf16-approx-v1"
                )
    if resident:
        for patch in (
            dict(bf16_weight_residency=False),
            dict(resident_weight_bytes=0),
            dict(resident_parameter_copies=0),
        ):
            with pytest.raises(AssertionError):
                validate_codec_qualification(
                    report | patch, selected, reference, "e13-bf16-approx-v1"
                )
    if storage:
        for patch in (dict(bf16_parameter_storage=False), dict(fp32_cpu_backup_bytes=0)):
            with pytest.raises(AssertionError):
                validate_codec_qualification(
                    report | patch, selected, reference, "e13-bf16-approx-v1"
                )


def test_encoder_bf16_proof_requires_real_execution_and_fp32_boundary():
    reference = dict(report_sha256="a" * 64, tensors_sha256="b" * 64, pcm_atol=1e-5, pcm_rtol=1e-5)
    selected = [
        "codec-unready-return",
        "codec-bias-broadcast",
        "codec-flow-bf16",
        "codec-sdpa-bf16",
        "codec-resident-bf16",
        "codec-storage-bf16",
        "codec-encoder-bf16",
    ]
    report = dict(
        optimizations=selected,
        reference=reference,
        precision_contract="e13-bf16-approx-v1",
        joint_pcm_within_frozen_tolerance=False,
        same_precision_pcm_exact=True,
        cross_precision_statistics=dict(
            pcm_max_abs=0.9, pcm_relative_l2=0.8, pcm_tensor_cosine=0.7
        ),
        code_ledger_exact=True,
        request_state_isolation_exact=True,
        rng_isolation_exact=True,
        cleanup_exact=True,
        executions={feature: 1 for feature in selected},
        bf16_sdpa_mask_broadcast=True,
        bf16_weight_residency=True,
        resident_weight_bytes=4,
        resident_parameter_copies=1,
        bf16_parameter_storage=True,
        fp32_cpu_backup_bytes=8,
        bf16_flow_encoder=True,
        encoder_fp32_boundaries=True,
        encoder_bf16_executions=1,
        encoder_bf16_convolution_executions=1,
    )
    validate_codec_qualification(report, selected, reference, "e13-bf16-approx-v1")
    for patch in [
        dict(bf16_flow_encoder=False),
        dict(encoder_fp32_boundaries=False),
        dict(encoder_bf16_executions=0),
        dict(encoder_bf16_convolution_executions=0),
    ]:
        with pytest.raises(AssertionError):
            validate_codec_qualification(report | patch, selected, reference, "e13-bf16-approx-v1")


def test_full_flow_bf16_requires_its_own_explicit_contract_and_solver_proof(tmp_path):
    features = [
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "codec-bias-broadcast",
        "codec-unready-return",
        "codec-full-flow-bf16",
    ]
    changes = dict(
        optimizations=features,
        precision_contract="e13-bf16-approx-v1",
        deployment=dict(
            ar_graph=True,
            decode_graph=True,
            decode_graph_serial_batch=True,
            broadcast_attention_bias=True,
            flow_precision="full_flow_bf16",
        ),
        run_vocoder=dict(broadcast_attention_bias=True, flow_precision="full_flow_bf16"),
    )
    config = load_suite(configuration(tmp_path, **changes))
    for gate in ("C01", "C03"):
        nodes = selected_nodes(
            gate, "cumulative", True, "eager-bugs", config["optimizations"], "independent"
        )
        assert any("test_full_flow_precision.py" in node for node in nodes)
        assert len(nodes) == len(set(nodes))
    for patch in [
        dict(precision_contract=None),
        dict(optimizations=[feature for feature in features if feature != "codec-full-flow-bf16"]),
        dict(optimizations=features + ["codec-flow-bf16"]),
        dict(
            run_vocoder=dict(
                broadcast_attention_bias=True, flow_precision="ffn_attention_sdpa_storage"
            )
        ),
    ]:
        with pytest.raises(ValueError):
            load_suite(configuration(tmp_path, **(changes | patch)))


def test_full_flow_bf16_proof_requires_real_solver_states_and_cpu_backups():
    selected = ["codec-bias-broadcast", "codec-unready-return", "codec-full-flow-bf16"]
    reference = dict(report_sha256="a" * 64, tensors_sha256="b" * 64, pcm_atol=1e-5, pcm_rtol=1e-5)
    proof = dict(
        optimizations=selected,
        reference=reference,
        precision_contract="e13-bf16-approx-v1",
        joint_pcm_within_frozen_tolerance=False,
        same_precision_pcm_exact=True,
        cross_precision_statistics=dict(
            pcm_max_abs=1.0, pcm_relative_l2=1.0, pcm_tensor_cosine=0.0
        ),
        code_ledger_exact=True,
        request_state_isolation_exact=True,
        rng_isolation_exact=True,
        cleanup_exact=True,
        executions={feature: 10 for feature in selected},
        full_flow_bf16=True,
        bf16_solver_steps=10,
        bf16_solver_state_exact_dtype=True,
        bf16_sdpa_mask_broadcast=True,
        full_flow_bf16_parameter_bytes=4,
        full_flow_cpu_backup_bytes=8,
    )
    validate_codec_qualification(proof, selected, reference, "e13-bf16-approx-v1")
    for patch in [
        dict(full_flow_bf16=False),
        dict(bf16_solver_steps=0),
        dict(bf16_solver_state_exact_dtype=False),
        dict(bf16_sdpa_mask_broadcast=False),
        dict(full_flow_bf16_parameter_bytes=0),
        dict(full_flow_cpu_backup_bytes=4),
        dict(same_precision_pcm_exact=False),
    ]:
        with pytest.raises(AssertionError):
            validate_codec_qualification(proof | patch, selected, reference, "e13-bf16-approx-v1")
    with pytest.raises(AssertionError):
        validate_codec_qualification(proof, selected, reference)

    asynchronous = proof | dict(
        optimizations=selected + ["codec-async"],
        executions=proof["executions"] | {"codec-async": 2},
        async_flow_slots=2,
        async_pcm_exact=True,
        async_bf16_flow_slots=2,
        async_bf16_parameter_bytes=8,
        async_fp32_cpu_backup_bytes=16,
    )
    selected_async = selected + ["codec-async"]
    validate_codec_qualification(asynchronous, selected_async, reference, "e13-bf16-approx-v1")
    for patch in [
        dict(async_bf16_flow_slots=1),
        dict(async_bf16_parameter_bytes=0),
        dict(async_fp32_cpu_backup_bytes=8),
        dict(async_pcm_exact=False),
    ]:
        with pytest.raises(AssertionError):
            validate_codec_qualification(
                asynchronous | patch, selected_async, reference, "e13-bf16-approx-v1"
            )

    middle = asynchronous | dict(
        optimizations=selected_async + ["codec-middle-bias-reuse"],
        executions=asynchronous["executions"] | {"codec-middle-bias-reuse": 22},
        middle_bias_exact=True,
        middle_bias_constructions=2,
        middle_bias_reuses=22,
        middle_bias_fallbacks=0,
        async_middle_bias_slots=2,
        async_middle_bias_reuses=44,
    )
    middle_selected = middle["optimizations"]
    validate_codec_qualification(middle, middle_selected, reference, "e13-bf16-approx-v1")
    for patch in [
        dict(middle_bias_exact=False),
        dict(middle_bias_constructions=0),
        dict(middle_bias_reuses=0),
        dict(middle_bias_fallbacks=1),
        dict(async_middle_bias_slots=1),
        dict(async_middle_bias_reuses=0),
    ]:
        with pytest.raises(AssertionError):
            validate_codec_qualification(
                middle | patch, middle_selected, reference, "e13-bf16-approx-v1"
            )


@pytest.mark.parametrize(
    "mismatch", [None, "undeclared", "deployment", "factory", "float", "dependencies"]
)
def test_async_codec_requires_explicit_two_slot_qualification(tmp_path, mismatch):
    selected = [
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "codec-bias-broadcast",
        "codec-cfm-graph",
        "codec-unready-return",
        "codec-async",
    ]
    deployment = dict(
        ar_graph=True,
        decode_graph=True,
        decode_graph_serial_batch=True,
        broadcast_attention_bias=True,
        cfm_graph_mel_length=1500,
        codec_flow_slots=2,
    )
    vocoder = dict(broadcast_attention_bias=True, cfm_graph_mel_length=1500, codec_flow_slots=2)
    if mismatch == "undeclared":
        selected.remove("codec-async")
    elif mismatch == "deployment":
        deployment["codec_flow_slots"] = 1
    elif mismatch == "factory":
        vocoder["codec_flow_slots"] = 1
    elif mismatch == "float":
        vocoder["codec_flow_slots"] = 2.0
    elif mismatch == "dependencies":
        selected.remove("codec-unready-return")
    path = configuration(
        tmp_path, optimizations=selected, deployment=deployment, run_vocoder=vocoder
    )
    if mismatch is None:
        config = load_suite(path)
        cpu = selected_nodes("C01", "cumulative", True, "eager-bugs", selected)
        cuda = selected_nodes("C03", "cumulative", True, "eager-bugs", selected)
        assert sum("async_codec_factory_defers_graph" in node for node in cpu) == 2
        assert sum("async_vocoder_fullweights" in node for node in cuda) == 2
        log = tmp_path / "server.log"
        log.write_text(
            "MOSS full decode graph replay active\nserial batch decode graph replay active batch=2\nserial batch decode graph replay active batch=4\nMOSS_CODEC_BIAS_BROADCAST_ACTIVE\n"
        )
        with pytest.raises(AssertionError):
            validate_activation(config, tmp_path / "no-real-async")
        with log.open("a") as output:
            output.write("MOSS_CODEC_ASYNC_ACTIVE flow_slots=2 cudnn_enabled=True\n")
        assert validate_activation(config, tmp_path / "actual-async")["status"] == "pass"
    else:
        with pytest.raises(ValueError):
            load_suite(path)


def test_async_codec_joint_proof_requires_actual_workers_and_exact_isolation():
    selected = ["codec-bias-broadcast", "codec-cfm-graph", "codec-unready-return", "codec-async"]
    reference = dict(report_sha256="a" * 64, tensors_sha256="b" * 64, pcm_atol=1e-5, pcm_rtol=1e-5)
    report = dict(
        optimizations=selected,
        reference=reference,
        joint_pcm_within_frozen_tolerance=True,
        code_ledger_exact=True,
        request_state_isolation_exact=True,
        rng_isolation_exact=True,
        cleanup_exact=True,
        executions={feature: 1 for feature in selected},
        async_flow_slots=2,
        async_pcm_exact=True,
    )
    validate_codec_qualification(report, selected, reference)
    for name, value in [
        ("async_flow_slots", 1),
        ("async_pcm_exact", False),
        ("rng_isolation_exact", False),
    ]:
        changed = copy.deepcopy(report)
        changed[name] = value
        with pytest.raises(AssertionError):
            validate_codec_qualification(changed, selected, reference)
    changed = copy.deepcopy(report)
    changed["executions"]["codec-async"] = 0
    with pytest.raises(AssertionError):
        validate_codec_qualification(changed, selected, reference)


def test_async_full_flow_bf16_requires_preserved_precision_and_actual_service_execution(tmp_path):
    selected = [
        "ar-decode-graph",
        "ar-serial-decode-graph",
        "codec-bias-broadcast",
        "codec-unready-return",
        "codec-full-flow-bf16",
        "codec-async",
    ]
    deployment = dict(
        ar_graph=True,
        decode_graph=True,
        decode_graph_serial_batch=True,
        broadcast_attention_bias=True,
        flow_precision="full_flow_bf16",
        codec_flow_slots=2,
    )
    vocoder = dict(
        broadcast_attention_bias=True, flow_precision="full_flow_bf16", codec_flow_slots=2
    )
    config = load_suite(
        configuration(
            tmp_path,
            optimizations=selected,
            deployment=deployment,
            run_vocoder=vocoder,
            precision_contract="e13-bf16-approx-v1",
        )
    )
    log = tmp_path / "server.log"
    log.write_text(
        "full decode graph replay active\nserial batch decode graph replay active batch=2\nserial batch decode graph replay active batch=4\nMOSS_CODEC_BIAS_BROADCAST_ACTIVE\nMOSS_CODEC_FULL_FLOW_BF16_ACTIVE\nMOSS_CODEC_ASYNC_ACTIVE flow_slots=2 cudnn_enabled=True\n"
    )
    with pytest.raises(AssertionError, match="codec-async"):
        validate_activation(config, tmp_path / "missing-active-precision")
    with log.open("a") as output:
        output.write(
            "MOSS_CODEC_ASYNC_ACTIVE flow_slots=2 cudnn_enabled=True flow_precision=full_flow_bf16\n"
        )
    assert validate_activation(config, tmp_path / "actual-bf16-concurrency")["status"] == "pass"


def test_async_parameterized_checks_are_enumerated_in_the_frozen_selection():
    selection = json.loads(
        (Path(selection_module.__file__).with_name("selection.json")).read_text()
    )
    guard = "tests/unit_test/moss_speech/test_config.py::test_async_codec_rejects_incompatible_precision_before_loading"
    stream = "tests/unit_test/moss_speech/test_streaming.py::test_async_vocoder_fullweights_stream_completion_and_abort"
    assert guard not in selection["C01-codec-async"]
    assert {
        guard + "[" + key + "]"
        for key in ["full_flow_bf16-1500", "ffn-None", "ffn_attention_sdpa_storage-None"]
    } <= set(selection["C01-codec-async"])
    assert set(selection["C03-codec-async"]) == {stream + "[fp32]", stream + "[full-bf16]"}


@pytest.mark.parametrize("mismatch", [None, "undeclared", "deployment", "factory"])
def test_middle_bias_reuse_requires_explicit_serving_flag_and_qualification(tmp_path, mismatch):
    selected = [
        "codec-bias-broadcast",
        "codec-unready-return",
        "codec-full-flow-bf16",
        "codec-middle-bias-reuse",
    ]
    deployment = dict(
        ar_graph=False,
        broadcast_attention_bias=True,
        flow_precision="full_flow_bf16",
        reuse_middle_attention_bias=True,
    )
    vocoder = dict(
        broadcast_attention_bias=True,
        flow_precision="full_flow_bf16",
        reuse_middle_attention_bias=True,
    )
    if mismatch == "undeclared":
        selected.remove("codec-middle-bias-reuse")
    elif mismatch == "deployment":
        deployment["reuse_middle_attention_bias"] = False
    elif mismatch == "factory":
        vocoder["reuse_middle_attention_bias"] = False
    path = configuration(
        tmp_path,
        optimizations=selected,
        deployment=deployment,
        run_factory={},
        run_vocoder=vocoder,
        precision_contract="e13-bf16-approx-v1",
    )
    if mismatch is not None:
        with pytest.raises(ValueError):
            load_suite(path)
    else:
        config = load_suite(path)
        cpu = selected_nodes("C01", "cumulative", True, "eager-bugs", selected)
        cuda = selected_nodes("C03", "cumulative", True, "eager-bugs", selected)
        assert sum("test_middle_bias_reuse" in node for node in cpu) == 10
        assert sum("test_middle_bias_reuse" in node for node in cuda) == 4
        assert "codec-middle-bias-reuse" in config["optimizations"]
