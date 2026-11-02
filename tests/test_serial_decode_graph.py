"""Explicit full-weight and actual-serving gates for serial B2/B4 replay."""

import copy
import json

import pytest

from moss_speech_bench.correctness import load_suite
from moss_speech_bench.providers.ar_combination import validate_activation
from moss_speech_bench.providers.serial_decode_graph import validate_qualification


def configuration(tmp_path, **changes):
    value = dict(
        workspace_root=".",
        core_sha="a" * 40,
        profile="ar-serial-decode-graph",
        context_remaining=True,
        deployment=dict(decode_graph_serial_batch=True, decode_graph=True, ar_graph=True),
        service_activation_log=str(tmp_path / "server.log"),
    )
    value.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return path


def proof():
    return dict(
        batch_sizes=[2, 4],
        executions={"2": 4, "4": 4},
        captures=4,
        kv_layers=40,
        complete_heads_exact=True,
        complete_hidden_exact=True,
        no_partial_retry=True,
        all_kv_exact=True,
        ownership_exact=True,
        all_row_rejection_exact=True,
        runtime_guards_exact=True,
        boundary_fallback_exact=True,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"deployment": {}},
        {"context_remaining": False},
        {"profile": "hf-reference"},
        {"deployment": dict(decode_graph_serial_batch=1, decode_graph=True, ar_graph=True)},
        {"deployment": dict(decode_graph_serial_batch=True, decode_graph=False, ar_graph=True)},
        {
            "deployment": dict(
                decode_graph_serial_batch=True,
                decode_graph=True,
                ar_graph=True,
                decode_rope_reuse=True,
            )
        },
        {"service_activation_required": []},
        {"service_activation_offset": -1},
        {"native_norm_library": "unqualified.so"},
    ],
)
def test_reject_unqualified_path(tmp_path, change):
    with pytest.raises(ValueError):
        load_suite(configuration(tmp_path, **change))


def test_complete_batch_proof_required():
    validate_qualification(proof())
    for key, value in [
        ("batch_sizes", [2]),
        ("executions", {"2": 1, "4": 0}),
        ("executions", {"2": True, "4": 1}),
        ("kv_layers", 39),
        ("captures", 0),
    ]:
        with pytest.raises(AssertionError):
            validate_qualification(dict(proof(), **{key: value}))
    for key in (
        "complete_heads_exact",
        "complete_hidden_exact",
        "no_partial_retry",
        "all_kv_exact",
        "ownership_exact",
        "all_row_rejection_exact",
        "runtime_guards_exact",
        "boundary_fallback_exact",
    ):
        value = copy.deepcopy(proof())
        value[key] = False
        with pytest.raises(AssertionError):
            validate_qualification(value)


def test_actual_both_batches_not_startup_or_stale(tmp_path):
    config = load_suite(configuration(tmp_path))
    log = tmp_path / "server.log"
    for text in (
        "decode_graph_serial_batch=True startup\n",
        "serial batch decode graph replay active batch=2\n",
        "serial batch decode graph replay active batch=24\n",
    ):
        log.write_text(text)
        with pytest.raises(AssertionError):
            validate_activation(config, tmp_path / "missing")
    log.write_text(
        "serial batch decode graph replay active batch=2\n"
        "serial batch decode graph replay active batch=4\n"
    )
    result = validate_activation(config, tmp_path / "pass")
    assert result["status"] == "pass"
    assert result["serial_decode_batches"] == {"2": 1, "4": 1}
    config["service_activation_offset"] = log.stat().st_size
    with pytest.raises(AssertionError):
        validate_activation(config, tmp_path / "stale")
