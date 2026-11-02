"""Reject silent eager fallback and unrelated Codec optimization combinations."""

import json

import pytest

from moss_speech_bench.correctness import load_suite
from moss_speech_bench.providers.ar_combination import validate_activation
from moss_speech_bench.providers.codec_bias_broadcast import validate_qualification


def config(tmp_path, **changes):
    data = dict(
        workspace_root=".",
        core_sha="a" * 40,
        profile="codec-bias-broadcast",
        context_remaining=True,
        deployment=dict(broadcast_attention_bias=True),
        service_activation_log=str(tmp_path / "server.log"),
    )
    data.update(changes)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    "change",
    [
        {"profile": "hf-reference"},
        {"deployment": {}},
        {"context_remaining": False},
        {"deployment": dict(broadcast_attention_bias=1)},
        {"deployment": dict(broadcast_attention_bias=True, cfm_graph_mel_length=1400)},
        {"deployment": dict(broadcast_attention_bias=True, decode_graph_serial_batch=True)},
        {"service_activation_required": []},
        {"service_activation_offset": -1},
    ],
)
def test_unqualified_profile_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load_suite(config(tmp_path, **change))


def test_actual_all_processors_required():
    data = dict(installed=56, calls=[20] * 56, fallback_calls=[0] * 56, head_strides=[0] * 56)
    validate_qualification(data)
    for bad in (
        dict(data, installed=55),
        dict(data, head_strides=[1] * 56),
        dict(data, calls=[0] * 56),
        dict(data, calls=[20] * 55),
        dict(data, calls=[True] * 56),
        dict(data, fallback_calls=[-1] * 56),
    ):
        with pytest.raises(AssertionError):
            validate_qualification(bad)


def test_actual_marker_and_fresh_slice(tmp_path):
    value = load_suite(config(tmp_path))
    log = tmp_path / "server.log"
    log.write_text("broadcast_attention_bias=True startup\n")
    with pytest.raises(AssertionError):
        validate_activation(value, tmp_path / "missing")
    log.write_text("MOSS_CODEC_BIAS_BROADCAST_ACTIVE\n")
    assert validate_activation(value, tmp_path / "pass")["status"] == "pass"
    value["service_activation_offset"] = log.stat().st_size
    with pytest.raises(AssertionError):
        validate_activation(value, tmp_path / "stale")
