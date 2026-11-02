"""Keep cached HF evidence additional, independent, complete and phase-matched."""

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from moss_speech_bench.providers.cached_reference import validate_cached_reference

REVISION = "cff025bb41d8459d59abac0b5e44aba7f659ec9e"
CASES = (
    "t2t_short",
    "t2s_cn",
    "s2t_cn",
    "s2s_cn",
    "mixed_multiturn",
    "unicode_no_system_text",
    "unicode_no_system_audio",
    "history_empty_no_system",
)
VOCABULARIES = {"text": 151680, "audio": 16512}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_sha(case):
    value = {key: case[key] for key in ("input_ids", "attention_mask")}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_tensor(path, header, data=None):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    length = max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        if data is None:
            stream.seek(length - 1, 1)
            stream.write(b"\0")
        else:
            stream.write(data)


@pytest.fixture
def bundle(tmp_path):
    model_path = tmp_path / "modeling_moss_speech.py"
    config_path = tmp_path / "configuration_moss_speech.py"
    model_path.write_text("# pinned official model\n")
    config_path.write_text("# pinned official configuration\n")
    sources = {path.name: sha(path) for path in (model_path, config_path)}
    inputs = dict(
        status="passed",
        expected_model_revision=REVISION,
        serving_core_modules_imported=[],
        source_sha256=sources,
        cases={},
    )
    cases = {}
    for index, name in enumerate(CASES):
        case = dict(
            input_ids=[[row + 1, 0] for row in range(index + 3)],
            attention_mask=[1] * (index + 3),
            output_modality="text",
        )
        inputs["cases"][name] = copy.deepcopy(case)
        length = len(case["input_ids"])
        cases[name] = dict(
            **case,
            grid_sha256=grid_sha(case),
            status="passed",
            compute_phase=dict(
                phase="cached_decode",
                prefix_rows=length - 1,
                decode_rows=1,
                cache_position=length - 1,
                full_attention_mask_rows=length,
                cache_rows_before={key: length - 1 for key in ("shared", "text", "audio")},
                cache_rows_after={key: length for key in ("shared", "text", "audio")},
            ),
            heads={
                head: dict(
                    tensor_key=f"{name}.{head}_last_logits",
                    forward_shape=[1, 1, size],
                    export_shape=[size],
                    forward_dtype="torch.bfloat16",
                    export_dtype="torch.bfloat16",
                    all_finite=True,
                )
                for head, size in VOCABULARIES.items()
            },
        )
    model = dict(
        parameter_count=9095694336,
        parameter_dtypes=["torch.bfloat16"],
        text_vocab_size=151680,
        audio_vocab_size=16512,
        attention_implementation="sdpa",
        trained_checkpoint_loaded=True,
        use_cache=True,
        logits_to_keep=0,
    )
    report = dict(
        schema_version=1,
        status="passed",
        compute_phase="cached_decode",
        original_prefill_reference_retained=True,
        model_revision=REVISION,
        serving_core_modules_imported=[],
        source_sha256=sources,
        case_order=list(CASES),
        determinism=dict(seed=20261027, algorithms_required=True, tf32=False),
        requested_profile=dict(attention_backend="sdpa", logits_to_keep=0),
        package_versions={"torch": "pinned", "transformers": "pinned"},
        model=model,
        tensor_count=16,
        cases=cases,
        checkpoint=dict(index_sha256="a" * 64, total_parameters=9095694336),
        loaded_model_module=dict(file=str(model_path), sha256=sha(model_path)),
        loaded_hf_module_audit={
            f"transformers_modules.{REVISION}.{path.stem}": dict(file=str(path), sha256=sha(path))
            for path in (model_path, config_path)
        },
    )
    prefill = copy.deepcopy(report)
    prefill.pop("compute_phase")
    prefill["model"]["use_cache"] = False
    for case in prefill["cases"].values():
        for head, size in VOCABULARIES.items():
            case["heads"][head]["forward_shape"] = [1, len(case["input_ids"]), size]
    prefill["tensor_artifact_sha256"] = "b" * 64
    paths = {
        key: tmp_path / name
        for key, name in dict(
            report="cached.json",
            tensor="cached.safetensors",
            inputs="inputs.json",
            prefill="prefill.json",
            policy="policy.json",
        ).items()
    }
    policy = dict(
        absolute_tolerance=0.25,
        relative_tolerance=0.02,
        finite_required=True,
        post_observation_tolerance_increase_allowed=False,
    )
    for key, value in [("inputs", inputs), ("policy", policy)]:
        paths[key].write_text(json.dumps(value))
    input_hash = sha(paths["inputs"])
    report["inputs_json_sha256"] = prefill["inputs_json_sha256"] = input_hash
    paths["prefill"].write_text(json.dumps(prefill))
    header = {
        "__metadata__": dict(
            model_revision=REVISION,
            inputs_json_sha256=input_hash,
            attention_backend="sdpa",
            logits_to_keep="0",
            dtype="BF16 raw head output",
        )
    }
    offset = 0
    for name in CASES:
        for head, size in VOCABULARIES.items():
            header[f"{name}.{head}_last_logits"] = dict(
                dtype="BF16", shape=[size], data_offsets=[offset, offset + size * 2]
            )
            offset += size * 2
    save_tensor(paths["tensor"], header)
    report["tensor_artifact_sha256"] = sha(paths["tensor"])
    report["pinned_source_metadata"] = dict(
        hf_model_revision=REVISION,
        inputs_json_sha256=input_hash,
        original_prefill_reference_json_sha256=sha(paths["prefill"]),
        original_prefill_tensors_sha256=prefill["tensor_artifact_sha256"],
        policy_sha256=sha(paths["policy"]),
    )
    paths["report"].write_text(json.dumps(report))
    return paths, report, header


def validate(paths):
    return validate_cached_reference(
        *(paths[key] for key in ("report", "tensor", "inputs", "prefill", "policy"))
    )


def replace_field(report, field, value):
    keys = field.split(".")
    for key in keys[:-1]:
        report = report[key]
    report[keys[-1]] = value


def test_valid_reference_is_additional_and_does_not_mutate_files(bundle):
    paths, report, _ = bundle
    before = {key: sha(path) for key, path in paths.items()}
    assert validate(paths) == report
    assert {key: sha(path) for key, path in paths.items()} == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("compute_phase", "prefill"),
        ("model_revision", "wrong"),
        ("original_prefill_reference_retained", False),
        ("serving_core_modules_imported", ["sglang_omni"]),
        ("determinism.seed", 1),
        ("determinism.algorithms_required", False),
        ("determinism.tf32", True),
        ("model.parameter_dtypes", ["torch.float32"]),
        ("model.attention_implementation", "eager"),
        ("model.use_cache", False),
        ("model.trained_checkpoint_loaded", False),
        ("model.text_vocab_size", 16),
        ("model.audio_vocab_size", 16),
        ("requested_profile.attention_backend", "eager"),
        ("tensor_count", 15),
        ("case_order", list(CASES[:-1])),
        ("inputs_json_sha256", "0" * 64),
        ("pinned_source_metadata.original_prefill_reference_json_sha256", "0" * 64),
        ("pinned_source_metadata.original_prefill_tensors_sha256", "0" * 64),
        ("pinned_source_metadata.policy_sha256", "0" * 64),
        ("loaded_model_module.sha256", "0" * 64),
        ("cases.t2t_short.grid_sha256", "0" * 64),
        ("cases.t2t_short.input_ids", [[9, 0], [1, 0]]),
        ("cases.t2t_short.attention_mask", [1, 0, 1]),
        ("cases.t2t_short.compute_phase.phase", "prefill"),
        ("cases.t2t_short.compute_phase.prefix_rows", 1),
        ("cases.t2t_short.compute_phase.decode_rows", 2),
        ("cases.t2t_short.compute_phase.cache_position", 0),
        ("cases.t2t_short.compute_phase.full_attention_mask_rows", 2),
        ("cases.t2t_short.compute_phase.cache_rows_before", {"shared": 2}),
        ("cases.t2t_short.compute_phase.cache_rows_after", {"shared": 3, "text": 3, "audio": 2}),
        ("cases.t2t_short.heads.audio.export_shape", [16]),
        ("cases.t2t_short.heads.audio.forward_dtype", "torch.float32"),
        ("cases.t2t_short.heads.audio.all_finite", False),
        ("cases.t2t_short.heads.audio.tensor_key", "wrong.audio_last_logits"),
    ],
)
def test_rejects_incomplete_or_changed_evidence(bundle, field, value):
    paths, report, _ = bundle
    replace_field(report, field, value)
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)


@pytest.mark.parametrize(
    "field", ["compute_phase", "pinned_source_metadata", "loaded_hf_module_audit", "model", "cases"]
)
def test_missing_required_fields_fail_closed(bundle, field):
    paths, report, _ = bundle
    report.pop(field)
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)


@pytest.mark.parametrize(
    "change", ["missing-head", "missing-case", "missing-module", "wrong-module-hash"]
)
def test_complete_head_case_and_origin_coverage(bundle, change):
    paths, report, _ = bundle
    if change == "missing-head":
        report["cases"][CASES[0]]["heads"].pop("audio")
    elif change == "missing-case":
        report["cases"].pop(CASES[-1])
    elif change == "missing-module":
        report["loaded_hf_module_audit"] = {}
    else:
        next(iter(report["loaded_hf_module_audit"].values()))["sha256"] = "0" * 64
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)


@pytest.mark.parametrize(
    "change", ["dtype", "shape", "missing-head", "offset", "nonfinite", "truncated", "trailing"]
)
def test_tensor_bytes_are_validated_even_when_hash_is_rebound(bundle, change):
    paths, report, header = bundle
    key = f"{CASES[0]}.audio_last_logits"
    if change in ("dtype", "shape", "offset"):
        header[key][{"dtype": "dtype", "shape": "shape", "offset": "data_offsets"}[change]] = {
            "dtype": "F32",
            "shape": [1],
            "offset": [0, 33024],
        }[change]
    elif change == "missing-head":
        header.pop(key)
    save_tensor(paths["tensor"], header)
    if change == "nonfinite":
        with paths["tensor"].open("r+b") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(8 + length)
            stream.write(struct.pack("<H", 0x7F80))
    elif change == "truncated":
        with paths["tensor"].open("r+b") as stream:
            stream.truncate(paths["tensor"].stat().st_size - 1)
    elif change == "trailing":
        with paths["tensor"].open("ab") as stream:
            stream.write(b"x")
    report["tensor_artifact_sha256"] = sha(paths["tensor"])
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)


@pytest.mark.parametrize(
    "changed", ["cached-tensor", "loaded-module", "original-prefill", "inputs", "policy"]
)
def test_tampered_bound_artifacts_are_rejected(bundle, changed):
    paths, report, _ = bundle
    path = {
        "cached-tensor": paths["tensor"],
        "loaded-module": Path(report["loaded_model_module"]["file"]),
        "original-prefill": paths["prefill"],
        "inputs": paths["inputs"],
        "policy": paths["policy"],
    }[changed]
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        validate(paths)


def test_retained_original_cannot_itself_be_a_cached_oracle(bundle):
    paths, report, _ = bundle
    prefill = json.loads(paths["prefill"].read_text())
    prefill["model"]["use_cache"] = True
    paths["prefill"].write_text(json.dumps(prefill))
    report["pinned_source_metadata"]["original_prefill_reference_json_sha256"] = sha(
        paths["prefill"]
    )
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)


def test_tolerances_cannot_be_weakened_even_with_rebound_policy(bundle):
    paths, report, _ = bundle
    policy = json.loads(paths["policy"].read_text())
    policy["absolute_tolerance"] = 1.0
    paths["policy"].write_text(json.dumps(policy))
    report["pinned_source_metadata"]["policy_sha256"] = sha(paths["policy"])
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)


def test_validation_needs_only_standard_library_without_site_packages(bundle):
    paths, _, _ = bundle
    source = Path(__file__).parents[1] / "src"
    command = "from moss_speech_bench.providers.cached_reference import validate_cached_reference; import sys; validate_cached_reference(*sys.argv[1:]); assert not any(n.split('.')[0] in {'torch','sglang','sglang_omni'} for n in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-S", "-c", command, *map(str, paths.values())],
        env=dict(os.environ, PYTHONPATH=str(source)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("change", ["missing-head", "one-row-prefill", "dtype"])
def test_original_full_prefill_coverage_cannot_be_weakened(bundle, change):
    paths, report, _ = bundle
    original = json.loads(paths["prefill"].read_text())
    heads = original["cases"][CASES[0]]["heads"]
    if change == "missing-head":
        heads.pop("audio")
    elif change == "one-row-prefill":
        heads["text"]["forward_shape"][1] = 1
    else:
        heads["audio"]["export_dtype"] = "torch.float32"
    paths["prefill"].write_text(json.dumps(original))
    report["pinned_source_metadata"]["original_prefill_reference_json_sha256"] = sha(
        paths["prefill"]
    )
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)


def test_noninteger_tensor_shape_is_not_accepted_as_equal_integer(bundle):
    paths, report, header = bundle
    header[f"{CASES[0]}.text_last_logits"]["shape"] = [151680.0]
    save_tensor(paths["tensor"], header)
    report["tensor_artifact_sha256"] = sha(paths["tensor"])
    paths["report"].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate(paths)
