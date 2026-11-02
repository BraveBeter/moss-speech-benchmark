"""Validate an additional independent cached-decode HF oracle without CUDA imports."""

import hashlib
import json
import struct
from pathlib import Path

MODEL_REVISION = "cff025bb41d8459d59abac0b5e44aba7f659ec9e"
CASE_NAMES = (
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
CACHE_BANKS = {"shared", "text", "audio"}


def require(condition, message):
    if not condition:
        raise ValueError(f"Invalid cached HF reference: {message}")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_hash(case):
    payload = {key: case[key] for key in ("input_ids", "attention_mask")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_tensors(path, expected, inputs_hash):
    with Path(path).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        require(0 < header_size <= Path(path).stat().st_size - 8, "tensor header size")
        header = json.loads(stream.read(header_size))
        payload = stream.read()
    metadata = header.pop("__metadata__")
    require(metadata["model_revision"] == MODEL_REVISION, "tensor model revision")
    require(metadata["inputs_json_sha256"] == inputs_hash, "tensor input identity")
    require(metadata["attention_backend"] == "sdpa", "tensor attention backend")
    require(metadata["logits_to_keep"] == "0", "tensor head projection shape")
    require(metadata["dtype"] == "BF16 raw head output", "tensor dtype metadata")
    require(set(header) == set(expected), "complete tensor head coverage")
    spans = []
    for key, size in expected.items():
        tensor = header[key]
        require(tensor["dtype"] == "BF16" and tensor["shape"] == [size], f"{key} dtype/shape")
        offsets = tensor["data_offsets"]
        require(isinstance(offsets, list) and len(offsets) == 2, f"{key} offsets")
        require(all(type(value) is int for value in offsets), f"{key} integer offsets")
        start, end = offsets
        require(0 <= start < end <= len(payload) and end - start == 2 * size, f"{key} span")
        require(
            all(type(dimension) is int for dimension in tensor["shape"]), f"{key} integer shape"
        )
        spans.append((start, end))
    previous_end = 0
    for start, end in sorted(spans):
        require(start == previous_end, "overlapping or incomplete tensor payload")
        previous_end = end
    require(previous_end == len(payload), "trailing tensor payload")
    require(
        not any(value & 0x7F80 == 0x7F80 for (value,) in struct.iter_unpack("<H", payload)),
        "nonfinite BF16 tensor value",
    )


def validate_cached_reference(report_path, tensor_path, inputs_path, prefill_path, policy_path):
    """Return cached evidence only when the original prefill oracle remains bound and valid."""
    try:
        report, inputs, prefill, policy = (
            json.loads(Path(path).read_text())
            for path in (report_path, inputs_path, prefill_path, policy_path)
        )
        require(
            type(report["schema_version"]) is int and report["schema_version"] == 1, "report schema"
        )
        require(report["status"] == inputs["status"] == prefill["status"] == "passed", "status")
        require(report["compute_phase"] == "cached_decode", "compute phase")
        require(report["original_prefill_reference_retained"] is True, "original oracle retained")
        require(
            report["model_revision"]
            == prefill["model_revision"]
            == inputs["expected_model_revision"]
            == MODEL_REVISION,
            "official model revision",
        )
        for document in (report, prefill, inputs):
            require(document["serving_core_modules_imported"] == [], "independent HF imports")
        require(report["source_sha256"] == prefill["source_sha256"], "original HF source identity")
        sources = report["source_sha256"]
        for name in ("modeling_moss_speech.py", "configuration_moss_speech.py"):
            require(sources[name] == inputs["source_sha256"][name], "canonical HF source identity")
        require(report["package_versions"] == prefill["package_versions"], "HF package versions")
        require(
            report["checkpoint"]["index_sha256"] == prefill["checkpoint"]["index_sha256"],
            "checkpoint index",
        )
        require(
            report["checkpoint"]["total_parameters"]
            == prefill["checkpoint"]["total_parameters"]
            == 9095694336,
            "checkpoint parameters",
        )
        for document in (report, prefill):
            determinism = document["determinism"]
            require(determinism["seed"] == 20261027, "HF seed")
            require(
                determinism["algorithms_required"] is True and determinism["tf32"] is False,
                "deterministic BF16 policy",
            )
            model = document["model"]
            require(model["parameter_dtypes"] == ["torch.bfloat16"], "BF16 checkpoint")
            require(model["trained_checkpoint_loaded"] is True, "trained checkpoint")
            require(model["parameter_count"] == 9095694336, "model parameter count")
            require(model["attention_implementation"] == "sdpa", "SDPA model")
            require(
                document["requested_profile"] == {"attention_backend": "sdpa", "logits_to_keep": 0},
                "HF compute profile",
            )
            require(
                type(model["logits_to_keep"]) is int and model["logits_to_keep"] == 0,
                "complete head projection",
            )
            require(
                model["text_vocab_size"] == 151680 and model["audio_vocab_size"] == 16512,
                "official vocabulary sizes",
            )
        require(report["model"]["use_cache"] is True, "cached model phase")
        require(prefill["model"]["use_cache"] is False, "retained full-prefill oracle phase")
        pins = report["pinned_source_metadata"]
        inputs_hash = file_hash(inputs_path)
        require(
            report["inputs_json_sha256"]
            == prefill["inputs_json_sha256"]
            == pins["inputs_json_sha256"]
            == inputs_hash,
            "input hash binding",
        )
        require(pins["hf_model_revision"] == MODEL_REVISION, "pinned model revision")
        require(
            pins["original_prefill_reference_json_sha256"] == file_hash(prefill_path),
            "original prefill report binding",
        )
        require(
            pins["original_prefill_tensors_sha256"] == prefill["tensor_artifact_sha256"],
            "original prefill tensor binding",
        )
        require(pins["policy_sha256"] == file_hash(policy_path), "policy binding")
        require(
            policy["absolute_tolerance"] == 0.25 and policy["relative_tolerance"] == 0.02,
            "unchanged numerical tolerances",
        )
        require(
            policy["finite_required"] is True
            and policy["post_observation_tolerance_increase_allowed"] is False,
            "unchanged finite/tolerance policy",
        )
        require(report["tensor_artifact_sha256"] == file_hash(tensor_path), "cached tensor hash")
        origin = report["loaded_model_module"]
        require(Path(origin["file"]).name == "modeling_moss_speech.py", "official loaded model")
        require(
            origin["sha256"] == sources["modeling_moss_speech.py"] == file_hash(origin["file"]),
            "loaded model source hash",
        )
        audited_sources = set()
        for name, module in report["loaded_hf_module_audit"].items():
            require(
                name.startswith(f"transformers_modules.{MODEL_REVISION}."), "HF module namespace"
            )
            path = Path(module["file"])
            require(path.name in sources, "declared HF module source")
            require(module["sha256"] == sources[path.name] == file_hash(path), "HF module hash")
            audited_sources.add(path.name)
        require(
            {"modeling_moss_speech.py", "configuration_moss_speech.py"} <= audited_sources,
            "complete HF module audit",
        )
        require(
            report["case_order"] == prefill["case_order"] == list(CASE_NAMES), "canonical order"
        )
        require(
            set(inputs["cases"])
            == set(report["cases"])
            == set(prefill["cases"])
            == set(CASE_NAMES),
            "eight canonical cases",
        )
        require(
            type(report["tensor_count"]) is int and report["tensor_count"] == 16,
            "sixteen complete heads",
        )
        tensors = {}
        for name in CASE_NAMES:
            canonical, case, original = (
                inputs["cases"][name],
                report["cases"][name],
                prefill["cases"][name],
            )
            grid = canonical["input_ids"]
            require(isinstance(grid, list) and len(grid) >= 2, f"{name} canonical grid")
            require(
                all(
                    isinstance(row, list)
                    and len(row) == 2
                    and all(type(token) is int for token in row)
                    and 0 <= row[0] < 151680
                    and 0 <= row[1] < 16512
                    for row in grid
                ),
                f"{name} canonical tokens",
            )
            length = len(grid)
            require(canonical["attention_mask"] == [1] * length, f"{name} canonical mask")
            for evidence in (case, original):
                require(
                    evidence["input_ids"] == grid
                    and evidence["attention_mask"] == canonical["attention_mask"],
                    f"{name} full input grid",
                )
                require(
                    evidence["output_modality"] == canonical["output_modality"],
                    f"{name} output modality",
                )
                require(evidence["grid_sha256"] == grid_hash(canonical), f"{name} grid hash")
                require(evidence["status"] == "passed", f"{name} status")
            phase = case["compute_phase"]
            require(phase["phase"] == "cached_decode", f"{name} cached phase")
            for field, expected in (
                ("prefix_rows", length - 1),
                ("decode_rows", 1),
                ("cache_position", length - 1),
                ("full_attention_mask_rows", length),
            ):
                require(type(phase[field]) is int and phase[field] == expected, f"{name} {field}")
            for field, expected in (
                ("cache_rows_before", length - 1),
                ("cache_rows_after", length),
            ):
                banks = phase[field]
                require(
                    set(banks) == CACHE_BANKS
                    and all(type(value) is int and value == expected for value in banks.values()),
                    f"{name} all cache banks",
                )
            for evidence, rows in ((case, 1), (original, length)):
                require(set(evidence["heads"]) == set(VOCABULARIES), f"{name} both heads")
                for head, size in VOCABULARIES.items():
                    value = evidence["heads"][head]
                    key = f"{name}.{head}_last_logits"
                    require(value["tensor_key"] == key, f"{name} head tensor key")
                    require(
                        value["forward_shape"] == [1, rows, size]
                        and value["export_shape"] == [size],
                        f"{name} complete head shape",
                    )
                    require(
                        all(
                            type(dimension) is int
                            for dimension in value["forward_shape"] + value["export_shape"]
                        ),
                        f"{name} integer head shape",
                    )
                    require(
                        value["forward_dtype"] == value["export_dtype"] == "torch.bfloat16",
                        f"{name} head dtype",
                    )
                    require(value["all_finite"] is True, f"{name} finite head")
                    tensors[key] = size
        validate_tensors(tensor_path, tensors, inputs_hash)
        return report
    except (KeyError, TypeError, AttributeError, OSError, struct.error, UnicodeError) as error:
        raise ValueError(f"Malformed cached HF reference: {error}") from error
