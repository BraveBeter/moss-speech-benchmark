"""Compare production grid formatting with a separately exported official HF oracle.

Audio codes are supplied inputs. This probe never calls or validates an encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sglang_omni.models.moss_speech.components.processor import (
    DEFAULT_SYSTEM_PROMPTS,
    MossSpeechGridProcessor,
)
from transformers import AutoTokenizer

JsonValue = Any

CORE = Path(os.environ["MOSS_CORE_ROOT"]).resolve()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(
    expected: JsonValue, actual: JsonValue, path: str = "$"
) -> list[dict[str, JsonValue]]:
    if type(expected) is not type(actual):
        return [{"path": path, "expected": expected, "actual": actual}]
    elif isinstance(expected, list) and isinstance(actual, list):
        found = []
        if len(expected) != len(actual):
            found.append(
                {
                    "path": f"{path}.length",
                    "expected": len(expected),
                    "actual": len(actual),
                }
            )
        else:
            pass
        for index, (left, right) in enumerate(zip(expected, actual)):
            found.extend(differences(left, right, f"{path}[{index}]"))
        return found
    elif isinstance(expected, dict) and isinstance(actual, dict):
        found = []
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                found.append(
                    {
                        "path": f"{path}.{key}",
                        "expected": expected.get(key),
                        "actual": actual.get(key),
                    }
                )
            else:
                found.extend(differences(expected[key], actual[key], f"{path}.{key}"))
        return found
    elif expected != actual:
        return [{"path": path, "expected": expected, "actual": actual}]
    else:
        return []


def comparison(expected: JsonValue, actual: JsonValue) -> dict[str, JsonValue]:
    found = differences(expected, actual)
    return {
        "equal": not found,
        "difference_count": len(found),
        "first_differences": found[:50],
    }


def compare_core(reference: dict, result: dict) -> None:
    assert reference["status"] == "passed" and reference["processor"]["status"] == "passed"
    assert reference["serving_core_modules_imported"] == []
    assert len(reference["cases"]) == 8
    assert os.environ.get("SGLANG_USE_CPU_ENGINE") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("PYTHONPATH") == str(CORE)
    model_path = Path(os.environ["MOSS_MODEL_PATH"])
    processor_path = Path(inspect.getfile(MossSpeechGridProcessor)).resolve()
    assert processor_path == CORE / "sglang_omni/models/moss_speech/components/processor.py"
    fixture_path = CORE / "tests/fixtures/moss_speech/canonical_inputs.json"
    assert digest(fixture_path) == reference["fixture_sha256"]
    fixture = json.loads(fixture_path.read_text())
    asset_hashes = {name: digest(model_path / name) for name in reference["source_sha256"]}
    assert asset_hashes == reference["source_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    processor = MossSpeechGridProcessor(tokenizer)
    tokenizer_module = importlib.import_module(type(tokenizer).__module__)
    result["origins"] = {
        "processor_class": f"{MossSpeechGridProcessor.__module__}.{MossSpeechGridProcessor.__name__}",
        "processor_path": str(processor_path),
        "processor_sha256": digest(processor_path),
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "tokenizer_module_path": tokenizer_module.__file__,
        "asset_sha256": asset_hashes,
        "fixture_sha256": digest(fixture_path),
    }
    result["defaults_comparison"] = comparison(
        reference["processor"]["default_system_prompts"], DEFAULT_SYSTEM_PROMPTS
    )
    result["pad_id_comparison"] = comparison(
        reference["processor"]["tokenizer_pad_token_id"], tokenizer.pad_token_id
    )
    actual_text = {
        text: tokenizer.encode(text, add_special_tokens=False)
        for text in reference["text_encodings"]
    }
    result["text_encodings"] = actual_text
    result["text_comparisons"] = {
        text: comparison(expected, actual_text[text])
        for text, expected in reference["text_encodings"].items()
    }
    result["canonical_text_comparisons"] = {
        text: comparison(expected, actual_text[text])
        for text, expected in fixture["text_encodings"].items()
    }
    result["cases"] = {}
    samples = []
    case_order = reference["batch"]["case_order"]
    assert set(case_order) == set(reference["cases"]) and len(case_order) == len(set(case_order))
    for name in case_order:
        case = reference["cases"][name]
        sample = processor.build(case["turns"], case["audio_codes"], case["output_modality"])
        record = {
            "source": case["source"],
            "output_modality": case["output_modality"],
            "audio_code_input_count": len(case["audio_codes"]),
            "audio_evidence": "Supplied known codes; grid formatting only, no encoder invoked or validated.",
            "input_ids": sample.grid[0].tolist(),
            "attention_mask": sample.attention_mask[0].tolist(),
            "prompt_len": sample.prompt_len,
        }
        record["input_ids_comparison"] = comparison(case["input_ids"], record["input_ids"])
        record["attention_mask_comparison"] = comparison(
            case["attention_mask"], record["attention_mask"]
        )
        record["prompt_len_comparison"] = comparison(len(case["input_ids"]), sample.prompt_len)
        result["cases"][name] = record
        samples.append(sample)
    original_grids = [sample.grid.clone() for sample in samples]
    original_masks = [sample.attention_mask.clone() for sample in samples]
    batch = processor.collate(samples)
    result["batch"] = {
        "case_order": case_order,
        "input_ids": batch.grid.tolist(),
        "attention_mask": batch.attention_mask.tolist(),
        "shape": list(batch.grid.shape),
        "padding_side": "left",
        "prompt_len": batch.prompt_len,
    }
    result["batch_comparisons"] = {
        key: comparison(reference["batch"][key], result["batch"][key])
        for key in (
            "case_order",
            "input_ids",
            "attention_mask",
            "shape",
            "padding_side",
        )
    }
    result["left_padding_comparisons"] = {}
    for index, name in enumerate(case_order):
        sample = samples[index]
        padding_length = batch.prompt_len - sample.prompt_len
        expected_grid = [[tokenizer.pad_token_id, 512]] * padding_length + original_grids[index][
            0
        ].tolist()
        expected_mask = [0] * padding_length + original_masks[index][0].tolist()
        result["left_padding_comparisons"][name] = {
            "padding_rows": padding_length,
            "grid": comparison(expected_grid, batch.grid[index].tolist()),
            "mask": comparison(expected_mask, batch.attention_mask[index].tolist()),
            "sample_grid_unchanged": torch.equal(sample.grid, original_grids[index]),
            "sample_mask_unchanged": torch.equal(sample.attention_mask, original_masks[index]),
        }
    checks = [result["defaults_comparison"], result["pad_id_comparison"]]
    checks.extend(result["text_comparisons"].values())
    checks.extend(result["canonical_text_comparisons"].values())
    checks.extend(
        record[key]
        for record in result["cases"].values()
        for key in (
            "input_ids_comparison",
            "attention_mask_comparison",
            "prompt_len_comparison",
        )
    )
    checks.extend(result["batch_comparisons"].values())
    checks.extend(
        record[key]
        for record in result["left_padding_comparisons"].values()
        for key in ("grid", "mask")
    )
    if os.environ.get("MOSS_CONTEXT_REMAINING") == "1":
        probes = []
        for target_length in (40959, 40960):
            repetitions = target_length
            for attempt in range(5):
                text = " hello" * repetitions
                turns = [dict(role="user", kind="text", text=text)]
                canonical = processor.build(turns, [], "text")
                if canonical.prompt_len == target_length:
                    break
                repetitions += target_length - canonical.prompt_len
            assert canonical.prompt_len == target_length, "Cannot construct exact context probe"
            probes.append(
                dict(
                    text=text,
                    prompt_tokens=canonical.prompt_len,
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        result["context_probes"] = probes
    result["comparison_count"] = len(checks)
    result["failed_comparison_count"] = sum(not check["equal"] for check in checks)
    result["samples_unchanged"] = all(
        record["sample_grid_unchanged"] and record["sample_mask_unchanged"]
        for record in result["left_padding_comparisons"].values()
    )
    result["cuda_initialized"] = torch.cuda.is_initialized()
    result["official_processor_modules_imported"] = sorted(
        name
        for name in sys.modules
        if name.startswith("transformers_modules.") and name.endswith("processing_moss_speech")
    )
    assert not result["official_processor_modules_imported"], (
        "Core comparison must not import the HF processor implementation"
    )
    assert not result["cuda_initialized"]
    assert result["samples_unchanged"]
    assert result["failed_comparison_count"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_json.exists(), "Use a fresh output path; retain prior evidence"
    reference = json.loads(args.reference_json.read_text())
    started = time.perf_counter()
    result = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "cwd": str(Path.cwd()),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "numpy",
                "sglang",
                "sglang-omni",
            )
        },
        "core_commit": subprocess.check_output(
            ["git", "-C", str(CORE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "core_worktree_status": subprocess.check_output(
            ["git", "-C", str(CORE), "status", "--porcelain"], text=True
        ),
        "reference_json": str(args.reference_json.resolve()),
        "reference_sha256": digest(args.reference_json),
        "reference_environment_versions": reference["package_versions"],
        "model_revision": reference["expected_model_revision"],
        "gpu_used": False,
        "limitations": [
            "Pure token-grid/tokenizer/collation parity against a separate official HF process only.",
            "Known audio codes are supplied; no audio encoder, waveform, codec weights, trained-model inference or quality validation.",
            "No tiny-model numerical parity is claimed by this comparison.",
        ],
    }
    try:
        compare_core(reference, result)
        result["status"] = "passed"
    except Exception as error:
        result.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result.get("cases", {})),
                "comparison_count": result.get("comparison_count"),
                "failed_comparison_count": result.get("failed_comparison_count"),
                "output": str(args.output_json),
            }
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
