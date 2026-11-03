"""Compare native full-checkpoint AR logits with a separately exported HF oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from .allocation import require_gpu_allocation
    from .ar_combination import features
    from .batched_rope import validate_qualification as validate_batched_rope
    from .cumulative import COMPUTE_AR_FEATURES
    from .serial_decode_graph import validate_qualification as validate_serial_decode_graph
else:
    from allocation import require_gpu_allocation
    from ar_combination import features
    from batched_rope import validate_qualification as validate_batched_rope
    from cumulative import COMPUTE_AR_FEATURES
    from serial_decode_graph import validate_qualification as validate_serial_decode_graph

PROFILE = {
    "profile": os.environ.get("MOSS_CORRECTNESS_PROFILE"),
    "ar_optimizations": json.loads(os.environ.get("MOSS_AR_OPTIMIZATIONS", "[]")),
    "optimizations": json.loads(os.environ.get("MOSS_OPTIMIZATIONS", "[]")),
}

MODEL_REVISION = "cff025bb41d8459d59abac0b5e44aba7f659ec9e"


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def grid_digest(case: dict) -> str:
    payload = {name: case[name] for name in ("input_ids", "attention_mask")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stream_weights(model_path: Path, schema: dict):
    from safetensors import safe_open

    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(str(model_path / filename), framework="pt", device="cpu") as shard:
            expected = {name for name, source in index["weight_map"].items() if source == filename}
            assert set(shard.keys()) == expected, filename
            for name in shard.keys():
                tensor = shard.get_tensor(name)
                schema[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "shard": filename,
                }
                yield name, tensor


def prefill(model, grid, backend):
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    device = grid.device
    token_count = grid.shape[0]
    text_tokens, audio_tokens = grid[:, 0], grid[:, 1]
    audio_positions = text_tokens == model.config.modality_pad_token_id
    text_embeddings = model.embed_tokens(text_tokens.masked_fill(audio_positions, 0))
    audio_embeddings = model.audio_embed(audio_tokens)
    text_selection = (~audio_positions).unsqueeze(-1).to(text_embeddings.dtype)
    prompt_embeddings = text_embeddings * text_selection + audio_embeddings * (1 - text_selection)
    backend.req_to_token_pool.req_to_token[1, :token_count] = torch.arange(
        1, token_count + 1, device=device
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.zeros(token_count, dtype=torch.long, device=device),
        req_pool_indices=torch.tensor([1], device=device),
        seq_lens=torch.tensor([token_count], dtype=torch.int32, device=device),
        out_cache_loc=torch.arange(1, token_count + 1, device=device),
        seq_lens_sum=token_count,
        positions=torch.arange(token_count, device=device),
        extend_seq_lens=torch.tensor([token_count], dtype=torch.int32, device=device),
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=[token_count],
        extend_prefix_lens_cpu=[0],
        extend_num_tokens=token_count,
        rids=["native-weight-probe"],
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        return model(batch.input_ids, batch.positions, batch, input_embeds=prompt_embeddings)


def compare_logits(actual, expected, policy: dict) -> dict:
    import torch

    actual, expected = actual.float(), expected.float()
    assert actual.ndim == 1 and actual.shape == expected.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    absolute_error = (actual - expected).abs()
    within = torch.isclose(
        actual,
        expected,
        atol=policy["absolute_tolerance"],
        rtol=policy["relative_tolerance"],
    )
    actual_values, actual_ids = actual.topk(10)
    expected_values, expected_ids = expected.topk(10)
    return {
        "finite": finite,
        "allclose": finite and bool(within.all()),
        "outside_tolerance_count": int((~within).sum()),
        "maximum_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "root_mean_squared_error": float(absolute_error.square().mean().sqrt()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)),
        "native_argmax": int(actual.argmax()),
        "hf_argmax": int(expected.argmax()),
        "native_argmax_ties": int((actual == actual.max()).sum()),
        "hf_argmax_ties": int((expected == expected.max()).sum()),
        "top1_equal": int(actual.argmax()) == int(expected.argmax()),
        "topk_first_equal": int(actual_ids[0]) == int(expected_ids[0]),
        "top10_overlap": len(set(actual_ids.tolist()) & set(expected_ids.tolist())),
        "native_top10": {
            "token_ids": actual_ids.tolist(),
            "logits": actual_values.tolist(),
        },
        "hf_top10": {
            "token_ids": expected_ids.tolist(),
            "logits": expected_values.tolist(),
        },
        "hf_top1_margin": float(expected_values[0] - expected_values[1]),
    }


def load_cached_reference(arguments):
    """Load a separately pinned phase-matched oracle; never derive it from native outputs."""
    from safetensors.torch import load_file

    if __package__:
        from .cached_reference import validate_cached_reference
    else:
        from cached_reference import validate_cached_reference

    report_path = Path(os.environ["MOSS_CACHED_HF_REPORT"])
    tensor_path = Path(os.environ["MOSS_CACHED_HF_TENSORS"])
    validate_cached_reference(
        report_path,
        tensor_path,
        arguments.inputs_json,
        arguments.hf_output_dir / "hf-weight-reference.json",
        arguments.policy_json,
    )
    identity = dict(
        report=str(report_path),
        report_sha256=digest(report_path),
        tensors=str(tensor_path),
        tensors_sha256=digest(tensor_path),
        compute_phase="cached_decode",
        original_prefill_reference_retained=True,
    )
    return identity, load_file(str(tensor_path), device="cpu")


def run(
    arguments: argparse.Namespace,
    report: dict,
    vectors: dict,
    report_path: Path,
    tensor_path: Path,
) -> None:
    import torch
    from safetensors.torch import load_file, save_file
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.runtime_context import get_context, get_parallel
    from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig
    from sglang_omni.models.moss_speech.sglang_model import MossSpeechSGLangModel

    device = torch.device(arguments.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(20261027)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(device)
    report["gpu"] = {
        "device": str(device),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report["determinism"] = {
        "seed": 20261027,
        "algorithms_required": True,
        "tf32": False,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "forced_sdp_backend": None,
    }
    model_path = arguments.model_path.resolve()
    configuration = MossSpeechConfig(**json.loads((model_path / "config.json").read_text()))
    assert (
        configuration.num_hidden_layers == 36
        and configuration.num_shared_layers == 32
        and configuration.num_modality_layers == 4
    )
    core_file = Path(inspect.getfile(MossSpeechSGLangModel)).resolve()
    core_repository = core_file.parents[3]
    report["core_head"] = subprocess.check_output(
        ["git", "-C", str(core_repository), "rev-parse", "HEAD"], text=True
    ).strip()
    report["origins"] = {}
    for component in (MossSpeechSGLangModel, MossSpeechConfig, TorchNativeAttnBackend):
        source_path = Path(inspect.getfile(component)).resolve()
        report["origins"][component.__name__] = {
            "file": str(source_path),
            "sha256": digest(source_path),
            "module": component.__module__,
        }
    model_layers_path = core_file.with_name("model_layers.py")
    report["origins"]["model_layers"] = {
        "file": str(model_layers_path),
        "sha256": digest(model_layers_path),
    }
    expected_vectors = load_file(
        str(arguments.hf_output_dir / "hf-weight-logits.safetensors"), device="cpu"
    )
    report["stage"] = "loading-checkpoint"
    write_json(report_path, report)
    started_seconds = time.perf_counter()
    with (
        get_context().override_server_args(
            device=device.type,
            dtype="bfloat16",
            attention_backend="torch_native",
            disable_cuda_graph=True,
            enable_torch_compile=False,
            disable_radix_cache=True,
            tp_size=1,
        ),
        get_parallel().override(
            tp_rank=0,
            pp_rank=0,
            attn_tp_rank=0,
            attn_dp_rank=0,
            attn_cp_rank=0,
            moe_tp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
            tp_group=None,
            attn_tp_group=None,
        ),
        torch.device(device),
        torch.inference_mode(),
    ):
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            # Projection guards require version counters, as in the serving loader.
            with (
                torch.inference_mode(PROFILE["profile"] not in ("ar-combination", "cumulative")),
                torch.no_grad(),
            ):
                model = MossSpeechSGLangModel(configuration, init_device=device).eval()
                assert model.total_attention_layers == 40 and model.attention_layer_ids() == list(
                    range(40)
                )
                schema = {}
                model.load_weights(stream_weights(model_path, schema))
                model.assert_heads_independent()
            if PROFILE["profile"] in ("ar-combination", "cumulative"):
                parameters = dict(model.named_parameters())
                assert parameters and all(not p.is_inference() for p in parameters.values()), (
                    "AR combination requires versioned checkpoint parameters"
                )
                report["weight_versioning"] = dict(
                    all_parameters_versioned=True,
                    parameter_count=len(parameters),
                    versions={name: parameter._version for name, parameter in parameters.items()},
                )
            if "native-norm" in features(PROFILE):
                from sglang_omni.models.moss_speech.model_layers import MossSpeechRMSNorm
                from sglang_omni.models.moss_speech.native_norm import NativeNorm

                native_norm = NativeNorm(os.environ["MOSS_NATIVE_NORM_LIBRARY"])
                for module in model.modules():
                    if isinstance(module, MossSpeechRMSNorm):
                        module.native_norm = native_norm
                report["native_norm_library"] = dict(
                    path=os.environ["MOSS_NATIVE_NORM_LIBRARY"],
                    sha256=digest(Path(os.environ["MOSS_NATIVE_NORM_LIBRARY"])),
                )
            assert len(schema) == 446
            parameter_count = sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name != "decode_input_embedding.weight"
            )
            assert parameter_count == report["checkpoint"]["total_parameters"]
            report["model"] = {
                "parameter_count_excluding_decode_scratch": parameter_count,
                "parameter_dtype": str(next(model.parameters()).dtype),
                "checkpoint_hidden_layers": configuration.num_hidden_layers,
                "allocated_attention_layers": model.total_attention_layers,
                "weight_load_report": model.weight_load_report,
                "source_tensor_schema": schema,
                "trained_checkpoint_loaded": True,
                "attention_backend": "TorchNativeAttnBackend",
                "loading_seconds": time.perf_counter() - started_seconds,
            }
            maximum_rows = max(len(case["input_ids"]) for case in report["cases"].values())
            request_pool = ReqToTokenPool(
                size=1,
                max_context_len=maximum_rows + 1,
                device=str(device),
                enable_memory_saver=False,
            )
            kv_pool = MHATokenToKVPool(
                size=maximum_rows + 1,
                page_size=1,
                dtype=torch.bfloat16,
                head_num=configuration.num_key_value_heads,
                head_dim=configuration.head_dim,
                layer_num=model.total_attention_layers,
                device=str(device),
                enable_memory_saver=False,
                enable_alt_stream=False,
            )
            backend_type = TorchNativeAttnBackend
            if "ar-host-metadata" in features(PROFILE):
                from sglang_omni.models.moss_speech.attention import MossSpeechTorchNativeBackend

                backend_type = MossSpeechTorchNativeBackend
                report["host_metadata_prefills"] = 0
            backend = backend_type(
                SimpleNamespace(
                    device=str(device),
                    req_to_token_pool=request_pool,
                    token_to_kv_pool=kv_pool,
                )
            )
            if PROFILE["profile"] == "ar-serial-decode-graph":
                from tests.unit_test.fixtures.moss_speech_model import assert_serial_decode_graph

                report["serial_decode_graph"] = assert_serial_decode_graph(model, capacity=1024)
                write_json(report_path, report)
                validate_serial_decode_graph(report["serial_decode_graph"])
            if os.environ.get("MOSS_CORRECTNESS_PROFILE") == "ar-decode-graph":
                from tests.unit_test.fixtures.moss_speech_model import assert_full_decode_graph

                report["full_decode_graph"] = assert_full_decode_graph(model, capacity=1024)
            if os.environ.get("MOSS_CORRECTNESS_PROFILE") == "context-prefill-graph":
                from tests.unit_test.fixtures.moss_speech_model import assert_full_prefill_graph

                report["full_prefill_graph"] = assert_full_prefill_graph(
                    model, lengths=tuple(json.loads(os.environ["MOSS_AR_PREFILL_LENGTHS"]))
                )
            if PROFILE["profile"] == "cumulative" and set(features(PROFILE)) & set(
                COMPUTE_AR_FEATURES
            ):
                compute = set(features(PROFILE)) & set(COMPUTE_AR_FEATURES)
                if compute == {"ar-decode-graph", "ar-serial-decode-graph"}:
                    from tests.unit_test.fixtures.moss_speech_model import (
                        assert_serial_decode_graph,
                    )

                    serial = assert_serial_decode_graph(model, capacity=1024)
                    serial_calls = sum(serial["executions"].values())
                    b1_calls = serial["row_replays"] - sum(
                        int(batch) * count for batch, count in serial["executions"].items()
                    )
                    assert b1_calls > 0 and serial_calls > 0
                    report["cumulative_ar"] = dict(
                        optimizations={
                            feature: dict(
                                executions=b1_calls
                                if feature == "ar-decode-graph"
                                else serial_calls,
                                complete_heads_exact=serial["complete_heads_exact"],
                                all_kv_exact=serial["all_kv_exact"],
                            )
                            for feature in compute
                        },
                        complete_heads_exact=serial["complete_heads_exact"],
                        all_kv_exact=serial["all_kv_exact"],
                        joint_optimizations=sorted(compute),
                        joint_complete_heads_exact=serial["complete_heads_exact"],
                        joint_all_kv_exact=serial["all_kv_exact"],
                        serial_decode_graph=serial,
                    )
                else:
                    from tests.unit_test.fixtures.moss_speech_model import assert_cumulative_ar

                    report["cumulative_ar"] = assert_cumulative_ar(
                        model,
                        optimizations=features(PROFILE),
                        projection_batch_size=int(os.environ["MOSS_AR_PROJECTION_BATCH"]),
                        graph_lengths=tuple(json.loads(os.environ["MOSS_AR_PREFILL_LENGTHS"])),
                        native_norm_library=os.environ.get("MOSS_NATIVE_NORM_LIBRARY") or None,
                    )
                if "ar-serial-decode-graph" in features(PROFILE):
                    report["serial_decode_graph"] = report["cumulative_ar"]["serial_decode_graph"]
                    validate_serial_decode_graph(report["serial_decode_graph"])
                write_json(report_path, report)
            if PROFILE["profile"] == "ar-combination":
                from tests.unit_test.fixtures.moss_speech_model import assert_ar_combination

                report["ar_combination"] = assert_ar_combination(
                    model,
                    optimizations=features(PROFILE),
                    projection_batch_size=int(os.environ["MOSS_AR_PROJECTION_BATCH"]),
                    graph_lengths=tuple(json.loads(os.environ["MOSS_AR_PREFILL_LENGTHS"])),
                    native_norm_library=os.environ.get("MOSS_NATIVE_NORM_LIBRARY") or None,
                )
            if "ar-batched-rope" in features(PROFILE):
                from tests.unit_test.fixtures.moss_speech_model import qualify_batched_rope

                grids = {
                    name: torch.tensor(case["input_ids"], dtype=torch.long, device=device)
                    for name, case in report["cases"].items()
                }
                cached_identity, cached_vectors = load_cached_reference(arguments)
                report["cached_decode_reference"] = cached_identity
                proof, outputs = qualify_batched_rope(model, grids)
                report["batched_rope"] = proof
                proof["cases"] = {}
                batched_vectors = {}
                assert set(outputs) == {"2", "4"}
                for batch, cases in outputs.items():
                    assert set(cases) == set(grids)
                    proof["cases"][batch] = {}
                    for name, heads in cases.items():
                        assert set(heads) == {"text", "audio"}
                        proof["cases"][batch][name] = {}
                        for modality, logits in heads.items():
                            assert logits.dtype == torch.bfloat16
                            vector = logits.float().cpu().contiguous()
                            key = f"{name}.{modality}_last_logits"
                            batched_vectors[f"b{batch}.{key}"] = vector
                            proof["cases"][batch][name][modality] = compare_logits(
                                vector, cached_vectors[key], report["policy"]
                            )
                batched_path = tensor_path.with_name("batched-rope-logits.safetensors")
                save_file(batched_vectors, str(batched_path))
                proof["tensor_artifact"] = str(batched_path)
                proof["tensor_artifact_sha256"] = digest(batched_path)
                write_json(report_path, report)
                validate_batched_rope(proof, list(grids))
            report["pool_configuration"] = {
                "request_count": 1,
                "maximum_context_rows": maximum_rows + 1,
                "token_slots": maximum_rows + 1,
                "kv_layers": 40,
                "kv_dtype": "torch.bfloat16",
                "page_size": 1,
            }
            print(
                json.dumps(
                    {
                        "event": "native-checkpoint-loaded",
                        "parameters": parameter_count,
                        "source_tensors": len(schema),
                        "loading_seconds": report["model"]["loading_seconds"],
                    }
                ),
                flush=True,
            )
            report["stage"] = "serial-case-forwards"
            for name in report["case_order"]:
                case = report["cases"][name]
                report["active_case"] = name
                write_json(report_path, report)
                grid = torch.tensor(case["input_ids"], dtype=torch.long, device=device)
                assert grid.ndim == 2 and grid.shape[1] == 2
                assert case["attention_mask"] == [1] * grid.shape[0]
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started_seconds = time.perf_counter()
                output = prefill(model, grid, backend)
                if "ar-host-metadata" in features(PROFILE):
                    assert backend.host_metadata is not None
                    report["host_metadata_prefills"] += 1
                if "ar-prefix-cache" in features(PROFILE):
                    from tests.unit_test.fixtures.moss_speech_prefix_cache import (
                        verify_cached_prefill,
                    )

                    case["prefix_cache"] = verify_cached_prefill(model, grid, output, backend)
                torch.cuda.synchronize(device)
                case["forward_seconds"] = time.perf_counter() - started_seconds
                case["heads"] = {}
                for modality, logits, vocabulary_size in (
                    ("text", output.text_logits, configuration.vocab_size),
                    ("audio", output.audio_logits, configuration.audio_vocab_size),
                ):
                    assert logits is not None and list(logits.shape) == [
                        1,
                        vocabulary_size,
                    ]
                    assert logits.dtype == torch.bfloat16
                    key = f"{name}.{modality}_last_logits"
                    vector = logits[0].float().cpu().contiguous()
                    vectors[key] = vector
                    comparison = compare_logits(vector, expected_vectors[key], report["policy"])
                    case["heads"][modality] = {
                        "tensor_key": key,
                        "native_forward_dtype": str(logits.dtype),
                        "export_dtype": str(vector.dtype),
                        "shape": list(vector.shape),
                        "comparison": comparison,
                    }
                case["status"] = (
                    "passed"
                    if all(head["comparison"]["allclose"] for head in case["heads"].values())
                    else "failed"
                )
                case["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                model.dual_logits_by_request_id.clear()
                save_file(
                    vectors,
                    str(tensor_path),
                    metadata={
                        "scope": "native real-checkpoint last-position raw logits",
                        "policy_sha256": report["policy_sha256"],
                        "inputs_sha256": report["inputs_json_sha256"],
                    },
                )
                report["tensor_artifact_sha256"] = digest(tensor_path)
                write_json(report_path, report)
                print(
                    json.dumps(
                        {
                            "event": "native-case-compared",
                            "case": name,
                            "status": case["status"],
                            "rows": grid.shape[0],
                            "forward_seconds": case["forward_seconds"],
                        }
                    ),
                    flush=True,
                )
                del output, logits, grid
        finally:
            torch.set_default_dtype(original_dtype)
    forbidden_modules = sorted(
        name
        for name in sys.modules
        if name.startswith("transformers_modules.") and "modeling_moss_speech" in name
    )
    report["independent_hf_model_modules_imported"] = forbidden_modules
    assert not forbidden_modules, forbidden_modules
    assert set(vectors) == set(expected_vectors)
    report.pop("active_case", None)
    report["top1_mismatches"] = [
        {"case": name, "head": modality}
        for name, case in report["cases"].items()
        for modality, head in case["heads"].items()
        if not head["comparison"]["top1_equal"]
    ]
    report["active_top1_mismatches"] = [
        mismatch
        for mismatch in report["top1_mismatches"]
        if mismatch["head"] == report["cases"][mismatch["case"]]["output_modality"]
    ]
    report["inactive_top1_mismatches"] = [
        mismatch
        for mismatch in report["top1_mismatches"]
        if mismatch["head"] != report["cases"][mismatch["case"]]["output_modality"]
    ]
    report["greedy_equivalence"] = (
        "unresolved: active-head first-index argmax differences"
        if report["active_top1_mismatches"]
        else "active-head first-index argmax agrees at all exported positions; generation not tested"
    )
    report["status"] = (
        "passed"
        if all(case["status"] == "passed" for case in report["cases"].values())
        else "failed"
    )
    report["stage"] = "complete"


def timeout_handler(signum, frame) -> None:
    raise TimeoutError("Native real-checkpoint comparison exceeded its wall-clock budget")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--inputs-json", type=Path, required=True)
    parser.add_argument("--hf-output-dir", type=Path, required=True)
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=Path(__file__).with_name("core-weight-policy.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=int, default=840)
    arguments = parser.parse_args()
    if not 1 <= arguments.max_seconds <= 840:
        parser.error("max-seconds must be 1..840")
    else:
        pass
    require_gpu_allocation()
    assert os.environ.get("SGLANG_USE_CPU_ENGINE") != "1", (
        "Native GPU platform must be selected before imports"
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = arguments.output_dir / "core-weight-reference.json"
    tensor_path = arguments.output_dir / "core-weight-logits.safetensors"
    assert not report_path.exists() and not tensor_path.exists(), "Choose a fresh output directory"
    inputs = json.loads(arguments.inputs_json.read_text())
    reference_path = arguments.hf_output_dir / "hf-weight-reference.json"
    reference = json.loads(reference_path.read_text())
    policy = json.loads(arguments.policy_json.read_text())
    assert policy["registered_before_weight_execution"] and policy["finite_required"]
    assert policy["absolute_tolerance"] == 0.25 and policy["relative_tolerance"] == 0.02
    assert inputs["status"] == reference["status"] == "passed"
    assert inputs["expected_model_revision"] == reference["model_revision"] == MODEL_REVISION
    assert reference["inputs_json_sha256"] == digest(arguments.inputs_json)
    assert not reference["serving_core_modules_imported"]
    assert reference["tensor_artifact_sha256"] == digest(
        arguments.hf_output_dir / "hf-weight-logits.safetensors"
    )
    model_path = arguments.model_path.resolve()
    assert json.loads((model_path / "config.json").read_text())["model_type"] == "moss_speech"
    index_path = model_path / "model.safetensors.index.json"
    assert digest(index_path) == reference["checkpoint"]["index_sha256"]
    index = json.loads(index_path.read_text())
    source_hashes = {
        filename: digest(model_path / filename) for filename in reference["source_sha256"]
    }
    assert source_hashes == reference["source_sha256"]
    cases = {}
    for name in reference["case_order"]:
        case = inputs["cases"][name]
        assert grid_digest(case) == reference["cases"][name]["grid_sha256"]
        cases[name] = {
            "input_ids": case["input_ids"],
            "attention_mask": case["attention_mask"],
            "grid_sha256": grid_digest(case),
            "output_modality": case["output_modality"],
            "status": "pending",
        }
    assert len(cases) == 8
    shards = [
        {
            "file": filename,
            "bytes": (model_path / filename).stat().st_size,
            "resolved_blob_id": (model_path / filename).resolve().name,
        }
        for filename in sorted(set(index["weight_map"].values()))
    ]
    versions = {
        package: importlib.metadata.version(package)
        for package in ("torch", "sglang", "transformers", "safetensors", "numpy")
    }
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stage": "initializing",
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "local_gpu_devices": os.environ.get("MOSS_LOCAL_GPU_DEVICES"),
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "inputs_json_sha256": digest(arguments.inputs_json),
        "hf_reference_sha256": digest(reference_path),
        "hf_tensor_sha256": reference["tensor_artifact_sha256"],
        "policy": policy,
        "policy_sha256": digest(arguments.policy_json),
        "probe_sha256": digest(Path(__file__)),
        "source_sha256": source_hashes,
        "case_order": list(cases),
        "cases": cases,
        "checkpoint": {
            "index_sha256": digest(index_path),
            "total_parameters": index["metadata"]["total_parameters"],
            "index_tensor_count": len(index["weight_map"]),
            "shards": shards,
            "full_shard_hashes_recomputed": False,
        },
        "limitations": [
            "Known audio codes in teacher-forced grids do not validate audio encoding or generation.",
            "One-request manual attention/KV setup uses the actual model and loader; it does not exercise scheduler admission, service transport, codec or output quality.",
            "BF16 elementwise tolerances are a preregistered engineering gate; raw top1 mismatches remain unresolved for greedy equivalence.",
            "Pinned snapshot/index/source hashes and resolved cache-blob identities are retained; complete multi-GB shard hashes are not recomputed.",
        ],
    }
    write_json(report_path, report)
    vectors = {}
    started_seconds = time.perf_counter()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(arguments.max_seconds)
    try:
        run(arguments, report, vectors, report_path, tensor_path)
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        if report.get("active_case"):
            report["cases"][report["active_case"]]["status"] = "failed"
        else:
            pass
    finally:
        signal.alarm(0)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["total_seconds"] = time.perf_counter() - started_seconds
        report["tensor_count"] = len(vectors)
        write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases_passed": sum(
                    case["status"] == "passed" for case in report["cases"].values()
                ),
                "tensor_count": len(vectors),
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
