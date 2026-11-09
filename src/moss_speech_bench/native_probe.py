"""Versioned native MOSS adapter using production factories, not model tests."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang_omni.models.moss_speech.codec import MossSpeechCodec
from sglang_omni.models.moss_speech.fsm import row_selected_token
from sglang_omni.models.moss_speech.hf_config import ensure_moss_speech_config_registered
from sglang_omni.models.moss_speech.request_builders import MossSpeechState, preprocess_request
from sglang_omni.models.moss_speech.stages import create_ar_engine_executor
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from transformers import AutoTokenizer

from moss_speech_bench.noise_tape import NoiseTape


def payload(config, case, tokenizer, request_id, budget):
    return preprocess_request(
        StagePayload(
            request_id,
            OmniRequest(
                inputs=case["text"],
                params=dict(
                    seed=config["sampling"]["seed"],
                    temperature=0,
                    top_k=1,
                    repetition_penalty=config["sampling"]["repetition_penalty"],
                    max_new_tokens=budget,
                ),
                metadata={"tts_params": {"ref_audio": config["voice_wav"]}},
            ),
            {},
        ),
        tokenizer,
    )


@torch.inference_mode()
def ar(config, output):
    ensure_moss_speech_config_registered()
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=False)
    scheduler = create_ar_engine_executor(
        config["model_path"],
        device="cuda",
        gpu_id=0,
        server_args_overrides=config["runtime"],
        total_gpu_memory_fraction=config["gpu_memory_fraction"],
    )
    runner = scheduler.model_runner
    actual = resolved_view(scheduler.server_args)
    tensors, bindings, observations, replays, failures = {}, {}, [], [], []
    original_sample = runner.sample_next_token_ids
    group = ""
    graph = runner.tp_worker.model_runner.decode_cuda_graph_runner
    if graph is not None:
        original_replay = graph.execute

        def replay(forward_batch, *args, **kwargs):
            replays.append(dict(group=group, batch_size=forward_batch.batch_size))
            return original_replay(forward_batch, *args, **kwargs)

        graph.execute = replay

    def sample(logits, forward_batch, schedule_batch, requests):
        observations.append(
            dict(group=group, decode=forward_batch.forward_mode.is_decode(), active=len(requests))
        )
        for index, request in enumerate(requests):
            bound = bindings[request.request_id]
            step = len(request.data.output_rows)
            for head in ("text", "audio"):
                tensors[f"{group}/{bound['case_id']}/{step}/{head}"] = (
                    getattr(logits, head + "_logits")[index].detach().float().cpu().contiguous()
                )
        selected = original_sample(logits, forward_batch, schedule_batch, requests)
        for index, request in enumerate(requests):
            bound = bindings[request.request_id]
            native = request.data
            step = len(native.output_rows) - 1
            if bound["teacher"] and step < len(config["teacher_rows"]):
                row = tuple(config["teacher_rows"][step])
                native.output_rows[-1] = row
                native.pending_feedback_queue[-1] = runner.rows_to_embeddings(
                    torch.tensor([row], dtype=torch.long)
                )[0].detach()
                native.finished = False
                native.stop_reason = None
                selected[index] = row_selected_token(row)
            bound["rows"] = [list(row) for row in native.output_rows]
        return selected

    runner.sample_next_token_ids = sample

    def serve():
        try:
            with torch.inference_mode():
                scheduler.start()
        except BaseException as error:
            failures.append(repr(error))

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    cases = []
    try:
        for kind in ("teacher", "greedy"):
            sizes = config["batch_sizes"] if kind == "teacher" else [1, max(config["batch_sizes"])]
            for size in sizes:
                group = f"{kind}-b{size}"
                budget = (
                    len(config["teacher_rows"]) + 1 if kind == "teacher" else config["greedy_steps"]
                )
                for start in range(0, len(config["cases"]), size):
                    cohort = config["cases"][start : start + size]
                    pending = set()
                    for case in cohort:
                        request_id = f"{group}-{case['id']}"
                        prepared = payload(config, case, tokenizer, request_id, budget)
                        state = MossSpeechState.from_dict(prepared.data)
                        bindings[request_id] = dict(
                            case_id=case["id"],
                            teacher=kind == "teacher",
                            prompt_rows=torch.stack(
                                [
                                    torch.tensor(state.text_token_ids),
                                    torch.full((state.prompt_token_count,), 512),
                                ],
                                dim=1,
                            ).tolist(),
                        )
                        pending.add(request_id)
                        scheduler.inbox.put(IncomingMessage(request_id, "new_request", prepared))
                    deadline = time.monotonic() + config.get("worker_timeout", 600)
                    while pending:
                        if failures:
                            raise RuntimeError("Native scheduler failed: " + str(failures))
                        if time.monotonic() > deadline:
                            raise TimeoutError("Native scheduler cohort timed out")
                        try:
                            message = scheduler.outbox.get(timeout=1)
                        except queue.Empty:
                            continue
                        if message.type == "error":
                            raise RuntimeError(str(message.data))
                        if message.type == "result":
                            assert message.request_id in pending
                            pending.remove(message.request_id)
                            cases.append(dict(group=group, **bindings[message.request_id]))
    finally:
        scheduler.stop()
        thread.join(timeout=20)
        assert not thread.is_alive(), "Native scheduler did not stop"
    save_file(tensors, str(output / "ar.safetensors"))
    return dict(
        cases=cases,
        observations=observations,
        graph_replays=replays,
        runtime=dict(
            attention_backend=actual.attention_backend,
            disable_cuda_graph=actual.disable_cuda_graph,
            dtype=actual.dtype,
            max_running_requests=actual.max_running_requests,
        ),
    )


@torch.inference_mode()
def codec(config, output):
    source = Path(config["hf_source_root"])
    sys.path[:0] = [str(source), str(source / "Matcha-TTS")]
    codec = MossSpeechCodec(config["codec_path"], torch.device("cuda:0"))
    try:
        voice = codec.encode_reference(config["voice_wav"])
        tensors = {
            "voice/" + key: (
                value.detach().cpu().long() if key == "prompt_token" else value.detach().cpu()
            ).contiguous()
            for key, value in voice.items()
        }
        reference_root = Path(config["reference"]).parent
        frozen = load_file(reference_root / "codec.safetensors")
        noise = codec.decoder.flow.decoder.rand_noise
        assert noise.shape == frozen["noise/flow_initial"].shape
        assert noise.dtype == frozen["noise/flow_initial"].dtype
        codec.decoder.flow.decoder.rand_noise = (
            frozen["noise/flow_initial"].to(noise.device).clone()
        )
        noise_calls = json.loads((reference_root / "codec.json").read_text())["noise_calls"]
        reference_voice = {
            key.removeprefix("voice/"): value.to("cuda")
            for key, value in frozen.items()
            if key.startswith("voice/")
        }
        for count in config["codec_code_counts"]:
            random.seed(config["sampling"]["seed"])
            np.random.seed(config["sampling"]["seed"])
            torch.manual_seed(config["sampling"]["seed"])
            codes = [config["codec_codes"][i % len(config["codec_codes"])] for i in range(count)]
            with NoiseTape(frozen, str(count), expected=noise_calls[str(count)]):
                waveform, _ = codec.decoder.token2wav(
                    torch.tensor([codes], device="cuda"),
                    uuid=f"native-{count}",
                    **reference_voice,
                    finalize=True,
                )
            tensors[f"pcm/{count}"] = waveform.detach().cpu().float().reshape(-1).contiguous()
        save_file(tensors, str(output / "codec.safetensors"))
        return dict(
            sample_rate=24000,
            dtype=str(codec.decoder.flow.dtype)
            if hasattr(codec.decoder.flow, "dtype")
            else "float32",
        )
    finally:
        codec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("ar", "codec"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result = (ar if args.phase == "ar" else codec)(config, args.output)
    origins = {
        name: str(Path(inspect.getfile(module)).resolve())
        for name, module in list(sys.modules.items())
        if name.startswith("sglang_omni") and getattr(module, "__file__", None)
    }
    assert origins and all(
        Path(config["core_root"]).resolve() in Path(path).parents for path in origins.values()
    )
    result.update(
        status="complete",
        origins=origins,
        torch=torch.__version__,
        sglang=importlib.metadata.version("sglang"),
        adapter="sglang-native-moss-v1",
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / (args.phase + ".json")).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
