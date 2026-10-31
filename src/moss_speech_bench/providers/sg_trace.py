"""Explicit terminal-only observation factories for isolated SGLang diagnosis.

Import model dependencies only in the serving worker. No sampling or model method
is replaced; these factories retain terminal grids after normal computation.
"""

import json
import os
import re
from dataclasses import asdict
from pathlib import Path


def write_trace(request_id, stage, record):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", request_id):
        raise ValueError("Invalid trace request ID")
    directory = Path(os.environ["MOSS_TRACE_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{request_id}.{stage}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(request_id=request_id, **record), ensure_ascii=False) + "\n"
    )
    temporary.replace(path)


def create_ar_engine_executor(
    model_path,
    *,
    dtype,
    context_length,
    device=None,
    gpu_id=None,
    total_gpu_memory_fraction=None,
    process_total_gpu_memory_fraction=None,
    server_args_overrides=None,
):
    from sglang_omni.models.moss_speech.request_builders import state_from_payload
    from sglang_omni.models.moss_speech.stages import create_ar_engine_executor as original

    scheduler = original(
        model_path,
        dtype=dtype,
        context_length=context_length,
        device=device,
        gpu_id=gpu_id,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        process_total_gpu_memory_fraction=process_total_gpu_memory_fraction,
        server_args_overrides=server_args_overrides,
    )
    adapt = scheduler.result_adapter

    def result_adapter(request_data):
        parameters = asdict(request_data.params)
        stop_reason = request_data.stop_reason
        result = adapt(request_data)
        state = state_from_payload(result)
        write_trace(
            result.request_id,
            "ar",
            dict(
                input_grid=state.input_grid,
                attention_mask=state.attention_mask,
                output_grid=state.output_grid,
                output_modality=state.output_modality,
                finish_reason=state.finish_reason,
                stop_reason=stop_reason,
                sampling=parameters,
                seed=state.effective_seed,
            ),
        )
        return result

    scheduler.result_adapter = result_adapter
    return scheduler


def create_text_decode_executor(model_path):
    from sglang_omni.models.moss_speech.request_builders import state_from_payload
    from sglang_omni.models.moss_speech.stages import load_tokenizer
    from sglang_omni.models.moss_speech.streaming import MossSpeechStreamingText

    class TracedText(MossSpeechStreamingText):
        def compute(self, payload):
            result = super().compute(payload)
            state = state_from_payload(payload)
            text_tokens = [row[0] for row in state.output_grid]
            write_trace(
                payload.request_id,
                "text",
                dict(
                    output_grid=state.output_grid,
                    raw_decode=self.tokenizer.decode(text_tokens, skip_special_tokens=False),
                    visible_decode=self.tokenizer.decode(text_tokens, skip_special_tokens=True),
                    token_strings=self.tokenizer.convert_ids_to_tokens(text_tokens),
                    final_text=result.data["text"],
                ),
            )
            return result

    return TracedText(load_tokenizer(model_path))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Diagnostic serving requires Slurm"
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.serve.launcher import launch_server

    config = ConfigManager.from_file(args.config).config
    before = config.model_dump()
    factories = {
        "ar_engine": "moss_speech_bench.providers.sg_trace.create_ar_engine_executor",
        "text_decode": "moss_speech_bench.providers.sg_trace.create_text_decode_executor",
    }
    for stage in config.stages:
        if stage.name in factories:
            stage.factory_path = factories[stage.name]
    after = config.model_dump()
    write_trace("deployment", "topology", dict(before=before, after=after))
    launch_server(config, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
