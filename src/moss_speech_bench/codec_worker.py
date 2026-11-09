"""Execute only Codec or local scoring APIs in the explicitly selected environment."""

import argparse
import importlib.metadata
import inspect
import os
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from moss_speech_bench.codec import file_hash, summarize_quality
from moss_speech_bench.storage import digest, read_json, source_identity, write_json


def infer(config, root):
    import numpy as np
    import soundfile as sf
    import torch

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = dict(
        status="running",
        ar_executed=False,
        backend=config["backend"],
        physical_gpu=config["physical_gpu"],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        decoder_mode="offline",
        rows=[],
        loaded_sources={},
    )
    write_json(root / "generation.json", report)
    scopes = []
    with ExitStack() as contexts:
        if config["backend"] == "hf-native":
            source = Path(config["hf_source_root"])
            sys.path[:0] = [str(source), str(source / "Matcha-TTS")]
            from transformers import AutoModel

            codec = (
                AutoModel.from_pretrained(
                    config["codec_path"], trust_remote_code=True, local_files_only=True
                )
                .to("cuda:0")
                .eval()
            )
            actual = Path(inspect.getfile(type(codec)))
            expected = Path(config["codec_path"]) / "modeling_moss_speech_codec.py"
            assert file_hash(actual) == file_hash(expected), "HF Codec source was changed"
            report["loaded_sources"]["hf_codec"] = dict(path=str(actual), sha256=file_hash(actual))
            assert all(parameter.dtype == torch.float32 for parameter in codec.parameters())
        else:
            sys.path.insert(0, config["model_root"])
            from sglang_omni.models.moss_speech.components.codec_adapter import (
                MossSpeechCodecAdapter,
            )

            codec = contexts.enter_context(
                MossSpeechCodecAdapter(
                    config["codec_path"], device="cuda:0", broadcast_attention_bias=True
                )
            )
            assert codec.decoder is not None
            if config["backend"] == "sg-bf16":
                from sglang_omni.models.moss_speech.components.flow_precision import FlowBF16Scope

                for stage in ("ffn", "attention"):
                    options = (
                        dict(bf16_sdpa=True)
                        if stage == "attention" and config["flow_precision"] == "ffn_attention_sdpa"
                        else {}
                    )
                    scopes.append(
                        contexts.enter_context(
                            FlowBF16Scope(codec.decoder.flow.decoder.estimator, stage, **options)
                        )
                    )
            report["loaded_sources"]["sg_codec"] = dict(
                path=inspect.getfile(type(codec)), sha256=file_hash(inspect.getfile(type(codec)))
            )
        report["transformers"] = importlib.metadata.version("transformers")
        report["inference_training"] = (
            bool(codec.training)
            if config["backend"] == "hf-native"
            else bool(codec.decoder.training)
        )
        assert not report["inference_training"]
        for sample in read_json(root / "samples.json"):
            row = dict(id=sample["id"], status="failed", language=sample["language"])
            try:
                random.seed(config["seed"])
                np.random.seed(config["seed"])
                torch.manual_seed(config["seed"])
                torch.cuda.manual_seed_all(config["seed"])
                started = time.monotonic()
                codes = codec.encode([sample["target_audio"]], batch_size=1)[0]
                prompt_codes = codec.encode([sample["prompt_audio"]], batch_size=1)[0]
                assert codes and all(type(code) is int and 0 <= code < 16384 for code in codes)
                write_json(
                    root / "codes" / (sample["id"] + ".json"),
                    dict(target=codes, prompt=prompt_codes),
                )
                row["codes_sha256"] = digest(codes)
                row["prompt_codes_sha256"] = digest(prompt_codes)
                if config["backend"] == "hf-native":
                    waveform = codec.decode(
                        torch.tensor(codes, device="cuda").reshape(1, 1, -1),
                        prompt_speech=sample["prompt_audio"],
                    )["syn_wav_list"][0]
                    assert not any(
                        name == "sglang_omni" or name.startswith("sglang_omni.")
                        for name in sys.modules
                    )
                else:
                    voice = codec.encode_voice_ref(sample["prompt_audio"])
                    _, waveform = codec.decode(codes, voice, request_id=sample["id"])
                waveform = waveform.detach().cpu().float().reshape(-1)
                torch.cuda.synchronize()
                row.update(
                    code_count=len(codes),
                    sample_count=waveform.numel(),
                    sample_rate=24000,
                    finite=bool(torch.isfinite(waveform).all()),
                    local_codec_seconds=time.monotonic() - started,
                )
                path = root / "audio" / (sample["id"] + ".wav")
                path.parent.mkdir(exist_ok=True)
                sf.write(path, waveform.numpy(), 24000, subtype="FLOAT")
                row.update(wav_path=str(path.relative_to(root)), wav_sha256=file_hash(path))
                assert row["finite"] and row["sample_count"] == len(codes) * 1920
                row["status"] = "complete"
            except Exception as error:
                row["error"] = type(error).__name__ + ": " + str(error)
            report["rows"].append(row)
            write_json(root / "generation.json", report)
            print(sample["id"], row["status"], flush=True)
        report["bf16_executions"] = sum(
            operator.forward.bf16_executions for scope in scopes for operator in scope.operators
        )
        if scopes:
            assert all(
                operator.forward.executions == operator.forward.bf16_executions > 0
                for scope in scopes
                for operator in scope.operators
            )
            assert all(parameter.dtype == torch.float32 for parameter in codec.decoder.parameters())
        if config.get("flow_precision") == "ffn_attention_sdpa":
            processors = [processor for scope in scopes for processor in scope.sdpa_processors]
            assert processors and all(
                processor.bf16_sdpa_calls == processor.broadcast_calls > 0
                and processor.fallback_calls == 0
                and processor.last_head_stride == 0
                and processor.last_sdpa_dtype is torch.bfloat16
                for processor in processors
            )
            report["bf16_sdpa_executions"] = sum(p.bf16_sdpa_calls for p in processors)
            report["bf16_sdpa_mask_broadcast"] = True
        report["status"] = (
            "complete" if all(row["status"] == "complete" for row in report["rows"]) else "partial"
        )
        write_json(root / "generation.json", report)
    return report


def score(config, root):
    from moss_speech_bench._vendor.asr_scoring import apply_wer
    from moss_speech_bench.quality import LocalScorers, configuration

    settings, contract = configuration(root / "execution/score-config.json")
    settings["device"] = "cuda:0"
    backend = LocalScorers(settings)
    samples = read_json(root / "samples.json")
    generation = {row["id"]: row for row in read_json(root / "generation.json")["rows"]}
    report = dict(
        status="running",
        protocol="moss-codec-only-v1",
        scoring_contract=contract,
        client=source_identity(),
        physical_gpu=config["physical_gpu"],
        sim_scale="cosine_times_100",
        english_decoding=dict(language="en", fp16=False, temperature=0.0, beam_size=5),
        rows=[],
    )
    write_json(root / "quality.json", report)
    for sample in samples:
        generated = generation[sample["id"]]
        metric = "wer" if sample["language"] == "en" else "cer"
        row = dict(
            id=sample["id"],
            language=sample["language"],
            prompt_sha256=sample["prompt_sha256"],
            metrics={},
        )
        path = root / generated["wav_path"]
        assert generated["status"] == "complete" and file_hash(path) == generated["wav_sha256"]
        try:
            if sample["language"] == "en":
                transcript = backend.model("asr_en").transcribe(
                    str(path), language="en", fp16=False, temperature=0.0, beam_size=5
                )["text"]
            else:
                transcript = backend.transcribe(path, sample["language"])
            output = apply_wer(
                SimpleNamespace(target_text=sample["text"], is_success=False),
                transcript,
                sample["language"],
            )
            if not output.is_success:
                raise ValueError("Empty normalized reference")
            row["metrics"][metric] = dict(
                status="scored",
                value={
                    key: vars(output)[key]
                    for key in ["wer", "substitutions", "deletions", "insertions", "hits"]
                },
            )
            row.update(
                transcript=transcript,
                reference_text=sample["text"],
                normalized_reference=output.ref_norm,
                normalized_hypothesis=output.hyp_norm,
            )
        except Exception as error:
            row["metrics"][metric] = dict(
                status="failed", reason=type(error).__name__ + ": " + str(error)
            )
        try:
            row["metrics"]["sim"] = dict(
                status="scored", value=100 * backend.sim(sample["prompt_audio"], path)
            )
        except Exception as error:
            row["metrics"]["sim"] = dict(
                status="failed", reason=type(error).__name__ + ": " + str(error)
            )
        report["rows"].append(row)
        write_json(root / "quality.json", report)
        print(
            sample["id"],
            {key: value["status"] for key, value in row["metrics"].items()},
            flush=True,
        )
    report["aggregate"] = summarize_quality(report["rows"])
    report["status"] = (
        "complete"
        if all(
            value["status"] == "scored"
            for row in report["rows"]
            for value in row["metrics"].values()
        )
        else "partial"
    )
    write_json(root / "quality.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("infer", "score"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = read_json(arguments.config)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(config["physical_gpu"])
    result = (infer if arguments.phase == "infer" else score)(config, arguments.output)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
