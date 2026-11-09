"""Independent local HF exports; never import candidate or SGLang modules."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from moss_speech_bench.noise_tape import NoiseTape


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


@torch.inference_mode()
def ar(config, output):
    attention = config.get("reference_attention_backend", "sdpa")
    flashinfer_attention = None
    if attention == "flashinfer":
        from moss_speech_bench.hf_flashinfer import FlashInferAttention

        flashinfer_attention = FlashInferAttention(torch.device("cuda:0"))
        flashinfer_attention.register()
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], trust_remote_code=True, local_files_only=True
    )
    formatter = get_class_from_dynamic_module(
        "processing_moss_speech.MossSpeechSampleProcessor",
        config["model_path"],
        local_files_only=True,
    )(tokenizer, None, {"audio": config["service"]["system"]})
    model = (
        AutoModel.from_pretrained(
            config["model_path"],
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention,
        )
        .to("cuda:0")
        .eval()
    )
    expected_source = Path(config["model_path"]) / "modeling_moss_speech.py"
    assert (
        hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).digest()
        == hashlib.sha256(expected_source.read_bytes()).digest()
    )
    tensors, cases = {}, []
    teacher = torch.tensor(config["teacher_rows"], device="cuda", dtype=torch.long)
    for case in config["cases"]:
        messages = [
            {"role": "system", "content": config["service"]["system"]},
            {"role": "user", "content": case["text"]},
        ]
        rows = formatter.prepare_sample(messages, "audio").input_ids_2d.T.contiguous()
        prompt = rows.unsqueeze(0).to("cuda")
        cases.append(dict(case, prompt_rows=rows.tolist()))
        cache = None
        for step in range(len(teacher) + 1):
            full = torch.cat([prompt, teacher[:step].unsqueeze(0)], dim=1)
            fresh = model(
                input_ids=full,
                attention_mask=torch.ones(full.shape[:2], device="cuda", dtype=torch.long),
                use_cache=False,
                logits_to_keep=0,
            )
            cached = model(
                input_ids=prompt if step == 0 else teacher[step - 1 : step].unsqueeze(0),
                attention_mask=torch.ones(full.shape[:2], device="cuda", dtype=torch.long),
                past_key_values_dict=cache,
                use_cache=True,
                logits_to_keep=0,
            )
            cache = cached.past_key_values_dict
            for head in ("text", "audio"):
                tensors[f"teacher/{case['id']}/{step}/{head}"] = (
                    getattr(cached, head + "_logits")[0, -1].float().cpu().contiguous()
                )
                tensors[f"uncached/{case['id']}/{step}/{head}"] = (
                    getattr(fresh, head + "_logits")[0, -1].float().cpu().contiguous()
                )
        cache = None
        for step in range(len(teacher) + 1):
            cached = model(
                input_ids=prompt if step == 0 else teacher[step - 1 : step].unsqueeze(0),
                attention_mask=torch.ones((1, len(rows) + step), device="cuda", dtype=torch.long),
                past_key_values_dict=cache,
                use_cache=True,
                logits_to_keep=0,
            )
            cache = cached.past_key_values_dict
            for head in ("text", "audio"):
                tensors[f"repeat/{case['id']}/{step}/{head}"] = (
                    getattr(cached, head + "_logits")[0, -1].float().cpu().contiguous()
                )
        generation = GenerationConfig(
            do_sample=False,
            min_new_tokens=0,
            repetition_penalty=config["sampling"]["repetition_penalty"],
            max_new_tokens=config["greedy_steps"],
            pad_token_id=151667,
            eos_token_id=[151643, 151645],
            return_dict_in_generate=False,
        )
        seed(config["sampling"]["seed"])
        generated = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones(prompt.shape[:2], device="cuda", dtype=torch.long),
            generation_config=generation,
        )
        cases[-1]["greedy_rows"] = generated[0].cpu().tolist()
    save_file(tensors, str(output / "ar.safetensors"))
    result = dict(
        cases=cases, model_class=type(model).__name__, attention=attention, dtype="bfloat16"
    )
    if flashinfer_attention is not None:
        result["flashinfer"] = flashinfer_attention.report()
        result["attention_adapter_sha256"] = hashlib.sha256(
            Path(inspect.getfile(type(flashinfer_attention))).read_bytes()
        ).hexdigest()
    return result


@torch.inference_mode()
def codec(config, output):
    source = Path(config["hf_source_root"])
    sys.path[:0] = [str(source), str(source / "Matcha-TTS")]
    model = (
        AutoModel.from_pretrained(
            config["codec_path"], trust_remote_code=True, local_files_only=True
        )
        .to("cuda:0")
        .eval()
    )
    assert all(p.dtype == torch.float32 for p in model.parameters())
    wave, rate = torchaudio.load(config["voice_wav"])
    if rate != 24000:
        wave = torchaudio.transforms.Resample(rate, 24000)(wave)
    codes = model.encode([config["voice_wav"]], batch_size=1)[0]
    voice_code_count = len(codes)
    mel, _ = model._extract_speech_feat(wave)
    count = min(mel.shape[1] // 4, len(codes))
    embedding = model._extract_spk_embedding(torchaudio.transforms.Resample(24000, 16000)(wave))
    tensors = {
        "voice/prompt_token": torch.tensor([codes[:count]], dtype=torch.long),
        "voice/prompt_feat": mel[:, : count * 4].float().cpu().contiguous(),
        "voice/embedding": embedding.float().cpu().contiguous(),
    }
    voice = {name.removeprefix("voice/"): value.to("cuda") for name, value in tensors.items()}
    # CausalCFM creates this stochastic input during construction, before token2wav.
    tensors["noise/flow_initial"] = model.audio_decoder.flow.decoder.rand_noise.cpu().clone()
    noise_calls = {}
    for count in config["codec_code_counts"]:
        seed(config["sampling"]["seed"])
        codes = [config["codec_codes"][i % len(config["codec_codes"])] for i in range(count)]
        with NoiseTape(tensors, str(count)) as tape:
            wave, _ = model.audio_decoder.token2wav(
                torch.tensor([codes], device="cuda"),
                uuid=f"reference-{count}",
                **voice,
                finalize=True,
            )
        noise_calls[str(count)] = tape.calls
        tensors[f"pcm/{count}"] = wave.detach().cpu().float().reshape(-1).contiguous()
    save_file(tensors, str(output / "codec.safetensors"))
    return dict(
        voice_codes=voice_code_count, dtype="float32", sample_rate=24000, noise_calls=noise_calls
    )


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
    seed(config["sampling"]["seed"])
    result = (ar if args.phase == "ar" else codec)(config, args.output)
    forbidden = [
        name
        for name in sys.modules
        if name == "sglang" or name.startswith(("sglang.", "sglang_omni"))
    ]
    assert not forbidden, "HF oracle imported candidate/serving code"
    result.update(
        status="complete",
        serving_modules_imported=forbidden,
        python=sys.version,
        torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        torchaudio=torchaudio.__version__,
        gpu=dict(
            name=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
            cuda=torch.version.cuda,
        ),
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        noise_helper_sha256=hashlib.sha256(
            Path(inspect.getfile(NoiseTape)).read_bytes()
        ).hexdigest(),
    )
    if args.phase == "codec":
        import onnxruntime

        result["onnxruntime"] = onnxruntime.__version__
    (args.output / (args.phase + ".json")).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
