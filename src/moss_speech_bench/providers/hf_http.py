"""Isolated original-HF diagnostic endpoint; never import from the client runtime.

Buffered responses only: native HF does not expose the SGLang streaming events.
Use the original checkpoint's processor, generate and decode without SGLang code.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType


def read_pcm16(path):
    """PCM WAV I/O only; normalized samples match int16 / 32768 exactly."""
    import numpy as np

    with wave.open(str(path)) as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("Reference PCM reader requires uncompressed PCM16 WAV")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if samples.size != wav.getnframes() * wav.getnchannels():
            raise ValueError("Truncated reference WAV")
        return (
            samples.reshape(-1, wav.getnchannels()).T.astype(np.float32) / np.float32(32768)
        ), wav.getframerate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "codec", "voice", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument(
        "--pcm16-reader",
        action="store_true",
        help="Explicit PCM16 I/O adapter for environments without TorchCodec",
    )
    parser.add_argument("--default-repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Reference generation requires Slurm"
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoProcessor, GenerationConfig

    if args.pcm16_reader:
        import torchaudio

        def load_pcm(path, *positional, **kwargs):
            if positional or kwargs:
                raise ValueError("PCM reader accepts only a file path")
            samples, rate = read_pcm16(path)
            return torch.from_numpy(samples.copy()), rate

        torchaudio.load = load_pcm  # Isolated process: I/O only, no model method replacement.
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        args.model,
        codec_path=args.codec,
        device="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = (
        AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda:0")
    )

    def lineage():
        assert not any(
            n == "sglang" or n.startswith(("sglang.", "sglang_omni")) for n in sys.modules
        )
        files = {}
        for name, module in tuple(sys.modules.items()):
            if name.startswith(("transformers_modules.", "cosyvoice.", "matcha.")) and isinstance(
                module, ModuleType
            ):
                filename = vars(module).get("__file__")
                if filename and Path(filename).is_file():
                    files[filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        (directory / "lineage.json").write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "cuda": torch.version.cuda,
                    "ar_dtype": "bfloat16",
                    "attention": "sdpa",
                    "codec_dtype": "float32",
                    "sglang_imported": False,
                    "audio_io": "explicit PCM16 reader"
                    if args.pcm16_reader
                    else "native torchaudio",
                    "sources": files,
                },
                indent=2,
            )
            + "\n"
        )

    lineage()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200, {"status": "healthy"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.respond(404, {"error": "Unknown endpoint"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if payload.get("stream"):
                    return self.respond(
                        400, {"error": "Original HF diagnostic provider supports buffered only"}
                    )
                key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                turns = []
                for turn in payload["messages"]:
                    content = turn["content"]
                    if isinstance(content, list):
                        assert len(content) == 1 and content[0]["type"] == "input_audio"
                        path = directory / (key + ".input.wav")
                        path.write_bytes(
                            base64.b64decode(content[0]["input_audio"]["data"], validate=True)
                        )
                        content = {"path": str(path)}
                    turns.append({"role": turn["role"], "content": content})
                modalities = payload["modalities"]
                seed = payload.get("seed", 42)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                inputs = processor([turns], modalities)
                limit = payload.get("max_tokens", 512)
                config = GenerationConfig(
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=limit,
                    min_new_tokens=0,
                    repetition_penalty=payload.get(
                        "repetition_penalty", args.default_repetition_penalty
                    ),
                    pad_token_id=model.config.modality_pad_token_id,
                    eos_token_id=[
                        processor.tokenizer.pad_token_id,
                        processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    ],
                )
                with torch.no_grad():
                    tokens = model.generate(
                        input_ids=inputs["input_ids"].cuda(),
                        attention_mask=inputs["attention_mask"].cuda(),
                        generation_config=config,
                    )
                    (directory / (key + ".generation.json")).write_text(
                        json.dumps(
                            {
                                "input_ids": inputs["input_ids"].tolist(),
                                "attention_mask": inputs["attention_mask"].tolist(),
                                "generated_grid": tokens.tolist(),
                                "raw_decode": processor.tokenizer.decode(
                                    tokens[0, :, 0].tolist(), skip_special_tokens=False
                                ),
                                "visible_decode": processor.tokenizer.decode(
                                    tokens[0, :, 0].tolist(), skip_special_tokens=True
                                ),
                                "token_strings": processor.tokenizer.convert_ids_to_tokens(
                                    tokens[0, :, 0].tolist()
                                ),
                                "generation_config": config.to_dict(),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    decoded = processor.decode(
                        tokens, modalities, decoder_audio_prompt_path=args.voice
                    )[0]
                message = {
                    "role": "assistant",
                    "content": decoded.generated_text if modalities == ["text"] else "",
                }
                if modalities == ["audio"]:
                    pcm = decoded.audio.squeeze().cpu().float().numpy()
                    buffer = io.BytesIO()
                    sf.write(buffer, pcm, decoded.sampling_rate, format="WAV", subtype="PCM_16")
                    message["audio"] = {
                        "id": "hf-" + key[:16],
                        "data": base64.b64encode(buffer.getvalue()).decode(),
                    }
                count = tokens.shape[1]
                prompt = inputs["input_ids"].shape[1]
                (directory / (key + ".json")).write_text(
                    json.dumps(
                        {
                            "payload": payload,
                            "input_ids": inputs["input_ids"].tolist(),
                            "attention_mask": inputs["attention_mask"].tolist(),
                            "generated_grid": tokens.tolist(),
                            "generated_text": decoded.generated_text,
                            "generation_config": config.to_dict(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
                lineage()
                self.respond(
                    200,
                    {
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "length" if count == limit else "stop",
                                "message": message,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt,
                            "completion_tokens": count,
                            "total_tokens": prompt + count,
                        },
                    },
                )
            except Exception as error:
                import traceback

                traceback.print_exc()
                self.respond(500, {"error": type(error).__name__ + ": " + str(error)})

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
