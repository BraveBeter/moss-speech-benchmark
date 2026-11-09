# Codec-only reconstruction quality

The versioned `moss-codec-only-v1` protocol evaluates real audio → Encoder → Decoder. It never loads an AR language model or sends speech-QA requests. Existing QA suites, timing and correctness gates are unchanged.

## Frozen inputs

Use a JSONL corpus with one record per target/prompt pair:

```json
{"id":"en-01","language":"en","text":"Actual target transcript.","target_audio":"/absolute/target.wav","target_sha256":"...","prompt_audio":"/absolute/same-speaker-prompt.wav","prompt_sha256":"..."}
```

The transcript belongs to the actual target recording and the speaker prompt belongs to that sample. Record the dataset revision, selection and complete hashes before inference. Do not substitute a global deployment voice or a QA answer.

## Inference

Start from `examples/codec-only.json`, replacing paths and hashes. Select `hf-native`, `sg-fp32` or `sg-bf16`. SG requires `model_root` and a complete clean `model_sha`; BF16 additionally requires `flow_precision: "ffn_attention"` (M1) or `"ffn_attention_sdpa"` (M2). HF requires its original CosyVoice/Matcha dependency source tree as `hf_source_root`. Explicitly select the model interpreter and an authorized physical GPU. Pin all Codec weight/config/source assets.

```bash
python -m moss_speech_bench --phase codec-infer --config hf.json --output-dir results/hf
python -m moss_speech_bench --phase codec-infer --config sg-fp32.json --output-dir results/sg-fp32
python -m moss_speech_bench --phase codec-infer --config sg-bf16.json --output-dir results/sg-bf16
```

Use one fresh directory per backend/configuration. All three use offline decoding to match the public native HF `decode` API. SG BF16 applies the selected branch's `FlowBF16Scope` to FFN and attention projections. M1 retains FP32 SDPA; M2 additionally uses BF16 SDPA Q/K/V and live bias. Encoder, noise, solver and HiFT remain FP32 in both profiles. This tests that precision strategy in offline reconstruction. Production streaming quality is a separate configuration and must not inherit these results automatically.

Each backend runs its own Encoder, freezes target/prompt code ledgers and saves complete FLOAT WAVs. Native HF source is hash-verified and SG imports are forbidden in that path. Generation requires finite PCM and exactly 1920 samples per code. Failed samples remain in the planned denominator. Source/client identities, runtime versions, asset/input hashes and commands are retained.

## Scoring

Supply local scoring assets in a JSON configuration: `asr_en` with `model: whisper-large-v3` and `path: .../large-v3.pt`; `asr_zh` with `model: paraformer-zh` and its local `path`; `sim` with `path: .../wavlm_large_finetune.pth` and `wavlm_base: .../wavlm_large.pt`; plus `python` and `physical_gpu`. Optional dependencies are installed in the selected scorer environment, outside the lightweight client.

```bash
python -m moss_speech_bench --phase codec-score --score-config scorers.json --output-dir results/hf
```

Repeat for the SG runs. English uses Whisper-large-v3 with language=en, FP32 inference, temperature=0 and beam_size=5. Chinese uses Paraformer-zh. Existing client text normalizers produce English word-level WER and Chinese character-level CER. Corpus rates divide total substitutions+deletions+insertions by total reference words/characters; the per-sample mean is separate. SIM uses the existing WavLM speaker model against each sample's corresponding prompt, reported as cosine×100. Missing/failed scores are explicit and never become zero-error results. Existing scores cannot be overwritten implicitly.

## Comparison

```json
{"runs":{"HF native":"results/hf","SG FP32":"results/sg-fp32","SG BF16":"results/sg-bf16"}}
```

```bash
python -m moss_speech_bench --phase codec-report --config comparison.json --output-dir results/comparison
```

Comparison requires identical corpus, Codec assets, seed and decode mode, and verifies target/prompt Encoder codes match. It retains absolute WER/CER/SIM, actual denominators and failures for each backend. A subset is a regression screen, not a full reproduction of paper numbers. Codec quality does not replace C00–C08, HTTP Quick or end-to-end latency tests.

M2 additionally records positive BF16 SDPA execution counts and requires zero-stride head bias broadcasting without fallback. Its existing FP32 solver, noise and HiFT remain unchanged. These offline results do not establish streaming reconstruction quality.
