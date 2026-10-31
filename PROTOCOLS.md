# MOSS-Speech TTS and spoken-QA protocols

Quality comparisons require the same input IDs, weights, voice, decoding profile,
sampling, budgets and scorer identity. Preserve the original run and score;
protocol changes create a new baseline. A completed score is not a quality pass.

## Text to speech

The default `moss-chat-v1` asks a dialogue model to read text, followed by its
trained default system turn. This is an instruction-following engineering test.
It is not the paper's isolated speech-decoder evaluation or SeedTTS voice cloning.

A dataset may explicitly declare a versioned prompt:

```json
"prompt": {
  "name": "tts-read-system-zh-v1",
  "system": "只用中文逐字朗读用户原文，不回答问题，不翻译，不添加解释。",
  "template": "{text}"
}
```

`template` must contain exactly one `{text}` field. `system` is a separate first
message. The default is unchanged when `prompt` is absent. The unmodified target
remains `expected`; prompt content/hash and actual request hashes are recorded.
Never strip a reply, translate its transcript back, or replace the reference to
improve WER/CER. Before accepting a protocol, examine a fixed bilingual challenge
set containing questions, instructions, numbers and normal declarative sentences.
Report language/task violations and empty/truncated output alongside error rates.
For ASR diagnosis, `asr_en/asr_zh.language_hint: "auto"` omits the language hint;
this is an explicitly different scorer identity. The normal default still sends
the declared sample language. Preserve both transcripts and their settings.

Use corpus edits/reference length as the primary WER/CER, per-sample values as
diagnostics. Insertions can make a sample exceed 100%. Compare the recognizer's
transcript with the audio and another recognizer/manual review for outliers.
SIM uses the configured service voice; per-sample clone SIM needs actual cloning
support. Neither instruction compliance nor a low WER alone proves model parity.

## Speech to speech

Keep natural dialogue QA and constrained short-answer QA as separate named
protocols. Speech inputs may declare `prompt.system`; they do not accept a text
`template`. A generic short-answer system instruction must not contain the gold
answer, and input audio, aliases and ASR outputs must remain available.

The existing `accuracy` field remains normalized whole-answer exact match (EM).
A correct explanation can fail EM. Do not present this value as semantic QA
accuracy. An optional scoring flag `qa_answer_mention: true` adds `answer_mention`,
using normalized token-boundary aliases. This is a diagnostic only: a negated or
otherwise incorrect sentence can mention an answer. It never replaces EM.

Semantic QA requires a separately versioned official evaluator or explicit answer
extractor/judge with its own validation and failure coverage. Do not introduce a
permissive substring metric and call it accuracy. UTMOS measures acoustic quality,
not answer correctness. Preserve refusal, empty, timeout and length-limited cases.

## HF comparison

Run `providers/hf_http.py` only in an isolated, explicitly selected HF environment
inside Slurm, then connect with the normal `infer --mode custom --no-stream` CLI.
The provider uses the original checkpoint's AutoProcessor, model.generate and
processor.decode; it imports no SGLang implementation. It records source hashes,
input grids, generated grids and generation configuration. It supports buffered
WAV output only and is a diagnosis tool, not a production serving backend.

Compare exact prompt/sampling inputs first, then change one prompt condition at a
time. Use the same full-decoder profile for waveform comparisons. Streaming and
buffered decoders, different precision/kernels and different attention libraries
can have different waveforms; behavioral ASR agreement is not tensor parity.
Use the independent `correctness` phase for numerical/greedy gates. Do not call
similar quality scores proof of correct adaptation or tuned prompts model fixes.
