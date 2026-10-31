# MOSS-Speech evaluation protocols

## Current suite: moss-spoken-qa-v3

Quick, Daily and Full begin with the existing C00–C08 correctness suite.
Inference is blocked unless all nine gates pass. Correctness artifacts and hashes
are bound to the run; offline scoring and report regeneration validate them.
The fixed eager-base profile is not certification for arbitrary optimizations.

End-to-end datasets are exclusively OpenAudioBench LLaMA Questions (300),
TriviaQA (1000) and WebQuestions (1000). Default tasks are both S2T and S2S.
Daily uses all 2300 questions, 4600 requests. Full scans the same complete data
at C1/C4/C8/C16, three repeats by default. Each dataset/task has a separate cell;
quality normally selects C4/r1 instead of rescoring every performance repeat.

Quick selects 20 fixed questions per dataset by source ID hash and seed42,
then evaluates each in both modalities: 60 questions, 120 requests.
The same source questions/audio must be used in S2T and S2S. Source manifests,
IDs, revisions, actual prompts, sampling and output hashes remain reproducible.
20–50 questions per dataset are configurable; a larger cohort needs a new
baseline and a fresh check of the 40-minute target.

S2T uses `spoken-qa-native-text-short-en-v1`: preserve the native system text
`You are a helpful assistant. Respond with text outputs.` and append the short
English answer instructions. Dataset and task-specific S2T system prompts also
retain this prefix; an existing complete prefix is not duplicated. An empty prompt
leaves the initial user turn intact so the Processor inserts its native default.
S2S keeps the original `spoken-qa-short-en-v1` request unchanged. Custom diagnostic
mode preserves explicitly configured raw prompts for historical reproduction.

The effective prompt contains no gold answer and is frozen with its hash in the
input contract. Suite v3 requires a fresh run and baseline; historical v2 outputs
and scores stay unchanged. This repair targets output modality, not answer
correctness or the unresolved repeated-text/long-answer cases. Do not silently truncate generated answers,
change max_tokens, or remove difficult examples to improve runtime or scores.

S2T uses normalized whole-answer exact match, retained as `accuracy` for format
compatibility. S2S transcribes saved audio with concurrent Qwen3-ASR, then applies
the same matching; UTMOS evaluates acoustic quality independently.
This EM is an engineering regression metric, not semantic QA accuracy or an
exact reproduction of a paper's evaluator. The optional `answer_mention` is only
a diagnostic: even an incorrect or negated answer may contain a valid alias.
Any semantic answer extractor/judge requires its own versioned protocol and
validation before formal adoption. Empty, failed, timed-out and truncated cases
remain in coverage and in the expected denominator.

## Deferred isolated Codec Decoder

Seed-TTS-Eval EN1088 and ZH2020 are reserved exclusively for an isolated Decoder
protocol in this project. Full records the planned 3108 samples as `deferred`.
No Decoder endpoint, tensor-input contract, or reference generation recipe is
currently implemented. Do not substitute chat-based TTS, end-to-end QA, or
correctness tensors for this quality evaluation. No fake HTTP request or score
is generated; the overall Full suite remains incomplete and exits 1.
WER/CER, speaker similarity and DNSMOS remain future Decoder metrics,
not correctness gates or end-to-end QA scores. Implementation is optional future
work, not a prerequisite for running the supported end-to-end portion.

## Historical diagnosis

The previous four-set Quick, Seed chat-reading prompts, StoryCloze and their
reports remain historical evidence. They do not become Decoder measurements
when the current suite changes. Old reports can still be reconstructed offline;
new named modes reject the old workload composition. New `kind=seedtts` chat
runs are rejected. Generic custom requests remain explicit diagnostic work.

The isolated `providers/hf_http.py` and explicit prompt/ASR language diagnostics
are retained. They compare original HF model behavior with eager serving and
record reference source, modality pad, generation conditions and PCM decoding.
They do not implement the deferred Decoder benchmark. The full independent
numerical and service checks remain in the correctness suite described in
[CORRECTNESS.md](CORRECTNESS.md).
