# Source provenance

The `_vendor` package reuses the Apache-2.0 standalone `tts-serving-benchmark`
provided by the user, at commit `86ab381fd9ff2fbbd7b638c50b9d48aaf623e0db`.
That project extracted its benchmark helpers from
[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/f96b10e95ea98801a28cf27dd5f6a2b0a9d878c9/benchmarks)
at commit `f96b10e95ea98801a28cf27dd5f6a2b0a9d878c9`.

Reused modules: request-result schema, runner, SeedTTS loader, performance metrics,
playback continuity metrics, and their formatting helper. Original SPDX headers
and Apache-2.0 LICENSE are retained. `origin.json` records original relative paths,
original file hashes, and installed copy hashes. It ships with source and wheel.

Changes to the copied code:

- Imports use `moss_speech_bench._vendor`.
- Runner HTTP pool follows requested concurrency instead of inheriting the
  implicit 100-connection limit for large closed-loop workloads.
- Runner cancels and drains pending tasks on interruption or an artifact-writing
  exception, so no request continues modifying a finalized attempt.

Metric formulas and dataset loading algorithms are unchanged. The MOSS wrapper
uses `audio_ttfp_*` for the first nonempty PCM SSE event (TTFA), rather than
claiming it observes a native codec chunk. Missing usage and interrupted wall
clocks are explicitly marked unavailable outside the copied metric code.

The new adapter, workload mapping, run artifacts, CLI and reports are maintained
outside `_vendor`. They do not import the inference engine or its model tests.

Quality helpers reuse SGLang-Omni at `305273a673c281f2d214e1bed457a4b5b8895740`:
`metrics/accuracy.py` and `metrics/speaker_similarity.py` are copied unchanged;
`tasks/asr.py` contributes `_get_en_normalizer`, `normalize_text`, and `apply_wer`;
`metrics/utmos.py` contributes `score_batch`, with a local TorchScript-path constructor
replacing model download/cache orchestration. Source and copy hashes and selected
symbols are recorded in `origin.json`. Text normalization, edit counts, speaker
embedding and UTMOS algorithms are preserved. QA alias matching and coverage/report
management are new code outside `_vendor`.

Correctness providers adapt the workspace's archived independent-reference
comparisons (original paths and SHA256 in `origin.json`). The executable copies
live in `providers/`; there are no execution imports of archived runners.
Changes remove directory-depth/old-core bindings, use explicit current asset
paths and retain the complete numerical algorithms. The CPU/CUDA selection is
fixed from the recorded eager-base JUnit node IDs and three accelerator nodes;
the target's tests and fixtures remain in its own checkout. The HTTP case
protocol and assertions follow eager-base `tests/test_model/test_moss_speech_ci.py`
and archived strict-C2 behavior, implemented using the standalone ChatClient
without importing service-management fixtures.

Qwen ASR scoring follows the current official `benchmarks/tasks/asr.py` multipart fields and default fan-out 32, verified on 2026-10-31. New async orchestration adds transcript identity/hash caching without importing model-serving code. UTMOS WAV decoding uses soundfile float32, matching the upstream audio fallback; the 16kHz torchaudio resampling, PCM16 conversion and prediction remain unchanged.

The current `moss-spoken-qa-v2` named suites reuse the existing OpenAudioBench
CSV mapping for LLaMA Questions, TriviaQA and WebQuestions, paired in S2T/S2S.
The vendored SeedTTS loader is retained unchanged for source provenance; the
current client rejects chat-based SeedTTS runs and defers isolated Decoder
quality evaluation. Existing correctness providers and numerical algorithms are
unchanged; named inference now gates on their complete report and offline
reporting validates its bound artifacts.

Run artifacts and reports are workspace data under `benchmark-results/<run_id>/`,
with `summary.md` as the entry point. They are not part of the client source
repository. Execution evidence and optional detailed analyses stay inside that
same run directory.
