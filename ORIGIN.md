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
