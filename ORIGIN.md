# Source provenance

The retained `_vendor` helpers originate in the Apache-2.0
`tts-serving-benchmark@86ab381fd9ff2fbbd7b638c50b9d48aaf623e0db`, extracted from
[SGLang-Omni benchmarks](https://github.com/sgl-project/sglang-omni/tree/f96b10e95ea98801a28cf27dd5f6a2b0a9d878c9/benchmarks).
Only the request-result schema, performance calculations and playback continuity
helpers remain. Unused dispatch, dataset loaders, printers and internal precision
providers were deleted in 0.7. Original SPDX headers and Apache-2.0 LICENSE remain.

`origin.json` records original paths, source hashes and retained copy hashes.
Performance formulas are preserved; the maintained wrapper defines TTFA as the
first nonempty PCM SSE event, leaves missing observations unavailable and counts
throughput only against a complete measured wall clock. It averages per-round
metrics rather than pooling percentiles.

Quality helpers originate in
[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/305273a673c281f2d214e1bed457a4b5b8895740/benchmarks):
WavLM speaker similarity, selected ASR normalization/edit-count functions and
UTMOS scoring with a local TorchScript-path constructor. Audio decoding uses
soundfile float32; resampling, PCM16 conversion and prediction are preserved.
The Qwen3-ASR HTTP adapter retains the recorded multipart protocol with bounded
fan-out and durable transcript caching. Model-serving code is not imported by E2E.

The QA Judge templates use the MIT-licensed
[Kimi-Audio-Evalkit](https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/a65bd4243dd280daa685e35a5cfd69c371315c06/almeval/datasets/ds_refqa.py)
at the pinned commit. See [Judge protocol](JUDGE_PROTOCOL.md) and
[upstream license](JUDGE_UPSTREAM_LICENSE.txt).

Historical removed code and its provenance remain available in Git history;
there are no compatibility copies in the installed package. Runtime artifacts
belong in the workspace's `benchmark-results/<run_id>/`, outside this source tree.
