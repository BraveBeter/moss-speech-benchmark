# Correctness against independent HF references

Unified entry for any explicitly pinned candidate branch/commit. The numerical
oracle is the independent HF reference bundle; Git ancestry and a performance
baseline are not correctness criteria. `hf-reference` is the default profile;
`eager-base` remains a legacy spelling with the same candidate-independent rules.
The candidate HEAD must match `core_sha` and the worktree must be clean. The
example requires you to fill in the candidate path and its full commit SHA.
The client remains free of Torch, SGLang and HF imports. Model checks run in a
separate, explicitly configured interpreter; HTTP checks connect to an existing
service. The suite never starts a server or discovers historical scripts.

The client is used for model correctness, quality and performance evaluations.
Routine unit tests, Graph/Kernel development tests, debugging, profiling, linting
and package checks run directly with the owning repository tools. Graph/Kernel
implementations and their specialized tests belong to the model/engine repository;
a formal evaluation may invoke those tests and collect their evidence. Development
checks do not automatically trigger a full suite or qualify a model branch.

## Gates and data

| Gate | Required coverage | Execution |
|---|---|---|
| C00 | Candidate SHA/clean tree, 50 pinned assets, independent golden/input/voice identities, declared deployment | Control process; full file hashing |
| C01 | Fixed 514 node IDs including MOSS's 10 unit-test files and shared regressions | Target environment, CPU engine |
| C02 | 8 processor profiles, 76 named exact comparisons | Separate candidate CPU process, admitted independent HF JSON |
| C03 | Native prefill/decode isolation + two CUDA seed parameters, 3 tests | Slurm GPU provider |
| C04 | 8 profiles × 2 complete last-position vocabulary vectors | Full AR weights vs independent HF logits |
| C05 | EN/CN 250 codes each, complete fixed voice prompt/mel/speaker features | Independent HF safetensors vs candidate encoder |
| C06 | 24/32/511/512 codes, complete FP32 PCM | Independent original flow.pt + HiFT golden vs candidate decoder |
| C07 | 16 modality/language/stream combinations + EOS/tail/disconnect/invalid recovery | 20 logical cases, normally 22 HTTP attempts |
| C08 | Red/blue serial vs simultaneous calls, text and audio | 2 checks, normally 8 HTTP attempts |

514 includes the historical 247 MOSS CPU cases; do not add the counts. Inputs:
canonical JSON (five source conversations expanded to eight processor profiles),
the pinned 20-second EN/CN WAVs and complete 27.43-second CN voice. Known fixture
codes do not substitute for encoder tests. Streaming and buffered PCM are not
mutual golden references. The 511/512 boundary inputs repeat a pinned code sequence
and are not natural long-speech quality data.

AR comparison uses the admitted HF SDPA/BF16/all-rows reference, seed 20261027,
with FP32 comparison `abs(candidate-reference) <= .25 + .02*abs(reference)`.
All 16 raw top1 IDs must also match. Encoder float features and full Codec PCM
use `atol=rtol=1e-5`; integer codes, shapes and the declared dtype policy are
strict. The migrated providers preserve the original numerical algorithms and
full output tensors. No trimming, padding, NaN equality or post-failure tolerance
adjustment is allowed. Codec CFM seed 42 and original deterministic/cuDNN state
checks remain in the provider.

## New eager-base optimization qualification

Set `context_remaining: true` independently of the compute `profile` to qualify the 40,960-row context and remaining default generation budget. C01 keeps the frozen base allowlist, replaces the obsolete 513-row rejection with 40961, and adds the 22 existing context/default/stream regression cases. C02 constructs exact 40959/40960 canonical prompts. C07 requires 26 logical checks, including explicit/omitted context admission, 1001-row text and audio, context-overflow rejection/recovery, omitted text EOS and null audio EOS. Pin `long_text_request` in the full-hash asset allowlist; all boundary requests and responses are retained.

| Compute profile | Extra qualification | C01 / C03 with context_remaining |
|---|---|---|
| hf-reference | Original HF and native CUDA checks | 536 / 3 |
| ar-decode-graph | Target fixture checks actual full-checkpoint capture/replay, both complete heads and all40 KV layers, plus tiny-model CUDA lifecycle/fallback | 536 / 4 |
| ar-projection-graph | Explicit projection_graph_batch_size1..4; CPU arithmetic/config/cleanup and four actual CUDA batch profiles with complete model/KV equality | 549 / 7 |
| mixed-modalities | Typed opt-in and lifecycle/cancel isolation; actual mixed CUDA batch model/KV checks; C08 additionally compares mixed text/audio serial vs overlapping calls | 550 / 4 |
| native-norm | Explicit pinned library and native-rounding/fallback fixture | 537 / 4 |

Deployment must declare the corresponding `ar_graph`, `projection_graph_batch_size` or `mixed_modalities` setting; a mismatched or unqualified compute profile is rejected rather than silently covered by eager results. Projection HF prefill remains eager by design, with replay correctness separately established in C03. Server-side activation evidence is still required to show that the measured deployment used Graph/native/mixed admission: HTTP overlap alone does not establish an engine batch, and the client does not infer server counters from its local checkout. These additions do not change numerical tolerances, reference goldens or named Quick workloads.

## Batched decode RoPE qualification

Use `profile: "ar-batched-rope"`, `context_remaining: true`, and deployment
`decode_rope_reuse: true`. This profile qualifies only per-forward cosine/sine
reuse for B2/B4 decode; unrelated Graph/native/cache strategies are rejected.
C01/C03 add fixed fallback, complete-head/KV, context-rounding and CUDA tests.
C04 retains the eight serial HF checks and additionally teacher-forces all eight
fixed HF inputs at both B2 and B4, compares both full vocabulary heads with the
unchanged HF tolerance and raw top1 requirements, and requires exact disabled vs
enabled full-head/all-40-layer KV proofs plus positive actual execution counts.
The extra vectors are preserved as `batched-rope-logits.safetensors` with their hash.

Pin `service_activation_log` and optional nonnegative `service_activation_offset`.
C08 requires the actual `batched RoPE reuse active` execution marker, not a startup
configuration message. After Quick, independently archive an activation audit
for that same service/run using `python -m moss_speech_bench.providers.ar_combination
--config <correctness.json> --output <run>/execution/activation`; do not infer an
engine batch from HTTP concurrency alone. This does not change named workloads,
reference tensors, tolerances, or qualify combinations of optimizations.

## Configure and run

Use [examples/correctness.json](examples/correctness.json). Its workspace root
is relative to the configuration file; every other path explicitly uses
`{workspace_root}`. Update the candidate interpreter, existing model assets,
reference locations and operator-supplied resolved deployment metadata. Set both
`core_sha` and `deployment.core_sha` to the actual candidate, with no ancestor
requirement. Asset
hashes are fixed before execution. The example binds this workspace's admitted
independent HF bundle; it does not distribute weights or tensor files.

C00 hashes all listed files, including weight shards, and checks cross-bundle
input/voice/tensor identities. It records fresh hashes and reference lineage.
The original reference JSON files retain their historical paths; providers use
explicit current paths and never rewrite those files. To introduce a different
reference, first independently export and admit its provenance, environment,
source closure, numerical settings and hashes. Candidate outputs are never golden.

```bash
# Result location is independent of the checkout/worktree working directory.
MOSS_BENCHMARK_RESULTS_ROOT=/remote-home1/xrluan/speech-rep/benchmark-results

# CPU selection and processor; intentionally incomplete as a full-suite result
moss-speech-bench --phase correctness --correctness-config examples/correctness.json \
  --checks C00,C01,C02 --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/correctness-single/correctness"

# After arranging Slurm resources, run model providers in their own environment.
# Configure providers.C03..C06.launcher as an argv prefix such as:
# ["srun", "--partition=a800", "--gres=gpu:1", "--time=00:20:00", "/usr/bin/env"]
moss-speech-bench --phase correctness --correctness-config examples/correctness.json \
  --checks C03,C04,C05,C06 --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/correctness-single/correctness" --resume

# Release provider processes; start the target service outside this client.
moss-speech-bench --phase correctness --correctness-config examples/correctness.json \
  --checks C07,C08 --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/correctness-single/correctness" --resume

# Validate saved artifact hashes and regenerate the report without sending requests
moss-speech-bench --phase correctness-report --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/correctness-single/correctness"
```

Providers require a Python 3.12 candidate runtime with the target dependencies.
CPU lanes must run on a compatible node; configure C01/C02 launchers as CPU-only
Slurm tasks (e.g. `--gres=none`) when the login-node ABI differs. Inside the job,
the entry sets CPU-engine mode and hides CUDA before imports. GPU lanes clear
that override and require a Slurm allocation or an explicit local allocation. The interpreter and launch prefix
are explicit argv, never shell snippets. The scheduler and GPU budget are owned
by the outer workflow; the client does not allocate GPUs without configuration.

Single/dual deployments use separate run directories. Dual placement is AR on
GPU0, encoder/vocoder on GPU1, TP1. Reuse of CPU/component evidence across runs
must be explicitly documented; this interface does not automatically merge
historical runs into a pass. Deployment metadata is an operator declaration,
not proof of server internals obtained through HTTP.

## Assertions, artifacts and recovery

C07 reuses ChatClient, SSE reconstruction, WAV parsing and request persistence,
then checks terminal/DONE ordering, usage arithmetic, modality exclusivity,
nonzero PCM, 24kHz mono format and chunk/code sample counts. Disconnect records
`client_closed=true` and `server_cancel_ack=null`; a successful follow-up does
not prove immediate KV cleanup. Expected HTTP 400 is checked separately from a
successful generation. All attempts and raw response events are retained.

C08 requires different serial red/blue content, positive overlapping call
intervals on independent connections, exact text/finish/usage and complete PCM.
WAV headers and response IDs are excluded. One LSB is a failure. Diagnostics
include mismatch counts and first positions. HTTP overlap does not prove engine
batch size > 1.

`correctness.json` and `correctness.md` are independent of performance/quality
artifacts. Per-gate directories retain commands, environment declarations,
module origins, JUnit, exact node IDs, logs, requests/WAV and numerical tensors.
Required status values are pass/fail/skip/not_run/unsupported. Only all nine
required gates passing yields exit 0; failures or incomplete coverage exit 1.
Invalid configurations/artifact corruption exit 2. Zero collection, required
skips and exit-code-only provider success cannot pass.

A resumed run requires the same configuration, client and provider data, valid
artifact hashes and unchanged clean candidate source. Passed gates are reused;
failed/interrupted gates rerun as a whole in a fresh attempt directory. Failed
results remain in history and their files are still verified. A partial subset
is always incomplete, including a standalone C08 diagnostic. Running the full
suite without `--checks` stops dependent work at the first failed gate.

This integration has local protocol/contract tests, fresh CPU 514 and processor
76 comparisons, and three CPU/Torch tensor-policy tests. Real GPU C03–C06 and
single/dual-service C07–C08 have not been rerun by this integration. Historical
successes are preserved as historical evidence, not new acceptance results.

正确性阶段也必须归入工作区`benchmark-results/<run_id>/correctness/`，与同次性能/质量结果共用run根。只运行正确性时，在run根保留简短summary.md入口，详细断言仍在correctness/correctness.md；不向.audit或源码仓库另存报告。


The explicit native-norm profile accepts any full clean candidate SHA and an operator-built library pinned by the same asset allowlist and deployment SHA256. It preserves all C00–C08 checks, numerical thresholds and golden identities; C01/C03 append fixed norm fallback/actual-CUDA qualification cases. C04 installs the same norm library on the candidate model before its full-vocabulary comparison. A successful eager-base certificate does not qualify the native-norm profile, and profile/library changes invalidate resume identity. Native service manifests must enable this exact library independently; the client never enables a service feature by changing its own environment.

The native-tail-research profile is deliberately incomplete: it pins the candidate SHA, the original Lt reference module and a helper compiled against the current headers. C03 retains the base native tests, checks actual struct sizes/offsets, observes the original version rejection, and runs a separately labelled in-memory version-override diagnostic on checkpoint affine weights. No on-disk guard is changed. Those local full/tail affine observations cannot qualify complete CFM, prefix ownership or a serving branch; C03 remains unsupported and subsequent HTTP/Quick gates stay blocked. The report retains every API failure and numerical mismatch.

## Compute-path coverage

Candidate identity and numerical-adapter support are separate. C00 admits any
clean pinned candidate and validates the declared placement and voice; capacity,
chunk size and GPU count are not required to match the historical eager service.
HF weights/revisions, deterministic inputs, independent goldens and all existing
numerical thresholds remain unchanged.

The current C04 adapter covers BF16, torch_native attention, TP1 and AR graph off;
C05/C06 cover FP32 Codec. A different declared compute path is `unsupported` at
the relevant gate, which blocks dependent inference. The client never silently
substitutes the eager path to qualify a graph/kernel configuration it did not run.
Extend the corresponding adapter to execute and compare that actual path before
claiming support. A passing branch-independent admission check is not a model
numerical pass. C01/C03 still require the fixed test node IDs in the candidate;
missing or skipped tests are never treated as success.


## New eager base and P2 migration profiles

`context-remaining` qualifies the 40960 logical/40962 physical context and remaining-budget default. `context-small-first`, `context-startup-schedule`, and `context-prefill-graph` include the same contract plus the corresponding existing strategy tests. Candidates remain identified by any explicit full SHA; Git ancestry is not an oracle.

All four require `long_text_request` in the pinned asset allowlist. C01 runs 536 baseline cases, 563 for small-first, 596 for startup schedule. C02 constructs exact 40959/40960 prompt probes. C07 requires acceptance/rejection, explicit 1001-token text/audio output, omitted/null budget with early EOS, and recovery: 26 named checks. C08 retains exact concurrent isolation for each selected strategy.

For small-first set `deployment.small_first_chunk=true`; for schedule set it false and `deployment.stream_startup_schedule="6_8_align25"`. These must match the actual `audio_vocoder.factory` fields in the externally started server. Other context profiles leave both disabled. All use `context_limit=context_length=40960` and `max_total_tokens=40962`.

Prefill graphs additionally require `graph_lengths` and identical `deployment.prefill_graph_lengths` (one to eight distinct lengths within 1..512), and `deployment.ar_graph=true`. C03 runs the existing full prefill graph GPU test; C04 invokes the candidate full-weight fixture, requires actual capture/replay and exact complete heads/scattered KV across all 40 layers. Unsupported shapes remain eager; a successful fallback alone does not qualify Graph. Example factory field: `ar_engine.factory.prefill_graph_lengths=[34,41]`.

C06 remains an independent HF full offline codec comparison. Startup changes the streaming chunk schedule and may change PCM; C07/C08 verify the actual strategy's protocol, full sample accounting, repeatability and isolation, not cross-strategy waveform or perceptual equality. Offline quality scoring is separate.

The eight independent HF AR cases remain the original prefill inputs. Finite Graph qualification additionally checks candidate complete-checkpoint Graph against eager heads/KV; if the HF cases are outside the configured bank, they are fallback coverage and are not a new direct Graph-HF numerical oracle.

## P4 migration regression selection

The `context-remaining` compute profile accepts an optional fixed `test_suite`: `p4-15` (codec unready return, 545 CPU/3 CUDA), `p4-16` (sampling return, 540/9), `p4-17` (host overlap, 536/4), or `p4-18` (shared HiFT, 539/4). Each retains the full new-context suite and explicitly maps renamed strategy tests. The default `legacy` selection preserves every existing compute profile. Unknown suites, duplicate or absent replacement nodes, skips and incomplete collection fail qualification. Both the isolated provider and validator use the same fixed composition; execution remains in the candidate repository.

The shared-HiFT CUDA test requires the provider's `MOSS_CODEC_PATH`, `MOSS_AUDIO_VOICE` and `MOSS_CODE_PROFILE` assets; a missing-asset skip cannot pass C03. These additions change no numeric threshold, HF reference, HTTP check, performance metric, workload, or scoring behavior.


### Optional P3 strategies on the40960-context candidate

Set `context_remaining: true` independently of `profile`. The fixed CPU/CUDA selectors include the new context/budget regressions and the strategy-specific tests; skipped or uncollected required tests fail qualification. Supported profiles are `ar-prefix-cache`, `ar-sampling-history`, `ar-host-metadata`, `direct-lt`, `codec-cfm-graph`, `codec-fairness`, and `codec-score-release`. Deployment activation must match the profile; DirectLt additionally pins its explicit current-ABI library in the asset allowlist.

C04 compares both full-checkpoint vocabulary heads to the independent HF oracle. Prefix cache also checks actual cached-prefill logits/allKV; host metadata runs the actual backend for all eight HF cases. C03 covers CUDA history/RNG, backend isolation, real DirectLt calls, and real CFM capture/replay or fairness with complete streaming PCM and cleanup. Streaming Graph/fairness fixtures run in the candidate repository; C06 separately retains the independent offline full-codec HF comparison. These scopes are separate: an eager fallback cannot qualify a graph. C07 includes missing/null remaining budget,40960 context boundaries, outputs exceeding1000, EOS and tail recovery; C08 requires actual concurrent isolation. Quick performance remains a separate120-request stage after allnine gates pass.

DirectLt native qualification stays in the two fixed C03 tests: actual qualified CUDA affine launches must match the original Torch linear bit for bit, and actual streaming native calls must preserve complete PCM, ownership, interleaving and cleanup against eager. Skips or a successful eager fallback cannot qualify this native scope.

C06 retains the independent HF offline 24/32/511/512 code comparisons, complete PCM and original tolerances. The provider records FULL_LENGTHS from the candidate source. The parent verifies the fixed current-ABI domain 1392/1400/1424/1500 and derives each unpadded mel length as `(actual_voice_prompt_tokens + code_count) * 4`. The current 342-token voice yields 1464/1496/3412/3416, all outside the domain: owner installation/closure remain mandatory, native calls must be zero and the extra check is labelled offline-shape-fallback. These HF results qualify the offline fallback, not a native oracle. If a future fixed voice/input produces any qualified length, C06 requires positive native calls; zero calls never receive a generic activation pass. This distinction changes no C03 requirement, numerical threshold, reference golden, HTTP check or Quick workload.


## Explicit AR combinations

`profile: ar-combination` qualifies an explicit `ar_optimizations` list of at least two strategies drawn only from `ar-decode-graph`, `ar-projection-graph`, `context-prefill-graph`, `native-norm`, `ar-host-metadata`, and `mixed-modalities`. Unknown or duplicate strategies, undeclared enabled paths, mismatched settings and unsupported extra optimizations fail admission. Set `context_remaining: true`; the independent HF assets, numerical tolerances and C00–C08 remain unchanged.

Declare `deployment.decode_graph` independently of prefill; `deployment.ar_graph` is true when any selected Graph is enabled. Match `projection_graph_batch_size`, `graph_lengths`/`deployment.prefill_graph_lengths`, `deployment.use_host_attention_metadata` and `deployment.mixed_modalities` exactly to the selection. Native norm requires the explicit library, full asset hash and matching `deployment.native_norm_library_sha256`.

C01/C03 retain the fixed remaining-context selection, resolve the integrated branch's named regression replacements explicitly, and append each selected strategy's qualification plus the combined CUDA regression. Missing nodes, skipped cases and failed collection cannot pass. C04 loads the full checkpoint and invokes the candidate's combined fixture. Each selected strategy must report actual execution and complete-head/all-KV equality; each selected Graph must additionally capture and replay. Eight independent HF full-vocabulary comparisons remain mandatory. Native execution during capture is labelled capture execution; Graph replay does not execute Python counters. Mixed admission retains the third concurrent-isolation group in C08.

Set `service_activation_log` to the fresh owned server log and, if it already contains data, `service_activation_offset` to its byte length before startup. C08 saves the observed log bytes, hash and per-strategy counts; unobserved paths remain `not_observed` because the HTTP gate may not form a batch eligible for projection replay. After Quick and the matched custom workloads, run the strict activation audit below. Its frozen JSON/log remain evidence even on failure. Every selected strategy requires a positive actual-execution marker by default. Only prefill may be omitted from `service_activation_required` when HTTP input lengths do not enter its finite bank; report that scope as C04 qualification with service replay unobserved, rather than a serving activation pass. Native startup markers alone do not prove request execution. The audit requires an actual selected Graph replay line with both `native_norm_capture_calls>0` and `native_owners>0`, or the explicit `native norm eager request execution active` marker. Owner installation with zero native capture calls cannot qualify native serving.

```bash
python -m moss_speech_bench.providers.ar_combination \
  --config /absolute/run/execution/correctness.json \
  --output /absolute/run/execution/final-activation
```

Use the same frozen client commit, input hashes, GPU UUID, dtype, voice, seed and explicit token budget for new eager-base, selected single-strategy and combination measurements. Run each final deployment's C00–C08 before named Quick. Keep standard Quick C4, four-modality C1 and mixed C2 in separate run directories. Mixed results must retain pair completion time and individual latency/TTFA/RTF tradeoffs. Previous single-strategy measurements are historical evidence and do not count as new combination results.

A projection batch profile is part of the measured deployment identity. B1 results do not establish B4 end-to-end benefit; tiny-model CUDA tests for B1–B4 establish correctness only. Select performance configurations from independently demonstrated workloads and retest each proposed stack.

### Serial B2/B4 full-decode graph extension

`profile: "ar-serial-decode-graph"` requires `context_remaining: true` and explicit
`deployment.decode_graph_serial_batch`, `deployment.decode_graph`, and
`deployment.ar_graph` all true. Other optimization combinations are rejected.
C01 retains baseline regressions and adds transaction/ownership/error cases; C03
retains B1 graph qualification and adds actual serial B2/B4 CUDA qualification.
C04 preserves the original eight independent HF full-prefill comparisons and
requires the full-trained-checkpoint serial fixture: actual B2/B4 execution,
complete hidden/dual-head/all40KV exactness, output ownership, all-row rejection,
runtime guards, boundary fallback and no partial-failure eager retry. No local
timing or eager-only execution substitutes for those checks.

C08 archives serving activation observations but does not assume its isolation
workload creates both batch sizes. **After Quick**, run the activation audit
command above against the same fresh service log. It must observe actual
`serial batch decode graph replay active batch=2` **and** `batch=4`; startup-only,
B1-only, one batch size or stale log slices fail. No E2E closure is claimed until
this final audit and complete-output comparisons pass. This extension does not
relax existing fixed-HF tolerances or change old profiles.

### Phase-matched independent HF decode reference

The `ar-batched-rope` C04 extension additionally requires
`golden.C04_cached_decode.{report,tensors}`, both pinned in the C00 asset allowlist.
This is an independently exported HF N-1-row prefill plus one cached decode step
on the same eight canonical inputs, official revision, BF16/SDPA settings and seed.
All three cache banks, positions, full input grids, module hashes and tensor bytes
are validated. Actual B2/B4 decode heads are checked against this phase-matched
reference with the unchanged finite/allclose/raw-top1 rules. The original HF
full-prefill reference and all serial checks remain mandatory; it is not replaced
by candidate/base outputs. Historical failed runs remain failed.

### Live Codec attention-bias broadcasting

`profile: "codec-bias-broadcast"` requires `context_remaining: true` and explicit
`deployment.broadcast_attention_bias: true`; other optimization combinations are
rejected. C01 adds strict config/factory/public-method fallback regressions; C03
adds actual-CUDA live-bias equality and full trained streaming PCM/intermediate/
request-state isolation checks, preserving the report in its gate directory.
C06 enables the real `AudioDecoder` production flag during the original fixed-HF
full-PCM comparisons, requiring all56 installed processors to execute and retain
zero head strides. Constructor flags or eager fallback alone cannot qualify it.
After Quick, the fresh serving activation audit must observe the actual
`MOSS_CODEC_BIAS_BROADCAST_ACTIVE` marker. No reference tolerance, backend policy,
input, sampling or named Quick workload is changed.

## Operator-owned local GPUs

On a dedicated host/container without Slurm, explicitly add `local_gpu_allocation: {"devices": [1], "authorization": "operator authorization for this run"}` to the correctness configuration. The external launcher must reserve those devices and use the same visible devices for serving. Providers set and validate CUDA_VISIBLE_DEVICES, require usable CUDA, and archive device properties and the declaration in each GPU gate. This is operator-supplied allocation metadata, not a scheduler reservation. Never fabricate SLURM_JOB_ID; existing Slurm configurations still work. Numerical checks, pinned assets, source cleanliness and C00–C08 coverage remain unchanged.

## Eager feedback fixes

Use `profile: context-remaining` with `test_suite: eager-bugs` for the candidate that removes the unused processor collate interface. The same explicit suite composes with the existing optimization profiles when `context_remaining: true`; each profile retains its full specialized qualification and activation requirements. AR combinations propagate the suite into their baseline regression selection. C01 keeps 536 baseline nodes, replacing the two collate-only tests with buffered/streaming zero-code completion regressions and updating the codec empty-final lifecycle test, then adds the selected strategy's checks. C02 uses the same independent HF reference, all eight request grids, masks, lengths, token encodings and defaults (55 exact comparisons); the 21 collate/padding comparisons no longer describe a serving interface. The explicit suite and report contract must agree. Legacy candidates retain all 76 comparisons. C03–C08, context probes and Quick requirements are unchanged. Empty audio is valid for the one-token boundary regression; normal Quick empty-output failures remain failures.

## Cumulative optimization stages

`profile: cumulative` requires `context_remaining: true`, `test_suite: eager-bugs`, and an explicit unique `optimizations` list. Every activation flag, library, finite graph shape and serving factory option must match the declaration. Serial decode retains its B1 dependency. Unknown paths and missing qualification nodes fail admission. The pinned HF assets, all eight request cases, numerical tolerances, request isolation and normal Quick failure rules remain unchanged.

C01/C03 retain the new eager regressions and compose the fixed selected-strategy nodes, including the original P4 CUDA/RNG/component-lifetime suites. Multiple AR paths require actual joint complete-head/all-KV evidence from `assert_cumulative_ar`; the serial-only AR subset retains the original trained B1/B2/B4 qualification, ownership, runtime guards and no-partial-retry proof. Batched RoPE still requires its independently exported cached-HF phase reference. The joint CUDA regression supplements rather than replaces the original eight C04 HF comparisons.

Set `cumulative_api: independent` when the candidate combines independent strategy ports rather than the historical integrated AR API. This composes every selected strategy's frozen atomic checks and a joint CUDA check; it does not require unrelated projection, prefill or mixed-modality APIs. Omitting this field preserves the integrated AR selection contract. Both modes require the same trained joint proof and independent HF checks.

When strategies replace the same baseline node with nested sets or distinct parameterizations of that same fixed test, cumulative selection retains every required case, independent of strategy order. Incompatible test destinations still fail. Full strategy lists reuse the selected baseline replacements, and explicitly empty extra gates retain their original baseline checks.

When at least two streaming Codec strategies are selected, C06 additionally requires `assert_cumulative_codec` from the candidate's trained-checkpoint test fixtures. Its evidence must bind the same frozen full-HF reference files, preserve the original PCM tolerance and code ledger, prove request/RNG/cleanup isolation, and demonstrate executions for every declared path. A missing helper or inactive path fails qualification; registering an option does not qualify a combination. Independent single-path C06 checks remain mandatory, including all56 live-bias processors and the pinned DirectLt domain when selected.

After complete named Quick, audit the same fresh serving log. Every required marker must be observed, including both serial B2/B4 sizes and native execution evidence. Exact-length prefill may retain the existing explicit activation exception when the fixed workload does not hit its registered lengths; this never establishes a prefill performance gain.

Freeze one client commit across matched measurements whenever possible. If cumulative qualification needs a client update, an earlier matched run can remain a reference only with an archived exact-commit source audit proving every transport, request, timing, parsing, statistics and workload module byte-identical, unchanged HF policy/assets, and identical legacy regression selections. Explicitly record both client identities and the allowed qualification-only difference; preserve the earlier manifest and raw metrics. Any measurement or workload change requires rerunning the affected references. This exception does not permit changing a numerical tolerance, relabeling an old runtime identity, or treating an unqualified candidate as passed.


### E13 BF16 approximate Flow contract

The explicit cumulative feature `codec-flow-bf16` requires `precision_contract: e13-bf16-approx-v1`, matching deployment and vocoder precision (`ffn`, `attention`, `ffn_attention` or `ffn_attention_sdpa`), bias broadcasting and unready-return. It excludes the existing FP32 CFM graph. All unchanged HF checks and offline FP32 Codec tolerances remain unchanged. C06 records cross-precision streaming PCM errors as diagnostics, requires actual BF16 operator execution and exact same-precision serial/interleaved PCM, sample accounting, RNG and cleanup. It does not claim streaming PCM passed the FP32 tolerance. End-to-end frozen-output quality is a separate required decision: English WER and Chinese CER may increase at most 1 percentage point, WavLM SIM (0–100) mean may decrease at most 2 points. Numerical error, tensor cosine, UTMOS and clipping are statistics only. C06 structural success alone does not qualify these quality gates. CLI, performance inputs, repetitions and timing are unchanged.

The `ffn_attention_sdpa` profile additionally requires `codec-sdpa-bf16`, four real CUDA mask/boundary checks, C06 proof of BF16 SDPA with zero-stride head bias broadcasting, and a fresh `MOSS_CODEC_SDPA_BF16_ACTIVE` service marker. The live bias is converted before head expansion; the source FP32 bias stays read-only. Omitting the new feature or its actual execution proof fails qualification.
