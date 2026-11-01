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
that override and require a Slurm allocation. The interpreter and launch prefix
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
