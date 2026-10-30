# Eager-base correctness and numerical alignment

Unified entry for the fixed eager base `084f62ef13c65fbf33f0c9a3f2cce5e87e0c01d6`.
The client remains free of Torch, SGLang and HF imports. Model checks run in a
separate, explicitly configured interpreter; HTTP checks connect to an existing
service. The suite never starts a server or discovers historical scripts.

## Gates and data

| Gate | Required coverage | Execution |
|---|---|---|
| C00 | Core SHA/clean tree, 50 pinned assets, independent golden/input/voice identities, deployment profile | Control process; full file hashing |
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

## Configure and run

Use [examples/correctness.json](examples/correctness.json). Its workspace root
is relative to the configuration file; every other path explicitly uses
`{workspace_root}`. Update the candidate interpreter, existing model assets,
reference locations and operator-supplied resolved deployment metadata. Asset
hashes are fixed before execution. The example binds this workspace's admitted
independent HF bundle; it does not distribute weights or tensor files.

C00 hashes all listed files, including weight shards, and checks cross-bundle
input/voice/tensor identities. It records fresh hashes and reference lineage.
The original reference JSON files retain their historical paths; providers use
explicit current paths and never rewrite those files. To introduce a different
reference, first independently export and admit its provenance, environment,
source closure, numerical settings and hashes. Candidate outputs are never golden.

```bash
# CPU selection and processor; intentionally incomplete as a full-suite result
moss-speech-bench --phase correctness --correctness-config examples/correctness.json \
  --checks C00,C01,C02 --output-dir results/correctness-single

# After arranging Slurm resources, run model providers in their own environment.
# Configure providers.C03..C06.launcher as an argv prefix such as:
# ["srun", "--partition=a800", "--gres=gpu:1", "--time=00:20:00", "/usr/bin/env"]
moss-speech-bench --phase correctness --correctness-config examples/correctness.json \
  --checks C03,C04,C05,C06 --output-dir results/correctness-single --resume

# Release provider processes; start the target service outside this client.
moss-speech-bench --phase correctness --correctness-config examples/correctness.json \
  --checks C07,C08 --output-dir results/correctness-single --resume

# Validate saved artifact hashes and regenerate the report without sending requests
moss-speech-bench --phase correctness-report --output-dir results/correctness-single
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
