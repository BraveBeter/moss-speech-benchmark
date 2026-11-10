# Independent HF correctness regression

`hf-regression-v1` defines correctness in this client. It does not collect candidate pytest nodes, import candidate tests, check branch names or require ancestry from a performance baseline. The reference runs official local HF code in an isolated interpreter; a versioned backend adapter calls production model APIs in the candidate interpreter. The lightweight coordinator compares frozen data and never imports model code.

The first adapter, `sglang-native-moss-v1`, covers single-turn text input with spoken answers, using the production MOSS request builder, AR factory and reference-voice Codec. Other model API versions or modalities require an adapter implementing the same data contract; missing coverage cannot become a pass. Performance transport can select supported T2T/T2S/S2T/S2S tasks independently, but formal inference must be covered by the bound correctness certificate.

## Gates and numerical policy

| Gate | Client-owned assertion |
|---|---|
| C00 | Clean full candidate SHA, complete asset hashes, independent HF exporter/source closure, frozen input/policy binding; exact repeatability of a fresh HF cached trace |
| C01 | Actual candidate two-channel prompt grids exactly equal official HF formatting for all declared cases |
| C02 | Complete unmasked text and audio vocabulary logits at prefill, compared with independent HF |
| C03 | Teacher-forced cached decode versus HF at every fixed step and each declared batch size; observed active batch sizes and the declared graph execution path |
| C04 | Unmodified greedy output rows exactly equal HF, serially and at the largest declared batch |
| C05 | Reference-voice codes, mel features and speaker embedding compared with independent HF |
| C06 | Fixed-code offline Decoder PCM with frozen HF conditioning and stochastic inputs at 24/32/511/512 codes, compared with independent HF |
| C07 | Supported HTTP requests, malformed input/budget/reference and unsupported streaming rejection, then successful recovery |
| C08 | Distinct serial requests versus overlapping concurrent requests, complete PCM equality and a repeat after concurrency |

For BF16 AR, compare FP32-converted full vocabulary vectors using `abs(actual-reference) <= 0.25 + 0.02*abs(reference)`, and require exact raw top1 identity separately. This is the previously registered engineering tolerance, not a quality score or a reason to ignore greedy divergence. Integer grids/codes are exact. Codec FP32 features and PCM use `atol=rtol=1e-5`; shape, dtype and finite values are mandatory. No cropping, padding, NaN equality or tolerance changes after observing results are allowed.

The canonical oracle is official HF cached inference. A second fresh-cache run must reproduce every full vocabulary vector exactly before admission. Complete-prefix recomputation is retained as a diagnostic: it can select a different BF16 attention kernel and is not silently substituted for the cached inference oracle. Development observed such HF-internal differences; no candidate tolerance was increased. The candidate teacher-forcing probe records logits before sampling, then feeds the fixed reference rows back through the production scheduler. It does not replace attention or model layers. C04 separately runs the candidate sampler without forced feedback, preventing teacher-forcing from being mistaken for generation equivalence.

C05 compares the actual reference-voice frontend. Integer prompt codes are serialized as int64 without changing inference inputs. C06 isolates the Decoder by feeding identical HF voice tensors, the frozen CausalCFM initial noise and a recorded trace of every Torch random draw into both implementations. It requires matching random-call function/shape/dtype/device and full trace consumption. Equal seeds alone are insufficient across framework versions. No layers or weights are replaced; timed inference uses the ordinary service.

Graph evidence observes SGLang's real decode executor and active request count. A replay of a padded graph is not evidence that its padded capacity was the active batch. HTTP overlap is not engine-batch evidence. Probe instrumentation is absent from timed service inference. Runtime settings must describe the actual deployment; no eager or alternate-attention fallback is used to manufacture coverage.

A numerical failure alone does not attribute the cause to the model adapter. Torch/CUDA, ONNX Runtime and attention-kernel changes can alter these vectors, greedy rows or PCM; the v1 binding does not require candidate and HF library versions to match. Record actual module versions, including ONNX Runtime when both sessions use the CPU provider, and isolate each factor with separate same-stack controls. Preserve the original oracle and failed certificate. A changed reference, backend or injected frontend is a diagnostic configuration, not a retroactive pass or a tolerance adjustment.

These checks qualify the declared cases, compute path and API. C08 does not prove immediate server KV reclamation or internal cancellation acknowledgement, and a buffered WAV API does not reveal completion-token usage or natural termination. Unreported fields remain unknown.

## Prepare and freeze the independent reference

All commands below run from the workspace root. Paths in the specification are relative to its `workspace_root`, itself relative to the specification file. Interpreter paths preserve virtual-environment symlinks. The result paths are relative to the shell working directory. No models are downloaded and no GPU allocation or service is created by the client.

Copy and edit [hf-regression.json](examples/hf-regression.json) for existing local model/Codec checkpoints, official Codec source dependencies, the actual candidate checkout and interpreters. Declare authorized local GPU indices or use an existing Slurm allocation. Preparation also writes `deployment.expected.json`: it is a comparison template, not an observed service record. The operator must verify the real deployment and supply its `server.commit/dirty` and `hf_regression` asset/runtime binding before formal inference. Preparation hashes every local model/Codec asset, the reference voice and external Python source closure before any inference, and freezes the candidate SHA when omitted:

```bash
moss-speech-bench --phase hf-prepare --config benchmark-client/examples/hf-regression.json \
  --output-dir benchmark-results/hf-setup/execution

moss-speech-bench --phase hf-reference \
  --correctness-config benchmark-results/hf-setup/execution/correctness.json \
  --output-dir benchmark-results/hf-reference
```

Each reference contains AR grids/full logits/greedy rows, Codec features/PCM, source/runtime metadata, the exporter snapshot and SHA256 file ledger. The HF process asserts that it imported no candidate or SGLang module. Checkpoint paths may move; stable asset IDs and their hashes, inputs, system, seed, decoder vectors and numerical policy must remain identical. A different candidate SHA or candidate attention backend does not redefine the HF oracle.

Use a new reference directory for a new binding. Failed exports and partial worker logs are retained; they are not admitted as references. Reference export alone does not qualify a candidate.

## Optional HF FlashInfer reference

The default HF backend remains SDPA. An explicitly authorized comparison can set `reference_attention_backend: "flashinfer"` in the correctness specification and export a separate reference directory. The field is part of the immutable oracle binding; this reference cannot be substituted for an existing SDPA reference. Existing references without the field retain their original binding and files.

The FlashInfer hook uses the official HF [attention registry and matching mask registry](https://huggingface.co/docs/transformers/attention_interface). Official model source files, projections, normalization, RoPE, DynamicCache and generation remain intact. The hook independently invokes FlashInfer's FA2 ragged-prefill and paged-decode wrappers, using NHD, page size one and tensor-core decode. It imports no SGLang or candidate code. It qualifies unpadded BF16 B1 HF reference requests, whose frozen traces are compared with candidate B1/B2/B4 execution; it does not implement general HF batch scheduling.

The hook validates the full bottom-right causal mask and rejects unsupported padding, batch shapes, sliding windows, training and dropout. There is no SDPA fallback. Reference metadata records the FlashInfer version, actual prefill/decode call counts and a hashed hook snapshot. C00 verifies the snapshot and both observed paths. AR comparisons require the same Torch and FlashInfer versions; Codec comparisons require matching Torch, Torchaudio and ONNX Runtime versions. The existing numerical thresholds, full-vocabulary/raw-top1 checks and exact greedy checks remain unchanged.

A pass under this binding establishes HF+FlashInfer equivalence for the recorded candidate, configuration and cases. Keep the original SDPA comparison available with its actual status. An observed implementation failure, required mismatch or incomplete gate still blocks formal Quick and release.

## Correctness, performance and offline quality

Pre-service numerical checks can be run first, keeping GPU capacity available to the isolated workers:

```bash
moss-speech-bench --phase correctness \
  --correctness-config benchmark-results/hf-setup/execution/correctness.json \
  --checks C00,C01,C02,C03,C04,C05,C06 \
  --output-dir benchmark-results/candidate-quick/correctness
```

Start the actual candidate service outside the client. Match its model assets, dtype, attention, graph policy, capacity, reference voice and full SHA in the resolved deployment record. Update [spoken-response.json](examples/spoken-response.json) for the three existing OpenAudioBench CSVs and the same voice. Correctness and timed workload must declare the same API, system, tasks, streaming policy and voice identity:

```bash
moss-speech-bench --mode quick --config benchmark-client/examples/spoken-response.json \
  --model moss-speech --base-url http://127.0.0.1:8000 --no-stream \
  --deployment deployment.local.json \
  --correctness-config benchmark-results/hf-setup/execution/correctness.json \
  --output-dir benchmark-results/candidate-quick

# Stop the generation service externally before loading scoring models.
moss-speech-bench --phase score --output-dir benchmark-results/candidate-quick \
  --score-config scoring.local.json
moss-speech-bench --phase report --output-dir benchmark-results/candidate-quick
```

Quick retains three datasets, stable 20-question subsets, concurrency four, three repeats and per-repeat warmup. One declared T2S path therefore sends 180 formal and 36 warmup requests. T2S answers carry `quality_task=qa`: ASR then semantic Judge, plus UTMOS. Custom reading tasks retain their existing WER/CER/SIM interpretation; the input question must not become the transcript target for a spoken answer. Missing finish reason/usage and non-streaming first-event times are N/A, never fabricated values.

The client can execute independent numerical gates even if another numerical gate fails, yielding a useful complete diagnosis. Only valid C00 permits dependent work; only all C00–C08 passing permits formal inference. A missing worker artifact, zero comparisons, unsupported path, required failure or incomplete gate blocks performance. No `--checks` subset substitutes for a complete certificate.

## Identity, replay and historical compatibility

Every run freezes the candidate/client identities, input configuration, reference hashes, tensor comparisons, actual graph/batch observations, raw requests, responses and failures. The correctness certificate is hash-bound to inference and revalidated by score/report. Resume requires unchanged configuration/client source and valid artifacts; failed attempts remain in history. It never recomputes a golden from candidate output or upgrades old evidence to a new pass.

Configurations without `protocol=hf-regression-v1` retain the [legacy providers](LEGACY_CORRECTNESS.md). Their historical test allowlists, numerical algorithms and reports remain available for reproduction. They are not the recommended entry for new model APIs. Backend adapter changes, candidate code/config changes or additional modalities require the affected checks again; same-name branches do not confer coverage.

`codec-storage-bf16` qualifies the existing selected Linear compute policy with real BF16 inference-device parameter storage. It requires the BF16 approximate contract, SDPA and residency features, storage activation, positive BF16 bytes, CPU-only exact FP32 backups and matching parameter/buffer pointers. CPU/CUDA storage/restoration tests and the full cumulative Codec proof are mandatory; numerical and quality thresholds are unchanged.

`codec-encoder-bf16` additionally requires the existing storage/SDPA/residency features, explicit Flow encoder activation, positive Linear and convolution BF16 execution counts and FP32 normalization/output/mask proof. CPU/CUDA mask/RNG/storage/restoration and failed-entry tests are declared independently; existing numerical and quality gates remain unchanged.
