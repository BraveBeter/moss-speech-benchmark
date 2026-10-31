# MOSS-Speech benchmark client

独立客户端，连接外层已启动的 SGLang-Omni 服务。常规运行统一按 **Correctness → Inference/Performance → Offline Quality → Report** 执行；正式推理输出保存一次，评分不重发生成请求。客户端不下载模型、不管理服务或申请 GPU。

## 测试范围

| 模式 | 端到端数据 | 默认任务/请求 | Codec Decoder |
|---|---|---|---|
| Quick / Quickly | 三集各固定20个问题，可设20–50 | 同一问题各跑S2T、S2S，C4/一轮，共120请求 | 不运行 |
| Daily | LLaMA Questions 300、TriviaQA 1000、WebQuestions 1000，全量2300问题 | S2T+S2S，C4/一轮，共4600请求 | 不运行 |
| Full | 同上三个完整集合 | S2T+S2S，C1/C4/C8/C16、三轮，默认55200性能请求；评分选每个数据集/任务的一个测量段 | Seed-TTS-Eval EN1088+ZH2020，**deferred，未实现** |

Full 的 Decoder 流程尚未确定：只登记目标集合和暂缓原因，不加载 Seed 数据、不发送聊天请求代替 Codec 测试，也不声称整个 Full 通过。端到端部分正常执行，报告 `suite_status=incomplete`，实际执行退出码1；dry-run只展示计划，退出码0。

Seed-TTS-Eval仅预留给隔离的Codec Decoder评估。当前拒绝 `kind=seedtts` 的新聊天运行。StoryCloze及其他AR/Encoder数据不属于当前三个常规模式。旧数据集loader/protocol和原WER/CER结果的历史来源保留在Git及旧报告中，不将旧高错误率作为标准Decoder质量结论。

`--tasks s2s`或`--tasks s2t`可显式只测一条路径，默认两者。`--mode quickly`是`quick`别名，归档统一写quick。`custom`保留为显式开发/协议诊断入口，不代替常规模式验收。

## 安装与数据

Python 3.10+：

```bash
pip install -e '.[dev]'
# 在独立评分环境中，同时安装客户端基础依赖和所需可选依赖
pip install -e '.[quality,utmos]'
```

配置参考 [daily.json](examples/daily.json)，三种模式共享这份配置。替换本地CSV/audio_root和固定Dataset Revision。CSV支持OpenAudioBench原始 `audio_filename` 与 `Answer`、`answers`或`answer_normalized_aliases`；本地JSON列表也可用于开发。数量校验不等于官方版本校验，必须保留来源revision和文件hash。

Quick从三个完整CSV清单选样，先核验300/1000/1000总数与全部ID唯一，再按 `SHA256(dataset_id, subset_seed, sample_id)` 固定选20个，默认seed42。只读选中音频；S2T/S2S共享同一组ID及原音频字节，分别保存各自payload/hash。改变行序不改变子集；换子集、任务、提示词、预算须建立新baseline。

数据源及公开规模见[固定OpenAudioBench版本](https://huggingface.co/datasets/baichuan-inc/OpenAudioBench/blob/98f8fa0d5fa7e530ec8e6f097b55f10b003d111b/README.md)。不在CLI中隐式下载数据或修改数据。

## 正确性先行

先设置结果根的绝对路径，避免从不同checkout/worktree启动时写到别处。本节点使用：

```bash
MOSS_BENCHMARK_RESULTS_ROOT=/remote-home1/xrluan/speech-rep/benchmark-results
```

其他机器仅修改该工作区配置，不把运行位置写死进客户端代码。

Quick、Daily、Full的真实infer都必须提供 [correctness.json](examples/correctness.json) 的本机配置。固定C00–C08全部通过后，才发送任何正式性能请求；失败/缺项/unsupported停止后续推理，不能用 `--checks` 绕过。正确性配置的服务地址必须与受测地址一致。

详细覆盖、独立HF golden与Slurm环境见[CORRECTNESS.md](CORRECTNESS.md)。所有模型测试和fixture仍在目标仓库；客户端通过隔离provider调用，不导入模型计算代码。当前正确性profile限定固定eager base，不能拿它直接认证其他模型配置。

可以由外层在启动服务前，通过同一运行目录的`correctness/`执行C00–C06；服务就绪后，infer会校验其源码/资产/配置身份，完成C07/C08。只复用同目录内且身份、产物hash有效的已通过gate，失败不自动重试。独立完整运行也保留：

```bash
moss-speech-bench --phase correctness --correctness-config correctness.local.json \
  --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/correctness-only/correctness"
```

infer保存正确性JSON的hash，score/report会重新检查九项状态及产物hash。缺失、损坏或过期证明不能继续评分。旧格式历史结果仍可离线重建，不能把其没有前置证明解释为新套件通过。

## 运行

以下命令在客户端checkout目录执行，输出统一使用上面的绝对结果根；从worktree启动也不改变结果位置。禁止将评测产物保存到`.audit/`或源码仓库中的`reports/`、`results/`。

```bash
# 只验证数据和请求规模，不运行正确性、不请求服务
moss-speech-bench --mode quick --config examples/daily.json --model moss-speech \
  --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/base-quick" --dry-run

# 正确性通过后执行120条正式请求
moss-speech-bench --mode quick --config examples/daily.json --model moss-speech \
  --base-url http://127.0.0.1:8000 --deployment deployment.local.json \
  --correctness-config correctness.local.json --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/base-quick"

# Daily/Full替换--mode；Full可显式修改扫描规模
moss-speech-bench --mode full --config examples/daily.json --model moss-speech \
  --base-url http://127.0.0.1:8000 --deployment deployment.local.json \
  --correctness-config correctness.local.json --concurrencies 1,4,8,16 --repeats 3 \
  --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/base-full"

# 外层释放MOSS服务，准备ASR服务后，复用原输出评分
moss-speech-bench --phase score --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/base-quick" \
  --score-config examples/scoring.json
moss-speech-bench --phase report --output-dir "$MOSS_BENCHMARK_RESULTS_ROOT/base-quick"
```

`python -m moss_speech_bench`等价。`--base-url`只传HTTP(S)主机与端口，不带`/v1`；密钥用`--api-key-env VARIABLE_NAME`，不进入归档。`all`要求拆分infer/score，避免评分模型争用受测GPU。

[deployment.example.json](examples/deployment.example.json)由服务启动方填写：实际源码提交/dirty、权重版本/hash、GPU/软件版本、dtype/kernel/cache/chunk/voice/调度；不以客户端本机信息冒充远端部署。

每份 `summary.md` 开头分别显示**被测服务代码**与**评测客户端代码**的运行时分支、完整 commit SHA 和工作区状态。服务身份使用 `--deployment` 的 `server.branch/commit/dirty`；客户端身份在推理时冻结。离线重建报告不读取当前 checkout；字段缺失显示“未记录”，wheel 无 Git 信息时另列 source_hash。 detached HEAD 明确标注。历史身份可在所属 run 的 `execution/records/code-identity.json` 补记 `server`/`client` 和证据来源，须以 `manifest_sha256` 绑定原 manifest，不能覆盖已记录的身份或修改原评测数据。

Quick目标为包含前置正确性、推理、评分及服务准备在内约40分钟。默认infer调用预算2040秒（34分，含此调用中的正确性）、score预算180秒（3分），为外层准备留约3分；`--time-budget SECONDS`覆盖当前阶段。外层预先执行的gate、服务启动/排队不由客户端控制，必须在实际总时间中计入；单次评分模型调用也可能超过软预算。不能承诺所有硬件/首次编译都达标。需要硬截止时由外层Slurm时限控制；未完成一律partial，不以取消样本制造通过。

可用`--samples-per-dataset 30`改成180请求，50改成300请求，40分钟目标需按实际负载重验。采样/最大输出预算保留配置值，不暗中减少生成长度。

## 质量协议

[PROTOCOLS.md](PROTOCOLS.md)定义当前 `moss-spoken-qa-v2` 与历史协议边界。三个集合默认使用相同的 `spoken-qa-short-en-v1` system提示：只给简短英文答案，不重复问题或解释；不包含参考答案。可显式配置prompt，内容/hash进入归档。

| 路径 | 离线指标 | 局限 |
|---|---|---|
| S2T | normalized whole-answer EM，保存名accuracy | 不是语义judge，正确解释也可能匹配失败 |
| S2S | Qwen3-ASR转录后同一EM；UTMOS | EM不证明语义能力，UTMOS不证明答题正确 |
| Codec Decoder | 预留WER/CER、SIM、DNSMOS | 未实现，不从端到端QA或聊天朗读推导 |

Qwen3-ASR-1.7B统一走官方CI同源HTTP协议，默认并发32；示例外层A800配置见[qwen3-asr-server.json](examples/qwen3-asr-server.json)。长音频准入容量与客户端并发需匹配。客户端不启动ASR服务；score只上传已保存的回答WAV，不发送新的MOSS请求。Whisper依赖仅用于旧WER英文归一化，没有默认加载其ASR模型。

主评分示例仅需英文ASR和UTMOS。SIM/双语ASR及原WER/CER实现保留供历史/custom报告复用，不进入当前三个端到端集合。可选`qa_answer_mention`仅是诊断，不能替代EM或称为官方QA Accuracy；语义评判器需独立定协议、版本及验证。

每个数据集/任务默认只评分一个测量段，优先C4/r1，避免Full重复12次评分。结果保存每样本转录、值、失败及hash；Accuracy主分母包含全部预期样本。缓存身份包含输入、源码、模型/依赖、提示词和部署声明。失败不静默重试，预算占位单独保留，可恢复未完成评分。

## 指标、产物与恢复

- TTFA：首个非空有效PCM SSE事件，可能含静音；TTFT：首个非空文本delta。原生Codec边界TTFC及无遥测的GPU Peak Memory为N/A；非流式首事件为N/A。
- Latency：发送请求至完整接收，不含客户端并发排队；RTF：每请求Latency/实际PCM时长；audio-s/s及QPS分别以完整测量段墙钟为分母。保存Mean/P50/P95/P99、length截断、失败/未执行、usage及流式连续性指标。
- 流式解析PCM16 little-endian单声道，默认24kHz，需与服务一致；非流式按默认WAV协议。事件/输出写盘属于整段墙钟，客户端CPU/磁盘可能限制吞吐。

每次运行只有一个`benchmark-results/<run_id>/`，以根`summary.md`为唯一主入口。每个run保存 `manifest.json`、`correctness/`、`inputs/`、分数据集/任务/并发/轮次的`cells/`、`requests.jsonl`、原始文本/WAV/事件、`performance.json`、`quality.json`及`summary.md/json`。Full另显示Codec deferred及不完整状态。性能原始文件不因质量评分而改变。外层脚本、配置、环境/Slurm日志和验证记录放本run的`execution/`；必要的详细分析放`analysis/`，不在源码仓库另存报告副本。

`--resume`仅跳过完整且hash有效的测量段；中断段新建attempt全量重跑，不拼接吞吐墙钟。原错误/部分输出保留。baseline通过`--baseline PATH`比较，核对完整输入/任务/部署/采样/指标及评分约定；实验变量需显式`--allow-difference FIELD`。不同suite版本不自动比较。

退出码：0当前阶段完成，1失败/partial/Full含deferred，2配置或产物错误，130中断。报告重生成无需生成或ASR服务，正确性产物须完整保留。

## 开发与历史证据

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
```

复用runner、性能与播放算法的来源/许可见[ORIGIN.md](ORIGIN.md)。客户端测试验证调度、门槛、协议、计时、校验和恢复，不冒称模型质量或数值正确性。

历史运行`eager-base-quick-20261031`与`quality-diagnosis-20261031`保持原始身份；其Seed聊天朗读与StoryCloze不属于当前套件，旧30分钟目标也不适用于新配置。

全部本地结果从`$MOSS_BENCHMARK_RESULTS_ROOT/README.md`查看，各run只打开其`summary.md`；运行目录不纳入客户端Git。历史脚本/绝对路径保留执行时身份，迁移后的当前位置以结果索引及迁移清单为准。
