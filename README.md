# MOSS-Speech benchmark client

独立客户端，连接已启动的 SGLang-Omni `/v1/chat/completions` 服务。
支持性能测试与基于已保存输出的离线质量评分。模型正确性测试保持独立。

复用用户指定 `tts-serving-benchmark` 中提取的 SGLang-Omni runner、SeedTTS loader、性能和播放连续性算法；新增 MOSS chat/SSE 适配与实验记录。来源和修改见 [ORIGIN.md](ORIGIN.md)。客户端不依赖 Torch/CUDA，不管理服务进程。

## 安装与首次运行

Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# 只检查数据与运行规模，不请求服务
moss-speech-bench --config examples/custom.json --model moss-speech \
  --output-dir results/smoke --dry-run

# 服务需由外层启动；示例两条数据仅用于接入检查
moss-speech-bench --config examples/custom.json --model moss-speech \
  --base-url http://127.0.0.1:8000 --concurrency 1 --output-dir results/smoke
```

`python -m moss_speech_bench` 与命令行入口等价。`--base-url` 只接受协议、主机与端口，不附加 `/v1`。认证用 `--api-key-env VARIABLE_NAME`；密钥不写入产物。

## 数据和模式

配置格式见 [custom.json](examples/custom.json) 和 [daily.json](examples/daily.json)。输入文件路径相对于配置文件；示例 Daily 数据路径须替换成已获得的真实资产。程序不会下载缺失的本地数据，不提供截取前 N 条或随机抽样选项。

| 模式 | 数据 | 默认并发 / 重复 | 正式请求数 |
|---|---|---|---:|
| custom | 配置选择的数据全量 | C4 / 1 | 配置决定 |
| daily | Seed EN 1,088 + ZH 2,020 + StoryCloze 1,871 + LLaMA Questions 300 | C4 / 1 | 5,279 |
| full（当前仅性能） | Seed EN、StoryCloze、LLaMA Questions | C1/C4/C8/C16 / 3 | 39,108 |

Daily 校验四组输入的 ID、模态、语言和条数；Full 只要求并调度三个性能集，也接受同一份 Daily 配置（不加载未调度的 ZH 数据）。`--repeats`、`--concurrencies` 可修改 Full 策略。Daily 限定一个并发、一次正式推理。条数校验不能证明数据是官方原版；必须同时核对 revision、split 和输入 hash。自定义的同名数据不可冒称官方集。

支持的读取方式：

- `seedtts`：Seed-TTS `meta.lst` 或固定 40 位 commit revision 的 HF 数据集（额外安装 `.[hf]`）。复用原 loader。目标文本前加固定逐字朗读指令；当前使用服务固定音色，不传逐样本参考语音。未来 SIM 必须使用实际音色参考，不能声称标准 voice-clone 协议。
- `storycloze`：2016 test CSV，四句故事 + A/B 结局。标准列 `InputStoryid`、`InputSentence1..4`、`RandomFifthSentenceQuiz1/2`、`AnswerRightEnding`。保存 A/B 标签；属于纯文本工程协议。
- `openaudio`：CSV 的 `audio_filename` 与 `answer_normalized_aliases`（也接受 `Answer` / `answers`）及 `audio_root`。音频在正式计时前读入并编码；保留答案供后续评分。
- `json`：列表，每条含唯一 `id`，文本输入用 `text`，语音输入用 `audio` 文件路径；可提供 `expected`。支持 `t2t/t2s/s2t/s2s` 和 `en/zh`。

```bash
moss-speech-bench --mode daily --config examples/daily.json --model moss-speech \
  --base-url http://127.0.0.1:8000 --deployment deployment.json \
  --output-dir results/base-daily

moss-speech-bench --mode full --config examples/daily.json --model moss-speech \
  --base-url http://127.0.0.1:8000 --deployment deployment.json \
  --output-dir results/base-full
```

`deployment.json` 由启动服务的一方提供，记录服务 Git SHA/dirty、模型和 Codec revision/hash、GPU 数量/型号、Driver/CUDA/PyTorch/SGLang 版本，以及 dtype、kernel、cache、chunk、voice、调度配置。见 [deployment.example.json](examples/deployment.example.json)；示例占位符必须换成实值。客户端无法验证远端硬件或推断其版本，不以客户端本机信息代替。

默认不预热。可在配置中增加一个与 `datasets` 单项同格式的 `warmup`；该集合在每个数据集/并发/重复前全部执行，另存输出，不计正式统计。其 payload 必须与正式集合不相同，任何预热失败均停止正式测量。轮次之间不重启服务。

## 指标口径

| 指标 | 口径 |
|---|---|
| TTFA (`audio_ttfp_*`) | 发请求至第一个完整 SSE 事件中非空有效 PCM 到达；可能含静音 |
| TTFT (`text_ttft_*`) | 发请求至首个非空文本 delta；客户端不能观测 delta 内 token 的独立到达时间 |
| TTFC | 原生 Codec chunk 边界未由当前协议暴露，记 N/A |
| Latency | 请求发送至完整响应接收结束；不含等待客户端并发名额的时间 |
| RTF | 每请求 Latency / 实际生成的 PCM 时长 |
| audio-s/s / QPS | 成功音频总秒数 / 成功请求数，分别除以完整测量段墙钟 |
| Output tokens/s | 服务端 `usage.completion_tokens` / 测量段墙钟；未提供 usage 则 N/A |
| Inter-chunk、C50/C100/C200 | 按 SSE 音频事件，复用上游播放断流算法；单块连续性 N/A |
| GPU Peak Memory | 无服务端遥测，N/A |

延迟、TTFA、TTFT、RTF 保存 Mean/P50/P95/P99；分位数按单个数据集/并发/重复统计，不混合多个重复。成功请求用于这些分布，失败数和失败率单独保留。`finish_reason=length` 作为完成请求计入，但在 `length_count` 单列，不能解释成质量成功。

流式输出按 PCM16 little-endian、单声道解析，默认 24 kHz（可用 `--sample-rate` 显式声明）；应与服务配置一致。非流式使用服务默认 WAV 格式，不发送请求级 `audio` 配置；解析 PCM16 WAV，TTFA/TTFT 为 N/A。音频时长以实际字节数计算，WAV 保留无损 PCM。

正式墙钟包括请求派发、网络、解码和输出写盘；每请求延迟在写盘之前截止。文件 I/O 移至线程，避免直接阻塞其他请求的网络读取，但客户端 CPU/磁盘仍可能限制吞吐，应保持对照环境一致。`engine_time_s` 是复用数据结构中的请求耗时代理，不是服务端纯计算时间。当前不提供 token 间隔或 engine RTF。

## 结果、恢复与对比

```text
results/<run>/
├── manifest.json          # 客户端身份、依赖、部署声明、输入 hash、运行配置/状态
├── inputs/*.jsonl         # 完整请求 payload 和预期答案，语音输入含 base64
├── cells/<dataset>-c<C>-r<R>/attempt-01/
│   ├── timing.json        # 完整测量段墙钟、请求记录 hash
│   ├── warmup/            # 可选独立预热
│   └── outputs/           # 每请求 events.jsonl、txt、wav、result.json
├── requests.jsonl         # 每段最新 attempt 的请求记录，带 cell/attempt 定位
├── performance.json      # 全部指标及 Expected/Attempted/Succeeded/Failed/Scored
├── summary.json
├── summary.md
└── comparison.json / comparison.md  # 指定 baseline 时生成
```

部分输出、HTTP 错误、超时、空输出、异常结束都保留；默认没有重试。推理阶段的performance.json保持原始性能记录；质量评分另存quality.json，合并报告记录实际Scored和quality_expected。infer后通过score单独执行评分；all明确要求分阶段执行，以便外层释放推理服务资源。

```bash
# 原命令原配置增加 --resume
moss-speech-bench --config examples/custom.json --model moss-speech \
  --base-url http://127.0.0.1:8000 --concurrency 1 --output-dir results/smoke --resume

# 离线重建：不需要服务和原始数据目录
moss-speech-bench --phase report --output-dir results/base-daily

# 比较时只允许预先声明的实验变量变化
moss-speech-bench --phase report --output-dir results/candidate-daily \
  --baseline results/base-daily --allow-difference deployment.settings.cuda_graph
```

恢复只跳过完整且 hash 有效的测量段；中断段全量新建 `attempt-02` 重跑，保留原 attempt，不拼接两次墙钟。正式段中已完成的失败不会被悄悄重试。中断段保留已持久化请求的诊断延迟，吞吐记 N/A，未获得记录的请求列为 `not_attempted`。输入、输出或完整请求记录损坏会拒绝复用。

baseline 必须提供双方部署元信息并匹配完整实验约定；差异全列出，未获显式允许的差异阻止计算收益。允许差异表示有意改变的变量，不表示实验自动公平。各轮报告保留原始统计与失败/截断变化，不自动生成“优化有效”的结论。

退出码：0 完成，1 有失败或未完成样本，2 配置/产物错误或未实现阶段，130 用户中断。终止进程可能发生在写盘前，缺失记录在汇总中可见；不把未持久化的请求推定为成功。

## 开发验证与后续范围

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
```

本地 HTTP 测试覆盖 UTF-8/SSE 分片、流式/非流式、音频时长、错误/usage、并发/预热、取消/恢复、产物校验、baseline 和安装后的 CLI。这些是客户端测试，不是 MOSS 模型正确性验证。真实 A800 运行、全量数据验收和质量评分仍待后续阶段。

## 离线质量评分

评分直接读取运行目录的冻结输入、文本和WAV，检查hash，不向MOSS服务发请求。无需保留原始数据集目录或启动生成服务。

```bash
# 选择题文本评分只需基础安装；音频错误率需质量依赖和本地ASR权重
pip install -e '.[quality]'
# 按需安装中文ASR、WavLM SIM或UTMOS；Torch与torchaudio应匹配
pip install -e '.[zh,sim,utmos]'

moss-speech-bench --phase score --output-dir results/base-daily \
  --score-config examples/scoring.json
moss-speech-bench --phase report --output-dir results/base-daily
```

[scoring.json](examples/scoring.json) 中的路径须换成本地资产，均相对该配置文件定位。省略某个模型会给对应指标记录unavailable；配置了不存在的文件则报配置错误。默认CPU；使用CUDA需在Slurm作业中运行并遵守资源预算。ASR、SIM和UTMOS不能争用受测服务的GPU。Whisper读WAV通常还需要环境提供ffmpeg。

| 任务 | 指标和协议 |
|---|---|
| T2S EN | Whisper转录；复用上游EnglishTextNormalizer和jiwer WER |
| T2S ZH | 本地FunASR/Paraformer转录；复用上游中文标点归一化与逐字CER |
| T2S SIM | 复用上游WavLM-large SV；固定音色参考，与deployment.settings.voice_reference_sha256核对；输出cosine，原实现的百分值除以100 |
| T2T | 复用上游选择题答案提取；默认A/B，可配置choices为ABCD等；不能解析计错并记录invalid |
| S2T / S2S QA | 已保存文本或回答音频ASR；小写、英文标点/冠词/空白归一化后与可接受答案集exact match，工程回归协议 |
| S2S UTMOS | 复用上游16kHz重采样、PCM16和预测过程；加载本地utmos.jit |

EN Whisper large-v3和ZH Paraformer权重由外层预先准备，客户端不下载评分权重。SIM只支持实际固定音色参考，不使用未传入模型的SeedTTS参考音色；缺少音色身份则SIM unavailable。

WER/CER报告全部可评分样本的corpus错误率、逐样本均值及评分覆盖率，不排除高错误率样本。Accuracy的主值为正确数/所选集合总条数；另存evaluated_accuracy，未完成/失败样本保留。没有任何可评分样本时值为null。每项指标均记录scored/failed/unavailable/generation_failed/not_attempted数量。length输出参与评分，原finish_reason保留。

默认每个数据集只选一个性能测量段，优先C4/r1，否则选manifest中的第一个；quality.json显式记录selection。可用评分配置的cells列表指定其他测量段。Full性能扫描不会触发12次重复ASR；Full额外质量数据的获取与请求准备仍属于后续全量验收范围。

每样本结果保存在quality/<评分约定与输出hash>/<sample-hash>.json，保存ASR转录、归一化文本、编辑计数、分值、错误及校验hash。相同评分约定和输入会复用已保存评分，包括失败记录；换模型/依赖/配置会生成新缓存，旧记录保留，不隐式重试。中断后重新执行score可继续未保存样本。quality.json和quality.md汇总，report会校验缓存并合并至summary。summary中的scored表示所需指标全部完成的样本数；quality_expected是实际选中的质量样本数，与性能扫描总请求数区分。

baseline比较还会核对评分器源码、模型资产hash、依赖版本、规范化配置和音色身份；完整且约定一致的评分才计算delta，coverage变化不会被当作质量收益。错误率越小越好、Accuracy/SIM/UTMOS越大越好，尚不设置任意质量合格阈值。

本地测试验证评分协议、复用和归档行为；真实Whisper/Paraformer/WavLM/UTMOS权重及全量数据质量尚未验收。
