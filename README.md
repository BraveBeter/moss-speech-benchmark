# MOSS-Speech benchmark client

独立客户端，连接已启动的 SGLang-Omni `/v1/chat/completions` 服务。
当前实现性能测试；后续在已保存输出上增加质量评分。模型正确性测试保持独立。

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

部分输出、HTTP 错误、超时、空输出、异常结束都保留；默认没有重试。`scored=0`、`quality_status=not_implemented`。`score` 和 `all` 当前明确报未实现。

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

质量阶段将复用本次输出增加 WER/CER、固定音色 SIM、QA Accuracy 和 UTMOS；ASR/评分模型的版本、规范化规则、覆盖率与失败样本会单独记录。
