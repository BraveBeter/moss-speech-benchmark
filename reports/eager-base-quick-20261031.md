# Eager base Quick 实测：2026-10-31

120/120 条推理完成，最终 120/120 条所需质量指标均完成，无未执行或预算耗尽。音频转录 90/90 成功，实际 HTTP 并发峰值 32。生成只运行一次，全部质量评分读取其保存输出。

**时间结论：正式推理 + 最终成功评分共 20 分 29 秒，低于 30 分钟；包含本次加载、排错与重试，实际历时 30 分 55 秒，未达到整个流程 30 分钟的目标。** 两个成功作业的工作时间合计约 23 分 14 秒，但它们不是一次连续无故障复跑，不能据此承诺下次一定达标。

## 配置与数据

- 服务固定 `084f62ef13c65fbf33f0c9a3f2cce5e87e0c01d6`，原 eager base 干净且未修改。单张 A800-SXM4-80GB，BF16 AR / FP32 Codec，torch_native，CUDA Graph 关闭，chunk=25，C4，一轮，无预热。
- PyTorch 2.14.1+cu130（CUDA 13.0，来自实际服务环境版本文件）、Transformers 5.19.0、SGLang 0.5.22；seed=42、temperature=0、top_k=1、max_tokens=512。固定服务音色，不传逐条克隆提示。
- Quick 推理使用已安装的 `8c0a02f` wheel；评分使用 `3746e26` 的不可变源码快照。二者各自源码 hash、模型/Codec revision、完整选样 ID、数据文件 hash、评分资产 hash 见[机器可读收据](eager-base-quick-20261031.json)。后续提交只整理文档、示例与测试 import，不改变这两次实际运行的生产源码身份。

| 数据 | 原集合数量 | 固定子集 | revision / split |
|---|---:|---:|---|
| seed_en | 1088 | 30 | `27f4c1adee83b5b29b7c4b375f6b976324bda308` / en |
| seed_zh | 2020 | 30 | `27f4c1adee83b5b29b7c4b375f6b976324bda308` / zh |
| storycloze | 1871 | 30 | `d22567c475c216b1a0bde785e4c0508d45d00d25` / 2016-test |
| llama_questions | 300 | 30 | `98f8fa0d5fa7e530ec8e6f097b55f10b003d111b` / llama_questions |

按 sha256-id-v1 / subset seed=42 从完整清单选样；不是 Daily 全量、Full 扫描或随机重新抽样。未在本轮重跑模型正确性/数值对齐，也未测双卡和其他并发。

## 耗时

| 阶段/口径 | 秒 |
|---|---:|
| 正式推理调用（含客户端准备与报告） | 1167.579 |
| 最终评分调用（含资产校验、模型加载、ASR、SIM、UTMOS） | 61.372 |
| 上述两项相加 | 1228.951 |
| MOSS 服务启动 | 72.395 |
| 成功评分服务启动（已有编译缓存） | 68.856 |
| 评分服务首次启动（含首次编译） | 227.344 |
| 两个成功作业工作时间之和，非连续复跑 | 1393.678 |
| 成功推理作业启动至最终评分收尾，含排错/重试/间隔 | 1854.921 |
| 从最初失败的服务启动尝试算起 | 2028.735 |

计时起点是对应服务/客户端调用；此前的下载、环境初次创建及后续推送没有另行累计。实际历时包含起点之后的全部等待、依赖修复与重试，不扣除排错时间。本轮从最初服务尝试算起共 33 分 49 秒；排错后未重复生成以制造一次更好看的计时。评分默认 300 秒预算内完成，推理默认 1500 秒预算内完成。

## 性能

延迟/首事件单位秒，RTF 无单位。每行分位数按该组 30 条请求分别计算；TTFA 为首个非空 PCM SSE 事件，可能含静音。TTFC 原生 Codec 边界未暴露，记 N/A。文本 delta 口径的 TTFT 只在 T2T 可观察。

| 数据 | 指标 | Mean | P50 | P95 | P99 |
|---|---|---:|---:|---:|---:|
| seed_en | Latency | 20.6430 | 19.8670 | 32.3030 | 45.1660 |
| seed_en | TTFA | 12.4585 | 11.8154 | 16.1300 | 17.7826 |
| seed_en | RTF | 4.3427 | 4.3106 | 4.9517 | 5.0898 |
| seed_zh | Latency | 28.1450 | 20.8250 | 69.1200 | 117.3600 |
| seed_zh | TTFA | 12.0499 | 11.5157 | 14.9555 | 16.2458 |
| seed_zh | RTF | 4.2674 | 4.2751 | 5.1260 | 5.2378 |
| storycloze | Latency | 5.4830 | 6.3610 | 9.4150 | 12.2680 |
| storycloze | TTFT | 0.1922 | 0.1761 | 0.2789 | 0.2790 |
| llama_questions | Latency | 92.7000 | 81.6470 | 206.7640 | 219.4970 |
| llama_questions | TTFA | 14.7506 | 14.7908 | 18.9195 | 22.1435 |
| llama_questions | RTF | 4.9132 | 4.8769 | 5.5755 | 6.1554 |

| 数据 | 推理段墙钟(s) | 成功 / 预期 | QPS | audio-s/s | length 截断 |
|---|---:|---:|---:|---:|---:|
| seed_en | 158.907 | 30/30 | 0.1888 | 0.91 | 0 |
| seed_zh | 235.809 | 30/30 | 0.1272 | 0.874 | 0 |
| storycloze | 43.414 | 30/30 | 0.6910 | N/A | 0 |
| llama_questions | 728.734 | 30/30 | 0.0412 | 0.774 | 2 |

RTF 使用每请求完成耗时 / 实际 PCM 时长，再求分布；audio-s/s 使用整组成功音频总时长 / 整组墙钟。120 条无 HTTP/生成失败；2 条 S2S length 输出仍参与全部统计和质量评分。当前音频生成慢于实时；这是 eager base 单次 C4 回归基线，不是优化收益或稳定性结论。

## 质量

ASR 对齐[官方流程](https://github.com/sgl-project/sglang-omni/blob/305273a673c281f2d214e1bed457a4b5b8895740/benchmarks/tasks/asr.py)：Qwen/Qwen3-ASR-1.7B，中英共用服务，multipart `/v1/audio/transcriptions`，不显式覆盖默认 greedy 温度，并发 32。核对时官方 main 文件与固定 base 文件的 SHA256 相同（`f0e32c26b045b04d7e9c21da05d518ab2bac38f9804549e9a595e941b9d8d7af`）。Whisper 仅提供英文文本归一化规则，没有加载 Whisper 识别模型。

| 数据 | 指标 | 分数 | 完成覆盖 |
|---|---|---:|---:|
| seed_en | SIM | 0.6633 | 30/30 |
| seed_en | WER | 34.10% | 30/30 |
| seed_zh | CER | 83.49% | 30/30 |
| seed_zh | SIM | 0.7543 | 30/30 |
| storycloze | ACCURACY | 96.67% | 30/30 |
| llama_questions | 严格答案 EM（工程协议） | 0.00% | 30/30 |
| llama_questions | UTMOS | 4.0374 | 30/30 |

WER/CER 主值是 corpus 编辑错误数 / 参考词数或字数，分别为 34.10% / 83.49%；逐样本均值另存 JSON（27.25% / 92.63%）。SIM 为实际固定音色的平均余弦相似度，未采用 SeedTTS 逐样本克隆音色；UTMOS 为 30 条输出的平均值。每项均无 failed/unavailable/budget_exhausted，最终 quality.status=complete。

**评分完成不表示模型质量合格。** 当前 TTS 提示无法保证严格逐字朗读：例如中文样本 `10003331-00000019` 的转录是英文译文，另有回复而非朗读的样本。需要后续听辨与 HF 对照定位生成行为和识别误差，不能直接与标准 SeedTTS 论文分数比较。

S2S 严格 EM 为 0/30，不能解释成语义问答正确率为零。例如参考为 Moscow 时，回答包含 “Moscow is the capital of Russia” 和后续解释，严格整句匹配仍计错。后续需要单独定义答案抽取或官方语义评分协议，再建立新 baseline；本轮没有为改善分数修改匹配规则。StoryCloze 为 29/30，均可解析。

## 失败记录与处理

| 作业 | 实际结果 | 处理 |
|---|---|---|
| 77805 | 服务加载后外层控制器导入失败，未运行正式推理 | 修复外层路径，不改模型；保留失败及清理记录 |
| 77807 | 120/120 正式推理成功 | 后续所有评分共用这次输出 |
| 77809 | 评分子进程退出 1：独立环境缺 aiohttp；无转录/质量记录 | 补齐独立环境中的客户端依赖，CPU 作业 77810 完整入口检查通过 |
| 77811 | 评分子进程退出 1，quality partial：5 条长音频被 ASR HTTP503 拒绝 | 旧 quality/summary 已归档，所有 90 条转录记录保留 |
| 77812 | 评分子进程退出 0，120/120 完整评分 | 新配置 max_concurrent_long_audio_requests=32、max_concurrent_chunks=1；原 30 秒分块语义保留，显式新 attempt，无静默重试 |

ASR 端 max_running_requests=16 时默认长音频容量仅 1；它与客户端 HTTP 并发不同。已增加[外层 A800 配置示例](../examples/qwen3-asr-server.json)，记录服务容量、代码和模型身份。原 85 成功 + 5 失败记录未删除；最终 attempt 新建 90 成功记录。Slurm 外层作业结束码不代替评分子进程退出码或 quality.status。

## 验证和证据

- 41 项客户端测试通过，3 项可选 Torch 数值政策测试在基础客户端环境跳过；Ruff、格式和 diff 检查通过，wheel/sdist 构建成功，新环境普通安装后离线重建的 summary 与原报告完全相同。UTMOS 的 soundfile 依赖已显式声明。这些不冒称本轮模型正确性测试。
- 独立复核 120 条输出 hash、WAV 实际时长、RTF、延迟 Mean/P50/P95/P99、QPS、audio-s/s、质量缓存 hash、全部质量聚合值及最终 90 条 ASR 的真实并发。评分前后 performance.json hash 一致，评分源码快照与当前生产 Python 文件逐字节相同。
- 本轮 8 个 Slurm 作业全部终态，其中 5 个 GPU 作业共 1,904 GPU 秒（0.528889 卡时，含失败），最多同时 1 卡。累计 10.184444/50 卡时，余 39.815556。自有服务/评分进程组 active=[]、zombies=[]，无待结算作业。

原始结果留在工作区 `.audit/quick-real-20261031/`：`infer-02/quick/` 保存 manifest、输入、120 条事件/文本/WAV、性能与最终质量报告；`score-02/partial-results/` 保存不完整评分快照；`score-03/` 保存最终启动/评分配置及日志。另有 `verify-final.log`、`verified-results.json`、`timing-final.json`、`accounting-final.json`、固定下载清单和环境记录。GitHub 仅提交本文和轻量 JSON 收据，不上传权重、原始音频或大体积实验缓存。

## 复跑

1. 外层用固定 eager base、部署配置和数据源启动 MOSS 服务；按 README 的 Quick infer 命令运行同一份 workload，默认 C4/30 条每集。
2. 释放 MOSS 服务，外层启动 Qwen3-ASR；替换示例里的权重路径、服务地址/ID，使用一致的长音频容量，保留 SIM 实际音色参考 hash。首次准备评分环境须同时安装客户端基础依赖和对应评分依赖。
3. 对同一 output-dir 执行 `--phase score --score-config ...`，再 `--phase report`；无须重发 MOSS 请求。原始执行命令和完整配置保存在上述 control/manifest 文件。

尚未验证：无故障连续重跑的整个 30 分钟上限、Daily/Full 全量、语义 QA 协议、标准 TTS 朗读/克隆协议及优化分支比较。
