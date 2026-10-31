# 三集双路径 Quick：2026-10-31

当前客户端候选 `fd99062a9d659d3ec986d33c1a56b8db6368f007`，eager base `084f62ef13c65fbf33f0c9a3f2cce5e87e0c01d6`，单张A800、C4、一轮。三个完整CSV各固定选20个问题，S2T/S2S共享问题及输入音频，共60问题、120请求；新套件为moss-spoken-qa-v2。

**实际总历时 1087.555 秒（18.13 分钟），包含本次C00–C08、服务启动/切换、正式推理、评分与收尾，低于40分钟。** 权重/数据预先准备，不包含最初下载及本次代码开发。没有重跑生成、删除失败或暗改max_tokens。

**已有正确性九项全部pass；正式请求120条全部尝试，成功114、失败6。模型整轮验收为 incomplete。** 评分保存120条覆盖记录，成功输出全部评分；生成失败保持generation_failed，不能声称全部样本通过。

## 模式与实现

| 模式 | 当前正式范围 | Codec质量 |
|---|---|---|
| Quick/Quickly | 三集各20–50问题，默认S2T+S2S；20时120请求，C4一轮 | 不执行 |
| Daily | 三集2300问题×2任务=4600请求，C4一轮 | 不执行 |
| Full | 同三集全量，默认4并发档×3轮=55200性能请求；每个数据集/任务优先C4/r1离线评分 | Seed EN1088/ZH2020仅登记deferred，整体incomplete/退出1 |

Seed-TTS-Eval的新chat运行已禁止；隔离Decoder质量流程未开发，可能作为以后独立任务。原C06 Decoder数值对齐仍保留并实际通过，它不是3108条Seed Decoder质量评测。旧四集Quick/原WER、CER及HF诊断均保持历史身份；当前数据、提示词和任务改变，不能与旧Quick计算优化收益。

## 正确性先行

每种named infer必须提供correctness-config，九gate全部通过后才能发正式性能请求。可在服务启动前执行同一run/correctness内的C00–C06；infer只复用源码/配置/资产和产物身份有效的通过gate，随后完成C07/C08。通过证明绑定manifest，score/report重新校验hash及状态。custom仍为显式开发入口。

| Gate | 实际检查数 | 状态 |
|---|---:|---|
| C00 | 1 | pass |
| C01 | 514 | pass |
| C02 | 76 | pass |
| C03 | 3 | pass |
| C04 | 32 | pass |
| C05 | 5 | pass |
| C06 | 4 | pass |
| C07 | 20 | pass |
| C08 | 2 | pass |

C01的514项包含247项MOSS测试，不相加；C02是8组输入/76项Processor断言，C03原3项CUDA边界；C04是8组真实权重双head/全部row logits与raw-top1，C05 Encoder/voice条件，C06保留24/32/511/512 Codes完整PCM；C07为20个HTTP Smoke，C08为2组严格C2。当前profile为单卡固定eager，未在本轮测试双卡。

## 性能

只在成功请求上计算Latency/首事件/RTF分布；QPS分母是完整测量段墙钟，失败没有剔出墙钟。TTFA为首非空PCM事件，可能含静音；TTFT为首非空文本delta。原生TTFC和远端GPU Peak N/A，完整P50/P95/P99保存于原performance.json。服务已处理前置正确性HTTP流量，无额外独立warmup，不能视为冷启动首请求。

| 数据/任务 | 成功/预期 | Length | Latency Mean/P95 s | TTFT Mean s | TTFA Mean s | RTF Mean | QPS |
|---|---:|---:|---|---:|---:|---:|---:|
| llama_questions-s2t | 19/20 | 0 | 1.0190/1.6730 | 0.2728 | N/A | N/A | 3.3148 |
| llama_questions-s2s | 20/20 | 0 | 16.9610/52.0360 | N/A | 9.1531 | 4.5457 | 0.2205 |
| trivia_qa-s2t | 17/20 | 0 | 1.3590/2.3000 | 0.2987 | N/A | N/A | 1.1459 |
| trivia_qa-s2s | 20/20 | 0 | 27.3640/74.9560 | N/A | 9.5050 | 4.5535 | 0.1118 |
| web_questions-s2t | 18/20 | 2 | 7.5030/54.9550 | 0.2040 | N/A | N/A | 0.3065 |
| web_questions-s2s | 20/20 | 1 | 31.7590/97.4010 | N/A | 11.7048 | 4.5594 | 0.1052 |

## 质量与失败

默认spoken-qa-short-en-v1要求简短英文答案，不含参考答案；seed42/temperature0/top_k1/max_tokens512，原输入音频、gold与实际prompt/hash均保存。QA主值为normalized whole-answer EM（仍存accuracy字段），不是论文语义judge；解释性正确回答可能匹配失败。S2S通过Qwen3-ASR-1.7B转录已保存WAV，然后评分EM和UTMOS，不使用Whisper识别模型。

| 数据/任务 | EM（完整20条分母） | EM scored | UTMOS |
|---|---:|---:|---:|
| llama_questions-s2t | 30.0% | 19/20 | N/A |
| llama_questions-s2s | 30.0% | 20/20 | 3.9355 |
| trivia_qa-s2t | 5.0% | 17/20 | N/A |
| trivia_qa-s2s | 5.0% | 20/20 | 3.9519 |
| web_questions-s2t | 5.0% | 18/20 | N/A |
| web_questions-s2s | 0.0% | 20/20 | 4.0103 |

最终ASR 60条全部成功，实际HTTP并发峰值32。没有评分器failed/unavailable/not_attempted/budget_exhausted；generation_failed作为模型/服务输出失败保留。评分前后performance.json hash一致。

空文本案例原始SSE只有finish与[DONE]，无content delta，HTTP200且usage显示生成token；客户端判失败符合协议。不能据此直接认定HF本体或适配哪一方出错，本轮没有扩展到这些样本的HF归因。已有小型正确性套件pass不能替代新QA样本的有效输出检查。

| 数据/任务 | 失败样本 | 错误 |
|---|---|---|
| llama_questions-s2t | llama_questions_188.wav | Empty text output（HTTP 200） |
| trivia_qa-s2t | trivia_qa_832.wav | Empty text output（HTTP 200） |
| trivia_qa-s2t | trivia_qa_842.wav | Empty text output（HTTP 200） |
| trivia_qa-s2t | trivia_qa_486.wav | Empty text output（HTTP 200） |
| web_questions-s2t | web_questions_129.wav | Empty text output（HTTP 200） |
| web_questions-s2t | web_questions_1976.wav | Empty text output（HTTP 200） |

未更换子集、未用重试提高成功率。生成CLI因失败返回1后，外层按设计停止受测服务；随后显式仅继续离线评分，保留其partial/退出1。原始失败和退出码都保留，不把单元实现检查或评分器成功等同于模型整轮验收。后续需用这些固定输入单独定位空文本，再做模型验收。

## 耗时与资源

| 阶段 | 秒 |
|---|---:|
| C00–C06调用（此时HTTP gates未执行，退出1=incomplete） | 190.891 |
| MOSS启动 | 60.178 |
| infer调用（含复核、C07/C08、正式推理） | 736.479 |
| 其中六段正式测量墙钟之和 | 538.862 |
| ASR启动 | 74.917 |
| 离线score调用 | 22.031 |
| 包括切换/等待/收尾的总历时 | 1087.555 |

两作业均终态，包含失败共1088GPU秒（0.302222卡时），峰值1卡；累计10.853333/50，余额39.146667。所有保存的自有进程组active=[]、zombies=[]。40分钟是此部署此子集的实测结果，不承诺首次下载/不同硬件/更大子集也达标。

## 交付与证据

51项客户端本地测试通过，3项可选Torch测试在基础客户端环境跳过；Ruff/格式/diff、wheel/sdist、普通安装后的CLI及离线完整报告重建通过，含6个失败的summary与原文件一致。新增回归验证配对子集、任务选择、三个模式正确性阻断、证明损坏拒绝评分、Codec deferred与实际Full退出1。没有新建模型测试/fixture或改变base。

独立复核全部请求/输出hash、PCM时长、成功分布分位数与吞吐、配对输入、九gate产物hash、逐样本质量聚合和真实ASR并发。结果收据见[JSON](eager-base-qa-quick-20261031.json)。

原始证据在工作区 `.audit/benchmark-e2e-suites-20261031/`：run-01/quick保存正确性、完整输入、事件、文本/音频、性能、质量与报告；score-01保存离线启动/配置/日志；verified-results.json、accounting-final.json及下载/引用清单可重算。Git仅提交代码、文档和轻量收据，不上传权重、音频或大体积Tensor。

Daily/Full全量未运行；Codec质量协议未实现；语义QA judge未实现；当前模型Quick仍含6个空文本失败，另3个length输出计入成功和评分但单列，不能标记为模型分支已验收。
