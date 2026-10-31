# eager base 质量异常诊断（2026-10-31）

本报告使用五个冻结的异常/控制样本，定位旧 Quick 评测的协议问题。不是 Daily/Full 重跑，也不将挑选样本的结果外推为完整数据集质量。

## 原指标复核

原 Quick 英文 WER = 118/346 = 34.104046%，中文 CER = 541/648 = 83.487654%。独立使用 jiwer 对全部 30 EN 和 30 ZH 的已归一化 ref/hyp 重算替换、删除、插入、命中，逐项与原记录一致。原报告与评分文件未覆盖。

错误集中在生成内容偏离目标：英文最差样本把 Tim Tebow 的文字谜题当作问题回答，34 次编辑/18 词；中文“单孔类”样本被识别为英文计数，121 次编辑/16 字。WER/CER 包含插入，单条可超过 100%。这说明高分不是百分数或均值换算错误；是否存在 ASR 误识别，需要下面的固定/自动语言对照。

S2S 旧值为整句归一化精确匹配（strict EM）0/30。独立答案提及检查为 20/30，仅作定位线索，不是语义准确率；错误回答、否定句也可能提及正确实体。

## 新执行的正确性检查

客户端 `--phase correctness` 全部 C00–C08 pass，绑定模型 eager base `084f62ef13c65fbf33f0c9a3f2cce5e87e0c01d6`。本次单卡，不冒称双卡重验。

| 检查 | 本次实际覆盖 |
|---|---|
| C00 | 资产预检，50 个固定资产重新计算 hash |
| C01 | 514 CPU 单元/边界测试 |
| C02 | 8 组 Processor 输入，76 项比较 |
| C03 | 3 项真实 CUDA 回归 |
| C04 | 8 profiles、32 项 AR 检查，含完整末位 logits 与 top-1 |
| C05 | 5 项原版 Encoder 特征/码比较 |
| C06 | 24/32/511/512 Codes，4 项完整 PCM 对照 |
| C07 | 20 项 HTTP/SSE 四模态、中英、stream/non-stream及异常恢复 |
| C08 | 2 项严格串行/并发 C2 对照 |

这缩小了适配错误的嫌疑范围，不等价于所有输入、完整自回归轨迹、任意并发均已证明无误。独立质量相近也不能代替数值对齐。

## 配对实验配置

固定两条英文（一异常一正常）、两条中文（计数、翻译）与一条 Moscow 语音问答。五条旧提示＋五条显式 system＋两条中文 raw，共每端 12 请求。均用普通 benchmark-client 推理入口、C1、seed42、temperature0、max_tokens512，固定同一参考音色，非流式完整解码；不是性能比较。旧 Quick 为 C4/流式，和本次波形不要求相等。

- HF 模型 revision：`cff025bb41d8459d59abac0b5e44aba7f659ec9e`。
- Codec revision：`eeec733e4e1dea7da444d332d8e1621ef257414c`。
- voice SHA256：`cde3e98fdf0c90d4a578aa5858e1d4f36d99d3dd96d7ebb302b8bb976c3d12da`。
- HF 使用原 checkpoint 的 Processor、generate、decode，独立环境，不导入 SGLang。BF16 AR/SDPA、FP32 Codec；Transformers4.56.2 保留原生成接口。为缺 TorchCodec 的环境显式使用 PCM16 文件读取适配（int16/32768），不修改模型源码或计算。
- 参考端点显式设 min_new_tokens0、音频模态 pad151667、结束 token151643/151645。它们与 tokenizer padding 的用途不同。
- Qwen3-ASR1.7B 与旧 Quick 相同权重/流程；主评分使用样本语言 en/zh，另存自动语言识别的诊断转录，不替换原分数。

## 配对结果与结论

**主要原因是聊天式朗读的任务遵循失败和质量协议错配。HF 也会答题、翻译或退化；同时发现 ASR 固定语言约束会改变转录的评分风险。当前证据不支持把高 WER/CER 全部归因于适配，也不能证明两端完整自回归计算处处相同。**

24/24 推理、24/24 正式质量记录完整；两端各 12 个请求的 payload hash 逐项相同。旧提示的五条 eager C1 流式复跑与原 Quick C4 WAV **5/5 字节相同**，说明这五条旧异常不是只在 C4 出现的状态污染。配对主实验是非流式，不把这两种解码的波形混作精确对齐证据。

| 样本/现象 | eager 旧提示 | HF 旧提示 | system 提示后的观察 |
|---|---|---|---|
| EN 文字谜题 `common_voice_en_155313-common_voice_en_155315` | 回答如何替换字母，而非朗读 | 朗读零散字母并重复 S，触及512上限 | 两端仍回答谜题，没有恢复逐字朗读 |
| EN 正常句 `common_voice_en_17784-common_voice_en_17786` | The virus spread faster than wildfire. | 同句 | 两端仍准确朗读 |
| ZH 单孔类 `00004713-00000110` | 英文计数 one…twenty-one | 固定中文ASR只转录“嗯。嗯。”，自动语言为空；40.88秒输出截断 | eager 改成“这个问题在问：单孔类属于哪一类生物？”，仍未逐字读；HF仍退化/截断 |
| ZH 可持续发展 `10003331-00000019` | 自动语言ASR为英文译文；固定中文ASR却为中文转述 | 两种ASR均为英文译文 | 两端均转录为原中文，CER0 |
| S2S `llama_questions_109.wav`，参考Moscow | Moscow is the capital of Russia，并继续解释 | Moscow is the capital of Russia. | 两端都缩为Moscow.，strict EM从0变1 |

以上转录完整内容、音频路径、请求hash、结束原因、逐样本指标均保存在工作区 `benchmark-results/quality-diagnosis-20261031/paired-analysis.json`。表中对原文含义的判断只用于这些诊断样本，不是整集语义评分。

### 数值对照

下表均是同一固定评分配置下的**单样本**编辑错误率；中文指标为CER。S2S是单样本strict EM（0/1）。这些是有意挑选的异常/控制样本，不汇总为“总体改善比例”。

| 样本 | eager旧提示 | HF旧提示 | eager system | HF system |
|---|---:|---:|---:|---:|
| EN文字谜题 WER | 188.89% | 2277.78% | 200.00% | 233.33% |
| EN正常句 WER | 0.00% | 0.00% | 0.00% | 0.00% |
| ZH单孔类 CER | 756.25% | 100.00% | 68.75% | 100.00% |
| ZH可持续发展 CER | 38.89% | 316.67% | 0.00% | 0.00% |
| S2S Moscow strict EM | 0 | 0 | 1 | 1 |

HF共有4条length输出（旧EN谜题、旧ZH单孔类、system ZH单孔类、raw ZH可持续发展），eager本轮12条均stop。所有截断仍参与评分；不将其剔除或当成HF/SG性能收益。中文raw直传没有解决问题：单孔类两端转为解释，另一条也出现解释或退化。

### ASR评分歧义

自动语言检查覆盖34个音频路径，按音频hash/语言/评分配置去重为29条，29/29返回；重复的5条恰是旧Quick与新C1流式完全相同的WAV。它是另存的诊断评分，没有改写原质量数据。

- 原Quick的中文翻译、英文计数，在自动语言下仍是英文；英文谜题仍是回答。高WER/CER的核心异常没有因移除语言hint而消失。
- 本次 eager 非流式“可持续发展”的**同一WAV**：指定zh转录为“我们将贡献于全球城市的可持续发展。”（CER38.89%），auto转录为“We will contribute to the sustainable development of global cities.”（CER316.67%）。因此语言hint会影响转录内容，可能掩盖语言违约。
- HF退化音频在固定zh下得到“嗯。嗯。”、auto为空，进一步说明不能只用识别文本代表真实发音。

自动与指定语言仍是同一个Qwen识别器，不能代替独立ASR或听辨真值。此处确定的是**转录/评分对语言约束敏感**；有分歧的音频应标为待听辨/独立识别复核，不能挑较低CER作为结果。原Quick 34.10%/83.49%不因本次分析被替换。

### 适配正确性的边界

C00–C08全部通过，加上HF原版也有相同类别的任务失败，支持优先修正评测协议。两端的长自回归内容、长度并不完全一致；HF用SDPA，eager服务实际用FlashInfer，且没有对这12组完整解码逐步导出并定位首个分歧。不能把这些差异直接定性为适配bug，也不能用小样本行为相似宣布适配已全域无误。若后续需要解释长轨迹差异，应固定token前缀、逐步teacher forcing检查对应logits，而不是放宽现有门槛。

## 完善后的评测协议

1. **TTS分开命名。** 当前是`chat-read`工程回归，评估聊天模型能否按要求朗读；论文Table4是独立speech decoder的SeedTTS评测。后者采用Whisper/Paraformer、参考音色SIM及DNSMOS，与当前Qwen ASR、固定服务音色和完整AR路径都不同，不能直接比较绝对分数。依据：[MOSS-Speech报告](https://arxiv.org/html/2510.00499)。
2. **显式system候选。** 新增 `examples/daily-protocol-candidate.json`：中英朗读规则放第一条system，user只放原文；S2S使用不泄漏答案的短答system。原`daily.json`不变。翻译单例已恢复，但谜题/问句仍失败，故该候选**尚未通过严格TTS协议验收**，不能宣传为彻底修复。
3. **ASR协议单独版本化。** 新增 `examples/scoring-protocol-candidate.json`，明确`language_hint:auto`，保留strict EM并可附加答案提及诊断。原官方同源固定语言配置保留；更换提示、ASR语言策略都必须生成新baseline，不允许混用两种分数。发现分歧时保存两份转录并标注歧义，禁止自动挑低分。
4. **TTS验收先用独立挑战集。** 冻结包含普通句、问句、祈使句、数字及中英文的样本后，再检查逐字朗读、语言违约、空输出/截断及WER/CER。当前system针对已观察症状设计，诊断集不是独立保留测试集；正式采用前需冻结新集合并复验。任务/语言违约在本报告按转录逐例分析，客户端尚未实现可靠的自动语义判定。
5. **S2S按任务分别评分。** 自然对话QA保留完整答案，strict EM清楚标注为格式严格的工程指标；短答QA用独立system和baseline。`answer_mention`只是实体提及诊断，不能替代语义QA Accuracy。正式语义QA仍需版本固定的官方评分器或经标注集验证的答案抽取/judge，UTMOS独立报告声学质量。本次Moscow恢复EM只证明单例格式可控。
6. **全量回归仍待执行。** 本轮只诊断5个样本、12请求/端，没有运行Daily5279条或Full；不能给出新协议的全量质量结论或保证模型适合严格TTS。

详细接口和约束见 [PROTOCOLS.md](../PROTOCOLS.md)。

## 复现与证据

证据根：工作区 `benchmark-results/quality-diagnosis-20261031/`。

| 资料 | 路径 |
|---|---|
| 正确性配置、命令、报告 | `correctness-config.json`、`run_correctness.<hash>.py`、`correctness/correctness.json` |
| 选样及原指标复核 | `selected-original.json`、`original-score-audit.json` |
| 12请求定义及数据 | `paired-workload.json`、`diag_0.json`至`diag_4.json`、`diag_4.wav` |
| 普通客户端推理命令/退出/清理 | `pairs-control.json`、`hf-pairs/`、`eager-pairs/` |
| HF原始输入/grid、生成配置、源码hash | `hf-native/`、`hf-native/lineage.json` |
| ASR/质量配置、命令、退出/清理 | `scoring.json`、`score-control.json`、两端`quality.json`及逐样本记录 |
| 自动语言诊断 | `asr-auto-configs.json`、`asr-auto-results.json`、`asr-auto/` |
| 配对汇总/请求hash、流式复跑hash | `paired-analysis.json`、`stream-repeat-audit.json` |
| 资源账本 | `accounting-final.json`、`sacct-final.txt` |

正式配对推理使用冻结的客户端`7980e3e`源码快照`client-source-v5/`，评分使用`43ac36f`快照`score-source/`；正确性使用已安装`27a63a3`版本。每份快照身份均有完整SHA及文件hash，不把之后的文档提交冒充推理时客户端。eager base保持084f62ef且未修改。

参考环境/端点排错记录全部保留：77814缺streamer兼容/TorchCodec，77815未设整数min_new_tokens，77816错误使用tokenizer pad导致空Codes，77817在错误pad配置完成前主动取消。77818才是正式24条成功配对；77819完成评分。上述失败属于参考环境/诊断工具，不计作模型质量样本。PCM I/O适配、Transformers版本和显式生成参数均已披露，不声称原依赖环境完全未改动。

七个GPU作业均终态；自有正常退出服务组检查均active=[]，取消的77817在节点查询已无作业进程。包含所有失败/取消，本轮1320GPU秒（0.366667卡时），累计10.551111/50卡时，余39.448889。没有恢复Daily/Full，没有修改或推送模型分支。
