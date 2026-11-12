# MOSS-Speech benchmark client

只维护两类评测：三个 OpenAudioBench 数据集的端到端 QA，以及全量 Seed-TTS-Eval 的 Codec 重建。E2E 连接已启动的 HTTP 服务；Codec 在显式指定的模型环境和 GPU 上调用组件。客户端不部署生成服务、不调度 GPU，不注入模型优化。

## 安装

```bash
pip install -e '.[dev]'
# 离线评分环境按需要安装
pip install -e '.[quality,utmos,sim,zh]'
moss-speech-bench --help
```

运行依赖仅有 aiohttp、numpy；模型和评分器依赖留在各自环境中。所有配置中的文件路径均相对于该配置文件所在目录，Codec corpus 中的音频路径相对于 corpus 所在目录。

## E2E

支持 LLaMA Questions 300、TriviaQA 1000、WebQuestions 1000。使用原始 OpenAudioBench CSV：`audio_filename`、`question`/`Questions`、`Answer`/`answers`/`answer_normalized_aliases`。文本题面和参考答案随输入冻结；语音输入使用该题对应的录音。

从 [e2e.json](examples/e2e.json) 配置本地 CSV、音频目录、固定数据 revision 和服务能力。配置可以只包含所需数据集。数据集的完整 CSV 必须分别有 300/1000/1000 条唯一 ID；数量校验不等于官方版本校验。

```bash
moss-speech-bench e2e run \
  --config examples/e2e.json --model moss-speech \
  --base-url http://127.0.0.1:8000 \
  --datasets llama_questions web_questions --tasks t2s s2s \
  --samples llama_questions=30 web_questions=50 --subset-seed 42 \
  --concurrency 16 --warmup 16 --repeats 3 --max-tokens 512 \
  --deployment examples/deployment.example.json \
  --output-dir ../benchmark-results/example
```

| 参数 | 规则与默认值 |
|---|---|
| `--datasets` | 一个或多个已配置的数据集；省略则运行配置中的全部数据集 |
| `--tasks` | `t2t t2s s2t s2s`，可分别或组合选择；省略则运行 `service.tasks` 声明的模态 |
| `--samples DATASET=COUNT ...` | 每个数据集独立设置 1..全集数量；未指定数量的数据集跑全量 |
| `--subset-seed` | 固定选样 seed，默认 42，与生成 seed 分开 |
| `--selection` | 复用另一运行的 `selection.json`；与数量和选样 seed 参数互斥 |
| `--concurrency` | 默认 4；可传 `1 4 8 16` 依次扫描，不代表模型内部实际 batch |
| `--warmup` | 每个数据集×模态×并发×轮次的预热数量；默认等于该组并发，0 关闭 |
| `--repeats` | 默认 3，允许任意正整数 |
| `--max-tokens` | 默认 512；配置文件及 CLI 均可覆盖，CLI 优先 |
| `--seed` | 生成 seed，默认 42；temperature 默认 0 |
| `--temperature/top-p/top-k/min-p/repetition-penalty` | 生成参数，CLI 覆盖配置；完整写入契约 |
| `--no-stream` | 关闭流式；服务未声明支持流式时必须使用 |
| `--timeout` | 每个已发出请求的超时，默认 300 秒 |
| `--time-budget` | 可选的本次推理总预算，默认不限；耗尽保留部分记录 |
| `--api-key-env` | 服务凭证的环境变量名；不保存变量值 |
| `--dry-run` | 验证数据并显示正式、预热、质量评分计划；不发请求、不创建结果目录 |
| `--resume` | 相同参数恢复：跳过已完成组，未完成组另建 attempt 后整组重跑 |

Chat 使用标准系统提示词：文本输出为 `You are a helpful assistant. Respond with text outputs.`，语音输出为 `You are a helpful assistant. Respond with spoken outputs.`。不添加复述或简短回答指令。`service.api=speech` 仅支持非流式 T2S，参考音色和服务内部系统提示词由配置声明并冻结。

预热循环使用所选样本的前 N 条，不足时循环复用；每条有独立 ID。预热失败阻止该轮正式请求。预热输出单独存储，不进入性能和质量分母；它可能预填服务缓存，同进程的多轮测试不能称为独立冷启动。

### 固定选样

沿用 `sha256-id-v1`：按 `SHA256([dataset_id, seed, sample_id])` 排序取前 N 条。相同数据和 seed 下，20 条是 30 条的前缀，30 条是 50 条的前缀；CSV 行顺序不影响排名。四种模态和所有轮次复用同一批 ID。

每次生成 `selection.json`，冻结算法版本、顺序、revision、完整 CSV hash、总体 ID hash、选中题面/答案和对应音频 hash。语音输入要求所选录音齐全；仅文本输入可在没有录音时运行，清单将缺失音频 hash 记为 null。已有录音的 hash 同样冻结；后续增加或更换音频需建立新清单。跨 HF/SG 或分支比较优先复用清单：

```bash
moss-speech-bench e2e run \
  --config examples/e2e.json --model moss-speech --no-stream \
  --selection ../benchmark-results/example/selection.json \
  --tasks t2t --concurrency 4 --warmup 0 --repeats 1 \
  --output-dir ../benchmark-results/reuse
```

复用时任何数据 revision、源文件或选中音频变化都会报错，不自动重新抽样。清单不绑定后端地址、模型名或输出模态。

### 离线质量与报告

推理只生成一次。释放生成服务的资源后，在评分环境运行：

```bash
moss-speech-bench e2e score \
  --output-dir ../benchmark-results/example --score-config examples/scoring.json
moss-speech-bench e2e report --output-dir ../benchmark-results/example
moss-speech-bench e2e report \
  --output-dir ../benchmark-results/candidate --baseline ../benchmark-results/example
```

质量覆盖每个数据集×模态×并发的首轮全部选中问题：文本直接判答案，语音先 ASR 再判答案，语音另评 UTMOS。Acc 使用固定模板的语义 Judge，不回退到 EM。失败、未执行和评分器缺失保留计划分母，同时报告有效评分准确率。成功评分可缓存复用，失败评分重试时保留历史尝试。详见 [Judge 协议](JUDGE_PROTOCOL.md)。

性能报告包含 TTFT P95、TTFA P95、RTF P95、audio-s/s、TPS、QPS、最大播放欠载 Max/P95、C50/C100。RTF 先逐请求计算完整响应耗时/实际 PCM 时长，再求 P95。TTFT/TTFA 分别从请求发出计时至首个非空文本 delta/PCM SSE 事件；非流式没有这些首事件指标，记 N/A。

收益只使用 TTFA P95 和 RTF P95。跨轮先逐轮计算再等权平均，标明“P95 轮均”，不能称为合并请求 P95。TPS 来自服务返回的 completion_tokens；任一成功请求缺少 usage 时为 N/A。连续性仅统计成功且有多块音频观测的请求；C50/C100 为最大欠载≤50/100ms 的百分比。最大欠载 Max 轮均不是全程最坏值。Mean 留作诊断。

比较要求选样、生成参数、系统提示词、参考音色、并发/预热/轮数、客户端指标实现一致，且双方提供被测服务代码身份；不同 HF/SG 实现身份单独展示。契约不同只列出差异，不计算收益，没有任意忽略字段的开关。客户端不证明 C16 等于服务内部 B16。

## Codec Only

每次固定运行 Seed-TTS-Eval EN1088 + ZH2020，共 3108 对 target/prompt。没有抽样、轮数或套餐参数。输入完整清单、revision、hash 和对应音频必须齐全；失败仍留在全量分母。支持 HF 原生 Codec 和所选 SG adapter 的离线 Encoder→Decoder 重建，评估 EN WER、ZH CER、SIM。详见 [Codec 协议](CODEC_ONLY.md)。

```bash
moss-speech-bench codec run \
  --config examples/codec.json --output-dir ../benchmark-results/codec-hf
moss-speech-bench codec score \
  --score-config examples/codec-scoring.json --output-dir ../benchmark-results/codec-hf
moss-speech-bench codec report --output-dir ../benchmark-results/codec-hf
```

## 结果与维护

一次运行一个 `benchmark-results/<run_id>/`，以 `summary.md` 为入口，见 [摘要模板](SUMMARY_TEMPLATE.md)。E2E 保存身份与契约、选样、冻结请求、各组原始输出和事件、计时及 hash、性能和质量记录；预热单独保存。`execution/` 存执行材料，`analysis/` 存详细分析。失败、超时、截断、未完成轮次均明确保留。

0.7 是有意的不兼容精简：删除 `--phase`、`--mode`、Quick/Daily/Full/custom、C00–C08、HF 数值精度回归及分支专属检查。旧配置不自动迁移；旧结果用其记录的原始客户端版本读取，不改写历史证据。保留的开发测试仅验证这两类评测的软件行为：

```bash
python -m pytest -q
python -m ruff check src tests
python -m build
```

复用代码和许可见 [来源说明](ORIGIN.md)。
