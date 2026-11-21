# MOSS-Speech benchmark client

客户端支持 OpenAudioBench 端到端 QA 和全量 Seed-TTS-Eval Codec 重建。端到端评测连接已启动的 HTTP 服务。Codec 重建在配置指定的模型环境和 GPU 上调用组件。

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

从 [e2e.json](examples/e2e.json) 配置本地 CSV、音频目录、固定数据 revision 和服务能力。配置可以只包含所需数据集。客户端校验 CSV 行数和 ID 唯一性，数据版本由配置声明。运行对照时，按[部署模板](examples/deployment.example.json)填写 `baseline-deployment.json` 和 `candidate-deployment.json`。下列示例在同一 shell 中使用临时输出路径。

```bash
run_root="$(mktemp -d)"
moss-speech-bench e2e run \
  --config examples/e2e.json --model moss-speech \
  --base-url http://127.0.0.1:8000 \
  --datasets llama_questions web_questions --tasks t2s s2s \
  --samples llama_questions=30 web_questions=50 --subset-seed 42 \
  --concurrency 16 --warmup 16 --repeats 3 --max-tokens 512 \
  --deployment baseline-deployment.json \
  --output-dir "$run_root/baseline"
```

| 参数 | 规则与默认值 |
|---|---|
| `--datasets` | 一个或多个已配置的数据集。省略时使用 `--selection` 中的数据集，未指定清单时使用配置中的全部数据集 |
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
| `--stream` / `--no-stream` | 默认启用流式。服务只支持非流式时使用 `--no-stream` |
| `--deployment` | 提供服务代码身份和部署设置。使用 `report --baseline` 比较时，双方均需提供服务身份 |
| `--timeout` | 每个已发出请求的超时，默认 300 秒 |
| `--time-budget` | 可选的本次推理总预算，默认不限；耗尽保留部分记录 |
| `--api-key-env` | 服务凭证的环境变量名；不保存变量值 |
| `--dry-run` | 验证数据并显示正式、预热、质量评分计划；不发请求、不创建结果目录 |
| `--resume` | 相同参数恢复：跳过已完成组，未完成组另建 attempt 后整组重跑 |

Chat 语音输出可设置 `service.reference_audio`，客户端将其 data URI 随 `audio.ref_audio` 传递，并冻结音色 hash；非流式使用 WAV，流式使用 PCM。Chat 使用标准系统提示词：文本输出为 `You are a helpful assistant. Respond with text outputs.`，语音输出为 `You are a helpful assistant. Respond with spoken outputs.`。不添加复述或简短回答指令。`service.api=speech` 仅支持非流式 T2S，参考音色和服务内部系统提示词由配置声明并冻结。

预热循环使用所选样本的前 N 条，不足时循环复用；每条有独立 ID。预热失败阻止该轮正式请求。预热输出单独存储，不进入性能和质量分母；它可能预填服务缓存，同进程的多轮测试不能称为独立冷启动。

### 固定选样

沿用 `sha256-id-v1`：按 `SHA256([dataset_id, seed, sample_id])` 排序取前 N 条。相同数据和 seed 下，20 条是 30 条的前缀，30 条是 50 条的前缀；CSV 行顺序不影响排名。四种模态和所有轮次复用同一批 ID。

每次生成 `selection.json`，冻结算法版本、顺序、revision、完整 CSV 和总体 ID 的 hash，以及选中题面、答案和音频的 hash。语音输入要求所选录音齐全。仅文本输入时，清单将缺失的音频 hash 记为 null。以下示例针对另一部署复用基线清单及评测参数。

```bash
moss-speech-bench e2e run \
  --config examples/e2e.json --model moss-speech \
  --base-url http://127.0.0.1:8001 \
  --selection "$run_root/baseline/selection.json" --tasks t2s s2s \
  --concurrency 16 --warmup 16 --repeats 3 --max-tokens 512 \
  --deployment candidate-deployment.json \
  --output-dir "$run_root/candidate"
```

复用时任何数据 revision、源文件或选中音频变化都会报错，不自动重新抽样。清单不绑定后端地址、模型名或输出模态。

### 离线质量与报告

在评分环境中对冻结输出评分。`score` 和 `report` 使用对应运行的 `--output-dir`。

```bash
moss-speech-bench e2e score \
  --output-dir "$run_root/baseline" --score-config examples/scoring.json
moss-speech-bench e2e report --output-dir "$run_root/baseline"
moss-speech-bench e2e report \
  --output-dir "$run_root/candidate" --baseline "$run_root/baseline"
```

质量评分覆盖每个数据集、模态和并发组合的首轮选中问题。文本输出直接判分，语音输出先经 ASR 转写再判分，并另评 UTMOS。Acc 使用固定模板的语义 Judge，以全部计划样本为分母，另报有效评分准确率。成功评分可复用缓存，失败评分重试时保留历史尝试。详见 [Judge 协议](JUDGE_PROTOCOL.md)。

`report` 从冻结运行重建 `summary.md`，见[摘要模板](SUMMARY_TEMPLATE.md)。TTFT/TTFA 从请求发出分别计时至首个非空文本 delta 和首个非空 PCM SSE 事件，非流式记为 N/A。RTF 按每条请求的完整响应耗时除以实际 PCM 时长计算。跨轮汇总对各轮指标等权平均。TPS 依赖成功请求的 `completion_tokens`，播放连续性指标使用成功且观测到多块音频的请求。

`report --baseline` 校验选样、生成参数、提示词、参考音色、并发、预热、轮数及客户端指标实现，并分别展示两端服务身份。双方提供服务身份且比较契约一致时，客户端的收益表计算 TTFA P95 和 RTF P95 的变化。契约不同会列出差异。

## Codec Only

每次固定运行 Seed-TTS-Eval EN1088 + ZH2020，共 3108 对 target/prompt。没有抽样、轮数或套餐参数。输入完整清单、revision、hash 和对应音频必须齐全；失败仍留在全量分母。支持 HF 原生 Codec 和所选 SG adapter 的离线 Encoder→Decoder 重建，评估 EN WER、ZH CER、SIM。详见 [Codec 协议](CODEC_ONLY.md)。

```bash
codec_dir="$(mktemp -d)/codec"
moss-speech-bench codec run \
  --config examples/codec.json --output-dir "$codec_dir"
moss-speech-bench codec score \
  --score-config examples/codec-scoring.json --output-dir "$codec_dir"
moss-speech-bench codec report --output-dir "$codec_dir"
```

## 开发检查

```bash
python -m pytest -q
python -m ruff check src tests
python -m build
```

复用代码和许可见 [来源说明](ORIGIN.md)。
