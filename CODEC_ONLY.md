# 全量 Codec 重建

协议 `moss-codec-only-v2-full` 固定运行 Seed-TTS-Eval EN1088 + ZH2020。每条真实音频经过 Encoder→Decoder，使用对应说话人的 prompt；不加载 AR 模型，不发送 QA 请求。当前使用 HF 原生公开 decode 所支持的离线模式。

## 输入

配置参考 [codec.json](examples/codec.json)。corpus 为 JSONL，每行包含：

```json
{"id":"en-0001","language":"en","text":"Actual target transcript.","target_audio":"en/target.wav","target_sha256":"REPLACE","prompt_audio":"en/prompt.wav","prompt_sha256":"REPLACE"}
```

音频路径相对 corpus 所在目录。配置记录 corpus revision 和整个 JSONL 的 SHA256，同时冻结 Codec 权重、配置和代码资产。启动前验证 3108 条唯一 ID、语言数量及所有 target/prompt hash；不接受子集或缺样本。样本数和 hash 用于检查本地冻结输入，官方版本来源由配置 revision 和准备记录说明，不能仅凭数量宣称复现论文。

`hf-native` 需要原始 Codec 及其 CosyVoice/Matcha 依赖。`sg-native` 需要 `model_root`、完整 `model_sha` 和干净 checkout；直接调用其 `MossSpeechCodecAdapter`。可用 `adapter_options` 传递该实现的构造参数，device 和资产路径由客户端控制。客户端不再注入 BF16 scope、Graph 或分支专属资格检查；行为属于被测实现，配置随运行冻结。

配置中的 python、资产和源码路径相对配置所在目录。`physical_gpu` 显式选择，生成/评分分阶段运行；客户端不选择空闲卡。

## 执行

```bash
moss-speech-bench codec run \
  --config examples/codec.json --output-dir ../benchmark-results/codec-hf
moss-speech-bench codec score \
  --score-config examples/codec-scoring.json --output-dir ../benchmark-results/codec-hf
moss-speech-bench codec report --output-dir ../benchmark-results/codec-hf
```

每个后端或配置使用新目录。worker 保存 Encoder codes、FLOAT WAV、实际样本数和耗时。完整生成要求有效有限 PCM 且每个 code 对应 1920 个采样点。单样本失败保留记录并继续，其余样本也不会代替它补齐分母。Codec 不支持隐式 resume；已有评分不自动覆盖。

## 质量

英文固定 Whisper-large-v3：language=en、FP32、temperature=0、beam_size=5；中文为 Paraformer-zh。WER/CER 按整个语料的编辑错误总数/参考词或字总数计算，另保留逐样本均值。SIM 使用 WavLM，与每条对应 prompt 比较，报告 cosine×100。缺失或失败的生成和评分保留全量分母，不当作零错误。

评分配置见 [codec-scoring.json](examples/codec-scoring.json)，评分前再次校验冻结输出及 target/prompt 音频。输出身份和 hash 保留在同一运行目录。

## 比较

比较配置只包含运行名称和路径，路径相对该配置所在目录：

```json
{"runs":{"HF":"../benchmark-results/codec-hf","SG":"../benchmark-results/codec-sg"}}
```

```bash
moss-speech-bench codec report \
  --config comparison.json --output-dir ../benchmark-results/codec-comparison
```

要求相同 corpus、资产、seed、解码模式和评分器契约。报告各次实际身份、绝对分数与分母；Encoder codes 是否一致作为诊断记录，不作为内部精度准入证书。离线重建质量不能代表流式性能或端到端回答质量。
