# QA 离线质量

LLaMA Questions、TriviaQA、WebQuestions 使用固定的 Kimi-Audio-Evalkit 语义裁判模板，来源提交为 `a65bd4243dd280daa685e35a5cfd69c371315c06`。模板以 MIT 许可复用，见 [许可](JUDGE_UPSTREAM_LICENSE.txt) 和 [上游实现](https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/a65bd4243dd280daa685e35a5cfd69c371315c06/almeval/datasets/ds_refqa.py)。

LLaMA 返回唯一 Correct/Incorrect；TriviaQA/WebQuestions 返回 JSON judgment。按含义等价或包含正确答案判分，不要求逐字匹配。默认配置见 [scoring.json](examples/scoring.json)，使用 DeepSeek 的 OpenAI 兼容端点，显式记录 model、revision 和 thinking 策略；凭证只从 `api_key_env` 读取，不写入缓存。

评分输入是已冻结的问题、参考答案和模型输出。T2T/S2T 直接使用输出文本；T2S/S2S 使用输出 WAV 的 ASR 转录并独立评 UTMOS。评分覆盖每个数据集×模态×并发的首轮全部计划样本。质量评分不重发生成请求。

Acc = 判为正确的数量/计划样本数，同时提供正确数量/有效评分数量。生成失败、评分失败、未执行和评分器缺失单独计数。HTTP 错误、超时、裁判格式无效均不回退到字符串匹配。成功结果缓存绑定问题、答案、候选回复、裁判模型/版本、模板 hash 和配置；失败重试保留历史记录。

系统提示词只用于生成；裁判模板只用于离线评分。客户端不把参考答案或裁判指令交给被测模型。本协议不声称复现论文中未固定的评分器快照；模板也不包含额外的提示注入防护。真实服务评测和客户端的本地模拟测试应分别记录。
