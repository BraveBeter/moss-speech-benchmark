# 官方提示词与Judge评分

MOSS-Speech论文§4.3/Table6使用LLaMA Questions、TriviaQA、WebQuestions并报告UTMOS，表注说明参照Kimi进行答案归一化后判分。论文未在该段固定Judge服务的模型快照、API采样参数或转录器版本，不能承诺本客户端分数逐项复现论文。

- 论文：https://arxiv.org/html/2510.00499v1#S4.SS3
- 官方模型及System：https://github.com/OpenMOSS/MOSS-Speech
- 评测实现：https://github.com/MoonshotAI/Kimi-Audio-Evalkit/blob/a65bd4243dd280daa685e35a5cfd69c371315c06/almeval/datasets/ds_refqa.py

Kimi Evalkit固定提交a65bd4243dd280daa685e35a5cfd69c371315c06对LLaMA使用Correct/Incorrect模板，对TriviaQA/WebQuestions使用JSON judgment模板；以含义等价或包含正确答案判定，不要求逐字匹配。其README默认gpt-4o-mini，支持gpt-4o；这不证明MOSS论文当时使用同一模型快照。模板按MIT许可复用，许可见JUDGE_UPSTREAM_LICENSE.txt。

v5生成只使用官方System：S2T为`You are a helpful assistant. Respond with text outputs.`；S2S为`You are a helpful assistant. Respond with spoken outputs.`。删除默认简短答案及朗读指令，命名套件拒绝附加prompt覆盖；不把参考答案或Judge指令传给被测模型。CSV原始题面只作为冻结评分元数据保存。

score默认使用第三方Judge；S2T判已有文本，S2S判已有WAV的ASR转录，UTMOS独立保留。配置examples/scoring.json的judge endpoint/model/revision，凭证仅从api_key_env读取，不写入缓存。Judge完整请求、原始答复、实际返回模型、固定模板提交及hash随质量契约记录。问题、答案、候选回复、模型或版本变化会改变缓存键；返回格式无效、HTTP错误、超时和未配置均不会回退到EM。失败保留计划分母，evaluated_accuracy同时报告有效评分分母。

Judge模板属于离线裁判输入，与生成System严格分开。未实现对题面/候选中的提示注入的额外裁判防护；不额外修改上游模板来宣称原论文复现。裁判错误应通过人工抽检和对照集检查。

质量协议升级为moss-quality-v2-llm-judge；v5提示词变化使新旧性能/质量不可直接比较。旧报告不重写，legacy_em仅为显式历史分析选项，不是默认QA指标；旧记录缺少题面时不猜题、不以参考答案代替题面。真实Judge验收需要用户配置可访问的第三方服务；单元测试使用本地模拟响应，不代表已调用真实Judge或重跑模型。

The scoring example uses the DeepSeek OpenAI-compatible endpoint with `model=deepseek-flash` and explicit `thinking.type=disabled`. Endpoint, model/revision and thinking policy are frozen in the score binding and cache key; the API key is read from the named environment variable and never archived.
