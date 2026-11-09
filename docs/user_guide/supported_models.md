# 支持模型



本文汇总 vLLM v0.27.1 已说明支持或已提供量化配置的模型、精度形态和使用注意事项。


## vLLM v0.27.1 支持模型



|厂商| 模型 | 运行态量化 |
| ---| --- | --- |
|DeepSeek| DeepSeek-R1 | W4A8 |
|DeepSeek| DeepSeek-V3.1 | W4A8 |
|DeepSeek| DeepSeek-V3.2 | W4A8 |
|DeepSeek| DeepSeek-V4 | W4A8 |
|智谱 GLM| GLM-5.2 | W4A8 |
|腾讯 Hunyuan| Hy3 | W4A8 |
|MiniMax| MiniMax-M2.5 | W4A8 |
|MiniMax| MiniMax-M2.7 | W4A8 |
|MoonShot| Kimi-K2.5 | W4A8 |
|阿里 Qwen| Qwen2 | W8A8 |
|阿里 Qwen| Qwen3 | W8A8 |
|阿里 Qwen| Qwen3.5 | W4A8 |
|阿里 Qwen| Qwen3.6-35B-A3B | W4A8 |
|阿里 Qwen| Qwen3.8 | W8A8 |
|阶跃 StepFun| Step-3.5 | W4A8 |
|阶跃 StepFun| Step-3.7-Flash | W4A8 |




## 通用使用说明

### 使用前须知



- **安装对版本**：本版是 `vllm-iluvatar` 插件（在 vLLM 之外单独发布），与 vLLM 版本一一对应，必须安装 `vllm>=0.27.1,<0.28.0`，否则插件无法加载。
- **旧权重要重新量化**：量化格式已升级为 `iluvatar-mixed`。旧版的 W4A8 / W8A8 权重不能直接沿用，需要用新版 `examples/iluvatar-compressed-tensors/` 下的 `quant.py`、`configs`、`rules` 重新做一次离线量化。
- **量化参数无需手动传**：重新量化后，量化产物会写进 checkpoint 的 `config.json`，启动时由插件自动读取，不再需要传旧的量化环境变量。
- 量化方法与布局的详细字段说明见[量化](config_features/quantization.md)页，使用上表中的模型一般无需关心。

### 运行默认行为

以下为插件的默认行为，通常无需手动指定，仅在对应场景下需要留意：

- **推理执行后端（Model Runner）**：v0.27.1 起使用 Model Runner V2。
- **投机解码**：启用 DSpark 投机解码时会强制切到 Model Runner V2。
- **量化 KV cache**：使用量化 KV cache 时，须显式设置 `VLLM_USE_V2_MODEL_RUNNER=0`。
- **CUDA Graph**：默认开启，但当前仍需显式指定 `mode=0` 和 `FULL_DECODE_ONLY`。

