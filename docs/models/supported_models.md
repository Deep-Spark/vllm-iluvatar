# 支持模型

本文汇总已说明支持或已提供量化配置的模型、精度形态和使用注意事项。

| 厂商 | 模型 | 运行态量化 |
| --- | --- | --- |
| DeepSeek | DeepSeek-R1 | W4A8 |
| DeepSeek | DeepSeek-V3.1 | W4A8 |
| DeepSeek | DeepSeek-V3.2 | W4A8 |
| DeepSeek | DeepSeek-V4 | W4A8 |
| 智谱 GLM | GLM-5.2 | W4A8 |
| 腾讯 Hunyuan | Hy3 | W4A8 |
| 腾讯 Hunyuan | HY4 | FP4（见下方预填充验证记录） |
| MiniMax | MiniMax-M2.5 | W4A8 |
| MiniMax | MiniMax-M2.7 | W4A8 |
| MoonShot | Kimi-K2.5 | W4A8 |
| 阿里 Qwen | Qwen2 | W8A8 |
| 阿里 Qwen | Qwen3 | W8A8 |
| 阿里 Qwen | Qwen3.5 | W4A8 |
| 阿里 Qwen | Qwen3.6-35B-A3B | W4A8 |
| 阿里 Qwen | Qwen3.8 | W8A8 |
| 阶跃 StepFun | Step-3.5 | W4A8 |
| 阶跃 StepFun | Step-3.7-Flash | W4A8 |

## HY4 FP4 预填充验证记录

2026-11-27 使用 vllm-iluvatar `0490b046`、IxFormer `27c5be0eb` 和
llm-benchmark `be12a2cf` 复测了 c44 用例。
16 张 TG-V300 设备运行在 1500 MHz，使用 TP1、PP16。每个请求输入 85,909 个 token，
共享前缀 79,040 个 token，输出 1 个 token。每轮并发 44 个请求，门限为每轮首 token 延迟（TTFT）P50 ≤ 10 秒。

| 轮次 | 成功请求 | 整机 QPM | TTFT P50（ms） | TTFT P90（ms） |
| --- | --- | --- | --- | --- |
| 1 | 44/44 | 192.19 | 7857.88 | 12554.58 |
| 2 | 44/44 | 192.43 | 7842.58 | 12538.50 |
| 3 | 44/44 | 192.04 | 7865.70 | 12565.79 |

三轮均通过门限，共完成 132 个请求，抢占次数为 0。停服后 `verify-session` 通过。
