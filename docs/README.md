# vLLM-Iluvatar

`vllm-iluvatar` 是天数 GPU 的 vLLM 插件。它以 out-of-tree plugin 形式接入 vLLM，
用户仍然用 `vllm serve` 启动服务，调用方式与 vLLM 生态保持一致。

## 从这里开始

| 你要做的事 | 去这里 |
| --- | --- |
| 第一次把服务跑起来 | [快速开始](getting_started/quickstart.md) |
| 装环境、确认装对了 | [安装](getting_started/installation.md) |
| 查支持哪些模型 | [支持模型](models/supported_models.md) |
| 查版本配套关系 | [版本说明](release/index.md) |

## 本文档使用说明

本文档用于在天数 GPU 上部署、验证和日常使用 vLLM 软件栈：怎么装、怎么起服务、
支持哪些模型、版本之间怎么配套。

插件开发、新模型适配和发版流程见左侧「开发者指南」。
公版 vLLM 的基础概念、通用 API 和生态集成请查 [vLLM 官方文档](https://docs.vllm.ai/)。
