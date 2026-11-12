# 版本说明

本页汇总 `vllm-iluvatar` 与天数 vLLM 的配套版本说明。部署或升级前，请先确认使用的
vLLM 本体、`vllm-iluvatar` 插件、镜像、driver 和 CoreX runtime 来自同一次发布。

## 已发布版本

| 版本 | 状态 | 说明 |
| --- | --- | --- |
| [vLLM v0.27.1](0.27.1/index.md) | 已发布 | 首个 out-of-tree plugin 架构版本，x86_64 和 aarch64 均已发布。 |

## 阅读建议

- 首次部署时，按[安装](../getting_started/installation.md)页确认镜像和驱动。
- 升级或替换模型权重前，阅读对应版本说明中的量化、环境变量和启动参数变化。
- 多版本共存或排查问题时，以具体版本页面中的版本配套关系为准。
