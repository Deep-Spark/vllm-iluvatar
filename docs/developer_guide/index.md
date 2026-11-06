# 开发者文档



本栏目写给开发、扩展和维护 vllm-iluvatar 插件的人。它关注插件如何接入 vLLM、哪些机制适合放在插件侧、遇到上游接口差异时应该怎样收口；用户如何安装、启动和调参仍见「用户指南」。

vllm-iluvatar 的定位是 out-of-tree plugin：尽量贴近公版 vLLM，不直接修改社区源码，同时把天数硬件平台、算子、模型、量化和通信相关改动承载在插件仓里。

## 当前页面



- [插件架构](plugin_architecture.md)：说明两个 vLLM entry point、主要接入路径和核心组件分工。
- [patch 与插件机制](patch_and_plugin.md)：说明早期 `.pth` 启动补丁、import hook、模块替身 shim 和 general patch 的边界。
- [开发环境搭建](dev_environment.md)：说明 CoreX 环境依赖、editable 安装、常用环境变量和 wheel 构建。
- [接入模型](model_integration.md)：说明快速适配社区模型与接入天数优化模型两种范式的判断、命名约定和注册点骨架。
- [自定义算子](custom_ops.md)：说明新增 `torch.ops._C` 算子时 community 与 plugin 两条流程的落位与注册点。
- [扩展量化](extend_quantization.md)：说明 compressed-tensors、KV Cache 与 ilu_runtime 三条量化线的扩展点与边界。

## 完整开发资料

以下三份资料不进入站点导航（`mkdocs.yaml` 的 `exclude_docs` 排除了 `design/`、`guides/`、`rfcs/`），
保留在仓库内供查阅：

- [插件设计文档](design/iluvatar_plugin_design.md)：插件目录结构、接入路径与完整设计。
- [开发指南目录](guides/)：面向具体功能的仓库内指南。
- [README](../../README.md)：Requirements、安装、Quick Start 与文档入口。
