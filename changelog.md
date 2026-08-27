# Changelog

本文件记录 `vllm-iluvatar` 的版本变更。格式参考
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)。

**版本号与所适配的 vLLM 版本对齐**：`0.26.0` 适配 vLLM 0.26.0，vLLM 走到
0.27.0 时插件同步跳到 0.27.0。同一个 vLLM 版本窗口内插件不再手工改版本号，
其间的日常构建由 CI 追加 `-<git日期>.<git短哈希>-<构建版本>` 后缀区分。

本文件自 0.26.0 起开始维护，此前的变更请查阅 git 历史。

## [Unreleased]

## [0.26.0] - 2026-08-27

首个正式标记的版本，适配 vLLM 0.26.0 / CoreX 5.0.0。

### Added

- `IluPlatform`：CUDA 兼容的 CoreX 平台适配层，经 `vllm.platform_plugins`
  入口点注册。
- `IluCommunicator`：基于 `ixformer.distributed` 集合通信的通信后端。
- `vllm_iluvatar._C`：原生扩展，提供 vLLM 兼容的 `_C` 算子（`rms_norm`、
  `weak_ref_tensor`、`get_cuda_view_from_cpu_tensor`）。
- `vllm_iluvatar.kernels`：面向 vLLM V1 的天数 IR 算子 provider。
- 模型与量化适配，以及 worker 启动期所需的运行时补丁，经
  `vllm.general_plugins` 入口点注册。
- `IluNixlConnector` / `IluNixlPullConnector` / `IluNixlPushConnector`：
  用于 PD 分离部署的 KV 传输连接器。

以上全部通过 vLLM 的 out-of-tree plugin 机制加载，不修改 vLLM 主仓源码。
