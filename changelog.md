# Changelog

本文件记录 `vllm-iluvatar` 的版本变更。格式参考
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)。

**版本号与适配的 vLLM 版本保持一致**：`0.26.0` 适配 vLLM 0.26.0;当 vLLM
更新至 0.27.x 时,插件版本号同步更新。同一 vLLM 版本窗口内不再手工修改插件
版本号,其间的日常构建由 CI 追加 `-<git 日期>.<git 短哈希>-<构建版本>`
后缀加以区分。

本文件自 0.26.0 起开始维护，此前的变更请查阅 git 历史。

## [Unreleased]

### Added

- 新增多阶段 `Dockerfile` 与 `scripts/build_image.sh`:从源码构建整链路(ixinfer/ixformer/vllm 按 pin 克隆编译 + 本仓库插件),产出 CoreX 运行时镜像,并按仓库地址 + 打包日期打 tag。
- 新增《如何搭建 vllm-iluvatar 开发环境》开发文档。
- 新增 `LICENSE`(Apache-2.0)与项目 logo;`pyproject.toml` 声明 `license` 字段。
- 新增 DeepSeek-V4 Flash FP8 的 DSpark draft model，以及 KV cache 跨距校验。
- 新增 DeepSeek-V4 Flash 社区 block-outermost KV 布局 CPU 回归测试。
- 将 DeepSeek-V4 / V4.1 插件适配到 vLLM 0.31，并打通 sparse SWA
  bounded replay（`replay_start`）。
- DeepSeek-V4 FP8 原生 MXFP4（f4f8）routed experts 在
  `VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency` 且配置 backend 不是
  `deepep_low_latency` 时走 batched DP+EP：`IluMxfp4MoEMethod` 返回
  `mxfp4_w4a8` quant config，`select_gemm_impl` 复用
  `IluMoEBatchedExpertsFp8`，并在 `silu`/`swiglu` 缺少正的
  `swiglu_limit` 或权重不是 ixmm 布局时直接报错。把
  `--all2all_backend` 本身配成 `deepep_low_latency` 不会走这条实现。
- `IluDeepseekV4Fp8MoE` 保留 feat 上的 decode-only 低延迟入口：非 W4A8
  且实现了 `select_gemm_impl` 的方法（含原生 MXFP4）在
  `VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency` 且所有 rank 都是 decode
  时走 `forward_dp_ep_deep_ep`。Prefill 仍走社区 runner。转换后的 W4A8
  仍在整个 DP+EP step 上走 `forward_dp_ep`。纯低延迟 decode 的数值结果
  尚未覆盖。

### Removed

- 移除 `third_party/ixformer`、`third_party/vllm` 两个 submodule、`.gitmodules` 及仅用于驱动 submodule 构建的 `scripts/build_third_party.sh`;ixformer / vllm 的锁定 commit 改由根目录 `Dockerfile` 的构建参数(`IXFORMER_COMMIT` / `VLLM_COMMIT`)管理。

### Changed

- 量化模块按职责重新组织:`configs/`(checkpoint 配置)、`backends/`(执行
  实现)、`ilu_runtime/`(runtime 配置驱动的加载期量化)、`conversions/`(格式转换)、`common/`。顶层
  `ilu_*.py` 不再逐个自动发现,改由 `_registration.py` 按依赖顺序显式导入。
  对外注册的量化方法名与行为不变。
- **(不兼容)** 四个 online 量化方法改名,`additional_config` 中的
  `iluvatar.method` 需同步更新:`IluFp8ToW4A8NNMoEMethod` →
  `IluFp8ToW4A8NNOnlineMoEMethod`、`IluFp8ToW8A8Int8LinearMethod` →
  `IluFp8ToW8A8Int8OnlineLinearMethod`、`IluMxfp4ToW4A8TNMoEMethod` →
  `IluMxfp4ToW4A8TNOnlineMoEMethod`、`IluNativeToW8A8Int8LinearMethod` →
  `IluNativeToW8A8Int8OnlineLinearMethod`。旧名字不再解析,沿用旧名的配置
  会在启动时报错并列出当前已注册的方法名。
- 面向开源:README 精简为公开版(logo/badge/License 小节);清理 guide 文档中的内部引用(私有数据路径改为 `/path/to/model`、移除内部 bitbucket commit 链接)。
- GLM / DeepSeek-V4 MoE 构造改为社区 `FusedMoEFactory`（0.27 去掉 `FusedMoE` 导出）。
- `IluMultiHeadLatentAttentionWrapper.__init__` 增加
  `non_causal_multi_token_decode` 与
  `allow_short_prefill_indexer_scoring_skip`，满足 OOT 前缀签名检查。
- `IluCommunicator` 接受并转发 `use_all2all`，对齐 0.27 `GroupCoordinator`。
- DeepSeek-V4 Flash FP8 KV spec 显式提供 `state_content_bytes=584`，适配
  vLLM 0.30 的 `AttentionSpec` 页宽计算，并保留社区 KV cache 分组、准入和
  block-outermost 分配。
- `version.json` 版本与依赖窗口改为 0.31；ixformer / vllm 锁定 commit 改由
  `Dockerfile` 的 `IXFORMER_COMMIT` / `VLLM_COMMIT` 管理。ixformer pin 跟随
  `master@573eda3f`（已包含 V4.1 `739abab2`）。

### Fixed

- 兼容 CoreX Triton 3.2 缺少的 `triton.knobs`、`tl.mul`、Gluon 和
  `core._aggregate` 等接口，并镜像修复 penalties/rejection kernel 不支持的
  Triton 语法。
- DeepSeek-V4 Flash FP8 sparse MLA 使用 ixformer a64 gather，修复单层 KV
  cache 跨距超过 4 GiB 时的 32 位 per-lane 地址截断。
- `VLLM_ILU_DISABLE_DEEPEP=1` 时低延迟 all2all 不再要求安装 `deep_ep`：
  基类断言放行，RDMA size hint 跳过，`deep_ep` 仅在构造真实 Buffer 时
  导入。high-throughput 路径仍依赖该包。

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
