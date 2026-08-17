# vLLM-Iluvatar DP + EP 使用指南

本文介绍如何在 Iluvatar GPU 上使用 vLLM 的 Data Parallel（DP）与 Expert
Parallel（EP），重点说明 DeepSeek-V4 的 Hybrid EP 配置、环境变量、服务
启动命令和调度机制。

## 目录

1. [适用范围](#1-适用范围)
2. [DP + EP 与 Hybrid EP](#2-dp--ep-与-hybrid-ep)
   - [基本执行流程](#21-基本执行流程)
   - [Hybrid EP 的目的](#22-hybrid-ep-的目的)
   - [必须保持的安全不变量](#23-必须保持的安全不变量)
   - [在模型中接入混合 EP Backend](#24-在模型中接入混合-ep-backend)
3. [环境变量](#3-环境变量)
   - [Hybrid backend](#31-hybrid-backend)
   - [DP 请求分配与 Prefill 同步](#32-dp-请求分配与-prefill-同步)
   - [DP + EP MoE 计算](#33-dp--ep-moe-计算)
   - [LL buffer 容量](#34-ll-buffer-容量)
4. [推荐启动配置](#4-推荐启动配置)
   - [DP + EP 核心环境变量](#41-dp--ep-核心环境变量)
   - [DP + EP 最小核心服务命令](#42-dp--ep-最小核心服务命令)
5. [Backend 选择规则](#5-backend-选择规则)
   - [当前限制](#51-当前限制)
6. [DP Prefill 调度为何必要](#6-dp-prefill-调度为何必要)
   - [短请求](#61-短请求)
   - [长输出](#62-长输出)
   - [已验证结果](#63-已验证结果)
7. [常见问题](#7-常见问题)
   - [TTFT 变成约两倍](#71-ttft-变成约两倍)
   - [LL buffer OOM](#72-ll-buffer-oom)
8. [相关提交](#8-相关提交)
9. [附录](#9-附录)
   - [关键文件](#91-关键文件)
   - [配置步骤速览](#92-配置步骤速览)

## 1. 适用范围

本文配置适用于以下场景：

1. 模型包含 routed experts，且已经由 `vllm-iluvatar` 支持 DP + EP。
2. 单机使用多个 DP rank，并让 expert 横跨 TP × DP rank 分布。
3. Prefill 使用吞吐优先的 backend，Decode 使用
   `deepep_low_latency`（下文简称 LL）。
4. 参考拓扑为 TP=4、DP=4、PP=1，共使用 16 张 GPU。
5. 参考模型为 DeepSeek-V4 W4A8 TN 模型。

启动参数中：

- `--tensor-parallel-size 4` 创建 4 路 TP；
- `--data-parallel-size 4` 创建 4 个 DP engine；
- `--enable-expert-parallel` 启用 EP；
- PP=1 时，总 GPU 数为 `TP × DP = 16`；
- EP world size 为 TP 与 DP 参与 MoE 的全部 rank 数，即本例中的 16。

## 2. DP + EP 与 Hybrid EP

### 2.1 基本执行流程

DP + EP 下，每个 DP rank 接收不同请求，但 MoE routed experts 分布在整个
EP group 上。一次 MoE forward 的简化流程为：

```text
输入 token
  -> 路由与 top-k
  -> EP dispatch
  -> 本地 expert 计算
  -> EP combine
  -> 输出 token
```

不同 DP rank 虽然处理不同请求，但 EP collective 必须按相同顺序执行。任意
rank 选择了不同的通信 backend、少执行或多执行一次 collective，都可能造成
全局等待或 hang。

### 2.2 Hybrid EP 的目的

Prefill 和 Decode 的通信形态不同：

1. Prefill token 较多，通常更适合吞吐优先的
   `allgather_reducescatter` 或 DeepEP High Throughput（HT）。
2. Decode 每个请求每步通常只有一个 token，更适合
   `deepep_low_latency`。

Hybrid EP 允许二者使用不同 backend：

```text
Prefill 或 DP rank 状态混合
  -> 使用 --all2all_backend 指定的 backend

所有 DP rank 都是 decode-only
  -> 使用 VLLM_ILU_DECODE_EP_BACKEND 指定的 LL backend
```

推荐组合为：

```bash
--enable-expert-parallel \
--all2all_backend=allgather_reducescatter

export VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency
```

本文的 “Hybrid EP” 是指 Prefill 使用 AG/RS、所有 DP rank 纯 Decode 时
切换到 LL，不是 vLLM 的 `--data-parallel-hybrid-lb`。参考脚本没有启用
后者。

### 2.3 必须保持的安全不变量

Hybrid EP 必须满足：

> 只有所有参与的 DP rank 都是 decode-only 时，才允许统一切换到 LL。

不能根据单个 rank 的本地 metadata 选择 backend。当前实现会在
[`v1/worker/dp_utils.py`](../../vllm_iluvatar/v1/worker/dp_utils.py)
中同步各 DP rank 的 decode-only 状态，通过 `DPMetadata` 和
`ForwardContext` 传递 `all_ranks_decode_only`，最后由
[`ilu_fused_moe_ep.py`](../../vllm_iluvatar/model_executor/layers/fused_moe/ilu_fused_moe_ep.py)
选择 backend。

当各 DP rank 处于 Prefill/Decode 混合状态时，实现还会同步关闭该 step 的
CUDA Graph replay，并使用 CUDA Graph padding 之前的 token 数计算 DP
padding，避免 rank 间进入不同执行路径。

### 2.4 在模型中接入混合 EP Backend

模型需要通过 `IluFusedMoEEP` 接入 Hybrid EP。参考当前 DeepSeek-V4 MoE
实现
[`models/deepseek_v4/model.py`](../../vllm_iluvatar/models/deepseek_v4/model.py)，
接入步骤如下。

1. 导入 `IluFusedMoEEP`，让模型的 MoE 类同时继承 `nn.Module` 和
   `IluFusedMoEEP`。完成 routed experts、shared experts、quant method
   和 `prefix` 初始化后调用 `self.init_ep()`。

   ```python
   from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
       IluFusedMoEEP,
   )


   class ModelMoE(nn.Module, IluFusedMoEEP):
       def __init__(self, config, ...):
           super().__init__()
           self.experts = ...

           # 需要在最后
           self.init_ep()
   ```

2. 在 MoE `forward()` 中先通过 `_should_use_forward_dp_ep()` 判断是否进入
   DP + EP 路径，再调用 `forward_dp_ep()`；未进入时保留模型原有的
   `FusedMoE` forward。如果模型复用了 shared-expert 的预量化输入，应继续
   传递 `shared_gate_up_input`。

   ```python
   org_shape = hidden_states.shape
   if self._should_use_forward_dp_ep():
       final_hidden_states = self.forward_dp_ep(
           hidden_states,
           input_ids,
       )
       return final_hidden_states.view(org_shape)
   ```

合并后的示例如下：

```python
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
)


class ModelMoE(nn.Module, IluFusedMoEEP):
    def __init__(self, config, ...):
        super().__init__()
        self.experts = ...

        # 需要在最后
        self.init_ep()

    def forward(self, hidden_states, input_ids=None):
        org_shape = hidden_states.shape
        if self._should_use_forward_dp_ep():
            final_hidden_states = self.forward_dp_ep(
                hidden_states,
                input_ids,
            )
            return final_hidden_states.view(org_shape)

        # 保留模型原有的 FusedMoE forward。
        final_hidden_states = ...
        return final_hidden_states.view(org_shape)
```

当配置的 Prefill backend 不是 LL、Decode override 为 LL 时，`init_ep()`
会提前创建并缓存 LL manager 和通信 handle，避免在 CUDA Graph capture
中初始化通信资源。未进入 DP + EP 路径时，应保留模型原有的 `FusedMoE`
forward。

模型层只负责接入 `IluFusedMoEEP`，不应自行根据本 rank 的 token 数判断
Prefill 或 Decode。`forward_dp_ep()` 会读取同步后的
`DPMetadata.all_ranks_decode_only`，统一选择 Prefill 或 Decode backend。
例如：

```bash
export VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency

vllm serve /path/to/model \
  --enable-expert-parallel \
  --all2all_backend=allgather_reducescatter
```

该配置下，Prefill 或 Prefill/Decode 混合 step 使用
`allgather_reducescatter`；所有 DP rank 都是 decode-only 时使用
`deepep_low_latency`。

## 3. 环境变量

### 3.1 Hybrid backend

#### `VLLM_ILU_DECODE_EP_BACKEND`

- 默认值：未设置。
- 推荐值：`deepep_low_latency`。
- 作用：当且仅当 `all_ranks_decode_only=true` 时，将 Decode 的 EP backend
  覆盖为 LL；Prefill 仍使用 `--all2all_backend`。
- 合法覆盖值：当前仅支持 `deepep_low_latency`。
- 对应提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。

设置该变量后，LL manager 和通信 handle 会由 `init_ep()` 在 CUDA Graph
capture 前预初始化，避免在 stream capture 期间执行 barrier。LL MoE
kernel 由 profile/warmup 或首次对应 forward 路径完成初始化。

#### `VLLM_ILU_DISABLE_DEEPEP`

- 默认值：`0`。
- 参考脚本：`1`。
- 作用：设为 `1` 时，LL manager 不创建原生 `deep_ep.Buffer`，而是使用
  `BufferWrapper`，通信 buffer 由 ixformer NVSM EP 路径初始化。
- 对应提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。

该变量名称表示是否禁用原生 DeepEP buffer，不表示关闭 Hybrid LL 语义。
参考脚本使用 `VLLM_ILU_DISABLE_DEEPEP=1` 和
`VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency`，实际走的是
Iluvatar/ixformer 的 LL 通信实现。

#### `VLLM_ILU_DEEPEP_USE_INT8`

- 默认值：`0`。
- 参考脚本：`1`。
- 作用：控制 LL dispatch buffer 是否使用 INT8，以降低通信量。
- 对应提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。

该开关需要与模型量化实现、dispatch scale 和当前 ixformer kernel 组合一起
验证精度，不能只根据 HTTP 请求成功判断正确性。

#### `VLLM_ILU_DEEPEP_USE_SCATTER`

- 默认值：`1`。
- 作用：在 TP > 1 时，先按 TP rank scatter token，再执行 EP dispatch，
  避免每个 TP rank 重复发送相同 token。
- 对应提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。

该变量同时作用于 HT 和 LL。关闭后会增加 dispatch 通信量，通常只用于
A/B 定位。

#### `VLLM_ILU_LL_FUSED_AG_AR`

- 默认值：`1`。
- 生效条件：`VLLM_ILU_DISABLE_DEEPEP=1`。
- 作用：在 LL 路径融合 TP All Gather/All Reduce 相关处理，减少独立通信和
  中间数据搬运。
- 对应提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。

### 3.2 DP 请求分配与 Prefill 同步

#### `VLLM_ILU_DP_ROUND_ROBIN_LB`

- 默认值：`0`。
- 推荐值：Hybrid EP 场景设为 `1`。
- 作用：使用跨 API server 进程共享的 round-robin 计数器，将请求稳定地
  分发到各 DP engine，减少 coordinator 负载统计滞后造成的 rank 失衡。
- 对应提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。
- 测试与边界补充：
  [`3f2a6e81`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/3f2a6e812df442b0d6d961ae433905929d935463)。

Round-robin 不能单独保证所有 DP rank 永远同时进入 Prefill，但能显著提高
请求 wave 对齐概率，是下面 Prefill 同步策略的推荐前置配置。

默认状态文件包含 API server 的父 PID。若多个 API server 不共享同一个父
PID，默认文件也不会共享；此时若需要跨 server 的全局 round-robin，必须
显式设置同一个可共享且权限正确的
`VLLM_ILU_DP_ROUND_ROBIN_LB_FILE`。

#### `VLLM_ILU_DP_ROUND_ROBIN_LB_FILE`

- 默认值：空。
- 作用：指定跨 API server 进程共享的 round-robin 状态文件。
- 未设置时：使用
  `/tmp/vllm-dp-rr-<parent-pid>-<num-engines>.state`。
- 对应提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。

只有需要让多个独立服务进程共享同一 DP 分配序列时才应显式设置。不同服务
实例不要误用同一个文件。

#### `VLLM_ILU_DP_ENABLE_SYNC_PREFILLS`

- 默认值：当 Decode override 为 `deepep_low_latency` 且 round-robin 开启
  时为 `1`，其它情况为 `0`。
- 参考脚本：显式设为 `1`。
- 作用：在调度前同步各 DP EngineCore 的 Prefill-ready 和 Decode-busy
  状态；当只有部分 rank ready 时，有界推迟该 Prefill wave。
- 安装条件：还须 `VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS > 0`，否则
  EngineCore wrapper 不会安装，本开关单独开启无效。
- 对应提交：
  [`76d338dd`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/76d338dd7b10d4d7b7be8e371e4ec9413ce377a0)。
- collective 顺序修复：
  [`924613a5`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/924613a55c0d0a5a9adc77ba702f4f19f2040cac)。

同步使用独立 CPU process group。参与一次 Prefill 同步的每个 rank 都必须
继续执行且只执行一个真实或 dummy worker step，否则 EngineCore collective
和 worker collective 会错位并 hang。

#### `VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS`

- 默认值：`8`。
- 负值处理：限制为 `0`。
- 作用：部分 DP rank 有 Prefill、其它 rank 仍 Decode 时，允许等待的最大
  模型 step 数；超过上限后 fail-open，避免 Prefill 饥饿。
- 对应提交：
  [`76d338dd`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/76d338dd7b10d4d7b7be8e371e4ec9413ce377a0)。

当部分 rank ready 但没有 rank Decode 时，实现最多只短暂聚合两个 step，
同时仍受该上限约束。

#### `VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS`

- 默认值：Hybrid LL + round-robin 时为 `5`，其它配置为 `0`。
- 负值处理：限制为 `0`。
- 作用：某个 rank 执行完模型并清空本地 scheduler 后，短暂等待后台线程把
  replacement request 放入输入队列，避免过早进入 dummy batch。
- 对应提交：
  [`76d338dd`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/76d338dd7b10d4d7b7be8e371e4ec9413ce377a0)。

该等待解决短请求完成后的输入投递竞态；长输出下的 DP phase drift 仍需由
`VLLM_ILU_DP_ENABLE_SYNC_PREFILLS` 解决。不要用较大的固定 sleep 代替跨
rank 同步。`patch_core.py` 只在本变量大于 0 且
`VLLM_ILU_DP_ENABLE_SYNC_PREFILLS` 开启时安装 wrapper。

### 3.3 DP + EP MoE 计算

#### `VLLM_ILU_MOE_EP_LL_MOE_ALGO`

- 默认值：`GEMV`。
- 参考脚本：`FUSED`。
- 作用：选择 naive/AG-RS DP + EP 分支中 routed experts 的实现。
- `GEMV`：使用 modular MoE/GEMV 路径。
- `FUSED`：当 quant method 提供 `apply_fused_moe` 时，使用 ixformer
  fused W4A8 MoE；否则回退到 modular 路径。
- 初始提交：
  [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)。
- 当前 fused 路径补充提交：
  [`c6e9872e`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/c6e9872e978bb934839d1bba0420fdc1c6b2cadb)。

当前 W4A8 fused MoE 要求 `format=TN`、`version=2`、`group_size=-1`。
其它格式会在加载期报 `NotImplementedError`，不应静默使用。

该变量在 Python 模块 import 时读取并缓存，不能在服务运行中动态切换。

### 3.4 LL buffer 容量

当 `VLLM_ILU_DISABLE_DEEPEP=1`、LL 使用 `BufferWrapper`/NVSM 路径时，
Hybrid LL buffer 使用 `scheduler_config.max_num_seqs` 估算每个 rank 的
最大 token 容量，而不是使用 `max_num_batched_tokens`。对应提交：
[`4f22186d`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/4f22186df9a4b7d385d72093b47f4102e4e69a59)。

当前实现中，当 TP scatter 或 fused AG/AR 开启时，容量还会除以 TP size。

当 `VLLM_ILU_DISABLE_DEEPEP=0`、使用原生 `deep_ep.Buffer` 时，handle
仍根据 `moe_config.max_num_tokens` 创建，不适用上述 NVSM buffer 估算。

启动时直接设置 `--max-num-seqs`，并让 CUDA Graph capture size 与其保持
一致：

```bash
--max-num-seqs 64
--compilation-config \
  '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0,"max_cudagraph_capture_size":64}'
```

容量过大会浪费数百 MiB 显存，并挤压 KV Cache 或 MoE workspace；容量过小
则可能造成越界或请求失败。修改 `--max-num-seqs` 后必须重新检查 LL buffer
和 CUDA Graph capture size。

## 4. 推荐启动配置

### 4.1 DP + EP 核心环境变量

```bash
export VLLM_ILU_DP_ROUND_ROBIN_LB=1
export VLLM_ILU_DP_ENABLE_SYNC_PREFILLS=1
export VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS=8
export VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS=5

export VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency
export VLLM_ILU_DISABLE_DEEPEP=1
export VLLM_ILU_DEEPEP_USE_INT8=1
```

### 4.2 DP + EP 最小核心服务命令

```bash
MODEL=/newdata/nlp/DeepSeek-V4/DeepSeek-V4-Pro-w4a8-v2-TN

export VLLM_ILU_DP_ROUND_ROBIN_LB=1
export VLLM_ILU_DP_ENABLE_SYNC_PREFILLS=1
export VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS=8
export VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS=5

export VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency
export VLLM_ILU_DISABLE_DEEPEP=1
export VLLM_ILU_DEEPEP_USE_INT8=1

VLLM_TP_SIZE=4
VLLM_DP_SIZE=4
VLLM_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0,"max_cudagraph_capture_size":64}'

cd /tmp
vllm serve "${MODEL}" \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --tensor-parallel-size "${VLLM_TP_SIZE}" \
  --data-parallel-size "${VLLM_DP_SIZE}" \
  --enable-expert-parallel \
  --all2all_backend allgather_reducescatter \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --compilation-config "${VLLM_COMPILATION_CONFIG}" \
  --served-model-name DeepSeek
```

## 5. Backend 选择规则

当前 DP + EP 自定义路径要求同时满足：

1. DP size 大于 1。
2. 已启用 EP。
3. `--all2all_backend` 是 `naive`、`allgather_reducescatter`、
   `deepep_low_latency` 或 `deepep_high_throughput`。

满足条件后，按以下顺序选择 backend：

1. 所有 DP rank 都是 decode-only，且
   `VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency`：使用 LL。
2. 配置 backend 本身是 `deepep_low_latency`：使用 LL。
3. 配置 backend 是 `deepep_high_throughput`：使用 HT。
4. 配置 backend 是 naive/AG-RS 别名：使用 naive AG/reduce 路径。
5. 已经进入 `forward_dp_ep()` 后，其它未命中 override 的情况回退到配置的
   modular MoE 路径。

### 5.1 当前限制

1. LL 路径不支持 `pcp_size > 1`，会抛出 `NotImplementedError`。
2. 当前 NVSM LL modular kernel 支持的 hidden size 为 2048、2560、3072、
   4096、5120、6144、7168 和 8192。
3. `VLLM_ILU_DISABLE_DEEPEP=1` 仍依赖 Iluvatar NVSM LL 能力，不表示关闭
   EP 通信。
4. sync-prefill 每个活跃 EngineCore step 会增加一次小型 CPU DP
   all-reduce；多节点和非 round-robin 部署必须单独压测。

## 6. DP Prefill 调度为何必要

### 6.1 短请求

某个 rank 完成本地请求后，输出线程尚未把结果传回 API，输入线程也可能尚未
把 replacement request 放入 EngineCore 队列。若 EngineCore 立即开始下一
轮，该 rank 会进入 dummy batch，而其它 rank 已经开始真实 Prefill。

`VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS=5` 为该后台往返提供有界等待窗口。

### 6.2 长输出

各 DP rank 的 Decode 完成时间会逐渐偏离。较早收到 replacement request
的 rank 可能开始 Prefill，而其它 rank 仍在 Decode。此时一个逻辑 Prefill
wave 会被拆成两个昂贵的全局 EP step。

`VLLM_ILU_DP_ENABLE_SYNC_PREFILLS=1` 会在调度前同步：

1. 本 rank 是否有 Prefill-ready 请求；
2. 本 rank 是否仍有 Decode；
3. ready rank 数和 decode-busy rank 数；
4. 是否需要暂时 throttle 当前 partial Prefill wave。

当所有 rank 都 ready 时一起释放 Prefill；超过 defer 上限后 fail-open。
被 skip 的 waiting 请求不计入 Prefill-ready。

### 6.3 已验证结果

ISL=1024、OSL=1024、并发 4 的 fixed-sleep 基线为：

- Mean TTFT：1181.65 ms；
- P99 TTFT：1399.35 ms；
- Mean TPOT：46.30 ms。

启用插件侧 DP Prefill 同步后：

- Mean TTFT：696.54 ms；
- P99 TTFT：705.46 ms；
- Mean TPOT：45.86 ms；
- 完成请求：40/40。

这些数据说明优化主要消除了 DP phase 不对齐导致的额外 Prefill step，而
不是通过改变模型计算降低 TPOT。

## 7. 常见问题

### 7.1 TTFT 变成约两倍

通常表示 Prefill wave 被拆分：

1. 一部分 rank 执行真实 Prefill；
2. 其它 rank 执行 dummy 或 Decode，并在 EP collective 中等待；
3. 下一 step 两组 rank 交换角色。

优先检查 round-robin、replacement wait 和 Prefill 同步，不要先对每层
kernel 做大范围 profiling。

### 7.2 LL buffer OOM

检查：

1. `--max-num-seqs` 是否远大于实际并发；
2. `max_cudagraph_capture_size` 是否与 `--max-num-seqs` 一致；
3. TP scatter/fused AG-AR 开启时容量是否按 TP size 缩放；
4. KV Cache 之后是否仍为 MoE workspace 留有足够显存。

## 8. 相关提交

相关提交：

1. [`6752b169`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/6752b1698841567fc98b12c465fbba8632e8de23)：
   初始支持 DP + EP、Hybrid EP backend 和 round-robin LB。
2. [`3f2a6e81`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/3f2a6e812df442b0d6d961ae433905929d935463)：
   补充 DP + EP 测试、边界检查和说明。
3. [`4f22186d`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/4f22186df9a4b7d385d72093b47f4102e4e69a59)：
   Hybrid LL buffer 容量改用 `max_num_seqs`。
4. [`76d338dd`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/76d338dd7b10d4d7b7be8e371e4ec9413ce377a0)：
   优化 DP Prefill 调度，增加同步 admission 和 replacement wait。
5. [`924613a5`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/924613a55c0d0a5a9adc77ba702f4f19f2040cac)：
   修复 Prefill 同步中的 collective 顺序，保证每轮同步对应一个 worker
   step。
6. [`c6e9872e`](http://bitbucket.iluvatar.ai:7990/projects/AS/repos/vllm-iluvatar/commits/c6e9872e978bb934839d1bba0420fdc1c6b2cadb)：
   naive/AG-RS 路径补充 fused MoE。

## 9. 附录

### 9.1 关键文件

1. [`envs.py`](../../vllm_iluvatar/envs.py)：定义 DP、Hybrid EP 和 LL
   相关环境变量及默认值。
2. [`ilu_fused_moe_ep.py`](../../vllm_iluvatar/model_executor/layers/fused_moe/ilu_fused_moe_ep.py)：
   实现 DP + EP MoE 数据流、Hybrid backend 选择、TP scatter、fused
   AG/AR 和 LL 初始化。
3. [`ilu_deepep_ll.py`](../../vllm_iluvatar/model_executor/layers/fused_moe/prepare_finalize/ilu_deepep_ll.py)：
   实现 LL prepare/finalize，以及 NVSM dispatch 和 combine。
4. [`all2all.py`](../../vllm_iluvatar/distributed/device_communicators/all2all.py)：
   创建 AG/RS、DeepEP HT 和 DeepEP LL all-to-all manager。
5. [`dp_utils.py`](../../vllm_iluvatar/v1/worker/dp_utils.py)：同步 DP batch、
   CUDA Graph mode 和 `all_ranks_decode_only`。
6. [`core_client.py`](../../vllm_iluvatar/v1/engine/core_client.py)：实现 DP
   round-robin 请求分配。
7. [`core.py`](../../vllm_iluvatar/v1/engine/core.py)：实现 replacement
   request 等待和跨 DP rank 的 Prefill admission 同步。
8. [`patch_core.py`](../../vllm_iluvatar/ilu_patch/general_patches/patch_core.py)：
   安装 EngineCore wrapper，并校验社区方法的 fingerprint 和签名。

### 9.2 配置步骤速览

1. 根据 GPU 数量确定 TP 和 DP，PP=1 时总 GPU 数为 `TP × DP`。
2. 使用 `--enable-expert-parallel` 启用 EP。
3. 使用 `--all2all_backend=allgather_reducescatter` 配置 Prefill backend。
4. 设置 `VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency`，仅在所有 DP
   rank 都是 decode-only 时切换到 LL。
5. 启用 DP round-robin 和 Prefill 同步，使各 rank 的请求 wave 与
   collective 顺序保持一致。
6. 根据目标并发直接设置 `--max-num-seqs`，并保持
   `max_cudagraph_capture_size` 与其一致。
7. 启动服务后检查各 DP rank 是否同步推进，并分别关注 TTFT 与 ITL。
