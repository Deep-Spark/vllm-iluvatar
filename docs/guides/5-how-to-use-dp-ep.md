# vLLM-Iluvatar DP + EP 使用指南

本文介绍如何在 Iluvatar GPU 上使用 vLLM 的 Data Parallel（DP）与 Expert
Parallel（EP），重点说明 DeepSeek-V4 的 Hybrid EP 配置、环境变量、服务
启动命令和调度机制，并区分 GLM-5.2 默认 MoERunner 与自定义 DP + EP 路径。

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
   - [确认配置生效](#43-确认配置生效)
   - [GLM-5.2 的入口配置](#44-glm-52-的入口配置)
5. [Backend 选择规则](#5-backend-选择规则)
   - [当前限制](#51-当前限制)
   - [本轮改动的验证范围](#52-本轮改动的验证范围)
6. [DP Prefill 调度为何必要](#6-dp-prefill-调度为何必要)
   - [短请求](#61-短请求)
   - [长输出](#62-长输出)
   - [已验证结果](#63-已验证结果)
7. [常见问题](#7-常见问题)
   - [TTFT 变成约两倍](#71-ttft-变成约两倍)
   - [LL buffer OOM](#72-ll-buffer-oom)
8. [变更记录](#8-变更记录)
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
以下示例描述 DeepSeek-V4 的接入方式。GLM-5.2 默认使用 MoERunner，
仅在纯 Decode 命中 override 时进入自定义路径，两者的入口不能直接互换。

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

GLM-5.2 的入口参考
[`models/glm_moe_dsa/model.py`](../../vllm_iluvatar/models/glm_moe_dsa/model.py)：

```python
# router_in 是展开后的浮点输入；shared_gate_up_input 是可选的预量化输入。
if self._use_decode_ep_override():
    return self.forward_dp_ep(
        router_in,
        shared_gate_up_input=shared_gate_up_input,
    ).view(org_shape)

# 其余情况交给 self.experts / MoERunner，保留原有内部或外部 gate 接口，
# 并将 norm 提供的量化结果继续传入 runner。
```

`_use_decode_ep_override()` 要求已设置 override 且
`DPMetadata.all_ranks_decode_only=true`；metadata 缺失时不触发。
初始化时 `init_ep()` 会校验 override 是否适用于该层。不要将 GLM 的入口
改成仅判断 `_should_use_forward_dp_ep()`，否则 AG/RS 会绕过默认 runner。

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

- 默认值：未设置（空字符串也视为未设置）。GLM 默认使用 MoERunner。
- 合法覆盖值：`naive`、`deepep_low_latency`；其它非空值报错。
- 作用：仅在所有 DP rank 都是 decode-only 时覆盖模型的执行入口；
  Prefill/混合批次继续使用模型原入口和配置的 backend。
- `naive`：调用 `forward_dp_ep_naive()`；要求配置 backend 为
  `allgather_reducescatter`（内部也接受 `ag-rs`、`ag_rs`），不能与
  DeepEP backend 搭配。CLI 的 `--all2all-backend=naive` 是已弃用的兼容值，
  当前 vLLM `ParallelConfig` 会先将它归一化为 `allgather_reducescatter`，
  因而能通过该校验；新配置应直接使用 `allgather_reducescatter`。
- `deepep_low_latency`：调用 LL 路径，用于本文的 Hybrid LL 配置。
- 两种 override 都要求该层满足第 5 节的自定义 DP + EP 准入条件，
  否则初始化报错。该变量在模块 import 时读取，修改后需重启服务。

仅当 override 为 `deepep_low_latency` 且配置 backend 不是 LL 时，
`init_ep()` 才安排 LL manager 和通信 handle 在 CUDA Graph capture 前
预初始化。`naive` override 不初始化 LL handle。LL MoE kernel 由
profile/warmup 或首次对应 forward 路径完成初始化。

#### `VLLM_ILU_DISABLE_DEEPEP`

- 默认值：`0`。
- 参考脚本：`1`。
- 作用：设为 `1` 时，LL manager 不创建原生 `deep_ep.Buffer`，而是使用
  `BufferWrapper`，通信 buffer 由 ixformer NVSM EP 路径初始化。

该变量名称表示是否禁用原生 DeepEP buffer，不表示关闭 Hybrid LL 语义。
参考脚本使用 `VLLM_ILU_DISABLE_DEEPEP=1` 和
`VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency`，实际走的是
Iluvatar/ixformer 的 LL 通信实现。

#### `VLLM_ILU_DEEPEP_USE_INT8`

- 默认值：`0`。
- 参考脚本：`1`。
- 作用：控制 LL dispatch buffer 是否使用 INT8，以降低通信量。

该开关需要与模型量化实现、dispatch scale 和当前 ixformer kernel 组合一起
验证精度，不能只根据 HTTP 请求成功判断正确性。

#### `VLLM_ILU_DEEPEP_USE_SCATTER`

- 默认值：`1`。
- 作用：在 TP > 1 时，先按 TP rank scatter token，再执行 EP dispatch，
  避免每个 TP rank 重复发送相同 token。

该变量同时作用于 HT 和 LL。关闭后会增加 dispatch 通信量，通常只用于
A/B 定位。

#### `VLLM_ILU_LL_FUSED_AG_AR`

- 默认值：`1`。
- 生效条件：`VLLM_ILU_DISABLE_DEEPEP=1`。
- 作用：在 LL 路径融合 TP All Gather/All Reduce 相关处理，减少独立通信和
  中间数据搬运。

#### `VLLM_ILU_GLM_LL_SHARED_DISPATCH_OVERLAP`

- 默认值：`0`；设置为 `1` 后重启模型进程。
- 作用：GLM-5.2 的 LL 路径在 dispatch 提交前，将 shared MLP 提交到辅助
  CUDA stream，与主流上的 dispatch 并行；dispatch receiver 返回后汇合，
  再执行 routed experts。shared MLP 不再延后到 combine/AG 阶段。
- 与 `VLLM_ILU_LL_FUSED_AG_AR` 兼容：fused combine/AG/AR 直接接收已完成的
  shared 输出，不再拆出独立的 shared TP all-reduce 和加法。
- 生效条件：LL 路径、有 shared expert、输入 token 数大于 0；TP=1 也可
  与 EP dispatch 重叠。该开关与全局 `VLLM_DISABLE_SHARED_EXPERTS_STREAM` 独立。

该开关仅改变执行调度。实际收益需在目标并发与 CUDA Graph 模式下对照验证。

#### `VLLM_ILU_MOE_AG_SHARED_OVERLAP`

- 默认值：`0`；设置为 `1` 后重启模型进程。
- 作用：符合调度条件的 MoERunner 使用 AG/RS 时，将本地 shared MLP 提交到辅助
  CUDA stream，与主流 dispatch 并行；dispatch（包括 PCP gather）后等待
  shared，再执行后置 gate 和 routed experts。关闭时仍在 dispatch 前于主流
  串行计算 shared，然后执行 AG，不恢复此前的 shared 调度。
- runner 初始化时直接读取该通用开关，模型无需转传；要求有 shared experts、
  未启用 shared/gate 权重融合，且 backend 不由 kernel 自行 dispatch、quant
  method 不是 monolithic。仅实际进入此 runner 调度的模型生效；DSv4 默认的
  自定义 DP + EP 路径不受此开关控制。
- 原 `VLLM_ILU_GLM_AG_SHARED_OVERLAP` 已移除，启动脚本需改用新名字。
- shared 输出保持本地，routed 输出完成 combine/reduce-scatter 后再相加。
- 支持 BF16 输入以及 fused norm/quant 提供的 INT8/scales；独立于
  `VLLM_DISABLE_SHARED_EXPERTS_STREAM`。LL dispatch 仍由
  `VLLM_ILU_GLM_LL_SHARED_DISPATCH_OVERLAP` 控制。

MoERunner 的 AG/RS 数据格式使用 `dp_metadata.all_ranks_decode_only` 判断：
该字段为真时通信 BF16，gate 后置；为假或缺少 `dp_metadata` 时在本地 gate，支持预量化输入的
W4A8 NN 方法传 INT8/scales/logits，并在适用时复用 norm 的量化结果。
不支持预量化的方法保留浮点输入；由 kernel 自行 dispatch 的 backend
继续使用原执行协议。PCP 的 gather/reduce-scatter 沿用 runner 的通信归属判断。

**DP=1 / PCP>1 的阶段边界：** 未启用 sequence-parallel MoE 时，生产
`set_forward_context()` 不创建 `dp_metadata`。因此即使当前请求全是 decode，
runner 也采用前置 gate；支持预量化输入且无 custom routing 的 W4A8 NN
走 INT8/scales/logits dispatch。PCP gather/reduce-scatter 仍正常执行，但
不会命中 BF16 dispatch + 后置 gate 的 decode 优化。当前不从本地 attention
metadata 或 token 数推断阶段，也不人为补造 DP metadata。

测试分别覆盖真实 context 缺少 DP metadata 时的分派，以及两卡 PCP 的
INT8 通信、shared 串行/overlap 和 Graph replay；后者不代表 PCP decode
后置 gate 策略已在生产可达。支持该策略需要后续单独设计阶段判断。

该开关只控制 shared 调度，不控制上述阶段策略。收益需在目标负载下验证。

### 3.2 DP 请求分配与 Prefill 同步

#### `VLLM_ILU_DP_ROUND_ROBIN_LB`

- 默认值：`0`。
- 推荐值：Hybrid EP 场景设为 `1`。
- 作用：使用跨 API server 进程共享的 round-robin 计数器，将请求稳定地
  分发到各 DP engine，减少 coordinator 负载统计滞后造成的 rank 失衡。

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
- collective 顺序修复：
  `924613a5`。

同步使用独立 CPU process group。参与一次 Prefill 同步的每个 rank 都必须
继续执行且只执行一个真实或 dummy worker step，否则 EngineCore collective
和 worker collective 会错位并 hang。

#### `VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS`

- 默认值：`8`。
- 负值处理：限制为 `0`。
- 作用：部分 DP rank 有 Prefill、其它 rank 仍 Decode 时，允许等待的最大
  模型 step 数；超过上限后 fail-open，避免 Prefill 饥饿。

当部分 rank ready 但没有 rank Decode 时，实现最多只短暂聚合两个 step，
同时仍受该上限约束。

#### `VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS`

- 默认值：Hybrid LL + round-robin 时为 `5`，其它配置为 `0`。
- 负值处理：限制为 `0`。
- 作用：某个 rank 执行完模型并清空本地 scheduler 后，短暂等待后台线程把
  replacement request 放入输入队列，避免过早进入 dummy batch。

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
  `6752b169`。
- 当前 fused 路径补充提交：
  `c6e9872e`。

当前 W4A8 fused MoE 要求 `format=TN`、`version=2`、`group_size=-1`。
其它格式会在加载期报 `NotImplementedError`，不应静默使用。

该变量在 Python 模块 import 时读取并缓存，不能在服务运行中动态切换。

### 3.4 LL buffer 容量

当 `VLLM_ILU_DISABLE_DEEPEP=1`、LL 使用 `BufferWrapper`/NVSM 路径时，
Hybrid LL buffer 使用 `scheduler_config.max_num_seqs` 估算每个 rank 的
最大 token 容量，而不是使用 `max_num_batched_tokens`。对应提交：
`4f22186d`。

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

### 4.3 确认配置生效

服务启动时检查以下 INFO 日志：

1. `Enabled ILU synchronized DP prefills`：sync-prefill patch 已安装；日志同时
   输出 `max_defer_steps` 和 `max_replacement_wait_ms`。
2. `Initialized Hybrid EP low-latency decode backend`：LL decode backend 和
   通信 handle 已初始化。

使用默认 runner 或 `naive` override 时，不应要求出现 LL 初始化日志。

若配置了上述功能却缺少第一条，检查 `VLLM_ILU_DP_ENABLE_SYNC_PREFILLS=1` 且
`VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS > 0`；若缺少第二条，检查
`VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency` 和 DP + EP 是否启用。

### 4.4 GLM-5.2 的入口配置

以下配置均配合 `--enable-expert-parallel` 和
`--all2all_backend=allgather_reducescatter`，并设置适合设备数的 TP/DP。

```bash
# 默认：Prefill、混合批次、Decode 均走 MoERunner。
unset VLLM_ILU_DECODE_EP_BACKEND

# shared MLP 串行；设为 1 才开启 AG/shared overlap。
export VLLM_ILU_MOE_AG_SHARED_OVERLAP=0
```

临时让所有 DP rank 纯 Decode 时回到 `forward_dp_ep_naive()`：

```bash
export VLLM_ILU_DECODE_EP_BACKEND=naive
```

若要纯 Decode 使用 LL，则改为：

```bash
export VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency
```

两种 override 下，GLM 的 Prefill/混合批次仍走 MoERunner。没有额外的
`VLLM_ILU_GLM_MOE_USE_RUNNER` 开关。AG/RS runner 的阶段策略不由
`VLLM_ILU_MOE_AG_SHARED_OVERLAP` 控制；该变量只改变 shared 调度。
运行 AG/RS runner 后可检查 `MoERunner phase=...` 日志中的
`gate_after_dispatch` 和 `shared_overlap`。日志只说明该 forward 路径曾执行，
不等同于逐步 CUDA Graph replay 的 trace。

## 5. Backend 选择规则

先区分模型入口，再判断自定义路径内部的 backend。

| 模型 | 进入 `forward_dp_ep()` 的条件 | 其它情况 |
| --- | --- | --- |
| GLM-5.2 | 设置了合法 override，且所有 DP rank 都是纯 Decode | 使用 `self.experts` / MoERunner，按配置 backend 执行 |
| DeepSeek-V4 | `_should_use_forward_dp_ep()` 为真 | 保留模型原有 `FusedMoE` forward |

`_should_use_forward_dp_ep()` 要求同时满足：

1. DP size 大于 1，且已启用 EP。
2. 配置 backend 是 `naive`、`allgather_reducescatter`、`ag-rs`、`ag_rs`、
   `deepep_low_latency` 或 `deepep_high_throughput`。
3. quant method 的 `is_monolithic=False`。
4. backend 为 DeepEP LL/HT，或者 quant method 的
   `supports_internal_mk=False`。

这些是自定义 DP + EP 路径及 override 的准入条件，不是 MoERunner 的
通用准入条件。`naive` override 另要求配置 AG/RS backend，详见第 3.1 节。

已经进入 `forward_dp_ep()` 后，按以下优先级执行：

1. 所有 DP rank 纯 Decode 且设置 override：`naive` 调用
   `forward_dp_ep_naive()`；`deepep_low_latency` 调用 LL。
2. 未命中 override 且配置 backend 为 LL：调用 LL。
3. 未命中 override 且配置 backend 为 HT：调用 HT。
4. 未命中 override 且配置 backend 为 naive/AG-RS：调用
   `forward_dp_ep_naive()`。
5. 其它情况保留 `forward_dp_ep_modular()` 兜底。

第 4 条不能用于推断 GLM 的默认入口：GLM 未命中 override 时根本不会进入
该函数，而是走 MoERunner。DSv4 仍遵循其原有入口判断。

AG/RS MoERunner 的纯 Decode 后置 gate 要求 runner 持有 gate 且未启用
shared/gate 权重融合；Prefill/混合批次的 INT8 dispatch 要求 quant method
支持预量化输入且没有 custom routing function。kernel 自行处理 dispatch 或
monolithic 方法不走这套阶段策略。PCP 通信继续由 runner 的
`_needs_pcp_dispatch_combine()` 判断归属，不能将自定义路径的 DP>1 条件
套用到 PCP runner。

### 5.1 当前限制

1. LL 路径不支持 `pcp_size > 1`，会抛出 `NotImplementedError`。
2. 当前 NVSM LL modular kernel 支持的 hidden size 为 2048、2560、3072、
   4096、5120、6144、7168 和 8192。
3. `VLLM_ILU_DISABLE_DEEPEP=1` 仍依赖 Iluvatar NVSM LL 能力，不表示关闭
   EP 通信。
4. sync-prefill 每个活跃 EngineCore step 会增加一次小型 CPU DP
   all-reduce；多节点和非 round-robin 部署必须单独压测。

### 5.2 本轮改动的验证范围

- 单元测试覆盖通用 runner 配置读取、shared 串行/overlap 调度、DP 阶段策略，
  以及真实 `set_forward_context()` 在 DP=1/PCP>1 时不产生 DP metadata。
- 使用真实 `ParallelConfig` 验证 CLI `naive` 的归一化及 naive override；
  GLM 与 DSv4 测试从各自真实模型 `forward()` 入口验证派发和输入传递。
  这些入口测试使用下游 test doubles，不代表完整模型初始化和 kernel 验证。
- 两卡测试覆盖原生 NN MoE、DP/PCP 通信、空 DP shard、预量化输入、shared
  串行/overlap 与改输入后的 CUDA Graph replay。PCP 无 metadata 的覆盖范围
  见第 3.1 节。
- 尚未完成 GLM 配置 HT/LL 后的完整 runner 初始化与数值 forward、DSv4
  整模精度/性能回归。LL 协议 mock 测试不能证明 LL kernel 数值正确。
  不据这些测试宣称所有模型和 backend 均无回归。

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

## 8. 变更记录

变更记录：

1. `6752b169`：
   初始支持 DP + EP、Hybrid EP backend 和 round-robin LB。
2. `3f2a6e81`：
   补充 DP + EP 测试、边界检查和说明。
3. `4f22186d`：
   Hybrid LL buffer 容量改用 `max_num_seqs`。
4. `76d338dd`：
   优化 DP Prefill 调度，增加同步 admission 和 replacement wait。
5. `924613a5`：
   修复 Prefill 同步中的 collective 顺序，保证每轮同步对应一个 worker
   step。
6. `c6e9872e`：
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
9. [`ilu_moe_runner.py`](../../vllm_iluvatar/model_executor/layers/fused_moe/runner/ilu_moe_runner.py)：
   AG/RS 阶段策略、预量化输入复用、shared 调度及 PCP dispatch/combine。

### 9.2 配置步骤速览

以下为本文 DSv4 Hybrid LL 示例；GLM 默认 runner 或 naive override 配置见第 4.4 节。

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
