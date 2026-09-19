# 在 vllm-iluvatar 接入天数优化模型

> **How-to 指南** · 通过 OOT registry 路由到 Iluvatar 实现 · 模型部分完全重写、尽量不走社区接口 · 以 GLM-5.2（`glm_moe_dsa`）为落地参考

本文说明如何在 `vllm-iluvatar` 插件里接入一个**天数（Iluvatar）专有优化模型**。与"复用社区模型 + 逐类覆盖"不同，天数优化模型的思路是：**通过 `ModelRegistry.register_model` 把目标架构路由到插件内的 Iluvatar 实现，并把模型算法链完全重写在插件内**，运行时尽量走 `vllm-iluvatar` 自己的接口（ilu attention backend、`ixf_ops` marlin / W4A8 kernel、plugin-local opaque op），而非社区的通用封装。改动全部落在插件仓库内，社区 vLLM 源码保持零改动。

| | |
| --- | --- |
| **适用对象** | 需要做天数专有优化、且优化深入到 attention / 量化 / 投机解码算法链的模型 |
| **改动范围** | 仅 `vllm-iluvatar` 插件仓库，不改上游 vLLM |
| **参考实现** | GLM-5.2 `glm_moe_dsa` 包中的 `IluGlmMoeDsaForCausalLM`（MLA + DSA lightning indexer + fine-grained MoE，W4A8 混合量化，支持 MTP），布局对齐 `deepseek_v4` |

---

## 目录

- [1. 背景与原则](#1-背景与原则)
- [2. 命名与目录约定](#2-命名与目录约定)
- [3. 端到端步骤](#3-端到端步骤)
  - [3.1 权重转换](#31-权重转换bf16fp8--w4a8-混合量化)
  - [3.2 模型适配](#32-模型适配)
  - [3.3 attention 抽离](#33-attention-抽离)
  - [3.4 MoE 与混合量化](#34-moe-与混合量化)
  - [3.5 三种运行模式：eager / cuda-graph / MTP](#35-三种运行模式eager--cuda-graph--mtp)
- [4. 测试](#4-测试)
- [5. 附录](#5-附录)

---

## 1. 背景与原则

天数优化模型的接入遵循三条硬性原则：

1. **通过 registry 路由到插件实现，不改社区源码**。优先使用 checkpoint / 社区已经暴露的目标 arch（如 `DeepseekV4ForCausalLM`、`DeepseekV32ForCausalLM`）作为 registry key，指向插件内的**同等社区模型位置**实现类；如果是社区没有的新增架构，也可以注册新的 `IluXXXForCausalLM`。社区 `deepseek_v2.py` 等源文件保持逐字节不变；checkpoint 的 `config.json::architectures` 指向 registry key，加载时即解析到插件实现。

2. **模型算法链完全重写，尽量走插件接口**。运行路径走 `vllm-iluvatar` 内部接口，**不走社区**的 `MultiHeadLatentAttentionWrapper` / `MLAAttention` / `SparseMLAAttentionImpl` / `sparse_attn_indexer`。attention 的 rope、吸收 up-proj、sparse kernel、indexer top-k 全部由插件自行实现或经 `ixf_ops` / ilu backend 完成。只有**基础设施**（`*Linear` / `RMSNorm` / `get_rope` / ilu FlashMLA backend 的 metadata 与 cache）才复用。

3. **通过官方 OOT 扩展点接入**。`vllm-iluvatar` 有三套后期接入机制，各司其职（见 [`vllm_iluvatar/ilu_register/__init__.py`](../../vllm_iluvatar/ilu_register/__init__.py)）：
   - **import-hook**：import 期的 `vllm._custom_ops` 覆盖；
   - **`ilu_patch`**：运行期 monkey-patch（带指纹 / 签名校验）；
   - **`ilu_register`**：官方 OOT 扩展点，即 `ModelRegistry.register_model` 与 `@register_quantization_config`（无指纹，因为它们是替换性的注册项，而非方法拷贝）。

   接入一个新模型主要用到 `ilu_register`（模型 + 量化配置），外加少量 `ilu_patch.general_patches`（仅投机解码需要）。多文件模型包需要显式加入 `_OOT_MODEL_PACKAGES`，并在包内暴露 `register_model()`，这样 `apply_oot_registrations()` 才能自动发现并注册它。

---

## 2. 命名与目录约定

- **arch 命名**：registry key 使用"要让 checkpoint 命中的名字"。已有社区 arch 时优先复用社区名（如 `DeepseekV4ForCausalLM` → `IluDeepseekV4ForCausalLM`）；新增架构时再使用 `IluXXXForCausalLM`。MTP draft 同理，优先使用稳定的模型语义名（如 `DeepseekV32MTPModel`、`DeepSeekV4MTPModel`），不强制带 `Ilu`。
- **包命名**：`vllm_iluvatar/models/` 下的包名按模型/架构语义命名，尽量对齐 HF `model_type` 或社区包名，例如 `deepseek_v32`、`deepseek_v4`、`glm_moe_dsa`；不要用 `ilu_` 前缀作为包名。
- **包布局**：每个天数模型是 `vllm_iluvatar/models/<model_type>/` 下的一个**多文件包**，布局对齐 `deepseek_v4`：

  ```text
  vllm_iluvatar/models/<model_type>/
    __init__.py     # 包说明；模块由 ilu_register 自动 import
    model.py        # 模型图 + 权重加载 + register_model() 钩子
    attention.py    # 自包含 attention 算法链（含 plugin-local opaque op）
    mtp.py          # （可选）MTP draft 模型
  ```

- **登记到 `_OOT_MODEL_PACKAGES`**：`ilu_register` 的自动发现不递归子包，多文件模型包须显式列进 [`vllm_iluvatar/ilu_register/__init__.py`](../../vllm_iluvatar/ilu_register/__init__.py) 的 `_OOT_MODEL_PACKAGES`：

  ```python
  _OOT_MODEL_PACKAGES = [
      "vllm_iluvatar.model_executor.models",
      "vllm_iluvatar.models.deepseek_v32",
      "vllm_iluvatar.models.deepseek_v4",
      "vllm_iluvatar.models.glm_moe_dsa",
  ]
  ```

  `apply_oot_registrations()` 会 import 包内每个模块并调用其 `register_model()` 钩子。

---

## 3. 端到端步骤

整体流程（前 4 步必做，第 5 步的 MTP 按需）：

```text
3.1 权重转换  →  3.2 模型适配  →  3.3 attention 抽离  →  3.4 MoE/量化
                                                          └─▶ 3.5 三种模式接入（eager / cuda-graph / MTP）
```

### 3.1 权重转换（BF16/FP8 → W4A8 混合量化）

天数优化通常伴随一次**离线权重转换**：把原始 checkpoint（BF16 或 FP8-block）重量化成天数专有的 W4A8 混合方案，并按需要改写 `config.json` 的 `architectures`，使其命中插件注册的 registry key。

转换器产出三部分：

1. **量化后的 safetensors**：按下表做混合量化。
2. **重建的 `model.safetensors.index.json`**：按新 shard 的实际 dtype/shape 重新计算 `total_size` 与 `weight_map`。
3. **改写的 `config.json`**：
   - 修改或增加 `quantization_config`，其 `quant_method="compressed-tensors"`（复用统一的混合量化 schema，见 [3.4](#34-moe-与混合量化)）。
   - 保持其它部分不变。

混合量化布局（以 GLM-5.2 为例）：

| 源权重（HF 命名） | 目标量化 |
| --- | --- |
| routed experts `mlp.experts.N.{gate,up,down}_proj` | INT4 W4A8（打包进 INT8 + fp32 channel scale） |
| shared experts `mlp.shared_experts.*` | INT8 W8A8（per-channel scale） |
| dense MLP（`first_k_dense_replace` 层）`mlp.{gate,up,down}_proj` | INT8 W8A8 |
| MLA 线性 `self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,o_proj}` | INT8 W8A8 |
| indexer `self_attn.indexer.wq_b` | INT8 W8A8 |
| MTP `eh_proj` | INT8 W8A8 |
| MLA `self_attn.kv_b_proj` | INT8（`mla_bmm_w8a16` 段）——加载期拆 `W_UK`/`W_UV` 走 marlin |
| indexer `self_attn.indexer.{wk,weights_proj}` | 反量化到 BF16（保持 BF16，无 scale） |
| norm / embed / head / `mlp.gate.*` / `*_layernorm` | 原样保留 |

关键约束（**必须遵守，否则加载会静默出错**）：

- **`targets` 正则须写 vLLM 运行期（post-mapper）前缀名**，而非 checkpoint 的源命名。例如 dense MLP 的 `gate_proj`/`up_proj` 在运行期已合并为 `gate_up_proj`（`MergedColumnParallelLinear`），配置里应匹配 `mlp.(gate_up_proj|down_proj)`。

可选做法（非强制，视模型与优化取舍而定）：

- **`q_a_proj` / `kv_a_proj_with_mqa` 是否离线合并**：GLM-5.2 的做法是 checkpoint 保持两者分离、由模型**加载期**经 `fused_qkv_a_proj`（`MergedColumnParallelLinear` + `stacked_params_mapping`）自动合并，好处是 checkpoint 无需重转即可兼容融合布局、运行时也没有额外的 merge monkeypatch。这并非强约束——也可在离线阶段直接合并，或干脆不融合。
- **`kv_b_proj` 是否走 W8A16 grouped marlin**：GLM-5.2 把它在磁盘上存成普通 INT8（per-row scale，与其它 W8A8 张量同格式），仅在**运行期消费方式**上不同（拆成两个 grouped marlin GEMM，无激活量化），因此单独走 `mla_bmm_w8a16` 段而非 `dense_linears`。这是一种性能/显存优化选择，非强制——也可让 `kv_b_proj` 保持 BF16 吸收 bmm 或走普通 W8A8。
- **离线转换的分片并行**：转换器可选支持按 safetensors shard 分片并行（多进程 round-robin），最后统一 `--finalize` 重建 index + 改写 config；单进程与多进程产出应完全一致（量化逻辑不变，只是 shard 分给不同 worker）。这只是提速手段，单进程串行同样可行。转换需在装好 `ixformer` 的 GPU 机上执行。

### 3.2 模型适配

以 [`glm_moe_dsa/model.py`](../../vllm_iluvatar/models/glm_moe_dsa/model.py) 为模板搭建模型图与权重加载：

- **类结构**：`IluXXXForCausalLM(nn.Module, SupportsPP)` → `IluXXXModel` → `IluXXXDecoderLayer` → `MoE` / `MLP`。顶层持有 `lm_head` / `logits_processor`，`model_cls` 指向内层 `Model`。
- **`register_model()` 钩子**：在文件末尾注册 causal-LM 与 MTP 两个 arch，用**字符串路径惰性引用**类，把 attention / mtp 的重依赖延迟到模型真正加载时：

  ```python
  def register_model() -> None:
      from vllm import ModelRegistry
      ModelRegistry.register_model(
          "<TargetArchForCausalLM>",
          "vllm_iluvatar.models.<model_type>.model:IluXXXForCausalLM",
      )
      ModelRegistry.register_model(
          "<TargetMTPModel>",
          "vllm_iluvatar.models.<model_type>.mtp:IluXXXMTP",
      )
  ```

- **官方 FP8 权重（可选第二套）**：W4A8 转换权重走 `IluXXXForCausalLM`；官方 FP8 checkpoint 仍声明社区 arch 名时，另写薄子类覆盖（GLM-5.2：`fp8_model.py` / `fp8_attention.py` / `fp8_mtp.py`），用类属性钩子换 decoder / attention / MTP 层，避免复制整份模型图。社区 arch 名覆盖 vLLM 自带实现；MTP draft 若社区路由到与其它模型共享的 generic draft arch，则不要覆盖那个名字，改在 `_ILU_MTP_ARCH_MAP` 里重指向插件 draft。

- **融合参数映射**：`packed_modules_mapping` 与 `load_weights` 中的 `stacked_params_mapping` 定义加载期融合——`gate_up_proj`（gate+up）、`fused_qkv_a_proj`（q_a + kv_a）、`wk_weights_proj`（indexer wk + weights_proj）；expert 权重走 `fused_moe_make_expert_params_mapping`。
- **严格 reverse 校验**：量化模式下，vLLM 的 `DefaultModelLoader` 只在 `quantization is None` 时才检查"是否所有参数都被填充"。天数模型在 `--quantization compressed-tensors` 下须自行确定所有层都有正确的后端方法， 避免缺 scale / 命名不匹配时静默回退到随机初始化。
- **MoE 要点**：sigmoid 打分 + `noaux_tc` 校正 bias（`e_score_correction_bias`）+ grouped top-k；`routed_scaling_factor` 在**输出侧**应用（`apply_routed_scale_to_output`，避免 router 侧重复缩放）。

### 3.3 attention 抽离

这是天数优化的核心。以 [`glm_moe_dsa/attention.py`](../../vllm_iluvatar/models/glm_moe_dsa/attention.py) 为模板，把整条 attention 算法链**自包含**在插件内，分三层（对齐 `deepseek_v4`）：

- **外层 `IluXXXMLAAttention`**（`nn.Module`）：持有投影（`fused_qkv_a_proj` / `q_b_proj` / `kv_b_proj` / `o_proj`）、norms、DSA indexer，等价于社区 `MultiHeadLatentAttentionWrapper.forward_opt` 的 pre/post 处理，但**不引入社区 wrapper**。
- **内层 `IluXXXSparseMLA`**（`nn.Module, AttentionLayerBase`）：负责 `get_attn_backend` / `get_kv_cache_spec`，驱动 rope + 吸收 k-up + `concat_and_cache_mla` + FlashMLA sparse kernel + v-up。吸收的 up-proj 是 grouped W8A16 marlin：k-up / v-up 直接读 `kv_b_proj.W_UK` / `W_UV`（加载期由量化 method 拆好并 repack）并调 `ixf_ops.marlin_w8a16`。它注册为 `AttentionLayerBase`，由 runner 为其分配 paged KV cache 并按 `prefix` 建 metadata。
- **`IluXXXIndexer` / `IndexerCache`**：DSA lightning indexer；`IndexerCache` 同样是 `AttentionLayerBase`（单向量 paged MLA cache，backend 为 ilu indexer backend）。

**数据依赖控制流封装成 plugin-local opaque op**（保证 cuda-graph / torch.compile 安全的关键）：

indexer 的 top-k 计算涉及 top-k、按 chunk 的循环、`attn_metadata` 字典访问等数据依赖控制流，直接写进 `forward` 会破坏 cuda-graph / torch.compile。做法是用 `direct_register_custom_op` 把它注册成一个**插件本地不透明算子**，并配 `fake_impl` 与 `mutates_args`：

```python
direct_register_custom_op(
    op_name="ilu_xxx_sparse_attn_indexer",
    op_func=_ilu_xxx_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=_ilu_xxx_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)
```

约束：

- **使用独立算子名**（如 `ilu_xxx_sparse_attn_indexer`），使其能与 fork 中已有的 `vllm::ilu_sparse_attn_indexer` 共存，**不复用社区 `sparse_attn_indexer`**。
- op 内部须对 cuda-graph padding 行做保护：`slot_mapping` 长度按（padded 的）grid 走，多余的 `k` 截断；padding 行 `slot_idx < 0` 在 kernel 内跳过；空 KV（`max_context_len == 0`）时 prefill `continue`、decode 写 `-1` 直接返回。
- top-k 结果写进一个**共享的 `topk_indices_buffer`**：`full` 层写入、`shared`（skip_topk）层复用同一 buffer。该 buffer 每个 PP rank 各持一份、不跨 rank 传递，这会影响 PP 层划分（见 [3.5](#35-三种运行模式eager--cuda-graph--mtp)）。

### 3.4 MoE 与混合量化

混合量化配置以 [`compressed_tensors.py`](../../vllm_iluvatar/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py) 为模板。它是统一的量化 config，新模型可直接复用.

- **注册与激活**：`IluQuantConfig` 标注 `@register_quantization_config("compressed-tensors")`（由 `ilu_register` 在 import 期触发），替换官方同名配置类。checkpoint 的 `quantization_config.quant_method` 为 `compressed-tensors` 时自动走插件后端，社区的 quantization 注册表与 `config/model.py` 保持零改动。详细内容，请见[`6-how-to-use-and-provide-iluvatar-extend-compress-tensors.md`](../guides/6-how-to-use-and-provide-iluvatar-extend-compress-tensors.md)

### 3.5 三种运行模式：eager / cuda-graph / MTP

三种运行模式由浅入深，通常按此顺序逐个打通。

#### （1）eager —— 基线

用 `--enforce-eager` 关闭 cuda-graph，先验证权重加载、attention 数值、MoE 路由正确。这是精度对齐与调试的基准。

#### （2）cuda-graph —— decode 整图

若模型没有 `@support_torch_compile`（天数模型通常不加），则无法走需要 torch.compile 的 piecewise 路径，只能捕获 decode 整图。用 `--compilation-config` 显式指定：

```json
{"cudagraph_mode": "FULL_DECODE_ONLY", "mode": 0, "custom_ops": ["all"]}
```

- `cudagraph_mode: FULL_DECODE_ONLY`：只对 decode batch 捕获整图，prefill 走 eager。
- `mode: 0`（`CompilationMode.NONE`）：完全不走 torch.compile，但仍可捕获 decode cuda-graph，与"无 `@support_torch_compile`"匹配。
- `custom_ops: ["all"]`：启用全部自定义算子的 `forward_cuda`（含 ilu sparse indexer），避免被拆成 native 小算子。

前提：[3.3](#33-attention-抽离) 中 opaque op 的 padding 行保护、以及底层 kernel（如 `indexer_k_cache`）对 `slot_idx < 0` 的跳过必须已就位，否则 cuda-graph 下 padding 行会越界 / 读脏。

#### （3）MTP —— 多 token 投机解码（可叠加 cuda-graph）

新增 [`glm_moe_dsa/mtp.py`](../../vllm_iluvatar/models/glm_moe_dsa/mtp.py)，实现 `IluXXXMTP`（标准 DeepSeek-V3.2 MTP 布局：`enorm` / `hnorm` + 融合 `eh_proj` + `shared_head` + 一个 `mtp_block` 解码层）。要点：

- **`mtp_block` 复用 target 的 ilu 解码层**（`IluXXXDecoderLayer`），使 draft 与 target 共享同一套 ilu attention / W8A16 `kv_b_proj` / DSA kernel，不引入社区 `DeepseekV2DecoderLayer`。
- `eh_proj` 用 `ReplicatedLinear` 并绑 `quant_config`（离线转换把它量化成 INT8 W8A8）。
- **`index_share_for_mtp_iteration`**：`set_skip_topk` / `compact_topk_indices` 直接操作 `mtp_block.self_attn.{skip_topk,topk_indices_buffer}`——draft step 0 计算自己的 top-k，step 1+ 复用 step 0 写入的 buffer。
- **权重 tie**：GLM 把 MTP 的 `shared_head.head` tie 到主模型 `lm_head`、`embed_tokens` tie 到主模型 embedding（checkpoint 只带 `lm_head.weight` / `model.embed_tokens.weight`），在 `load_weights` 中显式 tie。

MTP 还需两个 `ilu_patch.general_patches`（各暴露 `register()`，由 `apply_general_patches` 在 `register_general` 期自动发现）：

- [`patch_speculative_config.py`](../../vllm_iluvatar/ilu_patch/general_patches/patch_speculative_config.py)：包装社区 `SpeculativeConfig.hf_config_override`，在社区路由跑完后，把 target arch 的 MTP draft arch 重定向到插件专用 draft arch（实现见 [`config/speculative.py`](../../vllm_iluvatar/config/speculative.py) 的 `_ILU_MTP_ARCH_MAP`，例如 `DeepseekV32MTPModel`、`DeepSeekV4MTPModel`、`IluGlmMoeDsaMTPModel`）。
- [`patch_llm_base_proposer.py`](../../vllm_iluvatar/ilu_patch/general_patches/patch_llm_base_proposer.py)：包装 `SpecDecodeBaseProposer.model_returns_tuple`，让 proposer 识别插件 draft arch 的 `(logit_hidden, recycle_hidden)` tuple 返回契约（实现见 [`v1/spec_decode/llm_base_proposer.py`](../../vllm_iluvatar/v1/spec_decode/llm_base_proposer.py)）。

启动约束（以通用形式给出，具体拓扑按部署环境调整）：

- **PP>1 时 draft 须单 stage**：MTP draft 只有一层、未做流水线切分，须在投机配置里设 `draft_pipeline_parallel_size: 1`（TP 仍与 target 一致），否则 draft PP 会继承 target PP，在 config 校验阶段报 draft 模型不支持 PP。
- **PP 每个 stage 的首层须为 `full` indexer 层**：`shared` 层复用本 rank 上一个 `full` 层写的 top-k buffer，buffer 不跨 rank 传。因此须按 `full` 层位置对齐 PP 边界（可用层划分环境变量控制），否则某 rank 首层落在 `shared` 会读到脏值、输出错误。
- 权重几乎全在 MoE 层，PP 负载均衡应按 **MoE 层数**而非总层数分配；带 MTP 的最后一个 stage 宜少放一层目标 MoE，为 MTP 与 KV cache 腾空间。

---

## 4. 测试

对齐现有测试布局：

- **投机 patch 测试**：`tests/config/test_speculative_patch.py`（`hf_config_override` 重定向）、`tests/v1/spec_decode/test_llm_base_proposer_patch.py`（`model_returns_tuple` tuple 契约）。
- **indexer 回归**：`tests/v1/spec_decode/test_ragged_decode_indexer.py`（ragged / 非均匀 query 长度的 decode top-k）。
- **加载与量化**：验证严格 reverse 校验能捕获缺参，以及 `kv_b_proj` 拆分后的 `W_UK/W_UV` 形状与 scale 正确。

三种运行模式（eager / cuda-graph / MTP）各跑一次端到端冒烟 + 小样本精度抽查，再做全量精度评测。

---

## 5. 附录

### 关键文件清单

| 作用 | 文件 |
| --- | --- |
| OOT 注册入口（登记模型包） | [`vllm_iluvatar/ilu_register/__init__.py`](../../vllm_iluvatar/ilu_register/__init__.py) |
| 模型图 + 权重加载 + `register_model()` | [`vllm_iluvatar/models/glm_moe_dsa/model.py`](../../vllm_iluvatar/models/glm_moe_dsa/model.py) |
| 自包含 attention 链 + plugin-local opaque op | [`vllm_iluvatar/models/glm_moe_dsa/attention.py`](../../vllm_iluvatar/models/glm_moe_dsa/attention.py) |
| MTP draft 模型 | [`vllm_iluvatar/models/glm_moe_dsa/mtp.py`](../../vllm_iluvatar/models/glm_moe_dsa/mtp.py) |
| 官方 FP8 权重（社区 arch 覆盖） | [`fp8_model.py`](../../vllm_iluvatar/models/glm_moe_dsa/fp8_model.py) · [`fp8_attention.py`](../../vllm_iluvatar/models/glm_moe_dsa/fp8_attention.py) · [`fp8_mtp.py`](../../vllm_iluvatar/models/glm_moe_dsa/fp8_mtp.py) |
| 混合量化 config（W4A8/W8A8/W8A16 分发） | [`vllm_iluvatar/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`](../../vllm_iluvatar/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py) |
| MTP draft-arch 重定向实现 | [`vllm_iluvatar/config/speculative.py`](../../vllm_iluvatar/config/speculative.py) |
| MTP tuple 返回契约实现 | [`vllm_iluvatar/v1/spec_decode/llm_base_proposer.py`](../../vllm_iluvatar/v1/spec_decode/llm_base_proposer.py) |
| MTP 两个 patch 的 register-time 接线 | [`patch_speculative_config.py`](../../vllm_iluvatar/ilu_patch/general_patches/patch_speculative_config.py) · [`patch_llm_base_proposer.py`](../../vllm_iluvatar/ilu_patch/general_patches/patch_llm_base_proposer.py) |

### 步骤速览

1. **权重转换**：混合量化产出 safetensors + 重建 index + 改写 `config.json`。
2. **模型适配**：按 `model.py` 模板搭 `IluXXXForCausalLM`，实现融合参数映射与严格 reverse 校验，`register_model()` 注册 causal-LM + MTP，并把包加入 `_OOT_MODEL_PACKAGES`。
3. **attention 抽离**：三层自包含 attention，数据依赖控制流用 `direct_register_custom_op` 封成 plugin-local opaque op（独立算子名、padding 行保护）。
4. **MoE / 量化**：复用 `compressed-tensors`（`IluQuantConfig`），`kv_b_proj` 走 `IluGlmKVBProjW8A16Method` 拆 `W_UK/W_UV`，grouped-BMM method 的 `apply` 抛 `NotImplementedError`。
5. **三种模式**：先 eager 打基线；再用 `--compilation-config` 开 `FULL_DECODE_ONLY` cuda-graph；最后加 `mtp.py` + 两个 general patch 开 MTP（注意 `draft_pipeline_parallel_size: 1` 与 PP 首层须为 full 层）。

本文的所有代码引用均来自 GLM-5.2（`IluGlmMoeDsa`）这一落地实现，可作为对照。
