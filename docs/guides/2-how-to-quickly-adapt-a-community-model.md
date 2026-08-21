# 在 vllm-iluvatar 快速适配社区模型

> **How-to 指南** · 复用社区 arch · 子类化 + 逐类最小覆盖 · 只替换少数天数算子 · 以 hy_v3（`HYV3ForCausalLM`）为落地参考、qwen3（`Qwen3ForCausalLM`）为对照

本文说明如何在 `vllm-iluvatar` 插件里**快速适配一个社区已支持的模型**。与"新注册 `IluXXX`、模型算法链完全重写"（见 [3-how-to-add-an-iluvatar-optimized-model.md](3-how-to-add-an-iluvatar-optimized-model.md)）不同，快速适配的思路是：**复用社区模型，用官方 OOT 扩展点以社区 arch 名重新注册一个子类，只覆盖真正含天数（Iluvatar）差异的少数几处**（通常是一处融合算子），其余（`Attention` / `FusedMoE` / `get_rope` / 量化加载路径）全部原样继承社区。改动全部落在插件仓库内，社区 vLLM 源码保持零改动。

| | |
| --- | --- |
| **适用对象** | 社区已支持、天数差异很小的通用模型（只需替换个别算子，如 q/k norm 融合），无需重写 attention / 量化算法链 |
| **改动范围** | 仅 `vllm-iluvatar` 插件仓库，不改上游 vLLM |
| **参考实现** | hy_v3 `HYV3ForCausalLM`（HunYuan-V3，no-MTP，唯一 delta 是融合 q/k RMSNorm，支持 W4A8 加载）· qwen3 `Qwen3ForCausalLM`（dense，同样只融合 q/k norm） |

---

## 目录

- [1. 背景与原则](#1-背景与原则)
- [2. 何时用本范式 vs 天数优化范式](#2-何时用本范式-vs-天数优化范式)
- [3. 命名与目录约定](#3-命名与目录约定)
- [4. 端到端步骤](#4-端到端步骤)
  - [4.1 定位天数 delta](#41-定位天数-delta)
  - [4.2 子类化社区模型](#42-子类化社区模型)
  - [4.3 用 **class** 交换重写 forward + 安装天数 op](#43-用-__class__-交换重写-forward--安装天数-op)
  - [4.4 隔离可选精度取舍](#44-隔离可选精度取舍)
  - [4.5 register_model 覆盖社区 arch](#45-register_model-覆盖社区-arch)
  - [4.6 量化加载（W4A8）](#46-量化加载w4a8)
- [5. 测试](#5-测试)
- [6. 附录](#6-附录)

---

## 1. 背景与原则

快速适配遵循三条原则：

1. **子类化，不整文件拷贝**。继承社区模型类（如 `vllm.model_executor.models.hy_v3.HYV3ForCausalLM`），**只覆盖含天数 delta 的部分**，其余全部继承。这样社区后续 bugfix 与 W4A8 加载路径能自动继承，避免逐字节拷贝整份源码带来的维护负担与漂移（这也是 code review 的硬要求：一处算子融合不应以整文件拷贝承载）。

2. **只替换真正的天数差异**。绝大多数通用模型的天数差异极小——例如 hy_v3 / qwen3 相对社区**唯一**的功能改动就是把 attention 里两次独立的 `q_norm`/`k_norm` 融合成一次 `RMSNormQK`（ixformer `rms_norm_qk`）。运行路径其余部分（社区 `Attention` / `FusedMoE` / `get_rope` / 量化 method）全部复用，不重写。

3. **通过官方 OOT 扩展点接入**。用 `ilu_register` 的 `ModelRegistry.register_model` 以**社区 arch 名**（不是 `IluXXX`）重新注册指向插件子类，社区源文件与 checkpoint 的 `config.json::architectures` 都保持不变，加载时该 arch 即解析到插件子类。单文件模型放在 `vllm_iluvatar/model_executor/models/` 包下、暴露一个 `register_model()` 钩子即可被 [`apply_oot_registrations`](../../vllm_iluvatar/ilu_register/__init__.py) 自动发现，**无需改动 `ilu_register/__init__.py` 或 `_OOT_MODEL_PACKAGES`**（后者只用于登记多文件的天数优化模型子包）。注册完成后，`_validate_oot_overrides` 会对插件子类跑一次 prefix-compatible 的签名校验（覆盖的方法签名须与社区兼容）。

---

## 2. 何时用本范式 vs 天数优化范式

| 维度 | 快速适配社区模型（本文） | 天数优化模型（[3-...](3-how-to-add-an-iluvatar-optimized-model.md)） |
| --- | --- | --- |
| arch 命名 | 沿用社区名覆盖（`HYV3ForCausalLM`） | 保持使用 `XXXForCausalLM` |
| 与社区关系 | 子类化，继承为主 + 逐类最小覆盖 | 模型算法链完全重写，尽量不走社区接口 |
| 天数差异规模 | 少数算子（如一处 norm 融合） | attention / 量化 / 投机解码全链路优化 |
| 文件布局 | 单文件 `models/<arch>.py` | 多文件包 `models/ilu_xxx/` |
| 典型工作量 | 约几十~一百多行 + 测试 | 数百~上千行 + patch + 测试 |
| 参考实现 | hy_v3、qwen3 | GLM-5.2 `IluGlmMoeDsa`、deepseek_v4 |

判断准则：**社区已能正确跑该模型、天数只想换掉个别算子** → 本范式。**需要深入改写 attention / 量化 / MTP 算法链** → 天数优化范式。

---

## 3. 命名与目录约定

- **arch 命名**：沿用**社区原 arch 名**做覆盖（如 `HYV3ForCausalLM` / `Qwen3ForCausalLM`），而非新起 `IluXXX`——目的是"替换社区实现"而非"新增架构"。
- **文件布局**：单文件 `vllm_iluvatar/model_executor/models/<arch>.py`（如 `hy_v3.py` / `qwen3.py`），无需建子包。
- **许可头 + 溯源 docstring**：保留 Apache 许可头，并写溯源 docstring，字段对齐 `qwen3.py`：

  ```text
  Corresponds to: vllm/model_executor/models/<arch>.py
  Target   : 社区目标类与 arch 名
  Baseline : 对齐的社区基线 commit（如 dcfebf93f4e）
  Why      : 相对社区的天数 delta 是什么
  How      : 如何接入（register_model 指向子类；__init__ 后交换 self_attn.__class__ 等）
  Scope    : 适用范围
  Remove   : 何时可删除本覆盖（如社区原生融合该算子后）
  ```

- **后处理函数命名规范（按功能，以结尾词区分）**：在 `__init__` 里 `super().__init__()` 之后对社区模型做原地改写的辅助函数，统一命名为 `<arch>_<组件/功能>_<结尾词>`，结尾词按功能二选一：
  - **`...fusion`**——用天数融合算子替换社区多步计算（交换类 + 安装 op），如 `hy_v3_attention_fusion`（q/k norm 融合）、`qwen3_attention_fusion`。
  - **`...overrides`**——对某组件做的可选 / 可回退的属性或精度覆盖（不改变计算结构），如 `hy_v3_moe_bf16_router_overrides`（bf16 router）。
  - 命名即语义：`fusion` 是核心天数 delta（一般必留）；`overrides` 是可随上游演进删除的取舍项，其 docstring 应注明删除条件。

---

## 4. 端到端步骤

整体流程：

```text
4.1 定位 delta  →  4.2 子类化模型  →  4.3 交换 forward + 安装 op
                                       └─▶ 4.4 隔离可选取舍(overrides)  →  4.5 register_model  →  4.6 量化加载(继承，无需改)
```

下文代码以 [`hy_v3.py`](../../vllm_iluvatar/model_executor/models/hy_v3.py) 为准，qwen3 结构相同（无 MoE，故无 4.4）。

### 4.1 定位天数 delta

先把 fork 的社区源文件与其社区基线做**逐字节对比**，找出真正的功能差异，避免把无关改动一并塞进覆盖。以 hy_v3 为例，相对社区基线 `dcfebf93f4e` 的唯一有效 delta 是：attention fallback 路径里两次独立 `q_norm`/`k_norm` → 一次融合 `rms_norm_qk`。其余"改动"（如强制 `rope_support=False` 的死代码）不应带入子类。

只有确认了 delta，才能决定要覆盖哪个类的哪个方法。

### 4.2 子类化社区模型

以社区类为基类建子类，`__init__` 调 `super().__init__()` 后，仅做原地改写：

```python
from vllm.config import VllmConfig
from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM as _HYV3ForCausalLM


class HYV3ForCausalLM(_HYV3ForCausalLM):
    """HunYuan-V3 causal-LM wired for Iluvatar fused q/k norm attention."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        hy_v3_attention_fusion(self.model)           # fusion：核心天数 delta
        hy_v3_moe_bf16_router_overrides(self.model)  # overrides：可回退取舍
```

`super().__init__()` 先按社区逻辑把整张图（含权重占位、量化 method）建好，之后的后处理只改必要部分。

### 4.3 用 `__class__` 交换重写 forward + 安装天数 op

覆盖某层的 `forward` 而不动其权重的最小手段是**重赋 `__class__`**：把社区 attention 实例的类换成一个"只重写 `forward`"的子类。`__class__` 重赋会保留实例上所有权重 / 状态，仅改变方法解析。

融合 forward 用 mixin 承载（复刻社区 fallback，只把两次 norm 换成一次融合 op），并在交换时**按需安装天数 op**（`RMSNormQK`），op 的构造参数取自社区已建好的 `q_norm`：

```python
import torch
from vllm.model_executor.models.hy_v3 import HYV3Attention as _HYV3Attention
from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK


class _HYV3AttentionFusedMixin:
    """把社区两次 q_norm/k_norm 融合为一次 RMSNormQK。需实例上有 qk_norm 属性。"""

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        out_q, out_k = self.qk_norm(
            q_by_head, k_by_head, self.q_norm.weight.data, self.k_norm.weight.data,
        )
        q = out_q.view(q.shape)
        k = out_k.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        return output


class HYV3Attention(_HYV3AttentionFusedMixin, _HYV3Attention):
    """HYV3Attention with fused q/k RMSNorm."""


def hy_v3_attention_fusion(model) -> None:
    """把每个 decoder 层的 self_attn 换成融合子类，并挂上 qk_norm op。

    __class__ 重赋保留全部权重 / 状态，仅改变 forward 解析。只对启用 q/k norm
    (use_qk_norm) 的层生效；其它层留在社区类上。
    """
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not getattr(attn, "use_qk_norm", False):
            continue
        attn.__class__ = HYV3Attention
        attn.qk_norm = RMSNormQK(
            attn.head_dim, attn.head_dim, eps=attn.q_norm.variance_epsilon,
        )
```

要点：

- mixin 的 `forward` 应**严格复刻社区对应路径**，只替换被融合的部分，其余（rope、`self.attn`、`o_proj`）保持社区调用，保证数值/形状语义一致。
- op 参数（`head_dim`、`eps`）从社区已初始化的模块（`q_norm`）读取，避免与社区 config 解析重复/漂移。
- 用 `use_qk_norm` 之类的守卫跳过不适用的层，防止在没有 `q_norm` 的层上安装 op。

### 4.4 隔离可选精度取舍

把"可选 / 可回退"的覆盖单独放进一个 `...overrides` 函数，与核心 `fusion` 解耦，便于将来一键删除。hy_v3 的 MoE router 在天数上跑 bf16（社区默认 float32），因为 ixformer 的 grouped-topk kernel 当前不吃 float32 gating：

```python
from vllm.model_executor.models.hy_v3 import HYV3MoEFused


def hy_v3_moe_bf16_router_overrides(model) -> None:
    """MoE router（gate logits + expert_bias）跑 bf16 以匹配 ixformer grouped-topk。

    社区默认 float32。待 ixformer 支持 float32 gating_output 后，删除本调用即可
    回到社区 float32。
    """
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, HYV3MoEFused):
            continue
        mlp.gate.out_dtype = torch.bfloat16
        if mlp.expert_bias is not None:
            mlp.expert_bias.data = mlp.expert_bias.data.to(torch.bfloat16)
```

要点：

- **必须在 docstring 注明这是取舍项及其删除条件**（"待 ixformer 支持 float32 后删除"），因为它相对社区是精度下调，语义上不同于核心融合。
- 若该模型没有此类取舍（如 qwen3），则不需要 `overrides` 函数。

### 4.5 register_model 覆盖社区 arch

文件末尾暴露 `register_model()` 钩子，用**社区 arch 名**指向插件子类（用字符串路径惰性引用，避免 import 期重依赖）：

```python
def register_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "HYV3ForCausalLM",
        "vllm_iluvatar.model_executor.models.hy_v3:HYV3ForCausalLM",
    )
```

该模块位于 `vllm_iluvatar/model_executor/models/` 包下，`apply_oot_registrations()` 会自动 import 并调用其 `register_model()`。无需改 `_OOT_MODEL_PACKAGES`。

### 4.6 量化加载（W4A8）

子类**继承社区 `load_weights` 不改动**即可支持量化加载。以 hy_v3 W4A8 为例：量化由 checkpoint `config.json` 的 `quantization_config`（`quant_method` 命中天数量化 config）**自动激活**，配合社区的 `stacked_params_mapping` 与 expert params mapping 完成参数映射；运行期量化 method（如 `ilu_compressed_tensors_moe_w4a8_int8`）的 `w13_*`/`w2_*` 参数名与转换脚本产出的 `weight`/`weight_scale`/`i8_weight_scale`/`i8_weight_zero` 对应。因子类未改这些映射，**不会引入加载回归**。

注意（务必靠测试兜底）：**量化模式下 vLLM 的 `DefaultModelLoader` 会跳过"参数是否全部加载"的校验**。若某 scale 命名对不上，会静默使用未初始化参数而不报错。因此 W4A8 路径必须有加载覆盖测试（见 [5](#5-测试)）。

---

## 5. 测试

新增 `tests/model_executor/models/test_ilu_<arch>.py`（如 [`test_ilu_hy_v3.py`](../../tests/model_executor/models/test_ilu_hy_v3.py)），至少覆盖：

- **OOT 注册（无需 GPU）**：`apply_oot_registrations()` 后，该 arch 解析到插件模块，且插件类是社区类的子类。

  ```python
  def test_oot_registration_points_at_plugin_subclass():
      from vllm import ModelRegistry
      from vllm_iluvatar.ilu_register import apply_oot_registrations

      apply_oot_registrations()
      entry = ModelRegistry.models["HYV3ForCausalLM"]
      assert entry.module_name == "vllm_iluvatar.model_executor.models.hy_v3"
      assert entry.class_name == "HYV3ForCausalLM"

  def test_plugin_class_subclasses_community():
      from vllm.model_executor.models.hy_v3 import HYV3ForCausalLM as _Community
      from vllm_iluvatar.model_executor.models.hy_v3 import HYV3ForCausalLM
      assert issubclass(HYV3ForCausalLM, _Community)  # 继承而非拷贝
  ```

- **天数 op 数值对齐（需 GPU / ixformer）**：融合 op 的 CUDA 实现与纯 torch 参考一致。

  ```python
  from conftest import requires_cuda

  @requires_cuda
  def test_fused_qk_norm_matches_reference(cuda_device, default_vllm_config):
      from vllm_iluvatar.model_executor.layers.layernorm import RMSNormQK
      # 构造 q/k 与 q_norm/k_norm 权重，比较 RMSNormQK.forward_cuda vs forward_native
      op = RMSNormQK(head_dim, head_dim, eps=eps)
      ref_q, ref_k = op.forward_native(q, k, weight_q, weight_k)
      out_q, out_k = op.forward_cuda(q, k, weight_q, weight_k)
      torch.testing.assert_close(out_q, ref_q, rtol=2e-2, atol=2e-2)
      torch.testing.assert_close(out_k, ref_k, rtol=2e-2, atol=2e-2)
  ```

- **W4A8 加载覆盖（针对 4.6 的静默风险）**：用极小 config + 合成 state_dict（按转换脚本命名带 `weight`/`weight_scale`，group 量化再带 `i8_weight_scale`/`i8_weight_zero`）构图，`load_weights` 后断言各 expert 的 scale 参数都出现在 `loaded_params` 里。

无 GPU 的注册 / 子类关系测试可在 CI 常规环境跑；数值 / 加载测试标注 GPU / ixformer 依赖。可选做端到端 eager 精度冒烟兜底。

---

## 6. 附录

### 关键文件清单

| 作用 | 文件 |
| --- | --- |
| OOT 注册入口（自动发现 `register_model()`） | [`vllm_iluvatar/ilu_register/__init__.py`](../../vllm_iluvatar/ilu_register/__init__.py) |
| 落地示例：hy_v3 子类化（fusion + overrides + register） | [`vllm_iluvatar/model_executor/models/hy_v3.py`](../../vllm_iluvatar/model_executor/models/hy_v3.py) |
| 对照示例：qwen3 子类化（仅 fusion） | [`vllm_iluvatar/model_executor/models/qwen3.py`](../../vllm_iluvatar/model_executor/models/qwen3.py) |
| 天数融合 op | [`vllm_iluvatar/model_executor/layers/layernorm.py`](../../vllm_iluvatar/model_executor/layers/layernorm.py) |
| 测试示例 | [`tests/model_executor/models/test_ilu_hy_v3.py`](../../tests/model_executor/models/test_ilu_hy_v3.py) |

### 步骤速览

1. **定位 delta**：fork vs 社区基线逐字节比对，锁定真正的天数差异（如一处 norm 融合），剔除死代码。
2. **子类化**：`class <Arch>(_Community<Arch>)`，`__init__` 调 `super().__init__()` 后做原地后处理。
3. **fusion**：`<arch>_..._fusion` 用 `__class__` 交换把社区多步算子换成融合 op（保留权重），并安装天数 op（参数取自社区已建模块）。
4. **overrides**（按需）：`<arch>_..._overrides` 隔离可回退的精度 / 属性取舍，docstring 注明删除条件。
5. **register_model**：用社区 arch 名指向插件子类；模块放 `models/` 下自动发现。
6. **量化加载**：继承社区 `load_weights`，不改映射即无回归；W4A8 静默风险靠加载覆盖测试兜底。
7. **测试**：OOT 注册 + 子类关系（无 GPU）、融合 op 数值对齐（GPU）、W4A8 加载覆盖断言。

本文所有代码引用来自 hy_v3（`HYV3ForCausalLM`）与 qwen3（`Qwen3ForCausalLM`）两处落地实现，可作为对照。
