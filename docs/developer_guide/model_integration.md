# 接入模型



本页写给要在 vllm-iluvatar 插件里**新接入一个模型**的开发者。改动全部落在插件仓库内，社区 vLLM 源码保持零改动。只想知道**已支持哪些模型、怎么启动**的用户见「用户指南/支持模型」。

接入分两种范式，先判断用哪一种，再按对应路线搭骨架。

## 两种范式何时用



两种范式的判断准则：

| 维度 | 快速适配社区模型 | 接入天数优化模型 |
| --- | --- | --- |
| 与社区关系 | 子类化、继承为主 + 逐类最小覆盖 | 模型算法链完全重写，尽量不走社区接口 |
| arch 命名 | 沿用社区 arch 名做覆盖（如 `HYV3ForCausalLM`） | 优先复用社区名，新增架构时用 `IluXXXForCausalLM` |
| 天数差异规模 | 少数算子（如一处 q/k norm 融合） | attention / 量化 / 投机解码全链路优化 |
| 文件布局 | 单文件 `model_executor/models/<arch>.py` | 多文件包 `models/<model_type>/` |
| 参考实现 | hy_v3、qwen3 | GLM-5.2 `IluGlmMoeDsa`、deepseek_v4 |

判断准则：**社区已能正确跑该模型、天数只想换掉个别算子** → 快速适配；**需要深入改写 attention / 量化 / MTP 算法链** → 天数优化。

## 路线一：快速适配社区模型



思路是复用社区模型、只覆盖含天数 delta 的少数几处（通常一处融合算子），其余（`Attention` / `FusedMoE` / `get_rope` / 量化加载路径）全部原样继承。

### 命名与目录约定



- **arch 命名**：沿用社区原 arch 名做覆盖（如 `HYV3ForCausalLM` / `Qwen3ForCausalLM`），而非新起 `IluXXX`——目的是「替换社区实现」而非「新增架构」。
- **文件布局**：单文件 `vllm_iluvatar/model_executor/models/<arch>.py`，无需建子包。
- **后处理函数命名**：对社区模型做原地改写的辅助函数统一命名为 `<arch>_<组件/功能>_<结尾词>`，结尾词二选一——`...fusion`（核心天数融合，一般必留）、`...overrides`（可回退的精度/属性取舍，docstring 须注明删除条件）。

### 端到端步骤骨架



1. **定位 delta**：fork 的社区源文件与其社区基线逐字节比对，锁定真正的天数差异，剔除死代码。
2. **子类化**：`class <Arch>(_Community<Arch>)`，`__init__` 调 `super().__init__()` 后做原地后处理。
3. **fusion**：`<arch>_..._fusion` 用重赋 `__class__` 把社区多步算子换成融合 op（保留全部权重/状态），并安装天数 op（构造参数取自社区已建好的模块）。
4. **overrides**（按需）：`<arch>_..._overrides` 隔离可回退的精度/属性取舍。
5. **register_model**：文件末尾暴露 `register_model()` 钩子，用**社区 arch 名**调 `ModelRegistry.register_model` 指向插件子类（字符串路径惰性引用）。

关键注册点：模块放在 `vllm_iluvatar/model_executor/models/` 包下、暴露 `register_model()` 即被 `apply_oot_registrations` 自动发现，**无需改 `ilu_register/__init__.py` 或 `_OOT_MODEL_PACKAGES`**（后者只登记多文件的天数优化模型子包）。

## 路线二：接入天数优化模型



思路是通过 registry 把目标架构路由到插件内的 Iluvatar 实现，并把模型算法链完全重写在插件内，运行时尽量走插件自己的接口（ilu attention backend、`ixf_ops` marlin / W4A8 kernel、plugin-local opaque op）。

### 命名与目录约定



- **arch 命名**：registry key 用「要让 checkpoint 命中的名字」；已有社区 arch 时优先复用（如 `DeepseekV4ForCausalLM`），新增架构时用 `IluXXXForCausalLM`。
- **包布局**：每个天数模型是 `vllm_iluvatar/models/<model_type>/` 下的多文件包，布局对齐 `deepseek_v4`：`__init__.py`（包说明）、`model.py`（模型图 + 权重加载 + `register_model()`）、`attention.py`（自包含 attention 链）、`mtp.py`（可选 MTP draft）。
- **登记到 `_OOT_MODEL_PACKAGES`**：自动发现不递归子包，多文件模型包须显式列进 `ilu_register/__init__.py` 的 `_OOT_MODEL_PACKAGES`。

### 端到端步骤骨架



1. **权重转换**（通常需要）：离线把原始 checkpoint 混合量化成天数专有方案，重建 `model.safetensors.index.json`，并按需改写 `config.json` 的 `architectures` 命中 registry key。
2. **模型适配**：以 `glm_moe_dsa/model.py` 为模板搭 `IluXXXForCausalLM`，实现融合参数映射与严格 reverse 校验，`register_model()` 注册 causal-LM（及 MTP）arch。
3. **attention 抽离**（核心）：把整条 attention 算法链自包含在插件内（外层 wrapper / 内层 `SparseMLA` / indexer 三层），数据依赖控制流用 `direct_register_custom_op` 封成 **plugin-local opaque op**（独立算子名、padding 行保护）。
4. **MoE 与混合量化**：复用 `compressed-tensors`（`IluQuantConfig`，详见「扩展量化」）。
5. **三种运行模式**：先 eager 打基线；再用 `--compilation-config` 开 `FULL_DECODE_ONLY` cuda-graph；最后加 `mtp.py` + 两个 general patch 开 MTP。

关键注册点：模型 + 量化配置走 `ilu_register`（`ModelRegistry.register_model` 与 `@register_quantization_config`），MTP 另需少量 `ilu_patch.general_patches`。
