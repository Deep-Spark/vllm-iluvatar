# ivcore11 / ivcore30 默认加载 FP8 / MXFP8 / 未量化 checkpoint：按架构的默认计算路径

> **开发者设计文档** · 目标达成度与缺口分析 · 以文档 1（FP8 Linear）与文档 2（MoE）为基础
>
> **各架构默认目标（不传 `additional_config`）**：
>
> - **ivcore11**：Linear = `w8a8_int8`（W8A8 INT8），MoE = `ixinfer_w4a8_nn`（IxInfer，INT4 权重 + INT8 激活）。
> - **ivcore30**：Linear = `fp8`（FP8 执行），MoE = `ixmm_fp4fp8`（IxMM，FP4 权重 + FP8 激活）。
> - 两架构 MoE 的默认目标是**不同格式**，不可混用、不可跨架构复用；`ixmm_w4a8_nn` 仅为 ivcore30 的非默认显式 mode。

---

## 目录

- [1. 目标与范围](#1-目标与范围)
- [2. 现状：FP8 Linear 的计算选择与权重布局](#2-现状fp8-linear-的计算选择与权重布局)
- [3. 现状：MoE 的框架接入、权重加载与计算 backend](#3-现状moe-的框架接入权重加载与计算-backend)
- [4. ivcore11 目标达成度](#4-ivcore11-目标达成度)
- [5. 五道门槛](#5-五道门槛)
- [6. 改造方案](#6-改造方案)
- [7. 文件职责索引](#7-文件职责索引)
- [8. 验证入口与边界](#8-验证入口与边界)

---

## 1. 目标与范围

**目标**：在 ivcore11（150）与 ivcore30（190）上默认加载 `fp8`、`mxfp8` 以及**未量化（BF16/FP16）**等原始 checkpoint，不要求用户显式传 `additional_config`，直接走各自架构的默认计算路径（ivcore11：Linear `w8a8_int8` / MoE `ixinfer_w4a8_nn`；ivcore30：Linear `fp8` / MoE `ixmm_fp4fp8`）。本文正文以 ivcore11 的判定链与缺口为主线，ivcore30 的达成度与差异单列于 §4.4。

**结论先行**：Linear 侧目标**已经达成**（对 `fp8` 源），MoE 侧**仅对 MXFP4 源达成**。`fp8`、`mxfp8` 与未量化三类 checkpoint 的专家层在 ivcore11 上目前都达不到 W4A8 NN，Linear 侧 `mxfp8` 与未量化同样达不到 W8A8 INT8。

阻塞分两类，**不是算子缺失**：

- `fp8` / `mxfp8` / DeepSeek V4 Linear：四处显式的架构门槛（§5 的 A–D）。其中 **DeepSeek V4 的 Linear 是个例外情况**——它卡住的原因只是 Config 没接管 `LinearBase`，社区 `Fp8LinearMethod` 不会调用 `fp8_linear_backend()`；补一个分支即可，见 §4.1 与 §6.2。
- 未量化：**根本没有分发入口**（§5 的门槛 E）。社区的 `override_quantization_method` 检测循环整体包在 `if quant_cfg is not None:` 之内，没有 `quantization_config` 的 checkpoint 压根不会创建任何 `QuantizationConfig`，层直接实例化 `UnquantizedLinearMethod` / `UnquantizedFusedMoEMethod`。这是**与前两类性质不同的问题**，必须单独解决。

本文不重复文档 1 与文档 2 的完整设计，只汇总与本目标相关的判定链，并给出缺口与改造路径。涉及的通信适配、Hybrid EP 接线以文档 2 第 5 节与第 8 节为准。

---

## 2. 现状：FP8 Linear 的计算选择与权重布局

### 2.1 计算选择

执行偏好由 `additional_config.ilu_compute.linear.fp8_backend` 决定，不修改 checkpoint 的 `quantization_config`。解析在 `config/ilu_compute.py`：

| 取值 | ivcore11 | ivcore30 |
| --- | --- | --- |
| `auto`（默认） | **`w8a8_int8`** | `fp8` |
| `w8a8_int8` | 可显式选择 | 可显式选择 |
| `fp8` | 报不支持 | 可显式选择 |

即 **ivcore11 上不传任何配置时，FP8 Linear 就走 W8A8 INT8**，这正是目标状态。计算方式在 Method 初始化时确定，两条路径都按源 checkpoint 创建参数、绑定 TP/fused loader 并加载权重，运行格式在 `process_weights_after_loading()` 中准备完成。

### 2.2 权重转换

INT8 分支分两步，均在加载阶段完成，forward 与 CUDA Graph replay 只消费准备好的运行权重：

1. `common/fp8_weights.py` 的 `create_fp8_source_weights()` 创建源 `weight` 与 scale，保留社区 `weight_loader`。block 源用 `weight_scale_inv`，非 block 源用 `weight_scale`；不得为了目标是 INT8 而提前改写源参数或绕过 TP/fused 装配。
2. `finalize_fp8_to_int8()` 发布参数并清理旧 scale，数值转换在 `conversions/fp8_to_int8.py`：源 FP8 与 FP32/E8M0 scale 转 FP32，反量化后按本地输出通道求 `amax/127`，round、clamp 到 `[-127,127]`，输出 INT8 权重与 FP32 channel scale。全零通道 scale=1，按源行块执行以避免整层 FP32 临时副本。**转换不引入 BF16 中间舍入。**

随后由已有 `IluCompressedTensorsW8A8Int8` 准备布局、padding、scale 与 bias。

> **复用要点**：`quantize_fp8_block_to_w8a8_int8` 已同时接受 FP32 与 E8M0 scale，并在内部做 `block_scale.float()`。MXFP8 的 block-32 E8M0 scale 属于它已支持的输入形态，这是第 6 节 MXFP8 Linear 改造的直接复用点。

### 2.3 布局选择

转换先产出本地 `[N, K]` 权重与 FP32 channel scale。创建 W8A8 scheme 时使用 `gemm_mode="PERF"`，后处理的 `_choose_format(N, K)`：

- 优先查 `perf_table[arch][N, K]`；
- 未命中时，N 与 K 均为 64 的倍数选 **NN**（转置为连续 `[K, N]`），否则选 **TN**（保持 `[N, K]` 方向并 padding）。

初始化传入的 `gemm_format="TN"` 会被 `_choose_format()` 覆盖，**不能据此认定所有层最终都是 TN**。布局在加载后固定；forward 再按实际 M 选择 GEMV/GEMM，M 不参与布局决策。

---

## 3. 现状：MoE 的框架接入、权重加载与计算 backend

### 3.1 三段拆分

文档 2 把 MoE 拆成三个可独立扩展的部分，这是本文改造方案的基础：

- **源格式适配**：Method 对接 vLLM 生命周期，Loader 解释 checkpoint 并生成运行权重。
- **本地专家计算**：计算 backend 消费确定的权重格式，组织量化、两次投影、激活与本地归约。
- **通信协议适配**：Experts adapter 对接 Prepare/Finalize。

可复用的计算 backend **不依赖某个模型的 Config，也不解析 checkpoint**。新增源格式时可以复用已有 backend；新增 backend 时才需要提供匹配其目标格式的权重转换。

### 3.2 公共 mode 与架构默认

`backends/fp4/mxfp4_moe.py` 的 `_MXFP4_MODES` / `_MXFP4_DEFAULTS`：

| arch | 可选 mode | 默认 | 解析结果（backend, layout） |
| --- | --- | --- | --- |
| ivcore11 | `ixinfer_w4a8_nn_v2`、`ixinfer_w4a8_tn_v2` | **`ixinfer_w4a8_nn_v2`** | (`ixinfer_w4a8`, `NN`) |
| ivcore30 | `ixmm_fp4fp8`、`ixmm_w4a8_nn` | `ixmm_fp4fp8` | (`ixmm_fp4fp8`, `fp4fp8`) |

所以 **ivcore11 的 MoE 默认就是 W4A8 NN**，也符合目标——但要注意它是 **IxInfer** 的 W4A8 NN，与 ivcore30 的 `ixmm_w4a8_nn` 是两套实现，不可混称。

### 3.3 可达性限制

`select_mxfp4_compute()` 只被 `select_mxfp4_method()` 调用，而后者只从 `IluMxfp4MoEMethod` 可达。换言之，**上面那张默认值表只对 MXFP4 源专家生效**。`fp8` 与 `mxfp8` checkpoint 的专家层走的是另外的 Method，完全不经过这套统一 compute backend 选择。

### 3.4 加载协议

MXFP4 路径采用两阶段配对加载：阶段一运行模型原有映射建立完整配对计划，阶段二按 `(Loader, local expert, w1/w2/w3)` 成对读取，每对固定先 weight 后 scale，跨层复用固定上传槽。单槽容量按单个本地 N×K 投影计算，为 `N*K/2 + N*K/32` 字节。这是一槽顺序协议，当前没有读盘/上传/转换的流水并发。

新增源格式若要复用该协议，需满足：单 checkpoint、顺序 safetensors、无 bias、gated experts、clamped SiLU/SwiGLU、输出端 TopK 加权，以及 hidden/intermediate 满足 WeightSpec 的 256 对齐。

---

## 4. ivcore11 目标达成度

| checkpoint | 层 | 当前 Method / Kernel | ivcore11 现状 | 目标 | 差距 |
| --- | --- | --- | --- | --- | --- |
| `fp8` | Linear | `IluFp8LinearMethod` | ✅ 默认 W8A8 INT8 | W8A8 INT8 | 无 |
| `fp8` | MoE | `IluFp8MoEMethod` | ❌ 要求 ixmm + ivcore30 | W4A8 NN | 缺源格式接入 |
| `mxfp8` | Linear | 社区 `ModelOptMxFp8LinearMethod` + `IluMxfp8LinearKernel` | ❌ kernel patch 被 arch 门槛跳过 | W8A8 INT8 | 缺转换路径 |
| `mxfp8` | MoE | `IluMxFp8MoEMethod` | ✅ 默认 `ixinfer_w4a8_nn_v2`（见 §6.3 完成说明） | W4A8 NN | 无 |
| `modelopt_mxfp8` | Linear / MoE | `IluModelOptMxFp8Config` | ❌ 同 `mxfp8` | 同上 | 同上 |
| MXFP4（DeepSeek V4 / V4.1） | MoE | `IluMxfp4MoEMethod` | ✅ 默认 `ixinfer_w4a8_nn_v2` | W4A8 NN | 无 |
| DeepSeek V4（`deepseek_v4_fp8`） | Linear | **社区 `Fp8LinearMethod`** + `IluFp8BlockScaledMMKernel` | ❌ kernel patch 被 arch 门槛跳过 | W8A8 INT8 | 缺接管 |
| DeepSeek V4.1（`deepseek_v41_fp8`） | Linear | `IluDeepseekV41FP8LinearMethod`（继承 `IluFp8LinearMethod`） | ✅ 默认 W8A8 INT8 | W8A8 INT8 | 无 |
| 无 `quantization_config`（BF16/FP16） | Linear | 社区 `UnquantizedLinearMethod` + ILU `default_unquantized_gemm` | ❌ 现为 W16A16 | W8A8 INT8 | **缺默认入口**（转换器已有） |
| 无 `quantization_config`（BF16/FP16） | MoE | `IluUnquantizedFusedMoEMethod`（OOT） | ❌ 现为 W16A16，且仅 TN | W4A8 NN | **缺默认入口 + 缺转换** |

十项里三项已达成，七项受阻。**已达成的三项恰好证明目标是可行的**：Linear 的 INT8 转换器与 MoE 的 IxInfer W4A8 NN backend 在 ivcore11 上都已验证可用，缺的是把其余几类源接到它们上面。

### 4.1 DeepSeek V4 与 V4.1 的 Linear 不是同一条路

两者的 MoE 都走 `IluMxfp4MoEMethod`，**但 Linear 分发完全不同**，不要因为 MoE 相同就假定 Linear 也相同：

- **V4.1**：`IluDeepseekV41FP8Config.get_quant_method()` **显式接管 `LinearBase`**，返回 `IluDeepseekV41FP8LinearMethod`。它继承 `IluFp8LinearMethod`，因此走 §2.1 的 `fp8_linear_backend()`，ivcore11 上 auto 即 `w8a8_int8`。该文件注释也写明「Normal compute selection is owned by `IluFp8LinearMethod`, including ivcore11 INT8 execution」。
- **V4**：`IluDeepseekV4FP8Config.get_quant_method()` **只重写了 RoutedExperts 的 MXFP4 分支**，`LinearBase` 整个交给 `super()`——即社区 `DeepseekV4FP8Config` → `Fp8Config` → **社区 `Fp8LinearMethod`**。MRO 上没有任何 `IluFp8LinearMethod`，所以 `fp8_linear_backend()` 根本不会被调用，W8A8 INT8 无从谈起。

V4 的 Linear 实际依赖 `patch_fp8_block_scaled_kernel.py` 把 `IluFp8BlockScaledMMKernel` 插到 `_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.CUDA]` 的最前面。但那个 patch 的 `_apply()` 开头同样是 `if not is_ivcore30(): return`——**与门槛 C 是同一种写法**，见 §5。

V4.1 的 Linear 还有两条分支，评估改造时不要遗漏：

| 条件 | 返回 | 说明 |
| --- | --- | --- |
| `prefix.endswith(".wo_a")` 且 TP < 8 且 ivcore11 且 `VLLM_ILU_DS41_GROUPED_INT8_O_PROJ` | `IluDeepseekV41Fp8GroupedInt8Method` | `wo_a` 按 `o_groups` 块对角，TP≥8 时每 rank 一组可走普通 Linear；更少 rank 需要模型自有的 BMM 策略 |
| `int8_scheme is None` 且 `K % 64 != 0` | `IluDeepseekV41Bf16DequantLinearMethod` | 只在 FP8 执行路径触发。ivcore11 上 `int8_scheme` 已设置，**不会**走到这里 |

### 4.2 未量化路径的现状

未量化 checkpoint（如 MiniMax-M3 的 BF16 版本）在 ivcore11 上能跑，但跑的是 W16A16，既不是 W8A8 INT8 也不是 W4A8 NN：

- **Linear**：`ilu_patch/general_patches/patch_layer_utils.py` 把社区 `default_unquantized_gemm` 换成 ixformer 实现，BF16/FP16 走 `ixf_ops.linear`，FP32 保留 torch 路径，并替换 `UnquantizedLinearMethod.process_weights_after_loading`（后者只是把 `ModelWeightParameter` 降级成普通 `Parameter`，不改数值）。**没有架构门槛**，ivcore11 与 ivcore30 行为一致，也不会把 BF16 转成 INT8。
- **MoE**：`IluUnquantizedFusedMoEMethod` 通过 `@UnquantizedFusedMoEMethod.register_oot()` 注册，并在 `__init__` 里把 `unquantized_backend` 设为 `UnquantizedMoeBackend.OOT`，**绕开社区 oracle 对 CUDA 后端的搜索**（`IluPlatform` 继承 `CudaPlatform`，不这样做会选到 corex 上不可用的后端）。`process_weights_after_loading()` 接受 `is_ivcore11() or is_ivcore30()`，但 NN 分支的第一个条件就是 `is_ivcore30()`，所以 **ivcore11 上恒为 TN**，`forward_native` 以 `weight_format="TN"` 调用 `ilu_w16a16_fused_experts`。注意这里的 "TN/NN" 指 **W16A16 的两种 BF16 排布**，与 W4A8 的 TN/NN 不是一回事。

### 4.3 已有的可复用件与缺失件

未量化两条路的完成度差别很大，不要一概而论：

| 环节 | Linear（BF16 → W8A8 INT8） | MoE（BF16 → W4A8 NN） |
| --- | --- | --- |
| 数值转换函数 | ✅ `quantize_native_to_w8a8_int8()`（`ilu_runtime/linear/native_to_w8a8.py`，内部复用 `_quantize_per_channel_int8`） | ❌ 无。最近的 `quantize_fp8_to_w4a8_nn` 是 **block-FP8 源 + ixmm NN 目标** |
| 转换 Method | ✅ `IluNativeToW8A8Int8OnlineLinearMethod` | ❌ 无。两个 `*_to_w4a8_nn` 在线方法都继承 `IluW4A8NNMoEMethod`（ixmm，卡门槛 B） |
| 计算 backend | ✅ `IluCompressedTensorsW8A8Int8` | ✅ `IxInferW4A8Backend`（`ixinfer_w4a8`，ivcore11 支持 `NN`/`TN`） |
| 默认入口 | ❌ 只能经 `ilu_quant`（门槛 A） | ❌ 同左 |

**Linear 侧实际上只差一个入口**：转换函数和 Method 都已存在并被 MTP 的 `eh_proj` 用过（FP8 checkpoint 把该层留作 BF16，而离线 W4A8 NN checkpoint 会连同其他 dense Linear 一起量化）。

**MoE 侧还差一个转换**。可以从 `quantize_fp8_to_w4a8_nn` 改写——它的流程是「反量化 → per-channel INT8 → INT4 → NN 打包」，BF16 源只是跳过第一步。但有一处**必须改**：

```text
IxInferW4A8Backend.weight_specs = ixinfer_w4a8_nn_v2, torch.int8, code_multiplier=18
```

IxInfer v2 的 kernel **自己**按码值 ×18 解释权重，而 `quantize_fp8_to_w4a8_nn` 返回的是 `scale * q4_scale`（已把 18 折进 scale，这是 ixmm 的约定）。直接照搬会让权重被乘两次 18。

### 4.4 ivcore30 的达成度

ivcore30 的目标默认是 Linear `fp8` / MoE `ixmm_fp4fp8`，整体完成度明显高于 ivcore11，缺口性质也不同：

| checkpoint | 层 | 当前 Method / Kernel | ivcore30 现状 | 目标 | 差距 |
| --- | --- | --- | --- | --- | --- |
| `fp8` / DeepSeek V4 / V4.1 | Linear | `IluFp8LinearMethod` / 社区 `Fp8LinearMethod` + `IluFp8BlockScaledMMKernel` | ✅ 默认 `fp8`（auto→fp8；kernel patch 在 ivcore30 生效） | `fp8` | 无 |
| `mxfp8` / `modelopt_mxfp8` | Linear | 社区 `ModelOptMxFp8LinearMethod` + `IluMxfp8LinearKernel` | ✅ 可用（patch 生效） | `fp8` 执行 | 无 |
| MXFP4（DeepSeek V4 / V4.1） | MoE | `IluMxfp4MoEMethod` | ✅ 默认 `ixmm_fp4fp8` | `ixmm_fp4fp8` | 无 |
| `fp8` / `mxfp8` / NVFP4 | MoE | `IluFp8MoEMethod` / `IluMxFp8MoEMethod` / `IluModelOptNvFp4MoEMethod` | ✅ 三者都经 `select_four_bit_expert_compute` 选到 `ixmm_fp4fp8` | `ixmm_fp4fp8` | 无（见 §6.3 完成说明） |
| 无 `quantization_config`（BF16/FP16） | Linear / MoE | 未量化 Method | ❌ 现为 W16A16 | `fp8` / `ixmm_fp4fp8` | **缺默认入口（门槛 E，与 ivcore11 同）** |

要点：

- **Linear 与 MXFP4 MoE 已达标**，无必做改动。门槛 A–D 在 ivcore30 上本就不生效（它们是用来保护 ivcore30 专有路径、只卡 ivcore11 的）。
- `fp8`/`mxfp8`/NVFP4 专家当前就能加载推理，只是走各自独立 Method；是否收敛到统一 `ixmm_fp4fp8` 计算 backend 是**工程整洁度**的可选项，不改数值格式。
- **未量化路径**与 ivcore11 卡在同一处门槛 E（§5），需要同一套入口方案（§6.6）；入口就位后 ivcore30 的未量化 Linear 目标是 `fp8`、MoE 目标是 `ixmm_fp4fp8`。
- `ixmm_w4a8_nn`（INT4 权重）是 ivcore30 的另一个 backend，仅 MXFP4 源、显式 `mode=ixmm_w4a8_nn` 可达，**不是默认、不是本文目标**；不要与默认的 `ixmm_fp4fp8`（FP4）混为一谈。

---

## 5. 五道门槛

缺口不在算子，而在四处显式的 `ivcore30` 判定。改造必须逐一处理，绕过任何一处都会把问题推到更深的地方。

### 门槛 A：`ilu_quant` 整体仅对 ivcore30 开放

`quantization/ilu_runtime/config.py` 在解析入口即调用 `require_ivcore30("additional_config.ilu_quant")`。这意味着**现有的在线量化方法在 ivcore11 上一个都用不了**，包括看起来正好对口的 `IluFp8ToW4A8NNOnlineMoEMethod`。文档 1 也明确写了「默认 150 路径不经由仅向 ivcore30 开放的 `ilu_quant` 入口」。

因此 ivcore11 的默认行为**不能**通过 `ilu_quant` 实现，必须走 checkpoint Config 的默认分发。

### 门槛 B：`IluW4A8NNMoEMethod` 自身要求 ivcore30

`backends/w4a8/nn_moe.py` 的 `require_ivcore30_nn()` 在 `is_ivcore30()` 为假时直接 `RuntimeError`。它消费的是 ixformer ixmm bridge 的 NN 布局。`IluFp8ToW4A8NNOnlineMoEMethod` 继承它，因此即使解除门槛 A，这条路在 ivcore11 上仍然不通。

**这是最容易踩的坑**：名字里的 "W4A8 NN" 与 ivcore11 默认的 "W4A8 NN" 指向两套不同实现。ivcore11 的目标应是 `ixinfer_w4a8` backend 的 NN 布局，不是 ixmm NN。

### 门槛 C：两个 Linear kernel patch 都被跳过

两个 patch 的 `_apply()` 开头都是 `if not is_ivcore30(): return`：

| patch | prepend 的 kernel | 目标列表 | 受影响的 checkpoint |
| --- | --- | --- | --- |
| `patch_mxfp8_linear_kernel.py` | `IluMxfp8LinearKernel` | `_POSSIBLE_MXFP8_KERNELS[CUDA]` | `mxfp8` / `modelopt_mxfp8` |
| `patch_fp8_block_scaled_kernel.py` | `IluFp8BlockScaledMMKernel` | `_POSSIBLE_FP8_BLOCK_KERNELS[CUDA]` | **DeepSeek V4**（社区 `Fp8LinearMethod` 路径） |

两份社区候选列表里都没有 Iluvatar kernel，所以这两条 Linear 在 ivcore11 上都没有可用 kernel。

注意这两条的**解法不同**：`mxfp8` 需要新的转换路径（§6.1）；**DeepSeek V4 只需要让 Config 接管 Linear**，把它从社区 `Fp8LinearMethod` 换成 `IluFp8LinearMethod` 系，就能复用已经可用的 §2.1 计算选择（§6.2）。

### 门槛 D：`IluFp8MoEMethod` 硬依赖 ixmm

`backends/fp4/fp8_moe.py` 的 `process_weights_after_loading()` 开头即 `if not _ixmm_backend_available(): raise`，并在错误信息中要求「rebuild ixformer with HAVE_IXMM on ivcore30」。block-FP8 专家因此在 ivcore11 上无法完成加载。

### 门槛 E：未量化 checkpoint 没有任何 Config 入口

这一条与 A–D 性质不同，**不是架构判定，而是检测链本身不会执行**。

`vllm/config/model.py` 的 `_verify_quantization()` 把整个 `overrides` 检测循环包在 `if quant_cfg is not None:` 之内。未量化 checkpoint 没有 `quantization_config`，于是：

1. `override_quantization_method()` **一次都不会被调用**——插件无法像 `fp8` / `mxfp8` 那样靠改写注册名接管；
2. `model_config.quantization` 保持 `None`；
3. `get_quant_config()` 不会被调用（它第一行就是 `if model_config.quantization is None: raise`）；
4. `LinearBase` / `RoutedExperts` 在 `quant_config is None` 时直接实例化 `UnquantizedLinearMethod` / `UnquantizedFusedMoEMethod`。

所以 §6.1–6.3 那套「在 Config 的 ivcore11 分支改派 Method」的做法，**对未量化路径完全不适用**——没有 Config 可改。必须先解决入口，见 §6.6。

---

## 6. 改造方案

各项改造共用一条思路：**新增源格式 Loader，复用已有计算 backend**，而不是新写 backend。这正是文档 2 第 6.3 节给出的扩展方式，也避免了反向依赖。未量化路径多一步前置工作——先得有入口（§6.6）。

### 6.1 MXFP8 Linear → W8A8 INT8

把 MXFP8 当作 block 形状为 `[1, 32]`、scale dtype 为 E8M0 的 FP8 源，接入 §2.2 的既有两步转换：

- `create_fp8_source_weights()` 按 MXFP8 的参数命名创建源 `weight` 与 E8M0 scale，保留社区 loader；
- `finalize_fp8_to_int8()` 复用 `quantize_fp8_block_to_w8a8_int8`，该函数已接受 E8M0 scale 并在内部 `float()`，无需新转换器；
- 产出 `[N, K]` INT8 + FP32 channel scale 后交给 `IluCompressedTensorsW8A8Int8`，布局由 `_choose_format()` 自行决定。

接入点是一个 MXFP8 专用的 Linear Method，由 `IluMxFp8Config` / `IluModelOptMxFp8Config` 在 ivcore11 分支返回；ivcore30 维持现有 `ModelOptMxFp8LinearMethod` + `IluMxfp8LinearKernel`，门槛 C 不必解除。

**需要确认**：MXFP8 的 block 是 `[1, 32]`，而现有 block 转换按源行块执行、针对 `[128,128]` / `[32,32]` 验证过。`[1, 32]` 下每行一组 scale，等价于 per-row-group，需要确认转换器的尾部行块处理与 `amax` 聚合在这个粒度上仍然正确。

### 6.2 DeepSeek V4 Linear → W8A8 INT8

这一项是全部改造里最轻的，**不需要任何新转换或新 kernel**：`IluDeepseekV4FP8Config.get_quant_method()` 目前只重写 RoutedExperts，补一个 `LinearBase` 分支返回 `IluFp8LinearMethod`（或 V4.1 那样的薄子类）即可。`IluFp8LinearMethod.__init__` 会调用 `fp8_linear_backend()`，ivcore11 上自动得到 `w8a8_int8` 并建好 `IluCompressedTensorsW8A8Int8`，§2.2 / §2.3 的转换与布局原样复用。

两点需要注意：

- **V4 的 block 是 `[128, 128]`**（V4.1 是 `[32, 32]`），正是现有 FP8→INT8 转换器验证过的粒度，不存在 §6.1 那个 `[1, 32]` 的未知数。
- 不要顺手把门槛 C 的 `patch_fp8_block_scaled_kernel.py` 向 ivcore11 放开。那是 **FP8 执行路径**的 kernel，与本项要达成的 INT8 执行无关；ivcore30 继续走社区 `Fp8LinearMethod` + 该 kernel 的现状不应改变。

### 6.3 MXFP8 / NVFP4 MoE → IxInfer W4A8 NN

> **已完成（实现与下文的方案有出入）**：没有按「每个源各写一个 ivcore11 子类」落地，而是把三个类合并成两个与架构无关的 Method——`IluMxFp8MoEMethod`（MXFP8，继承 `IluFp8MoEMethod` 的配对流式加载）和 `IluModelOptNvFp4MoEMethod`（NVFP4，整层加载后分块转换）。目标 weight contract 由 `select_four_bit_expert_compute` 决定：ivcore11 `ixinfer_w4a8_nn_v2`，ivcore30 `ixmm_fp4fp8`。转换之后的生命周期全部来自 `MoEComputeBindingMixin`。`IluMxFp8FusedMoE` / `IluMxFp8ToFp4OnlineMoE` / `IluModelOptNvFp4ToW4A8MoEMethod` 已删除；下文保留为数值路径的设计依据。

ivcore11 上有三个方法曾把专家转成 **ix30 FP4**（ivcore30 路径），要落到 `ixinfer_w4a8_nn`：`IluMxFp8FusedMoE`（MXFP8 整层 host 暂存）、`IluMxFp8ToFp4OnlineMoE`（MXFP8 逐 shard 在线）、`IluModelOptNvFp4MoEMethod`（NVFP4）。计算 backend `IxInferW4A8Backend` 已有，只需为每个源写「反量化 → FP32 → 统一的 INT4 W4A8 NN 量化」的 Loader，复用各自已有的加载骨架。

**共用的 5 步（逐专家 shard `[N,K]`）：**

1. **反量化到 FP32 逻辑权重**（三者唯一不同的一步，见下表）。
2. 逐输出通道对称 INT8：`scale = amax(|value|, dim=K) / 127`，`q8 = round(value / scale)`。
3. INT4：`q4 = round(q8 / 18).clamp(-8, 7)`（码值倍率 18）。
4. NN 打包：复用 `ilu_runtime/moe/fp8_to_w4a8_nn.py` 的 `_pack_int4` 与 NN permute。
5. 目标 `WeightSpec` 取 `select_moe_backend("ivcore11", name="ixinfer_w4a8", layout="NN")`，按目标 spec 直接分配最终 `w13_weight` / `w2_weight` 与 scale，不先分配整层源参数。

**唯一必须改的点**：返回的 per-channel scale 是 **`scale` 本身，不是 `scale * q4_scale`**。`quantize_fp8_to_w4a8_nn` 是为 ixmm 写的，把 18 折进了 scale；而 IxInfer 的 `WeightSpec` 带 `code_multiplier=18`，**kernel 自己对码值做 ×18**，再折进 scale 会乘两次。因此**不要直接复用 `IluFp8ToW4A8NNOnlineMoEMethod` 这个类**（它继承 `IluW4A8NNMoEMethod`，卡门槛 B 且目标是 ixmm NN），只复用其中的纯函数（`_pack_int4`、NN permute、逐通道量化）。

**各源第 1 步的反量化与可复用骨架：**

| 方法 | 源格式 | 第 1 步反量化 | 复用骨架 |
| --- | --- | --- | --- |
| `IluMxFp8FusedMoE` | FP8 e4m3 + E8M0 block-`(1,32)` scale，整层 host 暂存 | `value = w.float() * e8m0_to_fp32(s)`（block 1×32 广播） | `process_weights_after_loading` 里 `preprocess_fp8_moe_weight_for_ixmm` 的**按专家分块**结构（限 HBM 峰值），把 FP4 输出换成 NN INT4；`create_weights` 改分配 NN 目标形状 |
| `IluMxFp8ToFp4OnlineMoE` | 同上，逐 shard 在线转换 | 同上 | `_convert()` 里把 `convert_to_ix30_fp4` 换成 NN 转换；保留 staging / pending / 流式骨架；`create_weights` 把 `ix30_fp4_shapes` 换成 NN `weight_spec.shapes` |
| `IluModelOptNvFp4MoEMethod` | FP4（uint8 打包）+ e4m3 block-16 scale + 每专家 fp32 global scale（`weight_scale_2`） | 解包 nibble → `fp4_mag * block16_scale.float() * global_scale`（两级 scale） | 复用其现有 `[E,n,k/2]→[E,n,k]` 解包 + 两级反量化（`nvfp4_moe.py`），只把第 4 步起的 per-256 FP4 换成 per-channel INT4 NN |

**两处要特别当心：**

- **`w2`（down_proj）的 per-channel scale 必须按完整 K 求 amax**。ixinfer NN 的 scale 逐输出通道、跨 K 聚合；TP 下 w2 按 K 行并行切分，必须像 `IluFp8ToW4A8NNOnlineMoEMethod` 那样**先保留完整 checkpoint K 再求 amax、再切片**（它为此保留 `*_weight_scale_inv` 的 full-K block scale 并传 `k_start/k_size`）。MXFP8 在线路径现在把 w2 按 K 切（`_load_w2(shard_dim=1)`），改 NN 时要补这一步。
- **NVFP4 是 FP4 → FP32 → INT4 的二次量化**，两个低比特格式之间重量化精度损失叠加，默认启用前应单独做精度评测（见 §8）。

如果 MXFP8 能复用 MXFP4 的两阶段配对加载协议，还需满足 §3.4 列出的准入条件；否则按 `IluMxFp8ToFp4OnlineMoE` 的在线逐 shard 转换组织，避免整层源常驻。

### 6.4 block-FP8 MoE → IxInfer W4A8 NN

与 6.2 同构，源是 block-FP8（`weight_block_size=[128,128]`）。已有 `quantize_fp8_to_w4a8_nn`（在 `ilu_runtime/moe/fp8_to_w4a8_nn.py`）实现了 FP8→W4A8 NN 的数值转换，可以把纯转换函数提到 `conversions/` 下共用，但**不能直接复用 `IluFp8ToW4A8NNOnlineMoEMethod` 这个类**——它继承的 `IluW4A8NNMoEMethod` 卡在门槛 B，且目标是 ixmm NN 而非 IxInfer NN，两者 `WeightSpec` 的 code/scale 语义不同。

文档 2 第 4.2 节的原则在这里直接适用：**仅凭 TN/NN 名称不能判断权重能否跨 backend 复用，必须比较完整 `WeightSpec`。**

### 6.5 默认分发的接线

`fp8` / `mxfp8` 三项改造完成后，需要在 Config 层把 ivcore11 的默认分发接上（未量化路径的接线见 §6.6）：

- `IluFp8Config.get_quant_method()` 的 MoE 分支：当前 `weight_block_size` 非空即返回 `IluFp8MoEMethod`。应在 ivcore11 上改为返回新的 FP8→W4A8 NN Method。
- `IluMxFp8Config` / `IluModelOptMxFp8Config`：ivcore11 分支返回新的 MXFP8 Linear Method 与 MXFP8 MoE Method。
- 公共 mode 列表：若希望用户能显式选择，需同步更新 `_MXFP4_MODES` 式的 mode 表（注册 backend 本身不会自动开放用户配置）。若只做默认行为，可以先不开放 mode。

门槛 A 与 B 不需要解除：新路径走 checkpoint Config 的默认分发，不经 `ilu_quant`；目标 backend 是 IxInfer 而非 ixmm NN。门槛 C 与 D 保持现状，继续保护 ivcore30 专有路径。

### 6.6 未量化 checkpoint 的入口

这一项必须先于 6.6、6.7 落地，因为门槛 E 决定了后两者没有可挂载的位置。两个候选入口：

#### 方案一：合成一个 ivcore11 默认量化名（推荐）

在 `IluPlatform.check_and_update_config()` 里，当 `model_config.quantization is None` 且 arch 为 ivcore11 时，把它置为一个插件注册的名字（例如 `ilu_native_auto`）。`get_quant_config()` 只依据 `model_config.quantization` 取类，所以后续 Config → Method 的整条链路可以原样复用，**与 §6.5 的接线方式一致**。

这条路的优势是**允许流式转换**：新 Method 的 `create_weights()` 可以直接按目标 spec 分配 INT8 / INT4 参数，像 `IluFp8ToW4A8NNOnlineMoEMethod` 那样逐 shard 转换，不需要整层 BF16 常驻。

需要确认的点：`get_quant_config()` 在读不到 HF 侧 quant 配置字典时，能否把空配置交给 `from_config()`；以及 `check_and_update_config()` 的执行时机确实早于量化配置对象的构造。

#### 方案二：在未量化 Method 内部自行量化

直接扩展 `IluUnquantizedFusedMoEMethod` 与已被 patch 的 `UnquantizedLinearMethod.process_weights_after_loading`。不需要新入口，但有两个实打实的代价：

- `create_weights()` 已按 BF16 分配完参数，转换只能发生在加载之后，**专家层会出现 BF16 与 INT4 同时常驻的峰值**，没有流式空间；
- 会把量化分支塞进 OOT 注册的 W16A16 路径，破坏其单一职责——该类的 docstring 明确写了「intentionally only for ILUVATAR_W16A16，不要加通用 fallback」。

**除非方案一的两个确认点被证伪，否则应走方案一。**

### 6.7 未量化 Linear → W8A8 INT8

入口就位后，这一项是六项里最轻的：数值转换 `quantize_native_to_w8a8_int8()` 与 Method `IluNativeToW8A8Int8OnlineLinearMethod` 都已存在且已有使用者，只需让新 Config 的 Linear 分支在 ivcore11 上返回它（或返回一个不依赖 `ilu_runtime` context 的同构 Method，以免被门槛 A 牵连）。

产出仍是 `[N, K]` INT8 + FP32 channel scale，交给 `IluCompressedTensorsW8A8Int8`，布局由 `_choose_format()` 决定，与 §2.3 完全一致。

### 6.8 未量化 MoE → IxInfer W4A8 NN

新增 BF16 源 Loader，目标 `WeightSpec` 同样取 `select_moe_backend("ivcore11", name="ixinfer_w4a8", layout="NN")`。数值转换需要新写，但可以从 `quantize_fp8_to_w4a8_nn` 改：去掉 block-FP8 反量化这一步，从 `value = weight.float()` 开始。

**两处不能照抄**：

1. **scale 约定**。`IxInferW4A8Backend` 的 spec 带 `code_multiplier=18`，kernel 自己做 ×18；`quantize_fp8_to_w4a8_nn` 返回的 `scale * q4_scale` 已经折进了 18（ixmm 约定）。照抄会乘两次。
2. **全局 K 统计**。`quantize_fp8_to_w4a8_nn` 刻意在 TP 切片**之前**用完整 checkpoint K 求 amax，并在注释里说明「在本地分片上重算不等价」。BF16 源同样要保留这个顺序。

不要复用 `IluFp8ToW4A8NNOnlineMoEMethod` 或 `IluMxfp4ToW4A8NNMoEMethod` 这两个类——它们都继承 `IluW4A8NNMoEMethod`，卡在门槛 B 且目标是 ixmm NN。可复用的是**转换函数**与 `_pack_int4` 这类纯工具。

### 6.9 改造顺序建议

1. **6.2（DeepSeek V4 Linear）** 最轻，补一个 Config 分支即可，复用链路全部现成且已验证。建议先做它来确认 §2.1–§2.3 这条路在 V4 的 `[128,128]` 源上确实跑通。
2. **6.1（MXFP8 Linear）** 复用面最大，能独立验证——只要 `[1,32]` block 的转换正确性确认通过。
3. **6.6（未量化入口）** 后两项的前置。单独落地时可以先只接 Linear，用一个 BF16 模型验证入口是否生效。
4. **6.7（未量化 Linear）** 紧随入口，几乎没有新代码。
5. **6.4（block-FP8 MoE）** 数值转换已有实现，主要工作是 `WeightSpec` 对齐与 Loader 组织。
6. **6.8（未量化 MoE）** 需要新转换函数，但源格式最简单（无 scale 需要解析）。
7. **6.3（MXFP8 MoE）** 最后，它同时涉及新源格式与新转换。

每一步都应能独立落地并验证，不要求多项同时完成。

---

## 7. 文件职责索引

以下路径相对仓库根目录。

### 7.1 需要修改

| 文件 | 改动 |
| --- | --- |
| `vllm_iluvatar/model_executor/layers/quantization/configs/fp8.py` | MoE 分支在 ivcore11 改派新 Method |
| `.../quantization/configs/deepseek/v4_fp8.py` | §6.2：补 `LinearBase` 分支，接管社区 `Fp8LinearMethod` |
| `vllm_iluvatar/model_executor/layers/quantization/configs/mxfp8.py` | ivcore11 分支返回新的 Linear / MoE Method |
| `vllm_iluvatar/model_executor/layers/quantization/configs/modelopt/mxfp8.py` | 同上 |
| `vllm_iluvatar/model_executor/layers/quantization/conversions/fp8_to_int8.py` | 确认 `[1,32]` block 粒度；必要时补尾部处理 |
| `vllm_iluvatar/model_executor/layers/quantization/_registration.py` | 新 Method 的显式导入 |
| `vllm_iluvatar/platforms/ilu_platform.py` | §6.6 方案一：ivcore11 + 无 `quantization_config` 时合成默认量化名 |
| `vllm_iluvatar/model_executor/layers/quantization/configs/`（新增） | 未量化 checkpoint 的 ivcore11 默认 Config |
| `vllm_iluvatar/model_executor/layers/quantization/conversions/`（新增） | BF16 → IxInfer W4A8 NN 的数值转换（勿折入 ×18） |

### 7.2 直接复用，不改

| 文件 | 复用的能力 |
| --- | --- |
| `vllm_iluvatar/config/ilu_compute.py` | `fp8_linear_backend()` 的架构默认值，已满足目标 |
| `vllm_iluvatar/model_executor/layers/quantization/common/fp8_weights.py` | 源参数创建与加载后发布 |
| `.../compressed_tensors/schemes/ilu_compressed_tensors_w8a8_int8.py` | INT8 布局选择、padding、GEMV/GEMM 调度 |
| `.../fused_moe/experts/ilu_compute_backend.py` | `MoEWeightSpec` / `select_moe_backend` / `make_moe_compute` |
| `.../fused_moe/experts/ilu_ixinfer_w4a8.py` | ivcore11 的 W4A8 NN 计算实现 |
| `.../backends/fp8/linear.py` | `IluFp8LinearMethod`，§6.2 直接实例化它即可 |
| `.../backends/fp8/deepseek_v41_linear.py` | V4.1 的薄子类写法，§6.2 可参照 |
| `.../fused_moe/experts/ilu_compute_experts.py` | Standard / Batched 协议适配 |
| `vllm_iluvatar/model_executor/model_loader/ilu_default_loader.py` | 两阶段配对加载会话（若复用该协议） |
| `.../ilu_runtime/linear/native_to_w8a8.py` | `quantize_native_to_w8a8_int8()` 与 BF16→INT8 Method，§6.7 直接复用 |
| `.../ilu_runtime/moe/fp8_to_w4a8_nn.py` | `_pack_int4` 与 NN 打包流程，§6.8 作为改写蓝本（**不复用其类与 scale 约定**） |

### 7.3 保持现状，不要改

| 文件 | 原因 |
| --- | --- |
| `.../ilu_runtime/config.py` | `require_ivcore30` 保护显式 `ilu_quant` 入口 |
| `.../backends/w4a8/nn_moe.py` | ixmm NN 实现，与 IxInfer NN 不同契约 |
| `ilu_patch/general_patches/patch_mxfp8_linear_kernel.py` | ivcore30 的 MXFP8 kernel 插队 |
| `ilu_patch/general_patches/patch_fp8_block_scaled_kernel.py` | ivcore30 的 block-FP8 kernel 插队，服务 FP8 执行路径而非 INT8 |
| `.../backends/fp4/fp8_moe.py` | ivcore30 的 block-FP8 → ix30 FP4 |
| `.../fused_moe/unquantized_fused_moe_method.py` | W16A16 OOT 路径。§6.6 走方案一时不动它；只有退而选方案二才需要改 |
| `ilu_patch/general_patches/patch_layer_utils.py` | 未量化 Linear 的 ixformer W16A16 GEMM，同上 |

---

## 8. 验证入口与边界

### 8.1 建议的测试覆盖

```bash
python -m pytest -q \
  tests/model_executor/layers/quantization/test_ilu_compute_config.py \
  tests/model_executor/layers/quantization/test_ilu_fp8_to_w8a8_linear.py \
  tests/model_executor/layers/quantization/test_ilu_moe_backend.py \
  tests/model_executor/layers/quantization/test_ilu_mxfp4_loader.py
```

按改动选择：转换器改动联动 `test_ilu_fp8_to_w8a8_linear.py`（它已覆盖 E8M0/FP32 scale、尾部行块与 FP32 数值参考）；新增 Loader 联动 loader 与 backend 测试；Config 分发改动联动 compute config 测试。

### 8.2 边界声明

- 配置与 mock 检查**不代替**设备数值验证、模型精度评测与性能测试。三者回答不同问题，具体运行记录归验证报告。
- Row-parallel 的 channel scale 按当前 rank 的本地 K 分片计算，**不保证**与离线 full-K 量化逐字节相同。这一约束对新增的 MXFP8 路径同样成立。
- MXFP4 到 W4A8 包含数值重量化；MXFP8 / block-FP8 / BF16 到 W4A8 同理。切换计算模式时，**精度与性能应分别比较**，不能用其中一项的结论代替另一项。
- **未量化路径的精度风险最高**：BF16 → W4A8 是本文六项里压缩比最大的一条，且源权重未经离线量化感知训练。默认启用前应有精度评测，必要时保留回退到 W16A16 的开关。
- 本文第 4 节的达成度表基于当前代码判定链推导。`mxfp8` 与 `modelopt_mxfp8` 在 ivcore11 上的失败模式尚未实机复现，若实测结果与表中不符，以实测为准并回头修正判定链。
- §6.6 方案一的两个确认点（`get_quant_config()` 能否接受空 HF quant 字典、`check_and_update_config()` 的时机）**尚未验证**。若其中任一不成立，入口方案需要重选。

---

## 相关文档

- [1. 原始 FP8 Linear 的计算选择与权重布局](1-fp8-linear-compute-and-weight-layout.md)
- [2. MoE 的框架接入、权重加载与计算 backend](2-moe-compute-and-weight-layout.md)
- [运行时量化指南](../guides/8-how-to-use-iluvatar-runtime-quant-via-additional-config.md)
- [DP / EP 使用指南](../guides/5-how-to-use-dp-ep.md)
