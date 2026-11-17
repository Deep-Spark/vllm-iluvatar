# MoE 的框架接入、权重加载与计算 backend

> **开发者设计文档** · OOT 量化接入 · 源权重、运行格式与通信适配分离

---

## 目录

- [1. 框架接入与改造思路](#1-框架接入与改造思路)
- [2. 以 MXFP4 为例的适配流程](#2-以-mxfp4-为例的适配流程)
- [3. Method 与 Loader 的职责](#3-method-与-loader-的职责)
- [4. 计算 backend 的公共设计与实现](#4-计算-backend-的公共设计与实现)
- [5. Standard 与 Batched 通信适配](#5-standard-与-batched-通信适配)
- [6. 配置与扩展方法](#6-配置与扩展方法)
- [7. 文件职责与验证入口](#7-文件职责与验证入口)
- [8. 支持矩阵](#8-支持矩阵)

---

## 1. 框架接入与改造思路

vLLM 通过量化 Config 的 `get_quant_method()` 为专家层选择 MoE Method。插件使用 `register_quantization_config` 注册自己的 Config，再返回继承并适配社区接口的 Method。量化接入采用 Config 注册机制，不使用 `CustomOp.register_oot`。

本次设计将 MoE 拆成三个可独立扩展的部分：

- **源格式适配**：Method 对接 vLLM 生命周期，Loader 解释 checkpoint 并生成运行权重。
- **本地专家计算**：计算 backend 消费确定的权重格式，组织量化、两次投影、激活和本地归约。
- **通信协议适配**：Experts adapter 对接 Prepare/Finalize，向 backend 提供相应布局的输入。

模型接入层继续提供路由、激活参数和并行配置。可复用计算 backend 不依赖某个模型的 Config，也不解析 checkpoint。新增源格式时，可以复用已有 backend；新增 backend 时，需要提供与其目标格式匹配的权重转换。

图 1：MoE 框架接入与改造思路

![MoE 框架接入与改造思路](images/moe-framework-adaptation.png)

[可编辑 SVG 源图](images/moe-framework-adaptation.svg) · [Mermaid 源码](images/moe-framework-adaptation.mmd)

本文以 vLLM 0.31 接口及 `IluMxfp4MoEMethod` 的统一 backend 路径说明设计。其他源格式可沿用相同职责划分；是否已经接入，以文末支持矩阵为准。

---

## 2. 以 MXFP4 为例的适配流程

`IluMxfp4MoEMethod` 继承社区 `Mxfp4MoEMethod`，保留 vLLM 的调用接口，替换其中的平台计算选择与加载处理。构造时调用 `FusedMoEMethodBase` 初始化，避免社区 Method 构造函数先选择不适用的计算实现。

当前入口包括通用 `configs/fp8.py` 的 `IluFp8Config` 与 `configs/deepseek/v4_fp8.py` 的 `IluDeepseekV4FP8Config`；V4.1 Config 继承后者的专家分支。两者最终都调用 `backends/fp4/mxfp4_moe.py` 的 `select_mxfp4_method()`，不各自维护 backend 选择逻辑。通用 FP8 Config 识别 `store_dtype` / `expert_dtype="mxfp4"`，两字段冲突时报错。顶层 Config 的 FP8 名称描述混合 checkpoint 的配置体系，本节的专家源权重仍是 MXFP4。

Config 先保留原有分发并解析显式 `ilu_quant`；已匹配 runtime MoE 转换时优先使用该 Method，与非 auto 的 `ilu_compute.moe` 同时指定则拒绝冲突。无显式 MoE compute 配置时，300 上旧 `torch` / `ixformer_native` 选择继续走原 MXFP4 实现，不启用本文的统一 Loader；旧 compressed-tensors 预转换 checkpoint 也保留独立 Config/Method，不冒充原始 MXFP4。

图 2：以 MXFP4 为例的 MoE 适配流程

![以 MXFP4 为例的 MoE 适配流程](images/moe-mxfp4-adaptation.png)

[可编辑 SVG 源图](images/moe-mxfp4-adaptation.svg) · [Mermaid 源码](images/moe-mxfp4-adaptation.mmd)

加载阶段先确定计算 backend 和 `MoEWeightSpec`，随后按目标格式分配最终参数。checkpoint 读取器先扫描元数据，通过现有模型加载逻辑将延迟来源路由到参数回调；公共会话检查 weight/scale 配对后，再连续读取每一对源分片。量化 Loader 完成切片和转换，直接写入最终参数。加载完成后，Method 校验运行权重并绑定计算实例。

forward 只消费准备好的运行权重。改变常驻权重格式的 backend 选择发生在加载前；兼容同一存储契约的 GEMV/GEMM 调度可以保留在执行阶段。

图中的 Standard/Batched 分支表示适配接口。当前模型使用哪一条通信调用链，见第 5 节，不能由图中的接口存在性推断所有通信组合均已接通。

---

## 3. Method 与 Loader 的职责

### 3.1 Method 组织生命周期

`quantization/backends/fp4/mxfp4_moe.py` 包含公共 mode 选择策略和 `IluMxfp4MoEMethod`：

1. **选择目标**：`select_mxfp4_compute()` 将架构和公共 mode 映射为 backend 名称与布局。
2. **创建参数**：`create_weights()` 检查层配置，取得目标 `WeightSpec`，构造 Loader 并委托其注册参数与加载回调。
3. **完成绑定**：`process_weights_after_loading()` 调用 Loader 的 `finalize()`，再通过 backend 校验权重，保存运行权重和计算实例。
4. **连接执行**：`apply()` 调用 `run_standard()`；`select_gemm_impl()` 根据 Prepare/Finalize 创建对应 Experts adapter。

runner 的输出缩放信息可能在创建参数后才写入层，因此最终计算配置在加载完成时绑定。Method 不应把模型构造早期的临时配置误当成最终运行状态。

### 3.2 Loader、会话与读取器的分工

`quantization/common/mxfp4_loader.py` 中每层一个 `Mxfp4WeightLoader`：

- 按目标 `MoEWeightSpec` 分配最终 `w13_weight`、`w2_weight` 和 scale，不先分配整层 MXFP4 源参数。
- 注册带 `mxfp4_loader` 标记的社区签名 `weight_loader` 回调。
- 用层已有的 expert 映射把全局 expert ID 映射为本地 ID；非本 rank 返回未加载。
- 校验 projection、源 shape/dtype，w1/w3 沿 N 分片，w2 沿 K 分片；TP 切片发生在转换前。
- 把本地源切片搬到公共上传槽，调用 IxFormer converter 直接写最终参数；同步后才把该 `(expert, w1/w2/w3)` 记为已提交。
- `finalize()` 检查所有预期分片已提交、无 pending 和 scale 有限值，返回引用最终参数的 `MoEWeights`。

`model_loader/ilu_default_loader.py` 的 `Mxfp4LoadSession` 负责跨层收集配对、完整性检查、调度读取与上传槽生命周期；`ilu_paired_weights.py` 负责来源描述、延迟视图与有界文件句柄缓存。Layer Loader 不根据文件排列猜配对，读取器也不重新实现模型的权重名称映射。

图 3：Method、公共加载会话与 Layer Loader 的职责

![MXFP4 Method 与 Loader 职责划分](images/moe-method-loader.png)

[可编辑 SVG 源图](images/moe-method-loader.svg) · [Mermaid 源码](images/moe-method-loader.mmd)

### 3.3 阶段一：运行现有模型映射，建立完整配对计划

`IluDefaultModelLoader` 遍历最终参数，从回调发现本 rank 的全部 `Mxfp4WeightLoader`。没有此类回调时完全委托社区 `DefaultModelLoader`；存在时创建并绑定公共会话，包含后来没有收到任何回调的层。模型、VL 包装和 DSpark 不增加配对专用接口。

社区 `load_weights()` 委托原 `model.load_weights()`，**模型映射只执行一次**。它开始消费 iterator 时，插件沿用社区 `_prepare_weights()` 的文件选择、index 去重过滤和下载规则，随后扫描选中 safetensors 的 header。`get_slice()` 提供 shape/dtype；来源描述保存文件和 key，实际字节位置由 safetensors 管理。没有 index 也可扫描社区选中的 safetensors；不依赖特定模型的文件排列。同名 tensor key 跨文件重复、未知 dtype 均报错。

`DeferredCheckpointTensor` 用 meta tensor 保存 shape/dtype，并记录支持的 view、slice、select、transpose、permute、detach、squeeze/unsqueeze 及 dtype 重解释。名称映射、`WeightsMapper`、PP 跳过、EP local expert 和 DSpark 编号继续由原模型处理。普通参数执行真实数据操作时会读取 payload 并写入参数，**阶段一不等于完全不读数据**；显式 `prefetch` 还会调用社区文件预读，尽管 MXFP4 tensor 此时尚不上传或转换。

MXFP4 回调校验目标与来源后，公共会话以 `(Loader, local expert, w1/w2/w3)` 为 key，分别记录 weight 和 scale 的来源/视图及回调信息。重复目标或其他会话的来源被拒绝。模型若在回调前做了不支持的非视图变换，tensor 会按常规操作物化，MXFP4 回调随即拒绝丢失延迟来源的输入；不能改为缓存所有已物化 tensor 来规避契约。

模型映射结束后，`session.finish()` 遍历所有已绑定 Loader、所有本地 expert 和 w1/w2/w3 的预期集合，检查缺半对、整对缺失、整层缺失。配对登记成功只表示来源已路由；不是数据已经转换或参数已经完整。

### 3.4 阶段二：成对读取，跨层复用固定上传槽

全部配对检查通过后，每个 device 按本次所有层中最大的本地投影 staging 大小分配一个 `Mxfp4UploadBuffer`：CUDA 路径为固定 pinned CPU 缓冲加固定 GPU 缓冲。单个本地 N×K 投影需要 `N*K/2 + N*K/32` 字节，分别保存 packed weight 与 E8M0 scale 字节；按单个 w1/w2/w3 计算，不是整层融合 w13 的双倍容量。

会话按配对登记顺序处理各对，**每对固定先 weight 后 scale**，不再依赖 checkpoint 的 key/file 顺序或 `max_pending_shards=8`：

1. `TensorReference.read()` 读取 weight 并重放记录的视图；Layer Loader 校验元数据、执行 TP 切片，直接 copy 到固定 CPU weight 区。
2. 用同一路径读取 scale，重放视图和 TP 切片，copy 到同一槽的 scale 区；第一份来源 tensor 已消费，保留的是槽中的数据。
3. `copy_(..., non_blocking=True)` 上传这对使用的字节区，调用现有 IxFormer `mxfp4_to_w4a8` 或 `mxfp4_to_fp4fp8` 原生 converter，直接写最终 weight/scale 参数切片。
4. 同步该 device 的 **current CUDA stream**，确认异步上传及转换完成，清除槽占用与 pending，再登记本对已提交；之后才允许处理下一对。

这是一槽顺序协议，当前没有读盘/上传/转换流水并发。同步同时保护 pinned CPU 源和 GPU 输入，不能只等待转换结果而提前覆写 host 槽；这里也不使用 `empty_cache()` 或全设备 synchronize。固定容量约束仅覆盖本路径的上传槽，不代表普通参数、converter 内部或框架没有其他临时分配。

payload 读取器默认缓存最多 16 个 safetensors 句柄，LRU 淘汰；同文件多 tensor 可复用句柄，跨越更多文件时可能重新打开。header 扫描单独打开并关闭文件，因此不能宣称每个文件只打开一次。日志记录 pairs/opens/reads/staging_bytes；opens/reads 是 payload reader 的计数，不包含 header 扫描和文件 prefetch。

### 3.5 实际加载流程与异常清理

图 6：通用映射、完整配对与固定上传槽（按实际调用顺序）

![MXFP4 两阶段配对加载与异常清理](images/moe-paired-loading.png)

[可编辑 SVG 源图](images/moe-paired-loading.svg) · [Mermaid 源码](images/moe-paired-loading.mmd)

策略检查与 `bind()` 的 fresh-state 检查在 `try/finally` 之前完成；拒绝时尚未读取 payload 或分配上传槽。会话的 `try/finally` 覆盖社区加载、源路由与 `finish()`。无论成功或失败，`close()` 都尝试关闭所有读取句柄，逐槽同步 current stream 并清除 CPU/GPU tensor 引用，解除所有 Loader 的会话/槽/pending 引用。转换、读取、校验或清理失败时，本次模型加载失败；未完成的会话及清理异常会把所有绑定 Loader 标为失败。转换可能已写部分最终参数，失败不会回滚这些参数，不能重用该模型继续推理。

已有加载异常时，额外的清理异常只记录日志，保留原异常；仅清理失败时抛出清理异常。清除上传槽引用不等于强制把 allocator reserved 显存还给驱动。图中的正常关闭和异常关闭都包含槽同步；不是只有“转换正常完成”的路径才保护 buffer 生命周期。

会话成功关闭后，框架 `load_model()` 才进入量化后处理。Method 调用 Layer Loader 的 `finalize()`，按 runner 此时已安装的输出缩放构造 compute，校验 WeightSpec/shape/dtype/device/连续性并绑定运行权重。此阶段异常直接终止初始化；会话已关闭，不再次进入其清理路径。通信适配器必须在绑定后创建，完成转换本身不等于整个模型已可推理。

### 3.6 当前边界与扩展条件

此路径接受单 checkpoint 的顺序 safetensors lazy/default 或 prefetch 策略；拒绝多线程文件加载、eager/torchao、secondary weights。loader 注册覆盖 `auto/safetensors/pt/npcache` 的社区入口名称，但**实际选中文件必须是 safetensors**，不能据此宣称已支持 PT/npcache payload。无 MXFP4 回调的模型不受上述新路径限制。

当前统一准入要求无 bias、无额外维度 padding 的 gated experts、clamped SiLU/SwiGLU、输出端 TopK 加权，以及 hidden/intermediate 满足 WeightSpec 的 256 对齐。预处理运行权重、EPLB 和不相容 expert placement 不属于此新路径支持范围。Method/Loader 只完成一轮创建和 finalize，重复创建或 finalize 拒绝。

换模型不新增模型专用配对接口：先确认该模型原有映射是否保持可追溯的来源和允许的视图、回调是否具有相容的 projection/TP/EP 语义，以及激活与通信是否满足 backend 契约。直接绕过回调写 expert、未知 fused 源布局、加载中提前消费尚未转换的参数都不能仅凭“权重是 MXFP4”判定兼容。

---

## 4. 计算 backend 的公共设计与实现

### 4.1 公共契约

`fused_moe/experts/ilu_compute_backend.py` 定义统一的运行接口：

- **`MoEWeightSpec`**：描述格式标识、布局、weight/scale dtype、scale 分组、码值倍率和对齐要求，并计算参数形状。
- **`MoEWeights`**：引用运行时 w13/w2 权重及 scale，不复制权重，不保存 checkpoint。
- **`MoEComputeConfig`**：保存本层维度、expert 数、EP rank、激活及输出缩放等静态计算信息。
- **`MoEWorkspaceSpec` / `MoEWorkspace`**：描述并管理一次调用期间的临时空间，按需分配；backend 不把这些空间作为全局或跨层缓存。
- **`MoEComputeBackend`**：提供层配置与权重校验，约定 `run_standard()` 和 `run_batched()`。
- **注册与工厂**：`register_moe_backend` 注册具体类，`select_moe_backend` 返回选择结果，`make_moe_compute` 创建本层实例。

backend 的注册名、架构和布局声明决定其可选范围；注册不等于自动完成所有外部库与原生算子的运行验证。自动选择使用明确的 priority，避免依赖模块导入顺序。MXFP4 的公共 mode 还会在 Method 层进一步限定可选组合。

### 4.2 以 IxMM FP4FP8 为例

`ilu_ixmm_fp4fp8.py` 的 `IxMMFP4FP8Backend` 继承公共基类，并使用 `@register_moe_backend` 注册。它声明支持 300（`ivcore30`）和 `fp4fp8` 布局，消费 uint8 packed FP4 权重、float32 scale，以及 FP8 激活。

对于单个投影的逻辑尺寸 N×K，运行权重形状为 `[E, K/2, N]`，scale 为 `[E, K/256, N]`。这里 E 是本地 expert 数；w13 的 N 为两倍 intermediate，w2 的 N 为 hidden。scale 的含义必须与转换算子和计算算子一致。

- **`run_standard()`**：根据 TopK/expert_map 重排 token 并量化，依次执行 W13、带 limit 的激活与再量化、W2，最后完成本地 TopK 加权归约。
- **`run_batched()`**：接收按 expert 分组的输入和有效 token 数，执行同样的两次投影与激活，写出未加权的 expert 结果。
- 两条路径通过 IxFormer facade 调用底层算子；`group_gemm_fp4(..., backend="ixmm")` 是该实现的 GEMM 调用入口。

图 4：公共 backend 契约与具体实现

![计算 backend 的公共设计与实现方式](images/moe-compute-backend.png)

[可编辑 SVG 源图](images/moe-compute-backend.svg) · [Mermaid 源码](images/moe-compute-backend.mmd)

仅凭 TN/NN 名称不能判断权重能否跨 backend 复用。以 W4A8 为例，IxInfer v2 按 INT4 码值乘 18 解释权重，IxMM 直接消费码值，因此对应转换把倍率计入 IxMM 的 scale。即使某些布局具有相同字节排列，也必须比较完整 `WeightSpec`。

对本地 N×K 投影，IxInfer TN/v2 使用 INT8 `[E,N,K/2]`，NN/v2 使用 INT8 `[E,K,N/2]`，两者 scale 为 FP32 `[E,1,N]`；IxMM W4A8 NN 的形状相同但 code/scale 语义不同。IxMM FP4FP8 使用 UINT8 `[E,K/2,N]` 和 FP32 `[E,K/256,N]`。`w13` 的 gate/up 分别写入两个 N 区间，w2 为 down 投影。Loader 根据完整 `format_id` 和 spec 选择 converter，compute 不持有整层对象或原始 checkpoint。

MXFP4 到 W4A8 包含数值重量化，不能当成纯转置；FP4FP8 运行格式也有自己的 scale 分组约定。切换计算模式时，精度与性能应分别比较。

---

## 5. Standard 与 Batched 通信适配

`ilu_compute_experts.py` 实现 vLLM 的 `FusedMoEExpertsModular` 协议。`bind_compute_experts()` 读取 `prepare_finalize.activation_format`，把已选定的 backend 和已加载权重交给相应适配器。选择依据是输入布局，而非直接根据 prefill/decode 标签。

图 5：Standard/Batched 协议如何连接通信与计算

![Standard 与 Batched 适配及通信计算流程](images/moe-communication-adapters.png)

[可编辑 SVG 源图](images/moe-communication-adapters.svg) · [Mermaid 源码](images/moe-communication-adapters.mmd)

### 5.1 适配器继承与职责边界

`IluStandardComputeExperts` 与 `IluBatchedComputeExperts` 并列继承 `_IluComputeExperts`，薄基类再实现社区 `FusedMoEExpertsModular` 接口。基类组合 backend 与 `MoEWeights` 引用，保存本地 expert 数，并共享设备能力声明、激活与问题尺寸等协议信息；输入布局、workspace 输出形状、计算入口和归约策略由各自的适配器定义。

Standard 与 Batched 的输入形状和输出语义不同，不能互相替代。让 Standard 继承 Batched 会使 token 布局路径继承 expert-major 路径的默认行为，尤其容易混淆有效 token 数、TopK 归约及输出缩放的归属。并列继承保留公共绑定逻辑，同时让每个适配器完整声明自己的协议。`bind_compute_experts()` 负责按输入格式构造适配器，不改变已确定的权重格式或计算 backend。

```mermaid
classDiagram
    FusedMoEExpertsModular <|-- _IluComputeExperts
    _IluComputeExperts <|-- IluStandardComputeExperts
    _IluComputeExperts <|-- IluBatchedComputeExperts
    _IluComputeExperts --> MoEComputeBackend : 组合
    _IluComputeExperts --> MoEWeights : 引用
```

### 5.2 两种协议的计算责任

**Standard** 传入 token 布局的浮点输入、TopK 和 expert_map。适配器要求 Prepare 暂不量化输入，由 backend 自己重排和量化。`run_standard()` 返回已完成本地 TopK 加权归约的 `[T,H]`，因此适配器提供 `TopKWeightAndReduceNoOP`，防止重复加权。NoOP 不会取消后续所需的跨 rank 通信。

**Batched** 传入 `[E,capacity,H]` 布局的输入、激活 scale 和各 expert 的有效 token 数。`capacity` 是分配容量，计算和 combine 必须依据有效数量处理数据。`run_batched()` 写入未加权的 expert 输出，适配器提供 `TopKWeightAndReduceDelegate`，由 Finalize/Combine 完成 TopK 加权归约。

两种入口都把调用方的 output 传给 backend。Standard 的 runner 输出缩放在 backend 中处理，因此其 `runner_scaling_factor` 固定为 `1.0`；Batched 通过 `runner_scaling_factor` 交给外层调用者，在 fused combine 中处理或在普通 combine 后处理。TopK、输出缩放和 shared residual 的责任不能在两层重复。

### 5.3 当前实际通信接线

协议适配能力与某个模型的完整通信入口需要分别看待。当前 MXFP4 接线为：

- **普通调用与 AG/RS**：通过 `IluMxfp4MoEMethod.apply() → backend.run_standard()` 执行。DP+EP 的聚合与分发由 runner 或模型的 Hybrid EP 入口负责；当前链路不需要先构造 `IluStandardComputeExperts`。
- **Hybrid LL decode**：以 AG/RS 为配置基础，通过 decode override 创建 `DeepEPLLPrepareAndFinalize`，调用 Method 的 `select_gemm_impl()` 绑定 `IluBatchedComputeExperts`，再执行 `backend.run_batched()`。
- **Standard adapter**：提供给采用 Standard Prepare/Finalize 的 modular kernel 使用。它是已实现的适配接口，不等于 MXFP4 已自动接通所有 Standard 通信 backend。

现有 DeepSeek 模型接入中，W4A8 backend 通过 `prefer_hybrid_ep` 进入 Hybrid EP 入口；FP4FP8 默认沿用 runner，显式启用 LL decode override 时才在全 DP rank 均为纯 decode 的步骤切换。新模型复用 Method 时，仍需接通对应模型/runner 入口。

当前 Method 保持 `moe_kernel=None`，不构造社区内部通信 kernel。因此不能把直接配置 `deepep_low_latency` 或 `deepep_high_throughput` 与上述 Hybrid LL decode 入口视为等价。

---

## 6. 配置与扩展方法

### 6.1 选择计算模式

执行偏好放在 `additional_config.ilu_compute.moe.mode`，保持 checkpoint 的量化身份不变。已有 `additional-config` 时，合并到同一个 JSON 对象。

例如，在 150 上选择 IxInfer TN：

```bash
--additional-config '{"ilu_compute":{"schema_version":1,"moe":{"mode":"ixinfer_w4a8_tn_v2"}}}'
```

在 300 上选择 IxMM W4A8 NN：

```bash
--additional-config '{"ilu_compute":{"schema_version":1,"moe":{"mode":"ixmm_w4a8_nn"}}}'
```

省略 mode 或选择 `auto` 时，150 默认 IxInfer W4A8 NN，300 默认 IxMM FP4FP8。显式 mode 必须符合当前架构的公共列表；不支持的组合报错。

为兼容已有启动方式，在未提供 MoE compute 配置时，150 的旧 `VLLM_ILU_MOE_RUNTIME_LAYOUT` 可以覆盖布局默认值。显式 mode 优先。本文默认项均指未作这类覆盖的行为。

### 6.2 启用 Hybrid LL decode

在已接入 Hybrid EP 的模型中，配置 DP>1、EP，并以 AG/RS 作为基础通信。LL/NVSM decode 的关键选项如下：

```bash
export VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency
export VLLM_ILU_DISABLE_DEEPEP=1

# 与模型、TP/DP 和容量配置一起使用
--enable-expert-parallel --all2all-backend allgather_reducescatter
```

这里的 `VLLM_ILU_DISABLE_DEEPEP=1` 选择 IxFormer NVSM 通信实现，不表示关闭 EP。它不替代可用的原生通信依赖，也不改变全 DP rank 共同选择通信路径的要求。混合批次和 prefill 保持基础通信。

### 6.3 新增源格式或计算 backend

新增源格式时，先确定 checkpoint 参数和 scale 的语义，再实现对应 Method/Loader；能转换为已有 `WeightSpec` 时，复用现有 backend 和 Experts adapter。

新增计算 backend 时：

1. 继承 `MoEComputeBackend`，声明名称、架构/布局、权重契约与必要限制。
2. 实现 Standard/Batched 中所承诺的入口；不支持的入口应明确拒绝。
3. 用 `@register_moe_backend` 注册，并加入 `_load_builtin_backends()` 的显式导入列表。
4. 为源格式增加匹配完整 `WeightSpec` 的 Loader。MXFP4 公共 mode 需要同步更新 `_MXFP4_MODES`，注册 backend 本身不会自动开放用户配置。
5. 对接适配器并验证真实调用链，再在支持矩阵加入该组合。无需让计算 backend 反向依赖源格式 Method。

新增通信协议时，先明确输入布局、有效 token 元数据及归约责任，再实现独立适配器。公共基类只承载确实相同的绑定信息，不能为了复用少量代码而继承另一种输入协议的默认行为。

---

## 7. 文件职责与验证入口

### 7.1 配置、加载与计算

以下路径均相对 `vllm-iluvatar` 仓库根目录。量化目录承接 main 的 `configs / backends / common / conversions / ilu_runtime` 职责拆分，去掉 `ivcore30/` 目录层级；`capabilities.py`、公开 compute 选择、显式 runtime 工厂及 backend 的架构检查仍保留。注册由 `quantization/_registration.py` 显式导入嵌套模块，backend/runtime method 先于 checkpoint Config。

- `vllm_iluvatar/config/ilu_compute.py`：解析与校验 compute 配置结构。
- `vllm_iluvatar/model_executor/layers/quantization/configs/fp8.py`、`configs/deepseek/v4_fp8.py`：通用与模型 checkpoint Config 的 MXFP4 分发，V4.1 继承后者；共享 `select_mxfp4_method()`。
- `vllm_iluvatar/model_executor/layers/quantization/backends/fp4/mxfp4_moe.py`：公共 mode、Method 生命周期和执行绑定。
- `vllm_iluvatar/model_executor/model_loader/ilu_default_loader.py`：社区加载器扩展、实际路由配对计划与上传会话生命周期。
- `vllm_iluvatar/model_executor/model_loader/ilu_paired_weights.py`：源元数据、延迟 tensor 及有界文件句柄缓存。
- `vllm_iluvatar/model_executor/layers/quantization/common/mxfp4_loader.py`：分片、固定上传缓冲、转换写入与加载完整性。
- `vllm_iluvatar/model_executor/layers/fused_moe/experts/ilu_compute_backend.py`：运行权重契约、backend 基类、registry 与工厂。
- 同目录 `ilu_ixmm_fp4fp8.py`、`ilu_ixinfer_w4a8.py`、`ilu_ixmm_w4a8.py`：具体计算实现。
- 同目录 `ilu_compute_experts.py`：Standard/Batched 协议适配。

### 7.2 通信接线

- `vllm_iluvatar/models/deepseek_v4/fp8_model.py`：模型侧选择 runner 或 Hybrid EP 入口。
- `vllm_iluvatar/model_executor/layers/fused_moe/ilu_fused_moe_ep.py`：AG/RS、LL decode kernel 组装、通信与计算串接、输出缩放。
- 同目录 `prepare_finalize/ilu_deepep_ll.py`：LL 的 dispatch/combine、激活格式与有效 token 元数据。
- 同目录 `runner/ilu_moe_runner.py`：普通 runner 的 dispatch/combine。
- `vllm_iluvatar/distributed/device_communicators/all2all.py`：通信 manager。
- `docs/guides/5-how-to-use-dp-ep.md`：并行配置及 Hybrid EP 的使用约束。

### 7.3 验证与图源维护

在配套的 vLLM、插件和 IxFormer 开发环境中，按改动选择相关测试：

```bash
python -m pytest -q \
  tests/model_executor/layers/quantization/test_ilu_compute_config.py \
  tests/model_executor/layers/quantization/test_ilu_moe_backend.py \
  tests/model_executor/layers/quantization/test_ilu_mxfp4_loader.py \
  tests/model_executor/test_ilu_paired_weights.py \
  tests/model_executor/layers/quantization/test_ilu_fp8_mxfp4_config.py \
  tests/model_executor/layers/quantization/test_ilu_moe_compute_device.py \
  tests/model_executor/models/test_ilu_deepseek_v4_fp8_ep.py
```

配置测试检查公共 mode 与架构默认；Loader 测试检查分片到达顺序、TP/EP 映射、最终参数与失败状态；配对测试检查散布在跨文件的 scale、整对/整层缺失、一次原映射、DSpark 路由、普通参数物化及异常清理；通用 Config 测试检查 MXFP4 识别和共享分发；backend 测试检查运行权重、计算和输出协议，包括 Standard/Batched 的输出复用、workspace 容量、有效 token 数及非 1 输出缩放；模型接线测试检查实际分支选择。GPU 数值、通信、Graph replay 和模型评测分别回答不同问题，不能互相替代。

改变权重契约时，应联动 Loader 与计算测试；改变通信适配时，应覆盖非 1 的 TopK/输出缩放、有效 token 数、空输入，以及所声明的通信路径。用独立数值参考检查数学语义，用新旧实现对比检查迁移一致性。

PNG 用于文档展示，SVG 保留可编辑图形，MMD 保留流程图源码。图 1～6 的图源位于 `images/`；调用关系变化时应同步更新正文与对应图源。

---

## 8. 支持矩阵

本表列出已接入统一 compute backend 的 **MoE Method** 及其计算、通信组合。当前仅包含 `IluMxfp4MoEMethod`，处理原始 MXFP4 专家权重，并在加载时转换为所选 backend 的运行格式。旧预转换 W4A8 checkpoint 使用已有 Config/Method，不因复用部分算子就归入本表；新增其他源格式 Method 时，应另列其适配范围。

![MoE Method、计算 backend 与通信支持矩阵](images/moe-support-matrix.png)

[可编辑 HTML 源表](images/moe-support-matrix.html)

**✓ 支持；✗ 不支持；— 未验证。** “默认”指架构的计算 mode 默认值，不代表自动启用某种通信方式。支持范围以 DeepSeek V4.1 的实际接线与验证组合为依据，其他模型复用仍需满足第 5 节的接口契约；性能与精度结果另见测试报告。

通信接口及适用条件：

- **AG/RS**：配置 `allgather_reducescatter`，本地专家计算经 `IluMxfp4MoEMethod.apply() → backend.run_standard()`。ivcore11 的已验证组合在 Hybrid 的 Prefill／混合步骤使用该路径；ivcore30 的已验证组合在 Prefill 与 Decode 均使用该路径。
- **Hybrid LL / NVSM**：基础通信仍为 AG/RS，显式配置 `VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency` 与 `VLLM_ILU_DISABLE_DEEPEP=1`。所有 DP rank 均为纯 Decode 时，经 `IluBatchedComputeExperts → backend.run_batched()` 使用 IxFormer NVSM dispatch/combine；其他步骤保持基础通信。这不包含原生 DeepEP LL/HT 或直接将 LL 配为基础 all2all backend。
- **未验证的组合**：ivcore30 的两种 mode 已覆盖 AG/RS；已有 backend/adapter 接口不足以构成 LL/NVSM 的整模支持承诺，不将“未验证”写成“不支持”。
- **适配器边界**：当前 AG/RS 接线直接调用 Method，不经过 `IluStandardComputeExperts`。后者提供 Standard modular 协议接口，其独立通信接线不能由上述 AG/RS 结论代替。
- **并行边界**：PP 属于模型并行拓扑，不是 MoE all2all backend。未启用 EP 时，本地专家同样通过 `apply → run_standard` 执行；P/D KV 传输也不属于本表的专家通信。

公共约束：

- Hybrid 自定义通信入口要求 DP>1、EP、PCP=1；LL hidden size 当前限定为 2048、2560、3072、4096、5120、6144、7168、8192，并需配置足够的 token 容量。
- 当前 MXFP4 Method 不支持 EPLB。W4A8 要求连续 expert placement；FP4FP8 backend 可使用 expert_map。
- 底层 registry 中的 ivcore30/IxInfer TN 不属于当前 MXFP4 公共 mode，不列入本表。

维护矩阵时同步更新 HTML 源表与 PNG，保证 Method、mode、通信状态及默认标注一致。
