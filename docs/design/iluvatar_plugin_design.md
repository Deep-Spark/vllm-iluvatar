# vllm-iluvatar 插件设计文档

本文档描述 `vllm-iluvatar` 插件包的当前设计。该包是一个**独立的 Python 包**，基于 vLLM 官方 plugin 机制接入 Iluvatar CoreX GPU（corex + ixformer），在**不修改社区 vLLM 源码**的前提下提供平台适配、通信器、算子覆盖与环境兼容能力。

---

## 1. 概览与目录结构

设计目标：

- **零社区源码改动**：所有适配都通过 vLLM 官方扩展点或运行期挂接完成，社区文件保持与上游字节一致。
- **以 Python 为主 + 一个原生扩展**：核心适配为纯 Python；另含一个原生扩展 `vllm_iluvatar._C`，提供与社区签名一致的 `torch.ops._C` 算子（`rms_norm`、`rotary_embedding`、`weak_ref_tensor`、`get_cuda_view_from_cpu_tensor`）及插件内部算子（`apply_rotary_emb`），见第 9 章。torch / ixformer 仍由 corex 环境提供、不经 pip 安装；该扩展依赖 CMake + CUDA/ixformer 工具链（Iluvatar 目标机），构建时始终编译。
- **硬件强约束**：本包面向 Iluvatar 硬件，缺少 `ixformer` 时直接报错，而非静默回落到 CUDA 平台。

目录结构（**目录级**；每个子目录的文件细节见对应章节，避免此处随功能增多而膨胀）：

```text
vllm-iluvatar/
├── pyproject.toml          # 包元数据 + 两个 vLLM entry point
├── setup.py                # 构建入口 + 启动期 .pth 注入 + 原生 _C 扩展构建（见第 3、9 章）
├── .pre-commit-config.yaml # 提交检查（见第 12 章）
├── CMakeLists.txt          # 原生 _C 扩展的 CMake 构建（见第 9 章）
├── cmake/                  # FindIXFORMER.cmake：定位已安装 ixformer 包（见第 9 章）
├── csrc/                   # 原生 _C 扩展源码（community/ + plugin/ 分目录，见第 9 章）
└── vllm_iluvatar/
    ├── __init__.py         # 插件入口（register_platform / register_general）+ 早期 shim 加载器
    ├── _custom_ops.py / envs.py / version.py # 算子实现 / 环境变量 / 版本（对齐上游）
    ├── config/             # speculative 等配置 override（见第 4 章）
    ├── platforms/          # IluPlatform：Iluvatar OOT 平台 + attention backend 注册（见 §2.3、第 7 章）
    ├── distributed/        # 镜像 vllm/distributed/：通信器 + KV connector + general patch 落点（见 §2.4、§2.7、第 4 章）
    ├── kernels/            # IR provider：iluvatar provider 实现（见第 10 章）
    ├── lora/               # LoRA punica wrapper override（见第 4 章）
    ├── model_executor/     # 镜像 vllm/model_executor/：算子 / 量化 / 模型 / MoE / kernel OOT 实现（见第 5、6、8 章）
    ├── v1/                 # 镜像 vllm/v1/：attention backend / executor / spec_decode override（见第 4、7 章）
    ├── ilu_patch/          # 运行期挂接层：general patch 编排 + import hook + 模块替身 shim（import_shims/，见第 3、4 章）
    ├── ilu_register/       # OOT 注册编排层：算子 register_oot + 量化 register_quantization_config + 模型 register_model（见第 5、6、8 章）
    └── ilu_util/           # 公共工具：签名 / 前缀兼容校验 + 原生算子加载器 torch_ops_c.py（见第 9 章）
```

分层原则：**实现**（`_custom_ops.py`、`distributed/*.py` 与 `model_executor/*.py` 的 override、`ilu_communicator.py`、`envs.py`）与**接线**分离。接线层只负责把实现挂接到 vLLM，不承载重逻辑。接线分两类：运行期 monkey patch 由 `ilu_patch/` 编排（general patch）；官方 OOT 注册（`register_oot` / `register_model`）由 `ilu_register/` 编排，二者相互独立。

---

## 2. 整体设计思路

### 2.1 插件接入：两个 entry point

`pyproject.toml` 声明两个 vLLM 官方分组的 entry point：

```toml
[project.entry-points."vllm.platform_plugins"]
iluvatar = "vllm_iluvatar:register_platform"

[project.entry-points."vllm.general_plugins"]
iluvatar = "vllm_iluvatar:register_general"
```

- `register_platform()`：vLLM 在首次解析 `current_platform` 时调用，返回平台类路径字符串。
- `register_general()`：vLLM 在 `load_general_plugins()`（所有模块初始化后）调用，作为 late phase 入口，依次执行 `apply_general_patches()`（general patch，见第 4 章）、`apply_oot_registrations()`（OOT 注册，见第 5、6 章）与 `apply_kv_transfer_registrations()`（KV connector 注册，见 §2.7）。

### 2.2 十一条接入路径（按生效时机划分）

插件核心适配分为十一条路径，在不同阶段生效：

| 适配对象 | 机制 | 生效时机 |
| --- | --- | --- |
| 平台 / 通信器 | 官方 entry point + `get_device_communicator_cls()` 字符串解析 | `current_platform` 首次访问 |
| Attention backend 覆盖 | `register_backend` 覆盖官方 `AttentionBackendEnum` 成员路径（见第 7 章） | `register_platform()` / `get_attn_backend_cls()` 时 |
| 算子覆盖（`vllm._custom_ops`） | 自研 import hook（meta_path splat） | 目标模块 import 那一刻 |
| 早期兼容 shim（两类：属性级 compat shim + 模块替身 shim `import_shims/`，见第 3 章） | `.pth`（worker）+ `register_platform()`（主进程） | 深层 `import vllm` 之前 |
| `torch.ops._C` 原生算子 | 自研 C++ 扩展（`TORCH_LIBRARY` 注册 `_C` 命名空间，见第 9 章） | 早期 `import vllm_iluvatar._C` 时（早于社区首次调用） |
| IR provider 注册 / 优先级 | `IluPlatform.import_ir_kernels` 注册 `iluvatar` provider + `get_default_ir_op_priority` 设优先级（见第 10 章） | `VllmConfig` 初始化与 worker 启动时 |
| 分布式 function 级覆盖 | general patch（`register_general()` 自动发现并应用，见第 4 章） | vLLM 初始化完成后 |
| CustomOp 算子 OOT 替换 | `CustomOp.register_oot`（`register_general` 自动发现，见第 5 章） | 对应 layer 实例化时 |
| 量化方法注册 | `register_quantization_config` 装饰器（`_OOT_OP_PACKAGES` import 触发，见第 8 章） | `register_general` 后、量化探测时 |
| 模型 OOT 替换 | `ModelRegistry.register_model`（`register_model()` 钩子，见第 6 章） | 模型加载时 |
| KV connector 注册 | `KVConnectorFactory.register_connector`（`apply_kv_transfer_registrations()` 显式注册，见 §2.7） | `register_general` 后、创建 KV connector 前 |

```text
vLLM 进程启动
  -> 加载 platform_plugins 并调用 register_platform
  -> 检查 ixformer 可用性
     -> 否: RuntimeError 直接报错
     -> 是: setdefault VLLM_ILUVATAR_ENABLE_STARTUP_PATCH=1
            -> _load_import_compat_patch（应用早期 shim 并注册 import hook）
            -> envs.apply（注入 Iluvatar 环境）
            -> register_attention_backends（覆盖官方 AttentionBackendEnum 路径）
            -> 返回 IluPlatform 类路径
  -> 后续 import vllm._custom_ops 时触发 import hook 覆盖

vLLM 进程启动
  -> 加载 general_plugins 并调用 register_general
  -> apply_general_patches（自动发现并应用 general patch）
  -> apply_oot_registrations（OOT 注册）
     -> 算子 OOT: CustomOp.register_oot
     -> 量化注册: register_quantization_config 装饰器
     -> 模型注册: ModelRegistry.register_model
  -> apply_kv_transfer_registrations（KV connector 注册）
     -> KVConnectorFactory.register_connector("IluNixl*", ...)
```

### 2.3 平台设计：IluPlatform

`platforms/ilu_platform.py` 中，`IluPlatform` 继承社区 `CudaPlatform`，复用全部 NVML 设备查询与 CUDA-like 执行路径，仅覆盖 Iluvatar 需要改写的点：

- `get_device_capability()` 固定返回 `(9, 0)`；`has_device_capability()` 恒为 `True`。
- `import_kernels()` 置空（Iluvatar 不走社区 kernel import）。
- `import_ir_kernels()`：`import vllm.kernels`（保留社区 provider）+ `import vllm_iluvatar.kernels`（注册 `iluvatar` IR provider）（详见第 10 章）。
- `get_default_ir_op_priority()`：eager 下把 `iluvatar` 置于 IR op 优先级首位，inductor 下回落 `native`（详见第 10 章）。
- `opaque_attention_op()`、`use_custom_op_collectives()` 改为 `False`。
- `get_device_communicator_cls()` 返回 `IluCommunicator` 的字符串路径。
- `get_attn_backend_cls()`：先 `register_attention_backends()` 用官方 `register_backend` 覆盖 `FLASH_ATTN` / `FLASHMLA` / `FLASHMLA_SPARSE` / `FLASHMLA_SPARSE_DSV4` 成员的类路径，再按 `use_mla` / `use_sparse` 选官方成员并返回 `get_path()`（详见第 7 章）。
- `check_and_update_config()`：为 OOT 注册的全新架构 `IluDeepseekV4ForCausalLM` 补 `tokenizer_mode="deepseek_v4"`（详见 §6.5）。

平台类在模块首次 import 时由 `_make_iluvatar_platform()` 动态构建。构建过程中会先 stub 掉 Iluvatar 上可能不存在的 `vllm._C` / `vllm._C_stable_libtorch` 扩展，并撤销社区 `cuda.py` 模块级关闭 cudnn sdp 的副作用。社区平台文件本身保持不变。

### 2.4 通信器：IluCommunicator + All2All

`distributed/device_communicators/ilu_communicator.py` 中的 `IluCommunicator` 继承 `CudaCommunicator`，用 `ixformer.distributed`（ixfd）覆盖 `all_reduce` / `all_gather` / `reduce_scatter` / `send` / `recv` / `broadcast` 等通信原语，ixfd 通信组由传入的标准 torch ProcessGroup 内部创建，社区 `parallel_state.py` 无需感知。

MoE 场景的 All2All 由 `all2all.py` 提供 `NaiveAll2AllManager`（基于 all-reduce 的朴素实现）与 `AgRsAll2AllManager`（基于 all-gather/reduce-scatter），由 `IluCommunicator` 按 `all2all_backend` 选择实例化。

平台通过返回**字符串类路径**而非直接 import 的方式暴露通信器，vLLM 在需要时用 `resolve_obj_by_qualname` 动态解析，从而与具体实现解耦。

### 2.5 环境变量：双名注入

`envs.py` 的 `apply()` 做两件事（由 `register_platform()` 调用，此时 `vllm.envs` 已可用）：

1. 用 `os.environ.setdefault` 把若干社区变量的默认值改成 Iluvatar 取值（如 `VLLM_USE_FLASHINFER_SAMPLER=0`）。
2. 向 `vllm.envs.environment_variables` 注册 Iluvatar 专属变量，采用**双名策略**：规范名 `VLLM_ILU_*` 与旧名 `VLLM_*` 指向同一 reader，读取时优先 `VLLM_ILU_*`、其次 `VLLM_*`、最后默认值。既隔离 Iluvatar 命名空间，又保证旧调用点和启动脚本不受影响。

### 2.6 一致性保障：签名校验

`ilu_util/signature.py` 的 `validate_patch_signatures` 在启动期比对「覆盖函数」与「社区原函数」的签名。`_custom_ops` 与 `IluCommunicator` 的覆盖都会经过严格校验（`strict=True`）：若社区升级导致签名变化而插件未跟进，启动会立即 fail-fast，避免运行期隐蔽错误。少数有意偏离社区签名的接口通过显式白名单跳过校验。

### 2.7 KV connector：IluNixlConnector

`distributed/kv_transfer/kv_connector/v1/ilu_nixl/` 镜像上游 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` 的拆分方式，提供 `IluNixlConnector` / `IluNixlPullConnector`。它继承社区 NIXL pull 连接器，复用官方传输、heartbeat 与握手基础设施，只在 Iluvatar 需要的异构 PP/HMA region 映射、PP-aware handshake metadata 和 descriptor 选择上保留 delta。

`distributed/kv_transfer/kv_connector/v1/ilu_nixl_push/` 提供 `IluNixlPushConnector`，用于 Iluvatar NIXL push 传输路径。

注册函数放在 `distributed.kv_transfer` 包内，由 `register_general()` 的第三步 `apply_kv_transfer_registrations()` 调用。它集中使用 vLLM 官方 `KVConnectorFactory.register_connector` 注册 pull / push connector 名称，使 `--kv-transfer-config '{"kv_connector":"IluNixlConnector", ...}'`、`IluNixlPullConnector` 或 `IluNixlPushConnector` 都能解析到插件实现。各 connector 包内不再保留单独的 `nixl_patch.py` 注册入口，避免重复注册。

---

## 3. WheelWithStartupPatch 设计逻辑

本节描述 `setup.py` 中 `BuildPyWithStartupPatch` / `EditableWheelWithStartupPatch` 两个自定义 cmdclass 的设计动机与实现。

### 3.1 为什么需要 .pth

vLLM 以 spawn 方式拉起 worker 子进程。子进程在反序列化 engine 模块时，会触发社区代码在 **import 期**调用某些 torch / 分布式 API（如 `vllm._custom_ops` 中的函数、`torch.distributed._symmetric_memory`）。而：

- `register_platform()` / `general_plugins` 在子进程里运行得**太晚**（甚至不在该进程触发），来不及在这些 import-time 调用前补齐兼容层；
- corex 的 torch / triton 与上游 API 表面并不完全对齐，缺失项会在 import 期直接抛错。

因此需要一个在 **Python 解释器启动期**（早于任何深层 `import vllm`）执行的钩子。`.pth` 文件正是 Python `site` 模块在启动时自动执行的机制。

### 3.2 env-gated：默认惰性

`.pth` 的内容是一行带条件的语句：

```python
import os; os.environ.get('VLLM_ILUVATAR_ENABLE_STARTUP_PATCH') == '1' and __import__('vllm_iluvatar')._load_import_compat_patch()
```

- 默认**完全惰性**：只有当 `VLLM_ILUVATAR_ENABLE_STARTUP_PATCH=1` 时才会真正动作。
- 该环境变量由 `register_platform()` 用 `os.environ.setdefault` 设置，spawn 出的子进程继承它。
- 因此该 `.pth` 只对「已确认走 Iluvatar 平台」的 vLLM 进程树生效，绝不影响机器上任何无关的 `python` 进程。

```text
主进程路径:
  register_platform
    -> setdefault 环境变量=1
    -> _load_import_compat_patch
    -> 应用早期 shim 并注册 import hook

spawn worker 子进程路径:
  解释器启动时由 site 执行 .pth
    -> 检查环境变量是否为 1
       -> 是（继承自主进程）: _load_import_compat_patch -> 应用早期 shim 并注册 import hook
       -> 否: 惰性跳过
```

### 3.3 两条注入路径：为何需要两个 cmdclass

`.pth` 必须落到 site-packages 才会被 `site` 执行，而不同安装方式落盘路径不同，故 `setup.py` 提供两个 cmdclass：

- `BuildPyWithStartupPatch(build_py)`：常规 wheel / sdist 安装路径。在 `build_py` 阶段把 `vllm_iluvatar_startup_patch.pth` 写入 `build_lib`，并登记进 `get_outputs`，使其随包安装到 site-packages。
- `EditableWheelWithStartupPatch(editable_wheel)`：`pip install -e .` 走 editable wheel，会**绕过 `build_py` 的产物**。这里包装 editable strategy，在生成的 editable wheel 里 `writestr` 写入同一个 `.pth`，保证可编辑安装下钩子同样就位。

两者写入的 `.pth` 文件名与内容完全一致（`PTH_NAME` / `PTH_CONTENT` 单一定义）。

### 3.4 加载侧：_load_import_compat_patch

`.pth` 与 `register_platform()` 最终都调用 `vllm_iluvatar._load_import_compat_patch()`（定义于 `__init__.py`）：

- 用 `importlib.util.spec_from_file_location` **直接按文件路径**加载 `ilu_patch/patch_import_compat.py`，而非常规 `import vllm_iluvatar.ilu_patch.patch_import_compat`。这样可在解释器启动早期运行，而**不必 eager import 整个 `ilu_patch` 包**。
- 以一个固定的 `sys.modules` 键去重，保证全进程只执行一次。

`patch_import_compat.py` 被加载时，模块底部直接 `apply()`，依次：

1. `_patch_torch_accelerator()`：把 vLLM 用到的 `torch.accelerator` 子集映射到 `torch.cuda` 等价物。
2. `_stub_symmetric_memory()` + `_disable_unsupported_symmetric_memory()`：为 corex 上不完整的 `torch.distributed._symmetric_memory` 注入 stub，并置 `VLLM_ALLREDUCE_USE_SYMM_MEM=0`。
3. `_install_import_shims()`：调用 `import_shims.install_all()` 向 `sys.modules` 注入**模块替身 shim**（如 `vllm.vllm_flash_attn`），在深层 `import vllm` 之前兜住那些会在 import-time 崩溃 / 拉起 NV 专有代码的整模块（见 §3.6）。
4. `_register_import_hooks()`：`import vllm_iluvatar.ilu_patch.patch_custom_ops`，触发其注册 import hook（见 §3.5 与第 2.2 节）。
5. `_load_native_torch_ops()`：`import vllm_iluvatar._C`，触发原生扩展的 `TORCH_LIBRARY` 静态注册，使 `torch.ops._C.{rms_norm, rotary_embedding, weak_ref_tensor, get_cuda_view_from_cpu_tensor, apply_rotary_emb}` 在社区首次调用之前就绪（见第 9 章）。

`apply()` 自身带 `_applied` 幂等守卫，且整体 `try/except` 包裹——兼容 shim 绝不能让解释器或 vLLM 启动失败。

### 3.5 算子覆盖的 import hook 机制

`ilu_patch/import_hook.py` 实现了一个挂在 `sys.meta_path` 最前端的 finder：当目标模块（如 `vllm._custom_ops`）被 import 时，它先让真实 finder 构建 spec，再包裹 `exec_module`，在目标模块执行完毕（函数已定义）后立即触发回调。`patch_custom_ops.py` 在 import 时调用 `register("vllm._custom_ops", _patch_custom_ops)`；回调把 `_custom_ops.py` 的 `_ILU_OVERRIDES`（覆盖社区同名函数）与 `_ILU_ADDED`（新增函数）逐一 `setattr` 到社区模块，并在此之前完成签名校验。`_custom_ops.py`（及 ixformer）只在回调中惰性 import，使启动保持轻量且不强制平台解析。

### 3.6 模块替身 shim（import_shims）

§3.4 第 1、2 步的 `torch.accelerator` / `_symmetric_memory` 是**属性级 compat shim**——给一个已存在的模块补 / 改属性，inline 在 `patch_import_compat.py` 里。但有时需要替换的是**整个社区模块**：它在 import-time 就会崩溃或拉起 NV 专有代码，补属性救不了。这类**模块替身 shim** 统一归到 `ilu_patch/import_shims/` 子包，与属性级 compat shim 分开维护。

- `import_shims/__init__.py` 的 `install_all()`：`pkgutil.iter_modules` 自动发现本包下每个 `*.py`，读取其 `SHIM_KEY`（要替身的 `sys.modules` 键）与 `build_module()`，若该键不在 `sys.modules` 则注入，并尽量绑定到已 import 的父包属性上。新增替身只需丢一个文件，编排不动。
- 由 `patch_import_compat.apply()` 第 3 步 `_install_import_shims()` 在深层 `import vllm` **之前**调用。

**示例 `import_shims/vllm_flash_attn.py`（fa_utils 的 import-time delta）**：`IluPlatform` 是 cuda-like（`_enum=CUDA`、`is_cuda()=True`，见 §5.4），故社区 `vllm/v1/attention/backends/fa_utils.py` 会进 `if current_platform.is_cuda():` 分支，在 **import-time** 执行 `from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata`；而 corex 上 `vllm/vllm_flash_attn/__init__.py` 缺 `_vllm_fa2_C` / `_vllm_fa3_C` 会直接 `raise ImportError`。因此用模块替身在 import 前把 `sys.modules["vllm.vllm_flash_attn"]` 换成暴露 ixformer 版 `flash_attn_varlen_func` + `get_scheduler_metadata`(stub) 的替身——import-hook 的「import 后 splat」此处来不及（整模块在 import 那一刻就崩）。

> 对比：成熟插件路线若用 OOT 平台（`is_cuda()=False`）会天然跳过该 cuda 分支、无需 shim。本插件选 cuda-like 路线以全面复用 CUDA 执行路径（见 §5.4），模块替身 shim 即是该路线的必然配套。

---

## 4. 通用补丁（general patch）规范

当官方扩展点无法覆盖、又必须改写社区某个 function/method 行为时，使用 **general patch**：在 `register_general()` 阶段对社区符号做运行期覆盖，实现「零社区源码侵入」。本章给出取舍、文件规则与 register 规则。

### 4.1 .pth 与 general patch 的取舍

二者都是「运行期挂接」机制，区别只在**生效时机**：

| 机制 | 生效时机 | 适用 |
| --- | --- | --- |
| `.pth` 启动期 import hook（第 3 章） | 解释器启动期，早于深层 `import vllm` | 目标符号在 **import-time** 就被调用，general patch 来不及（如 `vllm._custom_ops`） |
| general patch（本章） | `register_general()`，vLLM 初始化完成后 | 目标符号在初始化/运行期才被调用 |

挂接前先按以下优先级判定，**能不 patch 就不 patch**：

```text
挂接决策流程:
  有官方扩展点?
    -> 有: 用官方扩展点（platform plugin / 通信器字符串解析 / CustomOp.register_oot /
           ModelRegistry.register_model / entry point / envs）
    -> 无: 目标符号是否在 import-time 就被调用 / 绑定?
           -> 是（general patch 来不及）:
                整模块在 import 前不可用 / 会崩  -> import_shims 模块替身（§3.6）
                符号在"已 import 的社区模块"里被调用 -> .pth import hook splat（§3.5）
           -> 否（初始化或运行期才调用）: 用 general patch（本章）
                再按"符号消费方式"选改写落点（见下）
```

- **首选官方扩展点**：platform plugin、`get_device_communicator_cls()` 字符串解析、`CustomOp.register_oot`、entry point、`envs` 等都不算 patch，优先走它们。
- **必须运行期挂接时，优先 general patch，尽量避免 .pth**：`.pth` 在全局 site 启动期执行、影响面大、须 env-gate 才安全、调试困难、维护成本高（详见 §3.2）。只有「import-time 调用、general patch 来不及」才用 `.pth` / import_shims。
- **两类 patch 整体都要克制**：它们是运行期等价改写，随上游升级易碎；delta 越小越好，优先 rebind / wrapper，尽量不整方法 copy。
- **general patch 内部按"符号消费方式"选改写落点**（决定 setattr 改哪里、要不要传播）：
  - **类方法** → `setattr(cls, name, ...)`（实例经类解析；OOT 掉包同理）。
  - **模块级自由函数、被 `from module import name` 消费** → `rebind_module_func`（传播改绑，见 §4.3）；只 setattr 目标模块会漏掉已被 `from-import` 拷走的引用。
  - **import-time 绑定 / 调用** → 不属 general patch，回到上面的 import_shims / import hook。

### 4.2 patch 文件规则（对齐上游）

- **镜像树 + 同名**：override 实现文件落在与上游同路径的镜像位置、文件名与上游一致，便于逐文件对照。例：`vllm/distributed/parallel_state.py` → `vllm_iluvatar/distributed/parallel_state.py`。
- **实现 / 接线分离**：override 实现放在上游同名 `xxx.py`（**不含 `register()`**）；`register()` + `_apply()` 接线统一放在 `ilu_patch/general_patches/` 下。实现目录保持上游镜像，接线目录集中管理。
- **无自有实现的 patch 只写接线文件**：纯 rebind / 纯 wrapper（其覆盖逻辑只引用社区符号、无自己的实现体）没有需要镜像的实现，只在 `ilu_patch/general_patches/` 下放一个接线模块，不保留空的实现兄弟文件。例：`patch_stateless_coordinator.py`。
- **整方法 copy 的基线快照**内联在对应接线模块内，以字符串常量（三单引号）保存社区原方法源码，与源码漂移校验放在一处。例：`general_patches/patch_parallel_state.py` 内的 `BROADCAST_TENSOR_DICT`。
- **模块 docstring 固定字段**：`Corresponds to`（上游文件）、`Target`（覆盖符号）、`Baseline`（社区基线 commit）、`Why`、`How`、`Scope`、`Remove`（何时可移除）；整方法 copy 在其接线模块内追加 `Fingerprint`。
- **改动标注**：每处覆盖以 `# [ILU override] <symbol> — delta: …` 标注；整方法 copy 内的实际改动用 `# >>> ILU MODIFICATION >>>` … `# <<< ILU MODIFICATION <<<` 框出，使 delta 在大段 copy 中一眼可见。

### 4.3 register 规则

- **自动发现，无需手工编排**：`ilu_patch/general_patches/` 顶层、定义了 `register()` 的模块会被 `apply_general_patches()` 自动扫描并调用（见 `ilu_patch/__init__.py`）。新增 general patch 只需新增一个接线模块，不必改编排代码。
- **统一形态**：接线模块的 `register()` 只委托 `general_patch.register_once(__name__, _apply, detail=…)`；patch 专属逻辑全部放进 `_apply()`，override 实现引用上游镜像实现模块（见 `ilu_patch/general_patch.py`）。
- **写法 delta 优先**：rebind（符号重绑）> wrapper（包裹委托）> 整方法 copy。只有整方法 copy 才需在 `_apply()` 里先 `check_fingerprint`（源码漂移校验：核对内联基线快照与社区源码是否一致）+ `check_signatures`（核对签名）；delegation 型 wrapper 天然兼容，可免显式校验。
- **传播改绑 `rebind_module_func`（覆盖模块级自由函数时用）**：`general_patch.rebind_module_func(module, name, new_obj)` 先 `setattr` 目标模块，再按对象身份遍历 `sys.modules`，把已 `from module import name` 拷走的旧引用一并替换（解决 from-import 失效；身份匹配避免误伤同名异物）。与 `patch_stateless_coordinator` 那种"类级符号重绑（`setattr` 改类）"的区别：后者改类、实例自然解析到新类；前者改的是模块函数，且必须传播到拷贝引用。
- **lazy import**：`import vllm.*` 与重依赖只在 `_apply()` 内部 import，保证模块在自动发现 import 期不拉起 vllm。
- **幂等**：由 `register_once` 的进程级 `_APPLIED` 集合保证，patch 模块不再各自维护幂等标志。
- **env 开关（经 `vllm.envs` 读取）**：`VLLM_ILU_PATCH_CHECK_SIGNATURE` / `VLLM_ILU_PATCH_CHECK_FINGERPRINT`（默认开）、`VLLM_ILU_PATCH_VERBOSE`（默认关，开启则以 `INFO` 打印应用详情）。这些变量的 reader 与默认值登记在 `envs.py` 的 `_ILU_GUARD_ENV_VARS`，既供 `vllm.envs` 读取，又避免 `validate_environ()` 把它们当未知 `VLLM_` 变量告警。

### 4.4 示例：general patch

general patch 示范 copy / rebind 写法，均经 `general_patch` 脚手架 + `ilu_patch/general_patches/` 自动发现接入。下表「实现 / 接线」列分别给出 override 实现文件与集中接线文件：

| 实现 / 接线 | 覆盖符号 | 写法 | 校验 |
| --- | --- | --- | --- |
| `distributed/parallel_state.py` / `general_patches/patch_parallel_state.py` | `GroupCoordinator.broadcast` | wrapper | 签名 |
| `distributed/parallel_state.py` / `general_patches/patch_parallel_state.py` | `GroupCoordinator.broadcast_tensor_dict` | 整方法 copy（基线快照内联于接线模块） | 源码漂移 + 签名 |
| （无实现）/ `general_patches/patch_stateless_coordinator.py` | `CudaCommunicator` → `IluCommunicator` | 符号重绑 | 签名 |

`create_tcp_store` 由 `ilu_patch/patch_platform_import_hooks.py` 在 platform 阶段注册 post-import hook：目标模块仍由 vLLM 自然导入，导入完成后立即 wrap `create_tcp_store`，避免 `register_platform()` 期间 eager import `vllm.distributed.utils`。

### 4.5 worked example：fa_utils（还原公版 + 插件覆盖全部 delta）

`vllm/v1/attention/backends/fa_utils.py` 是「社区源码还原上游公版 + 插件零侵入覆盖」的样板：把社区该文件还原到上游 baseline（字节一致），改动全部由插件承接。同一文件相对魔改版的三类 delta 恰好分别落到三种机制，正好横跨第 3、4 章：

| delta | 消费方式 | 落到的机制 |
| --- | --- | --- |
| cuda 分支 import 块（绑定 ixformer `flash_attn_varlen_func` / `get_scheduler_metadata`） | **import-time** 绑定（整模块 `vllm.vllm_flash_attn` 在 corex 会 `raise`） | `import_shims/vllm_flash_attn.py` 模块替身（§3.6） |
| `get_flash_attn_version`→`None`、`flash_attn_supports_sinks`→`True` | **运行期**，被 `from fa_utils import` 消费 | general patch + `rebind_module_func`（本章；实现 `v1/attention/backends/fa_utils.py`，接线 `general_patches/patch_fa_utils.py`，`_apply()` 用 `rebind_module_func` + `check_signatures(strict=True)`） |
| 公版无的 re-export `flash_attn_with_kvcache` / `flash_attn_varlen_int8_func` | **import-time** 被 fork 文件 `ilu_flash_attn.py` 经 fa_utils 二次导出消费 | 改**消费方**直接 `from ixformer.contrib.vllm_flash_attn import ...`（正本清源），不让 fa_utils 继续转发 |

要点：

- 两个能力函数是「**有意脱钩的整体替换**」——ixformer flash_attn 无上游 FA 版本派发，故无条件 `return None` / `True`，不跟踪上游逻辑（实现里只保留前缀兼容签名，不做源码漂移校验）。
- 还原公版会**暴露隐藏的 re-export 依赖**：原魔改版把 ixformer 专有符号塞进 fa_utils 的 cuda 分支再被别处 `from fa_utils import`；还原后这些符号在公版不存在。处理范式是把消费方指向符号真身（ixformer），而非在 fa_utils 上重建转发。

---

## 5. 算子 OOT 注册（CustomOp.register_oot）

当社区把某个算子（`RMSNorm` 等）建成 `CustomOp` 时，vLLM 提供官方 OOT 扩展点 `CustomOp.register_oot`：可用一个子类替换社区实现，**无需 monkey patch、无社区源码改动**。本章讲算子这一路径；模型注册见第 6 章，二者同由 `ilu_register/` 编排。

### 5.1 ilu_register 编排概览（第 5、6、8 章共用）

`ilu_register/` 是继 import hook、general patch 之后的**第三种 late phase 机制**，并刻意与 `ilu_patch/` 分开：它走 vLLM **官方扩展点**，挂接的是「替换子类 / 注册表条目」而非「方法体 copy」，因此**不做源码漂移校验**，只做前缀兼容签名校验（§5.5）。

`register_general()` 在 `apply_general_patches()` 之后调用 `apply_oot_registrations()`，它按机制不同分两条独立路径自动发现：

- **算子 / 量化 OOT**（`_register_ops`，本章 + 第 8 章）：`import` `_OOT_OP_PACKAGES`（当前包括 `model_executor.layers`、`layers.fused_moe`、`layers.fused_moe.router`、`layers.mamba.gdn`、`layers.rotary_embedding`、`layers.quantization`）下的每个模块，注册靠模块级装饰器副作用完成（`@<Base>.register_oot` / `@register_quantization_config`），**无钩子**。
- **模型注册**（`_register_models`，第 6 章）：`import` `_OOT_MODEL_PACKAGES`（当前包括 `model_executor.models`、`models.deepseek_v4`、`models.glm_moe_dsa`；多文件模型子包需显式登记，见 §6.5）下的模块后，调用其 `register_model()` 钩子。

```text
register_general
  -> apply_oot_registrations
     -> _register_ops: import 算子包并触发装饰器
        -> CustomOp.register_oot 写入 op_registry_oot
     -> _register_models: import 模型包并调用 register_model 钩子
        -> ModelRegistry.register_model 覆盖同名架构
     -> _validate_oot_overrides: 前缀兼容签名校验
```

新增算子覆盖时，在 `_OOT_OP_PACKAGES` 下放一个带 `@<Base>.register_oot` 的模块即可，无需修改编排代码（量化配置同理，改用 `@register_quantization_config`，见第 8 章）。

### 5.2 算子 discovery（无钩子）

算子包的注册完全靠 import 时的模块级装饰器：`@<Base>.register_oot`（覆盖社区算子）与 `@CustomOp.register("name")`（新增插件自有算子）。`_register_ops` 只负责把包下每个模块 import 进来，触发这些装饰器，本身不调用任何钩子。

### 5.3 register_oot 机制

`CustomOp.register_oot` 把子类登记进全局 `op_registry_oot`，**key 为社区基类的 `__name__`**。社区算子实例化时 `CustomOp.__new__` 会按类名在 `op_registry_oot` 中查找，命中则把待建实例「掉包」为我们的子类——因此调用点（社区代码）无感，拿到的就是 Iluvatar 实现。

### 5.4 dispatch 取舍：为何覆盖 forward_cuda 而非 forward_oot

`CustomOp` 的 `dispatch_forward` 按 `current_platform._enum` 选择 `forward_cuda` / `forward_native` / `forward_oot` 等。`IluPlatform` **继承自 `CudaPlatform`、`_enum` 仍是 `CUDA`**（`is_out_of_tree()` 为 `False`），故 dispatch 选中的是 **`forward_cuda`**，而非名字上更「贴切」的 `forward_oot`。

因此 OOT 子类直接覆盖 `forward_cuda`（及 delta 实际所在的 `forward_native`）。不改成「OOT-typed 平台 + `forward_oot`」是经过权衡的：`IluPlatform` 全面复用 CUDA 执行路径，一旦把 `_enum` 改成 `OOT`，大量 `is_cuda` 判定会失效，需要补一堆补偿 patch，得不偿失。覆盖 `forward_cuda` 与之功能等价且代价最小。

### 5.5 前缀兼容签名校验

OOT 子类是整类替换、没有方法体 copy，故**不做源码漂移校验**，只对「覆盖了社区同名方法」的子类做一道 `validate_oot_override`（`ilu_util/signature.py`）。它是一条**前缀兼容**规则：

1. 覆盖方法的参数个数 ≥ 社区方法。
2. 前 N 个参数（N = 社区参数数）按位置对齐——同名、同序。
3. 多出来的尾部参数必须带默认值（社区调用点不传也能工作）。

这样既能容纳「在尾部追加 Iluvatar 专属可选参数」的有意扩展（如 `IluGemmaRMSNorm.forward_cuda` 的 `fused_int8_quant` / `return_origin_output`），又无需白名单，还能抓出真正的接口漂移。该校验复用 general patch 的开关 `VLLM_ILU_PATCH_CHECK_SIGNATURE` / `VLLM_ILU_PATCH_VERBOSE`。

### 5.6 示例：layernorm.py

`model_executor/layers/layernorm.py`（镜像 `vllm/model_executor/layers/layernorm.py`）示范算子 OOT 的两种形态——覆盖社区算子与新增插件算子：

| 类 | 注册方式 | delta | 校验 |
| --- | --- | --- | --- |
| `RMSNormQK` | `@CustomOp.register("rms_norm_qk")`（**新增**插件自有算子） | 社区无对应；融合 q/k RMSNorm，`forward_cuda` 走 ixformer `rms_norm_qk` | 仅前缀兼容（社区无同名方法可比，跳过） |
| `IluRMSNorm` | `@RMSNorm.register_oot` | 覆盖 `forward_native`，残差路径走 Iluvatar `fused_add_rms_norm`；社区 `forward_cuda` 委托 `forward_native` | 前缀兼容签名 |
| `IluGemmaRMSNorm` | `@GemmaRMSNorm.register_oot` | 覆盖 `forward_native` + `forward_cuda`（2D 走 ixformer `gemma_rms_norm`，尾部新增 `fused_int8_quant` / `return_origin_output`） | 前缀兼容签名（尾部新增带默认值） |

> `IluRMSNorm` 的**无残差**分支委托 `ir.ops.rms_norm(...)`，现 dispatch 到 `iluvatar` IR provider（见第 10 章）；**残差**分支仍走 `vllm._custom_ops.fused_add_rms_norm`。两层 dispatch（OOT 外层 + IR 内层）的完整关系见 §10.5。

---

## 6. 模型注册（ModelRegistry.register_model）

### 6.1 为何模型只能整体注册

vLLM 对模型**没有「按方法 patch」的官方扩展点**，唯一入口是 `ModelRegistry.register_model(arch, "module:Class")`。因此即便只想改一个 `forward`，也必须注册一个新的模型类。我们让该类继承上游模型，把 delta 收敛到最小（见 §6.3），而非整模型 copy。

### 6.2 模型 discovery（register_model() 钩子）

模型包用钩子而非纯装饰器：`_register_models` import `_OOT_MODEL_PACKAGES` 下的模块后，调用其模块级 `register_model()`，内部执行 `ModelRegistry.register_model`。`ModelRegistry` 是字典，**同名架构后注册者覆盖先注册者**；插件在 `register_general` 阶段（早于模型加载）注册，故加载时命中的是我们的类。

> 命名约定：模型模块用 `register_model()` 钩子，与算子 OOT 的「纯装饰器、无钩子」区分开；两者都不同于 general patch 的 `register()`。

### 6.3 子类继承 + 安装期 **class** 交换

为把 delta 收到最小，模型类继承上游模型，在 `super().__init__()` 之后做**安装期 `__class__` 交换**：把已建好的子模块（如各层 `self_attn`）的 `__class__` 改写为融合子类，并按需补建算子属性。`__class__` 重绑只改变解析到的 `forward`，**完整保留已加载的权重与状态**，避免复制整模型代码。

### 6.4 示例：qwen3.py（覆盖社区已有架构）

`model_executor/models/qwen3.py`（镜像 `vllm/model_executor/models/qwen3.py`）把 Qwen3 attention 里分离的 q/k RMSNorm 融合为单个 `RMSNormQK`（见 §5.6）：

- `Qwen3ForCausalLM(_Qwen3ForCausalLM)`：继承上游模型，`__init__` 末尾调 `install_qwen3_attention_fusion(self.model)`。
- `install_qwen3_attention_fusion`：遍历各层，把 `self_attn.__class__` 交换为 `Qwen3Attention`（= `_Qwen3AttentionFusedMixin` + 上游 `Qwen3Attention`，提供融合 `forward`），并用社区 `q_norm` 的 head dim / eps 补建 `qk_norm`(`RMSNormQK`)。
- `register_model()`：把架构 `"Qwen3ForCausalLM"` 指向 `vllm_iluvatar.model_executor.models.qwen3:Qwen3ForCausalLM`，由 `apply_oot_registrations` 调用。

### 6.5 全新架构注册（不覆盖社区）+ tokenizer_mode：deepseek_v4 示例

§6.4 的 qwen3 是**覆盖社区已有架构**（注册同名 `Qwen3ForCausalLM`，后注册者胜）。另一类场景是注册一个**社区不存在的全新架构**，对社区原生模型零副作用。`model_executor/models/deepseek_v4/`（镜像 `vllm/models/deepseek_v4/`）即此类：注册全新 arch `IluDeepseekV4ForCausalLM`，与社区原生 `DeepseekV4ForCausalLM` 并存、互不影响。

- **注册全新 arch**：`model.py` 的 `register_model()` 同时注册主模型与投机解码（MTP）两个全新架构，均为字符串类路径，由 `ModelRegistry` 懒解析：

```python
def register_model() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "IluDeepseekV4ForCausalLM",
        "vllm_iluvatar.model_executor.models.deepseek_v4.model:"
        "IluDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "IluDeepSeekV4MTPModel",
        "vllm_iluvatar.model_executor.models.deepseek_v4.mtp:IluDeepSeekV4MTP",
    )
```

- **多文件模型包：子包须显式登记**：`_iter_package_modules` 只扫顶层模块、**跳过子包**（见 §5.1）。故 deepseek_v4 这种「一个目录多文件（`model.py` / `mtp.py` / `attention.py`）」的模型包，必须把子包路径 `vllm_iluvatar.model_executor.models.deepseek_v4` **显式**加进 `_OOT_MODEL_PACKAGES`，否则其 `register_model()` 不会被调用。单文件模型（如 `qwen3.py`）则由 `model_executor.models` 顶层自动发现，无需额外登记。
- **tokenizer_mode 补齐**：社区 `ModelConfig` 只为社区名 `DeepseekV4ForCausalLM` 自动选 `deepseek_v4` tokenizer；全新 arch 在社区还原后不再命中该映射，`tokenizer_mode` 停在 `"auto"`。由 `IluPlatform.check_and_update_config`（社区在 `config/vllm.py` 的 `VllmConfig.__post_init__` 调用，早于 tokenizer 构建）在 arch 命中时补齐：

```python
@classmethod
def check_and_update_config(cls, vllm_config) -> None:
    super().check_and_update_config(vllm_config)
    mc = getattr(vllm_config, "model_config", None)
    if (
        mc is not None
        and "IluDeepseekV4ForCausalLM" in (mc.architectures or [])
        and mc.tokenizer_mode == "auto"
    ):
        mc.tokenizer_mode = "deepseek_v4"
```

- **时序安全**：`load_general_plugins()`（触发 `register_model`）在 `EngineArgs.__post_init__` 中执行，**早于** `ModelConfig` 解析架构；故加载时全新 arch 已在 `ModelRegistry` 中，架构解析不会失败。

### 6.6 其他已接入模型与新增指南

除 qwen3（§6.4，覆盖社区已有架构）和 deepseek_v4（§6.5，全新 arch）外，当前还接入了以下模型：`hy_v3`、`qwen2_5_vl`、`qwen3_moe`、`qwen3_vl_moe`、`glm_moe_dsa`。它们分别属于上述两类范式之一。

新增模型的具体步骤参见：

- [在 vllm-iluvatar 快速适配社区模型](../guides/2-how-to-quickly-adapt-a-community-model.md)（复用社区 arch，子类化 + 最小覆盖）
- [在 vllm-iluvatar 接入天数优化模型](../guides/3-how-to-add-an-iluvatar-optimized-model.md)（新注册 `IluXXX` arch，模型算法链完全重写）

---

## 7. Attention backend 注册（register_backend）

attention backend 既不是 `CustomOp`、也不是模型，vLLM 为它提供另一个官方扩展点 `register_backend`：在**不改社区 `AttentionBackendEnum` 定义**的前提下，把某个 backend 成员的类路径改指向 Iluvatar 实现。本章核心约束是：**必须复用官方 enum 成员，不能自定义 `ILU_*` 成员，否则触发反查 bug**。

### 7.1 流程

- `register_attention_backends()`（`platforms/ilu_platform.py`）在两处被调用：`register_platform()` 早期一次、`IluPlatform.get_attn_backend_cls` 选择时再保证一次；`_ATTN_BACKENDS_REGISTERED` 保证全进程幂等。
- 它对每个目标 backend 调官方 `register_backend(member, "插件 backend 类路径")`。本次覆盖四个官方成员：`FLASH_ATTN` / `FLASHMLA` / `FLASHMLA_SPARSE` / `FLASHMLA_SPARSE_DSV4`。
- `get_attn_backend_cls` 不再引用任何 `ILU_*`，而是按 `use_mla` / `use_sparse` 选官方成员，返回 `member.get_path()`（已被覆盖 → 解析到 Iluvatar 类）。

### 7.2 机制：register_backend 覆盖类路径

官方 `register_backend(member, class_path)`（`vllm/v1/attention/backends/registry.py`）把 `class_path` 写进进程级字典 `_ATTN_OVERRIDES[member]`。`AttentionBackendEnum.member.get_path()` 优先取 `_ATTN_OVERRIDES` 的覆盖路径，未覆盖才回落到 enum 默认值。因此覆盖后，社区调用点拿到的就是 Iluvatar backend，而**社区 `registry.py` 保持字节不变**。

### 7.3 为何必须复用官方 enum（自定义会有 bug）

社区 `vllm/model_executor/layers/attention/attention.py` 在建层时做一次**按名反查**：

```python
self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
```

`AttentionBackendEnum[name]`（元类 `__getitem__`）按**成员名**查 enum。若走「自定义」路线——给社区 registry 新增 `ILU_FLASH_ATTN` 成员、并让 backend 的 `get_name()` 返回 `"ILU_FLASH_ATTN"`——那么一旦坚持「零社区改动」把 `registry.py` 还原为 baseline，该成员就不存在，反查直接抛：

```text
ValueError: Unknown attention backend: 'ILU_FLASH_ATTN'. Valid options are: FLASH_ATTN, ...
```

正确做法是**借官方成员的壳**：

1. backend 的 `get_name()` 返回**社区名**（如 `IluFlashAttentionBackend.get_name()` → `"FLASH_ATTN"`），反查命中官方成员。
2. 用 `register_backend(FLASH_ATTN, "<Iluvatar backend 类路径>")` 把该官方成员的**类路径**改指向 Iluvatar 实现。

如此 `AttentionBackendEnum["FLASH_ATTN"]` 反查正常、`get_path()` 又落到 Iluvatar 类，两边都对，且 `registry.py` 零改动。

### 7.4 手动指定 backend 的行为

覆盖官方成员后：

- **官方名仍可手动指定**：`--attention-backend FLASH_ATTN`（或 `FLASHMLA` 等）会经 `get_attn_backend_cls` 的 user-selected 分支 `selected_backend.get_path()` → `_ATTN_OVERRIDES` → 解析到 **Iluvatar 实现**。
- **拿不到「社区原版」backend**：`register_backend` 是全局覆盖，CLI 不会调 `clear_override()`；在 Iluvatar 平台上，选官方名得到 Iluvatar 实现正是期望行为。
- **`ILU_*` 自定义名不可用**：对应 enum 成员不存在，按名选择 / 反查都会 `ValueError`。

### 7.5 流程

```text
register_platform / get_attn_backend_cls
  -> register_attention_backends（幂等）
  -> register_backend 写入 _ATTN_OVERRIDES[官方成员] = Iluvatar 类路径

get_attn_backend_cls 选官方成员
  -> member.get_path 读取 _ATTN_OVERRIDES
  -> 解析到 Iluvatar backend 类

社区 attention.py 建层
  -> AttentionBackendEnum[backend.get_name（社区名）]
  -> 命中官方成员，反查成功
```

---

## 8. 量化配置注册（register_quantization_config）

当 checkpoint 使用社区未内置的量化方法时，vLLM 提供官方扩展点 `register_quantization_config`：用装饰器登记一个自定义 `QuantizationConfig` 子类，**零社区源码改动**。

### 8.1 流程（与算子 OOT 同款发现）

量化模块与算子 OOT 共用同一条「import 触发装饰器」发现路径（见 §5）：量化包 `vllm_iluvatar.model_executor.layers.quantization` 已登记进 `_OOT_OP_PACKAGES`，`_register_ops` 把其下模块 import 进来，`@register_quantization_config(...)` 在 import 期生效。**无钩子**。

### 8.2 机制：一次装饰，三处生效

官方 `register_quantization_config(name)`（`vllm/model_executor/layers/quantization/__init__.py`）的装饰器一次完成：

1. **append 到 `QUANTIZATION_METHODS`**：使该方法名可被 `ModelConfig` 的量化探测发现。
2. **append 到 `current_platform.supported_quantization`**：使 `_verify_quantization` 视其为本平台支持、放行。
3. **写入 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`**：`get_quantization_config(name)` 据此返回我们的配置类。

### 8.3 零社区改动 + 自动绕开守卫

社区 `config/model.py` 对「写在 `QuantizationMethods` 字面量里的内置方法」有一道 overrides 探测报错守卫。自定义方法名**不在**该字面量里，因此既无需改 `quantization/__init__.py` 的字面量，也**不触发**该守卫；配合上面三处自动生效，实现真正的零社区改动。

### 8.4 时序

`register_general`（含 `_register_ops`）在 `EngineArgs.__post_init__` 阶段执行，**早于** `ModelConfig._verify_quantization`；故量化探测时，自定义方法已在 `QUANTIZATION_METHODS` / `supported_quantization` 中。

> 注意：量化探测发生在 `ModelConfig.__post_init__`，**早于** `IluPlatform.check_and_update_config`；故量化只能走 `register_quantization_config`，无法像 tokenizer_mode（§6.5）那样在 `check_and_update_config` 里补。

### 8.5 示例：compressed-tensors（IluQuantConfig）

`model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`：

```python
@register_quantization_config("compressed-tensors")
class IluQuantConfig(CompressedTensorsConfig):
    ...
```

插件用 `IluQuantConfig` 替换官方同名配置类。W4A8 / W8A8 的选用基于 checkpoint 的 `quantization_config`（`quant_method` 为 `compressed-tensors`），与模型 arch 解耦：`IluDeepseekV4ForCausalLM`（§6.5）负责架构，`IluQuantConfig` 负责权重量化方案，两条注册路径相互独立。详细字段与分发见 [`docs/guides/6-how-to-use-and-provide-iluvatar-extend-compress-tensors.md`](../guides/6-how-to-use-and-provide-iluvatar-extend-compress-tensors.md)。

---

## 9. 原生 torch.ops._C 扩展

前八章的适配都在 Python 层完成。本章描述插件唯一的原生（C++/CUDA）扩展 `vllm_iluvatar._C`：它注册一组与社区签名一致的 `torch.ops._C` 算子，使社区 `vllm/kernels/vllm_c.py` 与 `vllm/utils/torch_utils.py` 能**零改动**地直接调用 `torch.ops._C.*`，底层落到 Iluvatar 实现。

### 9.1 为什么需要

社区代码（`vllm/kernels/vllm_c.py`、`vllm/utils/torch_utils.py` 等）直接调用 `torch.ops._C.rms_norm`、`torch.ops._C.rotary_embedding`、`torch.ops._C.weak_ref_tensor`、`torch.ops._C.get_cuda_view_from_cpu_tensor`。若想保持这些社区文件与上游字节一致（符合「零社区源码改动」总原则，见第 1 章），就必须由插件自行提供这些 `_C` 算子，而非去 patch 调用点。此外，插件还注册了一个内部算子 `apply_rotary_emb`（非社区 `torch.ops._C` 算子，仅供插件 OOT override 使用）。

### 9.2 文件架构（按功能分目录）

`csrc/` 按**社区兼容算子（`community/`）与插件独有算子（`plugin/`）**分目录组织，头文件就近放置，各目录完全自治：

| 文件 | 作用 | 对应上游 |
| --- | --- | --- |
| `core/registration.h` | `REGISTER_EXTENSION` / `TORCH_LIBRARY_*_EXPAND` 宏 | `csrc/core/registration.h`（镜像） |
| `torch_bindings.cpp` | `TORCH_LIBRARY_FRAGMENT` 定义 schema + `TORCH_LIBRARY_IMPL` 按设备注册 + `REGISTER_EXTENSION(_C)` | `csrc/torch_bindings.cpp`（同范式） |
| `community/ops.h` | 声明社区兼容算子（`vllm_iluvatar` 命名空间） | `csrc/ops.h`（同概念） |
| `community/ixformer_ops.h` | **插件特有**：前置声明 `ixformer::infer::rms_norm` | 无（ixformer 只 ship `kernels/` / `comm/` 头，不含 `infer/`） |
| `community/layernorm.cpp` | `rms_norm` 实现：薄封装调 `ixformer::infer::rms_norm` | `csrc/layernorm.cpp`（同名） |
| `community/pos_encoding_kernels.cu` | `rotary_embedding` 实现：自带 CUDA kernel（不依赖 ixformer） | `csrc/pos_encoding_kernels.cu`（同名） |
| `community/cuda_view.cu` | `weak_ref_tensor` / `get_cuda_view_from_cpu_tensor` 自带实现（torch + CUDA） | 上游 `cuda_view.cu` 等价 |
| `plugin/ops.h` | 声明插件独有算子（`apply_rotary_emb`、`apply_rotary_emb_meta`） | 无 |
| `plugin/apply_rotary_emb.cu` | `apply_rotary_emb` + Meta impl：自带 CUDA kernel（不依赖 ixformer） | 无 |

要点：

- **`rms_norm` 借力 ixformer**：`community/layernorm.cpp` 经 `ixformer_ops.h` 前置声明、薄封装直接调 `ixformer::infer::rms_norm`，把 dtype dispatch 等复杂逻辑交给 ixformer 高层封装；构建期 link `libixformer.so`，由 `cmake/FindIXFORMER.cmake` 定位已安装的 ixformer 包并设置 rpath（运行期解析 `.so`）。
- **rotary / view 算子自带实现**：`rotary_embedding`、`weak_ref_tensor`、`get_cuda_view_from_cpu_tensor` 逻辑相对简单，直接在 `community/pos_encoding_kernels.cu` / `community/cuda_view.cu` 实现，不依赖 ixformer kernel。
- **插件独有算子单独存放**：`apply_rotary_emb`（非社区 `torch.ops._C` 算子）置于 `plugin/apply_rotary_emb.cu`，与社区兼容算子物理分离。
- **构建集成**：`setup.py` 用 `CMakeExtension` / `CMakeBuildExt` 驱动 `CMakeLists.txt`，源文件经 `file(GLOB_RECURSE ... CONFIGURE_DEPENDS csrc/*.cc/*.cpp/*.cu)` 递归收集 `community/` 和 `plugin/` 子目录；支持 Ninja 增量编译、按 CUDA arch / build type 分桶。该扩展始终构建（缺 CMake 即构建失败，与硬件强约束一致）。

### 9.3 注册时序（构建期 + 运行期）

运行期注册发生在**早期 shim 阶段**（`patch_import_compat.apply()` 第 5 步，见 §3.4）：主进程经 `register_platform → _load_import_compat_patch`、spawn worker 经 `.pth`（见第 3 章），两条路径都在社区 `vllm_c.py` / `torch_utils.py` **首次调用之前**完成 `torch.ops._C.*` 注册。

```text
构建期:
  pip install -e .
    -> setup.py 的 CMakeExtension 调 CMake/Ninja
    -> 产出 vllm_iluvatar/_C.<abi>.so（rpath 指向 ixformer 包目录）

运行期:
  register_platform / .pth worker
    -> _load_import_compat_patch -> apply()
    -> _register_import_hooks
    -> _load_native_torch_ops: import vllm_iluvatar._C
    -> TORCH_LIBRARY 注册 torch.ops._C.{rms_norm, rotary_embedding, weak_ref_tensor, get_cuda_view_from_cpu_tensor, apply_rotary_emb}
    -> 社区 vllm_c.py / torch_utils.py 首次调用命中插件实现
```

### 9.4 新增一个自定义算子的流程

当又有社区文件直接调 `torch.ops._C.<name>` 且需 Iluvatar 实现时：

1. **声明**：在 `csrc/community/ops.h`（社区兼容算子）或 `csrc/plugin/ops.h`（插件独有算子）的 `vllm_iluvatar` 命名空间加函数声明。
2. **实现**：借 ixformer kernel 的仿照 `community/layernorm.cpp`（必要时在 `community/ixformer_ops.h` 补前置声明）；自带实现的仿照 `community/cuda_view.cu`。新增 `.cpp` / `.cu` 文件会被 `GLOB_RECURSE CONFIGURE_DEPENDS` 自动纳入，无需改 `CMakeLists.txt`。
3. **注册**：在 `csrc/torch_bindings.cpp` 用 `ops.def("<schema>")`（schema **逐字抄上游 `vllm/csrc`**）定义，并在对应设备段（CUDA / CPU）`ops.impl(...)`。
4. **重新编译**（增量）。
5. **测试**：必须在 `tests/kernels/core/` 添加数值/行为用例。
6. **前提**：社区调用点确实走 `torch.ops._C.<name>`；否则应选 Python 层的扩展点。

**与 §5 `CustomOp.register_oot` 的分工边界**：`torch.ops._C.*` 是**底层 C++ dispatcher 算子**，被社区 kernel/util 直接按名调用；`CustomOp.register_oot` 是**上层 layer 整类替换**（Python）。两者处于不同层次，按社区调用点的形态选择——调用点是 `torch.ops._C.<name>` 走本章，调用点是 `CustomOp` 子类实例化走第 5 章。

---

## 10. IR provider 适配

vLLM 新引入 **IR（intermediate representation）op + provider** 机制：把 `rms_norm` 这类算子抽象成「语义 op」，底层实现由多个 provider 注册并按优先级 dispatch。本章描述插件如何注册一等 provider `iluvatar`，让这些 op 直接落到 ixformer。

### 10.1 背景：vLLM IR 三层结构

`from vllm import ir` 暴露 `ir.ops`。其 dispatch 是三层结构：

1. **Op 注册表** `IrOp.registry`：`@register_op` 定义的语义 op（含一份 PyTorch `native` 实现）。当前只有两个：`rms_norm`、`fused_add_rms_norm`（`vllm/ir/ops/layernorm.py`）。
2. **provider 字典** `IrOp.impls`：每个 op 下 `provider 名 → 实现`，由 `@ir.ops.<op>.register_impl("<provider>", ...)` 在 import 时填充（社区内置 `vllm_c` / `native` / `aiter` / `oink` 等）。
3. **优先级 dispatch**：调用 `ir.ops.<op>(...)` 时按 `KernelConfig.ir_op_priority` 列表顺序，选第一个 `supported && supports_args(...)` 命中的 provider；都不中则回落 `native`。

平台通过两个钩子接入：`import_ir_kernels()`（触发 provider 注册模块 import）与 `get_default_ir_op_priority()`（给出本平台默认优先级）。

### 10.2 为什么要自注册 provider

未覆盖时 `IluPlatform` 继承 CUDA 默认优先级 `["vllm_c","native"]`：

- `rms_norm` 经 `vllm_c` provider 调 `torch.ops._C.rms_norm` —— 由插件 `_C` 提供（第 9 章），**可跑通**。
- `fused_add_rms_norm` 的 `vllm_c` 实现调 `torch.ops._C.fused_add_rms_norm` —— 插件 `_C` **未注册**该算子，IR 路径在 Iluvatar 上**不可用**（这正是 `IluRMSNorm` 现在把 residual 分支绕开 IR 的原因，见 §5.6 / §10.5）。

故注册一个一等 provider `iluvatar`，两个 op 都**直调 ixformer**（不经 `torch.ops._C` 间接层），并通过优先级让其生效。

### 10.3 文件架构

`vllm_iluvatar/kernels/`（镜像上游 `vllm/kernels/`，命名遵循第 11 章规则 3 的 `ilu_` 前缀）：

| 文件 | 作用 | 对应上游 |
| --- | --- | --- |
| `kernels/__init__.py` | `from . import ilu_ops` —— import 即触发注册 | `vllm/kernels/__init__.py`（同范式） |
| `kernels/ilu_ops.py` | 注册 `iluvatar` provider 的两个 op 实现 | `vllm/kernels/vllm_c.py`（同概念，provider 名不同） |

`ilu_ops.py` 要点：

- `@ir.ops.rms_norm.register_impl("iluvatar", ...)` → 分配 `out` 后调 `ixf_ops.rms_norm(x, weight, epsilon, out)`。
- `@ir.ops.fused_add_rms_norm.register_impl("iluvatar", inplace=True, ...)` → 调 `ixf_ops.residual_rms_norm(x, weight, epsilon, 1.0, x_residual)`，返回 `(normed, residual)`。
- **签名逐字对齐社区 IR op**（`x, weight, epsilon, variance_size=None` 等），否则 `IrOpImpl.__init__` 的 schema 校验会拒绝注册。
- `supports_args` 沿用 `vllm_c` 同款约束：无 `variance_size` 覆盖、`weight` 与 `x` 同 dtype；不满足则 dispatch 自动落到后续 provider。

### 10.4 平台覆盖与优先级

`IluPlatform`（`vllm_iluvatar/platforms/ilu_platform.py`）覆盖两个方法：

- `import_ir_kernels()`：`import vllm.kernels`（保留社区 provider，`vllm_c` 作 `rms_norm` 兜底）+ `import vllm_iluvatar.kernels`（注册 `iluvatar`）。这是官方钩子——`KernelConfig._iter_op_priorities()` 会先调它，再校验优先级列表里每个 provider 已注册，故无需走 general_plugins。
- `get_default_ir_op_priority(vllm_config)`：

  - **inductor 编译**（`backend=="inductor"` 且 `mode != NONE`）：`["native"]`，让 inductor 自行融合（沿用社区 CUDA 策略）。
  - **eager**：
    - `rms_norm = ["iluvatar", "vllm_c", "native"]`（保留 `vllm_c` 兜底，其 `_C.rms_norm` 插件已提供）。
    - `fused_add_rms_norm = ["iluvatar", "native"]`（**不留** `vllm_c`，因其调的 `torch.ops._C.fused_add_rms_norm` 插件 `_C` 不存在）。

> 优先级是 per-op 的；dispatch 命中第一个匹配者。用户用 `--ir-op-priority.<op>=...` 显式指定时，平台默认会被 append 在用户列表之后（去重）。

### 10.5 与三条 rms_norm 路径的关系、dispatch 两层结构、收敛计划

插件目前有**三条**把 RMSNorm 落到 ixformer 的路径，分属不同层次：

| 路径 | 机制 | 触发点 |
| --- | --- | --- |
| `ir.ops`（本章 provider） | `iluvatar` provider 直调 ixformer | `IluRMSNorm.forward_native` 无残差分支 `ir.ops.rms_norm(...)` |
| `vllm._custom_ops` 补丁 | import hook splat（第 2.2 / 5 章） | `IluRMSNorm.forward_native` residual 分支 `fused_add_rms_norm(...)` |
| `torch.ops._C`（第 9 章） | 自研 `_C` 扩展 | 社区 `vllm_c` provider / `torch_utils.py` 等直接按名调用 |

**dispatch 是两层嵌套，不是平级二选一**：

- **外层 = layer 替换**：`RMSNorm` 是 `CustomOp`，实例化时被 `IluRMSNorm` OOT 掉包（第 5 章，恒生效）。`forward_cuda` 因 `_enum==CUDA` 被选中，委托 `forward_native`（`VLLM_BATCH_INVARIANT` 时改走确定性 `rms_norm_batch_invariant`，两条 ixformer 路都不进）。
- **内层 = op dispatch**：`forward_native` 的**无残差**分支才调 `ir.ops.rms_norm(...)`，由优先级选中 `iluvatar` provider（取代原 `vllm_c → _C` 间接路径，等价但更直接）；**residual** 分支直接走 `vllm._custom_ops`，**不进 IR**。

```text
RMSNorm 实例化
  -> OOT 掉包为 IluRMSNorm（外层，恒生效）
  -> forward_cuda（因 _enum=CUDA）
  -> 判断 VLLM_BATCH_INVARIANT?
     -> 是: rms_norm_batch_invariant（不进 IR / _custom_ops）
     -> 否: forward_native
            -> 判断 residual is None?
               -> 是: ir.ops.rms_norm -> IR dispatch（内层）
                      -> 按优先级 [iluvatar, vllm_c, native] 逐个试 supports_args
                         -> iluvatar 命中: iluvatar provider 直调 ixformer
                         -> 回落: vllm_c -> torch.ops._C -> 插件 _C
                         -> 再回落: native（PyTorch）
               -> 否: vllm._custom_ops.fused_add_rms_norm -> ixformer（绕开 IR）

IluPlatform.import_ir_kernels / get_default_ir_op_priority
  -> 影响 IR dispatch 的 provider 可见性与优先级
```

**并行策略与收敛计划**：本阶段不删 `IluRMSNorm` override —— 无残差分支已改由 `iluvatar` provider 承接；`iluvatar.fused_add_rms_norm` 已注册但处「待命态」（residual 分支仍走 `_custom_ops`）。后续「收敛阶段」可把 residual 分支也改成 `ir.ops.fused_add_rms_norm.maybe_inplace`，启用该 provider 并简化 `IluRMSNorm`。

---

## 11. 文件命名逻辑

命名规则的核心目标：让「对齐上游」与「新增实现」一眼可分，便于与社区版对照维护。

### 11.1 五条规则

1. **能对上 vLLM 上游文件树 -> 用上游同名**（方便对齐）：
   - `_custom_ops.py`（对应 `vllm/_custom_ops.py`）
   - `envs.py`（对应 `vllm/envs.py`）
   - `distributed/device_communicators/all2all.py`（对应上游同名文件）
   - general patch 的 override 实现文件同理用上游同名（如 `distributed/parallel_state.py`、`distributed/utils.py`、`v1/attention/backends/fa_utils.py`，见第 4 章）。
   - OOT 注册的实现文件也镜像上游同名（如 `model_executor/layers/layernorm.py`、`model_executor/models/qwen3.py`、`model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`、`model_executor/models/deepseek_v4/`，保留社区架构类名，见第 5、6、8 章）。
   - **模块替身 shim `ilu_patch/import_shims/` 内部文件镜像被替身的上游模块同名**（如 `vllm_flash_attn.py` ↔ `vllm/vllm_flash_attn/`），便于对照；这是规则 4「`ilu_` 文件夹内部不缀」之外、为「整模块替身」保留上游名的特例（与 `csrc/` 镜像上游同理，见 §3.6）。
   - **原生扩展 `csrc/` 按 community / plugin 分目录**：`community/` 内的文件名与上游 `vllm/csrc` 对应（`layernorm.cpp`、`pos_encoding_kernels.cu`、`cuda_view.cu`）；`plugin/` 存放插件独有算子（`apply_rotary_emb.cu`）；注册入口 `torch_bindings.cpp` 与 `core/registration.h` 保持在根，唯一插件特有的 `ixformer_ops.h` 置于 `community/`（ixformer 未 ship `infer/` 头，需自带前置声明，见第 9 章）。
2. **general patch 接线 -> 集中到 `ilu_patch/general_patches/`**：
   - 文件名统一使用 `patch_<target>.py`，例如 `patch_parallel_state.py`、`patch_stateless_coordinator.py`、`patch_fused_moe_layer.py`、`patch_fa_utils.py`。
   - 接线模块含 `register()`/`_apply()`、签名/指纹校验和基线快照；override 实现仍放在上游镜像目录。
3. **上游为「按后端分文件」的镜像目录中、我们新增的后端实现 -> `ilu_` 前缀**：
   - `platforms/ilu_platform.py`（`platforms/` 镜像 `vllm/platforms/`）
   - `distributed/device_communicators/ilu_communicator.py`
   - `distributed/kv_transfer/kv_connector/v1/ilu_nixl/`（镜像上游 `distributed/kv_transfer/kv_connector/v1/nixl/` 的 connector 拆分）
4. **新增文件夹（上游无对应概念）-> 文件夹以 `ilu_` 开头，内部子目录/文件不再加 `ilu`**（文件夹本身即标识）：
   - `ilu_patch/`（内含 `general_patch.py`、`general_patches/`、`import_hook.py`、`patch_custom_ops.py`、`patch_import_compat.py`、`patch_platform_import_hooks.py`、`patch_pynvml.py`）
   - `ilu_register/`（OOT 注册编排，见第 5、6、8 章）
   - `ilu_util/`（内含 `signature.py`）
5. **Python 包基础设施保持原名**：`__init__.py`、`version.py`。

### 11.2 类名规则

- 我们拥有、内部引用的类统一用短前缀 `Ilu`：`IluPlatform`、`IluCommunicator`。
- **OOT 覆盖子类用 `Ilu` 前缀 + 社区基类名**：`IluRMSNorm`(`@RMSNorm.register_oot`)、`IluGemmaRMSNorm`(`@GemmaRMSNorm.register_oot`)，一眼看出「覆盖哪个社区类」。
- 靠「名字」挂回社区模块的符号**保留社区名**，以便 `setattr` 覆盖或按名实例化：
  - `_custom_ops.py` 中的函数名（如 `rms_norm`、`rotary_embedding`，要按名 splat 回 `vllm._custom_ops`）。
  - `all2all.py` 中的 `NaiveAll2AllManager` / `AgRsAll2AllManager`（按 backend 名实例化）。
  - 模型 OOT 子类保留社区架构类名 `Qwen3ForCausalLM`（注册时按架构名覆盖）。
- **新增的插件自有算子**用 PascalCase 类名 + snake_case 注册名：`RMSNormQK` 经 `@CustomOp.register("rms_norm_qk")` 注册。

### 11.3 接线钩子约定

不同接入路径的「接线」入口名各异，据此即可判断一个模块走哪条路径：

- general patch：`ilu_patch/general_patches/*.py` 暴露 `register()`（见 §4.3）。
- platform-stage post-import hook：`ilu_patch/patch_platform_import_hooks.py` 中登记 `_HOOKS`（见 §3、§4.4）。
- pynvml 兼容：`ilu_patch/patch_pynvml.py`（独立于 general patch 和 import hook 的 pynvml 适配）。
- 模型 OOT：模型模块暴露 `register_model()` 钩子（见 §6.2）。
- 算子 OOT：**无钩子**，靠模块级 `@<Base>.register_oot` / `@CustomOp.register` 装饰器（见 §5.2）。

### 11.4 规则速查

```text
对齐上游同名      ->  _custom_ops.py / envs.py / all2all.py / parallel_state.py / layernorm.py / qwen3.py
                       / quantization/compressed_tensors/compressed_tensors.py / models/deepseek_v4/
general patch 接线 ->  ilu_patch/general_patches/patch_<target>.py
                       （patch_parallel_state.py / patch_fa_utils.py；
                       基线快照内联其中）
platform import hook ->  ilu_patch/patch_platform_import_hooks.py（平台确认后注册、目标模块 import 后回调）
模块替身 shim     ->  ilu_patch/import_shims/<被替身上游模块同名>.py（如 vllm_flash_attn.py）
镜像目录新增文件  ->  ilu_<name>.py（ilu_platform.py / ilu_communicator.py）
新增文件夹        ->  ilu_<name>/，内部不缀（ilu_patch/ / ilu_register/ / ilu_util/ / kv_connector/v1/ilu_nixl/）
包基础设施        ->  __init__.py / version.py（不变）
类名              ->  Ilu 前缀（含 OOT 子类 IluRMSNorm 等）；按名挂回社区的符号保留社区名
接线钩子          ->  register()（general patch）/ _HOOKS（platform import hook）/ register_model()（模型 OOT）/ 装饰器（算子 OOT）
```

---

## 12. 代码规范：pre-commit

为与 vLLM 社区保持一致的代码风格与提交习惯，本包提供 `.pre-commit-config.yaml`，并在 vLLM 上游配置基础上**裁剪为纯 Python 插件包适用的子集**。

### 12.1 取舍原则

vLLM 上游的 pre-commit 配置包含大量依赖其仓库基础设施的钩子（`tools/pre_commit/*` 脚本、`requirements/*` 锁定、`csrc/` 的 clang-format、`rust/`、`docker/` 等）。本包以 Python 为主，不引入上游那套工具链（仅有的原生 `csrc/` 扩展走本地 CMake 构建、不纳入 pre-commit，见第 9 章），故：

- **保留**：对任意 Python 包都适用、且**版本与上游对齐**的外部钩子。
- **丢弃**：依赖 vLLM 仓库特有目录 / 脚本的钩子（C++/CUDA、Rust、docker、pip-compile、`tools/pre_commit/*` 等），避免引用不存在的路径导致 pre-commit 直接报错。

### 12.2 启用的钩子

| 钩子 | 版本（对齐上游） | 作用 |
| --- | --- | --- |
| `ruff-check` / `ruff-format` | `v0.14.0` | Python lint + 格式化（`--fix` 自动修复） |
| `typos` | `v1.43.5` | 拼写检查 |
| `markdownlint-cli2` | `v0.21.0` | Markdown 文档规范（`--fix`） |
| `check-filenames`（local） | — | 禁止文件名含空格 |
| `signoff-commit`（local，commit-msg） | — | 自动补 `Signed-off-by`，对齐社区 DCO 习惯 |
| `suggestion`（local） | — | 提示如何用 `--no-verify` / `SKIP=` 跳过钩子 |

外部钩子版本与上游逐一对齐，便于将来同步升级。

### 12.3 使用

每个开发者克隆仓库后**一次性**安装钩子；之后每次 `git commit` 会自动对本次改动文件运行检查。

```bash
cd vllm-iluvatar
pip install pre-commit
pre-commit install --install-hooks   # 安装 pre-commit 与 commit-msg 钩子
```

### 12.4 ruff 规则对齐

ruff 的规则集与上游对齐，定义在 `pyproject.toml` 的 `[tool.ruff.lint]` / `[tool.ruff.format]`：

- `select`：`E`（pycodestyle）、`F`（Pyflakes）、`UP`（pyupgrade）、`B`（bugbear）、`ISC`、`SIM`、`I`（isort）、`G`（logging-format）。
- `ignore`：`F405`/`F403`（star import）、`E731`、`B905`、`B007`、`UP032`、`E501`（行宽，与上游 ruff-check 行为对齐——formatter 不强拆长行，故 linter 不再就此报错）。
- `format`：`docstring-code-format = true`。
- `per-file-ignores`：`vllm_iluvatar/version.py` 忽略 `F401`。

---

## 13. 测试约定

呼应第 1 章「硬件强约束」，测试侧也遵循同一原则：**依赖缺失即报错，绝不静默跳过**。

### 13.1 依赖强制，缺失即报错

- `ixformer` / `vllm` / 原生 `vllm_iluvatar._C` 是本 build 的**强依赖**。需要它们的测试**直接 `import`**，缺失时在 collection 期抛 `ImportError`、被 pytest 计为 **ERROR**，从而暴露环境问题。
- **不使用 `pytest.importorskip(...)` 跳过缺失依赖**——那会把「环境装错」伪装成「125 passed, 1 skipped」，掩盖问题。

### 13.2 仅硬件用 skip gating

唯一允许 skip 的是**物理 CUDA / Iluvatar 设备不在**：数值 / kernel 用例确实无法在无卡机跑。统一用 [tests/conftest.py](../../tests/conftest.py) 的 `requires_cuda` mark 与 `cuda_device` fixture（内部 `torch.cuda.is_available()` 判定）gating。

### 13.3 纯 Python 机制写 CPU-only 用例

不依赖 GPU 的机制（如 `rebind_module_func` 的传播 / 身份匹配、签名校验）应写成 CPU-only 用例，无卡环境也能跑通。例：`tests/v1/attention/backends/test_fa_utils_patch.py` 用合成伪模块验证 `rebind_module_func` 同时改到源模块与已 `from-import` 的拷贝引用、且按身份不误伤同名异物。
