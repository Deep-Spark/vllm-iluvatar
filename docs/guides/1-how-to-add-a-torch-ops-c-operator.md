# 在 vllm-iluvatar 新增 torch.ops._C 算子

> **How-to 指南** · 让上游 vLLM 代码零改动运行 · 以 `rotary_embedding` 为例

本文说明如何在 `vllm-iluvatar` 插件里自己实现一个底层算子，把它注册进
`torch.ops._C` 命名空间，从而让**上游 vLLM 代码零改动**就能调用到天数（Iluvatar）
的实现。全程只在插件仓库内改动，不需要修改上游 vLLM。

| | |
| --- | --- |
| **适用对象** | 任何上游 vLLM 通过 `vllm._custom_ops` → `torch.ops._C.<op>` 调用的底层算子 |
| **改动范围** | 仅 `vllm-iluvatar` 插件仓库，不改上游 vLLM |
| **参考实现** | `rotary_embedding`（本仓库已完整落地） |

---

## 目录

- [1. 背景与目标](#1-背景与目标)
- [2. 原理与调用链](#2-原理与调用链)
- [3. 移植前：抄准社区接口](#3-移植前抄准社区接口)
- [4. 端到端步骤（rotary_embedding 主线）](#4-端到端步骤rotary_embedding-主线)
  - [4.1 实现内核 `csrc/pos_encoding_kernels.cu`](#41-实现内核-csrcpos_encoding_kernelscu)
  - [4.2 在 `csrc/ops.h` 声明函数](#42-在-csrcopsh-声明函数)
  - [4.3 在 `csrc/torch_bindings.cpp` 注册](#43-在-csrctorch_bindingscpp-注册)
  - [4.4 构建：新增 `.cu` 无需改 CMake](#44-构建新增-cu-无需改-cmake)
  - [4.5（可选）清理 Python 侧 override](#45可选仅当替换已有-oot-覆盖时清理-python-侧-override)
- [5. 测试](#5-测试)
- [6. 附录](#6-附录)

---

## 1. 背景与目标

上游 vLLM 的很多算子最终都落到一个 C++/CUDA 扩展上。调用形态是：

- Python 层通过 `vllm._custom_ops`（简称 `ops`）里的包装函数调用；
- 包装函数内部再调 `torch.ops._C.<op>(...)`；
- `torch.ops._C.<op>` 由某个已编译的扩展在 import 时静态注册。

在 `vllm-iluvatar` 里，这个扩展就是 `vllm_iluvatar._C`。只要我们用**与上游完全一致的
schema**在 `_C` 命名空间里注册同名算子，上游的 Python 调用点就会自动解析到我们的实现，
无需改上游一行代码，也不再需要为每个类写 out-of-tree（OOT）覆盖。

本文的目标就是把这套"新增算子 → 注册进 `_C` → 上游零改动调用"的流程讲清楚。

---

## 2. 原理与调用链

关键机制：**`import vllm_iluvatar._C` 会触发扩展里的 `TORCH_LIBRARY_FRAGMENT`
静态注册**，于是社区的 `torch.ops._C.*` 调用点在运行时解析到插件实现。

以 `rotary_embedding` 为例，从模型到内核的完整调用链（纯文本）：

```
Model 前向（如 Qwen3Attention.forward）
  └─ self.rotary_emb(positions, q, k)              # RotaryEmbedding.__call__
       └─ RotaryEmbedding.forward_cuda(...)        # 社区实现，未改动
            └─ vllm._custom_ops.rotary_embedding(...)   # 社区包装函数，未改动
                 └─ torch.ops._C.rotary_embedding(...)  # PyTorch dispatcher
                      └─ vllm_iluvatar::rotary_embedding(...)  # 插件 CUDA 实现（本仓库）
```

要点：
- 上游 `RotaryEmbedding.forward_cuda` 和 `vllm._custom_ops.rotary_embedding` 全都
  **保持原样**；
- 我们只提供最底层的 `torch.ops._C.rotary_embedding` 实现；
- 因此上游一旦升级，这些上层代码自动跟随，不留维护包袱。

---

## 3. 移植前：抄准社区接口

新增算子的第一步是**照抄上游 schema**，一个字都不能差（参数名、顺序、默认值、
in-place 标记）。schema 来源是上游 vLLM 仓库的 `csrc/libtorch_stable/torch_bindings.cpp`。

以 `rotary_embedding` 为例，上游 schema 为：

```
rotary_embedding(Tensor positions, Tensor! query, Tensor!? key,
                 int head_size, Tensor cos_sin_cache, bool is_neox,
                 int rope_dim_offset=0, bool inverse=False) -> ()
```

抄 schema 时需同时确认的几件事：

- **in-place 语义**：`Tensor!` 表示原地修改（`query` 原地旋转写回），`Tensor!?`
  表示可选且原地（`key` 可为 `None`）；返回 `-> ()` 即无返回值。
- **参数默认值**：`rope_dim_offset=0`、`inverse=False` 必须与上游一致，否则调用点
  少传参数时行为会不同。
- **dtype / shape 约定**：`positions` 为 int64、支持 1D `[num_tokens]` 或 2D
  `[batch, seq]`；`query`/`key` 支持扁平 `[*, H*D]` 或带头维 `[*, H, D]`（最高 4D）；
  `cos_sin_cache` 的 dtype 可以与 `q/k` 不同（如 fp32 cache 配 bf16 的 q/k）。

把这些约定确认清楚后，实现时严格照做即可保证与上游行为对齐。

---

## 4. 端到端步骤（rotary_embedding 主线）

整体流程（前 4 步必做，第 5 步按需）：

```
4.1 写内核 .cu  →  4.2 ops.h 声明  →  4.3 torch_bindings 注册  →  4.4 构建
                                                                    └─▶ 4.5（可选）清理旧 OOT 覆盖
```

### 4.1 实现内核 `csrc/pos_encoding_kernels.cu`

在 `csrc/` 下新增一个 `.cu` 文件，实现 `vllm_iluvatar` 命名空间下的 host 函数。
host 端负责**shape / stride 推导与校验**（建议逐行对齐上游同名内核的 host 逻辑），
device 端负责实际计算。文件位于
[`csrc/pos_encoding_kernels.cu`](../../csrc/pos_encoding_kernels.cu)。

host 函数签名与上游一致，内部完成派生量推导（num_tokens / num_heads /
num_kv_heads / rot_dim / 各 stride）与校验：

```cpp
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      c10::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse) {
  const int64_t num_tokens = positions.numel();
  const int positions_ndim = positions.dim();
  TORCH_CHECK(positions_ndim == 1 || positions_ndim == 2, ...);
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");

  const int num_heads = (query.numel() / num_tokens) / head_size;
  const int num_kv_heads = key.has_value()
      ? (key->numel() / num_tokens) / head_size : num_heads;
  const int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK((rot_dim + rope_dim_offset) <= head_size, ...);

  // 头维 stride：带头维 [*, H, D] 用 stride(-2)（支持 padded/sliced）；
  // 扁平 [*, H*D] 时各头是长度 head_size 的连续块。
  const int64_t head_stride =
      (query.dim() == positions_ndim + 2) ? query.stride(-2) : head_size;
  ...
}
```

实现要点：

- **dtype dispatch**：分别对 `query` dtype（fp32/fp16/bf16）与 `cos_sin_cache`
  dtype 派发，允许二者不同；累加统一用 fp32 保证精度。
- **向量化**：连续布局下可按 `PACK_NUM`（如 2）向量化访存，非连续时回退到标量
  （`PACK=1`）路径。
- **原地写回**：直接对 `query`/`key` 的存储写回，符合 `Tensor!` 语义。

### 4.2 在 `csrc/ops.h` 声明函数

在 [`csrc/ops.h`](../../csrc/ops.h) 的 `vllm_iluvatar` 命名空间里补上声明，
注释里附上对应的社区 schema，并**带上默认参数值**：

```cpp
namespace vllm_iluvatar {

// rotary_embedding(Tensor positions, Tensor! query, Tensor!? key,
//                  int head_size, Tensor cos_sin_cache, bool is_neox,
//                  int rope_dim_offset=0, bool inverse=False) -> ()
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      c10::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse);

}  // namespace vllm_iluvatar
```

### 4.3 在 `csrc/torch_bindings.cpp` 注册

在 [`csrc/torch_bindings.cpp`](../../csrc/torch_bindings.cpp) 里做两件事：
用 `ops.def(...)` 声明 schema（与上游逐字一致），用 `ops.impl(...)` 把它绑定到
4.1 的实现，并**选对 dispatch key**（CUDA 算子注册在 `CUDA` 键上）。

```cpp
TORCH_LIBRARY_FRAGMENT_EXPAND(_C, ops) {
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox, "
      "int rope_dim_offset=0, bool inverse=False) -> ()");
  // ... 其它算子 ...
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CUDA, ops) {
  ops.impl("rotary_embedding", &vllm_iluvatar::rotary_embedding);
  // ... 其它算子 ...
}
```

说明：
- 使用 `TORCH_LIBRARY_FRAGMENT`（而非 `TORCH_LIBRARY`）——插件只是**扩展** `_C`
  命名空间，不独占，避免与其它定义冲突。
- 若算子输入是 CPU 张量，则应注册到 `CPU` 键（参考同文件里
  `get_cuda_view_from_cpu_tensor` 的写法）。

### 4.4 构建：新增 `.cu` 无需改 CMake

[`CMakeLists.txt`](../../CMakeLists.txt) 用 `GLOB_RECURSE ... CONFIGURE_DEPENDS`
自动收集 `csrc/` 下所有 `.cc/.cpp/.cu`：

```cmake
file(GLOB_RECURSE VLLM_ILUVATAR_EXT_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cu")
```

因此新增 `pos_encoding_kernels.cu` **不需要**手动改 CMake；`CONFIGURE_DEPENDS`
让 CMake 在下次构建时自动发现新文件并重新配置。

### 4.5（可选，仅当替换已有 OOT 覆盖时）清理 Python 侧 override

如果新增的算子是用来**取代插件里已有的 OOT 覆盖**（而不是全新算子），还需要清理旧路径，
让上游 Python 代码直接走到新的 `_C` 实现；纯新增算子可跳过本步。

以 `rotary_embedding` 为例，涉及三处：

- **`vllm_iluvatar/_custom_ops.py`**：删除该算子的 Python override 函数，并从
  `_ILU_OVERRIDES` 清单里移除对应条目。这样社区 `vllm._custom_ops.rotary_embedding`
  会原样使用、并分发到我们的 `_C` 实现。当前文件里保留了一条说明性注释：

  ```python
  # rotary_embedding is now provided by the plugin's native torch.ops._C kernel
  # (csrc/pos_encoding_kernels.cu), which matches the community signature exactly,
  # so community vllm._custom_ops.rotary_embedding is used unchanged (no override).
  ```

- **`vllm_iluvatar/model_executor/layers/.../base.py`**：删除逐类的
  `forward_cuda` OOT 子类（rope 场景下是 10 个 `RotaryEmbedding` 子类）。

- **`vllm_iluvatar/ilu_patch/patch_custom_ops.py`**：从 `_SKIP_SIGCHECK` 集合里移除
  该算子名。签名已与社区对齐后，就应恢复严格签名校验（不再豁免）。

---

## 5. 测试

### 放置位置

- 内核数值测试：`tests/kernels/core/test_<op>.py`（rope 为
  [`tests/kernels/core/test_pos_encoding.py`](../../tests/kernels/core/test_pos_encoding.py)）。
- 若做了 4.5 的 OOT 清理，把"覆盖已移除"的断言更新到对应
  `tests/model_executor/...` 测试里。

### 两层测试

- **模块级**：通过 `get_rope` 构造真实的 rope 模块，调用 `forward_cuda`
  （即走 `ops.rotary_embedding` → 我们的 `_C` 内核），与 `forward_native`（纯 torch
  参考）对比。这验证的是"删掉 OOT 覆盖之后的完整路径"。
- **裸算子级**：直接调用 `torch.ops._C.rotary_embedding`，与手写的纯 torch 参考实现
  对比，覆盖模块级测不到的维度。

### 对齐社区测试并补边界

先对照上游 `tests/kernels/core/test_pos_encoding.py`，把它覆盖的场景搬过来
（`is_neox` 两种、扁平/带头维/padded 布局、`key=None`、奇数头数、大 `seq_len`、
partial rotary dim、fp16/bf16/fp32），再补上天数/DeepSeek 特有的边界：

- `rope_dim_offset > 0`；
- `inverse=True`；
- GQA（`num_kv_heads < num_heads`）；
- `cos_sin_cache` dtype 与 `q/k` 不同（fp32 cache + bf16/fp16 q/k）；
- 非连续（strided）的 `query`/`key`。

### CustomOp 需要 VllmConfig 上下文

直接构造 `CustomOp`（如 `RotaryEmbedding`）时，`CustomOp.__init__` 会调用
`get_current_vllm_config()`，要求有活跃的 config 上下文，否则报
`Current vLLM config is not set`。测试通过
[`tests/conftest.py`](../../tests/conftest.py) 的 `default_vllm_config` fixture 提供：

```python
@pytest.fixture()
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config
    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config
```

在需要构造 rope 模块的测试里把它作为参数注入即可。

---

## 6. 附录

### 关键文件清单

| 作用 | 所在仓库 | 文件 |
| --- | --- | --- |
| 调用我们算子的社区接口包装函数 | 上游 vllm | `vllm/_custom_ops.py` |
| schema 来源（照抄用） | 上游 vllm | `csrc/libtorch_stable/torch_bindings.cpp` |
| 内核实现（host + device） | vllm-iluvatar | [`csrc/pos_encoding_kernels.cu`](../../csrc/pos_encoding_kernels.cu) |
| 函数声明 | vllm-iluvatar | [`csrc/ops.h`](../../csrc/ops.h) |
| schema 定义 + impl 绑定 | vllm-iluvatar | [`csrc/torch_bindings.cpp`](../../csrc/torch_bindings.cpp) |
| 构建（自动收集 `.cu`） | vllm-iluvatar | [`CMakeLists.txt`](../../CMakeLists.txt) |
| Python override 清单（可选清理） | vllm-iluvatar | [`vllm_iluvatar/_custom_ops.py`](../../vllm_iluvatar/_custom_ops.py) |
| 签名校验豁免集合（可选清理） | vllm-iluvatar | [`vllm_iluvatar/ilu_patch/patch_custom_ops.py`](../../vllm_iluvatar/ilu_patch/patch_custom_ops.py) |
| 内核测试 | vllm-iluvatar | [`tests/kernels/core/test_pos_encoding.py`](../../tests/kernels/core/test_pos_encoding.py) |
| 测试 fixture | vllm-iluvatar | [`tests/conftest.py`](../../tests/conftest.py) |

### 步骤速览

1. 抄准上游 schema（`csrc/libtorch_stable/torch_bindings.cpp`）。
2. 新增 `csrc/<op>_kernels.cu` 实现 host + device。
3. 在 `csrc/ops.h` 声明函数。
4. 在 `csrc/torch_bindings.cpp` 用 `ops.def` + `ops.impl`（选对 dispatch key）注册。
5. 直接构建，`CONFIGURE_DEPENDS` 自动收新文件。
6.（可选）替换已有 OOT 覆盖时，清理 `_custom_ops.py` / `base.py` / `_SKIP_SIGCHECK`。
7. 加 `tests/kernels/core/test_<op>.py`，模块级 + 裸算子级两层测试。

`rotary_embedding` 是一个已完整落地的参考实现，可直接对照上述文件学习。
