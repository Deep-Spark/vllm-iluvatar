# 新增一个 torch.ops._C 算子

> **开发者指南** · community / plugin 两条主线 · 全程仅改插件仓库

---

## 目录

- [1. 背景](#1-背景)
- [2. 前置判断：community 还是 plugin](#2-前置判断community-还是-plugin)
- [3. community 算子完整流程](#3-community-算子完整流程)（示例 `rotary_embedding`）
  - [3.1 调用链](#31-调用链)
  - [3.2 确认上游 schema](#32-确认上游-schema)
  - [3.3 编写 CUDA 内核](#33-编写-cuda-内核)
  - [3.4 声明函数签名](#34-声明函数签名)
  - [3.5 注册 schema 与 dispatch](#35-注册-schema-与-dispatch)
  - [3.6 构建](#36-构建)
  - [3.7 （可选）清理旧 OOT 覆盖](#37-可选清理旧-oot-覆盖)
  - [3.8 测试](#38-测试)
- [4. plugin 算子完整流程](#4-plugin-算子完整流程)（示例 `apply_rotary_emb`）
  - [4.1 调用链](#41-调用链)
  - [4.2 定义 schema](#42-定义-schema)
  - [4.3 编写 CUDA 内核 + Meta impl](#43-编写-cuda-内核--meta-impl)
  - [4.4 声明函数签名](#44-声明函数签名)
  - [4.5 注册 schema 与 dispatch](#45-注册-schema-与-dispatch)
  - [4.6 编写 OOT override](#46-编写-oot-override)
  - [4.7 构建](#47-构建)
  - [4.8 测试](#48-测试)
- [5. 附录](#5-附录)

---

## 1. 背景

上游 vLLM 的底层算子通过 `torch.ops._C.<op>` 调用，由编译期静态注册的 C++ 扩展提供实现。在 `vllm-iluvatar` 中，这个扩展是 `vllm_iluvatar._C`。

根据上游是否已有同名算子，插件中的 `_C` 算子分为两类，**调用机制和开发流程有本质区别**：

| | community 算子 | plugin 算子 |
| --- | --- | --- |
| **含义** | 上游 vLLM 已有同名同 schema 的 `torch.ops._C` 算子 | 上游没有，由插件自行新增的算子 |
| **调用机制** | 照抄上游 schema 后，社区 Python 代码通过 PyTorch dispatcher **自动命中**插件实现，无需改动上层 | 社区代码不会自动调用；**必须新增 OOT override** 将社区调用点接到插件实现 |
| **维护负担** | 低——上游升级时上层代码自动跟随 | 较高——OOT override 随上游变更需自行维护 |

两条开发主线：

- 上游有同名算子 → 走 [第 3 节 community 流程](#3-community-算子完整流程)
- 上游没有 → 走 [第 4 节 plugin 流程](#4-plugin-算子完整流程)

---

## 2. 前置判断：community 还是 plugin

新增算子前先确定归属：

```text
上游 torch.ops._C 有同名同 schema 的算子？
  ├── 是 → community/（社区兼容算子）→ 第 3 节
  └── 否 → plugin/（插件独有算子）  → 第 4 节
```

两类算子的差异对照：

| | community/ | plugin/ |
| --- | --- | --- |
| **定义** | 上游 vLLM 已有的 `torch.ops._C` 算子 | 上游没有、插件自行新增的算子 |
| **schema 来源** | 逐字照抄上游 `libtorch_stable/torch_bindings.cpp` | 自行定义 |
| **源文件位置** | `csrc/community/<op>.cu` 或 `.cpp` | `csrc/plugin/<op>.cu` |
| **头文件声明** | `csrc/community/ops.h` | `csrc/plugin/ops.h` |
| **Meta impl** | 上游已有，一般不需要 | **必须提供**（torch.compile / opcheck 需要） |
| **OOT override** | **不需要**（社区代码自动命中）；常需**清理**旧的 | **必须新增**（否则社区调用点无法触达） |
| **测试对照** | 对齐上游同名测试 + 补硬件特有边界 | 自行设计参考实现 |
| **典型示例** | `rms_norm`、`rotary_embedding`、`weak_ref_tensor` | `apply_rotary_emb` |

目录结构全貌：

```text
csrc/
├── core/registration.h              # 注册宏
├── community/                       # 社区兼容算子
│   ├── ops.h                       # 函数声明
│   ├── ixformer_ops.h              # ixformer 外部符号
│   ├── layernorm.cpp               # rms_norm（委托 ixformer）
│   ├── pos_encoding_kernels.cu     # rotary_embedding
│   └── cuda_view.cu               # weak_ref_tensor / get_cuda_view_from_cpu_tensor
├── plugin/                          # 插件独有算子
│   ├── ops.h                       # 函数声明
│   └── apply_rotary_emb.cu        # apply_rotary_emb + Meta impl
└── torch_bindings.cpp               # 统一注册入口
```

---

## 3. community 算子完整流程

> 示例：`rotary_embedding` · 参考实现：[`csrc/community/pos_encoding_kernels.cu`](../../csrc/community/pos_encoding_kernels.cu)

### 3.1 调用链

`import vllm_iluvatar._C` 触发 `TORCH_LIBRARY_FRAGMENT` 静态注册，社区调用点在运行时通过 PyTorch dispatcher 自动解析到插件实现，**上层代码无需改动**。

以 `rotary_embedding` 为例：

```text
Qwen3Attention.forward
  └─ self.rotary_emb(positions, q, k)
       └─ RotaryEmbedding.forward_cuda(...)          # 社区代码，不改
            └─ vllm._custom_ops.rotary_embedding(...)     # 社区包装，不改
                 └─ torch.ops._C.rotary_embedding(...)    # PyTorch dispatcher
                      └─ vllm_iluvatar::rotary_embedding(...)  # 我们的实现
```

我们只提供最底层实现，上层全部复用社区代码。上游版本升级时，这些上层代码自动跟随，无维护负担。

### 3.2 确认上游 schema

community 算子的第一步是照抄上游 schema，一字不差。

Schema 来源：上游 vLLM 仓库 `csrc/libtorch_stable/torch_bindings.cpp`。

以 `rotary_embedding` 为例：

```text
rotary_embedding(Tensor positions, Tensor! query, Tensor!? key,
                 int head_size, Tensor cos_sin_cache, bool is_neox,
                 int rope_dim_offset=0, bool inverse=False) -> ()
```

确认要点：

- **in-place 标记**：`Tensor!` = 原地修改；`Tensor!?` = 可选且原地；`-> ()` = 无返回值。
- **默认值**：`rope_dim_offset=0`、`inverse=False` 必须与上游一致。
- **dtype / shape 约定**：在上游实现文件中确认 positions 类型（int64）、query 布局（扁平 `[*, H*D]` 或带头维 `[*, H, D]`）、cache dtype 是否可异构等。

### 3.3 编写 CUDA 内核

在 `csrc/community/` 下新增 `.cu` 文件，实现 `vllm_iluvatar` 命名空间中的 host 函数。

host 端负责 shape / stride 推导与校验（建议逐行对齐上游同名内核的 host 逻辑），device 端负责实际计算。

参考：[`csrc/community/pos_encoding_kernels.cu`](../../csrc/community/pos_encoding_kernels.cu)

```cpp
void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      c10::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox,
                      int64_t rope_dim_offset, bool inverse) {
  const int64_t num_tokens = positions.numel();
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");

  const int num_heads = (query.numel() / num_tokens) / head_size;
  const int rot_dim = cos_sin_cache.size(1);
  TORCH_CHECK((rot_dim + rope_dim_offset) <= head_size, ...);

  const int64_t head_stride =
      (query.dim() == positions.dim() + 2) ? query.stride(-2) : head_size;
  ...
}
```

实现要点：

- **dtype dispatch**：对 query dtype（fp32/fp16/bf16）与 cache dtype 分别派发，累加用 fp32 保精度。
- **向量化**：连续布局下按 `PACK_NUM=2` 向量化访存，非连续时回退到标量路径。
- **原地写回**：`Tensor!` 语义要求直接修改 query/key 的存储。

### 3.4 声明函数签名

在 [`csrc/community/ops.h`](../../csrc/community/ops.h) 中补充声明（注释附上社区 schema 原文）：

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

### 3.5 注册 schema 与 dispatch

在 [`csrc/torch_bindings.cpp`](../../csrc/torch_bindings.cpp) 中完成两件事：

1. `ops.def(...)` 声明 schema（与上游逐字一致）
2. `ops.impl(...)` 绑定到实现（选对 dispatch key）

```cpp
TORCH_LIBRARY_FRAGMENT_EXPAND(_C, ops) {
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox, "
      "int rope_dim_offset=0, bool inverse=False) -> ()");
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CUDA, ops) {
  ops.impl("rotary_embedding", &vllm_iluvatar::rotary_embedding);
}
```

注意：

- 使用 `TORCH_LIBRARY_FRAGMENT`（非 `TORCH_LIBRARY`）——插件**扩展** `_C` 命名空间，不独占。
- CUDA 算子注册到 `CUDA` key；CPU 输入的算子注册到 `CPU` key（如 `get_cuda_view_from_cpu_tensor`）。
- community 算子的 Meta impl 由上游提供，插件侧一般不需要注册。

### 3.6 构建

[`CMakeLists.txt`](../../CMakeLists.txt) 用 `GLOB_RECURSE ... CONFIGURE_DEPENDS` 递归收集 `csrc/` 下所有源文件：

```cmake
file(GLOB_RECURSE VLLM_ILUVATAR_EXT_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cu")
```

新增的 `.cu` / `.cpp` 文件会被自动发现，**不需要手动改 CMake**。

### 3.7 （可选）清理旧 OOT 覆盖

community 算子注册后，社区调用点会自动通过 PyTorch dispatcher 命中插件实现。如果插件里此前已有 Python 层的 OOT 覆盖来实现同一功能，现在可以清理掉。纯新增算子跳过此步。

清理三处：

1. **`vllm_iluvatar/_custom_ops.py`**：删除该算子的 override 函数，从 `_ILU_OVERRIDES` 移除对应条目。
2. **OOT 子类文件**：删除 `model_executor/layers/` 下逐类的 `forward_cuda` 子类。
3. **`vllm_iluvatar/ilu_patch/patch_custom_ops.py`**：从 `_SKIP_SIGCHECK` 移除该算子名（签名已对齐，不再需要豁免）。

### 3.8 测试

**文件位置**：`tests/kernels/core/test_<op>.py`

**两层测试策略**：

- **模块级**：通过上游构建函数（如 `get_rope`）构造完整模块，验证 `forward_cuda` → `_C` 内核的完整路径。
- **裸算子级**：直接调用 `torch.ops._C.<op>`，与纯 torch 参考实现对比，覆盖模块级测不到的边界。

**覆盖范围**——对齐上游同名测试的场景，再补 Iluvatar 特有边界：

- `rope_dim_offset > 0`、`inverse=True`
- GQA（`num_kv_heads < num_heads`）
- cache dtype 与 q/k 异构（fp32 cache + bf16 q/k）
- 非连续（strided）张量

**注意事项**：构造 `CustomOp`（如 `RotaryEmbedding`）需要活跃的 `VllmConfig` 上下文，使用 [`tests/conftest.py`](../../tests/conftest.py) 中的 `default_vllm_config` fixture 即可。

---

## 4. plugin 算子完整流程

> 示例：`apply_rotary_emb` · 参考实现：[`csrc/plugin/apply_rotary_emb.cu`](../../csrc/plugin/apply_rotary_emb.cu)

plugin 算子是上游 `torch.ops._C` 中**不存在**的算子，由插件自行新增。社区 Python 代码不会自动调用它，因此除了 C++ 侧的内核实现外，还**必须编写 OOT override** 将社区调用点接到新算子。

### 4.1 调用链

与 community 算子不同，plugin 算子多了一层**插件侧的 OOT override**（`@<CustomOp>.register_oot` 子类），由它在 `forward_cuda` 中调用 `torch.ops._C.<plugin_op>`。

以 `apply_rotary_emb` 为例：

```text
Qwen3Attention.forward
  └─ self.rotary_emb(positions, q, k)
       └─ RotaryEmbedding.forward_cuda(...)
            └─ ApplyRotaryEmb.forward_cuda(...)        # 社区 CustomOp
                 └─ IluApplyRotaryEmb.forward_cuda(...)    # 插件 OOT override
                      └─ torch.ops._C.apply_rotary_emb(...)   # PyTorch dispatcher
                           └─ vllm_iluvatar::apply_rotary_emb(...)  # 我们的实现
```

注意：`IluApplyRotaryEmb` 是插件自行编写维护的 OOT 子类。上游 `ApplyRotaryEmb` 的接口变更时，此 override 需要同步更新。

### 4.2 定义 schema

plugin 算子在上游没有对应的 schema，需要自行定义。设计 schema 时注意：

- 参数类型和语义要与社区调用侧的数据流吻合（OOT override 会从社区拿到数据再传给 `_C` 算子）。
- 选择 in-place（`Tensor!`、`-> ()`）还是 out-of-place（`-> Tensor`）需考虑调用语义。
- out-of-place 算子**必须**配套 Meta impl（见 4.3）。

以 `apply_rotary_emb` 为例，它是 out-of-place 的（返回新 Tensor）：

```text
apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) -> Tensor
```

### 4.3 编写 CUDA 内核 + Meta impl

在 `csrc/plugin/` 下新增 `.cu` 文件，在同一文件中实现**两个**函数：

1. **CUDA 实现**：`vllm_iluvatar::apply_rotary_emb` —— 实际的 GPU 计算。
2. **Meta impl**：`vllm_iluvatar::apply_rotary_emb_meta` —— 仅做 shape 推导，用于 torch.compile / FakeTensor / opcheck。

参考：[`csrc/plugin/apply_rotary_emb.cu`](../../csrc/plugin/apply_rotary_emb.cu)

CUDA 实现（host 入口）：

```cpp
torch::Tensor apply_rotary_emb(torch::Tensor& x, torch::Tensor& cos,
                               torch::Tensor& sin, bool interleaved) {
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4, ...);
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, ...);

  const int64_t rot_dim = cos.size(1) * 2;
  auto out = x.contiguous().clone();
  if (num_tokens == 0 || num_heads == 0 || rot_dim == 0) return out;
  // ... launch kernel ...
  return out;
}
```

Meta impl：

```cpp
torch::Tensor apply_rotary_emb_meta(torch::Tensor& x, torch::Tensor& /*cos*/,
                                    torch::Tensor& /*sin*/,
                                    bool /*interleaved*/) {
  return at::empty_like(x, x.options(), at::MemoryFormat::Contiguous);
}
```

### 4.4 声明函数签名

在 [`csrc/plugin/ops.h`](../../csrc/plugin/ops.h) 中补充声明，**包含 CUDA 实现和 Meta impl 两个签名**：

```cpp
namespace vllm_iluvatar {

// apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) -> Tensor
torch::Tensor apply_rotary_emb(torch::Tensor& x, torch::Tensor& cos,
                               torch::Tensor& sin, bool interleaved);

torch::Tensor apply_rotary_emb_meta(torch::Tensor& x, torch::Tensor& cos,
                                    torch::Tensor& sin, bool interleaved);

}  // namespace vllm_iluvatar
```

### 4.5 注册 schema 与 dispatch

在 [`csrc/torch_bindings.cpp`](../../csrc/torch_bindings.cpp) 中完成**三件事**（比 community 多一个 Meta 注册）：

1. `ops.def(...)` 声明自定义 schema
2. `ops.impl(...)` 绑定 CUDA 实现
3. `ops.impl(...)` 绑定 Meta impl

```cpp
TORCH_LIBRARY_FRAGMENT_EXPAND(_C, ops) {
  ops.def(
      "apply_rotary_emb(Tensor x, Tensor cos, Tensor sin, bool interleaved) "
      "-> Tensor");
}

TORCH_LIBRARY_IMPL_EXPAND(_C, CUDA, ops) {
  ops.impl("apply_rotary_emb", &vllm_iluvatar::apply_rotary_emb);
}

TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, ops) {
  ops.impl("apply_rotary_emb", &vllm_iluvatar::apply_rotary_emb_meta);
}
```

注意：**Meta impl 是 plugin 算子的必要条件**。没有它，torch.compile 和 opcheck 都会报错。community 算子的 Meta impl 由上游提供，这里不需要。

### 4.6 编写 OOT override

这是 plugin 算子与 community 算子最大的区别。社区代码不会自动调用 plugin 算子，必须通过 vLLM 的 CustomOp OOT 扩展点将调用接入。

具体做法：在 `vllm_iluvatar/model_executor/layers/` 下对应模块中编写 `@<CustomOp>.register_oot` 子类，重写 `forward_cuda`。

参考：[`vllm_iluvatar/model_executor/layers/rotary_embedding/common.py`](../../vllm_iluvatar/model_executor/layers/rotary_embedding/common.py)

```python
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

@ApplyRotaryEmb.register_oot
class IluApplyRotaryEmb(ApplyRotaryEmb):
    def forward_cuda(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)
        interleaved = not self.is_neox_style
        output = torch.ops._C.apply_rotary_emb(x, cos, sin, interleaved)
        return self._post_process(output, origin_shape, origin_dtype)
```

编写要点：

- **最大化复用社区逻辑**：只替换社区 `forward_cuda` 中无法在 Iluvatar 上运行的部分（本例是 flash-attn 的 `apply_rotary_emb` import），其余逻辑（`_pre_process` / `_post_process`）直接复用。
- **标注删除条件**：在文件头部注释说明何时可以移除此 override（如"当社区 flash-attn rotary 在 Iluvatar 可用时"），便于后续清理。
- **维护义务**：上游 `ApplyRotaryEmb` 接口变更时，此 override 需同步更新。

### 4.7 构建

与 community 算子相同，CMake `GLOB_RECURSE` 会自动发现 `csrc/plugin/` 下新增的源文件，**不需要手动改 CMake**。

但 OOT override 的 Python 文件（如 `model_executor/layers/rotary_embedding/common.py`）需确保被插件的入口逻辑加载（通常通过 `vllm_iluvatar/ilu_register/` 中的自动导入机制完成）。

### 4.8 测试

**文件位置**：`tests/kernels/core/test_<op>.py`

**两层测试策略**：

- **裸算子级**：直接调用 `torch.ops._C.<op>`，与纯 torch 参考实现对比，验证内核数值正确性。
- **模块级**：通过 OOT override 所在的 CustomOp 模块（如 `ApplyRotaryEmb`）进行端到端验证，确认 override → `_C` 内核的完整路径。

plugin 算子没有上游同名测试可以对齐，需自行设计参考实现和覆盖范围。

**注意事项**：构造 `CustomOp` 需要活跃的 `VllmConfig` 上下文，使用 [`tests/conftest.py`](../../tests/conftest.py) 中的 `default_vllm_config` fixture 即可。

---

## 5. 附录

### 关键文件清单

**community 算子相关：**

| 作用 | 所在仓库 | 文件 |
| --- | --- | --- |
| schema 来源 | 上游 vllm | `csrc/libtorch_stable/torch_bindings.cpp` |
| 社区包装函数 | 上游 vllm | `vllm/_custom_ops.py` |
| 内核实现（示例） | vllm-iluvatar | [`csrc/community/pos_encoding_kernels.cu`](../../csrc/community/pos_encoding_kernels.cu) |
| 函数声明 | vllm-iluvatar | [`csrc/community/ops.h`](../../csrc/community/ops.h) |
| 统一注册入口 | vllm-iluvatar | [`csrc/torch_bindings.cpp`](../../csrc/torch_bindings.cpp) |
| 内核测试（示例） | vllm-iluvatar | [`tests/kernels/core/test_pos_encoding.py`](../../tests/kernels/core/test_pos_encoding.py) |

**plugin 算子相关：**

| 作用 | 所在仓库 | 文件 |
| --- | --- | --- |
| 内核实现 + Meta impl（示例） | vllm-iluvatar | [`csrc/plugin/apply_rotary_emb.cu`](../../csrc/plugin/apply_rotary_emb.cu) |
| 函数声明 | vllm-iluvatar | [`csrc/plugin/ops.h`](../../csrc/plugin/ops.h) |
| 统一注册入口 | vllm-iluvatar | [`csrc/torch_bindings.cpp`](../../csrc/torch_bindings.cpp) |
| OOT override（示例） | vllm-iluvatar | [`vllm_iluvatar/model_executor/layers/rotary_embedding/common.py`](../../vllm_iluvatar/model_executor/layers/rotary_embedding/common.py) |
| 内核测试（示例） | vllm-iluvatar | [`tests/kernels/core/test_apply_rotary_emb.py`](../../tests/kernels/core/test_apply_rotary_emb.py) |

**共用：**

| 作用 | 文件 |
| --- | --- |
| 构建 | [`CMakeLists.txt`](../../CMakeLists.txt) |
| 注册宏 | [`csrc/core/registration.h`](../../csrc/core/registration.h) |

### 步骤速览

**community 算子**（上游有同名 op）：

1. 照抄上游 schema（来源：`csrc/libtorch_stable/torch_bindings.cpp`）。
2. 在 `csrc/community/` 新增 `.cu` / `.cpp` 实现。
3. 在 `csrc/community/ops.h` 声明函数。
4. 在 `csrc/torch_bindings.cpp` 注册 schema + CUDA dispatch。
5. 构建（CMake 自动发现）。
6. （可选）清理旧 OOT 覆盖。
7. 添加 `tests/kernels/core/test_<op>.py`，模块级 + 裸算子级两层测试。

**plugin 算子**（上游无同名 op）：

1. 自行定义 schema。
2. 在 `csrc/plugin/` 新增 `.cu` 实现（含 **Meta impl**）。
3. 在 `csrc/plugin/ops.h` 声明函数（含 meta 声明）。
4. 在 `csrc/torch_bindings.cpp` 注册 schema + CUDA dispatch + **Meta impl**。
5. **编写 OOT override**（`@<CustomOp>.register_oot` 子类）。
6. 构建（CMake 自动发现 C++ 文件；确保 OOT override Python 文件被加载）。
7. 添加 `tests/kernels/core/test_<op>.py`，裸算子级 + OOT override 模块级两层测试。

`rotary_embedding`（`csrc/community/`）和 `apply_rotary_emb`（`csrc/plugin/`）分别是两类算子的完整参考实现。
