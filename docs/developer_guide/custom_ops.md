# 自定义算子



本页写给要在 vllm-iluvatar 插件里**新增一个 `torch.ops._C` 算子**的开发者。上游 vLLM 的底层算子通过 `torch.ops._C.<op>` 调用，在插件里这个扩展是 `vllm_iluvatar._C`（`import vllm_iluvatar._C` 触发 `TORCH_LIBRARY_FRAGMENT` 静态注册）。改动全程只在插件仓库内。

## 前置判断：community 还是 plugin



先按「上游是否已有同名同 schema 的算子」确定归属，两类算子的**调用机制和开发流程有本质区别**：

| | community 算子 | plugin 算子 |
| --- | --- | --- |
| 含义 | 上游 vLLM 已有同名同 schema 的 `torch.ops._C` 算子 | 上游没有、插件自行新增 |
| 调用机制 | 照抄上游 schema 后，社区代码经 PyTorch dispatcher **自动命中**插件实现，上层不改 | 社区代码不会自动调用，**必须新增 OOT override** 把调用点接到插件实现 |
| schema 来源 | 逐字照抄上游 `csrc/libtorch_stable/torch_bindings.cpp` | 自行定义 |
| 源文件位置 | `csrc/community/<op>.cu` 或 `.cpp` | `csrc/plugin/<op>.cu` |
| Meta impl | 上游已有，一般不需要 | **必须提供**（torch.compile / opcheck 需要） |
| 典型示例 | `rotary_embedding`、`rms_norm` | `apply_rotary_emb` |

判断流程：`上游 torch.ops._C 有同名同 schema 的算子？` 是 → community（第 3 章），否 → plugin（第 4 章）。

## 流程一：community 算子



（示例 `rotary_embedding`）我们只提供最底层实现，上层社区代码全部复用；上游升级时上层自动跟随，维护负担低。

关键步骤 + 文件落位 + 注册点：

1. **确认上游 schema**：一字不差照抄上游 `csrc/libtorch_stable/torch_bindings.cpp`（注意 in-place 标记、默认值、dtype/shape 约定）。
2. **编写 CUDA 内核**：在 `csrc/community/` 下新增 `.cu` / `.cpp`，实现 `vllm_iluvatar` 命名空间中的 host 函数（host 做 shape/stride 推导与校验，建议逐行对齐上游）。
3. **声明函数签名**：在 `csrc/community/ops.h` 补充声明（注释附社区 schema 原文）。
4. **注册 schema 与 dispatch**：在 `csrc/torch_bindings.cpp` 用 `ops.def(...)`（schema 与上游逐字一致）+ `ops.impl(...)` 绑定实现。用 `TORCH_LIBRARY_FRAGMENT`（扩展 `_C` 而非独占）；CUDA 算子注册到 `CUDA` key。community 算子的 Meta impl 由上游提供，一般不需注册。
5. **构建**：`CMakeLists.txt` 用 `GLOB_RECURSE ... CONFIGURE_DEPENDS` 递归收集 `csrc/`，新增文件自动发现，**不需手改 CMake**。
6. **（可选）清理旧 OOT 覆盖**：若此前有 Python 层 OOT 覆盖实现同功能，现可清理三处——`vllm_iluvatar/_custom_ops.py`（删 override、从 `_ILU_OVERRIDES` 移除）、`model_executor/layers/` 下逐类的 `forward_cuda` 子类、`ilu_patch/patch_custom_ops.py`（从 `_SKIP_SIGCHECK` 移除该算子名）。
7. **测试**：`tests/kernels/core/test_<op>.py`，模块级 + 裸算子级两层。

## 流程二：plugin 算子



（示例 `apply_rotary_emb`）plugin 算子是上游 `torch.ops._C` 中不存在的算子，社区代码不会自动调用它，因此除内核实现外**必须编写 OOT override** 把社区调用点接到新算子。

关键步骤 + 文件落位 + 注册点（比 community 多 Meta impl 与 OOT override）：

1. **定义 schema**：上游无对应，自行定义；注意参数类型要与社区调用侧数据流吻合，in-place / out-of-place 的选择，out-of-place 必须配 Meta impl。
2. **编写 CUDA 内核 + Meta impl**：在 `csrc/plugin/` 下新增 `.cu`，同一文件实现 CUDA 实现（`vllm_iluvatar::<op>`）与 Meta impl（`vllm_iluvatar::<op>_meta`，仅做 shape 推导）。
3. **声明函数签名**：在 `csrc/plugin/ops.h` 声明，**含 CUDA 实现与 Meta impl 两个签名**。
4. **注册 schema 与 dispatch**：在 `csrc/torch_bindings.cpp` 做三件事——`ops.def(...)` 声明自定义 schema、`ops.impl(..., CUDA)` 绑定 CUDA 实现、`ops.impl(..., Meta)` 绑定 Meta impl。**Meta impl 是 plugin 算子的必要条件**，缺它 torch.compile / opcheck 报错。
5. **编写 OOT override**（与 community 最大区别）：在 `vllm_iluvatar/model_executor/layers/` 下对应模块编写 `@<CustomOp>.register_oot` 子类，重写 `forward_cuda` 调 `torch.ops._C.<plugin_op>`。要点：最大化复用社区逻辑（`_pre_process` / `_post_process`）、文件头注明删除条件、上游接口变更需同步维护。
6. **构建**：C++ 文件同样由 CMake `GLOB_RECURSE` 自动发现；但 OOT override 的 Python 文件需确保被插件入口逻辑（`ilu_register/` 自动导入机制）加载。
7. **测试**：`tests/kernels/core/test_<op>.py`，裸算子级 + OOT override 模块级两层；无上游同名测试可对齐，需自行设计参考实现。
