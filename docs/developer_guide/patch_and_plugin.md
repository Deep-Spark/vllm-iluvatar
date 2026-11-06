# patch 与插件机制



vllm-iluvatar 的 patch 分两层：早期 import 兼容层解决「模块导入时就会失败」的问题，晚期 general patch 解决「模块已经初始化后再替换运行期行为」的问题。能用 vLLM 官方扩展点时优先使用官方扩展点，patch 只覆盖官方入口无法表达的部分。

## 为什么需要 `.pth`



vLLM 会启动 worker 进程，worker 可能在 platform/general plugin 生效前就导入社区模块。若这些模块在 import-time 访问了天数环境暂不支持的符号，单靠 entry point 已经太晚。

因此安装包会写入一个 `.pth` 启动钩子。它默认惰性，只有 `VLLM_ILUVATAR_ENABLE_STARTUP_PATCH=1` 时才调用 `vllm_iluvatar._load_import_compat_patch()`；这个环境变量由 `register_platform()` 设置，并被后续 worker 继承。

常规构建和 editable wheel 都要能带上这条 `.pth`，因此构建逻辑分别处理普通 `build_py` 和 editable wheel 的输出路径，确保两种安装方式行为一致。

## 早期 import 兼容层



`_load_import_compat_patch()` 直接按文件路径加载 `ilu_patch/patch_import_compat.py`，并用固定 `sys.modules` key 保证只执行一次。该阶段做的是最早必须完成的兼容工作，例如：

- 加载 `torch.accelerator`、symmetric memory、Triton Gluon 等 import-time shim。
- 注册模块替身 shim，例如 `vllm.vllm_flash_attn` 这类导入时必须存在的模块。
- 注册 post-import hook，用于在目标模块加载后再覆盖其中的符号。
- 提前导入 native `_C`，让 `torch.ops._C` 相关 schema 可用。

`import_hook.py` 使用 `sys.meta_path` 包装目标模块加载过程：先让原 loader 执行模块，再在 `exec_module` 之后触发回调。`patch_custom_ops.py` 就通过这个机制等待 `vllm._custom_ops` 导入完成，再替换或补充天数 custom ops。

## platform 阶段 import hook



部分 hook 不需要 `.pth` 那么早，但仍应避免 eager import 目标模块。这类逻辑由 `register_platform()` 调用 `patch_platform_import_hooks.py` 注册，等目标模块真正导入后再执行回调。

## general patch 规范



general patch 在 `vllm.general_plugins` 阶段执行，此时 vLLM 模块对象已经初始化，适合替换函数、绑定已经复制出去的引用，或注册运行期行为。它不负责 import-time 兜底。

新增 general patch 时遵循三条边界：

- 实现代码放在与上游路径镜像的目录中，patch wiring 放在 `ilu_patch/general_patches`。
- 每个 wiring 模块暴露 `register()`，内部通过 `general_patch.register_once(__name__, _apply, detail=...)` 注册，保持幂等。
- 改写函数后如目标函数已被其它模块复制引用，使用 `rebind_module_func` 同步回绑，避免只改到定义处。

`ilu_patch/__init__.py` 会非递归扫描 `ilu_patch.general_patches` 下的模块并调用 `register()`。
