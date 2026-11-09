# 如何搭建 vllm-iluvatar 开发环境

本仓库只包含 vLLM Iluvatar 插件。vLLM、CoreX PyTorch、Triton 和 ixformer
由匹配的 CoreX 软件环境提供，源码不作为 Git 子模块随本仓库发布。
版本要求见 [README](../../../README.md#requirements) 和
[version.json](../../../version.json) 的 `dependencies`。

## 环境检查

先安装 CoreX 驱动、SDK、匹配的 Python 软件包，以及 CMake 和 Ninja。
确认当前 Python 环境能导入运行依赖：

```bash
python3 -c "import torch, triton, ixformer, vllm; print(torch.__version__, triton.__version__, vllm.__version__)"
cmake --version
ninja --version
```

设置 SDK 路径，并将目标架构改成实际 GPU 对应的值。原生扩展需要在
CoreX 环境中编译；不要用普通 PyPI 的 NVIDIA PyTorch 替换 CoreX PyTorch。

```bash
export COREX_HOME=/usr/local/corex
export COREX_ARCH=ivcore11
```

## 安装插件

在仓库根目录安装 Python 运行依赖和构建工具，再进行 editable 安装：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install 'setuptools>=77.0.3,<81' wheel
python3 -m pip install -e . --no-build-isolation --no-deps
```

`--no-build-isolation` 使用已有的 CoreX 构建环境；`--no-deps` 避免
editable 安装重复解析已准备好的平台依赖。CMake 通过已安装的 ixformer
包定位头文件和动态库，不需要本地子模块源码。

## 验证

```bash
python3 -c "import torch; import vllm_iluvatar._C; from vllm.platforms import current_platform; print(current_platform)"
```

输出应包含 Iluvatar 平台。修改 Python 代码后，editable 安装会直接使用
工作区中的代码；修改 C++/CUDA 代码或切换 `COREX_ARCH` 后，需要重新编译。
不同架构之间不要复用旧的原生扩展构建产物。

## 构建 wheel

```bash
bash build_vllm_iluvatar.sh
VLLM_ILUVATAR_INSTALL_RUNTIME_DEPS=1 bash install_vllm_iluvatar.sh
```

生成的 wheel 位于 `build_pip/`。发布前应在目标 CoreX 环境验证导入、
实际模型推理和需要使用的分布式配置。
