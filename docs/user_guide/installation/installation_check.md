# 自检



本页说明安装完成后的环境自检项、验证命令和常见检查结果。

## 基础环境检查



安装完成后，先确认 CoreX runtime 文件和设备检查工具 ixsmi 可用，再检查 Python 依赖、vLLM 命令和核心 Python 包是否可用。

```bash
command -v ixsmi && ixsmi

for p in \
  /usr/local/corex/lib64/libcuda.so \
  /usr/local/corex/lib64/libcuda.so.1 \
  /usr/local/corex/lib64/libixml.so \
  /usr/local/corex/bin/ixsmi; do
  test -e "$p" || { echo "missing CoreX runtime file: $p"; exit 1; }
done
```

如果 `ixsmi` 不存在，或上述 CoreX runtime 文件缺失，先回到安装步骤确认驱动、toolkit 与容器映射是否来自同一轮发布物，不要继续做 vLLM 层自检。

```bash
pip3 check                              # 期望：No broken requirements found
vllm --version                          # 期望：退出码 0
python3 -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

## 插件加载检查



确认 `vllm-iluvatar` 插件确实被 vLLM 加载。vLLM v0.27.1 使用 out-of-tree plugin 架构后，这是必须检查的一项。

```bash
python3 -c "
from importlib.metadata import entry_points
for g in ('vllm.platform_plugins', 'vllm.general_plugins'):
    print(g, [e.name for e in entry_points(group=g)])
"
```

两组输出都应出现 `iluvatar`。如果为空，通常说明插件未安装、安装环境不一致，或者安装成了其它同名包。

## 自检通过之后



两组插件名都出现 `iluvatar` 后，回到[快速开始](../quickstart.md)启动示例模型。检查失败时留在本页和[安装页](index.md)。
