# 安装



先做安装前确认，再按侧边栏选择安装路径，装完做安装自检。

<a id="env"></a>

## 安装前确认



不要只替换单个 wheel 或单个库文件。vLLM 本体、`vllm-iluvatar` 插件、CoreX toolkit、驱动、模型权重格式和启动参数需要来自同一轮说明。

安装前至少确认：

- 安装路径用侧边栏「安装」下面的页面。
- 宿主机驱动与镜像或 run 包中的 CoreX runtime 匹配。
- 模型权重和量化配置适用于当前版本。

选择下面一页继续：

- 安装：[镜像安装](image_installation.md)
- 装完后：[安装自检](installation_check.md)。自检通过后回到[快速开始](../quickstart.md)
