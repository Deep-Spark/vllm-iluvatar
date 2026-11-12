# 安装

先做安装前确认，再按镜像安装，装完做安装自检。三部分都在本页。

## 安装前确认

不要只替换单个 wheel 或单个库文件。vLLM 本体、`vllm-iluvatar` 插件、CoreX toolkit、驱动、模型权重格式和启动参数需要来自同一轮说明。

安装前至少确认：

- 宿主机驱动与镜像或 run 包中的 CoreX runtime 匹配。
- 模型权重和量化配置适用于当前版本，旧量化格式需要重新生成，见[版本说明](../release/index.md)。

装完后到本页的[安装自检](#安装自检)，自检通过后回到[快速开始](quickstart.md)。

## 从开发者社区下载

镜像与驱动都在[天数智芯开发者社区](https://developer.iluvatar.com/)下载，按宿主机 CPU 架构（x86_64 或 aarch64）和 `vllm-iluvatar` 发布版本选择对应文件。

- 镜像：[大模型推理 · vllm-iluvatar](https://developer.iluvatar.com/mirrors?category=%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86&image=vllm-iluvatar)
- 驱动：[ix-driver](https://developer.iluvatar.com/mirrors?category=%E5%85%A8%E9%83%A8&image=ix-driver)

!!! note "版本需配套"
    镜像、驱动、CoreX toolkit 与模型权重格式要来自同一轮发布，版本不配套时不要继续。

## 从源码构建镜像

仓库自带多阶段 `Dockerfile`（基础环境与源码组件分层）与 `scripts/build_image.sh`，适合预发布版本或需要自定 pin 的场景。

构建方式、构建参数、缓存复用规则，以及如何查看镜像里锁定的源码版本，见 [docker/README.md](../../docker/README.md)。

!!! note "驱动仍需单独安装"
    CoreX 驱动**不打进镜像**，需要在（特权）容器内单独安装，见[驱动安装](#驱动安装)。

## 安装步骤

推荐顺序如下：

1. 在宿主机安装匹配的驱动（从上面的社区页按架构下载）。
2. 拉取或加载对应架构的 `vllm-iluvatar` 镜像。
3. 启动容器时映射天数设备，并注入运行所需的驱动能力。
4. 进入容器后检查驱动、设备、CoreX runtime 和 Python 包。
5. 按[安装自检](#安装自检)确认 `vllm` 与 `vllm-iluvatar` 版本配套，并确认插件 entry point 正常加载。

## 拉取镜像与启动容器

下面是一个可照抄的模板。命令里的 `tag` 是占位符，**实际 tag 以[开发者社区镜像页](https://developer.iluvatar.com/mirrors?category=%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86&image=vllm-iluvatar)上对应架构和版本的镜像为准**，替换后再执行：

```bash
# 拉取镜像（tag 替换为社区页上对应架构的实际 tag）
docker pull dev-community-acr-registry.cn-shanghai.cr.aliyuncs.com/dev-community/vllm-iluvatar:tag

# 采用「宿主机装驱动 + 映射」方式时，先确认四个 CoreX runtime 文件存在；
# 缺失时停住，不要带着缺失的映射启动容器
for p in \
  /usr/local/corex/lib64/libcuda.so \
  /usr/local/corex/lib64/libcuda.so.1 \
  /usr/local/corex/lib64/libixml.so \
  /usr/local/corex/bin/ixsmi; do
  test -e "$p" || { echo "missing CoreX runtime file: $p"; exit 1; }
done

# 创建并进入容器
docker run -dit \
  -v /usr/local/corex/lib64/libcuda.so:/usr/local/corex/lib64/libcuda.so \
  -v /usr/local/corex/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so.1 \
  -v /usr/local/corex/lib64/libixml.so:/usr/local/corex/lib64/libixml.so \
  -v /usr/local/corex/bin/ixsmi:/usr/local/corex/bin/ixsmi \
  -v /data:/data \
  -v /usr/src:/usr/src \
  -v /lib/modules:/lib/modules \
  -v /dev:/dev \
  --pid=host --ipc=host --privileged --cap-add=ALL --network=host \
  --name vllm-iluvatar \
  dev-community-acr-registry.cn-shanghai.cr.aliyuncs.com/dev-community/vllm-iluvatar:tag

docker exec -it vllm-iluvatar bash
```

采用「直接在容器里装驱动」方式时，去掉上面四条 CoreX 文件映射，进容器后按社区页说明安装驱动。

## 驱动安装

从社区页下载对应架构的驱动包后，按社区页说明安装。安装完成后驱动会自动加载，可用命令 `ixsmi` 确认设备可见。

驱动两种安装方式：

- **装在宿主机，再映射进容器**：宿主机装好驱动后，启动容器时把宿主机的 CoreX runtime 文件映射进容器。要映射哪些文件见[容器启动要点](#容器启动要点)，可照抄的命令见[拉取镜像与启动容器](#拉取镜像与启动容器)。
- **直接在容器里装**：不做映射，进容器后在容器内按社区页说明安装驱动，适合不想改动宿主机环境的场景。

!!! note "避免驱动冲突"
    宿主机已有旧版本时，先清理再装，避免冲突。

## 容器启动要点

用社区页下载的镜像启动容器。若采用「宿主机装驱动 + 映射」的方式，需要把以下 CoreX runtime 文件映射进容器；**文件缺失时先停住，不要带着缺失的映射启动容器**：

```text
/usr/local/corex/lib64/libcuda.so
/usr/local/corex/lib64/libcuda.so.1
/usr/local/corex/lib64/libixml.so
/usr/local/corex/bin/ixsmi
```

启动容器时还需映射天数设备与宿主机内核相关目录（如 `/dev`、`/usr/src`、`/lib/modules`），并开启容器访问设备所需的权限。上面模板为快速跑通使用了 `--privileged --cap-add=ALL` 等最宽权限，生产环境请按最小权限裁剪，仅授予访问天数设备所需的能力。

镜像已经包含 SDK、CoreX toolkit 和 `vllm-iluvatar` 运行依赖。普通部署不要在镜像里手动升级或重装 `vllm`、`vllm-iluvatar`、`ixformer` 等包。

## 什么时候不要继续安装

遇到以下情况时，不要继续启动服务，先回到[版本说明](../release/0.27.1/index.md)确认：

- `vllm` 与 `vllm-iluvatar` 版本不配套。
- 驱动、CoreX toolkit 和容器镜像来源不一致。
- 模型权重仍是旧量化格式，但当前版本要求重新量化。

## 安装自检

本页说明安装完成后的环境自检项、验证命令和常见检查结果。

### 基础环境检查

安装完成后，先确认 CoreX runtime 文件和设备检查工具 `ixsmi` 可用，再检查 Python 依赖、vLLM 命令和核心 Python 包是否可用。

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

如果 `ixsmi` 不存在，或上述 CoreX runtime 文件缺失，先回到[安装步骤](#安装步骤)确认驱动、toolkit 与容器映射是否来自同一轮发布物，不要继续做 vLLM 层自检。

```bash
pip3 check                              # 期望：No broken requirements found
vllm --version                          # 期望：退出码 0
python3 -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

### 插件加载检查

确认 `vllm-iluvatar` 插件确实被 vLLM 加载。vLLM 使用 out-of-tree plugin 架构后，这是必须检查的一项。

```bash
python3 -c "
from importlib.metadata import entry_points
for g in ('vllm.platform_plugins', 'vllm.general_plugins'):
    print(g, [e.name for e in entry_points(group=g)])
"
```

两组输出都应出现 `iluvatar`。如果为空，通常说明插件未安装、安装环境不一致，或者安装成了其它同名包。

## 自检通过之后

两组插件名都出现 `iluvatar` 后，回到[快速开始](quickstart.md)启动示例模型。检查失败时留在本页的[安装前确认](#安装前确认)与[安装自检](#安装自检)排查。
