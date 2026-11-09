# 使用镜像安装（推荐）



镜像与驱动从[天数智芯开发者社区](https://developer.iluvatar.com/)下载，下载位置见[从开发者社区下载](#download)。装完后继续执行[自检](installation_check.md)，确认当前服务加载的是天数 `vllm-iluvatar` 插件。

环境要求见[安装前确认](index.md#env)。

<a id="download"></a>

## 从开发者社区下载



镜像与驱动都在[天数智芯开发者社区](https://developer.iluvatar.com/)下载，按宿主机 CPU 架构（x86_64 或 aarch64）选择对应文件。

- 镜像：[大模型推理 · vllm-iluvatar](https://developer.iluvatar.com/mirrors?category=%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86&image=vllm-iluvatar)
- 驱动：[ix-driver](https://developer.iluvatar.com/mirrors?category=%E5%85%A8%E9%83%A8&image=ix-driver)

镜像、驱动、CoreX toolkit 与模型权重格式要来自同一轮发布，版本不配套时不要继续。

## 安装步骤



可以通过开发者社区下载对应版本的包。推荐顺序如下：

1. 在宿主机安装匹配的驱动（从上面的社区页按架构下载）。
2. 拉取或加载对应架构的 `vllm-iluvatar` 镜像。
3. 启动容器时映射天数设备，并注入运行所需的驱动能力。
4. 进入容器后检查驱动、设备、CoreX runtime 和 Python 包。
5. 按[自检](installation_check.md)确认 `vllm` 与 `vllm-iluvatar` 版本配套，并确认插件 entry point 正常加载。

<a id="run"></a>

## 拉取镜像与启动容器



下面是一个可照抄的模板。命令里的 `tag` 是占位符，**实际 tag 以[开发者社区镜像页](https://developer.iluvatar.com/mirrors?category=%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86&image=vllm-iluvatar)上对应架构的镜像为准**（x86_64 与 aarch64 的 tag 不同），替换后再执行：

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



从社区页下载对应架构的驱动包后，按社区页说明在宿主机安装。安装完成后驱动会自动加载，可用 `ixsmi` 确认设备可见。

两种使用方式：

- **装在宿主机，再映射进容器**：宿主机装好驱动后，启动容器时把宿主机的 CoreX runtime 文件映射进容器。要映射哪些文件见[容器启动要点](#container)，可照抄的命令见[拉取镜像与启动容器](#run)。
- **直接在容器里装**：不做映射，进容器后在容器内按社区页说明安装驱动，适合不想改动宿主机环境的场景。

宿主机已有旧版本时，先清理再装，避免冲突。

<a id="container"></a>

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



遇到以下情况时，不要继续启动服务，先回到[版本说明](../reference/release_notes/index.md)确认：

- `vllm` 与 `vllm-iluvatar` 版本不配套。
- 驱动、CoreX toolkit 和容器镜像来源不一致。
- 模型权重仍是旧量化格式，但当前版本要求重新量化。
