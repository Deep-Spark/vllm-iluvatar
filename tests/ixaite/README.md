# ixaite 测试与 benchmark

此目录集中维护 ixaite 接入、新增修复和双后端对照用例。原测试目录继续
验证 baseline，原用例保持 `9d6a55b` 原样。基线测试中的错误断言、
失效测试对象和架构条件等问题按冻结约定保留，不在本分支修正。

| 目录 | 内容 |
|---|---|
| `backend_isolation/` | 源码保护、后端选择、注册、启动及调用来源检查；也包含 baseline 的对照断言 |
| `benchmarks/` | 算子和整模型 benchmark、模型配置、报告输出及 benchmark 自身的测试 |
| `layers/`、`moe/`、`quantization/` | 算子数值、权重布局、量化及生命周期检查 |
| `attention/`、`models/`、`worker/`、`patches/`、`distributed/` | ixaite 私有接入及修复的回归检查 |

在仓库根目录、激活对应环境后运行：

```bash
# 原测试目录，保留 baseline 已有失败。
VLLM_ILU_KERNEL_PROVIDER=baseline python -m pytest tests --ignore=tests/ixaite

# ixaite 接入及对照测试，另启一个进程运行。
python -m pytest tests/ixaite

# benchmark 的两个入口。
python -m tests.ixaite.benchmarks --list
python -m tests.ixaite.benchmarks.e2e --list-models
```

两组测试分别运行，避免全局 OOT 注册表和模块缓存相互影响。不要将
两条 pytest 命令合并到同一进程，也不为合跑而修正旧用例或添加隐藏
的 baseline 兼容补丁。ixaite 目录内的 baseline 对照仍由具体用例管理。

benchmark 的用法见[使用说明](benchmarks/README.md)。默认输出为
`tests/ixaite/benchmarks/reports/` 和 `tests/ixaite/benchmarks/e2e-runs/`；
两者仍由 Git 忽略。

`tests` 使用明确的包名，避免 pytest 将本目录作为顶层 `ixaite` 导入，
遮蔽算子库。目录内的辅助模块通过 `tests.ixaite.*` 导入，共享 fixture
仍使用原来的入口。

baseline 保留已有问题。
本目录中的 baseline 对照用于标明行为边界，不表示对 baseline 增加修复。
CPU 或局部 GPU 用例通过也不代表整模型质量、性能及完整来源验收完成。
