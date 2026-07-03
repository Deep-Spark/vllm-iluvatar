# RFC 0001: vLLM-Iluvatar 分支管理与发版策略

> **状态:** 草案 | **作者:** Gaiqin.Bai | **日期:** 2026-07-03 | **适用:** `main` 及后续版本分支

---

## 目标

vLLM-Iluvatar 采用“主干优先、少量版本分支、临时分支及时删除”的分支策略。本文定义日常开发、版本维护、bug 修复同步、发版和分支清理规则。

这套策略的目标是避免长期分支过多导致的重复修复、CI 口径不一致、问题定位困难和版本说明混乱。

---

## 核心原则

1. **主干是唯一开发主线。** 通用 feature、bugfix、平台适配、重构都应先进入 `main`。
2. **版本分支只用于发版和维护。** 版本分支不承接新的通用功能开发，只接收必要修复、版本号、CI、构建和 release note 变更。
3. **临时分支不长期保留。** `feature/*`、`bugfix/*`、`sync/*` 只承载一次 PR，PR 合入或关闭后删除。
4. **版本分支修复必须检查主干。** 如果 bug 在版本分支上修复，必须确认 `main` 是否也存在相同问题；存在则也要向 `main` 提 PR。

---

## 分支模型

| 分支类型 | 示例 | 来源 | 用途 | 生命周期 |
|---|---|---|---|---|
| 主干分支 | `main` | 仓库默认分支 | 持续跟踪 vLLM upstream 和插件化主线，承接所有通用开发 | 长期保留 |
| 版本分支 | `v0.24.0` | 从 `main` 切出 | 对应一个 vLLM 基座版本，用于发版、验证和维护 | 仅保留维护窗口内版本 |
| 临时分支 | `feature/foo` / `bugfix/foo` / `sync/foo-to-main` | 从 `main` 切出 | 承载一次开发、修复或同步 PR | PR 结束后删除 |

默认长期分支只包含：

- 一个主干分支：`main`。
- 最多两个维护中的版本分支，例如 `v0.24.0` 和 `v0.23.0`。

版本分支和临时分支都必须从 `main` 切出。`feature/*`、`bugfix/*`、`sync/*` 只是临时分支在不同场景下的命名，不是新的分支类型。不为个人开发、客户交付或专项适配创建长期分支；确需短期隔离时，使用临时分支和 PR 管理，并在交付结束后删除，其中通用改动必须回到 `main`。

---

## 日常开发流程

所有开发都从 `main` 切出临时分支，通过 PR 合入 `main`。禁止直接 push 到 `main` 或受保护版本分支。

```bash
git checkout main
git pull
git checkout -b feature/my-change
# ... 开发、单测、必要的模型 smoke test ...
git push -u origin feature/my-change
gh pr create --base main
```

PR 合入前至少满足：

- 说明变更目的、影响范围和验证方式。
- 相关 CI 通过，或说明失败项与本 PR 无关。
- 涉及模型、算子、并行策略、量化、平台适配时，提供最小可复现验证命令或日志。
- 破坏兼容性的变更需要说明影响范围和迁移方式。

---

## 版本分支修复流程

版本分支只接收维护所需改动。允许进入版本分支的改动包括：

- 严重功能回归、性能回归、精度错误、死锁、崩溃、显著内存泄漏。
- 交付阻塞问题，且风险和验证范围可控。
- 对该版本已存在功能的修复或兼容性补齐。
- 版本分支自身的版本号、CI、构建脚本、release note、文档修正。

如果 bug 首先在版本分支上发现或修复，必须先检查 `main`：

1. `main` 也存在相同 bug：向 `main` 提修复 PR。版本分支急需交付时，可以同时向版本分支提等价修复 PR，但两个 PR 必须互相关联。
2. `main` 不存在该 bug：在版本分支 PR 中说明原因，例如主干已重构、相关逻辑已删除或已有等价修复。
3. 无法判断 `main` 是否受影响：在 PR 中说明验证方法、疑点和负责人，合入前由维护者确认是否需要主干 PR。

通用 bugfix 的最终状态必须是“`main` 已修复或明确不受影响”。版本分支不能成为唯一修复来源，否则后续从 `main` 切出的新版本可能重新带出同一个问题。

---

## 版本分支修复同步到主干流程

如果修复先进入版本分支，维护者必须把相同问题同步检查到 `main`。`main` 仍存在该 bug 时，从 `main` 切出临时 `sync/*` 分支，并通过 PR 合入 `main`。`sync/*` 归类为临时分支，不是独立分支类型。

```bash
# 1. 确认版本分支上的修复
git checkout v0.24.0
git pull

# 2. 检查 main 是否存在相同 bug
git checkout main
git pull
# ... 复现或代码检查 ...

# 3. 如果 main 也存在问题，从 main 切出临时同步分支
git checkout -b sync/fix-foo-to-main
# cherry-pick 版本分支上的修复，或按 main 代码重写等价修复
git cherry-pick <version-branch-fix-commit>

# 4. 提 PR 合入 main
git push -u origin sync/fix-foo-to-main
gh pr create --base main
```

冲突处理规则：

- 能直接 cherry-pick 的改动，优先直接 pick。
- 冲突较小且语义清晰的改动，解决冲突后继续 pick。
- 版本分支与 `main` 的 API 差异过大时，在 `main` 上重写等价修复，不强行保留原提交形态。

同步到 `main` 的 PR 必须说明对应的版本分支 PR 或 commit，以及是否为 cherry-pick 或等价修复。

---

## 发版流程

上游 vLLM 发版或主干升级时，先在 `main` 完成一整轮适配和验证，再从 `main` 切出版本分支。只有准备发布或进入维护窗口的版本才切分支，不为中间实验状态创建长期分支。

```bash
# 在主干完成适配
git checkout main
git pull
# 适配 platform / patch / models / ops / CI ...
# 全量编译、单测、核心模型 smoke test 通过

# 切出版本分支
git checkout -b v0.24.0
# 更新版本号、CHANGELOG、release note
git commit -m "release: v0.24.0"
git push -u origin v0.24.0

# 主干继续跟踪下一轮 upstream
git checkout main
# 更新开发版本号或下一阶段适配说明
git commit -m "version: bump dev after v0.24.0"
```

上游 vLLM 发 patch 版本时，vLLM-Iluvatar 复用对应的已有版本分支，不再为上游 patch 单独创建长期分支。

例如当前 vLLM-Iluvatar 已有 `v0.24.0` 版本分支，对应上游 vLLM `0.24.0`。当上游 vLLM 发布 `0.24.1` 时，处理方式是：

1. 在 `v0.24.0` 分支上评估上游 `0.24.1` 的 patch 内容，例如严重 bugfix、兼容性修复、构建修复或安全修复。
2. 只同步确实需要进入 vLLM-Iluvatar `v0.24.0` 维护线的上游 patch 改动。
3. 完成适配、验证和版本号更新后，在同一个 `v0.24.0` 分支上打 `v0.24.1` tag。
4. 不新建 `v0.24.1` 长期分支，避免 patch 版本继续扩散维护线。

```bash
git checkout v0.24.0
git pull
# 按需同步上游 vLLM 0.24.1 的必要 patch 改动
# 更新版本号、CHANGELOG、release note
# 完成编译、单测、核心模型 smoke test
git commit -m "release: v0.24.1"
git tag v0.24.1
git push origin v0.24.0 v0.24.1
```

---

## CI 与验证

`main` CI 面向最新适配目标，至少覆盖：

- 编译和基础单测。
- Iluvatar 平台核心算子、模型加载、离线推理和服务化 smoke test。
- 当前主干活跃能力，例如 MoE、量化、speculative decoding、并行策略、NIXL/PD。

版本分支 CI 固定对应的 vLLM 基座和依赖版本，不隐式漂移：

```yaml
# v0.24.0 示例
- pip3 install vllm==0.24.0 --no-deps
```

EOL 分支可以停止常规 CI；如果需要临时交付修复，必须在 PR 中补充必要的手工验证记录。

---

## 维护与清理

默认只维护最新两个 vLLM-Iluvatar 版本分支：

```text
v0.24.0  活跃维护：允许必要 feature、bugfix、平台适配
v0.23.0  稳定维护：仅严重 bug、交付阻塞、兼容性修复
v0.22.0  EOL：停止常规 CI 和修复，归档后删除远端分支
```

版本分支进入 EOL 时，应在 README、release note 或分支说明中标注，打归档 tag 后删除远端分支：

```bash
git tag archive/v0.22.0 v0.22.0
git push origin archive/v0.22.0
git push origin --delete v0.22.0
```

临时分支清理规则：

- PR 合入后删除对应的 `feature/*`、`bugfix/*`、`sync/*` 分支。
- 已关闭且不再推进的 PR 分支，由作者或维护者删除。
- 超过一个发布周期未更新的临时分支，默认视为废弃，可由维护者通知后删除。

---

## 参考

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
