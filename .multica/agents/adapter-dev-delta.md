## Framework Profile 读取（v6 新增）

在开始开发之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 开发环境分流

根据 `integration_mode` 切换开发环境：

#### standalone 模式（独立包）

维持现有开发流程：
- 在 adapter 仓库的 feature 分支上开发
- 实现框架 VectorStore 抽象接口
- TDD：先写测试 → 再写实现

#### native 模式（原生集成）

切换到原生集成开发流程：

##### A0. 环境准备

1. Fork `fork_target` 到工作空间
2. Clone fork 仓库
3. 创建 feature 分支：`feature/vastbase-backend`
4. 安装框架的开发依赖

```bash
gh repo fork <fork_target> --clone
cd <repo-name>
git checkout -b feature/vastbase-backend
pip install -e ".[dev]"
```

##### A4.5. 读取修改文件清单（扩展）

除了 Convention Spec，**额外读取** Profile 中的：
- `files_to_modify`：本次子 Issue 需要修改的目标框架文件
- `files_to_create`：本次子 Issue 需要新增的文件
- `reference_backend`：参照实现的绝对路径
- `required_methods`：本次子 Issue 负责的方法列表

##### A6. 实现（扩展 REFACTOR 检查项）

在 REFACTOR 逐项核对时，增加：

| 检查项 | 说明 |
|--------|------|
| 不破坏现有后端 | 修改配置/注册/路由文件时，确保现有后端的逻辑不受影响 |
| 遵循参照后端模式 | 新代码的命名、结构、错误处理应与 `reference_backend` 保持一致 |
| 配置项合规 | 新配置项遵循框架的命名规范和默认值模式 |
| 所有修改文件已覆盖 | `files_to_modify` 中所有文件都已按需修改 |

##### A8. 提交规范

```bash
git add <files_to_create> <files_to_modify>
git commit -m "feat: add Vastbase vector database backend

Implements: <required_methods list>
Modified: <files_to_modify list>
Ref: <reference_backend>"
```

#### plugin 模式

按框架插件规范开发：独立目录 + entry_point 注册。
