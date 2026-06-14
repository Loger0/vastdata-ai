## 前置检查：确认 Feature 分支已存在（v6.1 新增）

在拆分子任务之前，**必须先确认** feature 分支已创建：

```bash
# 检查当前分支
git branch --show-current

# 如果不在 feature 分支上，切换到 feature 分支
# 分支名从 Profile 的 framework 字段推导
git checkout feature/<framework-name>-vastbase-adapter

# 如果分支不存在，创建它
git checkout -b feature/<framework-name>-vastbase-adapter
```

**此分支必须在 framework-analyzer 阶段已创建。如果不存在，说明流程异常，应在 Issue 评论区发出警告。**

## Framework Profile 读取（v6 新增）

在拆分子任务之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 拆分策略分流

根据 `integration_mode` 切换子任务拆分策略：

#### standalone 模式（独立包）

维持现有拆分逻辑：
- 按 `required_methods` 中的方法拆分子 Issue
- 每个子 Issue 包含：实现 → 测试 → 审查

#### native 模式（原生集成）

按以下三类任务拆分：

##### A. 配置与注册任务（1 个子 Issue）

**范围**：所有配置文件、常量枚举、工厂注册的修改
**内容**：
- `files_to_modify` 中的配置类文件（settings, constants, __init__）
- 添加 Vastbase 的配置项、枚举值、后端注册逻辑
**优先级**：最高（其他子任务依赖此任务完成）

##### B. 后端实现任务（按方法拆分，2-4 个子 Issue）

**范围**：新增的 Vastbase 后端实现文件（`files_to_create`）
**拆分建议**：
- 子 Issue 1：核心 CRUD（create, insert, delete）—— 最基础，先做
- 子 Issue 2：向量检索（search, hybrid_search）—— 核心功能
- 子 Issue 3（可选）：高级功能（filter, metadata 操作、索引管理等）
**优先级**：依次依赖

##### C. 集成测试任务（1 个子 Issue）

**范围**：端到端集成测试 + Demo 脚本
**内容**：
- 基于 test-strategist 的 `TEST_PLAN.md` 编写测试
- 编写 Demo 脚本
**优先级**：最后（依赖 A 和 B 全部完成）

##### native 模式拆分模板

```
Parent Issue: [适配] RAGFlow Vastbase 向量数据库后端
├── Child 1: [配置] 添加 Vastbase 配置项和后端注册
├── Child 2: [实现] Vastbase 后端核心 CRUD
├── Child 3: [实现] Vastbase 向量检索和混合搜索
├── Child 4: [实现] Vastbase 过滤和元数据操作（如 Profile 要求）
└── Child 5: [集成测试] 端到端集成测试和应用级 Demo
```

#### plugin 模式

按插件规范拆分：插件元数据 + 后端实现 + 集成测试。

### 每个子 Issue 必须标注

从 Profile 中提取关键信息写入子 Issue description：
- 相关的 `files_to_modify` 或 `files_to_create`（精确到文件路径）
- 相关的 `required_methods`（精确到方法名）
- 参照的 `reference_backend` 文件路径
- 关联的 `convention_references`
