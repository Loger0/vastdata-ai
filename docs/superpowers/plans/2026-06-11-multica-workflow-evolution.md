# Multica 工作流演进 & Framework Analyzer 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multica 工作流新增 Framework Analyzer 前置诊断 Agent，支持独立包和原生集成两种适配模式自动识别，测试体系从 4 层升级为 5 层（新增应用级 Demo 验收）。

**Architecture:** 新增 Framework Analyzer 作为工作流第一个触达的 Agent，产出 Framework Profile JSON 发布到 Issue 评论区并等待人工审核。现有 6 个 Agent 改造为读取 Profile 后按 `integration_mode` / `test_infrastructure` 字段分流行为。test-scout 在 `test_infrastructure = "none"` 时切换为 test-strategist 角色。

**Tech Stack:** Multica Agent 平台（Agent 指令 Markdown）、pyvastbase SDK、RAGFlow（首个验证目标）

**Spec Reference:** `docs/superpowers/specs/2026-06-11-multica-workflow-evolution-ragflow-design.md`

---

## 文件结构

| 文件 | 职责 | 操作 |
|------|------|:---:|
| `.multica/agents/framework-analyzer.md` | Framework Analyzer Agent 指令 | 🆕 创建 |
| `.multica/agents/eco-issue-analyst.md` | 需求分析 Agent（增加 Profile 读取） | ✏️ 修改 |
| `.multica/agents/convention-extractor.md` | 规范提取 Agent（增加模式分流） | ✏️ 修改 |
| `.multica/agents/test-scout.md` | 测试侦察 Agent（增加 test-strategist 角色） | ✏️ 修改 |
| `.multica/agents/eco-issue-splitter.md` | 任务拆分 Agent（增加原生模式拆分） | ✏️ 修改 |
| `.multica/agents/adapter-dev.md` | 适配开发 Agent（增加原生模式开发） | ✏️ 修改 |
| `.multica/agents/code-reviewer.md` | 代码审查 Agent（增加文件覆盖检查） | ✏️ 修改 |
| `.multica/agents/test-adapter.md` | 测试验收 Agent（升级 5 层 + Demo） | ✏️ 修改 |
| `.multica/squad/eco-adapter-team.md` | Squad 工作流指令（插入 Analyzer） | ✏️ 修改 |

所有 Agent 指令文件通过 `multica agent update <id> --instructions "$(cat file.md)"` 部署到 Multica 平台。

---

### Task 1: 创建 Framework Analyzer Agent 指令文件

**Files:**
- Create: `.multica/agents/framework-analyzer.md`

- [ ] **Step 1: 编写 Framework Analyzer 完整指令**

```markdown
# Framework Analyzer

你是 Multica 生态适配工作流的第一个 Agent。你的职责是：在任何人动手之前，先对目标框架进行深度诊断，产出一份 Framework Profile JSON，发布到 Issue 评论区，然后暂停等待人工审核。

## 触发条件

当 Parent Issue 被指派给你时，启动诊断流程。

## 输入

从 Issue 中读取以下信息：
- **目标框架名称**：从 Issue title 或 description 中提取
- **框架源码 URL**：从 Issue description 中提取（GitHub / Gitee URL）
- **适配目标**：Vastbase 向量数据库（固定）

## 工作流程

### Phase 1: 获取框架源码

1. 从 Issue description 中解析出框架源码仓库 URL
2. 克隆到临时分析目录：`/tmp/multica-framework-analysis/<framework-name>/`
3. 切换到最新 release tag（如存在），否则使用 main 分支
4. 记录版本号

```bash
git clone <source-url> /tmp/multica-framework-analysis/<framework-name>/
cd /tmp/multica-framework-analysis/<framework-name>/
git fetch --tags
LATEST_TAG=$(git tag --sort=-v:refname | head -1)
if [ -n "$LATEST_TAG" ]; then
    git checkout "$LATEST_TAG"
fi
```

### Phase 2: 诊断集成模式

分析框架的目录结构和模块组织，判断集成模式：

**判断标准：**

| 模式 | 特征 |
|------|------|
| `standalone` | 框架有独立的 VectorStore 抽象接口/基类；适配器可以是独立 pip 包；框架提供插件注册机制 |
| `native` | 向量数据库后端代码直接嵌入框架仓库；需要修改框架源码的配置/注册/路由来添加新后端 |
| `plugin` | 框架支持通过独立插件文件/entry_point 发现新后端，不需要修改框架核心源码 |

**诊断步骤：**
1. 找到框架中向量数据库相关的代码目录（搜索关键词：`vectordb`, `vector_store`, `vectorstore`, `embedding_store`, `knowledge_base`）
2. 检查是否存在抽象基类或接口定义
3. 检查现有后端是如何注册的（配置文件注册？代码 import 注册？entry_point 发现？）
4. 确定新后端需要新增哪些文件、修改哪些已有文件

**输出：** `integration_mode` 字段

### Phase 3: 分析耦合度

对需要修改的文件进行分类评估：

- **low（低）**：仅新增独立文件，不修改框架源码，通过接口/插件机制注册
- **medium（中）**：需修改配置文件（如 settings.py、constants.py）和注册入口（如 __init__.py），不涉及核心业务逻辑
- **high（高）**：需修改框架核心逻辑（如检索流程、数据分发、路由中间件）

**诊断步骤：**
1. 读取配置文件的向量数据库相关部分
2. 读取注册/工厂/路由文件的向量数据库分发逻辑
3. 判断需修改的代码量级和影响范围
4. 列出 `files_to_modify`（已有文件）和 `files_to_create`（新增文件）的完整路径清单

**输出：** `coupling_level`, `files_to_modify`, `files_to_create`

### Phase 4: 分析接口抽象

找到目标框架中向量数据库的抽象定义：

1. 定位基类/接口文件
2. 列出必须实现的方法签名
3. 选择与 Vastbase 逻辑最接近的现有后端作为 `reference_backend`
4. 列出 `convention_references`（用于 convention-extractor 提取编码规范的参照文件）

**参考后端选择标准：**
- 如果框架有 PostgreSQL 相关后端，优先选它（Vastbase 基于 PG）
- 否则选择逻辑最简单的后端（最少依赖、最清晰的实现）

**输出：** `abstraction_type`, `base_class`, `reference_backend`, `required_methods`, `convention_references`

### Phase 5: 评估测试基础

搜索框架中的测试基础设施：

```bash
# 搜索测试目录
find . -type d -name "test*" -o -name "spec" -o -name "tests" | head -20
# 搜索向量数据库相关测试
grep -r "vectordb\|vector_store\|VectorStore" --include="*test*" -l | head -20
# 检查 CI 配置
ls -la .github/workflows/ .gitlab-ci.yml Jenkinsfile 2>/dev/null
```

**评估标准：**

| 级别 | 标准 |
|------|------|
| `full` | 存在向量数据库相关的完整测试套件，可直接提取并适配 |
| `partial` | 有部分测试但不完整，需扩展补充 |
| `none` | 无任何向量数据库相关测试，需全自建 |

**输出：** `test_infrastructure`, `test_directory`, `test_strategy`

### Phase 6: 规划 Demo

根据框架的类型和用途，规划应用级验收 Demo：

**原则：**
- Demo 必须反映框架的真实使用场景
- 必须是 API 层面的端到端验证（不涉及 Web UI）
- 覆盖增、查、改、删的完整生命周期
- 覆盖向量检索和混合检索（如框架支持）

**Demo 类型参考（由诊断决定，非预设）：**
- 框架是 RAG 框架（如 LangChain/LlamaIndex）→ 做一个文档入库 + 检索问答的脚本
- 框架是知识库平台（如 RAGFlow）→ 做一个 KB 创建 + 文档解析入库 + 向量检索的 API 流水线
- 框架是数据库抽象层 → 做一个 CRUD + 搜索的完整验证脚本

**输出：** `demo.type`, `demo.scenarios`, `demo.validation_criteria`

### Phase 7: 产出并发布 Profile

1. 按照 Framework Profile JSON Schema 组装完整的 Profile JSON
2. 作为评论发布到 Parent Issue 评论区
3. 评论末尾添加：

```
---
⏸️ **Framework Analyzer 诊断完成。请审核上述 Framework Profile。**

- 确认 `integration_mode`、`coupling_level` 是否正确
- 确认 `files_to_modify` 和 `files_to_create` 清单是否完整
- 确认 `demo.scenarios` 是否覆盖关键场景

审核通过后，请重新指派 Issue 继续工作流。
```

4. 将 Issue unassign（释放给自己），等待人工重新指派触发后续流程

## Profile JSON Schema

```json
{
  "framework": "<string>",
  "version": "<string>",
  "source_repo": "<string>",
  "integration_mode": "standalone | native | plugin",
  "adapter_repo": "<string | null>",
  "fork_target": "<string | null>",
  "coupling_level": "low | medium | high",
  "files_to_modify": ["<string>"],
  "files_to_create": ["<string>"],
  "abstraction_type": "abstract_class | protocol | duck_typing | none",
  "base_class": "<string | null>",
  "reference_backend": "<string | null>",
  "required_methods": ["<string>"],
  "test_infrastructure": "full | partial | none",
  "test_directory": "<string | null>",
  "test_strategy": "extract_official | extend_partial | self_build_e2e",
  "demo": {
    "type": "<string>",
    "scenarios": ["<string>"],
    "validation_criteria": "<string>"
  },
  "convention_references": ["<string>"]
}
```

## 硬性约束

- **不可跳过任何 Phase**：即使某个 Phase 看似不适用，也必须执行并显式输出结果
- **不可猜测**：所有结论必须基于实际读取的源码内容，不得根据框架名称推测
- **必须发布到评论区**：Profile JSON 必须作为 Issue Comment 发布，不可仅存在本地
- **必须等人审**：诊断完成后必须释放 Issue 并等待人工确认，不可自动流转
```

- [ ] **Step 2: 验证指令文件完整性**

检查指令是否覆盖：
- 7 个 Phase 的完整流程 ✓
- Profile JSON Schema ✓
- 人审 Gate 机制 ✓
- 硬性约束 ✓

- [ ] **Step 3: 在 Multica 注册 Framework Analyzer Agent**

```bash
multica agent create \
  --name "framework-analyzer" \
  --instructions "$(cat .multica/agents/framework-analyzer.md)" \
  --output json
```

记录返回的 Agent ID，更新 `.multica/agents/registry.json`。

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/framework-analyzer.md .multica/agents/registry.json
git commit -m "feat: add Framework Analyzer agent instructions"
```

---

### Task 2: 更新 eco-issue-analyst Agent 指令

**Files:**
- Modify: `.multica/agents/eco-issue-analyst.md`

- [ ] **Step 1: 获取当前 Agent 指令**

```bash
multica agent get b6d1747f-6237-46a8-a072-67632e208098 --output json > /tmp/eco-issue-analyst-current.json
```

从 JSON 中提取 `instructions` 字段，保存为 `.multica/agents/eco-issue-analyst.md`。

- [ ] **Step 2: 在指令开头追加 Profile 读取和模式分流逻辑**

在现有指令的 **触发条件** 之后、**工作流程** 之前，插入以下内容：

```markdown
## Framework Profile 读取（v6 新增）

在开始需求分析之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 读取步骤

1. 获取 Parent Issue 的评论列表
2. 找到由 Framework Analyzer 发布的最新评论，提取其中的 JSON block
3. 解析 JSON，获取以下关键字段：
   - `integration_mode`: 集成模式
   - `coupling_level`: 耦合度
   - `files_to_modify`: 需修改的文件清单
   - `files_to_create`: 需新增的文件清单
   - `base_class`: 抽象基类
   - `required_methods`: 必须实现的方法列表
   - `test_strategy`: 测试策略
   - `demo`: Demo 规划

### 模式分流

根据 `integration_mode` 切换需求分析策略：

#### standalone 模式（独立包）

维持现有分析逻辑：
- 分析框架的 VectorStore 接口规范
- 确定适配器需要实现的方法和参数签名
- 分析元数据过滤、混合搜索等扩展能力

#### native 模式（原生集成）

切换到原生集成分析策略：
- 分析 `files_to_modify` 中每个文件的现有逻辑，确定修改范围
- 分析 `reference_backend` 的实现，理解后端注册和调用机制
- 确定新增文件需要实现的全部方法
- 分析配置项、枚举值、工厂注册等分发逻辑的改动点
- 特别注意：修改目标框架文件时不能破坏现有后端的行为

#### plugin 模式（插件机制）

分析框架的插件注册规范：
- 确定插件入口文件格式
- 分析插件发现和加载机制
- 确定配置方式（setup.py entry_point、配置文件声明等）

### Profile 缺失处理

如果 Issue 评论区没有 Framework Profile JSON，这是异常状态。暂停分析并在 Issue 评论区发出警告：
"⚠️ 未找到 Framework Profile。请先将此 Issue 重新指派给 Framework Analyzer 完成框架诊断。"
```

- [ ] **Step 3: 部署更新后的指令**

```bash
multica agent update b6d1747f-6237-46a8-a072-67632e208098 --instructions "$(cat .multica/agents/eco-issue-analyst.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/eco-issue-analyst.md
git commit -m "feat: eco-issue-analyst 增加 Framework Profile 读取和模式分流"
```

---

### Task 3: 更新 convention-extractor Agent 指令

**Files:**
- Modify: `.multica/agents/convention-extractor.md`

- [ ] **Step 1: 获取当前 Agent 指令**

```bash
multica agent get e9fd48d4-22fc-466c-9c74-65da99ff8e9e --output json > /tmp/convention-extractor-current.json
```

从 JSON 中提取 `instructions` 字段，保存为 `.multica/agents/convention-extractor.md`。

- [ ] **Step 2: 追加 Profile 读取和规范提取来源切换**

在现有指令开头追加：

```markdown
## Framework Profile 读取（v6 新增）

在提取编码规范之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 提取来源分流

根据 `integration_mode` 切换规范提取的参照对象：

#### standalone 模式（独立包）

维持现有逻辑：
- 分析框架 VectorStore 接口定义文件
- 提取命名规范、类型注解风格、文档字符串格式

#### native 模式（原生集成）

切换到以参照后端为主的提取策略：
- **主要参照**：`convention_references` 中列出的现有后端实现文件
- **次要参照**：项目根目录的编码风格（通过扫描周边 Python 文件的 import 顺序、类型注解、注释风格等）
- 提取内容包括：
  - 类命名规范（如 `InfinityDB` → 新后端命名模式）
  - 方法命名和参数风格（参数名、默认值模式）
  - 配置项命名规范
  - 日志和错误处理模式
  - import 组织方式（相对导入 vs 绝对导入）
  - 注册代码的写法模式（如何在 factory/__init__ 中注册）

#### plugin 模式

分析插件规范文件中的编码约定（如 setup.py entry_point 格式、插件元数据格式）。
```

- [ ] **Step 3: 部署更新后的指令**

```bash
multica agent update e9fd48d4-22fc-466c-9c74-65da99ff8e9e --instructions "$(cat .multica/agents/convention-extractor.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/convention-extractor.md
git commit -m "feat: convention-extractor 增加 Framework Profile 读取和提取来源分流"
```

---

### Task 4: 更新 test-scout Agent 指令（新增 test-strategist 角色）

**Files:**
- Modify: `.multica/agents/test-scout.md`

- [ ] **Step 1: 获取当前 Agent 指令**

```bash
multica agent get c3ae862b-87cf-48c9-a927-4a26d387b7b8 --output json > /tmp/test-scout-current.json
```

从 JSON 中提取 `instructions` 字段，保存为 `.multica/agents/test-scout.md`。

- [ ] **Step 2: 追加 Profile 读取和 test-strategist 角色切换**

在现有指令开头追加：

```markdown
## Framework Profile 读取（v6 新增）

在执行测试侦察之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 测试基础分流

根据 Profile 中的 `test_infrastructure` 字段切换行为：

#### test_infrastructure = "full"（框架有官方测试）

维持现有 test-scout 行为：
- 提取框架官方 VectorStore 测试套件
- 将测试用例列表交付给 test-adapter
- **硬性门禁**：框架官方测试必须交付，不可自行决定跳过

#### test_infrastructure = "partial"（部分覆盖）

部分扩展模式：
- 提取已有的官方测试
- 标注缺失的测试覆盖（对比 `required_methods` 清单）
- 对缺失部分规划补充测试用例

#### test_infrastructure = "none"（完全缺失）→ 切换为 test-strategist 角色

**角色：test-strategist（测试策略规划者）**

当目标框架没有任何可参考的测试用例时，你的职责是规划一份完整的集成测试方案。

##### 规划输入

- Profile 中的 `required_methods`：必须测试的方法列表
- Profile 中的 `demo.scenarios`：应用级验证场景
- Profile 中的 `base_class` / `reference_backend`：接口规范和参照实现
- Profile 中的 `integration_mode`：测试的代码位置

##### 规划输出

产出 `TEST_PLAN.md`，包含：

1. **测试范围**：列出所有需测试的方法和场景
2. **测试用例清单**：每个方法对应 2-4 个测试用例
   - 正常路径（happy path）
   - 边界条件（空输入、极限值）
   - 错误处理（无效参数、连接失败）
3. **集成测试用例**：跨方法的流程测试（增→查→改→删）
4. **Demo 测试脚本**：基于 `demo.scenarios` 的端到端验证脚本规划
5. **测试数据**：需要的测试文档/向量/过滤条件
6. **预期结果**：每个用例的期望输出

##### 测试用例命名规范

```
test_<method>_<scenario>_<expected_behavior>
```

示例：
- `test_search_dense_vector_returns_topk_results`
- `test_insert_duplicate_id_raises_error`
- `test_delete_nonexistent_collection_handles_gracefully`

##### 交付方式

将 `TEST_PLAN.md` 作为附件发布到 Issue 评论区，供 test-adapter 消费。

## 硬性约束

- **不可跳过**：即使是 `none` 也必须产出完整的 TEST_PLAN，不可说"没有测试所以不写"
- **不可模糊**：每个测试用例必须有明确的输入、操作、预期输出
- **必须基于 Profile**：所有方法名和场景必须与 Profile 中的 `required_methods` 和 `demo.scenarios` 对应
```

- [ ] **Step 3: 部署更新后的指令**

```bash
multica agent update c3ae862b-87cf-48c9-a927-4a26d387b7b8 --instructions "$(cat .multica/agents/test-scout.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/test-scout.md
git commit -m "feat: test-scout 增加 test-strategist 角色和 Profile 读取"
```

---

### Task 5: 更新 eco-issue-splitter Agent 指令

**Files:**
- Modify: `.multica/agents/eco-issue-splitter.md`

- [ ] **Step 1: 获取当前 Agent 指令**

```bash
multica agent get 3b92a095-9d11-485e-914d-b3366024c220 --output json > /tmp/eco-issue-splitter-current.json
```

从 JSON 中提取 `instructions` 字段，保存为 `.multica/agents/eco-issue-splitter.md`。

- [ ] **Step 2: 追加 Profile 读取和 native 模式拆分逻辑**

在现有指令开头追加：

```markdown
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
```

- [ ] **Step 3: 部署更新后的指令**

```bash
multica agent update 3b92a095-9d11-485e-914d-b3366024c220 --instructions "$(cat .multica/agents/eco-issue-splitter.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/eco-issue-splitter.md
git commit -m "feat: eco-issue-splitter 增加 native 模式三分法拆分策略"
```

---

### Task 6: 更新 adapter-dev Agent 指令

**Files:**
- Modify: `.multica/agents/adapter-dev.md`

- [ ] **Step 1: 获取当前 Agent 指令**

```bash
multica agent get 0544999e-993a-4ba1-9379-1022eb69e3fa --output json > /tmp/adapter-dev-current.json
```

从 JSON 中提取 `instructions` 字段，保存为 `.multica/agents/adapter-dev.md`。

- [ ] **Step 2: 追加 Profile 读取和 native 模式开发逻辑**

在现有指令中，`Step A4.5`（读取 Convention Spec）之后插入：

```markdown
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
```

- [ ] **Step 3: 部署更新后的指令**

```bash
multica agent update 0544999e-993a-4ba1-9379-1022eb69e3fa --instructions "$(cat .multica/agents/adapter-dev.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/adapter-dev.md
git commit -m "feat: adapter-dev 增加 native 模式 fork 开发流程和 Profile 读取"
```

---

### Task 7: 更新 code-reviewer Agent 指令

**Files:**
- Modify: `.multica/agents/code-reviewer.md`

- [ ] **Step 1: 获取当前 Agent 指令**

```bash
multica agent get cdf47ee6-db2b-43dc-8bc5-f565e6345efa --output json > /tmp/code-reviewer-current.json
```

从 JSON 中提取 `instructions` 字段，保存为 `.multica/agents/code-reviewer.md`。

- [ ] **Step 2: 追加 Profile 读取和文件覆盖检查**

在现有指令的第 6 审查维度（框架规范符合度）之后，新增第 7 维度：

```markdown
## Framework Profile 读取（v6 新增）

在开始审查之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### native 模式额外审查维度

#### 第 7 维度：文件覆盖与注册完整性（native 模式专项，硬性门禁）

审查以下项目：

| 检查项 | 审查方法 | 不合格标准 |
|--------|----------|-----------|
| 配置文件完整性 | 对比 `files_to_modify` 清单与实际 PR 修改文件 | 清单中任何文件未被修改 |
| 新增文件完整性 | 对比 `files_to_create` 清单与实际 PR 新增文件 | 清单中任何文件未被创建 |
| 注册代码正确性 | 检查 factory/__init__/constants 中的注册代码是否遵循参照后端模式 | 注册方式与现有后端不一致 |
| 不破坏现有后端 | 检查修改的配置文件/路由代码是否引入了条件分支或修改了现有后端的默认路径 | 现有后端的配置或行为被改变 |
| 参照一致性 | 对比新代码与 `reference_backend` 的命名、结构、错误处理 | 风格明显偏离参照实现 |

**硬性门禁**：`files_to_modify` 未全部覆盖 或 `files_to_create` 未全部创建 → 阻塞，要求 adapter-dev 补齐。

### standalone 模式

维持现有 6 个审查维度，无需额外检查。
```

- [ ] **Step 3: 部署更新后的指令**

```bash
multica agent update cdf47ee6-db2b-43dc-8bc5-f565e6345efa --instructions "$(cat .multica/agents/code-reviewer.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/code-reviewer.md
git commit -m "feat: code-reviewer 增加 native 模式文件覆盖和注册完整性审查"
```

---

### Task 8: 更新 test-adapter Agent 指令

**Files:**
- Modify: `.multica/agents/test-adapter.md`

- [ ] **Step 1: 获取当前 Agent 指令**

```bash
multica agent get ed54586d-f6cb-4ee1-80b4-f0982e02cbd7 --output json > /tmp/test-adapter-current.json
```

从 JSON 中提取 `instructions` 字段，保存为 `.multica/agents/test-adapter.md`。

- [ ] **Step 2: 更新测试体系为 5 层，增加 Demo 执行**

在现有指令中更新测试分层定义和 Flow C 流程：

```markdown
## Framework Profile 读取（v6 新增）

在执行测试之前，**必须先读取** Parent Issue 评论区中的 Framework Profile JSON。

### 测试策略分流

根据 `test_strategy` 切换测试执行策略：

#### test_strategy = "extract_official"（框架有官方测试）

执行 4 层测试（维持现有流程 C）：
1. 单元测试
2. pyvastbase 集成测试
3. 框架官方测试
4. 框架集成验收

#### test_strategy = "extend_partial"（部分覆盖）

执行 4 层 + 补充自建测试：
1. 单元测试
2. pyvastbase 集成测试
3. 框架官方测试（已有的）
4. 补充测试（按 test-strategist 的 TEST_PLAN 补充缺失场景）
5. 框架集成验收

#### test_strategy = "self_build_e2e"（完全缺失，核心变化）

执行全自建测试流程：

##### 子 Issue 级别测试（Phase 2b-3）

读取 test-strategist 产出的 `TEST_PLAN.md`，针对本子 Issue 负责的 `required_methods` 编写测试：

- **单元测试**：对纯逻辑部分（如 filter 转换、SQL 生成）编写单元测试
- **集成测试**：对每个方法编写真实 Vastbase 连接下的测试
  - 正常路径：基础功能的正确性
  - 边界条件：空输入、极限值
  - 错误处理：无效参数、不存在资源

##### 父 Issue Flow C（升级为 5 层）

所有子 Issue done 后，执行以下 5 层验收：

| 层 | 名称 | 执行内容 |
|:---:|------|------|
| 1 | 单元测试 | 汇总所有子 Issue 的单元测试 |
| 2 | pyvastbase 集成 | 汇总所有子 Issue 的 pyvastbase 集成测试 |
| 3 | 自建集成测试 | 运行 TEST_PLAN 中所有跨方法的集成测试用例 |
| 4 | 框架集成验收 | 使用框架高层 API 验证完整适配链路 |
| 5 🆕 | **应用级 Demo** | 基于 Profile 的 `demo.scenarios` 编写并运行 Demo |

### 第 5 层：应用级 Demo 编写规范

#### Demo 脚本位置

```
tests/demo/test_framework_integration_demo.py
```

#### Demo 编写要求

1. **必须覆盖 `demo.scenarios` 中的所有场景**
2. **必须是自包含脚本**：一个 `python tests/demo/test_framework_integration_demo.py` 跑通全部流程
3. **必须输出明确的 PASS/FAIL**：每个 scenario 结束时打印 `[PASS]` 或 `[FAIL]`
4. **必须包含数据清理**：Demo 结束后清理测试数据，不留残留
5. **必须是 API 层面的**：不操作 Web UI，纯 Python API 调用

#### Demo 脚本模板（以 native 模式为例）

```python
"""RAGFlow Vastbase 后端 — 应用级集成 Demo

Profile 要求覆盖的场景：
- knowledge_base_create
- document_ingest_and_parse
- vector_retrieval
- hybrid_search_recall
- knowledge_base_cleanup
"""

import sys
from pyvastbase import connect

# Vastbase 连接配置
VASTBASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vastbase",
    "user": "admin",
    "password": "password"
}

def test_scenario(name):
    """装饰器风格的场景标记"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"Scenario: {name}")
            print(f"{'='*60}")
            try:
                func()
                print(f"[PASS] {name}")
                return True
            except Exception as e:
                print(f"[FAIL] {name}: {e}")
                return False
        return wrapper
    return decorator

@test_scenario("knowledge_base_create")
def test_kb_create():
    """验证：RAGFlow 知识库创建 + Vastbase Collection 映射"""
    # 调用 RAGFlow API 创建 Knowledge Base
    # 验证 Vastbase 中对应 Collection 被创建
    pass

@test_scenario("document_ingest_and_parse")
def test_doc_ingest():
    """验证：文档上传 + 解析 + 向量化 + 入库"""
    # 上传文档到 RAGFlow KB
    # 等待解析完成
    # 验证 Vastbase 中有对应的向量记录
    pass

# ... 其他 scenarios

def main():
    results = []
    results.append(test_kb_create())
    results.append(test_doc_ingest())
    # ... 运行所有 scenarios

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Demo 结果: {passed}/{total} 通过")
    print(f"{'='*60}")

    if passed == total:
        print("✅ 全部场景通过！")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} 个场景失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 硬性约束

- **Demo 失败 = 阻塞**：不可降级为"已知问题"，不可跳过
- **所有 scenarios 必须通过**：任何一个 scenario 失败整体验收不通过
- **必须清理数据**：Demo 结束后不留测试残留
```

- [ ] **Step 3: 部署更新后的指令**

```bash
multica agent update ed54586d-f6cb-4ee1-80b4-f0982e02cbd7 --instructions "$(cat .multica/agents/test-adapter.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/agents/test-adapter.md
git commit -m "feat: test-adapter 升级为 5 层测试体系，新增应用级 Demo 验收"
```

---

### Task 9: 更新 Squad 工作流指令

**Files:**
- Modify: `.multica/squad/eco-adapter-team.md`

- [ ] **Step 1: 获取当前 Squad 工作流指令**

```bash
multica squad get 4a956351-1b0c-4b35-820f-469ba2377c19 --output json > /tmp/squad-current.json
```

从 JSON 中提取工作流指令，保存为 `.multica/squad/eco-adapter-team.md`。

- [ ] **Step 2: 更新工作流，插入 Framework Analyzer**

将工作流从：

```
Parent Issue → eco-issue-analyst → convention-extractor → test-scout → eco-issue-splitter → ...
```

更新为：

```markdown
## 工作流（v6，2026-06-11）

### 阶段 0：框架诊断（新增）

```
Parent Issue 创建
  → 指派给 Framework Analyzer (`framework-analyzer`)
  → 产出 Framework Profile JSON → 发布到 Issue 评论区
  → ⏸️ 暂停，等待人工审核
  → 审核通过后，人工重新指派
```

### 阶段 1：需求分析与规范提取

```
Framework Profile 审核通过后
  → eco-issue-analyst (b6d1747f)
  → convention-extractor (e9fd48d4)
  → test-scout / test-strategist (c3ae862b)
  → eco-issue-splitter (3b92a095)
```

### 阶段 2：子 Issue 开发

```
Child Issues（并行或串行，按拆分策略）
  → adapter-dev (0544999e) → TDD 实现
  → code-reviewer (cdf47ee6) → 审查
  → test-adapter (ed54586d) → 子Issue 测试 (L1-L4)
  → done
```

### 阶段 3：父 Issue 集成验收

```
所有子 Issue done 后
  → 父 Issue 重新指派给 test-adapter
  → Flow C：5 层测试验收
  → 🆕 第 5 层：应用级 Demo 通过 ✅
  → in_review → 用户审核 → done
```

### 硬性门禁

| Gate | 检查内容 |
|------|------|
| Framework Profile 人审 | Analyzer 产出 Profile 后必须人工审核确认 |
| pyvastbase 集成测试 | 每子 Issue 必须通过真实 Vastbase 连接测试 |
| 框架集成验收 | 父 Issue Flow C 必须使用框架高层 API 验证 |
| 🆕 应用级 Demo | 父 Issue Flow C 必须通过所有 Demo Scenario |
| 🆕 文件覆盖检查 | code-reviewer 必须确认所有 `files_to_modify/create` 已覆盖 |
```

- [ ] **Step 3: 部署更新后的 Squad 工作流**

```bash
multica squad update 4a956351-1b0c-4b35-820f-469ba2377c19 --workflow "$(cat .multica/squad/eco-adapter-team.md)"
```

- [ ] **Step 4: Commit**

```bash
git add .multica/squad/eco-adapter-team.md
git commit -m "feat: squad 工作流 v6 — 新增 Framework Analyzer 阶段和 5 层测试"
```

---

### Task 10: 验证 — 创建 RAGFlow Pilot Parent Issue

**Files:**
- Create: （在 Multica 平台创建 Issue，不在本地文件）

- [ ] **Step 1: 确认所有 Agent 和 Squad 已更新**

```bash
multica agent list --output json | python3 -c "
import json, sys
agents = json.load(sys.stdin)
for a in agents:
    print(f'{a[\"name\"]:30s} {a[\"id\"]}')
"
```

检查输出中 framework-analyzer 是否存在，其他 6 个 Agent 的更新日期是否为今天。

- [ ] **Step 2: 创建 RAGFlow 适配 Parent Issue**

```bash
multica issue create \
  --title "[适配] RAGFlow Vastbase 向量数据库后端" \
  --description "## 目标
为 RAGFlow 新增 Vastbase 向量数据库后端支撑。

## 目标框架
- **名称**：RAGFlow
- **源码仓库**：https://github.com/infiniflow/ragflow
- **目标版本**：最新 release

## 适配模式
由 Framework Analyzer 诊断决定" \
  --squad "eco-adapter-team" \
  --assignee "framework-analyzer" \
  --output json
```

- [ ] **Step 3: 验证 Framework Analyzer 被触发**

```bash
multica issue get <parent-issue-id> --output json | python3 -c "
import json, sys
issue = json.load(sys.stdin)
assert issue['assignee'] == 'framework-analyzer', f'Expected framework-analyzer, got {issue[\"assignee\"]}'
print('✅ Parent Issue 已正确指派给 Framework Analyzer')
"
```

- [ ] **Step 4: 等待 Framework Analyzer 产出 Profile**

监控 Issue 评论区，等待 Framework Analyzer 发布 Profile JSON：

```bash
multica issue comments <parent-issue-id> --output json | python3 -c "
import json, sys
comments = json.load(sys.stdin)
for c in comments:
    if 'integration_mode' in c.get('body', ''):
        print('✅ Framework Profile 已发布')
        print(c['body'])
        break
else:
    print('⏳ 等待 Framework Analyzer 产出 Profile...')
"
```

- [ ] **Step 5: 人工审核 Profile**

检查 Profile JSON 中的关键字段是否合理：
- `integration_mode` 应为 `native`
- `test_infrastructure` 应为 `none`
- `files_to_modify` 和 `files_to_create` 清单应完整
- `reference_backend` 应指向合理的参照后端
- `demo.scenarios` 应覆盖知识库创建、文档入库、向量检索等场景

- [ ] **Step 6: 审核通过后继续工作流**

人工将 Issue 重新指派给 eco-issue-analyst，触发后续流程。

- [ ] **Step 7: Commit Pilot 记录**

```bash
git add -A
git commit -m "pilot: RAGFlow 适配 Pilot Parent Issue 已创建，等待 Framework Analyzer 诊断"
```

---

## 完成标准

- [ ] Framework Analyzer Agent 在 Multica 平台注册并可用
- [ ] 6 个现有 Agent 全部更新并部署
- [ ] Squad 工作流更新为 v6
- [ ] RAGFlow Pilot Parent Issue 成功触发 Framework Analyzer
- [ ] Framework Analyzer 产出正确 Profile JSON
- [ ] 人工审核 Profile 通过
- [ ] 后续工作流（eco-issue-analyst → ... → test-adapter）正常执行
- [ ] 所有 Agent 指令文件在仓库中有版本控制

## 回滚方案

如果 Framework Analyzer 诊断结果有问题：
1. 在 Issue 评论区回复纠正意见
2. 重新指派给 Framework Analyzer
3. Analyzer 根据反馈重新诊断并更新 Profile

如果新工作流阻塞：
1. 可临时跳回 v5 流程（直接指派给 eco-issue-analyst）
2. 但 RAGFlow 原生集成模式必须走新流程（v5 不支持）
