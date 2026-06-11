# Multica 工作流演进 & RAGFlow Vastbase 适配设计

**日期:** 2026-06-11  
**状态:** 设计完成，待实现  
**关联:** [[integration-test-mandatory-gate]] | [[multica-v5-convention-extractor]]

---

## 1. 背景与动机

### 1.1 问题

Multica v5 当前工作流为 LangChain / LlamaIndex 模式设计，隐含三个假设：

- 框架有清晰的 VectorStore 抽象接口，适配代码实现接口即可
- 适配代码是独立 pip 包，不修改目标框架源码
- 框架有官方测试套件可参考

RAGFlow 适配打破了这三个假设：

| 维度 | LangChain / LlamaIndex | RAGFlow |
|------|------------------------|---------|
| 适配形式 | 独立包，实现框架接口 | 在 RAGFlow 仓库内原生新增后端 |
| 耦合度 | 低（0 个目标文件修改） | 中高（需修改配置/注册/路由等 3-5 个文件） |
| 测试基础 | 框架有官方测试套件 | 完全缺失，需全自建 |

**用户指出：当前是人工分析后才知道 RAGFlow 是耦合的、没有测试用例的，Multica 应该具备自动诊断这些特征的能力。**

### 1.2 目标

1. **Multica 工作流演进**：新增前置诊断阶段，支持"独立包"和"原生集成"两种适配模式自动识别
2. **首个验证案例**：RAGFlow Vastbase 向量数据库后端适配，通过新工作流跑通全流程

---

## 2. 架构概览

### 2.1 两种适配模式

```
模式一：独立包（LangChain / LlamaIndex — 已有）
┌──────────────────────┐
│  Adapter 独立仓库      │ ← 不修改目标框架，只实现接口
│  langchain-vastbase   │
│  llama-index-vastbase │
└──────────┬───────────┘
           │ 调用
┌──────────▼───────────┐
│  pyvastbase SDK       │ ← 复用层，两种模式一致
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Vastbase             │
└──────────────────────┘

模式二：原生集成（RAGFlow — 本次新增）
┌──────────────────────┐
│  Fork RAGFlow 仓库     │ ← 在 fork 上直接修改源码
│  ├── rag/nlp/         │
│  │   └── vastbase_impl.py  (新增)
│  ├── api/settings.py  │     (修改)
│  ├── rag/utils/       │     (修改)
│  └── ...              │
└──────────┬───────────┘
           │ 调用
┌──────────▼───────────┐
│  pyvastbase SDK       │ ← 复用层，两种模式一致
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Vastbase             │
└──────────────────────┘
```

### 2.2 核心原则

- **下层不变**：无论哪种模式，最终都通过 pyvastbase SDK 连通 Vastbase，该层完全复用
- **上层分流**：Framework Analyzer 诊断后，下游 Agent 根据集成模式切换行为
- **测试升维**：从 4 层测试升级为 5 层，新增应用级 Demo 验收

---

## 3. Multica 工作流演进

### 3.1 新工作流总览

```
Parent Issue 创建
       │
       ▼
┌──────────────────────┐
│ 🆕 Framework Analyzer │ ← 第一个触达的 Agent
│                      │
│ Step 1: 获取框架源码  │
│ Step 2: 诊断集成模式  │
│ Step 3: 分析耦合度    │
│ Step 4: 评估测试基础  │
│ Step 5: 产出 Profile  │
└──────────┬───────────┘
           │ Profile JSON → Issue 评论区
           │ ⏸️  人工审核 Gate（确认后继续）
           ▼
┌──────────────────────┐
│ eco-issue-analyst     │ ← 读取 Profile，切换分析模式
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ convention-extractor  │ ← 读取 Profile，切换规范提取来源
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ test-scout            │ ← 读取 Profile
│                      │
│ test_infrastructure:  │
│  "full"    → 提取测试  │
│  "partial" → 扩展测试  │
│  "none"    → 切换为    │
│   🆕 test-strategist  │   规划自建测试方案
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ eco-issue-splitter    │ ← 读取 Profile，按模式拆子任务
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 子 Issue 开发循环      │
│ adapter-dev           │ ← 读取 Profile
│   → code-reviewer     │
│   → test-adapter      │   (1-4 层测试)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 父 Issue Flow C       │
│ test-adapter          │ ← 读取 Profile.demo
│                       │
│ 🆕 第 5 层：应用级 Demo │
│ 所有场景通过 ✅        │
└──────────────────────┘
```

### 3.2 代码仓库策略

| 集成模式 | 代码位置 | 操作 |
|----------|----------|------|
| `standalone` | Adapter Git 仓库 | 在 adapter 仓库上开发，框架源码只读分析 |
| `native` | Fork 目标框架仓库 | Fork → feature 分支 → 直接修改源码 |

### 3.3 Framework Profile 存放与人审 Gate

- **存放位置**：Issue 评论区（作为评论发布，方便 Agent 直接读取）
- **人审 Gate**：Analyzer 诊断完成后暂停，等待用户审核 Profile 内容，确认无误后手动触发继续

---

## 4. Framework Profile 规范

### 4.1 JSON Schema

```json
{
  "framework": "ragflow",
  "version": "0.18.0",
  "source_repo": "https://github.com/infiniflow/ragflow",

  // ── 集成模式诊断 ──
  "integration_mode": "native",
  // "standalone" | "native" | "plugin"
  "adapter_repo": null,
  // standalone 模式：adapter git repo URL；其他模式 null
  "fork_target": "infiniflow/ragflow",
  // native 模式：需 fork 的目标仓库；其他模式 null

  // ── 耦合度分析 ──
  "coupling_level": "medium",
  // "low" | "medium" | "high"
  "files_to_modify": [
    "api/settings.py",
    "rag/utils/constants.py",
    "rag/nlp/__init__.py"
  ],
  "files_to_create": [
    "rag/nlp/vastbase_impl.py"
  ],

  // ── 接口抽象分析 ──
  "abstraction_type": "class_inheritance",
  // "abstract_class" | "protocol" | "duck_typing" | "none"
  "base_class": "rag.nlp.abstract.VectorDB",
  "reference_backend": "rag.nlp.infinity_impl",
  // 最接近 Vastbase 的参照实现，用于 convention-extractor 和 adapter-dev 参考
  "required_methods": [
    "search", "insert", "delete", "create", "update"
  ],

  // ── 测试基础评估 ──
  "test_infrastructure": "none",
  // "full" | "partial" | "none"
  "test_directory": null,
  "test_strategy": "self_build_e2e",
  // "extract_official" | "extend_partial" | "self_build_e2e"

  // ── Demo 规划 ──
  "demo": {
    "type": "api_e2e_pipeline",
    // 由 Analyzer 根据框架用途诊断，非预设
    "scenarios": [
      "knowledge_base_create",
      "document_ingest_and_parse",
      "vector_retrieval",
      "hybrid_search_recall",
      "knowledge_base_cleanup"
    ],
    "validation_criteria": "召回结果与预期一致；过滤条件下结果数量正确；增删后数据一致性"
  },

  // ── 编码规范引用 ──
  "convention_references": [
    "rag/nlp/infinity_impl.py",
    "rag/nlp/elasticsearch_impl.py"
  ]
}
```

### 4.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `integration_mode` | enum | `standalone`（独立包）/ `native`（原生集成）/ `plugin`（插件机制） |
| `coupling_level` | enum | 低（仅实现接口）/ 中（需改配置和注册）/ 高（深度耦合，需改核心逻辑） |
| `files_to_modify` | array | 目标框架中需要修改的已有文件列表 |
| `files_to_create` | array | 需要在目标框架中新增的文件列表 |
| `abstraction_type` | enum | 框架的接口抽象方式 |
| `reference_backend` | string | 与 Vastbase 逻辑最接近的现有后端，作为实现参照 |
| `test_infrastructure` | enum | `full`（有完整测试套件）/ `partial`（部分覆盖）/ `none`（缺失） |
| `test_strategy` | enum | `extract_official`（提取官方测试）/ `extend_partial`（扩展补全）/ `self_build_e2e`（全自建） |
| `demo` | object | 由 Analyzer 根据框架类型和用途诊断，无预设 |

---

## 5. 测试体系升级

### 5.1 五层测试金字塔

| 层 | 名称 | 验证内容 | 谁执行 | 何时 |
|:---:|------|------|------|------|
| 5 🆕 | **应用级 Demo** | 框架 + Vastbase 真实应用场景 | test-adapter | 父 Issue Flow C（所有子 Issue done 后） |
| 4 | 框架集成验收 | 框架高层 API 完整链路 | test-adapter | 父 Issue Flow C |
| 3 | 框架官方测试 | 框架自身 VectorStore 测试套件 | test-adapter | 每子 Issue Phase 3 |
| 2 | pyvastbase 集成 | 真实 Vastbase 连接，SDK 兼容性 | test-adapter | 每子 Issue Phase 2b |
| 1 | 单元测试 | mock / 本地逻辑验证 | adapter-dev | 每子 Issue Phase 1 |

### 5.2 第 5 层 Demo 规范

- **Demo 形式由 Framework Analyzer 诊断决定**，不预设
- **示例（非规定）**：
  - LangChain → 简单聊天应用（文档入库 → 向量检索召回 → 多轮对话）
  - LlamaIndex → 知识库 RAG 管线（VectorStoreIndex + as_query_engine）
  - RAGFlow → API 端到端（Create KB → Ingest Doc → Search → 验证召回）
- **验证标准**：召回结果正确性、过滤条件生效、增删后数据一致性、异常路径处理
- **硬性门禁**：Demo 失败 → 阻塞，不可降级，不可标记为"已知问题"

### 5.3 测试缺失处理

当 `test_infrastructure = "none"` 时：

1. test-scout 切换为 **test-strategist** 角色
2. 基于 Profile 的 `required_methods` 和 `demo.scenarios` 规划集成测试方案
3. 产出测试用例清单，标注到 Issue
4. test-adapter 按测试方案编写并运行测试
5. 验收标准：所有场景通过 + 全流程 Demo 通过

---

## 6. Agent 变更清单

| Agent | 变更类型 | 变更内容 |
|-------|:---:|------|
| **Framework Analyzer** | 🆕 新增 | 第一个触达的 Agent。克隆源码 → 5 维诊断 → 产出 Profile JSON → 发布到 Issue 评论区 → 暂停等人审 |
| eco-issue-analyst | ✏️ 修改 | 读取 Profile，根据 `integration_mode` 和 `coupling_level` 切换分析策略 |
| convention-extractor | ✏️ 修改 | 读取 Profile，根据 `integration_mode` 选择规范提取来源（接口定义 vs 参照后端代码） |
| test-scout | ✏️ 修改 | 读取 `test_infrastructure`，`"none"` 时自动切换为 test-strategist 角色 |
| eco-issue-splitter | ✏️ 修改 | 读取 Profile，`native` 模式时拆分"修改配置/注册文件"子任务 |
| adapter-dev | ✏️ 修改 | 读取 Profile，`native` 模式在 fork 仓库开发，Step A4.5 增加"读取需修改文件清单" |
| code-reviewer | ✏️ 修改 | 读取 Profile，审查时增加"需修改文件是否全部覆盖"检查 |
| test-adapter | ✏️ 修改 | 升级为 5 层测试，Flow C 增加 Demo 编写与运行。读取 `demo` 字段和 `test_strategy` |

---

## 7. RAGFlow 适配（首次验证案例）

### 7.1 适配目标

- **仓库**：`infiniflow/ragflow` 官方最新 release
- **目标**：让 Vastbase 成为 RAGFlow 中可选的新向量数据库后端
- **模式**：原生集成（native）

### 7.2 前置步骤

1. Framework Analyzer 克隆 RAGFlow 源码并诊断
2. 产出 Profile JSON，标注到 Issue 评论区
3. 人工审核 Profile，确认后再启动后续流程

### 7.3 预期适配范围（待 Analyzer 确认）

- **新增**：`rag/nlp/vastbase_impl.py` — Vastbase 后端实现
- **修改（预估）**：
  - `api/settings.py` — 添加 Vastbase 配置项
  - `rag/utils/constants.py` — 添加 Vastbase 枚举值
  - `rag/nlp/__init__.py` — 注册 Vastbase 后端
  - 可能涉及 `rag/nlp/search.py` 或其他分发逻辑

### 7.4 测试策略

由于 RAGFlow 无官方测试（待 Analyzer 确认），采用 `self_build_e2e`：

- 编写 API 端到端测试脚本
- 覆盖场景：知识库创建 → 文档入库解析 → 向量检索 → 混合搜索 → 知识库清理
- 第 5 层 Demo：完整 API 流水线验证

### 7.5 SDK 层

复用 `pyvastbase` SDK，连接管理 / Collection 操作 / 类型映射全部复用现有实现。

---

## 8. 实现前置依赖

1. **Framework Analyzer Agent 实现** — 核心新增 Agent，必须先完成
2. **现有 6 个 Agent 更新** — 各 Agent 增加 Profile 读取和模式分流逻辑
3. **test-strategist 角色定义** — test-scout 的"无测试"分支行为规范
4. **RAGFlow 源码获取与分析** — Analyzer 完成后产出 Profile

---

## 9. 边界与不做的事

- **不做**：修改 pyvastbase SDK — 除非适配过程中发现 SDK 缺陷
- **不做**：RAGFlow Web UI 测试 — Demo 限定在 API 层面
- **不做**：一次性支持 RAGFlow 所有向量数据库操作 — 按 Profile 诊断结果确定范围
- **不做**：为 RAGFlow 设计独立的测试框架 — 用标准 pytest 脚本

---

## 10. 风险

| 风险 | 缓解 |
|------|------|
| RAGFlow 耦合度比预期高，需大量修改源码 | Analyzer 诊断时识别，Issue Splitter 按复杂度拆子任务 |
| RAGFlow 内部接口不稳定，版本升级后适配代码失效 | 指定 release 版本适配，Profile 记录版本号 |
| Framework Analyzer 诊断错误（判错模式/漏文件） | 人审 Gate 兜底 |
| 无参照测试导致测试方案不完整 | test-strategist 产出测试清单后在评论区公示，人工补漏 |
