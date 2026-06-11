# Multica 生态适配系统架构图

> 生成日期：2026-06-11 | 版本：v5

---

## 1. 系统全景图

```mermaid
graph TB
    subgraph USER["👤 用户层"]
        USER_CC[Claude Code CLI]
        USER_VS[VSCode Extension]
        USER_WEB[Multica Web]
    end

    subgraph SKILLS["🧩 本地 Skills 层"]
        S_CREATE[creating-multica-issues]
        S_OPS[multica-eco-adapter-ops]
        S_VB[vastbase-python-sdk]
        S_VEX[vexdb]
        S_VBDB[vastbase]
        S_TAPD[tapd-openapi]
        S_RAGFLOW[ragflow-dataset-ingest]
    end

    subgraph MEMORY["🧠 Memory 系统"]
        M_INTEGRATION[integration-test-mandatory-gate]
        M_V5[multica-v5-convention-extractor]
    end

    subgraph CONFIG["⚙️ 配置层"]
        C_SETTINGS[settings.json<br/>effortLevel: xhigh<br/>model routing]
        C_LOCAL[settings.local.json<br/>permissions allowlist]
        C_HOOKS[PreToolUse Hooks<br/>TAPD 自动路由]
    end

    subgraph PROXY["🔀 模型代理层"]
        P_OPUS[deepseek-v4-pro<br/>Opus 级]
        P_SONNET[qwen3.7-max<br/>Sonnet 级]
        P_HAIKU[glm-5.1<br/>Haiku 级]
    end

    subgraph MULTICA["☁️ Multica 平台"]
        M_WORKSPACE[Workspace: vastdata-ai<br/>1f1559b5]
        M_PROJECT[Project: Vastbase 生态适配<br/>ce17a693]
        M_SQUAD[Squad: eco-adapter-team<br/>4a956351]

        subgraph AGENTS["🤖 10 Agents"]
            A_DISPATCHER[task-dispatcher]
            A_ANALYST[eco-issue-analyst]
            A_CONVENTION[convention-extractor ✨v5]
            A_SCOUT[test-scout]
            A_SPLITTER[eco-issue-splitter]
            A_DEV[adapter-dev]
            A_REVIEWER[code-reviewer]
            A_TEST[test-adapter]
            A_BUGFIXER[bug-fixer]
            A_WATCHDOG[board-watchdog]
        end

        subgraph PLAT_SKILLS["📦 Platform Skills"]
            PS_VB[vastbase-python-sdk]
            PS_CLI[multica-cli-reference]
            PS_DEV[adapter-dev-vastbase]
            PS_REVIEW[code-reviewer-vastbase]
            PS_TEST[test-adapter-vastbase]
            PS_GIT[gitflow-branching]
        end
    end

    subgraph REPO["📁 Git 仓库层"]
        R_MAIN[main]
        R_FEAT_LANGCHAIN[feature/langchain-vastbase-vectorstore]
        R_FEAT_LLAMA[feature/llamaindex-vastbase-vector-store]
        R_AGENT_WT[agent/test-adapter/*<br/>5 个 Agent Worktrees]
    end

    subgraph ARTIFACTS["📄 产出物"]
        ART_CONVENTION[.multica/conventions/<br/>langchain-vectorstore.yaml]
        ART_CODE[适配代码包<br/>langchain_vastbase/<br/>llama_index/vector_stores/vastbase/]
        ART_TESTS[测试套件<br/>unit + integration + acceptance]
        ART_ISSUES[Multica Issues<br/>VAS-7 ~ VAS-12+]
    end

    USER --> SKILLS
    USER --> MULTICA
    SKILLS --> CONFIG
    SKILLS --> MEMORY
    CONFIG --> PROXY
    PROXY --> MULTICA
    MULTICA --> REPO
    REPO --> ARTIFACTS
    MEMORY --> SKILLS
```

---

## 2. 配置与模型路由

```mermaid
graph LR
    subgraph SETTINGS["settings.json"]
        S1["effortLevel: xhigh"]
        S2["model: sonnet"]
        S3["theme: dark"]
        S4["superpowers plugin v5.1.0"]
    end

    subgraph ENV["环境变量"]
        E1["ANTHROPIC_BASE_URL<br/>→ 172.16.105.104:3000"]
        E2["ANTHROPIC_MODEL<br/>→ deepseek-v4-pro"]
    end

    subgraph ROUTING["模型路由表"]
        R1["Opus 级 → deepseek-v4-pro [1M]"]
        R2["Sonnet 级 → qwen3.7-max [1M]"]
        R3["Haiku 级 → glm-5.1"]
    end

    subgraph HOOKS["PreToolUse Hooks"]
        H1["TAPD 关键词匹配<br/>自动加载 tapd-openapi Skill"]
    end

    SETTINGS --> ENV
    ENV --> ROUTING
    HOOKS --> ROUTING
```

---

## 3. Multica v5 工作流全景

```mermaid
flowchart TB
    START((🚀 新适配任务)) --> CREATE

    subgraph CREATE["📝 Phase 0: Issue 创建"]
        C1["1. 调研上游源码<br/>(PyPI/GitHub)"] --> C2["2. 按模板撰写 Issue"]
        C2 --> C3["3. 写入本地 .md"]
        C3 --> C4["4. 🛑 用户审核确认"]
        C4 --> C5["5. multica issue create"]
        C5 --> C6["6. 打 '设计' 标签"]
    end

    C6 --> DISPATCH

    DISPATCH["task-dispatcher<br/>读 label → 路由"] --> ANALYST

    subgraph PARENT["📋 Phase 1: 父 Issue 分析与设计"]
        ANALYST["eco-issue-analyst<br/>需求分析 + 方案设计"] --> CONVENTION
        CONVENTION["convention-extractor ✨v5<br/>分析上游 → 产出 Convention Spec<br/>↓<br/>.multica/conventions/*.yaml"]
        CONVENTION --> SCOUT
        SCOUT["test-scout<br/>识别 + 准备框架官方测试<br/>🚫 不可跳过"]
        SCOUT --> SPLITTER
        SPLITTER["eco-issue-splitter<br/>拆分为子 Issue"]
    end

    SPLITTER --> CHILDREN

    subgraph CHILDREN["🔧 Phase 2: 子 Issue 开发（并行/串行）"]
        direction TB
        C_START["子 Issue<br/>指派给 adapter-dev"] --> DEV

        subgraph DEV_LOOP["TDD 开发循环"]
            DEV["adapter-dev<br/>Step A4.5: 读取 Convention Spec<br/>Step A6: REFACTOR 逐项核对"]
            DEV --> REVIEW
            REVIEW["code-reviewer<br/>6 维度审查（含框架规范符合度）"]
            REVIEW -->|"通过"| TEST
            REVIEW -->|"不通过"| DEV
            TEST["test-adapter"]
        end

        subgraph GATES["🧪 硬性测试门禁"]
            G1["Gate 1: pyvastbase 集成测试<br/>Phase 2b"]
            G2["Gate 2: 框架官方测试<br/>Phase 3"]
            G1 --> G2
            G2 -->|"全部通过"| DONE_CHILD
            G1 -->|"失败"| BLOCK1["🚫 阻塞 @luoyj"]
            G2 -->|"失败"| BLOCK2["🚫 阻塞 @luoyj"]
        end

        TEST --> GATES
        DONE_CHILD["✅ 子 Issue done"]
    end

    DONE_CHILD --> LAST_CHECK

    LAST_CHECK{"是否最后一个<br/>子 Issue?"} -->|"否"| CHILDREN
    LAST_CHECK -->|"是"| FLOW_C

    subgraph FLOW_C["🏁 Phase 3: 父 Issue 框架集成验收（Flow C）"]
        FC1["Phase 9.5 触发<br/>父 Issue: done → todo<br/>unassign → test-adapter"]
        FC1 --> FC2["test-adapter 编写<br/>tests/test_framework_integration.py<br/>使用框架高层 API"]
        FC2 --> FC3["四层全量测试"]
        FC3 --> FC4["1. 单元测试"]
        FC4 --> FC5["2. pyvastbase 集成测试"]
        FC5 --> FC6["3. 框架官方测试"]
        FC6 --> FC7["4. 框架集成验收测试"]
        FC7 -->|"全部通过"| FC8["in_review → 用户审核 → done ✅"]
        FC7 -->|"失败"| BLOCK3["🚫 阻塞，不可降级"]
    end
```

---

## 4. Agent 协作关系图

```mermaid
graph TB
    subgraph ORCHESTRATION["🎯 编排层"]
        DISPATCHER["task-dispatcher<br/>08c6ff9d<br/>读 Label → 路由"]
        WATCHDOG["board-watchdog<br/>ab7e6e9a<br/>每 10min 巡逻 Board"]
        BUGFIXER["bug-fixer<br/>a0f924f2<br/>Bug 分析修复"]
    end

    subgraph DESIGN["📐 设计层"]
        ANALYST["eco-issue-analyst<br/>b6d1747f<br/>需求分析 + 方案设计"]
        CONVENTION["convention-extractor<br/>e9fd48d4 ✨v5<br/>提取上游编码规范"]
        SCOUT["test-scout<br/>c3ae862b<br/>框架官方测试准备"]
        SPLITTER["eco-issue-splitter<br/>3b92a095<br/>拆分任务"]
    end

    subgraph DEV["💻 开发层"]
        DEV_AGENT["adapter-dev<br/>0544999e<br/>TDD 开发<br/>消费 Convention Spec"]
        REVIEWER["code-reviewer<br/>cdf47ee6<br/>6 维审查<br/>含框架规范符合度"]
        TESTER["test-adapter<br/>ed54586d<br/>3 种测试流程<br/>A/B/C"]
    end

    subgraph PLATFORM_SKILLS["Platform Skills（注入到 Agent）"]
        PS1["adapter-dev-vastbase<br/>017e93a2"]
        PS2["code-reviewer-vastbase<br/>1f059118"]
        PS3["test-adapter-vastbase<br/>f5dc63b9"]
        PS4["vastbase-python-sdk<br/>11f90ebb"]
        PS5["multica-cli-reference<br/>f7ee8612"]
        PS6["gitflow-branching<br/>cdd82b08"]
    end

    DISPATCHER -->|"'设计' label"| ANALYST
    DISPATCHER -->|"Bug label"| BUGFIXER
    DISPATCHER -->|"开发 label"| DEV_AGENT
    WATCHDOG -.->|"监控"| DISPATCHER

    ANALYST --> CONVENTION
    CONVENTION --> SCOUT
    SCOUT --> SPLITTER
    SPLITTER --> DEV_AGENT

    DEV_AGENT -->|"提交 Review"| REVIEWER
    REVIEWER -->|"通过"| TESTER
    REVIEWER -->|"不通过 → 反馈"| DEV_AGENT
    TESTER -->|"Bug 发现"| BUGFIXER
    BUGFIXER -->|"修复后"| DEV_AGENT

    PS1 -.->|"注入"| DEV_AGENT
    PS2 -.->|"注入"| REVIEWER
    PS3 -.->|"注入"| TESTER
    PS4 -.->|"注入"| DEV_AGENT
    PS4 -.->|"注入"| TESTER
    PS6 -.->|"注入"| ALL[所有 Agent]

    CONVENTION -.->|"产出 Convention Spec"| ARTIFACT[".multica/conventions/"]
    ARTIFACT -.->|"消费"| DEV_AGENT
    ARTIFACT -.->|"消费"| REVIEWER
```

---

## 5. 测试门禁体系

```mermaid
flowchart LR
    subgraph CHILD_TEST["子 Issue 测试流程"]
        direction TB
        CT1["Phase 2a: 单元测试<br/>(pytest)"] --> CT2
        CT2["🔴 Gate 1: pyvastbase 集成测试<br/>连接真实 Vastbase"] -->|"通过"| CT3
        CT2 -->|"失败"| CT_BLOCK["🚫 阻塞<br/>不可标 known bug"]
        CT3["🔴 Gate 2: 框架官方测试<br/>(上游框架自带测试)"] -->|"通过"| CT4
        CT3 -->|"缺失/失败"| CT_BLOCK2["🚫 阻塞 @luoyj"]
        CT4["✅ 子 Issue 通过"]
    end

    subgraph PARENT_TEST["父 Issue Flow C: 框架集成验收"]
        direction TB
        PT1["编写 test_framework_integration.py<br/>使用框架高层 API"] --> PT2
        PT2["VectorStoreIndex.from_documents()<br/>as_query_engine()<br/>完整 RAG 链路"]
        PT2 --> PT3["四层全量回归"]
        PT3 --> PT4["Layer 1: 单元测试"]
        PT4 --> PT5["Layer 2: pyvastbase 集成"]
        PT5 --> PT6["Layer 3: 框架官方测试"]
        PT6 --> PT7["🔴 Gate 3: 框架集成验收"]
        PT7 -->|"全部通过"| PT8["✅ 父 Issue done"]
        PT7 -->|"失败"| PT_BLOCK["🚫 阻塞，不可降级"]
    end

    CHILD_TEST --> PARENT_TEST

    subgraph LEGEND["图例"]
        L1["🔴 硬性门禁"]
        L2["✅ 通过"]
        L3["🚫 阻塞"]
    end
```

---

## 6. Convention Spec 数据流（v5 新增）

```mermaid
flowchart TB
    UPSTREAM["上游框架源码<br/>(GitHub / PyPI)"] --> EXTRACTOR

    subgraph EXTRACT["convention-extractor Agent"]
        EXTRACTOR["e9fd48d4"]
        E1["分析上游代码风格"]
        E2["提取项目结构规范"]
        E3["提取 API 模式"]
        E4["提取测试规范"]
        E5["标记置信度<br/>(observed / inferred)"]
    end

    EXTRACTOR --> SPEC[".multica/conventions/<br/>langchain-vectorstore.yaml"]

    SPEC --> CONSUME

    subgraph CONSUME["下游消费"]
        C1["adapter-dev<br/>Step A4.5: 读取 Spec<br/>Step A6: REFACTOR 逐项核对"]
        C2["code-reviewer<br/>第 6 维：框架规范符合度<br/>硬性门禁"]
    end

    subgraph SPEC_CONTENT["Convention Spec 内容"]
        S1["project_structure<br/>包命名 / 模块布局 / pyproject 要求"]
        S2["code_style<br/>import 顺序 / 命名 / 类型注解 / docstring"]
        S3["api_patterns<br/>工厂方法 / 错误处理 / 连接管理 / 异步策略"]
        S4["serialization<br/>JSON 处理 / 列类型 / UUID PK / JSONB / VECTOR"]
        S5["test_conventions<br/>pytest / fixture 模式 / 命名"]
    end

    SPEC --> SPEC_CONTENT
    SPEC_CONTENT --> CONSUME
```

---

## 7. 分支策略与工作空间

```mermaid
graph TB
    subgraph BRANCHES["分支策略"]
        direction TB
        MAIN["main<br/>(仅 README)"]
        FEAT1["feature/langchain-vastbase-vectorstore<br/>11 commits<br/>LangChain 适配"]
        FEAT2["feature/llamaindex-vastbase-vector-store<br/>15 commits<br/>LlamaIndex 适配"]
        FIX["fix/<name><br/>Bug 修复 → 合并 → 删除"]

        MAIN --> FEAT1
        MAIN --> FEAT2
        FEAT1 -.-> FIX
        FEAT2 -.-> FIX
    end

    subgraph WORKTREES["Agent Worktrees"]
        WT1["agent/test-adapter/2a69aea8"]
        WT2["agent/test-adapter/3293e0a3"]
        WT3["agent/test-adapter/4a22674e"]
        WT4["agent/test-adapter/8c842fd1"]
        WT5["agent/test-adapter/df888a43"]
    end

    subgraph RULE["规则"]
        R1["一个项目 = 一个 feature 分支"]
        R2["所有子 Issue 共享同一分支"]
        R3["Agent 在隔离 worktree 中工作"]
        R4["worktree 无变更时自动清理"]
    end

    FEAT1 --> WORKTREES
    FEAT2 --> WORKTREES
```

---

## 8. 项目产出物结构

```mermaid
graph TB
    subgraph LANGCHAIN["LangChain 适配产出"]
        LC1["langchain_vastbase/__init__.py<br/>导出: VastbaseVectorStore, Filter, utils"]
        LC2["langchain_vastbase/vectorstore.py<br/>1162 行 | sync/async CRUD + 搜索"]
        LC3["langchain_vastbase/utils.py<br/>211 行 | Filter → SQL 翻译"]
        LC4["tests/test_vectorstore.py<br/>1069 行"]
        LC5["tests/test_utils.py<br/>298 行"]
        LC6["tests/conftest.py<br/>67 行"]
        LC7["pyproject.toml<br/>langchain-vastbase v0.1.0"]
    end

    subgraph LLAMAINDEX["LlamaIndex 适配产出"]
        LI1["llama_index/vector_stores/vastbase/__init__.py"]
        LI2["llama_index/vector_stores/vastbase/base.py<br/>648 行 | CRUD + DENSE/HYBRID/TEXT"]
        LI3["llama_index/vector_stores/vastbase/utils.py<br/>153 行 | Filter → SQL"]
        LI4["tests/test_vastbase_vector_store.py<br/>1037 行 | 51 tests"]
        LI5["tests/test_filter_translation.py<br/>332 行 | 35 tests"]
        LI6["tests/test_integration.py<br/>454 行 | 23 tests"]
        LI7["tests/test_supplementary.py<br/>224 行 | 10 tests"]
        LI8["tests/conftest.py<br/>527 行 | 7 monkey-patches"]
        LI9["pyproject.toml<br/>llama-index-vector-stores-vastbase v0.1.0"]
    end

    subgraph COMMON["公共配置"]
        CO1[".gitignore"]
        CO2[".python-version (LlamaIndex)"]
        CO3["uv.lock (LlamaIndex)"]
    end
```

---

## 9. 关键技术栈

```mermaid
graph LR
    subgraph DB["数据库"]
        DB1["Vastbase G100<br/>(PG 线协议兼容)"]
    end

    subgraph SDK["Python SDK"]
        SDK1["pyvastbase 0.2.6<br/>VastbaseClient<br/>AsyncCollection<br/>Collection API"]
    end

    subgraph FRAMEWORKS["目标框架"]
        FW1["LangChain<br/>langchain-core<br/>langchain-postgres"]
        FW2["LlamaIndex<br/>llama-index-core<br/>llama-index-vector-stores-postgres"]
    end

    subgraph SEARCH["搜索能力"]
        SR1["DENSE<br/>向量相似度<br/>(COSINE/L2/IP)"]
        SR2["HYBRID<br/>BM25 + DENSE<br/>RRF 融合"]
        SR3["TEXT_SEARCH<br/>BM25 全文<br/>ILIKE fallback"]
    end

    subgraph INFRA["基础设施"]
        INF1["Claude Code"]
        INF2["Multica 平台"]
        INF3["Git Worktrees"]
        INF4["pytest"]
    end

    DB --> SDK1
    SDK1 --> FW1
    SDK1 --> FW2
    FW1 --> SR1
    FW1 --> SR2
    FW2 --> SR1
    FW2 --> SR2
    FW2 --> SR3
```

---

## 10. 核心数据流总结

| 阶段 | 输入 | 处理者 | 输出 |
|------|------|--------|------|
| **Issue 创建** | 上游框架 URL | 用户 + creating-multica-issues Skill | VAS Issue (打"设计"标签) |
| **需求分析** | VAS Issue | eco-issue-analyst | 技术方案设计 |
| **规范提取** ✨v5 | 上游框架源码 | convention-extractor | `.multica/conventions/*.yaml` |
| **测试准备** | 上游框架仓库 | test-scout | 框架官方测试用例 |
| **任务拆分** | 设计方案 + 规范 + 测试 | eco-issue-splitter | 子 Issue 列表 |
| **TDD 开发** | Convention Spec + 子 Issue | adapter-dev | 适配代码 |
| **代码审查** | PR diff + Convention Spec | code-reviewer (6维) | Review 结果 |
| **子 Issue 测试** | 适配代码 | test-adapter (Flow A) | Gate 1+2 通过 |
| **集成验收** | 所有子 Issue 代码 | test-adapter (Flow C) | Gate 3 通过 ✅ |
| **持续监控** | Multica Board | board-watchdog | 每10min 状态更新 |
