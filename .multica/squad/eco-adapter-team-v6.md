## 工作流（v6，2026-06-11）

### 阶段 0：框架诊断（新增）

```
Parent Issue 创建
  → 指派给 Framework Analyzer (framework-analyzer)
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

### Agent 注册表

| Agent | ID | 阶段 |
|-------|-----|:---:|
| Framework Analyzer 🆕 | 9102869c-2312-47a5-8c0e-b0c58e57ab5b | 阶段 0 |
| eco-issue-analyst | b6d1747f-6237-46a8-a072-67632e208098 | 阶段 1 |
| convention-extractor | e9fd48d4-22fc-466c-9c74-65da99ff8e9e | 阶段 1 |
| test-scout | c3ae862b-87cf-48c9-a927-4a26d387b7b8 | 阶段 1 |
| eco-issue-splitter | 3b92a095-9d11-485e-914d-b3366024c220 | 阶段 1 |
| adapter-dev | 0544999e-993a-4ba1-9379-1022eb69e3fa | 阶段 2 |
| code-reviewer | cdf47ee6-db2b-43dc-8bc5-f565e6345efa | 阶段 2 |
| test-adapter | ed54586d-f6cb-4ee1-80b4-f0982e02cbd7 | 阶段 2, 3 |
| bug-fixer | a0f924f2-742d-4de8-b9c0-fd7dbc626c1f | 按需 |
| board-watchdog | ab7e6e9a-72ba-40a9-955b-56f45cdd2afb | 持续 |

### Branch & Repository Strategy（v6.1 新增）

**核心原则：所有 Agent 产出物必须落盘到 Git 仓库并 Push 到远程。不可仅在本地保存或仅发布 Issue 评论。**

#### 分支创建规则

- **task-dispatcher** 在阶段 0 首次路由到 framework-analyzer 之前创建 feature 分支（A0 前置步骤）
- 分支命名：`feature/<framework-name>-vastbase-adapter`
- 所有后续阶段（A1-A4、子 Issue 开发、集成验收）共用同一 feature 分支
- framework-analyzer Phase 0 负责**检出**已有分支（不创建）
- Bug 修复使用 `fix/<descriptive-name>`，完成后合并回 feature 分支

```bash
# task-dispatcher 创建：
git checkout -b feature/<framework-name>-vastbase-adapter
git push -u origin feature/<framework-name>-vastbase-adapter

# 后续 Agent 检出：
git checkout feature/<framework-name>-vastbase-adapter
```

#### 各 Agent 产出物落盘路径

| Agent | 产出物 | 落盘路径 |
|-------|--------|---------|
| framework-analyzer | Framework Profile JSON | `.multica/profiles/<framework>-profile.json` |
| eco-issue-analyst | 需求规格 + 方案设计 | `.multica/specs/<framework>-spec.md` |
| convention-extractor | Convention Spec YAML | `.multica/conventions/<framework>-conventions.yaml` |
| test-scout | TEST_PLAN.md | `framework-tests/TEST_PLAN.md` |
| test-scout | 测试用例骨架 | `framework-tests/test_<method>.py` |
| adapter-dev | 适配代码 | 按 `files_to_modify` / `files_to_create` |
| test-adapter | Demo 脚本 | `framework-tests/demo/` |
| test-adapter | 集成测试 | `framework-tests/integration/` |

#### Commit + Push 规则

每个 Agent 完成工作后，**必须执行**：

```bash
git add <产出物路径>
git commit -m "<type>: <agent-name> — <简短描述>"
git push origin feature/<framework-name>-vastbase-adapter
```

- `type`: `profile` / `spec` / `convention` / `test-plan` / `feat` / `test` / `fix`
- Issue 评论中**同时**附上产出物摘要 + 仓库文件链接
- 不可仅发布 Issue 评论而不 commit

#### .gitignore 例外

以下目录**不可**加入 .gitignore：
- `.multica/` — Agent 产出物目录
- `framework-tests/` — 测试计划与测试用例

### 硬性门禁

| Gate | 检查内容 | 阶段 |
|------|------|:---:|
| Framework Profile 人审 | Analyzer 产出 Profile 后必须人工审核确认 | 0 |
| pyvastbase 集成测试 | 每子 Issue 必须通过真实 Vastbase 连接测试 | 2 |
| 框架集成验收 | 父 Issue Flow C 必须使用框架高层 API 验证 | 3 |
| 🆕 应用级 Demo | 父 Issue Flow C 必须通过所有 Demo Scenario | 3 |
| 🆕 文件覆盖检查 | code-reviewer 必须确认所有 `files_to_modify/create` 已覆盖 | 2 |
