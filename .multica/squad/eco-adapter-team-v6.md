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
| Framework Analyzer 🆕 | （注册后填入） | 阶段 0 |
| eco-issue-analyst | b6d1747f-6237-46a8-a072-67632e208098 | 阶段 1 |
| convention-extractor | e9fd48d4-22fc-466c-9c74-65da99ff8e9e | 阶段 1 |
| test-scout | c3ae862b-87cf-48c9-a927-4a26d387b7b8 | 阶段 1 |
| eco-issue-splitter | 3b92a095-9d11-485e-914d-b3366024c220 | 阶段 1 |
| adapter-dev | 0544999e-993a-4ba1-9379-1022eb69e3fa | 阶段 2 |
| code-reviewer | cdf47ee6-db2b-43dc-8bc5-f565e6345efa | 阶段 2 |
| test-adapter | ed54586d-f6cb-4ee1-80b4-f0982e02cbd7 | 阶段 2, 3 |
| bug-fixer | a0f924f2-742d-4de8-b9c0-fd7dbc626c1f | 按需 |
| board-watchdog | ab7e6e9a-72ba-40a9-955b-56f45cdd2afb | 持续 |

### 硬性门禁

| Gate | 检查内容 | 阶段 |
|------|------|:---:|
| Framework Profile 人审 | Analyzer 产出 Profile 后必须人工审核确认 | 0 |
| pyvastbase 集成测试 | 每子 Issue 必须通过真实 Vastbase 连接测试 | 2 |
| 框架集成验收 | 父 Issue Flow C 必须使用框架高层 API 验证 | 3 |
| 🆕 应用级 Demo | 父 Issue Flow C 必须通过所有 Demo Scenario | 3 |
| 🆕 文件覆盖检查 | code-reviewer 必须确认所有 `files_to_modify/create` 已覆盖 | 2 |
