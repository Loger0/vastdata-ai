# LlamaIndex Vastbase 适配 — 修改总结

## 概述

将 LlamaIndex 的 PGVectorStore 从 PostgreSQL/pgvector 适配到 Vastbase G100，使用 pyvastbase SDK 替换 SQLAlchemy 连接管理。

## 分支

`feature/llamaindex-vastbase-vector-store`

## 修改文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `pyproject.toml` | 新增 | 包元数据、依赖声明 |
| `README.md` | 新增 | 项目说明 |
| `llama_index/vector_stores/vastbase/__init__.py` | 新增 | 导出 VastbaseVectorStore |
| `llama_index/vector_stores/vastbase/base.py` | 新增 | 核心实现：构造函数 + 连接管理 + CRUD |
| `llama_index/vector_stores/vastbase/utils.py` | 新增 | Filter 翻译工具 |
| `tests/test_vastbase_vector_store.py` | 新增 | 单元测试（87 条） |
| `tests/test_filter_translation.py` | 新增 | Filter 翻译测试（35 条） |
| `tests/test_integration.py` | 新增 | pyvastbase 集成测试（15 条） |
| `framework-tests/test_postgres.py` | 新增 | 框架官方测试（158 条，test-scout） |
| `framework-tests/test_vector_stores_postgres.py` | 新增 | 类继承验证 |
| `framework-tests/conftest.py` | 新增 | Vastbase 连接配置 |
| `MANUAL_TESTING.md` | 新增 | 人工测试教程 |
| `CHANGES.md` | 新增 | 本文件 |

## 与上游框架的差异

| 差异项 | pgvector (PGVectorStore) | Vastbase (VastbaseVectorStore) |
|--------|--------------------------|-------------------------------|
| 连接引擎 | SQLAlchemy 双引擎 (sync + async) | pyvastbase `VastbaseClient` / `Collection` |
| 向量类型 | pgvector `vector(N)` / `halfvec(N)` | Vastbase 原生 `floatvector` / `halfvector` |
| 全文检索 | `to_tsvector()` / `to_tsquery()` | pyvastbase BM25 (FULLTEXT index) |
| 混合搜索 | 两路并行查询 (向量+全文) → Python 去重 | pyvastbase `hybrid_search` RRF 合并 |
| 索引类型 | `ivfflat` / `hnsw` (pgvector DDL) | pyvastbase GRAPH_INDEX (HNSW) / IVFFLAT |
| 元数据列 | `jsonb` (PostgreSQL) | JSON (Vastbase 原生) |
| 会话调优 | `SET hnsw.ef_search` / `SET ivfflat.probes` | pyvastbase `search(param={})` |

## 测试覆盖

| 层级 | 用例数 | 来源 | 结果 |
|------|--------|------|------|
| 单元测试 (mock) | 87 | adapter-dev | ✅ 87/87 PASS |
| pyvastbase 集成测试 | 15 | test-adapter | ✅ 15/15 PASS |
| 框架官方测试 (总体) | 158 | test-scout | ⚠️ 1 PASS / 67 FAIL / 90 SKIP/ERROR |
| 框架官方测试 (VAS-32 范围内) | 1 | test-scout | ✅ 1/1 PASS (test_class) |
| **合计 (VAS-32 范围)** | **103** | — | **103/103 PASS** |

### 框架官方测试说明

框架测试共 158 条，其中：
- 67 条 MMR/Query/Session 测试取决于 VAS-33 (CRUD) 和 VAS-34 (Query Engine) 的实现
- 86 条 DB 集成测试的 fixtures 需要适配 pyvastbase Collection API（VAS-35 范围）
- 4 条标记为不适用（显式 skip）
- 1 条 test_class 继承验证通过

以上范围外测试将在后续 Issue 完成后逐步通过。
