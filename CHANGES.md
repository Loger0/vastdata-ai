# LlamaIndex Vastbase 适配 — 修改总结

## 概述

将 LlamaIndex 的 VectorStore 集成从 PostgreSQL/pgvector 适配到 Vastbase G100。实现了一个完整的向量存储后端，包含 CRUD 操作、14 种 FilterOperator 元数据过滤、4 种查询模式（DENSE/SPARSE/HYBRID/MMR）和异步 API。

## 修改文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `llama_index/vector_stores/vastbase/__init__.py` | 新增 | 包入口，导出 VastbaseVectorStore 和工具函数 |
| `llama_index/vector_stores/vastbase/base.py` | 新增 | VastbaseVectorStore 核心实现：连接管理 + CRUD + 4 种查询模式 + 异步 API |
| `llama_index/vector_stores/vastbase/utils.py` | 新增 | MetadataFilters → SQL WHERE 子句翻译（14 种操作符 + AND/OR 嵌套）+ 键值验证 |
| `pyproject.toml` | 新增 | 项目配置，依赖 pyvastbase>=0.2.0、llama-index-core>=0.13.0、psycopg>=3.0 |
| `tests/test_vastbase_vector_store.py` | 新增 | 77 个单元测试覆盖 CRUD + 连接管理 + 异步 CRUD |
| `tests/test_filter_translation.py` | 新增 | 52 个测试覆盖所有 14 种 FilterOperator、AND/OR 嵌套、键验证、key_prefix |
| `tests/test_query_engine.py` | 新增 | 32 个查询引擎单元测试（mock pyvastbase/psycopg） |
| `tests/test_integration.py` | 新增 | 23 个集成测试（需真实 Vastbase 实例） |
| `tests/test_vastbase.py` | 新增 | 22 个初始化/序列化测试 |

## 与上游框架的差异

| 差异项 | pgvector (上游) | Vastbase |
|--------|----------|----------|
| 向量引擎 | 需要 `CREATE EXTENSION vector` | 内置向量引擎（floatvector），无需扩展 |
| 距离类型 | `<=>` (cosine), `<->` (L2), `<#>` (IP) | pyvastbase metric_type (cosine/L2/IP) |
| 索引类型 | ivfflat / hnsw (SQL DDL) | pyvastbase IndexParams (graph_index / fulltext_index) |
| 客户端 | psycopg2 / asyncpg (通过 SQLAlchemy) | pyvastbase (VastbaseClient + Collection API) |
| 全文搜索 | to_tsvector() / to_tsquery() + GIN | PostgreSQL full-text search (psycopg 原生连接) |
| JSONB 操作符 | 全部 PG JSONB 操作符 (`->>`, `->`, `?\|`, `?&`, `@>`) | 相同（Vastbase PG 协议兼容） |
| 异步实现 | asyncpg (SQLAlchemy 异步引擎) | asyncio.to_thread() 包装同步路径 |
| 自定义查询 | SQLAlchemy Select 对象可定制 | 不支持（warned） |

## 已发现并修复的 pyvastbase 兼容性问题

| 问题 | 根因 | 修复 |
|------|------|------|
| 表创建后无法定位 | pyvastbase schema 反射硬编码 `table_schema='public'`，但用户默认 schema 为 `aidev` | `SET search_path TO public` 在连接时执行 |
| 搜索 filter 被丢弃 | `_VastbaseWrapper.search()` 不转发 `**kwargs` 到 `col.search()` | 添加 `**kwargs` 转发 |
| metric_type 大小写 | pyvastbase 期望小写 `cosine`，适配器传递大写 `COSINE` | 统一为小写 |
| AsyncCollection 加载 schema 失败 | pyvastbase 0.2.6 异步 `_load_schema_async` 传递 list 而非 dict 给 psycopg named placeholders | 异步方法改用 `asyncio.to_thread()` 包装同步路径 |
| 稀疏搜索结果 embedding 为字符串 | pyvastbase 将向量序列化为 JSON 字符串 | 添加 `json.loads()` 解析 |

## 测试覆盖

| 层级 | 用例数 | 来源 | 结果 |
|------|--------|------|------|
| Filter 翻译单元测试 | 52 | adapter-dev | ✅ 52/52 PASS |
| Query 引擎单元测试 | 32 | adapter-dev | ✅ 32/32 PASS |
| VectorStore 单元测试 | 77 | adapter-dev | ✅ 77/77 PASS |
| Vastbase 初始化测试 | 22 | adapter-dev | ✅ 22/22 PASS |
| **小计（单元测试）** | **183** | — | **✅ 183/183 PASS** |
| pyvastbase 集成测试 | 23 | adapter-dev | ✅ 23/23 PASS |
| **总计** | **206** | — | **✅ 206/206 PASS** |
| 框架官方测试 | 158 | test-scout | ⚠️ 未适配方（含 SQLAlchemy 代码） |
