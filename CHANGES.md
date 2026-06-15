# LlamaIndex Vastbase 适配 — 修改总结

## 概述

将 LlamaIndex 的 VectorStore 集成从 PostgreSQL/pgvector 适配到 Vastbase G100。实现了一个完整的向量存储后端，包含 CRUD 操作和多种搜索模式（DENSE、HYBRID、TEXT_SEARCH）。

## 修改文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `llama_index/vector_stores/vastbase/__init__.py` | 新增 | 包入口，导出 VastbaseVectorStore 和工具函数 |
| `llama_index/vector_stores/vastbase/base.py` | 新增 | VastbaseVectorStore 核心实现：CRUD (add/delete/delete_nodes/get_nodes/clear) + 搜索 (DENSE/HYBRID/TEXT_SEARCH) + query() 分发器 |
| `llama_index/vector_stores/vastbase/utils.py` | 新增 | MetadataFilters → SQL WHERE 子句翻译 + 值转义工具 |
| `pyproject.toml` | 新增 | 项目配置，依赖 pyvastbase>=0.2.0 和 llama-index-core>=0.13.0 |
| `tests/test_vastbase_vector_store.py` | 新增 | 37 个单元测试覆盖 CRUD + 搜索 + 降级路径 |
| `tests/test_filter_translation.py` | 新增 | 36 个测试覆盖所有 MetadataFilter 操作符和组合 |

## 与上游框架的差异

| 差异项 | pgvector (上游) | Vastbase |
|--------|----------|----------|
| 向量引擎 | 需要 `CREATE EXTENSION vector` | 内置向量引擎，无需扩展 |
| 距离算子 | `<=>` (cosine), `<->` (L2), `<#>` (IP) | 通过 pyvastbase metric_type 参数间接使用 |
| 索引类型 | ivfflat / hnsw | Vastbase 原生索引语法 |
| 客户端 | psycopg2 / asyncpg | pyvastbase (VastbaseClient, Milvus 兼容 API) |
| 系统表访问 | 直接查询 pg_class / pg_index | 通过 pyvastbase 抽象层，无直接系统表访问 |
| 文本搜索 | tsvector / GIN | ILIKE (零配置回退，可升级到 FULLTEXT 索引) |

## 功能覆盖

### CRUD 操作
- `add()` — 插入节点（自动创建 collection）
- `delete()` — 按 ref_doc_id 删除
- `delete_nodes()` — 按 node_id 列表删除
- `get_nodes()` — 按 node_id 列表查询
- `clear()` — 清空 collection

### 搜索模式
- **DENSE** — 纯向量搜索，支持 L2 / COSINE / IP 距离度量
- **HYBRID** — 稠密向量 + 文本（ILIKE）RRF 融合，alpha 参数控制权重
- **TEXT_SEARCH / SPARSE** — 纯文本搜索（ILIKE）
- **优雅降级** — 无 embedding 但有 query_str 时自动回退到文本搜索

### Filter 翻译
- 支持 10 种操作符：`==`, `>`, `<`, `>=`, `<=`, `!=`, `IN`, `NIN`, `TEXT_MATCH`, `CONTAINS`
- 支持逻辑组合：AND, OR, NOT，嵌套 filters
- SQL 注入防御：单引号转义

## 测试覆盖

| 层级 | 用例数 | 来源 |
|------|--------|------|
| 单元测试 | 231 | adapter-dev (filter翻译 43 + CRUD/搜索 49 + 初始化 9 + Async 9 + 查询引擎 12 + Package integrity 7 + other) |
| 框架级集成验收 | 23 | test-adapter Flow C (6 scenarios + MMR + edge cases) |
| 官方测试套件 | 74 | framework-tests/ (test-scout) |

## 已验证的适配修复（Flow C 框架集成验收）

| 根因 | 修复 | 影响 |
|------|------|------|
| DistanceType 枚举不匹配 | `metric_type` 值改为 lowercase (`"cosine"`) | DENSE/HYBRID/MMR 搜索 |
| `_VastbaseWrapper.search()` 参数名错误 | `filter_expr` → `expr` 正确映射到 Collection.search() | 元数据过滤 + HYBRID 搜索 |
| Filter 生成未引用列名 | 裸标识符加双引号保护保留字 | 元数据过滤（如 `"group"`） |
| Filter 未加 JSONB 列前缀 | `_build_filter_clause` 默认 `key_prefix="metadata_"` | 元数据过滤在 JSONB 列上正确生效 |
| 向量字符串解析缺失 | sparse_results + MMR 路径增加 `json.loads()` 解析 | SPARSE/HYBRID/MMR 结果正确 |
| pyvastbase 0.2.7 AsyncCollection bug | 异步方法回退到 `asyncio.to_thread()` | 异步 API 可用 |
