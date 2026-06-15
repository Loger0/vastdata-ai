# LlamaIndex Vastbase 适配 — 人工测试教程

## 环境要求

- Python 版本：>=3.10
- Vastbase G100 版本：V3 (3.0.8+)
- pyvastbase 版本：0.2.6
- 操作系统：Linux / macOS

## Vastbase 连接信息

确保 Vastbase 实例已启动且向量引擎已启用。

环境变量设置：
```bash
export VASTBASE_HOST=<vastbase-host>
export VASTBASE_PORT=5432
export VASTBASE_DATABASE=vastbase
export VASTBASE_USER=<user>
export VASTBASE_PASSWORD=<password>
```

## 安装步骤

### 1. 克隆适配仓库

```bash
git clone https://github.com/Loger0/vastdata-ai.git
cd vastdata-ai
git checkout feature/llamaindex-vastbase-vector-store
```

### 2. 安装依赖

```bash
pip install -e .
pip install pytest pytest-asyncio
```

### 3. 验证安装

```bash
python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print('OK')"
```

预期输出：`OK`

## 验证步骤

### 单元测试（无需 Vastbase 连接）

```bash
pytest tests/test_filter_translation.py tests/test_query_engine.py tests/test_vastbase_vector_store.py tests/test_vastbase.py -v
```

预期：183 passed

### pyvastbase 集成测试（需 Vastbase 连接）

```bash
VASTBASE_HOST=<host> VASTBASE_PORT=<port> VASTBASE_DATABASE=<db> \
VASTBASE_USER=<user> VASTBASE_PASSWORD=<password> \
pytest tests/test_integration.py -v
```

预期：23 passed（6 CRUD + 7 Search + 4 Async + 6 Package Integrity）

### 全量测试

```bash
VASTBASE_HOST=<host> VASTBASE_PORT=<port> VASTBASE_DATABASE=<db> \
VASTBASE_USER=<user> VASTBASE_PASSWORD=<password> \
pytest tests/ -v
```

预期：206 passed

## 测试覆盖

| 测试类别 | 文件 | 测试数 |
|---------|------|--------|
| Filter 翻译（14 种操作符） | `tests/test_filter_translation.py` | 52 |
| Query 引擎（DENSE/SPARSE/HYBRID/MMR） | `tests/test_query_engine.py` | 32 |
| CRUD + 异步 + 初始化 | `tests/test_vastbase_vector_store.py` | 77 |
| 初始化 + 序列化 | `tests/test_vastbase.py` | 22 |
| 真实 Vastbase 集成 | `tests/test_integration.py` | 23 |

## 在 LlamaIndex 中使用

```python
from llama_index.vector_stores.vastbase import VastbaseVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document

# 创建 Vastbase 向量存储
vector_store = VastbaseVectorStore(
    connection_string="postgresql://user:password@host:5432/database",
    table_name="my_documents",
    embed_dim=1536,
)

# DENSE 向量搜索
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

# 集成到 LlamaIndex
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    [Document(text="Vastbase G100 supports native vector search.")],
    storage_context=storage_context,
)

# 查询
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("vector search")
print(response)
```

## 已知限制

1. **`customize_query_fn` 不支持** — pyvastbase 不暴露 SQLAlchemy Select 对象
2. **Async 方法使用 `asyncio.to_thread()`** — pyvastbase 0.2.6 异步路径有 psycopg named placeholder bug
3. **pyvastbase schema 反射依赖 `search_path`** — 连接时自动执行 `SET search_path TO public`
4. **框架官方测试** — `framework-tests/test_postgres.py` 包含 SQLAlchemy 代码，需 test-scout 适配

## 常见问题

### 连接失败
检查 Vastbase 是否启动，端口是否可达：
```bash
psql -h <host> -p <port> -U <user> -d <database> -c "SELECT 1"
```

### Collection 创建后无法定位
确保 pyvastbase 连接后 search_path 已设置为 `public`：
```sql
ALTER USER <user> SET search_path TO public;
```

### 向量搜索返回空
确保向量引擎已启用，且 embedding 维度与 `embed_dim` 参数一致。
