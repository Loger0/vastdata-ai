# LlamaIndex Vastbase 适配 — 人工测试教程

## 环境要求

- Python 版本：>=3.9
- Vastbase G100 版本：V3 (3.0.8+)
- 操作系统：Linux / macOS

## Vastbase 连接信息

确保 Vastbase 实例已启动且向量引擎已启用。

环境变量设置：
```bash
export VASTBASE_HOST=localhost
export VASTBASE_PORT=5432
export VASTBASE_USER=vastbase
export VASTBASE_PASSWORD=<your-password>
export VASTBASE_DB=vastbase
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
pip install pyvastbase llama-index-core pytest pytest-asyncio
```

### 3. 验证安装

```bash
python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print('OK')"
```

预期输出：`OK`

## 验证步骤

### 单元测试

```bash
pytest tests/ -v
```

预期：全部 PASS（231 tests）

### 框架级集成验收测试

```bash
# 设置 Vastbase 连接环境变量
export VASTBASE_HOST=172.16.105.107
export VASTBASE_PORT=15432
export VASTBASE_DATABASE=vastbase
export VASTBASE_USER=aidev
export VASTBASE_PASSWORD=Vbase_123456

# 运行框架级集成验收
pytest tests/test_framework_integration.py -v
```

验收覆盖 6 大场景：文档摄入 / DENSE向量搜索 / HYBRID混合搜索 / 元数据过滤 / 异步API / 表复用

### 测试分类

| 测试类别 | 文件 | 测试数 |
|---------|------|--------|
| Filter 翻译 | `tests/test_filter_translation.py` | 43 |
| CRUD | `tests/test_vastbase_vector_store.py` | 23 |
| DENSE/HYBRID/TEXT 搜索 | `tests/test_vastbase_vector_store.py` | 12 |
| 初始化/异步 | `tests/test_vastbase_vector_store.py` | 14 |
| 包完整性 | `tests/test_integration.py` | 7 |
| 集成测试 | `tests/test_integration.py` | 37 |
| 查询引擎 | `tests/test_query_engine.py` | 12 |
| 框架级集成验收 | `tests/test_framework_integration.py` | 23 |

## 在 LlamaIndex 中使用

```python
from llama_index.vector_stores.vastbase import VastbaseVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# 创建 Vastbase 向量存储
vector_store = VastbaseVectorStore(
    connection_uri="postgresql://vastbase:password@localhost:5432/vastbase",
    table_name="my_documents",
    dimension=1536,
    distance_metric="COSINE",  # L2 / COSINE / IP
)

# 集成到 LlamaIndex
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# DENSE 向量搜索
retriever = index.as_retriever(similarity_top_k=5)
results = retriever.retrieve("your query")

# HYBRID 混合搜索（稠密 + 文本）
from llama_index.core.vector_stores.types import VectorStoreQueryMode
query_engine = index.as_query_engine(
    vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    similarity_top_k=5,
    alpha=0.7,  # 稠密权重
)
response = query_engine.query("your query")
```

## 常见问题

### 连接失败
检查 Vastbase 是否启动，端口是否可达：
```bash
psql -h localhost -p 5432 -U vastbase -d vastbase -c "SELECT 1"
```

### 向量搜索返回空
确保向量引擎已启用，且数据库中的 embedding 维度与配置的 `dimension` 一致。

### 测试中 mock 依赖
所有单元测试使用 mock，无需真实 Vastbase 连接即可运行。
