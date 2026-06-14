# LlamaIndex Vastbase 适配 — 人工测试教程

## 环境要求

- Python 版本：>=3.10
- Vastbase G100 版本：V3 (3.0.8+)
- 操作系统：macOS / Linux
- pyvastbase >= 0.2.3

## Vastbase 连接信息

确保 Vastbase 实例已启动且向量引擎已启用。

环境变量设置：
```bash
export VASTBASE_HOST=<your-host>
export VASTBASE_PORT=15432
export VASTBASE_USER=<your-user>
export VASTBASE_PASSWORD=<your-password>
export VASTBASE_DATABASE=vastbase
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
pip install pyvastbase llama-index-core pytest pytest-asyncio pytest-mock numpy
```

添加项目到 PYTHONPATH：
```bash
export PYTHONPATH=.
```

### 3. 验证安装

```bash
python -c "from llama_index.vector_stores.vastbase import VastbaseVectorStore; print('OK')"
```

预期输出：`OK`

## 验证步骤

### 单元测试（Mock，无需数据库）

```bash
pytest tests/test_vastbase_vector_store.py tests/test_filter_translation.py -v
```

预期：全部 PASS（87 条测试）

### pyvastbase 集成测试（需 Vastbase 实例）

```bash
pytest tests/test_integration.py -v
```

预期：全部 PASS（15 条测试）

### 框架官方测试（需 Vastbase 实例）

```bash
# 注意：部分测试依赖 query/search 功能（后续 Issue 实现）
pytest framework-tests/test_vector_stores_postgres.py -v
```

预期：`test_class` PASS（类继承验证）

## 在 LlamaIndex 中使用

```python
from llama_index.vector_stores.vastbase import VastbaseVectorStore

# 方式 1：连接字符串
store = VastbaseVectorStore(
    connection_string="postgresql://user:pass@host:15432/vastbase",
    table_name="my_documents",
    embed_dim=1536,
)

# 方式 2：from_params()
store = VastbaseVectorStore.from_params(
    host="172.16.105.107",
    port=15432,
    database="vastbase",
    user="aidev",
    password="...",
    table_name="my_documents",
)
```

## 常见问题

### 连接失败
- 检查 Vastbase 是否启动：`nc -zv <host> <port>`
- 检查端口是否可达
- 确认用户名和密码正确

### pyvastbase 兼容性
- 推荐使用 pyvastbase 0.2.3+ 
- `VastbaseClient(uri=...)` 在当前版本中不完全支持 URI 解析，建议通过 `from_params()` 构造
