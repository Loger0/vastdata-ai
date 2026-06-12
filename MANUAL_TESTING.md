# VastbaseChatStore — 人工测试教程

## 环境要求

- Python 版本：>= 3.10
- Vastbase G100 版本：V3 (3.0.8+)
- 操作系统：macOS / Linux
- pyvastbase >= 0.2.0

## Vastbase 连接信息

确保 Vastbase 实例已启动且可达。

环境变量设置：
```bash
export VASTBASE_HOST=172.16.105.107
export VASTBASE_PORT=15432
export VASTBASE_DATABASE=vastbase
export VASTBASE_USER=aidev
export VASTBASE_PASSWORD=<your-password>
```

## 安装步骤

### 1. 克隆适配仓库

```bash
git clone https://github.com/Loger0/vastdata-ai.git
cd vastdata-ai
git checkout feature/llamaindex-vastbase-stores
```

### 2. 安装依赖

```bash
pip install -e ".[dev]"
```

### 3. 验证安装

```bash
python3 -c "from llama_index.storage.chat_store.vastbase import VastbaseChatStore; print('OK')"
```

预期输出：`OK`

## 验证步骤

### 单元测试 + pyvastbase 集成测试

所有测试均需连接 Vastbase 实例运行（使用 pyvastbase SDK）：

```bash
export VASTBASE_HOST=172.16.105.107
export VASTBASE_PORT=15432
export VASTBASE_DATABASE=vastbase
export VASTBASE_USER=aidev
export VASTBASE_PASSWORD=Vbase_123456

pytest tests/test_chat_store.py -v
```

预期：27 条测试全部 PASS。

### 框架官方测试

框架官方测试由 test-scout 根据 LlamaIndex 上游 PostgresChatStore 测试套件适配，已整合到 `tests/test_chat_store.py` 中（含 14 条 ChatStore 框架测试 + 13 条 adapter-dev 扩展测试）。

预期：全部 PASS。

## 在 LlamaIndex 中使用

### 基本用法

```python
from llama_index.storage.chat_store.vastbase import VastbaseChatStore
from llama_index.core.llms import ChatMessage, MessageRole

# 创建 ChatStore 实例
chat_store = VastbaseChatStore(
    host="172.16.105.107",
    port=15432,
    database="vastbase",
    user="aidev",
    password="Vbase_123456",
    table_name="my_chatstore",
)

# 设置消息
chat_store.set_messages("conversation-1", [
    ChatMessage(role=MessageRole.USER, content="你好"),
    ChatMessage(role=MessageRole.ASSISTANT, content="你好！有什么可以帮助你的？"),
])

# 获取消息
messages = chat_store.get_messages("conversation-1")

# 追加单条消息
chat_store.add_message("conversation-1",
    ChatMessage(role=MessageRole.USER, content="今天天气怎么样？"))

# 获取所有会话 key
keys = chat_store.get_keys()

# 删除最后一条消息
chat_store.delete_last_message("conversation-1")

# 删除指定索引的消息
chat_store.delete_message("conversation-1", idx=1)

# 删除整个会话
chat_store.delete_messages("conversation-1")

# 关闭连接
chat_store.close()
```

### 异步用法

```python
import asyncio

async def main():
    chat_store = VastbaseChatStore(...)
    # 异步 API 通过 asyncio.to_thread() 桥接
    await chat_store.aset_messages("key", [...])
    messages = await chat_store.aget_messages("key")
    await chat_store.async_add_message("key", ChatMessage(...))

asyncio.run(main())
```

### from_uri 方式

```python
chat_store = VastbaseChatStore.from_uri(
    "postgresql://aidev:Vbase_123456@172.16.105.107:15432/vastbase",
    table_name="my_chatstore",
)
```

## 常见问题

### 连接失败
- 检查 Vastbase 是否启动：`telnet 172.16.105.107 15432`
- 检查端口是否可达
- 确认用户名密码正确

### 表已存在
- ChatStore 使用 `CREATE TABLE IF NOT EXISTS`，多次初始化安全
- 如需重建表，手动 `DROP TABLE` 后重新实例化

### 并发安全
- `add_message`、`delete_message`、`delete_last_message` 使用 SELECT FOR UPDATE 行级锁 + 3 次重试，确保并发安全
- 如遇高并发场景下的事务冲突，重试机制会自动恢复
