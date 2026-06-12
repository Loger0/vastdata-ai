# LlamaIndex VastbaseChatStore 适配 — 修改总结

## 概述

将 LlamaIndex 的 PostgresChatStore 从 PostgreSQL + psycopg 适配到 Vastbase G100 + pyvastbase SDK，实现聊天历史的持久化存储。

## 修改文件

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `llama_index/storage/chat_store/vastbase/__init__.py` | 新增 | 导出 VastbaseChatStore |
| `llama_index/storage/chat_store/vastbase/base.py` | 新增 | VastbaseChatStore 主实现 (~484行) |
| `tests/conftest.py` | 新增 | Vastbase 连接 fixture + 测试数据清理 |
| `tests/test_chat_store.py` | 新增 | 27 条测试用例（单元 + 集成 + 框架官方） |
| `pyproject.toml` | 新增 | 包元数据（依赖：llama-index-core + pyvastbase） |

## 与上游框架的差异

| 差异项 | PostgresChatStore (upstream) | VastbaseChatStore |
|--------|------------------------------|-------------------|
| 驱动 | psycopg >= 3.0.0 | pyvastbase >= 0.2.0 (内置 psycopg3) |
| 消息存储类型 | `ARRAY(JSON)` / `ARRAY(JSONB)` | `TEXT` (JSON 序列化) |
| 追加消息 | `array_cat(value, ARRAY[new])` | Python `list.append()` + UPDATE |
| 按索引删除 | 数组切片 `value[: :idx]` | Python `list.pop(idx)` |
| 删除末尾 | `value[1:array_length(value, 1)-1]` | Python `list.pop()` |
| 并发控制 | 无 | SELECT FOR UPDATE + 3 次重试 + 指数退避 |
| 异步 | asyncpg 原生 async | `asyncio.to_thread()` 桥接 |
| ORM | SQLAlchemy sessionmaker | pyvastbase VastbaseConnection |
| 连接管理 | 外部注入 sessionmaker | 内部缓存 pyvastbase 连接 |

## 测试覆盖

| 层级 | 用例数 | 来源 |
|------|--------|------|
| 单元测试 | 27 | adapter-dev（含 test-scout 框架测试 14 条） |
| pyvastbase 集成测试 | 27（同上，一体运行） | adapter-dev |
| 框架官方测试 | 14（已整合到 test_chat_store.py） | test-scout（适配自 LlamaIndex 上游） |
| 补充测试 | 0（覆盖完整，无需补充） | — |
| **合计** | **27 / 27 PASS** | — |

## 实现的接口

| 方法 | 说明 | 测试数 |
|------|------|--------|
| `set_messages` | 全量设置消息 (upsert) | 3 |
| `get_messages` | 获取消息列表 | 2 |
| `add_message` | 追加单条消息 (带并发锁) | 3 |
| `delete_messages` | 删除全部消息 | 2 |
| `delete_message` | 按索引删除 | 3 |
| `delete_last_message` | 删除末尾消息 | 4 |
| `get_keys` | 获取所有会话 key | 2 |
| 异步桥接 | 全部 async 方法 | 6 |
| 多模态消息 | TextBlock 往返 | 1 |
| 构造/继承 | from_params / from_uri / class_name | 3 |
