"""Tests for VastbaseChatStore.

Adapted from upstream PostgresChatStore tests:
llama-index-storage-chat-store-postgres/tests/test_chat_store_postgres_chat_store.py

Key differences from upstream:
- No Docker container — connects directly to Vastbase
- No SQLAlchemy session — uses pyvastbase SDK
- No PG array_cat / array slicing — uses Python-side list operations
- No ARRAY(JSON) type — uses TEXT / JSON serialization
"""

import pytest
from llama_index.core.llms import ChatMessage, MessageRole


# ============================================================================
# Test: class inheritance
# ============================================================================


def test_class(chat_store):
    """VastbaseChatStore must extend BaseChatStore."""
    from llama_index.core.storage.chat_store.base import BaseChatStore
    assert isinstance(chat_store, BaseChatStore)
    assert chat_store.class_name() == "VastbaseChatStore"


# ============================================================================
# Test: instance creation
# ============================================================================


def test_from_params():
    """from_params() constructs a valid instance."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore.from_params(
        host="localhost",
        port=15432,
        database="vastbase",
        user="test",
        password="test",
        table_name="test_from_params",
    )
    assert store.table_name == "test_from_params"
    assert store._connection_params["host"] == "localhost"
    assert store._connection_params["port"] == 15432


def test_from_uri():
    """from_uri() parses a PostgreSQL connection URI."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore.from_uri(
        "postgresql://user:pass@host:5432/dbname",
        table_name="test_uri_table",
    )
    assert store.table_name == "test_uri_table"
    assert store._connection_params["host"] == "host"
    assert store._connection_params["port"] == 5432
    assert store._connection_params["database"] == "dbname"
    assert store._connection_params["user"] == "user"
    assert store._connection_params["password"] == "pass"


# ============================================================================
# Test: set_messages + get_messages
# ============================================================================


def test_set_and_get_messages(chat_store):
    """Set messages for a key and retrieve them."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
    ]
    chat_store.set_messages("session-1", messages)

    result = chat_store.get_messages("session-1")
    assert len(result) == 2
    assert result[0].role == MessageRole.USER
    assert result[0].content == "Hello"
    assert result[1].role == MessageRole.ASSISTANT
    assert result[1].content == "Hi there!"


def test_get_messages_missing_key(chat_store):
    """get_messages() for a non-existent key returns empty list."""
    result = chat_store.get_messages("no-such-key")
    assert result == []


def test_set_messages_overwrite(chat_store):
    """Setting messages for an existing key overwrites."""
    messages_v1 = [ChatMessage(role=MessageRole.USER, content="v1")]
    messages_v2 = [ChatMessage(role=MessageRole.USER, content="v2")]

    chat_store.set_messages("overwrite-key", messages_v1)
    chat_store.set_messages("overwrite-key", messages_v2)

    result = chat_store.get_messages("overwrite-key")
    assert len(result) == 1
    assert result[0].content == "v2"


# ============================================================================
# Test: add_message
# ============================================================================


def test_add_message(chat_store):
    """add_message() appends to existing messages."""
    chat_store.set_messages("add-key", [
        ChatMessage(role=MessageRole.USER, content="first")
    ])
    chat_store.add_message("add-key", ChatMessage(
        role=MessageRole.ASSISTANT, content="second"
    ))

    result = chat_store.get_messages("add-key")
    assert len(result) == 2
    assert result[0].content == "first"
    assert result[1].content == "second"


def test_add_message_new_key(chat_store):
    """add_message() on a new key creates the key."""
    chat_store.add_message("new-key", ChatMessage(
        role=MessageRole.USER, content="hello"
    ))

    result = chat_store.get_messages("new-key")
    assert len(result) == 1
    assert result[0].content == "hello"


def test_add_message_preserves_blocks(chat_store):
    """add_message() preserves ChatMessage blocks (TextBlock, etc.)."""
    from llama_index.core.base.llms.types import TextBlock

    msg = ChatMessage(role=MessageRole.USER, content="with blocks")
    chat_store.add_message("blocks-key", msg)

    result = chat_store.get_messages("blocks-key")
    assert len(result) == 1
    assert result[0].content == "with blocks"


# ============================================================================
# Test: delete_messages
# ============================================================================


def test_delete_messages(chat_store):
    """delete_messages() removes all messages for a key."""
    chat_store.set_messages("del-key", [
        ChatMessage(role=MessageRole.USER, content="to delete")
    ])
    deleted = chat_store.delete_messages("del-key")
    assert len(deleted) == 1
    assert deleted[0].content == "to delete"

    # Verify key is gone
    assert chat_store.get_messages("del-key") == []


def test_delete_messages_nonexistent(chat_store):
    """delete_messages() on non-existent key returns None."""
    result = chat_store.delete_messages("no-key")
    assert result is None


# ============================================================================
# Test: delete_message
# ============================================================================


def test_delete_message_by_index(chat_store):
    """delete_message() removes message at specific index."""
    chat_store.set_messages("idx-key", [
        ChatMessage(role=MessageRole.USER, content="msg-0"),
        ChatMessage(role=MessageRole.ASSISTANT, content="msg-1"),
        ChatMessage(role=MessageRole.USER, content="msg-2"),
    ])

    removed = chat_store.delete_message("idx-key", 1)
    assert removed.content == "msg-1"

    result = chat_store.get_messages("idx-key")
    assert len(result) == 2
    assert result[0].content == "msg-0"
    assert result[1].content == "msg-2"


def test_delete_message_first(chat_store):
    """delete_message() at index 0 removes the first message."""
    chat_store.set_messages("first-key", [
        ChatMessage(role=MessageRole.USER, content="first"),
        ChatMessage(role=MessageRole.ASSISTANT, content="second"),
    ])

    removed = chat_store.delete_message("first-key", 0)
    assert removed.content == "first"

    result = chat_store.get_messages("first-key")
    assert len(result) == 1
    assert result[0].content == "second"


def test_delete_message_out_of_range(chat_store):
    """delete_message() with out-of-range index raises ValueError."""
    chat_store.set_messages("range-key", [
        ChatMessage(role=MessageRole.USER, content="only")
    ])

    with pytest.raises(ValueError):
        chat_store.delete_message("range-key", 5)

    with pytest.raises(ValueError):
        chat_store.delete_message("range-key", -1)


# ============================================================================
# Test: delete_last_message
# ============================================================================


def test_delete_last_message(chat_store):
    """delete_last_message() removes the last message."""
    chat_store.set_messages("last-key", [
        ChatMessage(role=MessageRole.USER, content="msg-a"),
        ChatMessage(role=MessageRole.ASSISTANT, content="msg-b"),
    ])

    removed = chat_store.delete_last_message("last-key")
    assert removed.content == "msg-b"

    result = chat_store.get_messages("last-key")
    assert len(result) == 1
    assert result[0].content == "msg-a"


def test_delete_last_message_single(chat_store):
    """delete_last_message() on a single-message key empties it."""
    chat_store.set_messages("single-key", [
        ChatMessage(role=MessageRole.USER, content="lonely")
    ])

    removed = chat_store.delete_last_message("single-key")
    assert removed.content == "lonely"

    result = chat_store.get_messages("single-key")
    assert result == []


def test_delete_last_message_empty(chat_store):
    """delete_last_message() on an empty key returns None."""
    chat_store.set_messages("empty-key", [])
    result = chat_store.delete_last_message("empty-key")
    assert result is None


def test_delete_last_message_nonexistent(chat_store):
    """delete_last_message() on non-existent key returns None."""
    result = chat_store.delete_last_message("ghost-key")
    assert result is None


# ============================================================================
# Test: get_keys
# ============================================================================


def test_get_keys(chat_store):
    """get_keys() returns all stored keys."""
    chat_store.set_messages("session-a", [
        ChatMessage(role=MessageRole.USER, content="a")
    ])
    chat_store.set_messages("session-b", [
        ChatMessage(role=MessageRole.USER, content="b")
    ])
    chat_store.set_messages("session-c", [
        ChatMessage(role=MessageRole.USER, content="c")
    ])

    keys = chat_store.get_keys()
    assert set(keys) == {"session-a", "session-b", "session-c"}


def test_get_keys_empty(chat_store):
    """get_keys() on empty store returns empty list."""
    keys = chat_store.get_keys()
    assert keys == []


# ============================================================================
# Test: async methods
# ============================================================================


@pytest.mark.asyncio
async def test_async_set_and_get_messages(chat_store):
    """aset_messages + aget_messages async bridge."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="Async hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Async reply"),
    ]
    await chat_store.aset_messages("async-1", messages)

    result = await chat_store.aget_messages("async-1")
    assert len(result) == 2
    assert result[0].content == "Async hello"
    assert result[1].content == "Async reply"


@pytest.mark.asyncio
async def test_async_add_message(chat_store):
    """async_add_message bridge."""
    chat_store.set_messages("async-add", [
        ChatMessage(role=MessageRole.USER, content="base")
    ])
    await chat_store.async_add_message(
        "async-add",
        ChatMessage(role=MessageRole.ASSISTANT, content="appended"),
    )

    result = chat_store.get_messages("async-add")
    assert len(result) == 2
    assert result[1].content == "appended"


@pytest.mark.asyncio
async def test_async_delete_messages(chat_store):
    """adelete_messages bridge."""
    chat_store.set_messages("async-del", [
        ChatMessage(role=MessageRole.USER, content="gone")
    ])
    deleted = await chat_store.adelete_messages("async-del")
    assert len(deleted) == 1
    assert deleted[0].content == "gone"


@pytest.mark.asyncio
async def test_async_delete_message(chat_store):
    """adelete_message bridge."""
    chat_store.set_messages("async-del-idx", [
        ChatMessage(role=MessageRole.USER, content="keep"),
        ChatMessage(role=MessageRole.USER, content="remove"),
    ])
    removed = await chat_store.adelete_message("async-del-idx", 1)
    assert removed.content == "remove"
    result = chat_store.get_messages("async-del-idx")
    assert len(result) == 1
    assert result[0].content == "keep"


@pytest.mark.asyncio
async def test_async_delete_last_message(chat_store):
    """adelete_last_message bridge."""
    chat_store.set_messages("async-last", [
        ChatMessage(role=MessageRole.USER, content="start"),
        ChatMessage(role=MessageRole.ASSISTANT, content="end"),
    ])
    removed = await chat_store.adelete_last_message("async-last")
    assert removed.content == "end"


@pytest.mark.asyncio
async def test_async_get_keys(chat_store):
    """aget_keys bridge."""
    chat_store.set_messages("ak1", [ChatMessage(role=MessageRole.USER, content="x")])
    chat_store.set_messages("ak2", [ChatMessage(role=MessageRole.USER, content="y")])

    keys = await chat_store.aget_keys()
    assert set(keys) == {"ak1", "ak2"}


# ============================================================================
# Test: Multimodal messages (blocks)
# ============================================================================


def test_multimodal_messages(chat_store):
    """Chat messages with TextBlock contents survive round-trip."""
    messages = [
        ChatMessage(role=MessageRole.USER, content="text message"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="reply with extra blocks",
        ),
    ]
    chat_store.set_messages("multi-key", messages)

    result = chat_store.get_messages("multi-key")
    assert len(result) == 2
    assert result[0].content == "text message"
    assert result[1].content == "reply with extra blocks"
    assert result[0].role == MessageRole.USER
    assert result[1].role == MessageRole.ASSISTANT
