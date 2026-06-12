"""Test fixtures for Vastbase LlamaIndex stores."""

import os
import pytest

# Vastbase connection defaults (from issue VAS-20)
VASTBASE_HOST = os.environ.get("VASTBASE_HOST", "172.16.105.107")
VASTBASE_PORT = int(os.environ.get("VASTBASE_PORT", "15432"))
VASTBASE_DATABASE = os.environ.get("VASTBASE_DATABASE", "vastbase")
VASTBASE_USER = os.environ.get("VASTBASE_USER", "aidev")
VASTBASE_PASSWORD = os.environ.get("VASTBASE_PASSWORD", "Vbase_123456")

CHAT_TABLE_NAME = "test_chatstore"


@pytest.fixture(scope="session")
def vastbase_config():
    """Vastbase connection configuration from environment."""
    return {
        "host": VASTBASE_HOST,
        "port": VASTBASE_PORT,
        "database": VASTBASE_DATABASE,
        "user": VASTBASE_USER,
        "password": VASTBASE_PASSWORD,
    }


@pytest.fixture(scope="function")
def chat_store():
    """Create a fresh VastbaseChatStore and clean up after test."""
    from llama_index.storage.chat_store.vastbase import VastbaseChatStore

    store = VastbaseChatStore(
        host=VASTBASE_HOST,
        port=VASTBASE_PORT,
        database=VASTBASE_DATABASE,
        user=VASTBASE_USER,
        password=VASTBASE_PASSWORD,
        table_name=CHAT_TABLE_NAME,
    )
    # Ensure table exists and is empty
    store._initialize()
    conn = store._get_connection()
    conn.connection.execute(f"DELETE FROM {CHAT_TABLE_NAME}")
    conn.connection.commit()

    yield store

    # Cleanup
    conn = store._get_connection()
    conn.connection.execute(f"DELETE FROM {CHAT_TABLE_NAME}")
    conn.connection.commit()


@pytest.fixture(scope="function")
def vector_store(vastbase_config):
    """Create a VastbaseVectorStore for testing with a unique table."""
    import uuid
    from llama_index.vector_stores.vastbase import VastbaseVectorStore

    table_name = f"test_llamaindex_{uuid.uuid4().hex[:12]}"
    vs = VastbaseVectorStore.from_params(
        host=vastbase_config["host"],
        port=vastbase_config["port"],
        database=vastbase_config["database"],
        user=vastbase_config["user"],
        password=vastbase_config["password"],
        table_name=table_name,
        embed_dim=8,
    )

    yield vs

    # Cleanup: drop the test collection
    try:
        vs._initialize()
        vs._collection.drop()
    except Exception:
        pass
