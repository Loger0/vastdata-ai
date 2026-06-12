"""Test fixtures for Vastbase LLamaIndex stores."""

import os
import pytest

# Vastbase connection defaults (from issue VAS-20)
VASTBASE_HOST = os.environ.get("VASTBASE_HOST", "172.16.105.107")
VASTBASE_PORT = int(os.environ.get("VASTBASE_PORT", "15432"))
VASTBASE_DATABASE = os.environ.get("VASTBASE_DATABASE", "vastbase")
VASTBASE_USER = os.environ.get("VASTBASE_USER", "aidev")
VASTBASE_PASSWORD = os.environ.get("VASTBASE_PASSWORD", "Vbase_123456")

CHAT_TABLE_NAME = "test_chatstore"


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
