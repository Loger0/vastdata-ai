"""VastbaseChatStore — LlamaIndex chat store backed by Vastbase.

ADAPT: Uses pyvastbase SDK (no PG ecosystem) for chat history persistence.
Messages are stored as JSON-serialized lists in a key-value table.
Python-side list operations replace upstream PG array_cat()/array slicing.
"""

import json
from typing import Any, Dict, List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore


class VastbaseChatStore(BaseChatStore):
    """Chat store backed by Vastbase using pyvastbase SDK.

    Table schema::

        CREATE TABLE {table_name} (
            id SERIAL PRIMARY KEY,
            key VARCHAR(512) UNIQUE NOT NULL,
            value TEXT  -- JSON-serialized list of ChatMessage.model_dump_json()
        )

    ADAPT: Messages stored as JSON text column (not PG ARRAY(JSON)).
    ADAPT: Python-side list.append/pop replace upstream array_cat/array slicing.
    ADAPT: No CREATE EXTENSION needed — Vastbase has no pgvector dependency.
    """

    table_name: str = "chatstore"

    # Private attrs — not serialized by Pydantic
    _connection_params: Dict[str, Any]
    _initialized: bool = False

    def __init__(
        self,
        host: str = "localhost",
        port: int = 15432,
        database: str = "vastbase",
        user: str = "aidev",
        password: str = "Vbase_123456",
        table_name: str = "chatstore",
        **kwargs: Any,
    ):
        """Initialize VastbaseChatStore.

        Args:
            host: Vastbase host address.
            port: Vastbase port.
            database: Database name.
            user: Database user.
            password: Database password.
            table_name: Table name for chat messages (default ``"chatstore"``).
        """
        super().__init__(table_name=table_name, **kwargs)
        self._connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        self._conn = None
        self._initialized = False

    @classmethod
    def from_params(
        cls,
        host: str = "localhost",
        port: int = 15432,
        database: str = "vastbase",
        user: str = "aidev",
        password: str = "Vbase_123456",
        table_name: str = "chatstore",
    ) -> "VastbaseChatStore":
        """Construct from connection parameters."""
        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
        )

    @classmethod
    def from_uri(cls, uri: str, table_name: str = "chatstore") -> "VastbaseChatStore":
        """Construct from a PostgreSQL connection URI."""
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 15432
        database = (parsed.path or "/vastbase").lstrip("/")
        user = parsed.username or "aidev"
        password = parsed.password or "Vbase_123456"

        # Parse query params for table_name override
        params = parse_qs(parsed.query)
        if "table_name" in params:
            table_name = params["table_name"][0]

        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
        )

    @classmethod
    def class_name(cls) -> str:
        return "VastbaseChatStore"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_connection(self):
        """Get or establish pyvastbase connection.

        Uses a per-instance cached connection to avoid global connection
        registry issues across tests.
        """
        from pyvastbase import connect

        if self._conn is not None and self._conn.is_connected:
            return self._conn

        self._conn = connect(**self._connection_params)
        return self._conn

    def _initialize(self):
        """Create table if it does not exist."""
        if self._initialized:
            return

        conn = self._get_connection()
        # ADAPT: Use psycopg3 raw SQL via pyvastbase connection (no SQLAlchemy)
        conn.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                key VARCHAR(512) UNIQUE NOT NULL,
                value TEXT
            )
            """
        )
        conn.connection.commit()
        self._initialized = True

    def _serialize_messages(self, messages: List[ChatMessage]) -> str:
        """Serialize list of ChatMessage to JSON string."""
        return json.dumps([m.model_dump_json() for m in messages])

    def _deserialize_messages(self, value: str) -> List[ChatMessage]:
        """Deserialize JSON string back to list of ChatMessage."""
        if not value:
            return []
        raw_list = json.loads(value)
        return [ChatMessage.model_validate_json(m) for m in raw_list]

    # ------------------------------------------------------------------
    # CRUD — sync
    # ------------------------------------------------------------------

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Set messages for a key (upsert)."""
        self._initialize()
        conn = self._get_connection()
        value = self._serialize_messages(messages)
        # ADAPT: Use INSERT ... ON CONFLICT DO UPDATE (Vastbase supports upsert)
        conn.connection.execute(
            f"""
            INSERT INTO {self.table_name} (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (key, value),
        )
        conn.connection.commit()

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        self._initialize()
        conn = self._get_connection()
        cur = conn._execute(
            f"SELECT value FROM {self.table_name} WHERE key = %s", (key,)
        )
        try:
            row = cur.fetchone()
        finally:
            cur.close()

        if row is None:
            return []
        return self._deserialize_messages(row[0])

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a single message to the list for a key.

        ADAPT: Python-side list.append() replaces upstream array_cat().
        """
        self._initialize()
        conn = self._get_connection()
        # Read current messages
        cur = conn._execute(
            f"SELECT value FROM {self.table_name} WHERE key = %s", (key,)
        )
        try:
            row = cur.fetchone()
        finally:
            cur.close()

        if row and row[0]:
            messages = json.loads(row[0])
        else:
            messages = []
        messages.append(message.model_dump_json())

        value = json.dumps(messages)
        conn.connection.execute(
            f"""
            INSERT INTO {self.table_name} (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (key, value),
        )
        conn.connection.commit()

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete all messages for a key. Returns the deleted messages."""
        self._initialize()
        existing = self.get_messages(key)
        if not existing:
            return None

        conn = self._get_connection()
        conn.connection.execute(
            f"DELETE FROM {self.table_name} WHERE key = %s", (key,)
        )
        conn.connection.commit()
        return existing

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a single message at index ``idx``.

        ADAPT: Python-side list.pop(idx) replaces upstream PG array slicing
        ``value[: :idx]`` (not supported by Vastbase).
        """
        self._initialize()
        conn = self._get_connection()
        cur = conn._execute(
            f"SELECT value FROM {self.table_name} WHERE key = %s", (key,)
        )
        try:
            row = cur.fetchone()
        finally:
            cur.close()

        if not row or not row[0]:
            return None

        messages = json.loads(row[0])
        if idx < 0 or idx >= len(messages):
            raise ValueError(
                f"Index {idx} out of range for key '{key}' "
                f"(got {len(messages)} messages)"
            )

        removed = messages.pop(idx)
        value = json.dumps(messages)
        conn.connection.execute(
            f"UPDATE {self.table_name} SET value = %s WHERE key = %s",
            (value, key),
        )
        conn.connection.commit()
        return ChatMessage.model_validate_json(removed)

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete the last message for a key.

        ADAPT: Python-side list.pop() replaces upstream
        ``value[1:array_length(value, 1) - 1]``.
        """
        self._initialize()
        conn = self._get_connection()
        cur = conn._execute(
            f"SELECT value FROM {self.table_name} WHERE key = %s", (key,)
        )
        try:
            row = cur.fetchone()
        finally:
            cur.close()

        if not row or not row[0]:
            return None

        messages = json.loads(row[0])
        if not messages:
            return None

        removed = messages.pop()
        value = json.dumps(messages)
        conn.connection.execute(
            f"UPDATE {self.table_name} SET value = %s WHERE key = %s",
            (value, key),
        )
        conn.connection.commit()
        return ChatMessage.model_validate_json(removed)

    def get_keys(self) -> List[str]:
        """Get all distinct keys in the store."""
        self._initialize()
        conn = self._get_connection()
        cur = conn._execute(f"SELECT key FROM {self.table_name} ORDER BY key")
        try:
            rows = cur.fetchall()
        finally:
            cur.close()
        return [r[0] for r in rows]
