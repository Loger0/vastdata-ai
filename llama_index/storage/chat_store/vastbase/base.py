"""VastbaseChatStore — LlamaIndex chat store backed by Vastbase.

ADAPT: Uses pyvastbase SDK (no PG ecosystem) for chat history persistence.
Messages are stored as JSON-serialized lists in a key-value table.
Python-side list operations replace upstream PG array_cat()/array slicing.

Concurrency: Read-modify-write operations (add_message, delete_message,
delete_last_message) use SELECT FOR UPDATE row-level locking within an
explicit transaction to prevent lost updates. A 3-attempt retry loop
provides resilience against transient deadlocks.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

# Maximum retry attempts for read-modify-write operations
_MAX_RETRY_ATTEMPTS = 3
# Base sleep seconds for exponential backoff between retries
_RETRY_BACKOFF_BASE = 0.1

# Allowed characters in table_name (alphanumeric + underscore)
_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_table_name(name: str) -> None:
    """Validate table_name contains only safe identifier characters.

    While table_name comes from class construction (not external input),
    this provides defence-in-depth against accidental injection.
    """
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table_name '{name}': must match {_TABLE_NAME_RE.pattern}"
        )


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
    ADAPT: SELECT FOR UPDATE + retry loop prevent concurrent write race conditions.
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
        _validate_table_name(table_name)
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
        """Construct from a connection URI.

        Accepts ``postgresql://`` scheme for compatibility with the upstream
        PostgresChatStore pattern. The underlying connection uses pyvastbase
        (not psycopg directly), but the URI format is the familiar DSN shape.
        """
        from urllib.parse import parse_qs, urlparse

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
        from pyvastbase import VastbaseConnectionError, connect

        if self._conn is not None and self._conn.is_connected:
            return self._conn

        try:
            self._conn = connect(**self._connection_params)
        except VastbaseConnectionError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to Vastbase at "
                f"{self._connection_params['host']}:{self._connection_params['port']}"
            ) from exc
        return self._conn

    def _get_pg_conn(self):
        """Return the underlying psycopg connection (public API path).

        Uses ``VastbaseConnection.connection`` property, which is the
        documented public accessor for the psycopg3 connection object.
        """
        return self._get_connection().connection

    def _initialize(self):
        """Create table if it does not exist.

        Uses the psycopg3 connection.execute() method accessed through
        pyvastbase's public ``.connection`` property.
        """
        if self._initialized:
            return

        pg_conn = self._get_pg_conn()
        try:
            pg_conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    key VARCHAR(512) UNIQUE NOT NULL,
                    value TEXT
                )
                """
            )
            pg_conn.commit()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize chat store table '{self.table_name}'"
            ) from exc
        self._initialized = True

    def close(self):
        """Close the underlying Vastbase connection and release resources."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None
                self._initialized = False

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
        """Set messages for a key (upsert).

        Uses INSERT ... ON CONFLICT DO UPDATE for idempotent write.
        No read-modify-write — single atomic statement.
        """
        self._initialize()
        pg_conn = self._get_pg_conn()
        value = self._serialize_messages(messages)
        try:
            pg_conn.execute(
                f"""
                INSERT INTO {self.table_name} (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, value),
            )
            pg_conn.commit()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to set messages for key '{key}'"
            ) from exc

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Get messages for a key."""
        self._initialize()
        pg_conn = self._get_pg_conn()
        try:
            cur = pg_conn.execute(
                f"SELECT value FROM {self.table_name} WHERE key = %s", (key,)
            )
            row = cur.fetchone()
            cur.close()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to get messages for key '{key}'"
            ) from exc

        if row is None:
            return []
        return self._deserialize_messages(row[0])

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a single message to the list for a key.

        ADAPT: Python-side list.append() replaces upstream array_cat().
        Uses SELECT FOR UPDATE + retry loop to prevent lost updates
        under concurrent access.
        """
        self._initialize()
        new_content = message.model_dump_json()

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRY_ATTEMPTS):
            pg_conn = self._get_pg_conn()
            try:
                with pg_conn.transaction():
                    # Row-level lock prevents concurrent read-modify-write
                    cur = pg_conn.execute(
                        f"SELECT value FROM {self.table_name} "
                        f"WHERE key = %s FOR UPDATE",
                        (key,),
                    )
                    row = cur.fetchone()
                    cur.close()

                    if row and row[0]:
                        messages = json.loads(row[0])
                    else:
                        messages = []
                    messages.append(new_content)

                    value = json.dumps(messages)
                    pg_conn.execute(
                        f"""
                        INSERT INTO {self.table_name} (key, value)
                        VALUES (%s, %s)
                        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                        """,
                        (key, value),
                    )
                # Transaction committed successfully
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                continue

        raise RuntimeError(
            f"Failed to add_message for key '{key}' "
            f"after {_MAX_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete all messages for a key. Returns the deleted messages.

        Uses a transaction to ensure the read and delete are atomic.
        """
        self._initialize()
        pg_conn = self._get_pg_conn()
        try:
            with pg_conn.transaction():
                # Read messages before deletion (FOR UPDATE to prevent races)
                cur = pg_conn.execute(
                    f"SELECT value FROM {self.table_name} "
                    f"WHERE key = %s FOR UPDATE",
                    (key,),
                )
                row = cur.fetchone()
                cur.close()

                if row is None:
                    return None

                existing = self._deserialize_messages(row[0])
                pg_conn.execute(
                    f"DELETE FROM {self.table_name} WHERE key = %s", (key,)
                )
            return existing
        except Exception as exc:
            raise RuntimeError(
                f"Failed to delete messages for key '{key}'"
            ) from exc

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a single message at index ``idx``.

        ADAPT: Python-side list.pop(idx) replaces upstream PG array slicing
        ``value[: :idx]`` (not supported by Vastbase).

        Uses SELECT FOR UPDATE + retry loop to prevent lost updates
        under concurrent access.
        """
        self._initialize()

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRY_ATTEMPTS):
            pg_conn = self._get_pg_conn()
            try:
                with pg_conn.transaction():
                    cur = pg_conn.execute(
                        f"SELECT value FROM {self.table_name} "
                        f"WHERE key = %s FOR UPDATE",
                        (key,),
                    )
                    row = cur.fetchone()
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
                    pg_conn.execute(
                        f"UPDATE {self.table_name} SET value = %s WHERE key = %s",
                        (value, key),
                    )
                return ChatMessage.model_validate_json(removed)
            except ValueError:
                # Re-raise validation errors immediately — no retry
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                continue

        raise RuntimeError(
            f"Failed to delete_message for key '{key}' at index {idx} "
            f"after {_MAX_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete the last message for a key.

        ADAPT: Python-side list.pop() replaces upstream
        ``value[1:array_length(value, 1) - 1]``.

        Uses SELECT FOR UPDATE + retry loop to prevent lost updates
        under concurrent access.
        """
        self._initialize()

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRY_ATTEMPTS):
            pg_conn = self._get_pg_conn()
            try:
                with pg_conn.transaction():
                    cur = pg_conn.execute(
                        f"SELECT value FROM {self.table_name} "
                        f"WHERE key = %s FOR UPDATE",
                        (key,),
                    )
                    row = cur.fetchone()
                    cur.close()

                    if not row or not row[0]:
                        return None

                    messages = json.loads(row[0])
                    if not messages:
                        return None

                    removed = messages.pop()
                    value = json.dumps(messages)
                    pg_conn.execute(
                        f"UPDATE {self.table_name} SET value = %s WHERE key = %s",
                        (value, key),
                    )
                return ChatMessage.model_validate_json(removed)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                continue

        raise RuntimeError(
            f"Failed to delete_last_message for key '{key}' "
            f"after {_MAX_RETRY_ATTEMPTS} attempts"
        ) from last_exc

    def get_keys(self) -> List[str]:
        """Get all distinct keys in the store."""
        self._initialize()
        pg_conn = self._get_pg_conn()
        try:
            cur = pg_conn.execute(
                f"SELECT key FROM {self.table_name} ORDER BY key"
            )
            rows = cur.fetchall()
            cur.close()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to get keys from '{self.table_name}'"
            ) from exc
        return [r[0] for r in rows]
