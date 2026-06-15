"""
VastbaseVectorStore — LlamaIndex vector store integration for Vastbase V3.

ADAPT: Vastbase V3 has a built-in vector engine compatible with
PostgreSQL/pgvector. We wrap pyvastbase (VastbaseClient) to provide a
Milvus-style interface that LlamaIndex's BasePydanticVectorStore expects.

The public constructor mirrors PGVectorStore's 19-parameter signature for
drop-in compatibility.  Internally, pyvastbase replaces SQLAlchemy:
``connect()`` + ``VastbaseClient`` / ``Collection`` / ``AsyncCollection``.
"""

import json
import logging
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlparse, urlunparse

from pydantic import Field, PrivateAttr

from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

_logger = logging.getLogger(__name__)

# ── Input validation (module-level) ─────────────────────────────────────

# ADAPT: allow only UUIDs and alphanumeric/hyphen/underscore identifiers
# in user-supplied values that are embedded in raw SQL expr strings.
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_safe_id(value: str, label: str) -> None:
    """Raise ``ValueError`` if *value* contains unsafe characters.

    Args:
        value: The identifier to validate.
        label: Human-readable label for the error message.

    Raises:
        ValueError: If *value* fails the safe-id pattern check.
    """
    if not _SAFE_ID_PATTERN.match(value):
        raise ValueError(
            f"{label} contains unsafe characters: {value!r}. "
            f"Only [a-zA-Z0-9_-] characters are permitted."
        )


# ADAPT: re-use PGVectorStore's PGType literal for indexed_metadata_keys
# compatibility.  Vastbase/PostgreSQL supports the same type set, so the
# same type names are valid.
PGType = str  # simplified — see PGVectorStore for the full Literal


_TOKENIZER_MAP: Dict[str, str] = {
    "english": "en_tokenizer",
    "simple": "en_tokenizer",
    "chinese": "cn_tokenizer",
}


def _map_text_search_config(config: str) -> str:
    """Map a PG ``text_search_config`` name to a pyvastbase tokenizer.

    ADAPT: Vastbase uses tokenizer names (``en_tokenizer``, ``cn_tokenizer``)
    instead of PostgreSQL text search configuration names.  Unknown config
    values fall back to ``en_tokenizer``.

    Args:
        config: A PG text search config name (e.g. ``"english"``, ``"chinese"``).

    Returns:
        Pyvastbase tokenizer name (e.g. ``"en_tokenizer"``, ``"cn_tokenizer"``).
    """
    mapped = _TOKENIZER_MAP.get(config, "en_tokenizer")
    if config not in _TOKENIZER_MAP:
        _logger.warning(
            "Unknown text_search_config '%s', falling back to 'en_tokenizer'",
            config,
        )
    return mapped


class _VastbaseWrapper:
    """Thin wrapper around standalone pyvastbase functions + Collection API.

    ADAPT: pyvastbase 0.2.x ``VastbaseClient`` passes ``using=`` to internal
    utility functions that do not accept it, causing ``TypeError`` on every
    call.  This wrapper delegates directly to the standalone functions (for
    DDL) and ``Collection`` objects (for data operations), avoiding the
    broken client path entirely.
    """

    # ── DDL operations (standalone functions) ──────────────────────────

    def has_collection(self, collection_name: str) -> bool:
        from pyvastbase import has_collection  # type: ignore[import-untyped]
        return has_collection(collection_name)

    def create_collection(self, collection_name: str, fields: list) -> None:
        # ADAPT: Use Collection + CollectionSchema (not standalone
        # create_collection) so the internal schema cache stays
        # consistent — Collection() can then find the collection.
        from pyvastbase import (  # type: ignore[import-untyped]
            Collection,
            CollectionSchema,
            FieldSchema,
            DataType as VBDataType,
        )
        # Convert the dict-style fields to FieldSchema objects
        schema_fields = []
        for f in fields:
            fs = FieldSchema(
                name=f["name"],
                dtype=f["dtype"],
                is_primary_key=f.get("is_primary_key", False),
                max_length=f.get("max_length"),
                dim=f.get("dim"),
            )
            schema_fields.append(fs)
        schema = CollectionSchema(name=collection_name, fields=schema_fields)
        col = Collection(collection_name, schema=schema)
        col.create()

    def create_index(
        self, *, collection_name: str, field_name: str, index_params: Any
    ) -> None:
        from pyvastbase import Collection  # type: ignore[import-untyped]
        col = Collection(collection_name)
        col.create_index(field_name=field_name, index_params=index_params)

    def truncate_collection(self, collection_name: str) -> None:
        from pyvastbase import Collection  # type: ignore[import-untyped]
        col = Collection(collection_name)
        col.truncate()

    def drop_collection(self, collection_name: str) -> None:
        from pyvastbase import drop_collection  # type: ignore[import-untyped]
        drop_collection(collection_name)

    # ── Data operations (via Collection) ───────────────────────────────

    def _col(self, collection_name: str):
        """Get a Collection instance for the named table."""
        from pyvastbase import Collection  # type: ignore[import-untyped]
        return Collection(collection_name)

    def insert(self, collection_name: str, data: list) -> Any:
        return self._col(collection_name).insert(data)

    def delete(self, collection_name: str, *, expr: str) -> Any:
        return self._col(collection_name).delete(expr=expr)

    def query(
        self,
        collection_name: str,
        *,
        expr: str = "",
        limit: int = 10000,
        output_fields: Optional[list] = None,
        **kwargs: Any,
    ) -> list:
        return self._col(collection_name).query(
            expr=expr,
            limit=limit,
            output_fields=output_fields or [],
        )

    def search(
        self,
        collection_name: str,
        *,
        data: list,
        limit: int = 10,
        output_fields: Optional[list] = None,
        filter_expr: str = "",
        metric_type: str = "L2",
        **kwargs: Any,
    ) -> list:
        return self._col(collection_name).search(
            data=data,
            limit=limit,
            output_fields=output_fields or [],
            filter_expr=filter_expr,
            metric_type=metric_type,
        )

    def close(self) -> None:
        from pyvastbase import remove_connection  # type: ignore[import-untyped]
        try:
            remove_connection("default")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# VastbaseVectorStore
# ═══════════════════════════════════════════════════════════════════════════

class VastbaseVectorStore(BasePydanticVectorStore):
    """LlamaIndex Vector Store backed by Vastbase V3 via pyvastbase.

    Uses pyvastbase's ``VastbaseClient`` (MilvusClient-compatible API) for
    collection management and data operations.  Vector operations are handled
    natively by Vastbase's built-in vector engine — no ``CREATE EXTENSION``
    needed.

    The constructor signature mirrors ``PGVectorStore`` for drop-in
    compatibility.  SQLAlchemy-specific parameters are accepted but emit
    deprecation warnings — pyvastbase handles connection management
    internally.

    Args:
        connection_string: PostgreSQL connection string for Vastbase
            (e.g. ``"postgresql://user:pass@host:5432/database"``).
        table_name: Name of the collection/table to store nodes in.
            Defaults to ``"llamaindex"``.
        embed_dim: Dimensionality of the embedding vectors.
            Defaults to 1536 (OpenAI text-embedding-ada-002).
    """

    # ── VectorStore protocol flags ──────────────────────────────────────
    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    # ── Connection / schema ─────────────────────────────────────────────
    connection_string: str = Field(
        default="",
        description="PostgreSQL connection URI for Vastbase (postgresql://...)",
    )
    async_connection_string: str = Field(
        default="",
        description="Async connection string (stored for PGVectorStore compat; "
        "pyvastbase does not use a separate async connection string)",
    )
    table_name: str = Field(
        default="llamaindex",
        description="Collection/table name for storing nodes",
    )
    schema_name: str = Field(
        default="public",
        description="PostgreSQL schema name (stored for PGVectorStore compat)",
    )

    # ── Vector / index configuration ────────────────────────────────────
    embed_dim: int = Field(
        default=1536,
        description="Vector embedding dimension",
    )
    hybrid_search: bool = Field(
        default=False,
        description="Enable hybrid search (reserved for future use)",
    )
    text_search_config: str = Field(
        default="english",
        description="Text search config for hybrid search (reserved for future use)",
    )
    hnsw_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="HNSW index creation kwargs (reserved for future use)",
    )
    use_halfvec: bool = Field(
        default=False,
        description="Use half-precision vectors (halfvec). "
        "Vastbase >=3.0.9 supports halfvector.",
    )

    # ── Behaviour flags ─────────────────────────────────────────────────
    cache_ok: bool = Field(
        default=False,
        description="SQLAlchemy cache_ok flag (warned — not used by pyvastbase)",
    )
    perform_setup: bool = Field(
        default=True,
        description="Whether to auto-create the collection on first use",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode (enables verbose logging)",
    )
    initialization_fail_on_error: bool = Field(
        default=False,
        description="If True, initialization errors propagate; if False, they are logged",
    )

    # ADAPT: SQLAlchemy-only parameters — accepted but warned.
    # Vastbase uses pyvastbase for connection management, not SQLAlchemy.
    use_jsonb: bool = Field(
        default=False,
        description="[DEPRECATED — Vastbase uses JSON natively] "
        "Use JSONB instead of JSON.  Accepted for PGVectorStore compat; "
        "has no effect with pyvastbase.",
    )
    create_engine_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="[DEPRECATED — pyvastbase manages connections] "
        "SQLAlchemy create_engine kwargs.  Accepted for PGVectorStore compat; "
        "has no effect with pyvastbase.",
    )
    indexed_metadata_keys: Optional[Any] = Field(
        default=None,
        description="[DEPRECATED — not used by pyvastbase] "
        "Metadata keys to index.  Accepted for PGVectorStore compat; "
        "has no effect with pyvastbase.",
    )

    # ── Private state ───────────────────────────────────────────────────
    _client: Any = PrivateAttr(default=None)
    _is_connected: bool = PrivateAttr(default=False)
    # ADAPT: customize_query_fn is accepted but unused — pyvastbase does not
    # expose a SQLAlchemy Select object to customize.
    _customize_query_fn: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)

    # ── Constructor ─────────────────────────────────────────────────────

    def __init__(
        self,
        connection_string: Optional[Union[str, Any]] = None,
        async_connection_string: Optional[Union[str, Any]] = None,
        table_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        initialization_fail_on_error: bool = False,
        use_halfvec: bool = False,
        # ADAPT: SQLAlchemy engine params — accepted for PGVectorStore compat
        # but not used internally.  pyvastbase manages its own connections.
        engine: Optional[Any] = None,
        async_engine: Optional[Any] = None,
        # ADAPT: additional PGVectorStore compat params
        indexed_metadata_keys: Optional[Any] = None,
        customize_query_fn: Optional[Callable[..., Any]] = None,
        # ADAPT: backward-compat aliases for earlier VastbaseVectorStore API
        connection_uri: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """Initialize VastbaseVectorStore.

        The constructor mirrors ``PGVectorStore.__init__`` for drop-in
        compatibility.  Internally pyvastbase replaces SQLAlchemy — see the
        ADAPT notes on each parameter.

        Args:
            connection_string: PostgreSQL connection string for Vastbase.
                Falls back to ``connection_uri`` (deprecated alias).
            async_connection_string: Async connection string (stored but
                unused — pyvastbase does not split sync/async connections).
            table_name: Collection/table name.  Defaults to ``"llamaindex"``.
            schema_name: PostgreSQL schema name.  Defaults to ``"public"``.
            hybrid_search: Enable hybrid search (reserved).  Defaults to False.
            text_search_config: Text search config (reserved).  Defaults to ``"english"``.
            embed_dim: Vector dimension.  Defaults to 1536.  Falls back to
                ``dimension`` (deprecated alias).
            cache_ok: SQLAlchemy cache flag (warned — not used).  Defaults to False.
            perform_setup: Auto-create collection on first use.  Defaults to True.
            debug: Enable verbose logging.  Defaults to False.
            use_jsonb: Use JSONB (warned — Vastbase uses JSON natively).
                Defaults to False.
            hnsw_kwargs: HNSW index creation kwargs (reserved).  Defaults to None.
            create_engine_kwargs: SQLAlchemy engine kwargs (warned — not used).
                Defaults to None.
            initialization_fail_on_error: Propagate init errors.  Defaults to False.
            use_halfvec: Use half-precision vectors.  Defaults to False.
            engine: SQLAlchemy sync engine (warned — not used).
            async_engine: SQLAlchemy async engine (warned — not used).
            indexed_metadata_keys: Metadata key index specs (warned — not used).
            customize_query_fn: Query customization hook (warned — not used).
            connection_uri: Deprecated alias for ``connection_string``.
            dimension: Deprecated alias for ``embed_dim``.
        """
        # ── Resolve deprecated aliases ─────────────────────────────────
        # ADAPT: backward-compat: connection_uri is a deprecated alias for
        # connection_string.  connection_string takes precedence when provided.
        if not connection_string and connection_uri is not None:
            connection_string = connection_uri
        connection_string = str(connection_string or "")
        async_connection_string = str(async_connection_string or "")

        # ADAPT: backward-compat: dimension is a deprecated alias for embed_dim.
        # embed_dim takes precedence when explicitly provided (non-default).
        if embed_dim == 1536 and dimension is not None:
            embed_dim = dimension
        table_name = (table_name or "llamaindex").lower()
        schema_name = (schema_name or "public").lower()
        create_engine_kwargs = create_engine_kwargs or {}

        # ADAPT: warn on SQLAlchemy-specific parameters that pyvastbase
        # does not use.  These are accepted for PGVectorStore drop-in
        # compatibility but are no-ops with pyvastbase.
        self._warn_unsupported_params(
            use_jsonb=use_jsonb,
            create_engine_kwargs=create_engine_kwargs,
            indexed_metadata_keys=indexed_metadata_keys,
            engine=engine,
            async_engine=async_engine,
            customize_query_fn=customize_query_fn,
            cache_ok=cache_ok,
        )

        if hybrid_search and text_search_config is None:
            raise ValueError(
                "Sparse vector index creation requires "
                "a text search configuration specification."
            )

        super().__init__(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs,
            initialization_fail_on_error=initialization_fail_on_error,
            use_halfvec=use_halfvec,
            indexed_metadata_keys=indexed_metadata_keys,
        )

        # ADAPT: store SQLAlchemy compatibility attrs as private state
        self._customize_query_fn = customize_query_fn

        # ADAPT: auto-initialize collection + HNSW/FULLTEXT indexes on
        # construction (mirrors PGVectorStore's __init__ behavior).
        # Skipped when perform_setup=False.
        if self.perform_setup:
            try:
                self._initialize()
            except Exception:
                if self.initialization_fail_on_error:
                    raise
                _logger.warning(
                    "Failed to auto-initialize collection '%s' — "
                    "it will be created on first add() call instead",
                    self.table_name,
                    exc_info=self.debug,
                )

    @staticmethod
    def _warn_unsupported_params(
        use_jsonb: bool = False,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        indexed_metadata_keys: Optional[Any] = None,
        engine: Optional[Any] = None,
        async_engine: Optional[Any] = None,
        customize_query_fn: Optional[Callable[..., Any]] = None,
        cache_ok: bool = False,
    ) -> None:
        """Emit one-time warnings for SQLAlchemy-specific parameters.

        ADAPT: pyvastbase manages connections via ``connect()`` +
        ``VastbaseClient``, so SQLAlchemy-oriented parameters have no effect.
        We warn instead of raising to maintain PGVectorStore drop-in compat.
        """
        if use_jsonb:
            _logger.warning(
                "use_jsonb=True has no effect — Vastbase/pyvastbase uses JSON "
                "natively for metadata columns.  This parameter is accepted "
                "for PGVectorStore compatibility only."
            )
        if create_engine_kwargs:
            _logger.warning(
                "create_engine_kwargs has no effect — pyvastbase manages "
                "connections internally via connect() / VastbaseClient.  "
                "This parameter is accepted for PGVectorStore compatibility only."
            )
        if indexed_metadata_keys is not None:
            _logger.warning(
                "indexed_metadata_keys has no effect — metadata indexing is "
                "not currently supported by the Vastbase pyvastbase backend.  "
                "This parameter is accepted for PGVectorStore compatibility only."
            )
        if engine is not None:
            _logger.warning(
                "engine parameter has no effect — pyvastbase manages its own "
                "connection pool.  This parameter is accepted for "
                "PGVectorStore compatibility only."
            )
        if async_engine is not None:
            _logger.warning(
                "async_engine parameter has no effect — pyvastbase manages "
                "its own async connections.  This parameter is accepted for "
                "PGVectorStore compatibility only."
            )
        if customize_query_fn is not None:
            _logger.warning(
                "customize_query_fn has no effect — pyvastbase does not "
                "expose a SQLAlchemy Select object to customize.  "
                "This parameter is accepted for PGVectorStore compatibility only."
            )
        if cache_ok:
            _logger.warning(
                "cache_ok=True has no effect — SQLAlchemy type caching is "
                "not relevant to pyvastbase.  This parameter is accepted for "
                "PGVectorStore compatibility only."
            )

    # ── Class methods ────────────────────────────────────────────────────

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for serialization."""
        return "VastbaseVectorStore"

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "llamaindex",
        schema_name: str = "public",
        connection_string: Optional[Union[str, Any]] = None,
        async_connection_string: Optional[Union[str, Any]] = None,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        use_halfvec: bool = False,
        indexed_metadata_keys: Optional[Any] = None,
        customize_query_fn: Optional[Callable[..., Any]] = None,
    ) -> "VastbaseVectorStore":
        """Construct a VastbaseVectorStore from individual connection parameters.

        ADAPT: mirrors ``PGVectorStore.from_params()``.  Builds a
        PostgreSQL connection string from ``host``/``port``/``database``/
        ``user``/``password`` when ``connection_string`` is not provided,
        then delegates to ``__init__``.

        Args:
            host: Vastbase host.  Defaults to ``"localhost"``.
            port: Vastbase port.  Defaults to ``5432``.
            database: Database name.  Defaults to ``"vastbase"``.
            user: Database user.
            password: Database password.
            table_name: Collection/table name.  Defaults to ``"llamaindex"``.
            schema_name: Schema name.  Defaults to ``"public"``.
            connection_string: Full connection string (overrides host/port/etc).
            async_connection_string: Async connection string (stored, unused).
            hybrid_search: Enable hybrid search (reserved).
            text_search_config: Text search config (reserved).
            embed_dim: Vector dimension.  Defaults to 1536.
            cache_ok: SQLAlchemy cache flag (warned — not used).
            perform_setup: Auto-create collection.
            debug: Debug mode.
            use_jsonb: Use JSONB (warned — Vastbase uses JSON natively).
            hnsw_kwargs: HNSW kwargs (reserved).
            create_engine_kwargs: SQLAlchemy engine kwargs (warned — not used).
            use_halfvec: Use half-precision vectors.
            indexed_metadata_keys: Metadata index specs (warned — not used).
            customize_query_fn: Query hook (warned — not used).

        Returns:
            VastbaseVectorStore instance.
        """
        # ADAPT: build a PostgreSQL connection string from individual params
        # when no explicit connection_string is provided.
        conn_str = connection_string or cls._build_connection_string(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )
        async_conn_str = async_connection_string or ""

        return cls(
            connection_string=str(conn_str),
            async_connection_string=str(async_conn_str),
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs,
            use_halfvec=use_halfvec,
            indexed_metadata_keys=indexed_metadata_keys,
            customize_query_fn=customize_query_fn,
        )

    @staticmethod
    def _build_connection_string(
        host: Optional[str] = None,
        port: Optional[Union[str, int]] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> str:
        """Build a PostgreSQL connection string from individual parameters.

        ADAPT: Vastbase uses standard PostgreSQL connection URI format:
        ``postgresql://[user[:password]@][host][:port][/database]``

        Args:
            host: Hostname or IP.  Defaults to ``"localhost"``.
            port: Port number.  Defaults to ``5432``.
            database: Database name.  Defaults to ``"vastbase"``.
            user: Username.
            password: Password.

        Returns:
            PostgreSQL connection URI string.
        """
        _host = host or "localhost"
        _port = str(port or 5432)
        _db = database or "vastbase"

        if user and password:
            auth = f"{user}:{password}@"
        elif user:
            auth = f"{user}@"
        else:
            auth = ""

        return f"postgresql://{auth}{_host}:{_port}/{_db}"

    # ── Connection management ─────────────────────────────────────────────

    @property
    def client(self) -> Any:
        """Lazily create and return the pyvastbase VastbaseClient.

        ADAPT: VastbaseClient accepts a PostgreSQL URI directly.
        On first access, ``_connect()`` is called to establish the
        connection if it hasn't been connected yet.
        """
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self) -> None:
        """Establish the connection to Vastbase via pyvastbase.

        ADAPT: Uses ``pyvastbase.connect()`` to register the connection,
        then creates a ``VastbaseClient()`` without arguments to pick up
        the default connection.  pyvastbase 0.2.x ``VastbaseClient(uri=...)``
        is unreliable due to ``using=`` parameter mismatch in internal
        utility calls.

        This replaces SQLAlchemy's ``create_engine()`` / ``create_async_engine()``
        dual-engine pattern.  pyvastbase manages connection pooling internally.
        """
        if self._is_connected:
            return

        from pyvastbase import connect  # type: ignore[import-untyped]

        uri = self.connection_string
        if not uri:
            raise ValueError(
                "connection_string is empty — provide a PostgreSQL URI "
                "or use from_params() with host/port/database/user/password"
            )

        # ADAPT: parse the PG URI into components for connect()
        conn_params = self._parse_connection_string(uri)

        connect(
            host=conn_params["host"],
            port=conn_params["port"],
            database=conn_params["database"],
            user=conn_params["user"],
            password=conn_params["password"],
        )
        self._client = _VastbaseWrapper()
        self._is_connected = True

        if self.debug:
            _logger.debug(
                "VastbaseVectorStore connected to %s (table=%s)",
                self._mask_uri(uri),
                self.table_name,
            )

    def close(self) -> None:
        """Close the pyvastbase client connection.

        ADAPT: calls ``VastbaseClient.close()`` which disposes the
        internal connection pool.  After calling ``close()`` the
        ``client`` property will re-connect on next access.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                _logger.debug("Error closing VastbaseClient", exc_info=True)
            finally:
                self._client = None
                self._is_connected = False

    @staticmethod
    def _parse_connection_string(conn_str: str) -> Dict[str, Optional[str]]:
        """Parse a PostgreSQL connection URI into its components.

        ADAPT: standard ``postgresql://`` URI parsing.  Vastbase uses the
        same URI format as PostgreSQL.

        Args:
            conn_str: PostgreSQL connection URI
                (e.g. ``"postgresql://user:pass@host:5432/database"``).

        Returns:
            Dict with keys ``scheme``, ``user``, ``password``, ``host``,
            ``port``, ``database``.
        """
        result: Dict[str, Optional[str]] = {
            "scheme": None,
            "user": None,
            "password": None,
            "host": None,
            "port": None,
            "database": None,
        }

        if not conn_str:
            return result

        try:
            parsed = urlparse(conn_str)
            result["scheme"] = parsed.scheme or "postgresql"
            result["user"] = parsed.username
            result["password"] = parsed.password
            result["host"] = parsed.hostname
            result["port"] = str(parsed.port) if parsed.port else None
            # Strip leading '/' from path to get database name
            result["database"] = parsed.path.lstrip("/") or None
        except Exception:
            _logger.warning(
                "Failed to parse connection string: %s", conn_str, exc_info=True
            )

        return result

    @staticmethod
    def _mask_uri(uri: str) -> str:
        """Return a copy of the URI with password masked for logging."""
        try:
            parsed = urlparse(uri)
            if parsed.password:
                masked = parsed._replace(
                    netloc=parsed.netloc.replace(
                        f":{parsed.password}@", ":***@"
                    )
                )
                return urlunparse(masked)
        except Exception:
            pass
        return uri

    # ── Initialization ────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Full initialization: create collection + HNSW index + optional FULLTEXT.

        ADAPT: Replaces PGVectorStore's _initialize() which used SQLAlchemy
        DDL (CREATE EXTENSION → CREATE SCHEMA → CREATE TABLE → CREATE INDEX).
        Vastbase V3 uses pyvastbase's collection management and native index API.

        Idempotent — safe to call multiple times.  Skipped entirely when
        ``perform_setup=False``.
        """
        if not self.perform_setup:
            return

        if self._is_initialized:
            return

        # Create collection if it doesn't exist
        if not self.client.has_collection(self.table_name):
            self._create_collection()

        # Create HNSW index on the embedding column
        self._create_hnsw_index()

        # Create FULLTEXT index if hybrid search is enabled
        if self.hybrid_search:
            self._create_fulltext_index()

        self._is_initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure the collection and indexes are created (idempotent).

        Delegates to ``_initialize()`` which is guarded by the
        ``_is_initialized`` flag — subsequent calls are a no-op.
        """
        self._initialize()

    def _create_collection(self) -> None:
        """Create the Vastbase collection/table if it does not exist.

        The collection schema mirrors the LlamaIndex node structure:
        ``id`` (primary key), ``text``, ``embedding`` (float vector),
        ``metadata_`` (JSON), and ``ref_doc_id`` for source-document
        tracking.

        ADAPT: Vastbase's vector engine is built-in — no ``CREATE EXTENSION``
        or pgvector-specific setup is needed.  ``create_collection()`` uses
        standard PostgreSQL types underneath.
        """
        if not self.client.has_collection(self.table_name):
            # ADAPT: use pyvastbase DataType enums for schema definition.
            # Vastbase maps these to native PostgreSQL types automatically.
            from pyvastbase import DataType  # type: ignore[import-untyped]

            # ADAPT: use_halfvec → FLOAT16_VECTOR (half-precision) instead of
            # FLOAT_VECTOR (full-precision).  Vastbase >=3.0.9 supports halfvector.
            vector_dtype = (
                DataType.FLOAT16_VECTOR if self.use_halfvec
                else DataType.FLOAT_VECTOR
            )

            self.client.create_collection(
                self.table_name,
                fields=[
                    {
                        "name": "id",
                        "dtype": DataType.VARCHAR,
                        "is_primary_key": True,
                        "max_length": 256,
                    },
                    {"name": "text", "dtype": DataType.TEXT},
                    {
                        "name": "embedding",
                        "dtype": vector_dtype,
                        "dim": self.embed_dim,
                    },
                    {"name": "metadata_", "dtype": DataType.JSON},
                    {
                        "name": "ref_doc_id",
                        "dtype": DataType.VARCHAR,
                        "max_length": 256,
                    },
                ],
            )

    def _create_hnsw_index(self) -> None:
        """Create HNSW (Hierarchical Navigable Small World) index on the
        embedding column.

        ADAPT: Uses pyvastbase ``IndexParams.graph_index()`` which maps to
        Vastbase's native HNSW implementation.  PGVectorStore used raw SQL
        ``CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)
        WITH (m=..., ef_construction=...)``.  pyvastbase abstracts this into
        ``create_index()`` with ``IndexParams``.

        The index is created with ``IF NOT EXISTS`` semantics — if an index
        already exists on the embedding column, it is not re-created.
        """
        from pyvastbase import IndexParams  # type: ignore[import-untyped]

        # ADAPT: extract HNSW kwargs with PGVectorStore-compatible names
        # (hnsw_m, hnsw_ef_construction) and map to pyvastbase names (m, ef_construction).
        hnsw_kwargs = self.hnsw_kwargs or {}
        m = hnsw_kwargs.get("hnsw_m", 16)
        ef_construction = hnsw_kwargs.get("hnsw_ef_construction", 64)

        params = IndexParams.graph_index(
            m=m,
            ef_construction=ef_construction,
        )
        try:
            self.client.create_index(
                collection_name=self.table_name,
                field_name="embedding",
                index_params=params,
            )
        except Exception as exc:
            # ADAPT: index may already exist — _initialize() is idempotent
            # and may be called on an already-initialised collection.
            # Only suppress "already exists"-type errors; re-raise real
            # errors (connection failure, permission denied, etc.).
            msg = str(exc).lower()
            if "already exists" in msg or "duplicate" in msg:
                _logger.debug(
                    "HNSW index on %s.embedding already exists — skipping",
                    self.table_name,
                )
            else:
                raise

    def _create_fulltext_index(self) -> None:
        """Create FULLTEXT index on the ``text`` column for hybrid search.

        ADAPT: PGVectorStore used ``to_tsvector()`` / ``to_tsquery()`` with
        PostgreSQL GIN indexes.  Vastbase uses native BM25 full-text search
        via pyvastbase's ``IndexParams.fulltext_index()``.

        The ``text_search_config`` parameter is mapped to Vastbase tokenizer
        dictionaries:
        - ``"english"`` → ``"en_tokenizer"``
        - ``"chinese"`` → ``"cn_tokenizer"``
        - any other value is passed through as-is.
        """
        from pyvastbase import IndexParams  # type: ignore[import-untyped]

        # ADAPT: use module-level _map_text_search_config for consistent
        # tokenizer mapping (english→en_tokenizer, chinese→cn_tokenizer,
        # simple→en_tokenizer, unknown→en_tokenizer with warning).
        dictionary = _map_text_search_config(self.text_search_config)

        params = IndexParams.fulltext_index(
            dictionary=dictionary,
            algorithm="BM25",
        )
        self.client.create_index(
            collection_name=self.table_name,
            field_name="text",
            index_params=params,
        )

    # ── Serialization helpers ────────────────────────────────────────────

    @staticmethod
    def _node_to_dict(node: BaseNode) -> Dict[str, Any]:
        """Convert a LlamaIndex ``BaseNode`` to a dict for insertion.

        ADAPT: replaces inline dict-building in ``add()``.  Extracted as a
        standalone helper so both sync ``add()`` and ``async_add()`` can
        re-use the same serialisation logic.

        Args:
            node: A LlamaIndex ``BaseNode`` with ``node_id``, ``text``,
                ``embedding``, ``metadata``, and optional ``ref_doc_id``.

        Returns:
            Dict with keys ``id``, ``text``, ``embedding``, ``metadata_``,
            ``ref_doc_id``.
        """
        return {
            "id": node.node_id,
            "text": node.get_content(),
            "embedding": node.embedding,
            "metadata_": json.dumps(node.metadata or {}),
            "ref_doc_id": node.ref_doc_id or "",
        }

    @staticmethod
    def _dict_to_node(row: Dict[str, Any]) -> TextNode:
        """Convert a single result-row dict into a LlamaIndex ``TextNode``.

        ADAPT: replaces inline dict-to-TextNode logic in ``_parse_results()``.
        Extracted as a standalone helper so both ``_parse_results()`` and
        ``aget_nodes()`` can re-use it.

        Args:
            row: Dict with keys ``id``, ``text``, ``embedding`` (optional),
                ``metadata_`` (optional), ``ref_doc_id`` (optional).

        Returns:
            ``TextNode`` populated from the row data.  If ``ref_doc_id`` is
            present, a ``SOURCE`` relationship is attached so that
            ``node.ref_doc_id`` returns the value.
        """
        node_id: Optional[str] = row.get("id")
        text: str = row.get("text", "")
        embedding: Optional[List[float]] = row.get("embedding")
        # ADAPT: pyvastbase returns vectors as strings (e.g. '[0.1,0.2,0.3]').
        # Parse them back to Python lists for LlamaIndex TextNode.
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except (json.JSONDecodeError, TypeError):
                embedding = None
        metadata: dict = row.get("metadata_") or {}
        # ADAPT: metadata_ is stored as JSON string for psycopg compatibility;
        # parse it back to a dict if it's still a string.
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        ref_doc_id: Optional[str] = row.get("ref_doc_id")

        node = TextNode(
            id_=node_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )
        # ADAPT: ref_doc_id is stored as a separate column but LlamaIndex
        # exposes it as a read-only property backed by source_node.
        # Reconstruct the SOURCE relationship so node.ref_doc_id works.
        if ref_doc_id:
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=ref_doc_id
            )
        return node

    @staticmethod
    def _parse_results(results: Any) -> List[TextNode]:
        """Convert raw query results from pyvastbase into LlamaIndex TextNode list.

        Args:
            results: Iterable of result rows.  Each row may be a ``dict``
                or an object with attribute access (both are supported).

        Returns:
            List of ``TextNode`` objects built from the result rows.
        """
        nodes: List[TextNode] = []
        for row in results:
            # Support both dict-style and object-style result rows
            if isinstance(row, dict):
                nodes.append(VastbaseVectorStore._dict_to_node(row))
            else:
                # Convert object-style row to dict for _dict_to_node
                row_dict: Dict[str, Any] = {
                    "id": getattr(row, "id", None),
                    "text": getattr(row, "text", ""),
                    "embedding": getattr(row, "embedding", None),
                    "metadata_": getattr(row, "metadata_", {}) or {},
                    "ref_doc_id": getattr(row, "ref_doc_id", None),
                }
                nodes.append(VastbaseVectorStore._dict_to_node(row_dict))
        return nodes

    # ── CRUD: Add ───────────────────────────────────────────────────────

    def add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Insert nodes into the Vastbase collection.

        Creates the collection automatically on first use.

        .. note::
            Nodes with ``ref_doc_id=None`` are stored with an empty string
            (``""``) as their ``ref_doc_id``.  Callers should avoid passing
            ``""`` to :meth:`delete` unless they intend to target those nodes.

        Args:
            nodes: Sequence of LlamaIndex BaseNode objects with embeddings.

        Returns:
            List of node IDs that were inserted.
        """
        # ADAPT: ensure collection and indexes exist before inserting.
        # _initialize() is no-op when already initialized or perform_setup=False.
        if self.perform_setup:
            self._ensure_initialized()
        else:
            # Ensure at least the collection exists (without index creation)
            self._create_collection()

        # ADAPT: use _node_to_dict helper for consistent serialisation
        data = [self._node_to_dict(node) for node in nodes]
        self.client.insert(self.table_name, data)
        return [node.node_id for node in nodes]

    # ── Async CRUD: Add ─────────────────────────────────────────────────

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Async version of :meth:`add`.

        ADAPT: wraps the synchronous ``add()`` via ``asyncio.to_thread()``.
        pyvastbase does not expose async APIs on ``VastbaseClient``, so we
        offload the blocking I/O to a thread instead of using
        ``AsyncCollection`` (which requires a separate connection setup).

        Args:
            nodes: Sequence of LlamaIndex BaseNode objects with embeddings.

        Returns:
            List of node IDs that were inserted.
        """
        import asyncio

        return await asyncio.to_thread(self.add, nodes, **kwargs)

    # ── CRUD: Delete ────────────────────────────────────────────────────

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all nodes with the given ``ref_doc_id``.

        Args:
            ref_doc_id: Source document ID whose nodes should be removed.
                Must be a non-empty string.  An empty string or ``None``
                will raise ``ValueError`` to avoid accidental mass-deletion
                of nodes whose ``ref_doc_id`` was stored as empty.

        Raises:
            ValueError: If ``ref_doc_id`` is empty or ``None``.
        """
        if not ref_doc_id:
            raise ValueError("ref_doc_id must be a non-empty string")
        # ADAPT: validate input before embedding in raw SQL expr string.
        # pyvastbase's Milvus-style API accepts raw SQL expr strings only;
        # parameterised expressions are not yet supported upstream.
        _validate_safe_id(ref_doc_id, "ref_doc_id")
        self.client.delete(
            self.table_name,
            expr=f"ref_doc_id = '{ref_doc_id}'",
        )

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async version of :meth:`delete`.

        Args:
            ref_doc_id: Source document ID whose nodes should be removed.

        Raises:
            ValueError: If ``ref_doc_id`` is empty or ``None``.
        """
        import asyncio

        return await asyncio.to_thread(self.delete, ref_doc_id, **delete_kwargs)

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by their IDs.

        Args:
            node_ids: List of node IDs to delete.  No-op if empty or None.
            filters: Optional metadata filters (not yet supported in
                delete_nodes — reserved for future implementation).

        Raises:
            NotImplementedError: If *filters* is provided (not yet supported).
        """
        if filters is not None:
            raise NotImplementedError(
                "Metadata filters in delete_nodes are not yet supported"
            )
        if not node_ids:
            return
        # ADAPT: validate each node_id before embedding in raw SQL.
        for nid in node_ids:
            _validate_safe_id(nid, "node_id")
        ids_literal = ", ".join(f"'{nid}'" for nid in node_ids)
        self.client.delete(
            self.table_name,
            expr=f"id IN ({ids_literal})",
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async version of :meth:`delete_nodes`.

        Args:
            node_ids: List of node IDs to delete.  No-op if empty or None.
            filters: Optional metadata filters (reserved for future use).
        """
        import asyncio

        return await asyncio.to_thread(
            self.delete_nodes, node_ids, filters, **delete_kwargs
        )

    # ── CRUD: Get ───────────────────────────────────────────────────────

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Retrieve nodes by their IDs.

        Args:
            node_ids: Node IDs to retrieve.  When ``None``, returns all
                nodes in the collection.  An empty list returns ``[]``.
            filters: Optional metadata filters (not yet supported in
                get_nodes — reserved for future implementation).

        Returns:
            List of BaseNode objects reconstructed from the stored data.

        Raises:
            NotImplementedError: If *filters* is provided (not yet supported).
        """
        if filters is not None:
            raise NotImplementedError(
                "Metadata filters in get_nodes are not yet supported"
            )

        # node_ids=None → retrieve all nodes
        if node_ids is None:
            results = self.client.query(
                self.table_name,
                output_fields=["id", "text", "embedding", "metadata_", "ref_doc_id"],
            )
            return self._parse_results(results)

        if not node_ids:
            return []

        # ADAPT: validate each node_id before embedding in raw SQL.
        for nid in node_ids:
            _validate_safe_id(nid, "node_id")
        ids_literal = ", ".join(f"'{nid}'" for nid in node_ids)
        results = self.client.query(
            self.table_name,
            expr=f"id IN ({ids_literal})",
            output_fields=["id", "text", "embedding", "metadata_", "ref_doc_id"],
        )
        return self._parse_results(results)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Async version of :meth:`get_nodes`.

        Args:
            node_ids: Node IDs to retrieve.  Returns empty list if None/empty.
            filters: Optional metadata filters (reserved for future use).

        Returns:
            List of BaseNode objects reconstructed from the stored data.
        """
        import asyncio

        return await asyncio.to_thread(self.get_nodes, node_ids, filters)

    # ── CRUD: Clear ─────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all nodes from the collection (truncate).

        ADAPT: VastbaseClient uses ``truncate_collection()`` which
        maps to PostgreSQL ``TRUNCATE`` — fast and non-transactional.
        """
        self.client.truncate_collection(self.table_name)

    async def aclear(self) -> None:
        """Async version of :meth:`clear`."""
        import asyncio

        return await asyncio.to_thread(self.clear)

    # ── Query (stub — reserved for next phase) ─────────────────────────

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store.

        .. note::
            Search/query functionality will be implemented in a subsequent
            issue.  Currently raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "query() will be implemented in the next phase"
        )
