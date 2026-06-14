"""
VastbaseVectorStore — LlamaIndex vector store integration for Vastbase V3.

ADAPT: Vastbase V3 has a built-in vector engine compatible with
PostgreSQL/pgvector. We wrap pyvastbase (VastbaseClient) to provide a
Milvus-style interface that LlamaIndex's BasePydanticVectorStore expects.

The public constructor mirrors PGVectorStore's 19-parameter signature for
drop-in compatibility.  Internally, pyvastbase replaces SQLAlchemy:
``connect()`` + ``VastbaseClient`` / ``Collection`` / ``AsyncCollection``.
"""

import logging
import re
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

# ADAPT: re-use PGVectorStore's PGType literal for indexed_metadata_keys
# compatibility.  Vastbase/PostgreSQL supports the same type set, so the
# same type names are valid.
PGType = str  # simplified — see PGVectorStore for the full Literal


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

        ADAPT: uses ``pyvastbase.VastbaseClient`` with the connection URI.
        This replaces SQLAlchemy's ``create_engine()`` / ``create_async_engine()``
        dual-engine pattern.  pyvastbase manages connection pooling internally.
        """
        if self._is_connected:
            return

        from pyvastbase import VastbaseClient  # type: ignore[import-untyped]

        uri = self.connection_string
        if not uri:
            raise ValueError(
                "connection_string is empty — provide a PostgreSQL URI "
                "or use from_params() with host/port/database/user/password"
            )

        self._client = VastbaseClient(uri=uri)
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

    # ── CRUD: helpers ─────────────────────────────────────────────────────

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
                        "dtype": DataType.FLOAT_VECTOR,
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
                node_id: Optional[str] = row.get("id")
                text: str = row.get("text", "")
                embedding: Optional[List[float]] = row.get("embedding")
                metadata: dict = row.get("metadata_", {}) or {}
                ref_doc_id: Optional[str] = row.get("ref_doc_id")
            else:
                node_id = getattr(row, "id", None)
                text = getattr(row, "text", "")
                embedding = getattr(row, "embedding", None)
                metadata = getattr(row, "metadata_", {}) or {}
                ref_doc_id = getattr(row, "ref_doc_id", None)

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
            nodes.append(node)
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
        self._create_collection()

        data: List[dict] = []
        for node in nodes:
            data.append(
                {
                    "id": node.node_id,
                    "text": node.get_content(),
                    "embedding": node.embedding,
                    "metadata_": node.metadata or {},
                    "ref_doc_id": node.ref_doc_id or "",
                }
            )

        self.client.insert(self.table_name, data)
        return [node.node_id for node in nodes]

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
        # ADAPT: VastbaseClient.delete() uses SQL expressions directly.
        # Escape single quotes for SQL safety.
        # NOTE: This is a known limitation of pyvastbase's Milvus-style API.
        # The expr parameter only accepts raw SQL strings; callers must ensure
        # ref_doc_id values are sanitised before passing them in.
        escaped = ref_doc_id.replace("'", "''")
        self.client.delete(
            self.table_name,
            expr=f"ref_doc_id = '{escaped}'",
        )

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
        """
        if not node_ids:
            return
        # ADAPT: Escape single quotes for SQL safety — same pattern as delete().
        # NOTE: pyvastbase's Milvus-style API accepts raw SQL expr strings.
        # Single-quote escaping is the only practical defense without parameterised
        # expressions; future maintainers should avoid adding unescaped user input.
        escaped_ids = ", ".join(
            "'" + nid.replace("'", "''") + "'" for nid in node_ids
        )
        self.client.delete(
            self.table_name,
            expr=f"id IN ({escaped_ids})",
        )

    # ── CRUD: Get ───────────────────────────────────────────────────────

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Retrieve nodes by their IDs.

        Args:
            node_ids: Node IDs to retrieve.  Returns empty list if None/empty.
            filters: Optional metadata filters (not yet supported in
                get_nodes — reserved for future implementation).

        Returns:
            List of BaseNode objects reconstructed from the stored data.
        """
        if not node_ids:
            return []

        escaped_ids = ", ".join(
            "'" + nid.replace("'", "''") + "'" for nid in node_ids
        )
        results = self.client.query(
            self.table_name,
            expr=f"id IN ({escaped_ids})",
            output_fields=["id", "text", "embedding", "metadata_", "ref_doc_id"],
        )
        return self._parse_results(results)

    # ── CRUD: Clear ─────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all nodes from the collection (truncate).

        ADAPT: VastbaseClient uses ``truncate_collection()`` which
        maps to PostgreSQL ``TRUNCATE`` — fast and non-transactional.
        """
        self.client.truncate_collection(self.table_name)

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
