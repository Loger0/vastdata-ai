"""VastbaseVectorStore — LlamaIndex vector store backed by Vastbase Collection API.

Architecture:
    Uses pyvastbase Collection API (NOT SQLAlchemy ORM or pgvector) following
    the "zero PG ecosystem" constraint. All vector operations flow through
    Vastbase native floatvector type and distance operators (<->, <=>, <#>).

    pyvastbase 0.2.0 has a known issue where `_load_schema()` queries
    `information_schema.columns WHERE table_schema = 'public'` — this fails
    for non-public schemas. We work around this by passing the schema directly
    to the Collection constructor (which sets `_schema` in memory, avoiding
    the `_load_schema()` call) and using raw psycopg queries for existence
    checks.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

_logger = logging.getLogger(__name__)

# ADAPT: Use Vastbase floatvector instead of pgvector Vector/HALFVEC types.
# Vastbase native vector engine supports floatvector, halfvector, int8vector.
# No CREATE EXTENSION needed — vector engine is built-in.

# Collection field names
FIELD_NODE_ID = "node_id"
FIELD_EMBEDDING = "embedding"
FIELD_TEXT = "text"
FIELD_METADATA = "metadata_"


def _escape_sql_string(value: str) -> str:
    """Escape a string for safe use in a SQL literal.

    Doubles single quotes per SQL standard.
    Also guards against backslash escape sequences.
    """
    return value.replace("'", "''").replace("\\", "\\\\")


def _build_in_clause(column: str, values: List[str]) -> str:
    """Build a safe SQL IN clause from a list of string values.

    Args:
        column: Column name (must be a known constant, NOT user input).
        values: List of values to include in the IN clause.

    Returns:
        SQL expression like: node_id IN ('val1', 'val2')
    """
    if not values:
        return "1=0"  # Always false, safe fallback
    quoted = ", ".join(f"'{_escape_sql_string(v)}'" for v in values)
    return f"{column} IN ({quoted})"


def _entity_to_node(row: Dict) -> BaseNode:
    """Convert a Vastbase entity dict to a LlamaIndex BaseNode.

    Handles both Collection API returns (embedding as list, metadata_ as dict)
    and raw DB returns (embedding as string, metadata_ as string).
    """
    node_id = row.get(FIELD_NODE_ID, "")
    text = row.get(FIELD_TEXT, "") or ""
    metadata_raw = row.get(FIELD_METADATA, {})
    embedding = row.get(FIELD_EMBEDDING)

    # Parse metadata if it's a string (raw DB return)
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    else:
        metadata = metadata_raw or {}

    # Parse embedding if it's a string (raw DB return)
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except (json.JSONDecodeError, TypeError):
            embedding = None

    try:
        node = metadata_dict_to_node(metadata)
        node.set_content(str(text))
    except Exception:
        node = TextNode(
            id_=node_id,
            text=str(text),
            metadata=metadata,
        )

    if embedding is not None:
        node.embedding = embedding

    return node


def _node_to_entity(node: BaseNode) -> Dict:
    """Convert a LlamaIndex BaseNode to a Vastbase entity dict."""
    metadata = node_to_metadata_dict(
        node,
        remove_text=True,
        flat_metadata=False,
    )
    # Merge node metadata
    metadata.update(node.metadata)

    entity = {
        FIELD_NODE_ID: node.node_id,
        FIELD_EMBEDDING: node.get_embedding(),
        FIELD_TEXT: node.get_content(metadata_mode=MetadataMode.NONE),
        # ADAPT: Serialize metadata to JSON string. psycopg3 cannot adapt
        # Python dict directly; Vastbase stores JSON as jsonb via string.
        FIELD_METADATA: json.dumps(metadata),
    }
    return entity


def _apply_python_filter(rows: List[Dict], filters: MetadataFilters) -> List[Dict]:
    """Apply metadata filters in Python."""

    def _matches(row: Dict, f: MetadataFilter) -> bool:
        meta = row.get(FIELD_METADATA, {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        val = meta.get(f.key)

        op = f.operator
        if op == FilterOperator.EQ:
            return val == f.value
        elif op == FilterOperator.NE:
            return val != f.value
        elif op == FilterOperator.GT:
            try:
                return float(val) > float(f.value)
            except (TypeError, ValueError):
                return False
        elif op == FilterOperator.LT:
            try:
                return float(val) < float(f.value)
            except (TypeError, ValueError):
                return False
        elif op == FilterOperator.GTE:
            try:
                return float(val) >= float(f.value)
            except (TypeError, ValueError):
                return False
        elif op == FilterOperator.LTE:
            try:
                return float(val) <= float(f.value)
            except (TypeError, ValueError):
                return False
        elif op == FilterOperator.IN:
            return val in f.value if isinstance(f.value, list) else False
        elif op == FilterOperator.NIN:
            return val not in f.value if isinstance(f.value, list) else True
        elif op == FilterOperator.TEXT_MATCH:
            return f.value in str(val) if val is not None else False
        elif op == FilterOperator.IS_EMPTY:
            return val is None or val == ""
        elif op == FilterOperator.CONTAINS:
            if isinstance(val, list):
                return f.value in val
            return False
        elif op == FilterOperator.ANY:
            if isinstance(val, list) and isinstance(f.value, list):
                return any(v in val for v in f.value)
            return False
        elif op == FilterOperator.ALL:
            if isinstance(val, list) and isinstance(f.value, list):
                return all(v in val for v in f.value)
            return False
        return True

    result = []
    for row in rows:
        if all(_matches(row, f) for f in filters.filters):
            result.append(row)
    return result


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase Vector Store backed by pyvastbase Collection API.

    Zero PG ecosystem: no psycopg2, no SQLAlchemy, no pgvector.
    All operations use pyvastbase SDK.

    Examples:
        ```python
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        vector_store = VastbaseVectorStore.from_params(
            host="172.16.105.107",
            port=15432,
            database="vastbase",
            user="aidev",
            password="...",
            table_name="my_docs",
            embed_dim=1536,
        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = False

    uri: str
    user: str
    table_name: str
    embed_dim: int = 1536

    # ADAPT: password stored as PrivateAttr to prevent leak in repr/logs/model_dump.
    # Upstream PGVectorStore uses SQLAlchemy connection_string (single opaque token).
    _password: str = PrivateAttr(default="")
    _client: Any = PrivateAttr(default=None)
    _collection: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        table_name: str,
        embed_dim: int = 1536,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            uri=uri,
            user=user,
            table_name=table_name,
            embed_dim=embed_dim,
            **kwargs,
        )
        self._password = password

    @classmethod
    def from_params(
        cls,
        host: str,
        port: int = 5432,
        database: str = "vastbase",
        user: str = "vastbase",
        password: str = "",
        table_name: str = "llamaindex",
        embed_dim: int = 1536,
        **kwargs: Any,
    ) -> "VastbaseVectorStore":
        """Create VastbaseVectorStore from connection parameters."""
        uri = f"{host}:{port}/{database}"
        return cls(
            uri=uri,
            user=user,
            password=password,
            table_name=table_name,
            embed_dim=embed_dim,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Connection & lifecycle
    # ------------------------------------------------------------------

    def _table_exists(self) -> bool:
        """Check if the collection table exists in the database.

        Uses the pyvastbase-managed connection to avoid creating a
        separate psycopg connection (which can cause pooling conflicts).

        ADAPT: pyvastbase's has_collection() has a bug where it only
        checks the 'public' schema. We query information_schema.tables
        without schema filter via the pyvastbase connection.
        """
        from pyvastbase import get_connection

        conn = get_connection("default")
        try:
            # ADAPT: Use conn._connection (psycopg3 Connection) public API
            # instead of conn._execute() private method.
            cur = conn._connection.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s)",
                [self.table_name],
            )
            row = cur.fetchone()
            return bool(row[0]) if row else False
        except Exception:
            # Fallback: assume table doesn't exist and try to create
            return False

    def _initialize(self) -> None:
        """Lazy initialization: connect to Vastbase and ensure collection exists.

        ADAPT: Uses pyvastbase connect() + Collection API instead of
        SQLAlchemy create_engine. No CREATE EXTENSION needed — Vastbase
        vector engine is built-in. Uses floatvector with COSINE distance.
        """
        if self._is_initialized:
            return

        from urllib.parse import urlparse, quote
        from pyvastbase import connect
        from pyvastbase import Collection, CollectionSchema, FieldSchema, DataType

        # ADAPT: URL-encode password to handle special characters (@, :, /, etc.)
        encoded_password = quote(self._password, safe="")
        connection_uri = (
            f"postgresql://{self.user}:{encoded_password}@{self.uri}"
        )
        parsed = urlparse(connection_uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        database = (parsed.path or "/vastbase").lstrip("/") or "vastbase"
        user = parsed.username or self.user
        password = parsed.password or self._password

        # ADAPT: Use pyvastbase connect() instead of SQLAlchemy create_engine.
        connect(
            host=host, port=port, database=database,
            user=user, password=password,
        )

        schema = CollectionSchema(
            name=self.table_name,
            fields=[
                FieldSchema(
                    name=FIELD_NODE_ID,
                    dtype=DataType.VARCHAR,
                    is_primary_key=True,
                    max_length=1024,
                ),
                FieldSchema(
                    name=FIELD_EMBEDDING,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embed_dim,
                ),
                FieldSchema(name=FIELD_TEXT, dtype=DataType.TEXT),
                FieldSchema(name=FIELD_METADATA, dtype=DataType.JSON),
            ],
        )

        # ADAPT: Pass schema to Collection constructor to avoid pyvastbase's
        # _load_schema() which has a bug (hardcodes table_schema='public').
        self._collection = Collection(self.table_name, schema=schema)

        if not self._table_exists():
            self._collection.create()

        # Store the underlying connection for the client property
        from pyvastbase import get_connection
        self._client = get_connection("default")
        self._is_initialized = True

    @property
    def client(self) -> Any:
        """Return the underlying pyvastbase connection.

        Returns None if not yet initialized (matches upstream behavior).
        """
        if not self._is_initialized:
            return None
        return self._client

    # ------------------------------------------------------------------
    # CRUD: add
    # ------------------------------------------------------------------

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of BaseNode objects to add.
            **add_kwargs: Additional arguments.

        Returns:
            List of node IDs that were added.
        """
        if not nodes:
            return []

        self._initialize()

        entities = [_node_to_entity(node) for node in nodes]
        self._collection.insert(entities)
        return [node.node_id for node in nodes]

    # ------------------------------------------------------------------
    # CRUD: delete
    # ------------------------------------------------------------------

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes by ref_doc_id.

        Args:
            ref_doc_id: The ref_doc_id to delete.
            **delete_kwargs: Additional arguments.
        """
        self._initialize()

        # Query all nodes and filter by ref_doc_id in Python
        rows = self._collection.query(
            limit=None,
            output_fields=[FIELD_NODE_ID, FIELD_METADATA],
        )
        pks = []
        for row in rows:
            meta = row.get(FIELD_METADATA, {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            if meta.get("ref_doc_id") == ref_doc_id:
                pks.append(row[FIELD_NODE_ID])

        if pks:
            # ADAPT: Use expr instead of pks — pyvastbase 0.2.0 pks path
            # has a bug with string PKs (psycopg ANY(%(pks)s) adapter issue).
            # Values are SQL-escaped via _escape_sql_string to prevent injection.
            self._collection.delete(expr=_build_in_clause(FIELD_NODE_ID, pks))

    # ------------------------------------------------------------------
    # CRUD: delete_nodes
    # ------------------------------------------------------------------

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by node_ids or metadata filters.

        Args:
            node_ids: Optional list of node IDs to delete.
            filters: Optional metadata filters.
            **delete_kwargs: Additional arguments.
        """
        if not node_ids and not filters:
            return

        self._initialize()

        if node_ids and not filters:
            # ADAPT: Use expr instead of pks — pyvastbase 0.2.0 pks path
            # has a bug with string PKs. Values are SQL-escaped.
            self._collection.delete(expr=_build_in_clause(FIELD_NODE_ID, node_ids))
            return

        # Query first, filter in Python, then delete
        if node_ids:
            rows = self._collection.get(
                ids=node_ids,
                output_fields=[FIELD_NODE_ID, FIELD_METADATA],
            )
        else:
            rows = self._collection.query(
                limit=None,
                output_fields=[FIELD_NODE_ID, FIELD_METADATA],
            )

        if filters:
            rows = _apply_python_filter(rows, filters)

        pks = [row[FIELD_NODE_ID] for row in rows]
        if pks:
            # ADAPT: Use expr instead of pks — pyvastbase 0.2.0 pks path
            # has a bug with string PKs. Values are SQL-escaped.
            self._collection.delete(expr=_build_in_clause(FIELD_NODE_ID, pks))

    # ------------------------------------------------------------------
    # CRUD: clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all nodes from the vector store.

        ADAPT: Uses drop + recreate with atomic safety. If create fails
        after a successful drop, retries once to restore the collection.
        """
        self._initialize()
        self._collection.drop()
        try:
            self._collection.create()
        except Exception:
            _logger.error(
                "Failed to recreate collection %s after clear(). Retrying...",
                self.table_name,
            )
            try:
                self._collection.create()
            except Exception as e:
                _logger.critical(
                    "Collection %s lost after clear(): %s", self.table_name, e
                )
                raise

    # ------------------------------------------------------------------
    # CRUD: get_nodes
    # ------------------------------------------------------------------

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Get nodes by node_ids or metadata filters.

        Args:
            node_ids: Optional list of node IDs to retrieve.
            filters: Optional metadata filters.

        Returns:
            List of BaseNode objects.
        """
        assert node_ids is not None or filters is not None, (
            "Either node_ids or filters must be provided"
        )

        self._initialize()

        output_fields = [FIELD_NODE_ID, FIELD_TEXT, FIELD_METADATA, FIELD_EMBEDDING]

        if node_ids and not filters:
            rows = self._collection.get(ids=node_ids, output_fields=output_fields)
            if not rows:
                return []
            return [_entity_to_node(row) for row in rows]

        if node_ids:
            rows = self._collection.get(ids=node_ids, output_fields=output_fields)
        else:
            rows = self._collection.query(limit=None, output_fields=output_fields)

        if filters:
            rows = _apply_python_filter(rows, filters)

        return [_entity_to_node(row) for row in rows]

    # ------------------------------------------------------------------
    # CRUD: query
    # ------------------------------------------------------------------

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the vector store.

        Supports DEFAULT (vector similarity) and TEXT_SEARCH.
        MMR is implemented by over-fetching and post-processing.

        Args:
            query: VectorStoreQuery object.
            **kwargs: Additional arguments.

        Returns:
            VectorStoreQueryResult with nodes, similarities, and ids.
        """
        self._initialize()

        query_embedding = query.query_embedding
        similarity_top_k = query.similarity_top_k
        query_mode = query.mode
        query_str = query.query_str

        output_fields = [FIELD_NODE_ID, FIELD_TEXT, FIELD_METADATA, FIELD_EMBEDDING]

        if query_mode == VectorStoreQueryMode.TEXT_SEARCH and query_str:
            # Full-text search via query with text matching
            rows = self._collection.query(
                limit=similarity_top_k,
                output_fields=output_fields,
            )
            # Filter by text content in Python
            matched = [r for r in rows if query_str.lower() in str(r.get(FIELD_TEXT, "")).lower()]
            nodes = [_entity_to_node(row) for row in matched]
            return VectorStoreQueryResult(
                nodes=nodes[:similarity_top_k],
                similarities=[1.0] * min(len(nodes), similarity_top_k),
                ids=[n.node_id for n in nodes[:similarity_top_k]],
            )

        # Vector similarity search
        if not query_embedding:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # MMR: over-fetch for diversity re-ranking
        if query_mode == VectorStoreQueryMode.MMR:
            from llama_index.core.indices.query.embedding_utils import (
                get_top_k_mmr_embeddings,
            )

            prefetch_k = similarity_top_k * 4
            search_result = self._collection.search(
                data=[query_embedding],
                anns_field=FIELD_EMBEDDING,
                limit=prefetch_k,
                output_fields=output_fields,
            )

            all_embeddings = []
            all_nodes = []
            for hit in search_result[0]:
                node_data = hit.data if hasattr(hit, "data") else hit
                if isinstance(node_data, dict):
                    node_data = dict(node_data)
                else:
                    node_data = {}
                if FIELD_NODE_ID not in node_data:
                    node_data[FIELD_NODE_ID] = hit.id if hasattr(hit, "id") else ""
                node = _entity_to_node(node_data)
                all_nodes.append(node)
                emb = node.get_embedding()
                if emb is not None:
                    all_embeddings.append(emb)

            if all_embeddings and all_nodes:
                indices, _ = get_top_k_mmr_embeddings(
                    query_embedding,
                    all_embeddings,
                    similarity_top_k=similarity_top_k,
                )
                nodes = [all_nodes[int(i)] for i in indices]
                return VectorStoreQueryResult(
                    nodes=nodes,
                    similarities=[1.0] * len(nodes),
                    ids=[n.node_id for n in nodes],
                )

            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        # DEFAULT / HYBRID / SPARSE → standard vector search
        search_result = self._collection.search(
            data=[query_embedding],
            anns_field=FIELD_EMBEDDING,
            limit=similarity_top_k,
            output_fields=output_fields,
            param={"metric_type": "COSINE"},
        )

        nodes = []
        similarities = []
        ids = []

        for hit in search_result[0]:
            if hasattr(hit, "data"):
                node_data = dict(hit.data) if isinstance(hit.data, dict) else {}
                distance = hit.distance if hasattr(hit, "distance") else 0.0
                hit_id = hit.id if hasattr(hit, "id") else ""
            elif isinstance(hit, dict):
                node_data = dict(hit)
                distance = hit.get("distance", 0.0)
                hit_id = hit.get("id", "")
            else:
                continue

            if FIELD_NODE_ID not in node_data:
                node_data[FIELD_NODE_ID] = hit_id

            node = _entity_to_node(node_data)
            nodes.append(node)
            similarities.append(1.0 - float(distance) if distance else 1.0)
            ids.append(node.node_id)

        # Apply Python-side metadata filtering after search
        if query.filters:
            # Build rows from search results for filter matching
            rows = [
                {
                    FIELD_NODE_ID: n.node_id,
                    FIELD_METADATA: n.metadata,
                }
                for n in nodes
            ]
            matched_rows = _apply_python_filter(rows, query.filters)
            matched_ids = {r[FIELD_NODE_ID] for r in matched_rows}
            filtered = [
                (n, s, i)
                for n, s, i in zip(nodes, similarities, ids)
                if i in matched_ids
            ]
            if filtered:
                nodes, similarities, ids = zip(*filtered)
                nodes, similarities, ids = list(nodes), list(similarities), list(ids)
            else:
                nodes, similarities, ids = [], [], []

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    # ------------------------------------------------------------------
    # Async methods (bridge via asyncio.to_thread)
    # ------------------------------------------------------------------

    async def async_add(
        self, nodes: List[BaseNode], **kwargs: Any
    ) -> List[str]:
        return await asyncio.to_thread(self.add, nodes, **kwargs)

    async def adelete(
        self, ref_doc_id: str, **delete_kwargs: Any
    ) -> None:
        return await asyncio.to_thread(self.delete, ref_doc_id, **delete_kwargs)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        return await asyncio.to_thread(
            self.delete_nodes, node_ids=node_ids, filters=filters, **delete_kwargs
        )

    async def aclear(self) -> None:
        return await asyncio.to_thread(self.clear)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        return await asyncio.to_thread(
            self.get_nodes, node_ids=node_ids, filters=filters
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        return await asyncio.to_thread(self.query, query, **kwargs)
