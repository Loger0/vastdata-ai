"""
VastbaseVectorStore — LlamaIndex vector store integration for Vastbase V3.

ADAPT: Vastbase V3 has a built-in vector engine compatible with
PostgreSQL/pgvector. We wrap pyvastbase (VastbaseClient) to provide a
Milvus-style interface that LlamaIndex's BasePydanticVectorStore expects.

This module implements the CRUD operations and search (DENSE / HYBRID / TEXT)
for the Vastbase vector store.
"""

from typing import Any, Dict, List, Optional, Sequence

from pydantic import Field, PrivateAttr

from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from llama_index.vector_stores.vastbase.utils import _to_vastbase_filter


# ── Text search config mapping ───────────────────────────────────────────


# ADAPT: Vastbase uses pyvastbase tokenizer names instead of PG
# text_search_config values.  Map PG config names to pyvastbase tokenizers.
_TEXT_SEARCH_CONFIG_TO_TOKENIZER: Dict[str, str] = {
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
    import logging

    logger = logging.getLogger(__name__)
    mapped = _TEXT_SEARCH_CONFIG_TO_TOKENIZER.get(config, "en_tokenizer")
    if config not in _TEXT_SEARCH_CONFIG_TO_TOKENIZER:
        logger.warning(
            "Unknown text_search_config '%s', falling back to 'en_tokenizer'",
            config,
        )
    return mapped


class VastbaseVectorStore(BasePydanticVectorStore):
    """LlamaIndex Vector Store backed by Vastbase V3 via pyvastbase.

    Uses pyvastbase's ``VastbaseClient`` (MilvusClient-compatible API) for
    collection management and data operations.  Vector operations are handled
    natively by Vastbase's built-in vector engine — no CREATE EXTENSION needed.

    Args:
        connection_uri: PostgreSQL connection string for Vastbase
            (e.g. ``"postgresql://user:pass@host:5432/database"``).
        table_name: Name of the collection/table to store nodes in.
            Defaults to ``"llamaindex_nodes"``.
        dimension: Dimensionality of the embedding vectors.
            Defaults to 1536 (OpenAI text-embedding-ada-002).
    """

    # ── VectorStore protocol flags ──────────────────────────────────────
    stores_text: bool = True
    is_embedding_query: bool = True

    # ── Configuration ───────────────────────────────────────────────────
    connection_uri: str = Field(
        description="Vastbase PostgreSQL connection URI (postgresql://...)"
    )
    table_name: str = Field(
        default="llamaindex_nodes",
        description="Collection/table name for storing nodes",
    )
    dimension: int = Field(
        default=1536,
        description="Vector embedding dimension",
    )
    distance_metric: str = Field(
        default="L2",
        description="Distance metric for vector search: L2, COSINE, or IP",
    )
    use_halfvec: bool = Field(
        default=False,
        description="Use FLOAT16_VECTOR (half precision) instead of FLOAT_VECTOR",
    )
    hnsw_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="HNSW index parameters (m, ef_construction). "
        "Defaults to m=16, ef_construction=64 when omitted.",
    )
    hybrid_search: bool = Field(
        default=False,
        description="Create a FULLTEXT (BM25) index on the text column "
        "for hybrid search support.",
    )
    text_search_config: str = Field(
        default="english",
        description="Text search config name (english→en_tokenizer, "
        "chinese→cn_tokenizer).",
    )

    # ADAPT: use PrivateAttr for the lazy VastbaseClient — not a Pydantic field
    _client: Any = PrivateAttr(default=None)

    # ADAPT: track whether _initialize() has been called
    _is_initialized: bool = PrivateAttr(default=False)

    # ── Client property (lazy init) ─────────────────────────────────────

    @property
    def client(self) -> Any:
        """Lazily create and return the pyvastbase VastbaseClient."""
        if self._client is None:
            # ADAPT: VastbaseClient accepts a PostgreSQL URI directly —
            # no separate host/port/user/password needed.
            from pyvastbase import VastbaseClient  # type: ignore[import-untyped]

            self._client = VastbaseClient(uri=self.connection_uri)
        return self._client

    # ── Internal helpers ────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Create the Vastbase collection, HNSW index, and optional FULLTEXT index.

        ADAPT: Vastbase's vector engine is built-in — no ``CREATE EXTENSION``
        or pgvector-specific setup is needed.  Indexes are created via
        ``VastbaseClient.create_index()`` with ``IndexParams``.

        The collection schema mirrors the LlamaIndex node structure:
        ``id`` (primary key), ``text``, ``embedding`` (float vector),
        ``metadata_`` (JSON), and ``ref_doc_id`` for source-document tracking.
        """
        if self.client.has_collection(self.table_name):
            return

        from pyvastbase import DataType, IndexParams  # type: ignore[import-untyped]

        # Determine vector data type based on use_halfvec flag
        vector_dtype = (
            DataType.FLOAT16_VECTOR if self.use_halfvec else DataType.FLOAT_VECTOR
        )

        # ADAPT: use pyvastbase DataType enums for schema definition.
        # Vastbase maps these to native PostgreSQL types automatically.
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
                    "dim": self.dimension,
                },
                {"name": "metadata_", "dtype": DataType.JSON},
                {
                    "name": "ref_doc_id",
                    "dtype": DataType.VARCHAR,
                    "max_length": 256,
                },
            ],
        )

        # ADAPT: Create HNSW graph index on the embedding column using
        # pyvastbase IndexParams.graph_index().  This replaces the raw SQL
        # ``CREATE INDEX ... USING hnsw`` approach of the PG reference.
        hnsw = self.hnsw_kwargs or {}
        hnsw_params = IndexParams.graph_index(
            m=hnsw.get("m", 16),
            ef_construction=hnsw.get("ef_construction", 64),
        )
        self.client.create_index(
            self.table_name,
            field_name="embedding",
            index_params=hnsw_params,
        )

        # ADAPT: Optionally create a FULLTEXT (BM25) index on the text column.
        # This replaces the PG ``GIN + tsvector`` approach with Vastbase's
        # built-in BM25 full-text search.
        if self.hybrid_search:
            tokenizer = _map_text_search_config(self.text_search_config)
            ft_params = IndexParams.fulltext_index(
                dictionary=tokenizer,
                algorithm="BM25",
            )
            self.client.create_index(
                self.table_name,
                field_name="text",
                index_params=ft_params,
            )

    def _ensure_initialized(self) -> None:
        """Call ``_initialize()`` once per instance.

        Subsequent calls are a no-op — the ``_is_initialized`` flag gates
        the initialization path.
        """
        if not self._is_initialized:
            self._initialize()
            self._is_initialized = True

    @staticmethod
    def _node_to_dict(node: BaseNode) -> Dict[str, Any]:
        """Serialize a LlamaIndex ``BaseNode`` to a dict for insert.

        Args:
            node: A LlamaIndex node with ``node_id``, ``text``, ``embedding``,
                ``metadata``, and optionally a ``ref_doc_id``.

        Returns:
            Dict with keys ``id``, ``text``, ``embedding``, ``metadata_``,
            and ``ref_doc_id`` ready for ``VastbaseClient.insert()``.
        """
        return {
            "id": node.node_id,
            "text": node.get_content(),
            "embedding": node.embedding,
            "metadata_": node.metadata or {},
            "ref_doc_id": node.ref_doc_id or "",
        }

    @staticmethod
    def _dict_to_node(data: Dict[str, Any]) -> TextNode:
        """Convert a raw dict (from pyvastbase query/result) into a ``TextNode``.

        Args:
            data: Dict with keys ``id``, ``text``, ``embedding``,
                ``metadata_``, and optionally ``ref_doc_id``.

        Returns:
            ``TextNode`` with the stored data and SOURCE relationship set.
        """
        node_id: Optional[str] = data.get("id")
        text: str = data.get("text", "")
        embedding: Optional[List[float]] = data.get("embedding")
        metadata: dict = data.get("metadata_") or {}
        ref_doc_id: Optional[str] = data.get("ref_doc_id")

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
    def _parse_results(results: Sequence[Any]) -> List[TextNode]:
        """Convert raw query results from pyvastbase into LlamaIndex TextNode list.

        Args:
            results: Iterable of result rows.  Each row may be a ``dict``
                or an object with attribute access (both are supported).

        Returns:
            List of ``TextNode`` objects built from the result rows.
        """
        nodes: List[TextNode] = []
        for row in results:
            # Support both dict-style and object-style result rows.
            # Convert object-style rows to dict so _dict_to_node handles
            # relationship reconstruction uniformly.
            if isinstance(row, dict):
                node = VastbaseVectorStore._dict_to_node(row)
            else:
                data: Dict[str, Any] = {
                    "id": getattr(row, "id", None),
                    "text": getattr(row, "text", ""),
                    "embedding": getattr(row, "embedding", None),
                    "metadata_": getattr(row, "metadata_", {}) or {},
                    "ref_doc_id": getattr(row, "ref_doc_id", None),
                }
                node = VastbaseVectorStore._dict_to_node(data)
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
        self._ensure_initialized()

        data: List[dict] = [self._node_to_dict(node) for node in nodes]
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

    # ── Query / Search ────────────────────────────────────────────────────

    def _prepare_search(self, query: VectorStoreQuery) -> str:
        """Build a SQL WHERE clause from the query's metadata filters.

        Args:
            query: A ``VectorStoreQuery`` that may carry ``MetadataFilters``.

        Returns:
            SQL WHERE clause string (without ``WHERE``), or empty string.
        """
        if query.filters is not None:
            return _to_vastbase_filter(query.filters)
        return ""

    @staticmethod
    def _parse_search_hits(
        hits: Sequence[Any],
    ) -> tuple[List[TextNode], List[float], List[str]]:
        """Convert pyvastbase search hits into parallel lists.

        Supports both ``client.search()`` results (objects with ``.id``,
        ``.distance``, ``.entity``) and ``client.query()`` results
        (plain dicts with top-level ``id``, ``text``, etc.).

        Args:
            hits: Iterable of search-hit objects or dicts.

        Returns:
            Tuple of ``(nodes, similarities, ids)`` where each is a list.
            For L2/COSINE metrics, similarities are converted from distances
            so higher = more similar.
        """
        nodes: List[TextNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for hit in hits:
            # ADAPT: client.search() returns objects with .entity;
            # client.query() returns plain dicts.  Handle both.
            if isinstance(hit, dict):
                node_id = hit.get("id")
                text = hit.get("text", "")
                embedding = hit.get("embedding")
                metadata = hit.get("metadata_", {}) or {}
                ref_doc_id = hit.get("ref_doc_id")
                # dicts from query() have no distance field; default to 0.0
                distance = 0.0
            else:
                entity = getattr(hit, "entity", None) or {}
                if isinstance(entity, dict):
                    node_id = entity.get("id", getattr(hit, "id", None))
                    text = entity.get("text", "")
                    embedding = entity.get("embedding")
                    metadata = entity.get("metadata_", {}) or {}
                    ref_doc_id = entity.get("ref_doc_id")
                else:
                    node_id = getattr(entity, "id", getattr(hit, "id", None))
                    text = getattr(entity, "text", "")
                    embedding = getattr(entity, "embedding", None)
                    metadata = getattr(entity, "metadata_", {}) or {}
                    ref_doc_id = getattr(entity, "ref_doc_id", None)
                distance = float(getattr(hit, "distance", 0.0))

            node = TextNode(
                id_=node_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
            )
            # ADAPT: reconstruct SOURCE relationship so node.ref_doc_id works.
            if ref_doc_id:
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=ref_doc_id
                )

            nodes.append(node)
            similarities.append(distance)
            ids.append(str(node_id) if node_id is not None else "")

        return nodes, similarities, ids

    def _dense_search(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Pure dense vector search using Vastbase's built-in vector engine.

        Uses ``VastbaseClient.search()`` with the configured
        ``distance_metric`` (L2, COSINE, or IP).

        ADAPT: Vastbase V3 supports L2 (``<->``), COSINE (``<=>``), and
        IP (``<#>``) distance operators natively — no pgvector extension.

        Distance→similarity conversion:
        * L2 / COSINE — smaller distance = more similar.
          ``similarity = 1.0 / (1.0 + distance)``.
        * IP — larger value = more similar (already a similarity metric).
          Kept as-is.

        Args:
            query: ``VectorStoreQuery`` with ``query_embedding`` populated.

        Returns:
            ``VectorStoreQueryResult`` with ranked nodes, similarities, and ids.

        Raises:
            ValueError: If ``query_embedding`` is missing.
        """
        if not query.query_embedding:
            raise ValueError("query_embedding is required for dense search")

        filter_expr = self._prepare_search(query)

        # ADAPT: VastbaseClient.search() accepts metric_type directly.
        # Vastbase maps these to the native vector distance operators.
        search_results = self.client.search(
            self.table_name,
            data=[query.query_embedding],
            filter_expr=filter_expr,
            limit=query.similarity_top_k,
            output_fields=["id", "text", "embedding", "metadata_", "ref_doc_id"],
            metric_type=self.distance_metric,
        )

        if not search_results or not search_results[0]:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        hits = search_results[0]
        nodes, raw_scores, ids = self._parse_search_hits(hits)

        # ADAPT: Convert distance to similarity for L2/COSINE metrics.
        # IP is already a similarity (higher = more similar), keep as-is.
        metric_upper = self.distance_metric.upper()
        if metric_upper in ("L2", "COSINE"):
            similarities = [1.0 / (1.0 + s) for s in raw_scores]
        else:
            similarities = raw_scores

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _hybrid_search(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Hybrid search combining dense vector + text (BM25) recall.

        Strategy:
        1. Dense recall — ``VastbaseClient.search()`` with oversampling.
        2. Text recall — ``VastbaseClient.query()`` with ILIKE on the
           ``text`` column when ``query_str`` is available.
        3. Score fusion — weighted reciprocal-rank fusion (RRF-ish) with
           the ``alpha`` parameter controlling dense weight (1.0 = pure
           dense, 0.0 = pure text).

        ADAPT: Text recall uses standard SQL ILIKE — Vastbase is
        PostgreSQL-compatible and supports this natively.  For BM25-grade
        relevance scoring a FULLTEXT index can be added later via
        ``VastbaseClient.create_index()``; the ILIKE path provides a
        zero-config fallback.

        Args:
            query: ``VectorStoreQuery`` with both ``query_embedding`` and
                ``query_str``.  Falls back to pure dense if ``query_str``
                is missing.

        Returns:
            ``VectorStoreQueryResult`` with fused results.
        """
        import logging

        logger = logging.getLogger(__name__)

        alpha = query.alpha if query.alpha is not None else 0.7
        filter_expr = self._prepare_search(query)
        top_k = query.similarity_top_k

        # ADAPT: Helper to extract the id from either a dict (client.query())
        # or an object (client.search()) hit — both appear in fusion loops.
        def _hit_id(hit: Any) -> str:
            if isinstance(hit, dict):
                return hit.get("id", "")
            return str(getattr(hit, "id", ""))

        # ── 1. Dense recall (oversample for fusion headroom) ──────────
        dense_limit = max(top_k * 3, 10)
        dense_raw = self.client.search(
            self.table_name,
            data=[query.query_embedding] if query.query_embedding else [[0.0] * self.dimension],
            filter_expr=filter_expr,
            limit=dense_limit,
            output_fields=["id", "text", "embedding", "metadata_", "ref_doc_id"],
            metric_type=self.distance_metric,
        )
        dense_hits = dense_raw[0] if dense_raw else []

        # ── 2. Text recall ────────────────────────────────────────────
        text_hits: List[Any] = []
        if query.query_str:
            # ADAPT: escape the query string for SQL ILIKE safety.
            escaped_str = query.query_str.replace("'", "''")
            text_expr = f"text ILIKE '%{escaped_str}%'"
            if filter_expr:
                text_expr = f"({filter_expr}) AND ({text_expr})"
            try:
                text_raw = self.client.query(
                    self.table_name,
                    expr=text_expr,
                    limit=dense_limit,
                    output_fields=["id", "text", "embedding", "metadata_", "ref_doc_id"],
                )
                text_hits = list(text_raw) if text_raw else []
            except Exception:
                # ADAPT: log and continue — text recall failure should not
                # abort the entire search; dense results are still usable.
                logger.warning(
                    "Text recall query failed for table=%s expr=%r — "
                    "falling back to pure dense results.",
                    self.table_name,
                    text_expr,
                    exc_info=True,
                )
                text_hits = []

        # ── 3. Score fusion ───────────────────────────────────────────
        dense_scores: dict[str, float] = {}
        for hit in dense_hits:
            hid = _hit_id(hit)
            dist = float(getattr(hit, "distance", 1.0))
            # Convert distance to similarity: 1/(1+distance)
            dense_scores[hid] = 1.0 / (1.0 + dist)

        text_rank: dict[str, int] = {}
        for rank, hit in enumerate(text_hits):
            hid = _hit_id(hit)
            if hid:
                text_rank[hid] = rank + 1  # 1-indexed rank

        fused: dict[str, tuple] = {}  # id → (combined_score, hit)
        for hit in dense_hits:
            hid = _hit_id(hit)
            dense_s = dense_scores.get(hid, 0.0)
            t_rank = text_rank.get(hid, len(text_hits) + 1)
            text_s = 1.0 / float(t_rank)  # reciprocal rank
            combined = alpha * dense_s + (1.0 - alpha) * text_s
            fused[hid] = (combined, hit)

        for hit in text_hits:
            hid = _hit_id(hit)
            if hid in fused:
                continue
            t_rank = text_rank.get(hid, 1)
            text_s = 1.0 / float(t_rank)
            dense_s = 0.0
            combined = alpha * dense_s + (1.0 - alpha) * text_s
            fused[hid] = (combined, hit)

        # Sort by combined score descending, take top_k
        ranked = sorted(fused.values(), key=lambda x: x[0], reverse=True)[:top_k]

        if not ranked:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        nodes, _, ids = self._parse_search_hits(
            [item[1] for item in ranked]
        )
        # Override similarities with fused scores
        fused_scores = [item[0] for item in ranked]
        return VectorStoreQueryResult(nodes=nodes, similarities=fused_scores, ids=ids)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store.

        Dispatches based on ``query.mode``:

        * ``DEFAULT`` / ``None`` → dense vector search
        * ``SPARSE`` / ``TEXT_SEARCH`` → text-only search (requires ``query_str``)
        * ``HYBRID`` → dense + text fusion

        Args:
            query: ``VectorStoreQuery`` carrying embedding, query string,
                similarity_top_k, alpha, and optional metadata filters.

        Returns:
            ``VectorStoreQueryResult`` with matching nodes and scores.
        """
        mode = query.mode

        if mode == VectorStoreQueryMode.HYBRID:
            return self._hybrid_search(query)

        if mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
            # ADAPT: text-only search uses ILIKE on the text column.
            # For BM25-grade search a FULLTEXT index should be created first.
            if not query.query_str:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            filter_expr = self._prepare_search(query)
            escaped_str = query.query_str.replace("'", "''")
            text_expr = f"text ILIKE '%{escaped_str}%'"
            if filter_expr:
                text_expr = f"({filter_expr}) AND ({text_expr})"
            raw = self.client.query(
                self.table_name,
                expr=text_expr,
                limit=query.similarity_top_k,
                output_fields=["id", "text", "embedding", "metadata_", "ref_doc_id"],
            )
            nodes, similarities, ids = self._parse_search_hits(raw)
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        # DEFAULT / fallback → dense vector search
        # If query_embedding is missing, fall back to text search when query_str
        # is available; otherwise return empty.
        if not query.query_embedding:
            if query.query_str:
                # Use text-only search as graceful fallback
                return self.query(
                    VectorStoreQuery(
                        query_str=query.query_str,
                        mode=VectorStoreQueryMode.TEXT_SEARCH,
                        similarity_top_k=query.similarity_top_k,
                        filters=query.filters,
                        alpha=query.alpha,
                    ),
                    **kwargs,
                )
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        return self._dense_search(query)
