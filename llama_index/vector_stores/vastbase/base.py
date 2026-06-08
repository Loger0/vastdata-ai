"""
VastbaseVectorStore — LlamaIndex vector store integration for Vastbase V3.

ADAPT: Vastbase V3 has a built-in vector engine compatible with
PostgreSQL/pgvector. We wrap pyvastbase (VastbaseClient) to provide a
Milvus-style interface that LlamaIndex's BasePydanticVectorStore expects.

This module implements the CRUD operations: add, delete, delete_nodes,
get_nodes, and clear. Search/query is stubbed out for a later phase.
"""

from typing import Any, List, Optional, Sequence

from pydantic import Field, PrivateAttr

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


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

    # ADAPT: use PrivateAttr for the lazy VastbaseClient — not a Pydantic field
    _client: Any = PrivateAttr(default=None)

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
            # Support both dict-style and object-style result rows
            if isinstance(row, dict):
                node_id: Optional[str] = row.get("id")
                text: str = row.get("text", "")
                embedding: Optional[List[float]] = row.get("embedding")
                metadata: dict = row.get("metadata_", {}) or {}
            else:
                node_id = getattr(row, "id", None)
                text = getattr(row, "text", "")
                embedding = getattr(row, "embedding", None)
                metadata = getattr(row, "metadata_", {}) or {}

            node = TextNode(
                id_=node_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
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
        """
        # ADAPT: VastbaseClient.delete() uses SQL expressions directly.
        # Escape single quotes for SQL safety.
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
        # ADAPT: Escape single quotes for SQL safety — same pattern as delete()
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
            output_fields=["id", "text", "embedding", "metadata_"],
        )
        return self._parse_results(results)

    # ── CRUD: Clear ─────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all nodes from the collection (truncate).

        ADAPT: VastbaseClient uses ``truncate_collection()`` which
        maps to PostgreSQL ``TRUNCATE`` — fast and non-transactional.
        """
        self.client.truncate_collection(self.table_name)

    # ── Query (stub — reserved for next phase) ──────────────────────────

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
            "query() will be implemented in the next phase (VAS-10)"
        )
