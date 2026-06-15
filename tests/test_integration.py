"""
Integration tests for VastbaseVectorStore — real Vastbase CRUD + search.

ADAPT: Connects to a live Vastbase V3 instance for end-to-end validation.
Set environment variables (VASTBASE_HOST, VASTBASE_PORT, VASTBASE_DATABASE,
VASTBASE_USER, VASTBASE_PASSWORD) or provide a full VASTBASE_URI connection
string.  Tests are skipped gracefully when no instance is reachable.

Run:
    pytest tests/test_integration.py -v
"""

import os
import uuid

import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


# ── Connection helpers ─────────────────────────────────────────────────

def _build_connection_uri() -> str:
    """Build a Vastbase connection URI from environment variables."""
    uri = os.environ.get("VASTBASE_URI")
    if uri:
        return uri

    host = os.environ.get("VASTBASE_HOST", "127.0.0.1")
    port = os.environ.get("VASTBASE_PORT", "5432")
    database = os.environ.get("VASTBASE_DATABASE", "test")
    user = os.environ.get("VASTBASE_USER", "postgres")
    password = os.environ.get("VASTBASE_PASSWORD", "Vexdb@123")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _check_vastbase_reachable() -> bool:
    """Return True if the Vastbase instance is reachable.

    ADAPT: Uses ``pyvastbase.connect()`` (not ``VastbaseClient(uri=...)``)
    because pyvastbase 0.2.x ``VastbaseClient`` passes ``using=`` to internal
    utility functions that do not accept it.  ``connect()`` + standalone
    utilities work correctly.
    """
    host = os.environ.get("VASTBASE_HOST", "127.0.0.1")
    port = int(os.environ.get("VASTBASE_PORT", "5432"))
    database = os.environ.get("VASTBASE_DATABASE", "test")
    user = os.environ.get("VASTBASE_USER", "postgres")
    password = os.environ.get("VASTBASE_PASSWORD", "Vexdb@123")

    try:
        from pyvastbase import connect, list_collections  # type: ignore[import-untyped]

        connect(
            host=host, port=port, database=database,
            user=user, password=password,
        )
        list_collections()
        return True
    except Exception:
        return False


_VASTBASE_REACHABLE = _check_vastbase_reachable()

def _build_connection_uri() -> str:
    """Build a Vastbase connection URI from environment variables."""
    uri = os.environ.get("VASTBASE_URI")
    if uri:
        return uri

    host = os.environ.get("VASTBASE_HOST", "127.0.0.1")
    port = os.environ.get("VASTBASE_PORT", "5432")
    database = os.environ.get("VASTBASE_DATABASE", "test")
    user = os.environ.get("VASTBASE_USER", "postgres")
    password = os.environ.get("VASTBASE_PASSWORD", "Vexdb@123")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


_CONNECTION_URI = _build_connection_uri()

# Per-class skip marker — only integration tests (CRUD / Search / Async)
# need a live Vastbase; package integrity tests run regardless.
_NEEDS_VASTBASE = pytest.mark.skipif(
    not _VASTBASE_REACHABLE,
    reason="Vastbase instance not reachable — set VASTBASE_HOST/VASTBASE_PORT/"
    "VASTBASE_DATABASE/VASTBASE_USER/VASTBASE_PASSWORD or VASTBASE_URI",
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_nodes(count: int = 3, prefix: str = "node") -> list[TextNode]:
    """Build synthetic TextNode objects with deterministic embeddings.

    Uses a simple repeating pattern so dense search returns predictable
    relative rankings.
    """
    nodes = []
    for i in range(count):
        base = (i + 1) * 0.1
        nodes.append(
            TextNode(
                id_=f"{prefix}-{i}",
                text=f"Integration test text for {prefix}-{i}",
                embedding=[base, base + 0.05, base + 0.1, base + 0.15],
                metadata={"idx": i, "label": prefix},
            )
        )
    return nodes


def _unique_table() -> str:
    """Generate a collision-free collection name for this test run."""
    return f"itest_{uuid.uuid4().hex[:12]}"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def vastbase_store():
    """Create a VastbaseVectorStore with a unique table, clean up after.

    The table is dropped on teardown, regardless of test outcome.
    """
    from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

    table_name = _unique_table()
    store = VastbaseVectorStore(
        connection_string=_CONNECTION_URI,
        table_name=table_name,
        embed_dim=4,
    )
    # Lazy-init the client so we can clean up
    _ = store.client

    yield store

    # Teardown: drop the test collection
    try:
        store.client.drop_collection(table_name)
    except Exception:
        pass
    finally:
        try:
            store.client.close()
        except Exception:
            pass


# ── CRUD integration tests ────────────────────────────────────────────


@pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
@pytest.mark.usefixtures("vastbase_store")
class TestCRUDIntegration:
    """Full CRUD lifecycle against a live Vastbase instance."""

    def test_add_and_retrieve_nodes(self, vastbase_store):
        """Insert nodes and retrieve them by ID."""
        nodes = _make_nodes(3, "crud")
        ids = vastbase_store.add(nodes)

        assert ids == ["crud-0", "crud-1", "crud-2"]

        retrieved = vastbase_store.get_nodes(["crud-0", "crud-2"])
        assert len(retrieved) == 2
        assert {n.node_id for n in retrieved} == {"crud-0", "crud-2"}
        for node in retrieved:
            assert node.text.startswith("Integration test text")
            assert node.metadata["label"] == "crud"

    def test_get_nodes_returns_empty_for_unknown_ids(self, vastbase_store):
        """Querying non-existent IDs returns an empty list."""
        nodes = _make_nodes(1, "empty")
        vastbase_store.add(nodes)

        result = vastbase_store.get_nodes(["nonexistent-id"])
        assert result == []

    def test_delete_nodes_by_ids(self, vastbase_store):
        """delete_nodes() removes specific nodes by ID."""
        nodes = _make_nodes(3, "delid")
        vastbase_store.add(nodes)

        vastbase_store.delete_nodes(["delid-0", "delid-2"])
        remaining = vastbase_store.get_nodes(["delid-0", "delid-1", "delid-2"])

        assert len(remaining) == 1
        assert remaining[0].node_id == "delid-1"

    def test_delete_by_ref_doc_id(self, vastbase_store):
        """delete() removes all nodes sharing a ref_doc_id."""
        # ADAPT: Use add() with nodes that have ref_doc_id set.
        # LlamaIndex TextNode.ref_doc_id is a property backed by
        # node.relationships[NodeRelationship.SOURCE].node_id.
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

        nodes = _make_nodes(3, "refdel")
        for n in nodes:
            n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id="shared-doc"
            )

        vastbase_store.add(nodes)
        vastbase_store.delete("shared-doc")

        remaining = vastbase_store.get_nodes(["refdel-0", "refdel-1", "refdel-2"])
        assert len(remaining) == 0

    def test_clear_removes_all_nodes(self, vastbase_store):
        """clear() truncates the entire collection."""
        nodes = _make_nodes(5, "clear")
        vastbase_store.add(nodes)

        vastbase_store.clear()
        result = vastbase_store.get_nodes(["clear-0", "clear-1", "clear-2"])
        assert result == []

    def test_add_after_clear(self, vastbase_store):
        """After clearing, new nodes can be added and retrieved."""
        vastbase_store.add(_make_nodes(2, "before"))
        vastbase_store.clear()
        vastbase_store.add(_make_nodes(1, "after"))

        retrieved = vastbase_store.get_nodes(["after-0"])
        assert len(retrieved) == 1
        assert retrieved[0].node_id == "after-0"


# ── Search integration tests ──────────────────────────────────────────


class TestSearchIntegration:
    """DENSE / HYBRID / TEXT_SEARCH against a live Vastbase instance."""

    @pytest.fixture(autouse=True)
    def _mark_needs_vastbase(self):
        """Per-class skip marker — checked before any other fixture."""
        if not _VASTBASE_REACHABLE:
            pytest.skip(_NEEDS_VASTBASE.kwargs["reason"])

    @pytest.fixture(autouse=True)
    def _seed(self, vastbase_store):
        """Populate the store with 5 nodes before each search test."""
        nodes = _make_nodes(5, "search")
        vastbase_store.add(nodes)
        return nodes

    def test_dense_search_returns_top_k(self, vastbase_store):
        """DENSE vector search returns similarity_top_k results."""
        query = VectorStoreQuery(
            query_embedding=[0.5, 0.55, 0.6, 0.65],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) == 3
        assert len(result.similarities) == 3
        assert len(result.ids) == 3
        # Results should be ordered by similarity (descending)
        for i in range(1, len(result.similarities)):
            assert result.similarities[i - 1] >= result.similarities[i], (
                f"Results not sorted by similarity: {result.similarities}"
            )

    def test_dense_search_no_results_for_far_vector(self, vastbase_store):
        """DENSE search returns low-similarity results for orthogonal vectors.

        Uses a query vector pointing in the opposite direction to stored
        embeddings so cosine similarity is near -1 / distance near 2.
        """
        query = VectorStoreQuery(
            # Negative components are opposite to stored positive embeddings
            # → very low cosine similarity / high cosine distance
            query_embedding=[-10.0, -10.0, -10.0, -10.0],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vastbase_store.query(query)

        # May return some results (distance is never infinite), but
        # similarities should be very low
        if result.nodes:
            for sim in result.similarities:
                assert sim < 0.1, f"Expected low similarity for opposite-direction query: {sim}"

    def test_dense_search_with_metadata_filter(self, vastbase_store):
        """DENSE search with metadata filters returns only matching nodes."""
        # Add a uniquely-tagged node
        tagged = TextNode(
            id_="search-tagged",
            text="Tagged node content",
            embedding=[0.5, 0.55, 0.6, 0.65],
            metadata={"special": "yes"},
        )
        vastbase_store.add([tagged])

        filters = MetadataFilters(
            filters=[MetadataFilter(key="special", value="yes", operator="==")]
        )
        query = VectorStoreQuery(
            query_embedding=[0.5, 0.55, 0.6, 0.65],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1
        for node in result.nodes:
            assert node.metadata.get("special") == "yes"

    def test_hybrid_search_returns_results(self, vastbase_store):
        """HYBRID search combines dense + text recall."""
        query = VectorStoreQuery(
            query_embedding=[0.5, 0.55, 0.6, 0.65],
            query_str="Integration test",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1
        assert len(result.ids) >= 1

    def test_text_search_returns_matches(self, vastbase_store):
        """TEXT_SEARCH returns nodes whose text matches the query."""
        query = VectorStoreQuery(
            query_str="search-2",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1
        texts = [n.text for n in result.nodes]
        assert any("search-2" in t for t in texts), f"No text matched: {texts}"

    def test_text_search_no_match_returns_empty(self, vastbase_store):
        """TEXT_SEARCH returns empty for unmatched query."""
        query = VectorStoreQuery(
            query_str="zzz_nonexistent_phrase_zzz",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) == 0

    def test_default_mode_no_embedding_falls_back_to_text(self, vastbase_store):
        """DEFAULT mode with query_str but no embedding → text search fallback."""
        query = VectorStoreQuery(
            query_str="search-0",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1
        texts = [n.text for n in result.nodes]
        assert any("search-0" in t for t in texts), f"No text matched: {texts}"


# ── Async API integration tests ───────────────────────────────────────
# ADAPT: These test the async wrapper methods inherited from
# BasePydanticVectorStore against a live Vastbase instance.


class TestAsyncIntegration:
    """Async API methods against a live Vastbase instance."""

    @pytest.fixture(autouse=True)
    def _mark_needs_vastbase(self):
        """Per-class skip marker — checked before any other fixture."""
        if not _VASTBASE_REACHABLE:
            pytest.skip(_NEEDS_VASTBASE.kwargs["reason"])

    @pytest.mark.asyncio
    async def test_async_add_and_aget(self, vastbase_store):
        """async_add() + aget_nodes() end-to-end."""
        nodes = _make_nodes(3, "async")
        ids = await vastbase_store.async_add(nodes)

        assert ids == ["async-0", "async-1", "async-2"]

        retrieved = await vastbase_store.aget_nodes(["async-0", "async-1"])
        assert len(retrieved) == 2
        assert {n.node_id for n in retrieved} == {"async-0", "async-1"}

    @pytest.mark.asyncio
    async def test_async_add_and_aquery(self, vastbase_store):
        """async_add() + aquery() end-to-end (DENSE)."""
        nodes = _make_nodes(3, "aq")
        await vastbase_store.async_add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.3, 0.35, 0.4, 0.45],
            similarity_top_k=2,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = await vastbase_store.aquery(query)

        assert len(result.nodes) == 2

    @pytest.mark.asyncio
    async def test_async_add_adelete_nodes(self, vastbase_store):
        """async_add() → adelete_nodes() → aget_nodes() lifecycle."""
        nodes = _make_nodes(1, "adel")
        await vastbase_store.async_add(nodes)

        await vastbase_store.adelete_nodes(["adel-0"])

        remaining = await vastbase_store.aget_nodes(["adel-0"])
        assert remaining == []

    @pytest.mark.asyncio
    async def test_async_add_aclear(self, vastbase_store):
        """async_add() → aclear() → aget_nodes() lifecycle."""
        nodes = _make_nodes(2, "acl")
        await vastbase_store.async_add(nodes)

        await vastbase_store.aclear()
        remaining = await vastbase_store.aget_nodes(["acl-0", "acl-1"])
        assert remaining == []


# ── Import chain & package integrity ──────────────────────────────────
# These tests do NOT require a live Vastbase instance.


class TestPackageIntegrity:
    """Verify the package structure and import chain."""

    def test_import_vastbase_vector_store(self):
        """VastbaseVectorStore should be importable from the package."""
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        assert VastbaseVectorStore is not None

    def test_import_vastbase_vector_store_from_base(self):
        """VastbaseVectorStore should be importable from base module."""
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        assert VastbaseVectorStore is not None

    def test_import_utils(self):
        """Filter utilities should be importable from the package."""
        from llama_index.vector_stores.vastbase import (
            _to_vastbase_filter,
            _escape_value,
        )

        assert _to_vastbase_filter is not None
        assert _escape_value is not None

    def test_all_exports_complete(self):
        """__all__ should list all public API symbols."""
        from llama_index.vector_stores.vastbase import __all__

        assert "VastbaseVectorStore" in __all__
        assert "_to_vastbase_filter" in __all__
        assert "_escape_value" in __all__

    def test_vastbase_vector_store_is_subclass_of_base(self):
        """VastbaseVectorStore should be a BasePydanticVectorStore."""
        from llama_index.core.vector_stores.types import BasePydanticVectorStore
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        assert issubclass(VastbaseVectorStore, BasePydanticVectorStore)

    def test_async_methods_present(self):
        """All async API methods should be available on the store class."""
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        assert hasattr(VastbaseVectorStore, "async_add")
        assert hasattr(VastbaseVectorStore, "aquery")
        assert hasattr(VastbaseVectorStore, "adelete")
        assert hasattr(VastbaseVectorStore, "adelete_nodes")
        assert hasattr(VastbaseVectorStore, "aget_nodes")
        assert hasattr(VastbaseVectorStore, "aclear")
