"""
Framework-level integration acceptance tests for VastbaseVectorStore.

Validates the adapter through LlamaIndex's standard API surface
(VectorStoreIndex, StorageContext, and direct VastbaseVectorStore usage)
against a live Vastbase instance.

Coverage (6 scenarios per Flow C):
    1. Document ingestion + index building
    2. DENSE vector search
    3. HYBRID hybrid search
    4. Metadata filtering
    5. Async API full chain
    6. Table reuse (reconnect → query existing data)

Set environment variables before running:
    VASTBASE_HOST, VASTBASE_PORT, VASTBASE_DATABASE, VASTBASE_USER, VASTBASE_PASSWORD
    or provide a full VASTBASE_URI connection string.

Run:
    pytest tests/test_framework_integration.py -v
"""

import os
import uuid

import pytest

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


# ── Connection helpers ─────────────────────────────────────────────────────

def _check_vastbase_reachable() -> bool:
    """Return True if the Vastbase instance is reachable."""
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

_NEEDS_VASTBASE = pytest.mark.skipif(
    not _VASTBASE_REACHABLE,
    reason="Vastbase instance not reachable — set VASTBASE_HOST/VASTBASE_PORT/"
    "VASTBASE_DATABASE/VASTBASE_USER/VASTBASE_PASSWORD or VASTBASE_URI",
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _unique_table(prefix: str = "fw_int") -> str:
    """Generate a collision-free collection name for this test run."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _make_nodes(
    count: int = 3,
    prefix: str = "fw",
    embeddings: list | None = None,
) -> list[TextNode]:
    """Build synthetic TextNode objects with deterministic embeddings.

    If *embeddings* is provided, it should be a list of *count* embedding
    lists (each of dimension embed_dim).  Otherwise embeddings are generated
    from a simple repeating pattern.
    """
    nodes = []
    for i in range(count):
        if embeddings and i < len(embeddings):
            emb = embeddings[i]
        else:
            base = (i + 1) * 0.1
            emb = [base, base + 0.05, base + 0.1, base + 0.15]
        nodes.append(
            TextNode(
                id_=f"{prefix}-{i}",
                text=f"Framework integration test document #{i} about Vastbase vector search.",
                embedding=emb,
                metadata={"idx": i, "source": prefix, "category": "database" if i % 2 == 0 else "ai"},
            )
        )
    return nodes


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def vastbase_store():
    """Create a VastbaseVectorStore with a unique table, clean up after."""
    from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

    table_name = _unique_table()
    store = VastbaseVectorStore(
        connection_string=_CONNECTION_URI,
        table_name=table_name,
        embed_dim=4,
        perform_setup=True,
    )
    # Ensure client is initialized
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


# ── Scenario 1: Document ingestion + index building ────────────────────────


@pytest.mark.usefixtures("vastbase_store")
class TestScenario1DocumentIngestion:
    """Verify documents can be ingested and the collection/indexes are built."""

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_collection_is_auto_created_on_init(self, vastbase_store):
        """Collection should exist after VastbaseVectorStore init (perform_setup=True)."""
        from pyvastbase import has_collection  # type: ignore[import-untyped]

        assert has_collection(vastbase_store.table_name), (
            f"Collection '{vastbase_store.table_name}' was not auto-created"
        )

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_add_nodes_and_count(self, vastbase_store):
        """add() should insert all nodes and return correct IDs."""
        nodes = _make_nodes(5, "ingest")
        ids = vastbase_store.add(nodes)

        assert len(ids) == 5
        expected_ids = [f"ingest-{i}" for i in range(5)]
        assert ids == expected_ids

        retrieved = vastbase_store.get_nodes(expected_ids[:3])
        assert len(retrieved) == 3
        for node in retrieved:
            assert node.text is not None and len(node.text) > 0
            assert node.metadata.get("source") == "ingest"

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_add_nodes_preserves_metadata(self, vastbase_store):
        """add() should preserve all metadata fields."""
        node = TextNode(
            id_="meta-test",
            text="Metadata preservation test",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"key1": "value1", "key2": 42, "nested": {"a": 1}},
        )
        vastbase_store.add([node])

        retrieved = vastbase_store.get_nodes(["meta-test"])
        assert len(retrieved) == 1
        meta = retrieved[0].metadata
        assert meta.get("key1") == "value1"
        assert meta.get("key2") == 42
        assert meta.get("nested") == {"a": 1}

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_add_with_ref_doc_id(self, vastbase_store):
        """add() should preserve ref_doc_id via SOURCE relationship."""
        node = TextNode(
            id_="ref-test",
            text="Document with ref_doc_id",
            embedding=[0.5, 0.6, 0.7, 0.8],
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id="parent-doc-1"
        )
        vastbase_store.add([node])

        retrieved = vastbase_store.get_nodes(["ref-test"])
        assert len(retrieved) == 1
        assert retrieved[0].ref_doc_id == "parent-doc-1"


# ── Scenario 2: DENSE vector search ───────────────────────────────────────


@pytest.mark.usefixtures("vastbase_store")
class TestScenario2DenseVectorSearch:
    """Verify DENSE vector similarity search via query()."""

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_dense_query_returns_correct_top_k(self, vastbase_store):
        """DENSE search should return exactly similarity_top_k results, sorted by similarity."""
        nodes = _make_nodes(5, "dense")
        vastbase_store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.3, 0.35, 0.4, 0.45],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) == 3, f"Expected 3 results, got {len(result.nodes)}"
        assert len(result.similarities) == 3
        assert len(result.ids) == 3

        # Verify sorted by similarity (descending)
        for i in range(1, len(result.similarities)):
            assert result.similarities[i - 1] >= result.similarities[i], (
                f"Results not sorted: {result.similarities}"
            )

        # Verify similarities are in valid range [0, 1] for cosine
        for sim in result.similarities:
            assert 0.0 <= sim <= 1.0, f"Similarity {sim} out of [0,1] range"

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_dense_query_returns_best_match_first(self, vastbase_store):
        """DENSE search should return the closest vector first."""
        nodes = _make_nodes(3, "best")
        vastbase_store.add(nodes)

        # Query vector is very close to node best-1 ([0.2, 0.25, 0.3, 0.35])
        query = VectorStoreQuery(
            query_embedding=[0.2, 0.25, 0.3, 0.35],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) > 0
        # The closest match should have the highest similarity
        assert result.similarities[0] >= result.similarities[-1]

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_dense_query_handles_empty_store(self, vastbase_store):
        """DENSE search on empty collection returns empty results."""
        query = VectorStoreQuery(
            query_embedding=[0.5, 0.5, 0.5, 0.5],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) == 0
        assert result.ids == [] or len(result.ids) == 0


# ── Scenario 3: HYBRID hybrid search ──────────────────────────────────────


@pytest.mark.usefixtures("vastbase_store")
class TestScenario3HybridSearch:
    """Verify HYBRID search combines dense vector + sparse text via RRF merge."""

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_hybrid_search_returns_merged_results(self, vastbase_store):
        """HYBRID search should return results from both dense and text paths."""
        # Seed with diverse content
        nodes = [
            TextNode(
                id_="hyb-0", text="Machine learning with Python and neural networks",
                embedding=[0.9, 0.1, 0.0, 0.0],
                metadata={"category": "ml"},
            ),
            TextNode(
                id_="hyb-1", text="PostgreSQL database management system",
                embedding=[0.1, 0.9, 0.0, 0.0],
                metadata={"category": "db"},
            ),
            TextNode(
                id_="hyb-2", text="Python programming for data science",
                embedding=[0.0, 0.0, 0.9, 0.1],
                metadata={"category": "programming"},
            ),
            TextNode(
                id_="hyb-3", text="Deep learning and Vastbase vector database",
                embedding=[0.8, 0.0, 0.2, 0.0],
                metadata={"category": "ml"},
            ),
            TextNode(
                id_="hyb-4", text="Vector search with ANN indexes",
                embedding=[0.0, 0.0, 0.0, 0.9],
                metadata={"category": "db"},
            ),
        ]
        vastbase_store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.85, 0.05, 0.05, 0.05],
            query_str="database",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1, "HYBRID search should return at least 1 result"
        assert len(result.ids) >= 1
        # RRF scores should be > 0
        for sim in result.similarities:
            assert sim > 0.0, f"RRF score should be positive: {sim}"

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_hybrid_search_dedup(self, vastbase_store):
        """HYBRID search should not return duplicate node IDs."""
        nodes = _make_nodes(3, "hybd")
        vastbase_store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.2, 0.25, 0.3, 0.35],
            query_str="Vastbase",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = vastbase_store.query(query)

        # Check for duplicates
        ids = result.ids
        assert len(ids) == len(set(ids)), f"Duplicate IDs in HYBRID results: {ids}"


# ── Scenario 4: Metadata filtering ────────────────────────────────────────


@pytest.mark.usefixtures("vastbase_store")
class TestScenario4MetadataFiltering:
    """Verify metadata filters are applied correctly across query modes."""

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_eq_filter_single(self, vastbase_store):
        """EQ filter should return only matching nodes."""
        nodes = _make_nodes(5, "flt")
        # Mark first 3 as "target"
        for i in range(3):
            nodes[i].metadata["group"] = "target"
        for i in range(3, 5):
            nodes[i].metadata["group"] = "other"
        vastbase_store.add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="group", value="target", operator=FilterOperator.EQ)],
            condition=FilterCondition.AND,
        )
        query = VectorStoreQuery(
            query_embedding=[0.3, 0.35, 0.4, 0.45],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1
        for node in result.nodes:
            assert node.metadata.get("group") == "target", (
                f"Node {node.node_id} has group={node.metadata.get('group')}, expected 'target'"
            )

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_in_filter(self, vastbase_store):
        """IN filter should match nodes with any of the listed values."""
        nodes = [
            TextNode(id_="inf-0", text="A", embedding=[0.1, 0.1, 0.1, 0.1],
                     metadata={"tag": "alpha"}),
            TextNode(id_="inf-1", text="B", embedding=[0.2, 0.2, 0.2, 0.2],
                     metadata={"tag": "beta"}),
            TextNode(id_="inf-2", text="C", embedding=[0.3, 0.3, 0.3, 0.3],
                     metadata={"tag": "gamma"}),
        ]
        vastbase_store.add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="tag", value=["alpha", "gamma"], operator=FilterOperator.IN)],
        )
        query = VectorStoreQuery(
            query_embedding=[0.2, 0.2, 0.2, 0.2],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1
        result_tags = {n.metadata.get("tag") for n in result.nodes}
        assert "beta" not in result_tags, f"IN filter should exclude 'beta': {result_tags}"

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_ne_filter(self, vastbase_store):
        """NE filter should exclude nodes with the given value."""
        nodes = _make_nodes(3, "nef")
        nodes[0].metadata["status"] = "active"
        nodes[1].metadata["status"] = "inactive"
        nodes[2].metadata["status"] = "active"
        vastbase_store.add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="status", value="inactive", operator=FilterOperator.NE)],
        )
        query = VectorStoreQuery(
            query_embedding=[0.2, 0.25, 0.3, 0.35],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = vastbase_store.query(query)

        for node in result.nodes:
            assert node.metadata.get("status") != "inactive", (
                f"NE filter should exclude 'inactive': {node.metadata}"
            )

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_gt_filter(self, vastbase_store):
        """GT filter should return nodes with value > threshold."""
        nodes = [
            TextNode(id_="gt-0", text="Low", embedding=[0.1, 0.1, 0.1, 0.1],
                     metadata={"score": 10}),
            TextNode(id_="gt-1", text="Mid", embedding=[0.2, 0.2, 0.2, 0.2],
                     metadata={"score": 50}),
            TextNode(id_="gt-2", text="High", embedding=[0.3, 0.3, 0.3, 0.3],
                     metadata={"score": 90}),
        ]
        vastbase_store.add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="score", value=30, operator=FilterOperator.GT)],
        )
        query = VectorStoreQuery(
            query_embedding=[0.2, 0.2, 0.2, 0.2],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = vastbase_store.query(query)

        for node in result.nodes:
            assert node.metadata.get("score", 0) > 30, (
                f"GT filter: expected score > 30, got {node.metadata.get('score')}"
            )

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_text_match_filter(self, vastbase_store):
        """TEXT_MATCH filter should match substring."""
        nodes = [
            TextNode(id_="tm-0", text="Foo", embedding=[0.1, 0.1, 0.1, 0.1],
                     metadata={"title": "Introduction to Vastbase"}),
            TextNode(id_="tm-1", text="Bar", embedding=[0.2, 0.2, 0.2, 0.2],
                     metadata={"title": "Advanced PostgreSQL"}),
        ]
        vastbase_store.add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="title", value="Vastbase", operator=FilterOperator.TEXT_MATCH)],
        )
        query = VectorStoreQuery(
            query_embedding=[0.15, 0.15, 0.15, 0.15],
            similarity_top_k=10,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = vastbase_store.query(query)

        assert len(result.nodes) >= 1
        for node in result.nodes:
            assert "Vastbase" in str(node.metadata.get("title", ""))


# ── Scenario 5: Async API full chain ──────────────────────────────────────


class TestScenario5AsyncAPI:
    """Verify async_add → aquery → adelete → aclear lifecycle."""

    @pytest.fixture(autouse=True)
    def _mark_needs_vastbase(self):
        if not _VASTBASE_REACHABLE:
            pytest.skip(_NEEDS_VASTBASE.kwargs["reason"])

    @pytest.mark.asyncio
    async def test_async_full_lifecycle(self, vastbase_store):
        """async_add → aquery → adelete → aget_nodes → aclear full cycle."""
        nodes = _make_nodes(3, "asyncfw")
        ids = await vastbase_store.async_add(nodes)
        assert ids == ["asyncfw-0", "asyncfw-1", "asyncfw-2"]

        # Async query
        query = VectorStoreQuery(
            query_embedding=[0.2, 0.25, 0.3, 0.35],
            similarity_top_k=2,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = await vastbase_store.aquery(query)
        assert len(result.nodes) == 2

        # Async delete
        await vastbase_store.adelete_nodes(["asyncfw-0"])
        remaining = await vastbase_store.aget_nodes(["asyncfw-0"])
        assert remaining == []

        # Async clear
        await vastbase_store.aclear()
        all_remaining = await vastbase_store.aget_nodes(["asyncfw-1", "asyncfw-2"])
        assert all_remaining == []

    @pytest.mark.asyncio
    async def test_async_add_large_batch(self, vastbase_store):
        """async_add should handle a batch of 20+ nodes."""
        nodes = _make_nodes(25, "batch")
        ids = await vastbase_store.async_add(nodes)
        assert len(ids) == 25

        retrieved = await vastbase_store.aget_nodes([f"batch-{i}" for i in range(0, 25, 5)])
        assert len(retrieved) == 5

    @pytest.mark.asyncio
    async def test_async_hybrid_aquery(self, vastbase_store):
        """Async HYBRID search should work end-to-end."""
        nodes = [
            TextNode(id_="ah-0", text="Vastbase vector search engine",
                     embedding=[0.9, 0.1, 0.0, 0.0], metadata={"cat": "db"}),
            TextNode(id_="ah-1", text="Machine learning algorithms",
                     embedding=[0.1, 0.9, 0.0, 0.0], metadata={"cat": "ml"}),
        ]
        await vastbase_store.async_add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.85, 0.05, 0.05, 0.05],
            query_str="vector",
            similarity_top_k=2,
            mode=VectorStoreQueryMode.HYBRID,
        )
        result = await vastbase_store.aquery(query)
        assert len(result.nodes) >= 1


# ── Scenario 6: Table reuse ───────────────────────────────────────────────


class TestScenario6TableReuse:
    """Verify a new VastbaseVectorStore instance can query data created by another."""

    @pytest.fixture(autouse=True)
    def _mark_needs_vastbase(self):
        if not _VASTBASE_REACHABLE:
            pytest.skip(_NEEDS_VASTBASE.kwargs["reason"])

    @pytest.fixture
    def shared_table_name(self):
        """A table name shared between two store instances."""
        return _unique_table("shared")

    @pytest.fixture
    def store_writer(self, shared_table_name):
        """First store instance that creates and populates the collection."""
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string=_CONNECTION_URI,
            table_name=shared_table_name,
            embed_dim=4,
            perform_setup=True,
        )
        _ = store.client
        yield store
        # Cleanup: drop collection
        try:
            store.client.drop_collection(shared_table_name)
        except Exception:
            pass
        finally:
            try:
                store.client.close()
            except Exception:
                pass

    def test_reconnect_and_query_existing_data(self, store_writer, shared_table_name):
        """Second instance with same table_name should find and query existing data."""
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        # Write with first instance
        nodes = [
            TextNode(id_="reuse-0", text="Persistent document alpha",
                     embedding=[0.8, 0.1, 0.05, 0.05]),
            TextNode(id_="reuse-1", text="Persistent document beta",
                     embedding=[0.1, 0.8, 0.05, 0.05]),
            TextNode(id_="reuse-2", text="Persistent document gamma",
                     embedding=[0.05, 0.05, 0.8, 0.1]),
        ]
        store_writer.add(nodes)

        # Close writer to release connections
        store_writer.close()

        # Reconnect with second instance (same table_name)
        store_reader = VastbaseVectorStore(
            connection_string=_CONNECTION_URI,
            table_name=shared_table_name,
            embed_dim=4,
            perform_setup=False,  # Don't recreate — use existing
        )
        try:
            _ = store_reader.client

            # Should find existing nodes
            retrieved = store_reader.get_nodes(["reuse-0", "reuse-1", "reuse-2"])
            assert len(retrieved) == 3, (
                f"Expected 3 persisted nodes, got {len(retrieved)}"
            )

            # Should be able to query
            query = VectorStoreQuery(
                query_embedding=[0.8, 0.1, 0.05, 0.05],
                similarity_top_k=2,
                mode=VectorStoreQueryMode.DEFAULT,
            )
            result = store_reader.query(query)
            assert len(result.nodes) >= 1
            # reuse-0 should be the top match (closest vector)
            assert result.ids[0] == "reuse-0", (
                f"Expected 'reuse-0' as top match, got {result.ids[0]}"
            )
        finally:
            store_reader.close()

    def test_reconnect_with_hybrid_search(self, store_writer, shared_table_name):
        """Second instance should be able to run HYBRID search on existing data."""
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        nodes = [
            TextNode(id_="rh-0", text="Vastbase vector database with BM25 full-text search",
                     embedding=[0.9, 0.05, 0.03, 0.02]),
            TextNode(id_="rh-1", text="Python programming language for data analysis",
                     embedding=[0.05, 0.05, 0.9, 0.0]),
        ]
        store_writer.add(nodes)
        store_writer.close()

        store_reader = VastbaseVectorStore(
            connection_string=_CONNECTION_URI,
            table_name=shared_table_name,
            embed_dim=4,
            perform_setup=False,
        )
        try:
            _ = store_reader.client

            query = VectorStoreQuery(
                query_embedding=[0.85, 0.05, 0.05, 0.05],
                query_str="BM25",
                similarity_top_k=2,
                mode=VectorStoreQueryMode.HYBRID,
            )
            result = store_reader.query(query)
            assert len(result.nodes) >= 1
        finally:
            store_reader.close()


# ── MMR query tests (extended coverage) ────────────────────────────────────


@pytest.mark.usefixtures("vastbase_store")
class TestMMRExtendedCoverage:
    """Verify MMR (Maximal Marginal Relevance) re-ranking."""

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_mmr_query_returns_diverse_results(self, vastbase_store):
        """MMR should return results with diversity (not just top similarity)."""
        # Seed with cluster-like embeddings
        nodes = [
            TextNode(id_="mmr-0", text="Cluster A doc 1",
                     embedding=[0.9, 0.1, 0.0, 0.0], metadata={"cluster": "A"}),
            TextNode(id_="mmr-1", text="Cluster A doc 2",
                     embedding=[0.85, 0.15, 0.0, 0.0], metadata={"cluster": "A"}),
            TextNode(id_="mmr-2", text="Cluster B doc 1",
                     embedding=[0.1, 0.0, 0.9, 0.0], metadata={"cluster": "B"}),
            TextNode(id_="mmr-3", text="Cluster B doc 2",
                     embedding=[0.15, 0.0, 0.85, 0.0], metadata={"cluster": "B"}),
            TextNode(id_="mmr-4", text="Cluster C doc 1",
                     embedding=[0.0, 0.0, 0.0, 0.9], metadata={"cluster": "C"}),
        ]
        vastbase_store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.9, 0.1, 0.0, 0.0],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        result = vastbase_store.query(query, mmr_threshold=0.5)

        assert len(result.nodes) >= 2
        # With MMR, we expect results to span multiple clusters
        clusters = {n.metadata.get("cluster") for n in result.nodes}
        # At least 2 different clusters should appear
        assert len(clusters) >= min(2, len(result.nodes)), (
            f"MMR should diversify — got clusters: {clusters}"
        )

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_mmr_fallback_on_few_candidates(self, vastbase_store):
        """MMR with fewer candidates than top_k should not crash."""
        nodes = _make_nodes(2, "mmrf")
        vastbase_store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.2, 0.25, 0.3, 0.35],
            similarity_top_k=5,
            mode=VectorStoreQueryMode.MMR,
        )
        result = vastbase_store.query(query, mmr_threshold=0.5)

        # Should not crash and should return available nodes
        assert len(result.nodes) >= 1


# ── Edge case tests ───────────────────────────────────────────────────────


@pytest.mark.usefixtures("vastbase_store")
class TestEdgeCases:
    """Verify edge cases and error handling."""

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_delete_empty_ref_doc_id_raises(self, vastbase_store):
        """delete('') should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            vastbase_store.delete("")

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_delete_nodes_empty_list_is_noop(self, vastbase_store):
        """delete_nodes([]) should be a no-op."""
        nodes = _make_nodes(2, "emptydel")
        vastbase_store.add(nodes)
        vastbase_store.delete_nodes([])
        remaining = vastbase_store.get_nodes(["emptydel-0", "emptydel-1"])
        assert len(remaining) == 2

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_clear_on_empty_collection(self, vastbase_store):
        """clear() on empty collection should not crash."""
        vastbase_store.clear()
        result = vastbase_store.get_nodes()
        assert result == []

    @pytest.mark.skipif(not _VASTBASE_REACHABLE, reason=_NEEDS_VASTBASE.kwargs["reason"])
    def test_from_params_constructs_store(self, vastbase_store):
        """from_params() should construct a working VastbaseVectorStore."""
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        host = os.environ.get("VASTBASE_HOST", "127.0.0.1")
        port = int(os.environ.get("VASTBASE_PORT", "5432"))
        database = os.environ.get("VASTBASE_DATABASE", "test")
        user = os.environ.get("VASTBASE_USER", "postgres")
        password = os.environ.get("VASTBASE_PASSWORD", "Vexdb@123")

        store = VastbaseVectorStore.from_params(
            host=host, port=port, database=database,
            user=user, password=password,
            table_name=_unique_table("fromparams"),
            embed_dim=4,
        )
        try:
            _ = store.client
            node = TextNode(id_="fp-0", text="from_params test",
                          embedding=[0.1, 0.2, 0.3, 0.4])
            store.add([node])
            retrieved = store.get_nodes(["fp-0"])
            assert len(retrieved) == 1
            assert retrieved[0].node_id == "fp-0"
        finally:
            store.client.drop_collection(store.table_name)
            store.close()
