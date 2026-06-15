"""
Tests for VastbaseVectorStore — CRUD and search operations.

Covers add(), delete(), delete_nodes(), get_nodes(), clear(),
_create_collection(), _parse_results(), query() with DENSE/HYBRID modes,
and helper methods (_dense_search, _hybrid_search, _prepare_search).
"""
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """Return a MagicMock standing in for VastbaseClient."""
    mc = MagicMock()
    mc.has_collection.return_value = False
    mc.create_collection.return_value = None
    mc.insert.return_value = MagicMock(insert_count=1)
    mc.delete.return_value = None
    mc.query.return_value = []
    mc.truncate_collection.return_value = None
    return mc


@pytest.fixture
def sample_nodes():
    """Return a list of 3 TextNode objects with embeddings and metadata."""
    return [
        TextNode(
            id_="n1",
            text="Hello world",
            embedding=[0.1, 0.2, 0.3],
            metadata={"author": "Alice", "topic": "greeting"},
        ),
        TextNode(
            id_="n2",
            text="Vector databases are fast",
            embedding=[0.4, 0.5, 0.6],
            metadata={"author": "Bob", "topic": "tech"},
        ),
        TextNode(
            id_="n3",
            text="Vastbase V3 rocks",
            embedding=[0.7, 0.8, 0.9],
            metadata={"author": "Carol", "topic": "tech"},
        ),
    ]


# ── _initialize tests ───────────────────────────────────────────────────

class TestCreateCollection:
    """_initialize() should create a Vastbase collection with the
    correct schema and HNSW index when it does not already exist."""

    def test_creates_collection_when_missing(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        mock_client.has_collection.return_value = False
        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_nodes",
            dimension=128,
        )
        # Inject mock client
        store._client = mock_client

        store._initialize()

        mock_client.has_collection.assert_called_once_with("test_nodes")
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[0][0] == "test_nodes"  # first positional: collection name

    def test_skips_creation_when_exists(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        mock_client.has_collection.return_value = True
        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_nodes",
            dimension=128,
        )
        store._client = mock_client

        store._initialize()

        mock_client.create_collection.assert_not_called()
        mock_client.create_index.assert_not_called()


# ── _parse_results tests ────────────────────────────────────────────────

class TestParseResults:
    """_parse_results() should convert raw query results into TextNode list."""

    def test_parses_single_result(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        raw_results = [
            {
                "id": "n1",
                "text": "Hello",
                "embedding": [0.1, 0.2],
                "metadata_": {"k": "v"},
            },
        ]
        nodes = store._parse_results(raw_results)
        assert len(nodes) == 1
        assert nodes[0].node_id == "n1"
        assert nodes[0].text == "Hello"
        assert nodes[0].metadata == {"k": "v"}

    def test_parses_multiple_results(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        raw_results = [
            {"id": "a", "text": "Text A", "metadata_": {}},
            {"id": "b", "text": "Text B", "metadata_": {}},
            {"id": "c", "text": "Text C", "metadata_": {}},
        ]
        nodes = store._parse_results(raw_results)
        assert len(nodes) == 3
        assert [n.node_id for n in nodes] == ["a", "b", "c"]

    def test_parses_empty_results(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        nodes = store._parse_results([])
        assert nodes == []

    def test_parse_result_without_embedding(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        raw_results = [
            {"id": "x", "text": "No embedding", "metadata_": {}},
        ]
        nodes = store._parse_results(raw_results)
        assert nodes[0].embedding is None


# ── add() tests ─────────────────────────────────────────────────────────

class TestAdd:
    """add() should insert nodes into the Vastbase collection."""

    def test_add_single_node(self, mock_client, sample_nodes):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = True

        node_ids = store.add([sample_nodes[0]])

        assert node_ids == ["n1"]
        mock_client.has_collection.assert_called_once()
        mock_client.insert.assert_called_once()
        inserted_data = mock_client.insert.call_args[0][1]
        assert len(inserted_data) == 1
        assert inserted_data[0]["id"] == "n1"
        assert inserted_data[0]["text"] == "Hello world"
        assert inserted_data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert inserted_data[0]["metadata_"] == {"author": "Alice", "topic": "greeting"}

    def test_add_multiple_nodes(self, mock_client, sample_nodes):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = True

        node_ids = store.add(sample_nodes)

        assert node_ids == ["n1", "n2", "n3"]
        inserted_data = mock_client.insert.call_args[0][1]
        assert len(inserted_data) == 3

    def test_add_creates_collection_if_needed(self, mock_client, sample_nodes):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store.add([sample_nodes[0]])

        mock_client.has_collection.assert_called_once()
        mock_client.create_collection.assert_called_once()

    def test_add_node_without_metadata(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = True

        node = TextNode(id_="bare", text="No metadata", embedding=[0.0])
        store.add([node])

        inserted = mock_client.insert.call_args[0][1]
        assert inserted[0]["metadata_"] == {}

    def test_add_node_with_ref_doc_id(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = True

        node = TextNode(
            id_="n-ref",
            text="With ref_doc_id",
            embedding=[0.0],
        )
        # ADAPT: ref_doc_id is a read-only property backed by source_node;
        # set it via the SOURCE relationship.
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id="my-doc"
        )
        store.add([node])

        inserted = mock_client.insert.call_args[0][1]
        assert inserted[0]["ref_doc_id"] == "my-doc"

    def test_add_node_ref_doc_id_none_stored_as_empty(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = True

        node = TextNode(
            id_="n-no-ref",
            text="No ref_doc_id",
            embedding=[0.0],
        )
        store.add([node])

        inserted = mock_client.insert.call_args[0][1]
        assert inserted[0]["ref_doc_id"] == ""


# ── delete() tests ──────────────────────────────────────────────────────

class TestDelete:
    """delete() should remove nodes by ref_doc_id."""

    def test_delete_by_ref_doc_id(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        store.delete("doc-123")

        mock_client.delete.assert_called_once()
        call_expr = mock_client.delete.call_args[1]["expr"]
        assert "doc-123" in call_expr
        assert "ref_doc_id" in call_expr

    def test_delete_empty_ref_doc_id_raises(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        with pytest.raises(ValueError, match="ref_doc_id must be a non-empty string"):
            store.delete("")


# ── delete_nodes() tests ────────────────────────────────────────────────

class TestDeleteNodes:
    """delete_nodes() should remove nodes by their node_ids."""

    def test_delete_nodes_by_ids(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        store.delete_nodes(["n1", "n2", "n3"])

        mock_client.delete.assert_called_once()
        call_expr = mock_client.delete.call_args[1]["expr"]
        assert "id" in call_expr
        for nid in ("n1", "n2", "n3"):
            assert nid in call_expr

    def test_delete_nodes_empty_list(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        store.delete_nodes([])

        # Should be a no-op for empty list
        mock_client.delete.assert_not_called()


# ── get_nodes() tests ───────────────────────────────────────────────────

class TestGetNodes:
    """get_nodes() should retrieve nodes by their node_ids."""

    def test_get_nodes_returns_correct_nodes(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "n1", "text": "Hello", "metadata_": {"k": "v"}, "ref_doc_id": "doc-1"},
            {"id": "n3", "text": "World", "metadata_": {}, "ref_doc_id": "doc-3"},
        ]

        nodes = store.get_nodes(["n1", "n3"])

        assert len(nodes) == 2
        assert nodes[0].node_id == "n1"
        assert nodes[0].ref_doc_id == "doc-1"
        assert nodes[1].node_id == "n3"
        assert nodes[1].ref_doc_id == "doc-3"
        mock_client.query.assert_called_once()
        call_expr = mock_client.query.call_args[1]["expr"]
        assert "n1" in call_expr
        assert "n3" in call_expr

    def test_get_nodes_empty_list(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        nodes = store.get_nodes([])

        assert nodes == []
        mock_client.query.assert_not_called()

    def test_get_nodes_single_result(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "n42", "text": "Only one", "metadata_": {}, "ref_doc_id": "doc-42"},
        ]

        nodes = store.get_nodes(["n42"])

        assert len(nodes) == 1
        assert nodes[0].node_id == "n42"
        assert nodes[0].ref_doc_id == "doc-42"


# ── clear() tests ───────────────────────────────────────────────────────

class TestClear:
    """clear() should truncate the collection."""

    def test_clear_truncates_collection(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        store.clear()

        mock_client.truncate_collection.assert_called_once_with(
            store.table_name
        )


# ── client property tests ───────────────────────────────────────────────

class TestClientProperty:
    """client property should lazily create a VastbaseClient."""

    def test_client_lazy_init(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        # _client should be None initially
        assert store._client is None

        # ADAPT: VastbaseClient is imported lazily inside the client property,
        # so we patch the upstream pyvastbase module.
        with patch("pyvastbase.VastbaseClient") as mock_vc:
            mock_vc.return_value = MagicMock()
            _ = store.client
            mock_vc.assert_called_once()


# ── stores_text / is_embedding_query ────────────────────────────────────

class TestFlags:
    """VectorStore protocol flags."""

    def test_stores_text_true(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        assert store.stores_text is True

    def test_is_embedding_query_true(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        assert store.is_embedding_query is True


# ── Search helpers ────────────────────────────────────────────────────────


def _make_search_hit(node_id, text, embedding, metadata, ref_doc_id, distance=0.15):
    """Build a MagicMock that mimics a pyvastbase search hit.

    Each hit has .id (primary key), .distance (score), and .entity (dict of
    output fields).
    """
    hit = MagicMock()
    hit.id = node_id
    hit.distance = distance
    hit.entity = {
        "id": node_id,
        "text": text,
        "embedding": embedding,
        "metadata_": metadata,
        "ref_doc_id": ref_doc_id,
    }
    return hit


# ── DENSE search tests ────────────────────────────────────────────────────


class TestDenseSearch:
    """_dense_search() — pure vector search via client.search()."""

    def test_dense_search_returns_nodes(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        hit1 = _make_search_hit("n1", "Hello", [0.1, 0.2, 0.3], {"k": "v"}, "doc-1", 0.1)
        hit2 = _make_search_hit("n2", "World", [0.4, 0.5, 0.6], {}, "doc-2", 0.5)
        mock_client.search.return_value = [[hit1, hit2]]

        query = VectorStoreQuery(
            query_embedding=[0.15, 0.25, 0.35],
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store._dense_search(query)

        assert len(result.nodes) == 2
        assert result.nodes[0].node_id == "n1"
        assert result.nodes[0].text == "Hello"
        assert result.nodes[1].node_id == "n2"
        # L2 distance→similarity: 1/(1+distance)
        expected_s0 = 1.0 / (1.0 + 0.1)  # ≈0.9091
        expected_s1 = 1.0 / (1.0 + 0.5)  # ≈0.6667
        assert result.similarities == [expected_s0, expected_s1]
        assert result.ids == ["n1", "n2"]

        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["limit"] == 5
        assert "L2" in str(call_kwargs.get("metric_type", ""))

    def test_dense_search_empty_results(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.search.return_value = [[]]

        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store._dense_search(query)

        assert result.nodes == []
        assert result.similarities == []
        assert result.ids == []

    def test_dense_search_with_filters(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        hit = _make_search_hit("n1", "Filtered", [0.1], {}, "", 0.05)
        mock_client.search.return_value = [[hit]]

        filters = MetadataFilters(
            filters=[MetadataFilter(key="author", value="Alice", operator="==")]
        )
        query = VectorStoreQuery(
            query_embedding=[0.1],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
        )
        result = store._dense_search(query)

        assert len(result.nodes) == 1
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert "author" in str(call_kwargs.get("filter_expr", ""))

    def test_dense_search_cosine_metric(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            distance_metric="COSINE",
        )
        store._client = mock_client

        hit = _make_search_hit("n1", "Cosine", [0.1, 0.2], {}, "", 0.02)
        mock_client.search.return_value = [[hit]]

        query = VectorStoreQuery(
            query_embedding=[0.1, 0.2],
            similarity_top_k=1,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store._dense_search(query)

        assert len(result.nodes) == 1
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs.get("metric_type") == "COSINE"

    def test_dense_search_ip_metric(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            distance_metric="IP",
        )
        store._client = mock_client

        hit = _make_search_hit("n1", "IP", [0.5, 0.5], {"key": "val"}, "ref-1", 0.95)
        mock_client.search.return_value = [[hit]]

        query = VectorStoreQuery(
            query_embedding=[0.5, 0.5],
            similarity_top_k=1,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store._dense_search(query)

        assert len(result.nodes) == 1
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs.get("metric_type") == "IP"


# ── HYBRID search tests ───────────────────────────────────────────────────


class TestHybridSearch:
    """_hybrid_search() — dense vector + text search fusion."""

    def test_hybrid_search_basic(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        dense_hit = _make_search_hit("n1", "Dense result", [0.1], {"k": "v"}, "doc-1", 0.1)
        mock_client.search.return_value = [[dense_hit]]

        query = VectorStoreQuery(
            query_embedding=[0.1],
            query_str="Dense result",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.7,
        )
        result = store._hybrid_search(query)

        assert len(result.nodes) >= 1
        # Dense search should have been called
        mock_client.search.assert_called()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["limit"] >= 3  # oversamples for fusion

    def test_hybrid_search_with_alpha_weighting(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        dense_hit = _make_search_hit("n1", "A", [0.1], {}, "", 0.05)
        mock_client.search.return_value = [[dense_hit]]

        query = VectorStoreQuery(
            query_embedding=[0.1],
            query_str="A",
            similarity_top_k=2,
            mode=VectorStoreQueryMode.HYBRID,
            alpha=0.9,
        )
        result = store._hybrid_search(query)

        assert len(result.nodes) >= 1

    def test_hybrid_search_no_query_str_falls_back_to_dense(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        hit = _make_search_hit("n1", "Only dense", [0.1], {}, "", 0.1)
        mock_client.search.return_value = [[hit]]

        query = VectorStoreQuery(
            query_embedding=[0.1],
            similarity_top_k=3,
            mode=VectorStoreQueryMode.HYBRID,
            # No query_str → should fall back to pure dense
        )
        result = store._hybrid_search(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].node_id == "n1"


# ── TEXT_SEARCH / SPARSE tests ────────────────────────────────────────────


class TestTextSearch:
    """query() with TEXT_SEARCH / SPARSE mode — text-only via ILIKE."""

    def _make_text_hits(self):
        """Build dict hits that mimic client.query() return values."""
        return [
            {
                "id": "t1",
                "text": "First match",
                "embedding": [0.1],
                "metadata_": {"key": "val1"},
                "ref_doc_id": "doc-a",
            },
            {
                "id": "t2",
                "text": "Second match",
                "embedding": [0.2],
                "metadata_": {},
                "ref_doc_id": "doc-b",
            },
        ]

    def test_text_search_returns_dict_results(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = self._make_text_hits()

        query = VectorStoreQuery(
            query_str="First match",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
        )
        result = store.query(query)

        assert len(result.nodes) == 2
        assert result.nodes[0].node_id == "t1"
        assert result.nodes[0].text == "First match"
        assert result.nodes[1].node_id == "t2"
        # Verify client.query() was called (not client.search())
        mock_client.query.assert_called_once()
        mock_client.search.assert_not_called()

    def test_sparse_mode_behaves_like_text_search(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = self._make_text_hits()

        query = VectorStoreQuery(
            query_str="First match",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.SPARSE,
        )
        result = store.query(query)

        assert len(result.nodes) == 2
        mock_client.query.assert_called_once()

    def test_text_search_no_query_str_returns_empty(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        query = VectorStoreQuery(
            similarity_top_k=3,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            # No query_str → should return empty
        )
        result = store.query(query)

        assert result.nodes == []
        assert result.similarities == []
        assert result.ids == []
        # Neither client.search() nor client.query() should be called
        mock_client.search.assert_not_called()
        mock_client.query.assert_not_called()

    def test_text_search_with_filters(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "t1", "text": "Filtered", "embedding": None, "metadata_": {}, "ref_doc_id": ""}
        ]

        filters = MetadataFilters(
            filters=[MetadataFilter(key="author", value="Alice", operator="==")]
        )
        query = VectorStoreQuery(
            query_str="Filtered",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.TEXT_SEARCH,
            filters=filters,
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args.kwargs
        assert "author" in str(call_kwargs.get("expr", ""))
        assert "ILIKE" in str(call_kwargs.get("expr", ""))


# ── Graceful fallback tests ────────────────────────────────────────────────


class TestGracefulFallback:
    """query() with no embedding but with query_str → graceful text-search fallback."""

    def test_default_mode_no_embedding_falls_back_to_text(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "fb1", "text": "Fallback result", "embedding": None, "metadata_": {}, "ref_doc_id": ""}
        ]

        query = VectorStoreQuery(
            query_str="Fallback result",
            similarity_top_k=3,
            mode=VectorStoreQueryMode.DEFAULT,
            # No query_embedding → graceful text-search fallback
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].node_id == "fb1"
        assert result.nodes[0].text == "Fallback result"
        mock_client.query.assert_called_once()

    def test_fallback_preserves_filters(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "fb2", "text": "Filtered fallback", "embedding": None, "metadata_": {"k": "v"}, "ref_doc_id": ""}
        ]

        filters = MetadataFilters(
            filters=[MetadataFilter(key="tag", value="urgent", operator="==")]
        )
        query = VectorStoreQuery(
            query_str="fallback",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=filters,
            # No query_embedding
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        call_kwargs = mock_client.query.call_args.kwargs
        assert "tag" in str(call_kwargs.get("expr", ""))


# ── Async API delegation tests ──────────────────────────────────────────
# ADAPT: VastbaseVectorStore inherits async methods (async_add, aquery,
# adelete, adelete_nodes, aget_nodes, aclear) from LlamaIndex's
# BasePydanticVectorStore.  These methods delegate synchronously to their
# sync counterparts.  The tests below verify correct delegation so callers
# can safely use the async API.


class TestAsyncAdd:
    """async_add() delegates to add() and returns the same node IDs."""

    @pytest.mark.asyncio
    async def test_async_add_delegates_to_add(self, mock_client, sample_nodes):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = True

        node_ids = await store.async_add(sample_nodes)

        assert node_ids == ["n1", "n2", "n3"]
        mock_client.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_add_single_node(self, mock_client, sample_nodes):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = True

        node_ids = await store.async_add([sample_nodes[0]])

        assert node_ids == ["n1"]

    @pytest.mark.asyncio
    async def test_async_add_creates_collection_if_needed(self, mock_client, sample_nodes):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.has_collection.return_value = False

        await store.async_add([sample_nodes[0]])

        mock_client.create_collection.assert_called_once()


class TestAsyncQuery:
    """aquery() delegates to query() and returns matching results."""

    @pytest.mark.asyncio
    async def test_aquery_dense_returns_results(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.vector_stores.types import VectorStoreQuery

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        hit = _make_search_hit("n1", "Hello", [0.1, 0.2], {}, "", 0.1)
        mock_client.search.return_value = [[hit]]

        q = VectorStoreQuery(query_embedding=[0.1, 0.2], similarity_top_k=5)
        result = await store.aquery(q)

        assert len(result.nodes) == 1
        assert result.nodes[0].node_id == "n1"
        assert result.ids == ["n1"]
        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_aquery_no_embedding_returns_empty(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.vector_stores.types import VectorStoreQuery

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        q = VectorStoreQuery(query_embedding=None, query_str=None, similarity_top_k=5)
        result = await store.aquery(q)

        assert result.nodes == []
        assert result.similarities == []
        assert result.ids == []

    @pytest.mark.asyncio
    async def test_aquery_text_search_fallback(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.vector_stores.types import VectorStoreQuery

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "t1", "text": "Fallback result", "embedding": None, "metadata_": {}, "ref_doc_id": ""},
        ]

        q = VectorStoreQuery(query_str="Fallback result", similarity_top_k=3)
        result = await store.aquery(q)

        assert len(result.nodes) == 1
        assert result.nodes[0].text == "Fallback result"


class TestAsyncDelete:
    """adelete() delegates to delete()."""

    @pytest.mark.asyncio
    async def test_adelete_delegates_to_delete(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        await store.adelete("doc-xyz")

        mock_client.delete.assert_called_once()
        call_expr = mock_client.delete.call_args[1]["expr"]
        assert "doc-xyz" in call_expr

    @pytest.mark.asyncio
    async def test_adelete_empty_ref_doc_id_raises(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        with pytest.raises(ValueError, match="ref_doc_id must be a non-empty string"):
            await store.adelete("")


class TestAsyncDeleteNodes:
    """adelete_nodes() delegates to delete_nodes()."""

    @pytest.mark.asyncio
    async def test_adelete_nodes_delegates_to_delete_nodes(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        await store.adelete_nodes(["n1", "n2"])

        mock_client.delete.assert_called_once()
        call_expr = mock_client.delete.call_args[1]["expr"]
        assert "n1" in call_expr
        assert "n2" in call_expr

    @pytest.mark.asyncio
    async def test_adelete_nodes_empty_list_noop(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        await store.adelete_nodes([])

        mock_client.delete.assert_not_called()


class TestAsyncGetNodes:
    """aget_nodes() delegates to get_nodes()."""

    @pytest.mark.asyncio
    async def test_aget_nodes_returns_correct_nodes(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "n1", "text": "Hello", "metadata_": {}, "ref_doc_id": None},
            {"id": "n2", "text": "World", "metadata_": {}, "ref_doc_id": None},
        ]

        nodes = await store.aget_nodes(["n1", "n2"])

        assert len(nodes) == 2
        assert nodes[0].node_id == "n1"
        assert nodes[1].node_id == "n2"
        mock_client.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_nodes_empty_list(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        nodes = await store.aget_nodes([])

        assert nodes == []
        mock_client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_aget_nodes_with_ref_doc_id(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client
        mock_client.query.return_value = [
            {"id": "n42", "text": "With ref", "metadata_": {}, "ref_doc_id": "doc-42"},
        ]

        nodes = await store.aget_nodes(["n42"])

        assert len(nodes) == 1
        assert nodes[0].node_id == "n42"
        assert nodes[0].ref_doc_id == "doc-42"


class TestAsyncClear:
    """aclear() delegates to clear()."""

    @pytest.mark.asyncio
    async def test_aclear_delegates_to_clear(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        store._client = mock_client

        await store.aclear()

        mock_client.truncate_collection.assert_called_once_with(store.table_name)
