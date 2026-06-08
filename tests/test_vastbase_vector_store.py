"""
Tests for VastbaseVectorStore — CRUD operations.

Covers add(), delete(), delete_nodes(), get_nodes(), clear(),
_create_collection(), and _parse_results() using mock VastbaseClient.
"""
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from llama_index.core.schema import TextNode


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


# ── _create_collection tests ────────────────────────────────────────────

class TestCreateCollection:
    """_create_collection() should create a Vastbase collection with the
    correct schema when it does not already exist."""

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

        store._create_collection()

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

        store._create_collection()

        mock_client.create_collection.assert_not_called()


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
            {"id": "n1", "text": "Hello", "metadata_": {"k": "v"}},
            {"id": "n3", "text": "World", "metadata_": {}},
        ]

        nodes = store.get_nodes(["n1", "n3"])

        assert len(nodes) == 2
        assert nodes[0].node_id == "n1"
        assert nodes[1].node_id == "n3"
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
            {"id": "n42", "text": "Only one", "metadata_": {}},
        ]

        nodes = store.get_nodes(["n42"])

        assert len(nodes) == 1
        assert nodes[0].node_id == "n42"


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
