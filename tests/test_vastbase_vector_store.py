"""Tests for VastbaseVectorStore — client, add, delete, delete_nodes, clear, get_nodes."""

import pytest
from unittest.mock import MagicMock

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)


# ---------------------------------------------------------------------------
# Unit tests (no Vastbase connection required)
# ---------------------------------------------------------------------------

def _make_mock_collection():
    """Create a mock pyvastbase Collection."""
    mock = MagicMock()
    mock.insert.return_value = MagicMock(insert_count=1, primary_keys=["node-1"])
    mock.delete.return_value = MagicMock(delete_count=1)
    mock.query.return_value = []
    mock.get.return_value = []
    mock.drop.return_value = None
    mock.create.return_value = None
    return mock


def _make_vector_store(collection_mock):
    """Create a VastbaseVectorStore with mocked collection, skipping real init."""
    from llama_index.vector_stores.vastbase import VastbaseVectorStore

    store = VastbaseVectorStore(
        uri="localhost:5432/vastbase",
        user="test",
        password="test",
        table_name="test_table",
        embed_dim=8,
    )
    store._collection = collection_mock
    store._client = MagicMock()  # Mock connection
    store._is_initialized = True
    return store


class TestVastbaseVectorStoreUnit:
    """Unit tests with mocked pyvastbase Collection."""

    @pytest.fixture
    def mock_collection(self):
        """Create a mock Collection."""
        return _make_mock_collection()

    @pytest.fixture
    def vs(self, mock_collection):
        """Create VastbaseVectorStore with mocked collection."""
        return _make_vector_store(mock_collection)

    # ---------- client ----------

    def test_client_property(self, vs):
        """client property returns the underlying connection."""
        vs._client = MagicMock()
        assert vs.client is vs._client

    def test_client_returns_none_before_init(self):
        """client returns None before initialization."""
        from llama_index.vector_stores.vastbase import VastbaseVectorStore
        store = VastbaseVectorStore(
            uri="localhost:5432/vastbase",
            user="test",
            password="test",
            table_name="test_table",
            embed_dim=8,
        )
        assert store.client is None

    # ---------- add ----------

    def test_add_returns_node_ids(self, vs, mock_collection):
        """add() should return list of node IDs."""
        nodes = [
            TextNode(id_="id1", text="hello", embedding=[0.1] * 8),
            TextNode(id_="id2", text="world", embedding=[0.2] * 8),
        ]
        result = vs.add(nodes)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == ["id1", "id2"]

    def test_add_returns_empty_list_for_empty_input(self, vs):
        """add() with empty list returns empty list."""
        result = vs.add([])
        assert result == []

    # ---------- delete ----------

    def test_delete_called_with_ref_doc_id(self, vs, mock_collection):
        """delete() should succeed when removing nodes by ref_doc_id."""
        mock_collection.query.return_value = [
            {"node_id": "id1", "metadata_": {"ref_doc_id": "doc-1"}},
        ]
        node = TextNode(
            id_="id1",
            text="hello",
            embedding=[0.1] * 8,
            metadata={"ref_doc_id": "doc-1"},
        )
        vs.add([node])
        vs.delete(ref_doc_id="doc-1")
        # Verify delete was called with expr
        mock_collection.delete.assert_called_with(expr="node_id IN ('id1')")

    def test_delete_no_match_does_nothing(self, vs, mock_collection):
        """delete() with no matching ref_doc_id is a no-op."""
        mock_collection.query.return_value = []
        vs.delete(ref_doc_id="nonexistent")
        mock_collection.delete.assert_not_called()

    # ---------- delete_nodes ----------

    def test_delete_nodes_by_ids(self, vs, mock_collection):
        """delete_nodes() by node_ids should delegate to collection.delete."""
        nodes = [
            TextNode(id_="id-a", text="a", embedding=[0.1] * 8),
            TextNode(id_="id-b", text="b", embedding=[0.2] * 8),
        ]
        vs.add(nodes)
        vs.delete_nodes(node_ids=["id-a", "id-b"])
        mock_collection.delete.assert_called_with(expr="node_id IN ('id-a', 'id-b')")

    def test_delete_nodes_by_filters(self, vs, mock_collection):
        """delete_nodes() by metadata filters should filter and delete."""
        mock_collection.query.return_value = [
            {"node_id": "id1", "metadata_": {"source": "test"}},
        ]
        node = TextNode(
            id_="id1",
            text="hello",
            embedding=[0.1] * 8,
            metadata={"source": "test"},
        )
        vs.add([node])
        filters = MetadataFilters(
            filters=[MetadataFilter(key="source", value="test", operator=FilterOperator.EQ)]
        )
        vs.delete_nodes(filters=filters)
        mock_collection.delete.assert_called_with(expr="node_id IN ('id1')")

    def test_delete_nodes_no_args_is_noop(self, vs, mock_collection):
        """delete_nodes() with no args should be a no-op."""
        vs.delete_nodes()
        mock_collection.delete.assert_not_called()

    # ---------- clear ----------

    def test_clear_removes_all_nodes(self, vs, mock_collection):
        """clear() should drop and recreate the collection."""
        nodes = [
            TextNode(id_="id1", text="hello", embedding=[0.1] * 8),
            TextNode(id_="id2", text="world", embedding=[0.2] * 8),
        ]
        vs.add(nodes)
        vs.clear()
        mock_collection.drop.assert_called_once()
        mock_collection.create.assert_called_once()

    # ---------- get_nodes ----------

    def test_get_nodes_by_ids(self, vs, mock_collection):
        """get_nodes() by node_ids should return nodes."""
        mock_collection.get.return_value = [
            {
                "node_id": "id1",
                "text": "hello world",
                "metadata_": {"source": "test", "ref_doc_id": "doc-1"},
                "embedding": [0.1] * 8,
            }
        ]
        node = TextNode(
            id_="id1",
            text="hello world",
            embedding=[0.1] * 8,
            metadata={"source": "test", "ref_doc_id": "doc-1"},
        )
        vs.add([node])
        result = vs.get_nodes(node_ids=["id1"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].node_id == "id1"

    def test_get_nodes_returns_empty_for_missing(self, vs, mock_collection):
        """get_nodes() returns empty list for non-existent ids."""
        mock_collection.get.return_value = []
        result = vs.get_nodes(node_ids=["nonexistent"])
        assert result == []

    def test_get_nodes_by_filters(self, vs, mock_collection):
        """get_nodes() by metadata filters should return matching nodes."""
        mock_collection.query.return_value = [
            {
                "node_id": "id1",
                "text": "hello",
                "metadata_": {"category": "A"},
                "embedding": [0.1] * 8,
            },
            {
                "node_id": "id2",
                "text": "world",
                "metadata_": {"category": "B"},
                "embedding": [0.2] * 8,
            },
        ]
        filters = MetadataFilters(
            filters=[MetadataFilter(key="category", value="A", operator=FilterOperator.EQ)]
        )
        result = vs.get_nodes(filters=filters)
        assert len(result) == 1
        assert result[0].node_id == "id1"

    def test_get_nodes_requires_args(self, vs):
        """get_nodes() requires node_ids or filters (matches upstream behavior)."""
        with pytest.raises(AssertionError):
            vs.get_nodes()

    def test_add_preserves_embedding(self, vs, mock_collection):
        """add() should preserve embedding vectors in the store."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        mock_collection.get.return_value = [
            {
                "node_id": "id1",
                "text": "test",
                "metadata_": {},
                "embedding": embedding,
            }
        ]
        node = TextNode(id_="id1", text="test", embedding=embedding)
        vs.add([node])
        result = vs.get_nodes(node_ids=["id1"])
        assert len(result) == 1
        assert result[0].embedding is not None

    def test_add_preserves_metadata(self, vs, mock_collection):
        """add() should preserve metadata in the store."""
        metadata = {"source": "wiki", "page": 42}
        mock_collection.get.return_value = [
            {
                "node_id": "id1",
                "text": "test",
                "metadata_": metadata,
                "embedding": [0.1] * 8,
            }
        ]
        node = TextNode(id_="id1", text="test", embedding=[0.1] * 8, metadata=metadata)
        vs.add([node])
        result = vs.get_nodes(node_ids=["id1"])
        assert len(result) == 1
        assert result[0].metadata.get("source") == "wiki"
        assert result[0].metadata.get("page") == 42

    # ---------- from_params ----------

    def test_from_params_builds_uri(self):
        """from_params() should construct a VastbaseVectorStore with correct URI."""
        from llama_index.vector_stores.vastbase import VastbaseVectorStore

        store = VastbaseVectorStore.from_params(
            host="testhost",
            port=9999,
            database="testdb",
            user="testuser",
            password="testpass",
            table_name="my_table",
            embed_dim=512,
        )
        assert store.uri == "testhost:9999/testdb"
        assert store.user == "testuser"
        assert store.password == "testpass"
        assert store.table_name == "my_table"
        assert store.embed_dim == 512


# ---------------------------------------------------------------------------
# Integration tests (require Vastbase connection)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestVastbaseVectorStoreIntegration:
    """Integration tests against a live Vastbase instance."""

    def test_initialize_creates_collection(self, vector_store):
        """_initialize() should create a collection in Vastbase."""
        vector_store._initialize()
        assert vector_store._is_initialized
        assert vector_store._client is not None
        assert vector_store._collection is not None
        # Verify collection exists by querying (not has_collection, which has a bug)
        rows = vector_store._collection.query(limit=1, output_fields=["node_id"])
        assert isinstance(rows, list)

    def test_add_and_get_nodes_roundtrip(self, vector_store):
        """add() then get_nodes() should return the same node data."""
        nodes = [
            TextNode(
                id_="node-1",
                text="Hello Vastbase",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                metadata={"source": "test", "ref_doc_id": "doc-1"},
            ),
            TextNode(
                id_="node-2",
                text="World Vastbase",
                embedding=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                metadata={"source": "test", "ref_doc_id": "doc-1"},
            ),
        ]
        ids = vector_store.add(nodes)
        assert ids == ["node-1", "node-2"]

        # Retrieve by node_ids
        result = vector_store.get_nodes(node_ids=["node-1", "node-2"])
        assert len(result) == 2
        result_ids = {n.node_id for n in result}
        assert result_ids == {"node-1", "node-2"}

        # Check text content
        texts = {n.get_content() for n in result}
        assert "Hello Vastbase" in texts
        assert "World Vastbase" in texts

    def test_delete_by_ref_doc_id(self, vector_store):
        """delete() removes all nodes with matching ref_doc_id."""
        nodes = [
            TextNode(
                id_="keep-1",
                text="Keep me",
                embedding=[0.1] * 8,
                metadata={"ref_doc_id": "doc-keep"},
            ),
            TextNode(
                id_="del-1",
                text="Delete me",
                embedding=[0.2] * 8,
                metadata={"ref_doc_id": "doc-del"},
            ),
            TextNode(
                id_="del-2",
                text="Delete me too",
                embedding=[0.3] * 8,
                metadata={"ref_doc_id": "doc-del"},
            ),
        ]
        vector_store.add(nodes)

        vector_store.delete(ref_doc_id="doc-del")

        result_del = vector_store.get_nodes(node_ids=["del-1", "del-2"])
        assert result_del == []

        result_keep = vector_store.get_nodes(node_ids=["keep-1"])
        assert len(result_keep) == 1
        assert result_keep[0].node_id == "keep-1"

    def test_delete_nodes_by_ids(self, vector_store):
        """delete_nodes() with node_ids removes specific nodes."""
        nodes = [
            TextNode(id_="a", text="A", embedding=[0.1] * 8),
            TextNode(id_="b", text="B", embedding=[0.2] * 8),
            TextNode(id_="c", text="C", embedding=[0.3] * 8),
        ]
        vector_store.add(nodes)

        vector_store.delete_nodes(node_ids=["a", "c"])

        result = vector_store.get_nodes(node_ids=["a", "b", "c"])
        assert len(result) == 1
        assert result[0].node_id == "b"

    def test_delete_nodes_by_filters(self, vector_store):
        """delete_nodes() with filters removes matching nodes."""
        nodes = [
            TextNode(
                id_="x",
                text="X",
                embedding=[0.1] * 8,
                metadata={"type": "remove"},
            ),
            TextNode(
                id_="y",
                text="Y",
                embedding=[0.2] * 8,
                metadata={"type": "keep"},
            ),
        ]
        vector_store.add(nodes)

        filters = MetadataFilters(
            filters=[MetadataFilter(key="type", value="remove", operator=FilterOperator.EQ)]
        )
        vector_store.delete_nodes(filters=filters)

        result = vector_store.get_nodes(node_ids=["x", "y"])
        assert len(result) == 1
        assert result[0].node_id == "y"

    def test_clear(self, vector_store):
        """clear() removes all nodes from the store."""
        nodes = [
            TextNode(id_="n1", text="t1", embedding=[0.1] * 8),
            TextNode(id_="n2", text="t2", embedding=[0.2] * 8),
        ]
        vector_store.add(nodes)

        vector_store.clear()

        result = vector_store.get_nodes(node_ids=["n1", "n2"])
        assert result == []

    def test_add_empty_list(self, vector_store):
        """add() with empty list returns empty list."""
        result = vector_store.add([])
        assert result == []

    def test_get_nodes_nonexistent(self, vector_store):
        """get_nodes() for non-existent IDs returns empty list."""
        result = vector_store.get_nodes(node_ids=["not-there"])
        assert result == []

    def test_client_returns_connection(self, vector_store):
        """client property returns the underlying connection."""
        vector_store._initialize()
        assert vector_store.client is not None
