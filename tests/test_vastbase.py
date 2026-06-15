"""
Tests for VastbaseVectorStore — Collection Initialization, Serialization,
CRUD, and Async CRUD operations.

Covers _initialize(), _ensure_initialized(), _node_to_dict(), _dict_to_node(),
use_halfvec support, text_search_config mapping, and HNSW/FULLTEXT index creation.
"""
import json
from unittest.mock import MagicMock, patch, call

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """Return a MagicMock standing in for VastbaseClient."""
    mc = MagicMock()
    mc.has_collection.return_value = False
    mc.create_collection.return_value = None
    mc.create_index.return_value = None
    mc.insert.return_value = MagicMock(insert_count=1)
    mc.delete.return_value = None
    mc.query.return_value = []
    mc.truncate_collection.return_value = None
    return mc


@pytest.fixture
def sample_node():
    """Return a single TextNode with embedding, metadata, and ref_doc_id."""
    node = TextNode(
        id_="n1",
        text="Hello Vastbase",
        embedding=[0.1, 0.2, 0.3],
        metadata={"author": "Alice", "topic": "greeting"},
    )
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id="doc-1"
    )
    return node


# ── TestInitialize tests ────────────────────────────────────────────────────


class TestInitialize:
    """_initialize() should create collection schema with HNSW index,
    optionally FULLTEXT index, and handle use_halfvec."""

    def test_initialize_creates_collection_with_default_schema(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init",
            dimension=128,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        mock_client.has_collection.assert_called_once_with("test_init")
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[0][0] == "test_init"
        # Verify schema fields are passed
        fields = call_args[1].get("fields") or call_args[0][1] if len(call_args[0]) > 1 else None
        if fields is None and "fields" in call_args[1]:
            fields = call_args[1]["fields"]
        assert fields is not None

    def test_initialize_creates_hnsw_index(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init",
            dimension=128,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        # HNSW index should be created on embedding field
        create_index_calls = mock_client.create_index.call_args_list
        embedding_index_calls = [
            c for c in create_index_calls
            if c[0][0] == "test_init" and c[1].get("field_name") == "embedding"
        ]
        assert len(embedding_index_calls) >= 1, (
            f"Expected create_index call for 'embedding' field, "
            f"got calls: {create_index_calls}"
        )

    def test_initialize_skips_when_already_exists(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init",
            dimension=128,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = True

        store._initialize()

        mock_client.create_collection.assert_not_called()
        mock_client.create_index.assert_not_called()

    def test_initialize_with_halfvec(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init_halfvec",
            dimension=128,
            use_halfvec=True,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        fields = call_args[1].get("fields") or (call_args[0][1] if len(call_args[0]) > 1 else [])
        # Find the embedding field
        embedding_field = None
        for f in fields:
            if f.get("name") == "embedding":
                embedding_field = f
                break
        assert embedding_field is not None
        # FLOAT16_VECTOR should be used when use_halfvec=True
        from pyvastbase import DataType
        assert embedding_field["dtype"] == DataType.FLOAT16_VECTOR

    def test_initialize_with_fullvec_default(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init_fullvec",
            dimension=128,
            use_halfvec=False,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        call_args = mock_client.create_collection.call_args
        fields = call_args[1].get("fields") or (call_args[0][1] if len(call_args[0]) > 1 else [])
        embedding_field = None
        for f in fields:
            if f.get("name") == "embedding":
                embedding_field = f
                break
        assert embedding_field is not None
        from pyvastbase import DataType
        assert embedding_field["dtype"] == DataType.FLOAT_VECTOR

    def test_initialize_with_hnsw_kwargs(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init_hnsw",
            dimension=128,
            hnsw_kwargs={"m": 32, "ef_construction": 128},
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        # Check that create_index was called with graph_index params
        embedding_index_call = None
        for c in mock_client.create_index.call_args_list:
            if c[1].get("field_name") == "embedding":
                embedding_index_call = c
                break
        assert embedding_index_call is not None
        index_params = embedding_index_call[1].get("index_params")
        assert index_params is not None

    def test_initialize_with_hybrid_search_creates_fulltext_index(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init_hybrid",
            dimension=128,
            hybrid_search=True,
            text_search_config="english",
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        # Check that a FULLTEXT index was created on the text field
        text_index_calls = [
            c for c in mock_client.create_index.call_args_list
            if c[1].get("field_name") == "text"
        ]
        assert len(text_index_calls) >= 1, (
            f"Expected FULLTEXT index on 'text' when hybrid_search=True, "
            f"got: {mock_client.create_index.call_args_list}"
        )

    def test_initialize_no_fulltext_when_hybrid_search_false(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_init_no_ft",
            dimension=128,
            hybrid_search=False,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        text_index_calls = [
            c for c in mock_client.create_index.call_args_list
            if c[1].get("field_name") == "text"
        ]
        assert len(text_index_calls) == 0


class TestEnsureInitialized:
    """_ensure_initialized() should call _initialize() only once."""

    def test_ensure_initialized_calls_initialize_once(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_ensure",
            dimension=128,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        # First call should trigger _initialize()
        store._ensure_initialized()
        first_create_count = mock_client.create_collection.call_count
        assert first_create_count == 1

        # Second call should be a no-op
        store._ensure_initialized()
        assert mock_client.create_collection.call_count == first_create_count

    def test_ensure_initialized_respects_flag(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_flag",
            dimension=128,
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        assert store._is_initialized is False
        store._ensure_initialized()
        assert store._is_initialized is True
        store._ensure_initialized()  # should be no-op
        mock_client.create_collection.assert_called_once()


# ── TestSerialization tests ─────────────────────────────────────────────────


class TestSerialization:
    """_node_to_dict() and _dict_to_node() should round-trip BaseNode data."""

    def test_node_to_dict_basic(self, sample_node):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        result = store._node_to_dict(sample_node)

        assert isinstance(result, dict)
        assert result["id"] == "n1"
        assert result["text"] == "Hello Vastbase"
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["metadata_"] == {"author": "Alice", "topic": "greeting"}
        assert result["ref_doc_id"] == "doc-1"

    def test_node_to_dict_without_ref_doc_id(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        node = TextNode(
            id_="n2",
            text="No ref doc",
            embedding=[0.5],
            metadata={},
        )
        result = store._node_to_dict(node)

        assert result["ref_doc_id"] == ""

    def test_node_to_dict_without_metadata(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        node = TextNode(
            id_="n3",
            text="No metadata",
            embedding=[0.0],
        )
        result = store._node_to_dict(node)

        assert result["metadata_"] == {}

    def test_dict_to_node_basic(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        data = {
            "id": "n1",
            "text": "Hello Vastbase",
            "embedding": [0.1, 0.2, 0.3],
            "metadata_": {"author": "Alice", "topic": "greeting"},
            "ref_doc_id": "doc-1",
        }
        node = store._dict_to_node(data)

        assert node.node_id == "n1"
        assert node.text == "Hello Vastbase"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.metadata == {"author": "Alice", "topic": "greeting"}
        assert node.ref_doc_id == "doc-1"

    def test_dict_to_node_without_ref_doc_id(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        data = {
            "id": "n2",
            "text": "No ref",
            "embedding": [0.5],
            "metadata_": {},
        }
        node = store._dict_to_node(data)

        assert node.node_id == "n2"
        assert node.ref_doc_id is None

    def test_dict_to_node_without_embedding(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        data = {
            "id": "n3",
            "text": "No embedding",
            "metadata_": None,
        }
        node = store._dict_to_node(data)

        assert node.node_id == "n3"
        assert node.embedding is None
        assert node.metadata == {}

    def test_round_trip_node_to_dict_to_node(self, sample_node):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        data = store._node_to_dict(sample_node)
        restored = store._dict_to_node(data)

        assert restored.node_id == sample_node.node_id
        assert restored.text == sample_node.text
        assert restored.embedding == sample_node.embedding
        assert restored.metadata == sample_node.metadata
        assert restored.ref_doc_id == sample_node.ref_doc_id


# ── TestTextSearchConfigMapping ─────────────────────────────────────────────


class TestTextSearchConfigMapping:
    """text_search_config should map PG config names to pyvastbase tokenizers."""

    def test_english_maps_to_en_tokenizer(self):
        from llama_index.vector_stores.vastbase.base import (
            VastbaseVectorStore,
            _map_text_search_config,
        )

        assert _map_text_search_config("english") == "en_tokenizer"

    def test_chinese_maps_to_cn_tokenizer(self):
        from llama_index.vector_stores.vastbase.base import (
            VastbaseVectorStore,
            _map_text_search_config,
        )

        assert _map_text_search_config("chinese") == "cn_tokenizer"

    def test_simple_maps_to_en_tokenizer(self):
        from llama_index.vector_stores.vastbase.base import (
            VastbaseVectorStore,
            _map_text_search_config,
        )

        assert _map_text_search_config("simple") == "en_tokenizer"

    def test_unknown_config_falls_back_to_en_tokenizer(self):
        from llama_index.vector_stores.vastbase.base import (
            VastbaseVectorStore,
            _map_text_search_config,
        )

        # Unknown configs should fall back to en_tokenizer
        result = _map_text_search_config("german")
        assert result == "en_tokenizer"

    def test_store_passes_text_search_config(self, mock_client):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db",
            table_name="test_tsc",
            dimension=128,
            hybrid_search=True,
            text_search_config="chinese",
        )
        store._client = mock_client
        mock_client.has_collection.return_value = False

        store._initialize()

        # Find the FULLTEXT index creation call for the text field
        text_indices = [
            c for c in mock_client.create_index.call_args_list
            if c[1].get("field_name") == "text"
        ]
        assert len(text_indices) >= 1
        index_params = text_indices[0][1].get("index_params")
        assert index_params is not None
