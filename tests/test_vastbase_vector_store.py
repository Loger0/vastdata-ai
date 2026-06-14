"""
Tests for VastbaseVectorStore — CRUD operations.

Covers add(), delete(), delete_nodes(), get_nodes(), clear(),
_create_collection(), and _parse_results() using mock VastbaseClient.
"""
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode


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


# ── Constructor tests ──────────────────────────────────────────────────

class TestConstructor:
    """Constructor should accept all 19 PGVectorStore params and set defaults."""

    def test_default_values(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore()
        assert store.connection_string == ""
        assert store.async_connection_string == ""
        assert store.table_name == "llamaindex"
        assert store.schema_name == "public"
        assert store.embed_dim == 1536
        assert store.hybrid_search is False
        assert store.text_search_config == "english"
        assert store.cache_ok is False
        assert store.perform_setup is True
        assert store.debug is False
        assert store.use_jsonb is False
        assert store.hnsw_kwargs is None
        assert store.create_engine_kwargs == {}
        assert store.initialization_fail_on_error is False
        assert store.use_halfvec is False
        assert store.indexed_metadata_keys is None
        # Private attrs
        assert store._client is None
        assert store._is_connected is False
        assert store._customize_query_fn is None

    def test_custom_values(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://user:pass@host:5432/db",
            async_connection_string="postgresql://user:pass@host:5432/db",
            table_name="my_table",
            schema_name="myschema",
            hybrid_search=True,
            text_search_config="simple",
            embed_dim=768,
            cache_ok=True,
            perform_setup=False,
            debug=True,
            use_jsonb=True,
            hnsw_kwargs={"hnsw_m": 16},
            create_engine_kwargs={"pool_size": 5},
            initialization_fail_on_error=True,
            use_halfvec=True,
            indexed_metadata_keys=[("key1", "text")],
        )
        assert store.connection_string == "postgresql://user:pass@host:5432/db"
        assert store.table_name == "my_table"
        assert store.schema_name == "myschema"
        assert store.embed_dim == 768
        assert store.hybrid_search is True
        assert store.text_search_config == "simple"
        assert store.perform_setup is False
        assert store.debug is True
        assert store.use_halfvec is True
        assert store.hnsw_kwargs == {"hnsw_m": 16}

    def test_hybrid_search_requires_text_search_config(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        with pytest.raises(ValueError, match="text search configuration"):
            VastbaseVectorStore(hybrid_search=True, text_search_config=None)

    def test_table_name_lowercased(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(table_name="MyTable")
        assert store.table_name == "mytable"

    def test_schema_name_lowercased(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(schema_name="MySchema")
        assert store.schema_name == "myschema"

    def test_backward_compat_connection_uri_alias(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        # connection_uri is a deprecated alias for connection_string
        store = VastbaseVectorStore(
            connection_uri="postgresql://localhost:5432/db"
        )
        assert store.connection_string == "postgresql://localhost:5432/db"

    def test_backward_compat_dimension_alias(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        # dimension is a deprecated alias for embed_dim
        store = VastbaseVectorStore(dimension=256)
        assert store.embed_dim == 256

    def test_connection_string_takes_precedence_over_alias(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        # connection_string should take precedence over connection_uri
        store = VastbaseVectorStore(
            connection_string="postgresql://a:1/db",
            connection_uri="postgresql://b:2/db",
        )
        assert store.connection_string == "postgresql://a:1/db"

    def test_embed_dim_takes_precedence_over_alias(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        # embed_dim should take precedence over dimension
        store = VastbaseVectorStore(embed_dim=512, dimension=256)
        assert store.embed_dim == 512

    def test_all_19_params_accepted(self):
        """Smoke test: all PGVectorStore-compatible params should be accepted."""
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://u:p@h:5432/db",
            async_connection_string="postgresql://u:p@h:5432/db",
            table_name="tbl",
            schema_name="sch",
            hybrid_search=False,
            text_search_config="english",
            embed_dim=1536,
            cache_ok=False,
            perform_setup=True,
            debug=False,
            use_jsonb=False,
            hnsw_kwargs=None,
            create_engine_kwargs={},
            initialization_fail_on_error=False,
            use_halfvec=False,
            engine=None,
            async_engine=None,
            indexed_metadata_keys=None,
            customize_query_fn=None,
        )
        assert store is not None

    def test_class_name(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        assert VastbaseVectorStore.class_name() == "VastbaseVectorStore"


# ── Connection management tests ────────────────────────────────────────

class TestConnection:
    """_connect(), close(), from_params(), _parse_connection_string()."""

    def test_connect_creates_client(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://localhost:5432/vastbase"
        )
        with patch("pyvastbase.VastbaseClient") as mock_vc:
            mock_vc.return_value = MagicMock()
            store._connect()
            mock_vc.assert_called_once_with(
                uri="postgresql://localhost:5432/vastbase"
            )
            assert store._is_connected is True
            assert store._client is not None

    def test_connect_noop_when_already_connected(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://localhost:5432/vastbase"
        )
        store._is_connected = True
        store._client = MagicMock()

        with patch("pyvastbase.VastbaseClient") as mock_vc:
            store._connect()
            mock_vc.assert_not_called()

    def test_connect_raises_on_empty_connection_string(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore()  # connection_string defaults to ""
        with pytest.raises(ValueError, match="connection_string is empty"):
            store._connect()

    def test_client_property_lazy_connects(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://localhost:5432/vastbase"
        )
        with patch("pyvastbase.VastbaseClient") as mock_vc:
            mock_client = MagicMock()
            mock_vc.return_value = mock_client
            result = store.client
            assert result is mock_client
            assert store._is_connected is True

    def test_client_returns_existing_without_reconnect(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://localhost:5432/vastbase"
        )
        mock_client = MagicMock()
        store._client = mock_client
        store._is_connected = True

        result = store.client
        assert result is mock_client

    def test_close_disposes_client(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://localhost:5432/vastbase"
        )
        mock_client = MagicMock()
        store._client = mock_client
        store._is_connected = True

        store.close()
        mock_client.close.assert_called_once()
        assert store._client is None
        assert store._is_connected is False

    def test_close_noop_when_no_client(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://localhost:5432/vastbase"
        )
        store._client = None
        store._is_connected = False

        # Should not raise
        store.close()
        assert store._client is None

    def test_close_handles_client_error(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string="postgresql://localhost:5432/vastbase"
        )
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("connection lost")
        store._client = mock_client
        store._is_connected = True

        # Should not raise — exception is caught and logged
        store.close()
        assert store._client is None
        assert store._is_connected is False

    def test_from_params_builds_connection_string(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore.from_params(
            host="db.example.com",
            port=18000,
            database="vastbase",
            user="admin",
            password="secret",
            table_name="my_nodes",
            embed_dim=768,
        )
        assert "db.example.com" in store.connection_string
        assert "18000" in store.connection_string
        assert "vastbase" in store.connection_string
        assert "admin" in store.connection_string
        assert "secret" in store.connection_string
        assert store.table_name == "my_nodes"
        assert store.embed_dim == 768

    def test_from_params_defaults(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore.from_params()
        assert "localhost" in store.connection_string
        assert "5432" in store.connection_string
        assert "vastbase" in store.connection_string
        assert store.table_name == "llamaindex"
        assert store.embed_dim == 1536

    def test_from_params_connection_string_overrides(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore.from_params(
            host="ignored-host",
            connection_string="postgresql://override:5432/mydb",
        )
        assert store.connection_string == "postgresql://override:5432/mydb"

    def test_from_params_user_only(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore.from_params(
            user="reader",
            host="myhost",
            database="mydb",
        )
        assert "reader@" in store.connection_string
        assert "myhost" in store.connection_string
        assert "mydb" in store.connection_string
        # No password — should not contain ":password"
        assert ":reader@" not in store.connection_string.replace("reader@", "")
        # Verify no extraneous ':'
        assert store.connection_string.count(":@") == 0

    def test_parse_connection_string_full(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        result = VastbaseVectorStore._parse_connection_string(
            "postgresql://user:pass@host:5432/database"
        )
        assert result["scheme"] == "postgresql"
        assert result["user"] == "user"
        assert result["password"] == "pass"
        assert result["host"] == "host"
        assert result["port"] == "5432"
        assert result["database"] == "database"

    def test_parse_connection_string_minimal(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        result = VastbaseVectorStore._parse_connection_string(
            "postgresql://localhost/vastbase"
        )
        assert result["scheme"] == "postgresql"
        assert result["user"] is None
        assert result["password"] is None
        assert result["host"] == "localhost"
        assert result["port"] is None
        assert result["database"] == "vastbase"

    def test_parse_connection_string_empty(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        result = VastbaseVectorStore._parse_connection_string("")
        assert result["host"] is None
        assert result["database"] is None

    def test_parse_connection_string_no_port(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        result = VastbaseVectorStore._parse_connection_string(
            "postgresql://user@host/db"
        )
        assert result["user"] == "user"
        assert result["host"] == "host"
        assert result["port"] is None
        assert result["database"] == "db"

    def test_parse_connection_string_malformed(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        # Should not raise on malformed input
        result = VastbaseVectorStore._parse_connection_string("not-a-uri!!!!")
        # Returns best-effort parse; host may be None or a partial match
        assert isinstance(result, dict)


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
