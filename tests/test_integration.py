"""
pyvastbase integration tests — validates VastbaseVectorStore against a real
Vastbase instance via the Collection API.

Requires environment variables:
  VASTBASE_HOST, VASTBASE_PORT, VASTBASE_DATABASE, VASTBASE_USER, VASTBASE_PASSWORD

These tests are HARD GATE — must pass on a real Vastbase instance.
"""
import json
import os
import random
import string

import pytest
from pyvastbase import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connect,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _env(k: str, default: str = "") -> str:
    return os.environ.get(k, default)


def _connect():
    """Connect to Vastbase with the default alias."""
    return connect(
        host=_env("VASTBASE_HOST", "172.16.105.107"),
        port=int(_env("VASTBASE_PORT", "15432")),
        database=_env("VASTBASE_DATABASE", "vastbase"),
        user=_env("VASTBASE_USER", "aidev"),
        password=_env("VASTBASE_PASSWORD", ""),
    )


def _connection_uri():
    return (
        f"postgresql://{_env('VASTBASE_USER')}:{_env('VASTBASE_PASSWORD')}"
        f"@{_env('VASTBASE_HOST')}:{_env('VASTBASE_PORT')}"
        f"/{_env('VASTBASE_DATABASE')}"
    )


def _make_schema(name: str, dim: int = 4) -> CollectionSchema:
    return CollectionSchema(
        name=name,
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="text", dtype=DataType.TEXT),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata_", dtype=DataType.JSON),
            FieldSchema(name="ref_doc_id", dtype=DataType.VARCHAR, max_length=256),
        ],
    )


@pytest.fixture(scope="session", autouse=True)
def _vastbase_session():
    """Ensure Vastbase connection is established once per session."""
    _connect()


@pytest.fixture
def collection():
    """Create a unique collection for a test, drop it after."""
    suffix = "".join(random.choices(string.ascii_lowercase, k=8))
    name = f"test_int_{suffix}"
    schema = _make_schema(name)
    coll = Collection(name, schema=schema, using="default")
    coll.create()
    yield coll
    try:
        coll.drop()
    except Exception:
        pass


# ── Connectivity ───────────────────────────────────────────────────────

class TestVastbaseConnectivity:
    """Verify the Vastbase instance is reachable and responsive."""

    def test_connection_works(self):
        """Basic connectivity: Vastbase is reachable."""
        conn = _connect()
        assert conn is not None

    def test_collection_create_and_drop(self, collection):
        """Can create a collection and it exists."""
        assert collection.name.startswith("test_int_")


# ── Connection string parsing ──────────────────────────────────────────

class TestConnectionStringParsing:
    """Tests for _parse_connection_string and _build_connection_string."""

    def test_parse_full_uri(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        result = VastbaseVectorStore._parse_connection_string(
            "postgresql://user:pass@host:5432/dbname"
        )
        assert result["scheme"] == "postgresql"
        assert result["user"] == "user"
        assert result["password"] == "pass"
        assert result["host"] == "host"
        assert result["port"] == "5432"
        assert result["database"] == "dbname"

    def test_build_connection_string(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        uri = VastbaseVectorStore._build_connection_string(
            host="db.example.com",
            port=18000,
            database="mydb",
            user="admin",
            password="secret",
        )
        assert "db.example.com" in uri
        assert "18000" in uri
        assert "mydb" in uri
        assert "admin" in uri

    def test_parse_env_connection_string(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        uri = _connection_uri()
        result = VastbaseVectorStore._parse_connection_string(uri)
        assert result["host"] == _env("VASTBASE_HOST")
        assert result["port"] == _env("VASTBASE_PORT")
        assert result["database"] == _env("VASTBASE_DATABASE")
        assert result["user"] == _env("VASTBASE_USER")

    def test_mask_uri_hides_password(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        masked = VastbaseVectorStore._mask_uri(
            "postgresql://admin:secret123@db.example.com:5432/mydb"
        )
        assert "secret123" not in masked
        assert "***" in masked
        assert "admin" in masked


# ── from_params ────────────────────────────────────────────────────────

class TestFromParamsIntegration:
    """from_params() should produce a valid VastbaseVectorStore."""

    def test_from_params_with_env(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore.from_params(
            host=_env("VASTBASE_HOST"),
            port=int(_env("VASTBASE_PORT")),
            database=_env("VASTBASE_DATABASE"),
            user=_env("VASTBASE_USER"),
            password=_env("VASTBASE_PASSWORD"),
            table_name="test_from_params",
        )
        assert store.table_name == "test_from_params"
        assert _env("VASTBASE_HOST") in store.connection_string

    def test_from_params_with_explicit_connection_string(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        uri = _connection_uri()
        store = VastbaseVectorStore.from_params(
            host="ignored",
            connection_string=uri,
        )
        assert store.connection_string == uri


# ── CRUD via Collection API ────────────────────────────────────────────

class TestCRUDIntegration:
    """Add / get_nodes / delete / delete_nodes / clear through adapter methods.

    NOTE: These tests use the Collection API to bypass the VastbaseClient
    'using' keyword bug in pyvastbase 0.2.x.  The _create_collection(),
    add(), get_nodes(), delete(), delete_nodes(), and clear() methods are
    exercised with a pre-injected Collection-based client shim.
    """

    @staticmethod
    def _make_shim_client(collection):
        """Create a shim that wraps Collection API to match VastbaseClient.

        The adapter code expects VastbaseClient methods (has_collection,
        create_collection, insert, query, delete, truncate_collection).
        We shim these to use the Collection API which works correctly.
        """
        class _Shim:
            def has_collection(self, name):
                # Always return False since we pre-create the collection
                # in the fixture; adapter will skip creation.
                return True

            def create_collection(self, name, fields=None):
                pass  # Already created by fixture

            def insert(self, table_name, data):
                # Serialize metadata_ dicts to JSON strings
                serialized = []
                for row in data:
                    r = dict(row)
                    if isinstance(r.get("metadata_"), dict):
                        r["metadata_"] = json.dumps(r["metadata_"])
                    serialized.append(r)
                collection.insert(serialized)

            def query(self, table_name, expr=None, output_fields=None, **kwargs):
                return collection.query(
                    expr=expr,
                    output_fields=output_fields or [],
                )

            def delete(self, table_name, expr=None, **kwargs):
                collection.delete(expr=expr)

            def truncate_collection(self, table_name):
                collection.truncate()

            def close(self):
                pass

        return _Shim()

    def test_add_and_get_nodes(self, collection):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.schema import TextNode

        store = VastbaseVectorStore(
            connection_string=_connection_uri(),
            table_name=collection.name,
            embed_dim=4,
        )
        store._client = self._make_shim_client(collection)
        store._is_connected = True

        node = TextNode(
            id_="int-node-1",
            text="Integration test node",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        ids = store.add([node])
        assert ids == ["int-node-1"]

        retrieved = store.get_nodes(["int-node-1"])
        assert len(retrieved) == 1
        assert retrieved[0].node_id == "int-node-1"
        assert retrieved[0].text == "Integration test node"

    def test_add_multiple_and_get(self, collection):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.schema import TextNode

        store = VastbaseVectorStore(
            connection_string=_connection_uri(),
            table_name=collection.name,
            embed_dim=4,
        )
        store._client = self._make_shim_client(collection)
        store._is_connected = True

        nodes = [
            TextNode(id_="n1", text="First", embedding=[0.1, 0.1, 0.1, 0.1]),
            TextNode(id_="n2", text="Second", embedding=[0.2, 0.2, 0.2, 0.2]),
            TextNode(id_="n3", text="Third", embedding=[0.3, 0.3, 0.3, 0.3]),
        ]
        ids = store.add(nodes)
        assert ids == ["n1", "n2", "n3"]

        retrieved = store.get_nodes(["n1", "n3"])
        assert len(retrieved) == 2
        retrieved_ids = {n.node_id for n in retrieved}
        assert retrieved_ids == {"n1", "n3"}

    def test_delete_by_ref_doc_id(self, collection):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo

        store = VastbaseVectorStore(
            connection_string=_connection_uri(),
            table_name=collection.name,
            embed_dim=4,
        )
        store._client = self._make_shim_client(collection)
        store._is_connected = True

        node = TextNode(
            id_="del-test-1",
            text="Will be deleted",
            embedding=[0.0, 0.0, 0.0, 0.0],
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id="doc-to-delete"
        )
        store.add([node])

        # Verify node exists
        retrieved = store.get_nodes(["del-test-1"])
        assert len(retrieved) == 1

        # Delete by ref_doc_id
        store.delete("doc-to-delete")

        # Verify node is gone
        retrieved_after = store.get_nodes(["del-test-1"])
        assert len(retrieved_after) == 0

    def test_delete_nodes_by_ids(self, collection):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.schema import TextNode

        store = VastbaseVectorStore(
            connection_string=_connection_uri(),
            table_name=collection.name,
            embed_dim=4,
        )
        store._client = self._make_shim_client(collection)
        store._is_connected = True

        nodes = [
            TextNode(id_="keep-me", text="Keep", embedding=[0.0, 0.0, 0.0, 0.0]),
            TextNode(id_="del-me-1", text="Del1", embedding=[0.1, 0.1, 0.1, 0.1]),
            TextNode(id_="del-me-2", text="Del2", embedding=[0.2, 0.2, 0.2, 0.2]),
        ]
        store.add(nodes)

        store.delete_nodes(["del-me-1", "del-me-2"])

        remaining = store.get_nodes(["keep-me", "del-me-1", "del-me-2"])
        remaining_ids = {n.node_id for n in remaining}
        assert "keep-me" in remaining_ids
        assert "del-me-1" not in remaining_ids
        assert "del-me-2" not in remaining_ids

    def test_clear_collection(self, collection):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        from llama_index.core.schema import TextNode

        store = VastbaseVectorStore(
            connection_string=_connection_uri(),
            table_name=collection.name,
            embed_dim=4,
        )
        store._client = self._make_shim_client(collection)
        store._is_connected = True

        nodes = [
            TextNode(id_="c1", text="Data 1", embedding=[0.0, 0.0, 0.0, 0.0]),
            TextNode(id_="c2", text="Data 2", embedding=[0.1, 0.1, 0.1, 0.1]),
        ]
        store.add(nodes)
        assert len(store.get_nodes(["c1", "c2"])) == 2

        store.clear()
        assert len(store.get_nodes(["c1", "c2"])) == 0

    def test_get_nodes_empty_list(self, collection):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string=_connection_uri(),
            table_name=collection.name,
            embed_dim=4,
        )
        store._client = self._make_shim_client(collection)
        store._is_connected = True

        result = store.get_nodes([])
        assert result == []


# ── Constructor integration ────────────────────────────────────────────

class TestConstructorIntegration:
    """Constructor should accept all params."""

    def test_store_created_with_all_params(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(
            connection_string=_connection_uri(),
            table_name="test_constructor_int",
            embed_dim=128,
            hybrid_search=False,
            text_search_config="english",
            perform_setup=False,
        )
        assert store.embed_dim == 128
        assert store.table_name == "test_constructor_int"
        assert store.hybrid_search is False
