"""
Tests for VastbaseVectorStore query engine.

Covers _build_filter_clause, _build_query (DENSE), _build_sparse_query (SPARSE/TEXT_SEARCH),
_hybrid_query (HYBRID), _mmr_query (MMR), query() dispatch, and aquery() async query.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """Return a MagicMock standing in for VastbaseClient with search support."""
    mc = MagicMock()
    mc.has_collection.return_value = True
    mc.search.return_value = MagicMock()
    mc.query.return_value = []
    mc.hybrid_search.return_value = MagicMock()
    return mc


@pytest.fixture
def store_with_mock(mock_client):
    """Return a VastbaseVectorStore with mock client injected."""
    from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

    store = VastbaseVectorStore(
        connection_uri="postgresql://localhost:5432/vastbase",
        table_name="test_nodes",
        dimension=128,
    )
    store._client = mock_client
    store._is_connected = True
    return store


# ── Helpers for building mock search results ────────────────────────────────

def _make_search_result_item(id_, distance, text="", metadata_=None, embedding=None, ref_doc_id=None):
    """Create a mock SearchResultItem."""
    from pyvastbase.core.search_result import SearchResultItem

    return SearchResultItem(
        id=id_,
        distance=distance,
        data={
            "id": id_,
            "text": text,
            "metadata_": metadata_ or {},
            "embedding": embedding or [],
            "ref_doc_id": ref_doc_id or "",
        },
    )


def _make_search_result(items):
    """Create a mock SearchResult from a list of SearchResultItem."""
    from pyvastbase import SearchResult
    sr = SearchResult()
    sr.results = [items]
    sr.num_queries = 1
    return sr


# ── _build_filter_clause tests ──────────────────────────────────────────────

class TestBuildFilterClause:
    """_build_filter_clause() converts MetadataFilters to SQL expr string."""

    def test_none_filters(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        assert store._build_filter_clause(None) == ""

    def test_empty_filters(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        result = store._build_filter_clause(MetadataFilters(filters=[]))
        assert result == ""

    def test_single_filter(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        mf = MetadataFilter(key="status", value="active", operator=FilterOperator.EQ)
        result = store._build_filter_clause(MetadataFilters(filters=[mf]))
        assert "status" in result
        assert "active" in result

    def test_multiple_filters_and(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore
        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        f1 = MetadataFilter(key="x", value=1, operator=FilterOperator.EQ)
        f2 = MetadataFilter(key="y", value=2, operator=FilterOperator.GT)
        result = store._build_filter_clause(
            MetadataFilters(filters=[f1, f2], condition=FilterCondition.AND)
        )
        assert "x" in result
        assert "y" in result
        assert "AND" in result


# ── _build_query (DENSE) tests ──────────────────────────────────────────────

class TestBuildQueryDense:
    """_build_query() performs DENSE vector similarity search."""

    def test_dense_search_basic(self, store_with_mock, mock_client):
        """Basic dense search with query embedding."""
        items = [
            _make_search_result_item("n1", 0.1, text="Doc 1"),
            _make_search_result_item("n2", 0.3, text="Doc 2"),
        ]
        mock_client.search.return_value = _make_search_result(items)

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store_with_mock._build_query(query)

        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.ids) == 2
        assert result.ids == ["n1", "n2"]
        mock_client.search.assert_called_once()

    def test_dense_search_with_filters(self, store_with_mock, mock_client):
        """Dense search with metadata filters passes expr to search."""
        items = [_make_search_result_item("n1", 0.2, text="Match")]
        mock_client.search.return_value = _make_search_result(items)

        mf = MetadataFilter(key="topic", value="tech", operator=FilterOperator.EQ)
        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=3,
            filters=MetadataFilters(filters=[mf]),
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store_with_mock._build_query(query)

        assert len(result.ids) == 1
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["filter_expr"] is not None
        assert "topic" in str(call_kwargs["filter_expr"])

    def test_dense_search_no_embedding_raises(self, store_with_mock):
        """query_embedding=None should raise ValueError."""
        query = VectorStoreQuery(
            query_embedding=None,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        with pytest.raises(ValueError, match="query_embedding"):
            store_with_mock._build_query(query)

    def test_dense_search_empty_results(self, store_with_mock, mock_client):
        """Empty results from search should return empty VectorStoreQueryResult."""
        mock_client.search.return_value = _make_search_result([])

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store_with_mock._build_query(query)

        assert result.ids == []
        assert result.nodes == []
        assert result.similarities == []

    def test_dense_search_preserves_node_metadata(self, store_with_mock, mock_client):
        """Dense search should preserve text, metadata, and ref_doc_id."""
        items = [
            _make_search_result_item(
                "n1", 0.15, text="Hello", metadata_={"k": "v"}, ref_doc_id="doc-1"
            ),
        ]
        mock_client.search.return_value = _make_search_result(items)

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store_with_mock._build_query(query)

        assert len(result.nodes) == 1
        node = result.nodes[0]
        assert node.node_id == "n1"
        assert node.text == "Hello"
        assert node.metadata == {"k": "v"}
        assert node.ref_doc_id == "doc-1"

    def test_dense_search_sim_to_distance_conversion(self, store_with_mock, mock_client):
        """Similarity scores are computed from cosine distances (1 - distance)."""
        items = [
            _make_search_result_item("n1", 0.05, text="Close"),
            _make_search_result_item("n2", 0.8, text="Far"),
        ]
        mock_client.search.return_value = _make_search_result(items)

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store_with_mock._build_query(query)

        # Similarity = 1 - distance (cosine distance)
        assert result.similarities[0] == pytest.approx(0.95)
        assert result.similarities[1] == pytest.approx(0.20)


# ── _build_sparse_query (SPARSE/TEXT_SEARCH) tests ──────────────────────────

class TestBuildSparseQuery:
    """_build_sparse_query() performs BM25/full-text search."""

    def test_sparse_search_basic(self, store_with_mock, mock_client):
        """Basic sparse search using query_str."""
        # Mock psycopg connection + cursor for raw SQL execution
        mock_rows = [
            {"id": "n1", "text": "Hello world", "metadata_": {}, "embedding": [], "ref_doc_id": "", "rank": 0.8},
            {"id": "n2", "text": "World peace", "metadata_": {}, "embedding": [], "ref_doc_id": "", "rank": 0.3},
        ]

        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [
                ("id",), ("text",), ("metadata_",), ("embedding",), ("ref_doc_id",), ("rank",),
            ]
            mock_cursor.fetchall.return_value = [
                ("n1", "Hello world", {}, [], "", 0.8),
                ("n2", "World peace", {}, [], "", 0.3),
            ]
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            query = VectorStoreQuery(
                query_str="hello",
                similarity_top_k=5,
                mode=VectorStoreQueryMode.SPARSE,
            )
            result = store_with_mock._build_sparse_query(query)

        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.ids) == 2
        assert result.ids == ["n1", "n2"]
        # Similarities should reflect rank scores
        assert result.similarities[0] == pytest.approx(0.8)

    def test_sparse_search_no_query_str_raises(self, store_with_mock):
        """query_str=None should raise ValueError."""
        query = VectorStoreQuery(
            query_str=None,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.SPARSE,
        )
        with pytest.raises(ValueError, match="query_str"):
            store_with_mock._build_sparse_query(query)

    def test_sparse_search_with_filters(self, store_with_mock):
        """Sparse search with metadata filters."""
        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [
                ("id",), ("text",), ("metadata_",), ("embedding",), ("ref_doc_id",), ("rank",),
            ]
            mock_cursor.fetchall.return_value = []
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            mf = MetadataFilter(key="topic", value="news", operator=FilterOperator.EQ)
            query = VectorStoreQuery(
                query_str="breaking",
                similarity_top_k=5,
                filters=MetadataFilters(filters=[mf]),
                mode=VectorStoreQueryMode.TEXT_SEARCH,
            )
            result = store_with_mock._build_sparse_query(query)

        assert result.ids == []

    def test_sparse_search_uses_sparse_top_k(self, store_with_mock):
        """sparse_top_k should override similarity_top_k for sparse search."""
        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [
                ("id",), ("text",), ("metadata_",), ("embedding",), ("ref_doc_id",), ("rank",),
            ]
            mock_cursor.fetchall.return_value = []
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            query = VectorStoreQuery(
                query_str="test",
                similarity_top_k=10,
                sparse_top_k=3,
                mode=VectorStoreQueryMode.SPARSE,
            )
            store_with_mock._build_sparse_query(query)

            # Verify the limit parameter passed to execute is sparse_top_k (3)
            params = mock_cursor.execute.call_args[0][1]
            # Last param should be the LIMIT value
            assert params[-1] == 3


# ── _hybrid_query (HYBRID) tests ────────────────────────────────────────────

class TestHybridQuery:
    """_hybrid_query() combines dense + sparse with RRF dedup."""

    def test_hybrid_query_combines_results(self, store_with_mock, mock_client):
        """Hybrid should merge dense + sparse results, deduplicate, and sort."""
        dense_items = [
            _make_search_result_item("n1", 0.1, text="Dense match"),
            _make_search_result_item("n2", 0.2, text="Another dense"),
        ]
        mock_client.search.return_value = _make_search_result(dense_items)

        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [
                ("id",), ("text",), ("metadata_",), ("embedding",), ("ref_doc_id",), ("rank",),
            ]
            mock_cursor.fetchall.return_value = [
                ("n2", "Another dense", {}, [], "", 0.9),
                ("n3", "Sparse only", {}, [], "", 0.7),
            ]
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            query = VectorStoreQuery(
                query_embedding=[0.1] * 128,
                query_str="test query",
                similarity_top_k=3,
                mode=VectorStoreQueryMode.HYBRID,
            )
            result = store_with_mock._hybrid_query(query)

        # n2 appears in both, should be deduped; n1 and n3 are unique
        assert len(result.ids) == 3
        assert set(result.ids) == {"n1", "n2", "n3"}

    def test_hybrid_query_no_embedding(self, store_with_mock):
        """Hybrid without embedding should raise ValueError."""
        query = VectorStoreQuery(
            query_embedding=None,
            query_str="test",
            similarity_top_k=5,
            mode=VectorStoreQueryMode.HYBRID,
        )
        with pytest.raises(ValueError, match="query_embedding"):
            store_with_mock._hybrid_query(query)

    def test_hybrid_query_no_query_str(self, store_with_mock):
        """Hybrid without query_str should raise ValueError."""
        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            query_str=None,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.HYBRID,
        )
        with pytest.raises(ValueError, match="query_str"):
            store_with_mock._hybrid_query(query)


# ── _mmr_query (MMR) tests ──────────────────────────────────────────────────

class TestMMRQuery:
    """_mmr_query() implements Maximal Marginal Relevance."""

    def test_mmr_query_basic(self, store_with_mock, mock_client):
        """MMR should prefetch candidates and apply MMR algorithm."""
        # Create enough results for MMR to work
        items = []
        for i in range(10):
            # Vary embeddings to give MMR something to work with
            emb = [0.1 * (i + 1)] * 128
            items.append(
                _make_search_result_item(
                    f"n{i}", 0.1 * i, text=f"Doc {i}", embedding=emb
                )
            )
        mock_client.search.return_value = _make_search_result(items)

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        result = store_with_mock._mmr_query(query)

        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.ids) == 3  # top_k results
        # Should have called search with prefetch_k > similarity_top_k
        search_call_limit = mock_client.search.call_args.kwargs["limit"]
        assert search_call_limit >= query.similarity_top_k

    def test_mmr_query_no_embedding_raises(self, store_with_mock):
        """MMR without embedding should raise ValueError."""
        query = VectorStoreQuery(
            query_embedding=None,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.MMR,
        )
        with pytest.raises(ValueError, match="query_embedding"):
            store_with_mock._mmr_query(query)

    def test_mmr_query_fallback_on_few_results(self, store_with_mock, mock_client):
        """If prefetch returns fewer than top_k results, fallback gracefully."""
        items = [_make_search_result_item("n1", 0.1, text="Only one", embedding=[0.1] * 128)]
        mock_client.search.return_value = _make_search_result(items)

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.MMR,
        )
        result = store_with_mock._mmr_query(query)

        # Should return whatever is available
        assert len(result.ids) == 1

    def test_mmr_query_respects_mmr_threshold(self, store_with_mock, mock_client):
        """MMR should pass mmr_threshold parameter through."""
        items = []
        for i in range(10):
            items.append(
                _make_search_result_item(
                    f"n{i}", 0.1 * i, text=f"Doc {i}", embedding=[0.1 * (i + 1)] * 128
                )
            )
        mock_client.search.return_value = _make_search_result(items)

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
            mmr_threshold=0.8,
        )
        result = store_with_mock._mmr_query(query)

        assert len(result.ids) == 3

    def test_mmr_query_empty_prefetch(self, store_with_mock, mock_client):
        """Empty prefetch results should return empty query result."""
        mock_client.search.return_value = _make_search_result([])

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.MMR,
        )
        result = store_with_mock._mmr_query(query)

        assert result.ids == []
        assert result.nodes == []


# ── query() dispatch tests ──────────────────────────────────────────────────

class TestQueryDispatch:
    """query() dispatches to the correct internal method based on mode."""

    def test_dispatch_default_mode(self, store_with_mock, mock_client):
        """DEFAULT mode → _build_query."""
        mock_client.search.return_value = _make_search_result([])

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store_with_mock.query(query)

        assert isinstance(result, VectorStoreQueryResult)
        mock_client.search.assert_called_once()

    def test_dispatch_sparse_mode(self, store_with_mock):
        """SPARSE mode → _build_sparse_query."""
        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [
                ("id",), ("text",), ("metadata_",), ("embedding",), ("ref_doc_id",), ("rank",),
            ]
            mock_cursor.fetchall.return_value = []
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            query = VectorStoreQuery(
                query_str="test",
                similarity_top_k=5,
                mode=VectorStoreQueryMode.SPARSE,
            )
            result = store_with_mock.query(query)

        assert isinstance(result, VectorStoreQueryResult)

    def test_dispatch_text_search_mode(self, store_with_mock):
        """TEXT_SEARCH mode → _build_sparse_query."""
        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [
                ("id",), ("text",), ("metadata_",), ("embedding",), ("ref_doc_id",), ("rank",),
            ]
            mock_cursor.fetchall.return_value = []
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            query = VectorStoreQuery(
                query_str="test",
                similarity_top_k=5,
                mode=VectorStoreQueryMode.TEXT_SEARCH,
            )
            result = store_with_mock.query(query)

        assert isinstance(result, VectorStoreQueryResult)

    def test_dispatch_hybrid_mode(self, store_with_mock, mock_client):
        """HYBRID mode → _hybrid_query."""
        mock_client.search.return_value = _make_search_result([])

        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.description = [
                ("id",), ("text",), ("metadata_",), ("embedding",), ("ref_doc_id",), ("rank",),
            ]
            mock_cursor.fetchall.return_value = []
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            query = VectorStoreQuery(
                query_embedding=[0.1] * 128,
                query_str="test",
                similarity_top_k=5,
                mode=VectorStoreQueryMode.HYBRID,
            )
            result = store_with_mock.query(query)

        assert isinstance(result, VectorStoreQueryResult)

    def test_dispatch_mmr_mode(self, store_with_mock, mock_client):
        """MMR mode → _mmr_query."""
        items = []
        for i in range(10):
            items.append(
                _make_search_result_item(
                    f"n{i}", 0.1 * i, text=f"Doc {i}", embedding=[0.1 * (i + 1)] * 128
                )
            )
        mock_client.search.return_value = _make_search_result(items)

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=3,
            mode=VectorStoreQueryMode.MMR,
        )
        result = store_with_mock.query(query)

        assert isinstance(result, VectorStoreQueryResult)
        assert len(result.ids) == 3

    def test_dispatch_unknown_mode_raises(self, store_with_mock):
        """Unknown mode should raise ValueError."""
        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode="UNKNOWN_MODE",
        )
        with pytest.raises(ValueError, match="Unsupported query mode"):
            store_with_mock.query(query)


# ── _search_results_to_query_result tests ───────────────────────────────────

class TestSearchResultsToQueryResult:
    """_search_results_to_query_result() converts SearchResult to VectorStoreQueryResult."""

    def test_converts_valid_results(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        items = [
            _make_search_result_item("n1", 0.1, text="Doc 1", metadata_={"k": "v"}, ref_doc_id="d1"),
            _make_search_result_item("n2", 0.2, text="Doc 2", metadata_={}, ref_doc_id="d2"),
        ]
        sr = _make_search_result(items)

        result = store._search_results_to_query_result(sr)

        assert len(result.nodes) == 2
        assert result.ids == ["n1", "n2"]
        assert result.similarities == [0.9, 0.8]
        assert result.nodes[0].ref_doc_id == "d1"
        assert result.nodes[1].ref_doc_id == "d2"

    def test_converts_empty_results(self):
        from llama_index.vector_stores.vastbase.base import VastbaseVectorStore

        store = VastbaseVectorStore(connection_uri="postgresql://localhost:5432/db")
        sr = _make_search_result([])

        result = store._search_results_to_query_result(sr)

        assert result.ids == []
        assert result.nodes == []
        assert result.similarities == []


# ── _dedup_results tests ────────────────────────────────────────────────────

class TestDedupResults:
    """_dedup_results() removes duplicates by node_id."""

    def test_dedup_removes_duplicates(self):
        from llama_index.vector_stores.vastbase.base import _dedup_results

        class FakeRow:
            def __init__(self, node_id, similarity):
                self.node_id = node_id
                self.similarity = similarity

        rows = [
            FakeRow("a", 0.9),
            FakeRow("b", 0.8),
            FakeRow("a", 0.7),  # duplicate
        ]
        deduped = _dedup_results(rows)
        assert len(deduped) == 2
        assert {r.node_id for r in deduped} == {"a", "b"}


# ── query() integration tests ───────────────────────────────────────────────

class TestQueryIntegration:
    """End-to-end query behaviour."""

    def test_query_initializes_client(self, store_with_mock, mock_client):
        """query() should work without explicit _connect() call."""
        # _is_connected is True from fixture, but client should work
        mock_client.search.return_value = _make_search_result([])

        query = VectorStoreQuery(
            query_embedding=[0.1] * 128,
            similarity_top_k=5,
            mode=VectorStoreQueryMode.DEFAULT,
        )
        result = store_with_mock.query(query)
        assert isinstance(result, VectorStoreQueryResult)
