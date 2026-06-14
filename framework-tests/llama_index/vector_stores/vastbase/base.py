"""Stub VastbaseVectorStore for framework test collection.

This is a MINIMAL stub that allows pytest --collect-only to succeed.
All methods raise NotImplementedError — adapter-dev replaces this
with the real implementation as development progresses.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.types import FilterOperator as PGType


# Constants from reference implementation
DEFAULT_MMR_PREFETCH_FACTOR = 2
DBEmbeddingRow = Dict[str, Any]


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase-backed LlamaIndex Vector Store (STUB — TDD target)."""

    stores_text: bool = True
    flat_metadata: bool = False

    def __init__(
        self,
        connection_string: Optional[str] = None,
        async_connection_string: Optional[str] = None,
        table_name: str = "llamaindex",
        schema_name: str = "public",
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        initialization_fail_on_error: bool = False,
        use_halfvec: bool = False,
        engine: Optional[Any] = None,
        async_engine: Optional[Any] = None,
        indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
        customize_query_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self._connection_string = connection_string
        self._async_connection_string = async_connection_string
        self._table_name = table_name
        self._schema_name = schema_name
        self._hybrid_search = hybrid_search
        self._text_search_config = text_search_config
        self._embed_dim = embed_dim
        self._cache_ok = cache_ok
        self._perform_setup = perform_setup
        self._debug = debug
        self._hnsw_kwargs = hnsw_kwargs or {}
        self._initialization_fail_on_error = initialization_fail_on_error
        self._use_halfvec = use_halfvec
        self._collection = None
        self._async_collection = None
        self._is_initialized = False

        # Warn on unsupported params (PGVectorStore compatibility)
        if customize_query_fn is not None:
            warnings.warn(
                "customize_query_fn is not supported by VastbaseVectorStore. Ignored.",
                UserWarning,
            )
        if create_engine_kwargs is not None:
            warnings.warn("create_engine_kwargs is not supported. Ignored.", UserWarning)
        if engine is not None:
            warnings.warn("engine parameter is not supported. Ignored.", UserWarning)
        if async_engine is not None:
            warnings.warn("async_engine parameter is not supported. Ignored.", UserWarning)
        if indexed_metadata_keys is not None:
            warnings.warn("indexed_metadata_keys is not supported. Ignored.", UserWarning)
        if use_jsonb:
            warnings.warn("use_jsonb is not applicable. Ignored.", UserWarning)

    @property
    def client(self) -> Any:
        raise NotImplementedError

    @classmethod
    def from_params(cls, **kwargs) -> "VastbaseVectorStore":
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        raise NotImplementedError

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        raise NotImplementedError

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        raise NotImplementedError

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        raise NotImplementedError

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    async def aclear(self) -> None:
        raise NotImplementedError

    # Internal methods — stub signatures for test compatibility
    def _connect(self) -> None:
        raise NotImplementedError

    def _initialize(self) -> None:
        raise NotImplementedError

    def _build_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        raise NotImplementedError

    def _build_sparse_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        raise NotImplementedError

    def _hybrid_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        raise NotImplementedError

    def _mmr_query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError

    @staticmethod
    def _prepare_mmr_query(
        query: VectorStoreQuery,
        mmr_threshold: Optional[float],
        mmr_prefetch_factor: Optional[float],
        mmr_prefetch_k: Optional[int],
    ) -> VectorStoreQuery:
        raise NotImplementedError

    @staticmethod
    def _get_query_session_settings(query: VectorStoreQuery) -> List[str]:
        raise NotImplementedError
