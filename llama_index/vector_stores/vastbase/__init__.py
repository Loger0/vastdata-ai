# llama-index-vector-stores-vastbase
# ADAPT: LlamaIndex vector store integration for Vastbase V3.
# Vastbase is PostgreSQL-compatible and supports pgvector-compatible
# vector operators natively — no CREATE EXTENSION needed.

from llama_index.vector_stores.vastbase.base import (
    VastbaseVectorStore,
    _map_text_search_config,
)
from llama_index.vector_stores.vastbase.utils import (
    _to_vastbase_filter,
    _escape_value,
)

__all__ = [
    "VastbaseVectorStore",
    "_to_vastbase_filter",
    "_escape_value",
    "_map_text_search_config",
]
