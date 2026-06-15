#!/usr/bin/env python3
"""Demo: LlamaIndex + Vastbase Vector Store — 8 scenarios.

Usage::

    VASTBASE_HOST=localhost \\
    VASTBASE_PORT=15432 \\
    VASTBASE_DATABASE=vastbase \\
    VASTBASE_USER=vexdb \\
    VASTBASE_PASSWORD=Vexdb@123 \\
    python demo/llamaindex_vastbase_demo.py

Scenarios:
    1.  Connect to Vastbase and initialize the vector store
    2.  Document vector ingestion (add)
    3.  Vector similarity search (DEFAULT mode)
    4.  Metadata-filtered search (EQ FilterOperator)
    5.  Full-text search (TEXT_SEARCH mode, BM25)
    6.  Hybrid search (HYBRID mode, RRF merge)
    7.  MMR Maximal Marginal Relevance re-ranking
    8.  Delete + clear lifecycle
"""

import os
import sys

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from llama_index.vector_stores.vastbase import VastbaseVectorStore


# ── Helpers ────────────────────────────────────────────────────────────────


def _conn_str() -> str:
    """Build Vastbase connection URI from environment variables."""
    uri = os.environ.get("VASTBASE_URI")
    if uri:
        return uri
    host = os.environ.get("VASTBASE_HOST", "127.0.0.1")
    port = os.environ.get("VASTBASE_PORT", "5432")
    database = os.environ.get("VASTBASE_DATABASE", "test")
    user = os.environ.get("VASTBASE_USER", "postgres")
    password = os.environ.get("VASTBASE_PASSWORD", "Vexdb@123")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _make_node(
    id_: str,
    text: str,
    embedding: list,
    metadata: dict | None = None,
    ref_doc_id: str | None = None,
) -> TextNode:
    """Build a TextNode with optional ref_doc_id."""
    node = TextNode(id_=id_, text=text, embedding=embedding, metadata=metadata or {})
    if ref_doc_id:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=ref_doc_id
        )
    return node


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    conn_str = _conn_str()
    print(f"🔌 Connecting to Vastbase...\n  URI: {conn_str}")

    # ── Scenario 1: Connect & Initialize ─────────────────────────────────
    store = VastbaseVectorStore(
        connection_string=conn_str,
        table_name="demo_llamaindex",
        embed_dim=4,
        hybrid_search=True,
        text_search_config="english",
        perform_setup=True,
    )
    store._initialize()
    print(
        "\n✅ Scenario 1: Connected and initialized\n"
        f"   table_name = {store.table_name}\n"
        f"   embed_dim  = {store.embed_dim}\n"
        f"   hybrid_search = {store.hybrid_search}"
    )

    # ── Scenario 2: Document vector ingestion ────────────────────────────
    nodes = [
        _make_node(
            "n1",
            "Vastbase vector database with BM25 search",
            [0.9, 0.1, 0.0, 0.0],
            {"category": "database", "priority": 1},
            "ref_doc_a",
        ),
        _make_node(
            "n2",
            "PostgreSQL relational database management",
            [0.1, 0.9, 0.0, 0.0],
            {"category": "database", "priority": 2},
            "ref_doc_a",
        ),
        _make_node(
            "n3",
            "Python programming language for AI",
            [0.0, 0.0, 0.9, 0.1],
            {"category": "programming", "priority": 1},
            "ref_doc_b",
        ),
        _make_node(
            "n4",
            "Machine learning with neural networks",
            [0.0, 0.0, 0.1, 0.9],
            {"category": "ai", "priority": 3},
            "ref_doc_c",
        ),
        _make_node(
            "n5",
            "Deep learning with transformers",
            [0.0, 0.0, 0.05, 0.95],
            {"category": "ai", "priority": 2},
            "ref_doc_c",
        ),
    ]
    ids = store.add(nodes)
    print(f"\n✅ Scenario 2: Ingested {len(ids)} documents")
    for nid in ids:
        print(f"   - {nid}")

    # ── Scenario 3: DEFAULT vector similarity search ─────────────────────
    q3 = VectorStoreQuery(
        query_embedding=[0.85, 0.15, 0.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    r3 = store.query(q3)
    print("\n✅ Scenario 3: DEFAULT vector similarity search")
    print(f"   top result: {r3.ids[0] if r3.ids else 'none'}")
    if r3.similarities:
        print(f"   similarity: {r3.similarities[0]:.3f}")

    # ── Scenario 4: Metadata-filtered search ─────────────────────────────
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category", value="database", operator=FilterOperator.EQ
            ),
        ],
        condition=FilterCondition.AND,
    )
    q4 = VectorStoreQuery(
        query_embedding=[0.5, 0.5, 0.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.DEFAULT,
        filters=filters,
    )
    r4 = store.query(q4)
    print("\n✅ Scenario 4: Metadata-filtered search (category = 'database')")
    print(f"   found {len(r4.nodes)} result(s)")
    for n in r4.nodes:
        print(f"   - {n.node_id}: {n.text[:60]}...")

    # ── Scenario 5: TEXT_SEARCH (BM25 full-text) ─────────────────────────
    q5 = VectorStoreQuery(
        query_str="vector database",
        similarity_top_k=3,
        mode=VectorStoreQueryMode.TEXT_SEARCH,
    )
    r5 = store.query(q5)
    print("\n✅ Scenario 5: TEXT_SEARCH 'vector database'")
    print(f"   found {len(r5.nodes)} result(s)")
    for n in r5.nodes:
        print(f"   - {n.node_id}: {n.text[:60]}...")

    # ── Scenario 6: HYBRID search (RRF merge) ────────────────────────────
    q6 = VectorStoreQuery(
        query_embedding=[0.9, 0.1, 0.0, 0.0],
        query_str="database",
        similarity_top_k=3,
        mode=VectorStoreQueryMode.HYBRID,
    )
    r6 = store.query(q6)
    print("\n✅ Scenario 6: HYBRID search (RRF merge)")
    print(f"   found {len(r6.nodes)} result(s)")
    for n, sim in zip(r6.nodes, r6.similarities):
        print(f"   - {n.node_id}: {n.text[:60]}... (RRF={sim:.4f})")

    # ── Scenario 7: MMR diverse re-ranking ───────────────────────────────
    q7 = VectorStoreQuery(
        query_embedding=[0.0, 0.0, 0.5, 0.5],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    q7.mode = "mmr"  # type: ignore[assignment]
    r7 = store.query(q7, mmr_threshold=0.5, mmr_lambda=0.7)
    print("\n✅ Scenario 7: MMR diverse re-ranking")
    print(f"   found {len(r7.nodes)} result(s)")
    for n in r7.nodes:
        print(f"   - {n.node_id}: {n.text[:60]}...")

    # ── Scenario 8: Delete + clear lifecycle ─────────────────────────────
    store.delete("ref_doc_a")
    remaining = store.get_nodes()
    print(f"\n✅ Scenario 8a: Deleted ref_doc_a → {len(remaining)} remaining")
    for n in remaining:
        print(f"   - {n.node_id}")

    store.clear()
    after_clear = store.get_nodes()
    print(f"✅ Scenario 8b: Cleared → {len(after_clear)} remaining")

    store.close()
    print("\n🎉 All 8 scenarios completed!")


if __name__ == "__main__":
    main()
