"""
Filter translation utilities for LlamaIndex → Vastbase SQL.

ADAPT: Vastbase V3 is PostgreSQL-compatible with built-in vector engine.
All SQL generated here uses standard PostgreSQL syntax that Vastbase supports
natively — no CREATE EXTENSION, no pgvector-specific functions.
"""

from typing import Any

from llama_index.core.vector_stores.types import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

# Mapping from LlamaIndex FilterOperator to SQL comparison operator.
_OPERATOR_MAP: dict[FilterOperator, str] = {
    FilterOperator.EQ: "=",
    FilterOperator.GT: ">",
    FilterOperator.LT: "<",
    FilterOperator.GTE: ">=",
    FilterOperator.LTE: "<=",
    FilterOperator.NE: "!=",
}


def _escape_value(value: Any) -> str:
    """Convert a Python value to a SQL-safe literal string.

    ADAPT: PostgreSQL-compatible string escaping — single quotes doubled.

    Args:
        value: Python value (str, int, float, bool, None, list, tuple).

    Returns:
        SQL literal string suitable for embedding in a WHERE clause.
    """
    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    if isinstance(value, (list, tuple)):
        items = ", ".join(_escape_value(v) for v in value)
        return f"({items})"

    raise TypeError(
        f"Unsupported value type for SQL escaping: {type(value).__name__}. "
        f"Expected str, int, float, bool, None, list, or tuple."
    )


def _render_metadata_filter(mf: MetadataFilter) -> str:
    """Render a single MetadataFilter to a SQL condition string.

    Args:
        mf: A MetadataFilter with key, value, and operator.

    Returns:
        SQL condition string (e.g. ``"key = 'value'"``).
    """
    key = mf.key
    op = mf.operator
    value = mf.value

    if op in _OPERATOR_MAP:
        return f"{key} {_OPERATOR_MAP[op]} {_escape_value(value)}"

    if op == FilterOperator.IN:
        return f"{key} IN {_escape_value(value)}"

    if op == FilterOperator.NIN:
        return f"{key} NOT IN {_escape_value(value)}"

    if op == FilterOperator.TEXT_MATCH:
        escaped = str(value).replace("'", "''")
        # ADAPT: LIKE with % wildcards is standard SQL, Vastbase-compatible
        return f"{key} LIKE '%{escaped}%'"

    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        escaped = str(value).replace("'", "''")
        return f"{key} ILIKE '%{escaped}%'"

    if op == FilterOperator.CONTAINS:
        # ADAPT: Vastbase supports PostgreSQL-compatible ANY(array) syntax
        return f"{_escape_value(value)} = ANY({key})"

    if op == FilterOperator.IS_EMPTY:
        return f"{key} IS NULL"

    raise ValueError(f"Unsupported filter operator: {op}")


def _to_vastbase_filter(filters: MetadataFilters) -> str:
    """Convert LlamaIndex MetadataFilters to a SQL WHERE clause.

    Recursively handles nested ``MetadataFilters`` for AND/OR/NOT logic.

    ADAPT: All operators use standard SQL syntax compatible with Vastbase V3.
    Vastbase supports PostgreSQL-compatible ANY(array), LIKE/ILIKE, IN, etc.

    Args:
        filters: A ``MetadataFilters`` instance with ``.filters`` (list) and
                 ``.condition`` (FilterCondition or None).

    Returns:
        SQL WHERE clause string (without leading ``WHERE``), or empty string
        if no filters are present.
    """
    if not filters.filters:
        return ""

    condition = filters.condition or FilterCondition.AND
    parts: list[str] = []

    for item in filters.filters:
        if isinstance(item, MetadataFilters):
            # Nested filter group — recurse
            sub = _to_vastbase_filter(item)
            if sub:
                parts.append(f"({sub})")
        elif isinstance(item, MetadataFilter):
            parts.append(_render_metadata_filter(item))

    if not parts:
        return ""

    if condition == FilterCondition.NOT:
        inner = " AND ".join(parts)
        return f"NOT ({inner})"

    if len(parts) == 1:
        return parts[0]

    joiner = f") {condition.value.upper()} ("
    return f"({joiner.join(parts)})"
