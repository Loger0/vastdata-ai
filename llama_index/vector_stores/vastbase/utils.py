"""
Filter translation utilities for LlamaIndex → Vastbase SQL.

ADAPT: Vastbase V3 is PostgreSQL-compatible with built-in vector engine.
All SQL generated here uses standard PostgreSQL syntax that Vastbase supports
natively — no CREATE EXTENSION, no pgvector-specific functions.
"""

import json
import re
from typing import Any

from llama_index.core.vector_stores.types import (
    FilterOperator,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

# ADAPT: allow only alphanumeric/hyphen/underscore identifiers in metadata keys
# to prevent SQL injection via user-supplied filter key names.
_SAFE_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Mapping from LlamaIndex FilterOperator to SQL comparison operator.
_OPERATOR_MAP: dict[FilterOperator, str] = {
    FilterOperator.EQ: "=",
    FilterOperator.GT: ">",
    FilterOperator.LT: "<",
    FilterOperator.GTE: ">=",
    FilterOperator.LTE: "<=",
    FilterOperator.NE: "!=",
}


def _validate_key_name(key: str) -> None:
    """Raise ``ValueError`` if *key* contains unsafe characters.

    Args:
        key: The metadata filter key name to validate.

    Raises:
        ValueError: If *key* fails the safe-id pattern check.
    """
    if not _SAFE_KEY_PATTERN.match(key):
        raise ValueError(
            f"Metadata filter key contains unsafe characters: {key!r}. "
            f"Only [a-zA-Z0-9_-] characters are permitted."
        )


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


def _render_metadata_filter(mf: MetadataFilter, key_prefix: str = "") -> str:
    """Render a single MetadataFilter to a SQL condition string.

    ADAPT: When *key_prefix* is provided (e.g. ``"metadata_"``), all key
    references use PostgreSQL JSON extraction syntax
    (``metadata_->>'key'``) so that the filter works against a single JSON
    metadata column.  When *key_prefix* is empty (the default), the key is
    used as-is — suitable for Milvus-style expressions passed to pyvastbase.

    Args:
        mf: A MetadataFilter with key, value, and operator.
        key_prefix: Optional column prefix for JSON extraction.
            When set, keys are rendered as ``{prefix}->>'key'``
            (text extraction) or ``{prefix}->'key'`` (JSONB operators).

    Returns:
        SQL condition string (e.g. ``"metadata_->>'key' = 'value'"``).
    """
    key = mf.key
    op = mf.operator
    value = mf.value

    # ADAPT: validate key name to prevent SQL injection via user-supplied
    # metadata filter keys.
    _validate_key_name(key)

    # Build key reference with optional JSON column prefix
    if key_prefix:
        # Text-extraction path: metadata_->>'key' (returns text)
        key_ref = f"{key_prefix}->>'{key}'"
        # JSONB path: metadata_->'key' (returns jsonb, for ?|, ?&, @>)
        key_ref_json = f"{key_prefix}->'{key}'"
    else:
        key_ref = key
        key_ref_json = key

    if op in _OPERATOR_MAP:
        return f"{key_ref} {_OPERATOR_MAP[op]} {_escape_value(value)}"

    if op == FilterOperator.IN:
        return f"{key_ref} IN {_escape_value(value)}"

    if op == FilterOperator.NIN:
        return f"{key_ref} NOT IN {_escape_value(value)}"

    if op == FilterOperator.TEXT_MATCH:
        escaped = str(value).replace("'", "''")
        # ADAPT: LIKE with % wildcards is standard SQL, Vastbase-compatible
        return f"{key_ref} LIKE '%{escaped}%'"

    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        escaped = str(value).replace("'", "''")
        return f"{key_ref} ILIKE '%{escaped}%'"

    if op == FilterOperator.CONTAINS:
        # ADAPT: JSONB containment check using @> operator.
        # Metadata values may be JSON arrays or objects; @> checks
        # whether the JSONB value at key contains the given value.
        json_val = json.dumps(value)
        return f"{key_ref_json} @> '{json_val}'::jsonb"

    if op == FilterOperator.IS_EMPTY:
        # ADAPT: IS_EMPTY should match both NULL and empty string values.
        # Vastbase/PostgreSQL-compatible: check IS NULL OR = ''.
        return f"({key_ref} IS NULL OR {key_ref} = '')"

    if op == FilterOperator.ANY:
        # ADAPT: JSONB ?| operator — checks if the JSON array at key
        # contains ANY of the given values.  Requires -> (jsonb), not ->> (text).
        items = ", ".join(_escape_value(v) for v in value)
        return f"{key_ref_json} ?| array[{items}]"

    if op == FilterOperator.ALL:
        # ADAPT: JSONB ?& operator — checks if the JSON array at key
        # contains ALL of the given values.  Requires -> (jsonb), not ->> (text).
        items = ", ".join(_escape_value(v) for v in value)
        return f"{key_ref_json} ?& array[{items}]"

    raise ValueError(f"Unsupported filter operator: {op}")


def _to_vastbase_filter(filters: MetadataFilters, key_prefix: str = "") -> str:
    """Convert LlamaIndex MetadataFilters to a SQL WHERE clause.

    Recursively handles nested ``MetadataFilters`` for AND/OR/NOT logic.

    ADAPT: All operators use standard SQL syntax compatible with Vastbase V3.
    Vastbase supports PostgreSQL-compatible ANY(array), LIKE/ILIKE, IN, etc.
    When *key_prefix* is set, key references are prefixed for JSON column
    extraction (e.g. ``metadata_->>'key'``).

    Args:
        filters: A ``MetadataFilters`` instance with ``.filters`` (list) and
                 ``.condition`` (FilterCondition or None).
        key_prefix: Optional column prefix for JSON extraction
            (e.g. ``"metadata_"``).

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
            sub = _to_vastbase_filter(item, key_prefix=key_prefix)
            if sub:
                parts.append(f"({sub})")
        elif isinstance(item, MetadataFilter):
            parts.append(_render_metadata_filter(item, key_prefix=key_prefix))

    if not parts:
        return ""

    if condition == FilterCondition.NOT:
        inner = " AND ".join(parts)
        return f"NOT ({inner})"

    if len(parts) == 1:
        return parts[0]

    joiner = f") {condition.value.upper()} ("
    return f"({joiner.join(parts)})"
