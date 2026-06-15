"""
Tests for Vastbase filter translation utilities.

Covers _escape_value() and _to_vastbase_filter() — converting LlamaIndex
MetadataFilters to SQL WHERE clauses compatible with Vastbase V3.
"""
import pytest
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

from llama_index.vector_stores.vastbase.utils import (
    _escape_value,
    _to_vastbase_filter,
)


# ── _escape_value tests ──────────────────────────────────────────────

class TestEscapeValue:
    """Value-to-SQL-literal escaping."""

    def test_string_value(self):
        assert _escape_value("hello") == "'hello'"

    def test_integer_value(self):
        assert _escape_value(42) == "42"

    def test_float_value(self):
        assert _escape_value(3.14) == "3.14"

    def test_boolean_true(self):
        assert _escape_value(True) == "TRUE"

    def test_boolean_false(self):
        assert _escape_value(False) == "FALSE"

    def test_none_value(self):
        assert _escape_value(None) == "NULL"

    def test_single_quote_escaping(self):
        # ADAPT: PostgreSQL-compatible string escaping — single quotes doubled
        assert _escape_value("it's a test") == "'it''s a test'"

    def test_list_value(self):
        assert _escape_value([1, 2, 3]) == "(1, 2, 3)"

    def test_tuple_value(self):
        assert _escape_value(("a", "b")) == "('a', 'b')"

    def test_unsupported_type_raises(self):
        """Unsupported types (dict, datetime, etc.) should raise TypeError."""
        with pytest.raises(TypeError):
            _escape_value({"key": "val"})


# ── _to_vastbase_filter operator tests ───────────────────────────────

class TestOperatorEQ:
    """EQ operator → key = value."""

    def test_eq_string(self):
        mf = MetadataFilter(key="name", value="test", operator=FilterOperator.EQ)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "name = 'test'"

    def test_eq_integer(self):
        mf = MetadataFilter(key="count", value=10, operator=FilterOperator.EQ)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "count = 10"


class TestOperatorGT:
    """GT operator → key > value."""

    def test_gt(self):
        mf = MetadataFilter(key="score", value=0.5, operator=FilterOperator.GT)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "score > 0.5"


class TestOperatorLT:
    """LT operator → key < value."""

    def test_lt(self):
        mf = MetadataFilter(key="age", value=100, operator=FilterOperator.LT)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "age < 100"


class TestOperatorGTE:
    """GTE operator → key >= value."""

    def test_gte(self):
        mf = MetadataFilter(key="price", value=9.99, operator=FilterOperator.GTE)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "price >= 9.99"


class TestOperatorLTE:
    """LTE operator → key <= value."""

    def test_lte(self):
        mf = MetadataFilter(key="stock", value=0, operator=FilterOperator.LTE)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "stock <= 0"


class TestOperatorNE:
    """NE operator → key != value."""

    def test_ne(self):
        mf = MetadataFilter(key="status", value="deleted", operator=FilterOperator.NE)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "status != 'deleted'"


class TestOperatorIN:
    """IN operator → key IN (...)."""

    def test_in(self):
        mf = MetadataFilter(key="color", value=["red", "green", "blue"], operator=FilterOperator.IN)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "color IN ('red', 'green', 'blue')"


class TestOperatorNIN:
    """NIN operator → key NOT IN (...)."""

    def test_nin(self):
        mf = MetadataFilter(key="category", value=[1, 2, 3], operator=FilterOperator.NIN)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "category NOT IN (1, 2, 3)"


class TestOperatorTEXTMATCH:
    """TEXT_MATCH operator → key LIKE '%value%'."""

    def test_text_match(self):
        # ADAPT: LIKE with % wildcards is standard SQL, Vastbase-compatible
        mf = MetadataFilter(key="description", value="vector", operator=FilterOperator.TEXT_MATCH)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "description LIKE '%vector%'"

    def test_text_match_with_single_quote(self):
        """Single quotes in value must be escaped to prevent SQL injection."""
        mf = MetadataFilter(key="desc", value="it's tricky", operator=FilterOperator.TEXT_MATCH)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "desc LIKE '%it''s tricky%'"

    def test_text_match_with_percent(self):
        """Percent sign in value — LIKE wildcard, stored as literal."""
        mf = MetadataFilter(key="desc", value="100%", operator=FilterOperator.TEXT_MATCH)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "desc LIKE '%100%%'"

    def test_text_match_with_underscore(self):
        """Underscore in value — LIKE wildcard, stored as literal."""
        mf = MetadataFilter(key="desc", value="hello_world", operator=FilterOperator.TEXT_MATCH)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "desc LIKE '%hello_world%'"


class TestOperatorCONTAINS:
    """CONTAINS operator → key @> 'value'::jsonb (JSONB containment)."""

    def test_contains(self):
        # ADAPT: JSONB containment check using @> operator
        mf = MetadataFilter(key="tags", value="ml", operator=FilterOperator.CONTAINS)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert "@>" in result
        assert "tags" in result
        assert "ml" in result


class TestOperatorISEMPTY:
    """IS_EMPTY operator → (key IS NULL OR key = '')."""

    def test_is_empty(self):
        mf = MetadataFilter(key="deleted_at", value=None, operator=FilterOperator.IS_EMPTY)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert "IS NULL" in result
        assert "= ''" in result
        assert "deleted_at" in result


class TestOperatorANY:
    """ANY operator → key ?| array[...] (JSONB any-of check)."""

    def test_any_strings(self):
        # ADAPT: JSONB ?| operator checks if array contains ANY of the values
        mf = MetadataFilter(
            key="tags", value=["ml", "ai", "db"],
            operator=FilterOperator.ANY,
        )
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert "?|" in result
        assert "array" in result
        assert "ml" in result
        assert "ai" in result
        assert "db" in result

    def test_any_single_value(self):
        mf = MetadataFilter(
            key="status", value=["active"],
            operator=FilterOperator.ANY,
        )
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert "?|" in result
        assert "'active'" in result


class TestOperatorALL:
    """ALL operator → key ?& array[...] (JSONB all-of check)."""

    def test_all_strings(self):
        # ADAPT: JSONB ?& operator checks if array contains ALL of the values
        mf = MetadataFilter(
            key="tags", value=["a", "b"],
            operator=FilterOperator.ALL,
        )
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert "?&" in result
        assert "array" in result
        assert "a" in result
        assert "b" in result


# ── Logical combination tests ────────────────────────────────────────

class TestANDCombination:
    """Multiple filters with AND condition."""

    def test_two_filters_and(self):
        f1 = MetadataFilter(key="x", value=1, operator=FilterOperator.EQ)
        f2 = MetadataFilter(key="y", value=2, operator=FilterOperator.GT)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[f1, f2], condition=FilterCondition.AND)
        )
        assert result == "(x = 1) AND (y > 2)"

    def test_three_filters_and(self):
        f1 = MetadataFilter(key="a", value=1, operator=FilterOperator.EQ)
        f2 = MetadataFilter(key="b", value=2, operator=FilterOperator.EQ)
        f3 = MetadataFilter(key="c", value=3, operator=FilterOperator.EQ)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[f1, f2, f3], condition=FilterCondition.AND)
        )
        assert result == "(a = 1) AND (b = 2) AND (c = 3)"


class TestORCombination:
    """Multiple filters with OR condition."""

    def test_two_filters_or(self):
        f1 = MetadataFilter(key="status", value="active", operator=FilterOperator.EQ)
        f2 = MetadataFilter(key="status", value="pending", operator=FilterOperator.EQ)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[f1, f2], condition=FilterCondition.OR)
        )
        assert result == "(status = 'active') OR (status = 'pending')"


class TestNOTCondition:
    """FilterCondition.NOT negates the inner filter group."""

    def test_not_single(self):
        mf = MetadataFilter(key="deleted", value=1, operator=FilterOperator.EQ)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf], condition=FilterCondition.NOT)
        )
        # NOT should wrap the expression
        assert "NOT" in result
        assert "deleted = 1" in result


# ── Nested / complex tests ───────────────────────────────────────────

class TestNestedFilters:
    """Deeply nested MetadataFilters (list-of-lists recursion)."""

    def test_nested_and_or(self):
        # (a = 1 OR b = 2) AND (c = 3)
        inner1 = MetadataFilters(
            filters=[
                MetadataFilter(key="a", value=1, operator=FilterOperator.EQ),
                MetadataFilter(key="b", value=2, operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )
        inner2 = MetadataFilters(
            filters=[
                MetadataFilter(key="c", value=3, operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.AND,
        )
        outer = MetadataFilters(
            filters=[inner1, inner2],
            condition=FilterCondition.AND,
        )
        result = _to_vastbase_filter(outer)
        assert "(a = 1) OR (b = 2)" in result
        assert "c = 3" in result
        assert result.startswith("(")
        assert result.endswith(")")

    def test_three_level_nesting(self):
        # ((x = 1) AND (y = 2 OR z = 3))
        innermost = MetadataFilters(
            filters=[
                MetadataFilter(key="y", value=2, operator=FilterOperator.EQ),
                MetadataFilter(key="z", value=3, operator=FilterOperator.EQ),
            ],
            condition=FilterCondition.OR,
        )
        middle = MetadataFilters(
            filters=[
                MetadataFilter(key="x", value=1, operator=FilterOperator.EQ),
                innermost,
            ],
            condition=FilterCondition.AND,
        )
        result = _to_vastbase_filter(middle)
        assert "x = 1" in result
        assert "y = 2" in result
        assert "z = 3" in result


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary and edge-case behaviours."""

    def test_empty_filters_list(self):
        """Empty filters should return empty string (no WHERE clause)."""
        result = _to_vastbase_filter(MetadataFilters(filters=[]))
        assert result == ""

    def test_single_filter_no_condition(self):
        """Single filter with default condition should not add extra parens."""
        mf = MetadataFilter(key="id", value=100, operator=FilterOperator.EQ)
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "id = 100"

    def test_single_filter_default_and(self):
        """Single filter with explicit AND — same as default."""
        mf = MetadataFilter(key="id", value=100, operator=FilterOperator.EQ)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf], condition=FilterCondition.AND)
        )
        assert result == "id = 100"


# ── TEXT_MATCH_INSENSITIVE ───────────────────────────────────────────

class TestTextMatchInsensitive:
    """TEXT_MATCH_INSENSITIVE → ILIKE (Vastbase/PostgreSQL compatible)."""

    def test_text_match_insensitive(self):
        mf = MetadataFilter(
            key="title", value="hello",
            operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
        )
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "title ILIKE '%hello%'"

    def test_text_match_insensitive_with_single_quote(self):
        """Single quotes escaped in ILIKE value."""
        mf = MetadataFilter(
            key="title", value="it's a test",
            operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
        )
        result = _to_vastbase_filter(MetadataFilters(filters=[mf]))
        assert result == "title ILIKE '%it''s a test%'"


# ── Key prefix (metadata_->> JSON column) tests ──────────────────────

class TestKeyPrefix:
    """_to_vastbase_filter with key_prefix wraps keys in JSON extraction."""

    def test_eq_with_metadata_prefix(self):
        """key_prefix='metadata_' → metadata_->>'key' = value."""
        mf = MetadataFilter(key="status", value="active", operator=FilterOperator.EQ)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf]), key_prefix="metadata_"
        )
        assert result == "metadata_->>'status' = 'active'"

    def test_in_with_metadata_prefix(self):
        """IN with prefix uses ->> (text extraction)."""
        mf = MetadataFilter(key="color", value=["red", "blue"], operator=FilterOperator.IN)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf]), key_prefix="metadata_"
        )
        assert "metadata_->>'color'" in result
        assert "IN" in result

    def test_text_match_with_metadata_prefix(self):
        """TEXT_MATCH with prefix uses ->> extraction."""
        mf = MetadataFilter(
            key="desc", value="vector", operator=FilterOperator.TEXT_MATCH
        )
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf]), key_prefix="metadata_"
        )
        assert "metadata_->>'desc'" in result
        assert "LIKE" in result

    def test_contains_with_metadata_prefix(self):
        """CONTAINS with prefix uses -> (JSONB, not text) extraction."""
        mf = MetadataFilter(
            key="tags", value="ml", operator=FilterOperator.CONTAINS
        )
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf]), key_prefix="metadata_"
        )
        assert "metadata_->'tags'" in result
        assert "@>" in result

    def test_is_empty_with_metadata_prefix(self):
        """IS_EMPTY with prefix wraps both sides."""
        mf = MetadataFilter(
            key="deleted_at", value=None, operator=FilterOperator.IS_EMPTY
        )
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf]), key_prefix="metadata_"
        )
        assert "metadata_->>'deleted_at' IS NULL" in result
        assert "= ''" in result

    def test_any_with_metadata_prefix(self):
        """ANY with prefix uses -> (JSONB) for ?| operator."""
        mf = MetadataFilter(
            key="tags", value=["ml", "ai"], operator=FilterOperator.ANY
        )
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf]), key_prefix="metadata_"
        )
        assert "metadata_->'tags'" in result
        assert "?|" in result

    def test_all_with_metadata_prefix(self):
        """ALL with prefix uses -> (JSONB) for ?& operator."""
        mf = MetadataFilter(
            key="required", value=["a", "b"], operator=FilterOperator.ALL
        )
        result = _to_vastbase_filter(
            MetadataFilters(filters=[mf]), key_prefix="metadata_"
        )
        assert "metadata_->'required'" in result
        assert "?&" in result

    def test_nested_with_metadata_prefix(self):
        """Nested AND/OR filters should all get the prefix."""
        f1 = MetadataFilter(key="x", value=1, operator=FilterOperator.EQ)
        f2 = MetadataFilter(key="y", value=2, operator=FilterOperator.GT)
        result = _to_vastbase_filter(
            MetadataFilters(filters=[f1, f2], condition=FilterCondition.AND),
            key_prefix="metadata_",
        )
        assert "metadata_->>'x'" in result
        assert "metadata_->>'y'" in result
        assert "AND" in result


# ── Key validation tests ───────────────────────────────────────────

class TestKeyValidation:
    """Filter key names are validated for safe characters only."""

    def test_valid_key_passes(self):
        """Alphanumeric + hyphen + underscore keys should pass."""
        from llama_index.vector_stores.vastbase.utils import _validate_key_name
        _validate_key_name("normal_key")
        _validate_key_name("key-with-dashes")
        _validate_key_name("UPPERCASE_123")

    def test_key_with_spaces_raises(self):
        """Spaces in key should raise ValueError."""
        from llama_index.vector_stores.vastbase.utils import _validate_key_name
        with pytest.raises(ValueError, match="unsafe characters"):
            _validate_key_name("bad key")

    def test_key_with_special_chars_raises(self):
        """Special characters in key should raise ValueError."""
        from llama_index.vector_stores.vastbase.utils import _validate_key_name
        with pytest.raises(ValueError, match="unsafe characters"):
            _validate_key_name("key;DROP TABLE")

    def test_key_with_quotes_raises(self):
        """Quote characters in key should raise ValueError."""
        from llama_index.vector_stores.vastbase.utils import _validate_key_name
        with pytest.raises(ValueError, match="unsafe characters"):
            _validate_key_name("key'name")

    def test_filter_with_invalid_key_raises(self):
        """_to_vastbase_filter should validate keys and raise on unsafe chars."""
        mf = MetadataFilter(
            key="bad;key", value="x", operator=FilterOperator.EQ
        )
        with pytest.raises(ValueError, match="unsafe characters"):
            _to_vastbase_filter(MetadataFilters(filters=[mf]))
