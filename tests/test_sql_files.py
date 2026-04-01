"""Tests for SQL file parameterization.

Validates that SQL files in sql/ can be parameterized with catalog/schema
placeholders. Skips gracefully if the sql/ directory hasn't been created yet.
"""

from pathlib import Path

import pytest

SQL_DIR = Path(__file__).parent.parent / "sql"


@pytest.mark.skipif(not SQL_DIR.exists(), reason="sql/ directory not created yet")
class TestSqlFileParameterization:
    """Tests for SQL files with {catalog} and {schema} placeholders."""

    def _get_sql_files(self):
        return list(SQL_DIR.glob("*.sql"))

    def test_sql_files_exist(self):
        """At least one SQL file should exist in sql/."""
        sql_files = self._get_sql_files()
        assert len(sql_files) > 0, "No .sql files found in sql/ directory"

    def test_sql_files_can_be_read_and_parameterized(self):
        """All SQL files should be readable and accept catalog/schema formatting."""
        for sql_file in self._get_sql_files():
            content = sql_file.read_text()
            formatted = content.format(catalog="test_catalog", schema="test_schema")
            assert "test_catalog" in formatted or "test_schema" in formatted, (
                f"{sql_file.name} did not contain catalog/schema placeholders"
            )

    def test_no_placeholders_remain_after_formatting(self):
        """After .format(), no {catalog} or {schema} placeholders should remain."""
        for sql_file in self._get_sql_files():
            content = sql_file.read_text()
            formatted = content.format(catalog="test_catalog", schema="test_schema")
            assert "{catalog}" not in formatted, (
                f"{sql_file.name} still has {{catalog}} after formatting"
            )
            assert "{schema}" not in formatted, (
                f"{sql_file.name} still has {{schema}} after formatting"
            )

    def test_parameterized_sql_has_expected_values(self):
        """Parameterized SQL should contain the exact catalog/schema values."""
        for sql_file in self._get_sql_files():
            content = sql_file.read_text()
            if "{catalog}" in content:
                formatted = content.format(catalog="my_catalog", schema="my_schema")
                assert "my_catalog" in formatted
            if "{schema}" in content:
                formatted = content.format(catalog="my_catalog", schema="my_schema")
                assert "my_schema" in formatted
