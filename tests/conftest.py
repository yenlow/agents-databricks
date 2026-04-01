"""Shared test fixtures for agents-databricks."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_workspace_client():
    """Mock WorkspaceClient for tests that don't need real Databricks auth."""
    w = MagicMock()
    w.config.host = "https://test.cloud.databricks.com"
    w.current_user.me.return_value.user_name = "test@example.com"
    return w


@pytest.fixture
def sample_config():
    """Sample config matching the structure of 2_agent/config.yml."""
    return {
        "host": "https://test.cloud.databricks.com/",
        "catalog": "test_catalog",
        "schema": "test_schema",
        "llm_endpoint": "test-endpoint",
        "genie_space_id": "test-genie-id",
        "genie_table": "test_catalog.test_schema.cust_service_data",
        "uc_functions": [
            "test_catalog.test_schema.extract_product",
            "test_catalog.test_schema.get_requests_history",
            "test_catalog.test_schema.get_return_policy",
        ],
        "retriever": {
            "tool_name": "search_product_docs",
            "vs_endpoint": "test-endpoint",
            "vs_index": "test_catalog.test_schema.product_docs_vs",
            "vs_source": "test_catalog.test_schema.product_docs",
            "k": 5,
        },
        "lakebase": {
            "instance_name": "test-instance",
            "database": "test_db",
        },
    }
