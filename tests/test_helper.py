"""Tests for helper.py functions."""

import sys
from base64 import b64encode
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path so we can import helper
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pre-mock WorkspaceClient before importing helper, because helper.py evaluates
# WorkspaceClient() as a default argument at class-definition time (line 55).
_mock_ws_default = MagicMock()
with patch("databricks.sdk.WorkspaceClient", return_value=_mock_ws_default):
    import helper


class TestGetSPCredentials:
    """Tests for get_SP_credentials()."""

    def test_decodes_base64_secrets(self):
        """Mock WorkspaceClient().secrets.get_secret() to return base64."""
        mock_ws = MagicMock()

        client_id_b64 = b64encode(b"test-client-id").decode()
        client_secret_b64 = b64encode(b"test-client-secret").decode()

        mock_ws.secrets.get_secret.side_effect = [
            MagicMock(value=client_id_b64),
            MagicMock(value=client_secret_b64),
        ]

        with patch.object(helper, "WorkspaceClient", return_value=mock_ws):
            cid, csecret = helper.get_SP_credentials(
                scope="test-scope",
                client_id_key="client_id",
                client_secret_key="client_secret",
            )

        assert cid == "test-client-id"
        assert csecret == "test-client-secret"
        assert mock_ws.secrets.get_secret.call_count == 2

    def test_uses_provided_values_when_given(self):
        """When client_id_value and client_secret_value are provided, skip secrets."""
        mock_ws = MagicMock()

        with patch.object(helper, "WorkspaceClient", return_value=mock_ws):
            cid, csecret = helper.get_SP_credentials(
                scope="test-scope",
                client_id_key="unused",
                client_secret_key="unused",
                client_id_value="provided-id",
                client_secret_value="provided-secret",
            )

        assert cid == "provided-id"
        assert csecret == "provided-secret"
        mock_ws.secrets.get_secret.assert_not_called()


class TestGetLatestModelVersion:
    """Tests for get_latest_model_version()."""

    def test_returns_highest_version(self):
        """Mock MlflowClient to return multiple versions, verify max is returned."""
        mock_client = MagicMock()
        mock_versions = [
            MagicMock(version="1"),
            MagicMock(version="3"),
            MagicMock(version="2"),
        ]
        mock_client.search_model_versions.return_value = mock_versions

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            result = helper.get_latest_model_version("test-model")

        assert result == 3

    def test_returns_1_when_no_versions(self):
        """When no model versions exist, should return 1 (the default)."""
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            result = helper.get_latest_model_version("empty-model")

        assert result == 1


class TestLakebaseConnect:
    """Tests for LakebaseConnect.__init__()."""

    def test_init_sets_host_from_instance(self):
        """Verify __init__ fetches instance and sets host from read_write_dns."""
        mock_ws = MagicMock()
        mock_ws.database.get_database_instance.return_value = MagicMock(
            read_write_dns="test-host.databricks.com"
        )
        mock_ws.current_user.me.return_value.user_name = "test@example.com"

        conn = helper.LakebaseConnect(
            user="test-user",
            instance_name="test-instance",
            database="test-db",
            wsClient=mock_ws,
        )

        assert conn.host == "test-host.databricks.com"
        assert conn.instance_name == "test-instance"
        assert conn.database == "test-db"
        assert conn.user == "test-user"
        mock_ws.database.get_database_instance.assert_called_once_with(
            name="test-instance"
        )


class TestEvaldfMlflow2To3:
    """Tests for evaldf_mlflow2_to_3()."""

    def test_converts_dataframe(self):
        """Test eval dataset format conversion with either polars or pandas."""
        try:
            import polars as pl

            df = pl.DataFrame(
                {
                    "request": ["test question"],
                    "expected_facts": [["expected answer"]],
                }
            )
        except ImportError:
            import pandas as pd

            df = pd.DataFrame(
                {
                    "request": ["test question"],
                    "expected_facts": [["expected answer"]],
                }
            )

        result = helper.evaldf_mlflow2_to_3(df)

        assert len(result) == 1
        assert result[0]["inputs"]["request"] == "test question"
        assert result[0]["outputs"]["response"] == "expected answer"
        assert result[0]["expectations"]["expected_response"] == "expected answer"

    def test_converts_multiple_rows(self):
        """Test conversion with multiple rows."""
        try:
            import polars as pl

            df = pl.DataFrame(
                {
                    "request": ["q1", "q2"],
                    "expected_facts": [["answer1"], ["answer2"]],
                }
            )
        except ImportError:
            import pandas as pd

            df = pd.DataFrame(
                {
                    "request": ["q1", "q2"],
                    "expected_facts": [["answer1"], ["answer2"]],
                }
            )

        result = helper.evaldf_mlflow2_to_3(df)

        assert len(result) == 2
        assert result[0]["inputs"]["request"] == "q1"
        assert result[1]["inputs"]["request"] == "q2"
