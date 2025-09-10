from databricks.sdk import WorkspaceClient
import psycopg
from psycopg_pool import ConnectionPool
#from sqlalchemy import create_engine, text
from uuid import uuid4
import mlflow
from typing import List
import pandas as pd


def get_latest_model_version(model_name):
    from mlflow.tracking import MlflowClient
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


class LakebaseConnect:
    """A class to manage database connections to Lakebase."""

    def __init__(
        self,
        user: str,
        instance_name: str = "fe_shared_demo",
        database: str = "yen_lakebase",
        password: str = None,
        port: int = 5432,
        wsClient: WorkspaceClient = WorkspaceClient()
    ):
        """
        Initialize the database connection.

        Args:
            instance_name: Name of the database instance
            database: Name of the database
            port: Port number for the connection
        """
        self.instance_name = instance_name
        self.database = database
        self.port = port
        self.user = user
        self.password = password
        self.connection_pool = None
        self.url = None
        self.w = wsClient
        instance = self.w.database.get_database_instance(name=self.instance_name)
        self.host = instance.read_write_dns
        print(f"WorkspaceClient initialized with user {self.w.current_user.me().user_name}")

    def _connect(self):
        """Set up the database connection using Databricks workspace client."""
        # generate ephemeral password (1h)
        # https://docs.databricks.com/aws/en/oltp/authentication?language=Python+SDK#obtain-an-oauth-token-in-a-user-to-machine-flow
        if self.password is None:
            cred = self.w.database.generate_database_credential(
                request_id=str(uuid4()), instance_names=[self.instance_name]
            )
            self.password = cred.token

        self.url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode=require"
        self.conninfo = f"dbname={self.database} user={self.user} password={self.password} host={self.host}"

        self.connection_pool = ConnectionPool(
            conninfo=self.conninfo,
            kwargs={'autocommit': True},
            min_size=1,
            max_size=10,
            open=True
        )
        # If using sqlalchemy
        # Ensure psycopg and not psycopg2 (default)
        # self.connection_pool = create_engine(self.url.replace("postgresql", "postgresql+psycopg"))

    def query(self, query: str):
        """
        Execute a SQL query against the database.

        Args:
            query: SQL query string to execute

        Returns:
            Query result
        """
        if self.connection_pool is None:
            raise RuntimeError(
                "Database connection not initialized. Call _connect() first."
            )

        with self.connection_pool.connection() as conn:
            result = conn.execute(query)
            return result.fetchall()

    def close(self):
        """Close the database connection."""
        if self.connection_pool:
            self.connection_pool.close()
            self.connection_pool = None

    def test_query(self):
        query = "SELECT version()"
        self._connect()
        try:
            result = self.query(query)
            print(f"Successfully queried lakebase with result: {result}")
            return result
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.close()


# Convert for mlflow 3
# https://docs.databricks.com/aws/en/mlflow3/genai/agent-eval-migration
def evaldf_mlflow2_to_3(df: pd.DataFrame) -> List:
    eval_dataset_v3 = []
    for idx, row in df.iterrows():
        response = row["expected_facts"][0]
        eval_dataset_v3.append({
            "inputs": {"request": row["request"]},
            "outputs": {"response": response},
            "expectations": {"expected_response": response}
        })
    return eval_dataset_v3
