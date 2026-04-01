"""One-time workspace setup: catalog, schema, tables, vector search, lakebase."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance
from databricks.vector_search.client import VectorSearchClient
from mlflow.models import ModelConfig

# Add project root to path so we can import helper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helper import get_SP_credentials

SOURCE_CATALOG_SCHEMA = "retail_prod.agents"
TABLES_TO_COPY = ["cust_service_data", "policies", "product_docs"]
# CDF is required for VS delta sync
CDF_TABLE = "product_docs"


def execute_sql(
    w: WorkspaceClient,
    warehouse_id: str,
    statement: str,
    catalog: str | None = None,
    schema: str | None = None,
):
    """Execute a SQL statement via the Statement Execution API."""
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        catalog=catalog,
        schema=schema,
        wait_timeout="30s",
    )
    if (
        response.status
        and response.status.state
        and response.status.state.value == "FAILED"
    ):
        raise RuntimeError(f"SQL failed: {response.status.error}")
    return response


def setup_catalog_and_schema(
    w: WorkspaceClient, warehouse_id: str, catalog: str, schema: str
):
    """Create UC catalog and schema if they don't exist."""
    print(f"Creating catalog '{catalog}' (if not exists)...")
    execute_sql(w, warehouse_id, f"CREATE CATALOG IF NOT EXISTS {catalog}")
    print(f"Creating schema '{catalog}.{schema}' (if not exists)...")
    execute_sql(w, warehouse_id, f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")


def copy_tables(w: WorkspaceClient, warehouse_id: str, catalog: str, schema: str):
    """Copy tables from source and enable Change Data Feed on product_docs."""
    for table in TABLES_TO_COPY:
        dest = f"{catalog}.{schema}.{table}"
        src = f"{SOURCE_CATALOG_SCHEMA}.{table}"
        print(f"Copying {src} -> {dest}...")
        execute_sql(w, warehouse_id, f"DROP TABLE IF EXISTS {dest}")
        execute_sql(w, warehouse_id, f"CREATE TABLE {dest} AS SELECT * FROM {src}")

    cdf_dest = f"{catalog}.{schema}.{CDF_TABLE}"
    print(f"Enabling Change Data Feed on {cdf_dest}...")
    execute_sql(
        w,
        warehouse_id,
        f"ALTER TABLE {cdf_dest} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
    )


def setup_vector_search(cfg: ModelConfig):
    """Create VS endpoint (if needed) and delta sync index."""
    retriever = cfg.get("retriever")
    endpoint_name = retriever["vs_endpoint"]
    source_table = retriever["vs_source"]
    index_name = retriever["vs_index"]

    client = VectorSearchClient()

    endpoints = client.list_endpoints()
    existing = [ep.get("name") for ep in endpoints.get("endpoints", [])]
    if endpoint_name not in existing:
        print(f"Creating Vector Search endpoint '{endpoint_name}'...")
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        # Wait a moment for the endpoint to register before creating the index
        print("Waiting for endpoint to become available...")
        time.sleep(5)
    else:
        print(f"Vector Search endpoint '{endpoint_name}' already exists.")

    print(f"Creating delta sync index '{index_name}' from '{source_table}'...")
    index = client.create_delta_sync_index(
        endpoint_name=endpoint_name,
        source_table_name=source_table,
        index_name=index_name,
        pipeline_type="TRIGGERED",
        primary_key="product_id",
        embedding_source_column="product_doc",
        embedding_model_endpoint_name="databricks-gte-large-en",
    )
    print(f"Index created: {index}")


def setup_lakebase(
    w: WorkspaceClient,
    instance_name: str,
    client_id: str,
    client_secret: str,
):
    """Create Lakebase database instance if it doesn't exist."""
    # The notebook authenticates a separate WorkspaceClient with SP creds for Lakebase
    host = w.config.host
    w_sp = WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret)

    if any(db.name == instance_name for db in w_sp.database.list_database_instances()):
        print(f"Lakebase instance '{instance_name}' already exists.")
    else:
        print(f"Creating Lakebase instance '{instance_name}'...")
        w_sp.database.create_database_instance(
            DatabaseInstance(name=instance_name, capacity="CU_1")
        )
        print(f"Lakebase instance '{instance_name}' created.")

    return w_sp


def test_lakebase_connection(
    w_sp: WorkspaceClient,
    client_id: str,
    instance_name: str,
    database: str,
):
    """Test the Lakebase connection using the helper module."""
    from helper import LakebaseConnect

    print("Testing Lakebase connection...")
    db_client = LakebaseConnect(
        user=client_id,
        password=None,  # generates ephemeral token (1h)
        instance_name=instance_name,
        database=database,
        wsClient=w_sp,
    )
    db_client.test_query()


def main():
    parser = argparse.ArgumentParser(
        description="One-time workspace setup: catalog, schema, tables, vector search, lakebase."
    )
    parser.add_argument(
        "--profile", default="dev", help="Databricks CLI profile (default: dev)"
    )
    parser.add_argument(
        "--warehouse-id", required=True, help="SQL Warehouse ID for statement execution"
    )
    parser.add_argument(
        "--config",
        default="2_agent/config.yml",
        help="Path to config.yml (default: 2_agent/config.yml)",
    )
    parser.add_argument(
        "--secret-scope",
        default="yen",
        help="Databricks secret scope for SP credentials (default: yen)",
    )
    parser.add_argument(
        "--client-id",
        default=None,
        help="Service Principal client ID (overrides secret scope lookup)",
    )
    parser.add_argument(
        "--client-secret",
        default=None,
        help="Service Principal client secret (overrides secret scope lookup)",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip table copy step",
    )
    parser.add_argument(
        "--skip-vector-search",
        action="store_true",
        help="Skip vector search setup",
    )
    parser.add_argument(
        "--skip-lakebase",
        action="store_true",
        help="Skip Lakebase setup and connection test",
    )
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)
    cfg = ModelConfig(development_config=args.config)

    catalog = cfg.get("catalog")
    schema = cfg.get("schema")

    # Resolve SP credentials: CLI args take precedence, then fall back to secrets
    client_id = args.client_id
    client_secret = args.client_secret
    if not client_id or not client_secret:
        print(f"Fetching SP credentials from secret scope '{args.secret_scope}'...")
        client_id, client_secret = get_SP_credentials(
            scope=args.secret_scope,
            client_id_key="client_id",
            client_secret_key="client_secret",
            client_id_value=args.client_id,
            client_secret_value=args.client_secret,
        )

    # 1. Catalog + schema
    setup_catalog_and_schema(w, args.warehouse_id, catalog, schema)

    # 2. Copy tables
    if not args.skip_tables:
        copy_tables(w, args.warehouse_id, catalog, schema)
    else:
        print("Skipping table copy (--skip-tables).")

    # 3. Vector search
    if not args.skip_vector_search:
        setup_vector_search(cfg)
    else:
        print("Skipping vector search setup (--skip-vector-search).")

    # 4. Lakebase
    if not args.skip_lakebase:
        lakebase_cfg = cfg.get("lakebase")
        instance_name = lakebase_cfg["instance_name"]
        database = lakebase_cfg["database"]

        w_sp = setup_lakebase(w, instance_name, client_id, client_secret)
        test_lakebase_connection(w_sp, client_id, instance_name, database)
    else:
        print("Skipping Lakebase setup (--skip-lakebase).")

    print("\nSetup complete.")


if __name__ == "__main__":
    main()
