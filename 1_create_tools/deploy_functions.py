"""Deploy UC SQL/Python functions to the workspace."""

import argparse
from pathlib import Path

from databricks.sdk import WorkspaceClient
from mlflow.models import ModelConfig


def execute_sql(
    w: WorkspaceClient,
    warehouse_id: str,
    statement: str,
    catalog: str | None = None,
    schema: str | None = None,
):
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


def deploy_functions(
    w: WorkspaceClient, warehouse_id: str, catalog: str, schema: str
):
    sql_dir = Path(__file__).parent.parent / "sql"
    for sql_file in sorted(sql_dir.glob("create_*.sql")):
        print(f"Deploying {sql_file.name}...")
        statement = sql_file.read_text().format(catalog=catalog, schema=schema)
        execute_sql(w, warehouse_id, statement, catalog=catalog, schema=schema)
        print(f"  Done: {sql_file.stem}")


def main():
    parser = argparse.ArgumentParser(description="Deploy UC functions")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--config", default="2_agent/config.yml")
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)
    cfg = ModelConfig(development_config=args.config)
    deploy_functions(w, args.warehouse_id, cfg.get("catalog"), cfg.get("schema"))


if __name__ == "__main__":
    main()
