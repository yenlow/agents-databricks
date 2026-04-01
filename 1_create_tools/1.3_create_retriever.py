"""Create a VectorSearchRetrieverTool and set the retriever schema.

Reads retriever configuration from 2_agent/config.yml. The tool can be used
directly in-memory or optionally registered as a UC function.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models import ModelConfig

# Allow running from project root: `uv run python 1_create_tools/1.3_create_retriever.py`
sys.path.insert(0, str(Path(__file__).parent))
from deploy_functions import execute_sql  # noqa: E402


def create_retriever_tool(config: ModelConfig) -> VectorSearchRetrieverTool:
    """Build a VectorSearchRetrieverTool from the project config."""
    retriever_cfg = config.get("retriever")
    tool = VectorSearchRetrieverTool(
        index_name=retriever_cfg["vs_index"],
        num_results=retriever_cfg["k"],
        columns=[
            "product_category",
            "product_sub_category",
            "product_name",
            "product_doc",
            "product_id",
            "indexed_doc",
        ],
        tool_name=retriever_cfg["tool_name"],
        tool_description="Use this tool to search for product documentation.",
    )

    mlflow.models.set_retriever_schema(
        primary_key="product_id",
        text_column="indexed_doc",
        doc_uri="product_id",
        name=retriever_cfg["vs_index"],
    )
    return tool


def register_retriever_uc_function(
    w: WorkspaceClient,
    warehouse_id: str,
    config: ModelConfig,
):
    """Optionally register the retriever as a UC function."""
    catalog = config.get("catalog")
    retriever_cfg = config.get("retriever")

    execute_sql(w, warehouse_id, f"CREATE SCHEMA IF NOT EXISTS {catalog}.vs")

    statement = f"""
CREATE OR REPLACE FUNCTION {catalog}.vs.{retriever_cfg['tool_name']} (
  query STRING
  COMMENT "Use this tool to search for product documentation."
) RETURNS TABLE
COMMENT 'Executes a search on product documentation to retrieve text documents most relevant to the input query.'
RETURN
SELECT *
FROM vector_search(
    index => "{retriever_cfg['vs_index']}",
    query => query,
    num_results => {retriever_cfg['k']}
)
"""
    execute_sql(w, warehouse_id, statement, catalog=catalog)


def main():
    parser = argparse.ArgumentParser(description="Create retriever tool")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--warehouse-id", help="SQL warehouse ID (for UC registration)")
    parser.add_argument("--config", default="2_agent/config.yml")
    parser.add_argument(
        "--register-uc",
        action="store_true",
        help="Also register as a UC function",
    )
    args = parser.parse_args()

    cfg = ModelConfig(development_config=args.config)
    tool = create_retriever_tool(cfg)

    # Smoke test
    query = "Can you give me some troubleshooting steps for SoundWave X5 Pro Headphones that won't connect?"
    print(f"Testing retriever with: {query!r}")
    print(tool.invoke(query))

    if args.register_uc:
        if not args.warehouse_id:
            parser.error("--warehouse-id is required when using --register-uc")
        w = WorkspaceClient(profile=args.profile)
        register_retriever_uc_function(w, args.warehouse_id, cfg)
        print("Registered retriever as UC function.")


if __name__ == "__main__":
    main()
