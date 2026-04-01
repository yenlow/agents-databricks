"""Log, evaluate, and deploy the customer service agent."""

from __future__ import annotations

import argparse
import os
import sys
from uuid import uuid4

import mlflow
import polars as pl
from mlflow.models import ModelConfig
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksLakebase,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksVectorSearchIndex,
)

# Allow importing helper.py from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def log_agent(cfg: ModelConfig, registered_name: str, artifact_path: str = "agent"):
    """Log the agent model to MLflow and register in UC."""
    mlflow.set_registry_uri("databricks-uc")

    input_example = {
        "input": [
            {"role": "user", "content": "What was the latest customer service request?"}
        ],
        "custom_inputs": {"thread_id": uuid4().int},
    }

    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.get("llm_endpoint")),
        DatabricksServingEndpoint(endpoint_name="databricks-gte-large-en"),
        DatabricksVectorSearchIndex(index_name=cfg.get("retriever")["vs_index"]),
        DatabricksFunction(function_name="yen_training.agents.get_requests_history"),
        DatabricksFunction(function_name="yen_training.agents.get_return_policy"),
        DatabricksFunction(function_name="yen_training.agents.extract_product"),
        DatabricksFunction(function_name="yen_training.agents.get_recall_api"),
        DatabricksFunction(function_name="yen_training.agents.list_repos"),
        DatabricksFunction(function_name="system.ai.python_exec"),
        DatabricksUCConnection(connection_name="yen_github_conn"),
        DatabricksGenieSpace(genie_space_id=cfg.get("genie_space_id")),
        DatabricksTable(table_name=cfg.get("genie_table")),
        DatabricksSQLWarehouse(warehouse_id="4b9b953939869799"),
        DatabricksLakebase(database_instance_name=cfg.get("lakebase")["instance_name"]),
    ]

    with mlflow.start_run():
        logged_agent_info = mlflow.pyfunc.log_model(
            python_model=os.path.join(os.getcwd(), "agent"),
            name=artifact_path,
            registered_model_name=registered_name,
            model_config="config.yml",
            pip_requirements="../requirements.txt",
            code_paths=["../helper.py", "mcp_utils.py"],
            input_example=input_example,
            resources=resources,
        )

    print(f"Logged model: {logged_agent_info.model_uri}")
    return logged_agent_info


def build_eval_dataset():
    """Build the hardcoded evaluation dataset (10 Q&A pairs)."""
    data = {
        "request": [
            "What color options are available for the Aria Modern Bookshelf?",
            "How should I clean the Aurora Oak Coffee Table to avoid damaging it?",
            "How should I clean the BlendMaster Elite 4000 after each use?",
            "How many colors is the Flexi-Comfort Office Desk available in?",
            "What sizes are available for the StormShield Pro Men's Weatherproof Jacket?",
            "What should I do if my SmartX Pro device won't turn on?",
            "How many people can the Elegance Extendable Dining Table seat comfortably?",
            "What colors is the Urban Explorer Jacket available in?",
            "What is the water resistance rating of the BrownBox SwiftWatch X500?",
            "What colors are available for the StridePro Runner?",
        ],
        "expected_facts": [
            [
                "The Aria Modern Bookshelf is available in natural oak finish",
                "The Aria Modern Bookshelf is available in black finish",
                "The Aria Modern Bookshelf is available in white finish",
            ],
            [
                "Use a soft, slightly damp cloth for cleaning.",
                "Avoid using abrasive cleaners.",
            ],
            [
                "The jar of the BlendMaster Elite 4000 should be rinsed.",
                "Rinse with warm water.",
                "The cleaning should take place after each use.",
            ],
            ["The Flexi-Comfort Office Desk is available in three colors."],
            [
                "The available sizes for the StormShield Pro Men's Weatherproof Jacket"
                " are Small, Medium, Large, XL, and XXL."
            ],
            [
                "Press and hold the power button for 20 seconds to reset the device.",
                "Ensure the device is charged for at least 30 minutes before"
                " attempting to turn it on again.",
            ],
            ["The Elegance Extendable Dining Table can comfortably seat 6 people."],
            [
                "The Urban Explorer Jacket is available in charcoal, navy,"
                " and olive green"
            ],
            ["The water resistance rating of the BrownBox SwiftWatch X500 is 5 ATM."],
            [
                "The colors available for the StridePro Runner should include"
                " Midnight Blue.",
                "The colors available for the StridePro Runner should include"
                " Electric Red.",
                "The colors available for the StridePro Runner should include"
                " Forest Green.",
            ],
        ],
    }

    return pl.DataFrame(data)


def evaluate_agent(run_id: str, eval_data: pl.DataFrame):
    """Run MLflow GenAI evaluation using Correctness and RelevanceToQuery scorers."""
    from mlflow.genai.scorers import Correctness, RelevanceToQuery

    from helper import evaldf_mlflow2_to_3

    eval_dataset_v3 = evaldf_mlflow2_to_3(eval_data.to_pandas())

    with mlflow.start_run(run_id=run_id):
        eval_results = mlflow.genai.evaluate(
            data=eval_dataset_v3,
            scorers=[Correctness(), RelevanceToQuery()],
        )

    print(f"Evaluation complete. Results: {eval_results}")
    return eval_results


def deploy_agent(registered_name: str, endpoint_name: str = "aatww"):
    """Deploy agent to Model Serving with service principal credentials."""
    from databricks import agents

    from helper import get_latest_model_version, get_SP_credentials

    latest_version = get_latest_model_version(registered_name)
    print(f"Deploying model version {latest_version}")

    client_id, client_secret = get_SP_credentials(
        scope="yen",
        client_id_key="client_id",
        client_secret_key="client_secret",
    )

    agents.deploy(
        model_name=registered_name,
        model_version=latest_version,
        endpoint_name=endpoint_name,
        scale_to_zero=True,
        tags={"endpointSource": "docs"},
        environment_vars={
            "MAX_MODEL_LOADING_TIMEOUT": "600",
            "DATABRICKS_CLIENT_ID": client_id,
            "DATABRICKS_CLIENT_SECRET": client_secret,
        },
    )
    print(f"Deployed {registered_name} v{latest_version} to endpoint '{endpoint_name}'")


def main():
    parser = argparse.ArgumentParser(description="Agent deployment pipeline")
    parser.add_argument(
        "--action",
        choices=["log", "eval", "deploy", "all"],
        default="all",
        help="Pipeline phase to run",
    )
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    parser.add_argument(
        "--endpoint", default="aatww", help="Model serving endpoint name"
    )
    args = parser.parse_args()

    cfg = ModelConfig(development_config=args.config)
    catalog = cfg.get("catalog")
    schema = cfg.get("schema")
    registered_name = f"{catalog}.{schema}.customer_service"

    logged_info = None

    if args.action in ("log", "all"):
        logged_info = log_agent(cfg, registered_name)

    if args.action in ("eval", "all"):
        if logged_info is None:
            parser.error("--action=eval requires a prior log run; use --action=all")
        eval_data = build_eval_dataset()
        evaluate_agent(logged_info.run_id, eval_data)

    if args.action in ("deploy", "all"):
        deploy_agent(registered_name, args.endpoint)


if __name__ == "__main__":
    main()
