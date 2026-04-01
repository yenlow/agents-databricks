"""Set up a GenieAgent for natural language SQL queries against structured tables."""

import argparse

from databricks_langchain.genie import GenieAgent
from mlflow.models import ModelConfig


def create_genie_agent(config: ModelConfig) -> GenieAgent:
    """Create a GenieAgent from the project config."""
    genie_space_id = config.get("genie_space_id")
    return GenieAgent(
        genie_space_id,
        "Customer Service",
        description="Chat with your Customer Service table",
    )


def main():
    parser = argparse.ArgumentParser(description="Create and test a GenieAgent")
    parser.add_argument("--config", default="2_agent/config.yml")
    args = parser.parse_args()

    cfg = ModelConfig(development_config=args.config)
    genie_agent = create_genie_agent(cfg)

    # Smoke test
    query = "Which customer had the most interactions?"
    print(f"Testing GenieAgent with: {query!r}")
    response = genie_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    print(response["messages"][0].content)


if __name__ == "__main__":
    main()
