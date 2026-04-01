# ruff: noqa: E402
# Imports are interspersed with module-level execution by design --
# this file is an MLflow code model where the top-to-bottom flow matters.

import mlflow
from mlflow.models import ModelConfig

cfg = ModelConfig(development_config="config.yml")

from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint=cfg.get("llm_endpoint"))

from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)

set_uc_function_client(DatabricksFunctionClient())
uc_functions = cfg.get("uc_functions")
sql_tools = UCFunctionToolkit(function_names=uc_functions).tools
print(f"Functions in {uc_functions}: {[i.name for i in sql_tools]}")

from langchain.agents import create_agent

sql_prompt = """You are a helpful agent that can use these 3 tools:
1. extract the product name from the customer request
2. get request history of a customer
3. query policies for return, refund or exchange
"""
sql_agent = create_agent(llm, tools=sql_tools, system_prompt=sql_prompt, name="sql")

python_tool = UCFunctionToolkit(function_names=["system.ai.python_exec"]).tools
python_prompt = "You are a helpful agent that can use the python REPL to calculate transactions from customer service requests."
calculator_agent = create_agent(
    llm, tools=python_tool, system_prompt=python_prompt, name="calculator"
)

api_tool = UCFunctionToolkit(
    function_names=["yen_training.agents.get_recall_api"]
).tools
api_prompt = "You are a helpful agent that can query the Consumer Product Safety Commission recall API to enquire product recall information and its remedy if any"
api_agent = create_agent(llm, tools=api_tool, system_prompt=api_prompt, name="api")

ext_mcp_tool = UCFunctionToolkit(
    function_names=["yen_training.agents.list_repos"]
).tools
ext_mcp_prompt = "You are a helpful agent connected to an external Github MCP server that provides Github related tools like listing repositories."
ext_mcp_agent = create_agent(
    llm, tools=ext_mcp_tool, system_prompt=ext_mcp_prompt, name="ext_mcp"
)

import asyncio
from mcp_utils import create_mcp_tools, workspace_client

custom_mcp_tools = asyncio.run(
    create_mcp_tools(
        ws=workspace_client,
        managed_server_urls=None,  # for Databricks-managed MCP servers
        custom_server_urls=[
            "https://mcp-nitin-1444828305810485.aws.databricksapps.com/mcp"
        ],
    )
)
custom_mcp_prompt = "You are a helpful agent connected on a Databricks MCP server that provides news and the weather information."
custom_mcp_agent = create_agent(
    llm, tools=custom_mcp_tools, system_prompt=custom_mcp_prompt, name="custom_mcp"
)

from databricks_langchain.genie import GenieAgent

genie_space_id = cfg.get("genie_space_id")
genie_agent = GenieAgent(
    genie_space_id, genie_agent_name="Chat with customer service table"
)

from databricks_langchain import VectorSearchRetrieverTool

retriever_tool = VectorSearchRetrieverTool(
    index_name=cfg.get("retriever")["vs_index"],
    num_results=cfg.get("retriever")["k"],
    columns=[
        "product_category",
        "product_sub_category",
        "product_name",
        "product_doc",
        "product_id",
        "indexed_doc",
    ],
    tool_name=cfg.get("retriever")["tool_name"],
    tool_description="Use this tool to search for product documentation.",
)

mlflow.models.set_retriever_schema(
    primary_key="product_id",
    text_column="indexed_doc",
    doc_uri="product_id",
    name=cfg.get("retriever")["vs_index"],
)

retriever_prompt = (
    "You are a helpful retriever agent that can look up product documentation"
)
retriever_agent = create_agent(
    llm, tools=[retriever_tool], system_prompt=retriever_prompt, name="retriever"
)

from langgraph_supervisor import create_supervisor

supervisor_prompt = """You are a supervisor managing several agents:
1. SQL agent: assign specific SQL query tasks to this agent such as extracting product names and looking up return policies and request history
2. calculator agent: assign calculation tasks to this agent
3. API agent: look up the Consumer Product Safety Commission recall API to enquire product recall information and its remedy if any
4. External MCP agent: access Github external MCP server for related repositories
5. Custom MCP agent: access MCP server with custom weather and news tools
6. genie agent: assign chat with customer service data tasks to this agent
7. retriever agent: assign product documentation search tasks to this agent
Assign work to one agent at a time, do not call agents in parallel.
Do not do any work yourself."""

workflow = create_supervisor(
    [
        sql_agent,
        calculator_agent,
        api_agent,
        genie_agent,
        retriever_agent,
        ext_mcp_agent,
        custom_mcp_agent,
    ],
    model=llm,
    prompt=supervisor_prompt,
    output_mode="last_message",
)

from langgraph.checkpoint.postgres import PostgresSaver
from helper import LakebaseConnect, get_SP_credentials
from databricks.sdk import WorkspaceClient

client_id, client_secret = get_SP_credentials(
    scope="yen",
    client_id_key="client_id",
    client_secret_key="client_secret",
    client_id_value="your_client_id",
    client_secret_value="your_client_secret",
)

w = WorkspaceClient(
    host=cfg.get("host"),
    client_id=client_id,
    client_secret=client_secret,
)

dbClient = LakebaseConnect(
    user=client_id,
    password=None,
    instance_name=cfg.get("lakebase").get("instance_name"),
    database=cfg.get("lakebase").get("database"),
    wsClient=w,
)

import json
from uuid import uuid4
from langchain_core.messages import (
    AIMessageChunk,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from typing import Any, Generator, Union
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg_pool import ConnectionPool


class WrappedAgent(ResponsesAgent):
    def __init__(
        self, agent: Union[CompiledStateGraph, StateGraph], conninfo: str = None
    ):
        # if without memory
        if isinstance(agent, CompiledStateGraph):
            self.agent = agent
            self.workflow = None
            self.conninfo = conninfo
            self.pool = None
            self.checkpointer = None
        # if with memory
        elif isinstance(agent, StateGraph):
            self.agent = None
            self.workflow = agent
            self.conninfo = conninfo
            self.pool = ConnectionPool(
                conninfo=self.conninfo,
                kwargs={"autocommit": True},
                min_size=1,
                max_size=10,
                open=True,
            )
            self.checkpointer = PostgresSaver(self.pool)
        else:
            raise Exception(
                "agent must be either a langgraph CompiledStateGraph or a StateGraph"
            )

    def _add_memory(self):
        if self.workflow is not None and self.checkpointer is not None:
            self.agent = self.workflow.compile(checkpointer=self.checkpointer)
        elif self.workflow is not None and self.checkpointer is None:
            # No memory
            self.agent = self.workflow.compile()
            print("No checkpointer found so compiling workflow without memory")

    def _langchain_to_responses(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        "Convert from ChatCompletion dict to Responses output item dictionaries"
        for message in messages:
            message = message.model_dump()
            role = message["type"]
            if role == "ai":
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or uuid4().int,
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                        for tool_call in tool_calls
                    ]
                else:
                    return [
                        self.create_text_output_item(
                            text=message["content"],
                            id=message.get("id") or uuid4().int,
                        )
                    ]
            elif role == "tool":
                return [
                    self.create_function_call_output_item(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                    )
                ]
            elif role == "user":
                return [message]

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = []
        for event in self.predict_stream(request):
            if event.type == "response.output_item.done":
                outputs.append(event.item)
                # overwrite with latest as thread_id is constant through the stream
                custom_outputs = event.custom_outputs
        return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        try:
            config = {
                "configurable": {
                    "thread_id": request.custom_inputs.get("thread_id", uuid4().int)
                }
            }
        except Exception:
            config = {"configurable": {"thread_id": uuid4().int}}

        cc_msgs = []
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        if self.checkpointer:
            self._add_memory()
        for event in self.agent.stream(
            {"messages": cc_msgs, "recursion_limit": 2},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=item,
                            custom_outputs={
                                "thread_id": config["configurable"]["thread_id"]
                            },
                        )
            # filter the streamed messages to just the generated text messages
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                            custom_outputs={
                                "thread_id": config["configurable"]["thread_id"]
                            },
                        )
                except Exception as e:
                    print(e)


# If with memory
dbClient._connect()
conninfo = dbClient.conninfo
agent = WrappedAgent(workflow, conninfo)

# The entrypoint for MLflow code model logging
mlflow.models.set_model(agent)
