# Databricks notebook source
# MAGIC %md
# MAGIC # Part 2: Create agent (Agent notebook)
# MAGIC This is very similar to an auto-generated notebook created by an AI Playground export. There are three notebooks in the same folder as a set:
# MAGIC 1. [**agent**]($./agent): contains the code to build the agent (only the code in `mlflow.models.set_model` will be served)
# MAGIC 2. [driver]($./driver): references the agent code then logs, registers, evaluates and deploys the agent.
# MAGIC 3. [config.yml]($./config.yml): contains the configuration settings.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) to create your agent. It defines a supervisory LangChain agent that decides which other 4 ReAct agents to assign tasks too. These 4 agents are:
# MAGIC 1. **SQL agent** that can run SQL functions (including batch AI functions) as tools
# MAGIC 2. **Calculator agent** that can run python code for mathematical calculations
# MAGIC 3. **Genie agent** that can do Q&A on structured table(s) using natural language
# MAGIC 4. **Retriever agent** that can do Q&A on unstructured text in a Vector Store
# MAGIC ![](../graph.png)
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, however Mosaic AI Agent Framework is [compatible](%md
# MAGIC
# MAGIC To further customize your LangGraph agent, you can refer to:
# MAGIC * [LangGraph - Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/) for explanations of the concepts used in this LangGraph agent
# MAGIC * [LangGraph - How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/) to expand the functionality of your agent
# MAGIC
# MAGIC ## Prerequisites
# MAGIC 1. Create the required tools (SQL functions, UC functions, Genie Space, Vector Store) in Part 1.
# MAGIC 2. Check  [config.yml]($./config.yml) settings
# MAGIC
# MAGIC ## Next steps
# MAGIC After testing and iterating on your agent in this notebook, go to the auto-generated [driver]($./driver) notebook in this folder to log, register, evaluate, and deploy the agent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip freeze > requirements_agent.txt

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig

cfg = ModelConfig(development_config="config.yml")
cfg.to_dict()

# COMMAND ----------

query = "What was the latest customer service request?"

# COMMAND ----------

# MAGIC %md
# MAGIC # Create the agents
# MAGIC An [agent](https://langchain-ai.github.io/langgraph/agents/overview/) consists of three components: 
# MAGIC 1. a LLM
# MAGIC 2. tool(s) it can use
# MAGIC 3. a prompt that instructs the LLM how to use the tools
# MAGIC
# MAGIC The LLM operates in a loop. In each iteration, it selects a tool to invoke, provides input, receives the result (an observation), and uses that observation to inform the next action. The loop continues until a stopping condition is met — typically when the agent has gathered enough information to respond to the user.
# MAGIC
# MAGIC ### 1. Set the LLM for the ReAct Agents
# MAGIC The LLM can be different for different agents. For simplicity, we will use the same LLM endpoint

# COMMAND ----------

from databricks_langchain import ChatDatabricks

llm = ChatDatabricks(endpoint=cfg.get("llm_endpoint"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Create a SQL agent managing SQL functions tools

# COMMAND ----------

from databricks_langchain import UCFunctionToolkit
from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

set_uc_function_client(DatabricksFunctionClient())
uc_functions = cfg.get("uc_functions")
sql_tools = UCFunctionToolkit(function_names=uc_functions).tools
print(f"Functions in {uc_functions}:")
[i.name for i in sql_tools]

# COMMAND ----------

from langgraph.prebuilt import create_react_agent

sql_prompt = """You are a helpful agent that can use these 3 tools:
1. extract the product name from the customer request
2. get request history of a customer
3. query policies for return, refund or exchange
"""
sql_agent = create_react_agent(llm, tools=sql_tools, 
                               prompt=sql_prompt, name="sql")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create a calculator agent that do math using python code

# COMMAND ----------

python_tool = UCFunctionToolkit(function_names=["system.ai.python_exec"]).tools
python_prompt = "You are a helpful agent that can use the python REPL to calculate transactions from customer service requests."
calculator_agent = create_react_agent(llm, tools=python_tool, 
                                      prompt=python_prompt, name="calculator")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Create a API agent to call the CPSC product recall API to get recall and remedy information

# COMMAND ----------

api_tool = UCFunctionToolkit(function_names=["yen_training.agents.get_recall_api"]).tools
api_prompt = "You are a helpful agent that can query the Consumer Product Safety Commission recall API to enquire product recall information and its remedy if any"
api_agent = create_react_agent(llm, tools=api_tool, 
                               prompt=api_prompt, name="api")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Create Genie Agent that lets you chat with structured table(s)
# MAGIC This assumes you have set up a Genie space earlier in [1.2_create_genie_space]($../01_create_tools/1.2_create_genie_space)
# MAGIC
# MAGIC Note: unlike SQL functions who perform highly specific queries, Genie space will generate free-form SQL code in response to your chat requests and query the customer service table it is attached to.

# COMMAND ----------

from databricks_langchain.genie import GenieAgent

# Get you Genie space ID from the URL 
# https://workspace_host/genie/rooms/<genie_id>/chats/...
genie_space_id = cfg.get("genie_space_id")
genie_agent = GenieAgent(genie_space_id, genie_agent_name="Chat with customer service table")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Create a retriever agent that queries unstructured text
# MAGIC This assumes you have set up a Vector Store earlier in [0_setup]($../01_create_tools/0_setup).<br>
# MAGIC Note: While `VectorSearchRetrieverTool` was instantiated in [1.3_create_retriever](($../01_create_tools/1.3_create_retriever) to persist as a UC function, `VectorSearchRetrieverTool` exists only in memory and will need to be re-instantiated here (or imported)

# COMMAND ----------

from databricks_langchain import VectorSearchRetrieverTool
import mlflow

retriever_tool = VectorSearchRetrieverTool(
  index_name=cfg.get('retriever')['vs_index'],
  num_results=cfg.get('retriever')['k'],
  columns=[
    "product_category",
    "product_sub_category",
    "product_name",
    "product_doc",
    "product_id",
    "indexed_doc"
  ],
  tool_name=cfg.get('retriever')['tool_name'],
  tool_description="Use this tool to search for product documentation.",
)

# Set retriever schema to be returned
# Map the column names in the returned table to MLflow's expected fields: primary_key, text_column, and doc_uri
mlflow.models.set_retriever_schema(
    primary_key="product_id",
    text_column="indexed_doc",
    doc_uri="product_id",
    name=cfg.get('retriever')['vs_index'],
)

retriever_prompt = "You are a helpful retriever agent that can look up product documentation"
retriever_agent = create_react_agent(llm, tools=[retriever_tool], 
                                     prompt=retriever_prompt, name="retriever")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Create a supervisor agent
# MAGIC The supervisor will reason and plan the requests and assigns them to the appropriate agent(s).

# COMMAND ----------

from langgraph_supervisor import create_supervisor

supervisor_prompt = """You are a supervisor managing several agents:
1. SQL agent: assign specific SQL query tasks to this agent such as extracting product names and looking up return policies and request history
2. calculator agent: assign calculation tasks to this agent
3. API agent: look up the Consumer Product Safety Commission recall API to enquire product recall information and its remedy if any
4. genie agent: assign chat with customer service data tasks to this agent
5. retriever agent: assign product documentation search tasks to this agent
Assign work to one agent at a time, do not call agents in parallel.
Do not do any work yourself."""

workflow = create_supervisor(
    [sql_agent, calculator_agent, api_agent, genie_agent, retriever_agent],
    model=llm,
    prompt=supervisor_prompt,
    output_mode="last_message",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding memory
# MAGIC - **Short-term** memory for in-session multi-turn interactions
# MAGIC - **Long-term** memory for between-session interactions
# MAGIC In both cases, add Lakebase/Postgres DB backend to save the memory (either as short-term `PostgresSaver` or long-term `PostgresStore`)
# MAGIC See [langgraph docs](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/?h=#use-in-production)

# COMMAND ----------

# If without DB backend for quick testing
# from langgraph.checkpoint.memory import InMemorySaver

# memory = InMemorySaver()
# full_agent = workflow.compile(checkpointer=memory)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to Lakebase 

# COMMAND ----------

# If with DB backend for production
from langgraph.checkpoint.postgres import PostgresSaver
from helper import LakebaseConnect
from databricks.sdk import WorkspaceClient

client_id=cfg.get("lakebase").get("client_id")
w = WorkspaceClient(
    host=cfg.get("host"),
    client_id=client_id,
    client_secret=cfg.get("lakebase").get("client_secret")
)

dbClient = LakebaseConnect(
    user = client_id,
    password = None, # leave None to generate ephemeral token (1h)
    instance_name = cfg.get("lakebase").get("instance_name"), 
    database = cfg.get("lakebase").get("database"),
    wsClient = w
)
dbClient._connect()
conninfo = dbClient.conninfo

# COMMAND ----------

dbClient.test_query() # connects and closes pool too

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test unwrapped langgraph

# COMMAND ----------

# Keep commented for fast mlflow logging in driver
# from uuid import uuid4
# from psycopg_pool import ConnectionPool

# input_example = {
#     "messages": [
#         {
#             "role": "user",
#             "content": query
#         }
#     ]
# }
# config = {"configurable": {"thread_id": str(uuid4())}}

# # Using langgraph docs
# # with PostgresSaver.from_conn_string(db_uri) as checkpointer:
# #     checkpointer.setup() # if setting up for the first time
# #     full_agent = workflow.compile(checkpointer=checkpointer)
# #     response = full_agent.invoke(input_example, config=config)

# # Better to use a connection pool
# checkpointer = PostgresSaver(dbClient.connection_pool)
# # checkpointer.setup() # if setting up for the first time
# full_agent = workflow.compile(checkpointer=checkpointer)
# response = full_agent.invoke(input_example, config=config)
# # dbClient.close()

# COMMAND ----------

# MAGIC %md
# MAGIC The langgraph checkpoints will be saved in the Postgres database in a table called `checkpoints`
# MAGIC
# MAGIC You can run a PostgresSQL query by going to
# MAGIC Compute > Lakebase Postgres > <your instance> > New Query
# MAGIC ```
# MAGIC select * from <your_database>.public.checkpoints;
# MAGIC ```

# COMMAND ----------

# Keep commented for fast mlflow logging in driver
# import pandas as pd

# dbClient._connect()
# data = dbClient.query("SELECT * FROM checkpoints")
# pd.DataFrame(data).tail()
# # dbClient.close()

# COMMAND ----------

# Uncomment to test stream mode
# Keep commented for fast mlflow logging in driver
# for event in full_agent.stream(
#     input_example,
#     config=config, 
#     stream_mode=["updates", "messages"]
# ):
#     print(event)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap into a ResponsesAgent 
# MAGIC Required by mlflow with custom inputs/outputs

# COMMAND ----------

import json
from uuid import uuid4
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from typing import Any, Generator, Optional, Union
from langgraph.graph.state import CompiledStateGraph, StateGraph
from psycopg_pool import ConnectionPool

class WrappedAgent(ResponsesAgent):
    def __init__(self, 
                 agent: Union[CompiledStateGraph, StateGraph], 
                 conninfo: str = None):
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
                kwargs={'autocommit': True},
                min_size=1,
                max_size=10,
                open=True)
            self.checkpointer = PostgresSaver(self.pool)
        else:
            raise Exception("agent must be either a langgraph CompiledStateGraph or a StateGraph")

    def _add_memory(self):
        if self.workflow is not None and self.checkpointer is not None:
            self.agent = self.workflow.compile(checkpointer=self.checkpointer)
        elif self.workflow is not None and self.checkpointer is None:
            # No memory
            self.agent = self.workflow.compile()
            print("No checkpointer found so compiling workflow without memory")

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert from a Responses API output item to ChatCompletion messages."""
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def _prep_msgs_for_cc_llm(self, responses_input) -> list[dict[str, Any]]:
        "Convert from Responses input items to ChatCompletion dictionaries"
        cc_msgs = []
        for msg in responses_input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

    def _langchain_to_responses(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        "Convert from ChatCompletion dict to Responses output item dictionaries"
        for message in messages:
            message = message.model_dump()
            role = message["type"]
            if role == "ai":
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
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
                            id=message.get("id") or str(uuid4()),
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
            config = {"configurable": {"thread_id": request.custom_inputs.get("thread_id", str(uuid4()))}}
        except Exception as e:
            config = {"configurable": {"thread_id": str(uuid4())}}

        cc_msgs = []
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        if self.checkpointer:
            self._add_memory()
        for event in self.agent.stream(
            {
                "messages": cc_msgs, 
                "recursion_limit": 2
            }, 
            config=config, 
            stream_mode=["updates", "messages"]
        ):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done", 
                            item=item,
                            custom_outputs={"thread_id": config["configurable"]["thread_id"]})
            # filter the streamed messages to just the generated text messages
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                            custom_outputs={"thread_id": config["configurable"]["thread_id"]}
                        )
                except Exception as e:
                    print(e)

# COMMAND ----------

# If without memory
# agent = WrappedAgent(full_agent)
# If with memory
# Disable gssencmode to avoid GSSAPI-encrypted connection in Serving

agent = WrappedAgent(workflow, conninfo)

# COMMAND ----------

# The defines the object (i.e. agent) that will be logged in the driver NB even if the driver NB references this entire agent NB.
mlflow.models.set_model(agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test inferencing

# COMMAND ----------

# # Comment this out so mlflow logging of this NB will be faster
# # Input as dict
# response1 = agent.predict({
#     "input": [{"role": "user", "content": "What is 6*7 in Python?"}], 
#     "custom_inputs": {"thread_id": str(uuid4())}
#     })

# # or input as ResponseAgentRequest
# request = ResponsesAgentRequest(input = input_example['messages'], 
#                                 custom_inputs={"thread_id": str(uuid4())})
# response2 = agent.predict(request)

# # Pass in thread_id via custom input from previous response2.custom_outputs
# request = ResponsesAgentRequest(
#     input = [{
#         'role': 'user',
#         'content': "I tried your suggestions but it still won't connect. What should I do?"
#     }],
#     custom_inputs=response2.custom_outputs)
# response3 = agent.predict(request)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC You can rerun the cells above to iterate and test the agent.
# MAGIC
# MAGIC Go to the auto-generated [driver]($./driver) notebook in this folder to log, register, and deploy the agent.

# COMMAND ----------

# from IPython.display import display, Image

# display(Image(full_agent.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)))
