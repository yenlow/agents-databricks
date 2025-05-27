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

# MAGIC %pip install -U -qqqq langgraph==0.4.3 langgraph-checkpoint==2.0.25 langgraph-supervisor==0.0.21 langchain==0.3.25 langchain_core==0.3.59 langchain-community==0.3.24 pydantic==2.11.4 databricks-sdk==0.52.0 mlflow-skinny==2.22.0 databricks-langchain==0.5.0 databricks-vectorsearch==0.56
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze > requirements_agent.txt

# COMMAND ----------

# MAGIC %md
# MAGIC Use `mlflow.langchain.autolog()` to set up [MLflow traces](https://docs.databricks.com/en/mlflow/mlflow-tracing.html).

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig

mlflow.langchain.autolog()
config = ModelConfig(development_config="config.yml")
config.to_dict()

# COMMAND ----------

query = "Can you give me some troubleshooting steps for SoundWave X5 Pro Headphones that won't connect?"

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

llm = ChatDatabricks(endpoint=config.get("llm_endpoint"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Create a SQL agent managing SQL functions tools

# COMMAND ----------

from databricks_langchain import UCFunctionToolkit
from unitycatalog.ai.core.base import set_uc_function_client
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

set_uc_function_client(DatabricksFunctionClient())
uc_functions = config.get("uc_functions")
sql_tools = UCFunctionToolkit(function_names=uc_functions).tools
print(f"Functions in {uc_functions}:")
[i.name for i in sql_tools]

# COMMAND ----------

from langgraph.prebuilt import create_react_agent

sql_prompt = "You are helpful agent that can use these SQL queries to get latest interaction from a queue of customer service requests, extract the product name from the customer request, get request history of a customer and query policies for return, refund or exchange."
sql_agent = create_react_agent(llm, tools=sql_tools, 
                               prompt=sql_prompt, name="sql")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create a calculator agent that do math using python code

# COMMAND ----------

python_tool = UCFunctionToolkit(function_names=["system.ai.python_exec"]).tools
python_prompt = "You are helpful agent that can use these python functions to calculate transactions from customer service requests."
calculator_agent = create_react_agent(llm, tools=python_tool, 
                                      prompt=python_prompt, name="calculator")

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
genie_space_id = config.get("genie_space_id")
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
  index_name=config.get('retriever')['vs_index'],
  num_results=config.get('retriever')['k'],
  columns=[
    "product_category",
    "product_sub_category",
    "product_name",
    "product_doc",
    "product_id",
    "indexed_doc"
  ],
  tool_name=config.get('retriever')['tool_name'],
  tool_description="Use this tool to search for product documentation.",
)

# Set retriever schema to be returned
# Map the column names in the returned table to MLflow's expected fields: primary_key, text_column, and doc_uri
mlflow.models.set_retriever_schema(
    primary_key="product_id",
    text_column="indexed_doc",
    doc_uri="product_id",
    name=config.get('retriever')['vs_index'],
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
3. genie agent: assign chat with customer service data tasks to this agent
4. retriever agent: assign product documentation search tasks to this agent
Assign work to one agent at a time, do not call agents in parallel.
Do not do any work yourself."""

workflow = create_supervisor(
    [sql_agent, calculator_agent, genie_agent, retriever_agent],
    model=llm,
    prompt=supervisor_prompt,
    output_mode="last_message",
)

full_agent = workflow.compile()

# COMMAND ----------

from IPython.display import display, Image

display(Image(full_agent.get_graph().draw_mermaid_png()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define output parsers to pretty print output
# MAGIC The Databricks UI, such as the AI Playground, can pretty-print tool calls (e.g. markdown text).
# MAGIC Use the following helper functions to parse the LLM's output into the expected format.

# COMMAND ----------

from typing import Iterator, Dict, Any
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    MessageLikeRepresentation,
)

import json

# Pretty-print requests to tool
def stringify_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Convert a raw tool call into a formatted string that the playground UI expects if there is enough information in the tool_call
    """
    try:
        request = json.dumps(
            {
                "id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "arguments": json.dumps(tool_call.get("args", {})),
            },
            indent=2,
        )
        return f"<tool_call>{request}</tool_call>"
    except:
        return str(tool_call)


# Pretty-print responses from tool
def stringify_tool_result(tool_msg: ToolMessage) -> str:
    """
    Convert a ToolMessage into a formatted string that the playground UI expects if there is enough information in the ToolMessage
    """
    try:
        result = json.dumps(
            {"id": tool_msg.tool_call_id, "content": tool_msg.content}, indent=2
        )
        return f"<tool_call_result>{result}</tool_call_result>"
    except:
        return str(tool_msg)


# Parse messages using the above 2 pretty-print functions
def parse_message(msg) -> str:
    """Parse different message types into their string representations"""
    # tool call result
    if isinstance(msg, ToolMessage):
        return stringify_tool_result(msg)
    # tool call
    elif isinstance(msg, AIMessage) and msg.tool_calls:
        tool_call_results = [stringify_tool_call(call) for call in msg.tool_calls]
        return "".join(tool_call_results)
    # normal HumanMessage or AIMessage (reasoning or final answer)
    elif isinstance(msg, (AIMessage, HumanMessage)):
        return msg.content
    else:
        print(f"Unexpected message type: {type(msg)}")
        return str(msg)


# Handle both outputs from invoke or stream
def wrap_output(stream: Iterator[MessageLikeRepresentation]) -> Iterator[str]:
    """
    Process and yield formatted outputs from the message stream.
    The invoke and stream langchain functions produce different output formats.
    This function handles both cases.
    """
    for event in stream:
        # the agent was called with invoke()
        if "messages" in event:
            for msg in event["messages"]:
                yield parse_message(msg) + "\n\n"
        # the agent was called with stream()
        else:
            for node in event:
                for key, messages in event[node].items():
                    if isinstance(messages, list):
                        for msg in messages:
                            yield parse_message(msg) + "\n\n"
                    else:
                        print("Unexpected value {messages} for key {key}. Expected a list of `MessageLikeRepresentation`'s")
                        yield str(messages)

# COMMAND ----------

from langchain_core.runnables import RunnableGenerator
from mlflow.langchain.output_parsers import ChatCompletionsOutputParser

agent = full_agent | RunnableGenerator(wrap_output) | ChatCompletionsOutputParser()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.

# COMMAND ----------

agent.invoke({"messages": [{"role": "user", "content": query}],
              "recursion_limit": 2})

# COMMAND ----------

for event in agent.stream({"messages": [{"role": "user", "content": query}],
                           "recursion_limit": 2}):
    print(event, "---" * 20 + "\n")

# COMMAND ----------

# The defines the object (i.e. agent) that will be logged in the driver NB even if the driver NB references this entire agent NB.
mlflow.models.set_model(agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC You can rerun the cells above to iterate and test the agent.
# MAGIC
# MAGIC Go to the auto-generated [driver]($./driver) notebook in this folder to log, register, and deploy the agent.
