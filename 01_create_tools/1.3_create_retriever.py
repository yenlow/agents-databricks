# Databricks notebook source
# MAGIC %md
# MAGIC # 1.3 Define our retriever tool
# MAGIC A Retrieval Augmented Generation (RAG) agent allows you to do Q&A on unstructured text stored in a Vector Store. You pose a question to the agent and it looks up the vector store it has access to for relevant context to augment its answer. 
# MAGIC
# MAGIC Databricks simplifies the RAG agent by seamlessly integrating Vector Store and Mosaic AI Agent Framework. If you already have a Databricks vector search index and endpoint, you can easily create a tool that retrieves from the vector store index and passes the results to your agent. See [more](https://docs.databricks.com/aws/en/generative-ai/agent-framework/unstructured-retrieval-tools).

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC %pip install databricks-sdk==0.52.0 mlflow-skinny==2.22.0 databricks-langchain==0.5.0 databricks-vectorsearch==0.56 unitycatalog-langchain[databricks] databricks-ai-bridge --no-deps
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use config.yml to set and manage retriever's (many) parameters
# MAGIC - VS endpoint `product_doc_endpoint` in 
# MAGIC - VS index e.g. `yen_training.agents.product_docs_vs`
# MAGIC
# MAGIC [Link](https://e2-demo-field-eng.cloud.databricks.com/explore/data/yen_training/agents/product_docs_vs?o=1444828305810485)
# MAGIC ![](vs_index.png)

# COMMAND ----------

from mlflow.models import ModelConfig

config = ModelConfig(development_config="../02_agent/config.yml")
config.to_dict()

# COMMAND ----------

# Catalog and schema have been automatically created thanks to lab environment
catalog_name = config.get('catalog')
schema_name = config.get('schema')

query = "Can you give me some troubleshooting steps for SoundWave X5 Pro Headphones that won't connect?"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a retriever tool that retrieves from a pre-built vector search index
# MAGIC This assumes you have already set up a Vector Store Index and Endpoint (see [0_setup]($./0_setup)). To create it, see this [guide](https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search).
# MAGIC

# COMMAND ----------

from databricks_langchain import VectorSearchRetrieverTool

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

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set retriever schema to be returned
# MAGIC Map the column names in the returned table to MLflow's expected fields: primary_key, text_column, and doc_uri

# COMMAND ----------

import mlflow

mlflow.models.set_retriever_schema(
    primary_key="product_id",
    text_column="indexed_doc",
    doc_uri="product_id",
    name=config.get('retriever')['vs_index'],
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test query on Vector Store

# COMMAND ----------

# Run a query against the vector search index locally for testing
retriever_tool.invoke(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## [OPTIONAL] Register retriever tool as a UC Function
# MAGIC By registering the vector store query as a UC Function, you'll be able to access and re-use it from the Unity Catalog without having to re-instantiate `VectorSearchRetrieverTool`.
# MAGIC
# MAGIC In this workshop, we will use VS as a `VectorSearchRetrieverTool` in memory instead of a UC function although you can use either.
# MAGIC
# MAGIC NB: the vector store index and endpoint persist regardless

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.vs")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog_name}.vs.{config.get('retriever')['tool_name']} (
  -- The agent uses this comment to determine how to generate the query string parameter.
  query STRING
  COMMENT "Use this tool to search for product documentation."
) RETURNS TABLE
-- The agent uses this comment to determine when to call this tool. It describes the types of documents and information contained within the index.
COMMENT 'Executes a search on product documentation to retrieve text documents most relevant to the input query.' RETURN
SELECT
  *
FROM
  vector_search(
    -- Specify your Vector Search index name here
    index => "yen_training.agents.product_docs_vs",
    query => query,
    num_results => {config.get('retriever')['k']}
  )
""")
