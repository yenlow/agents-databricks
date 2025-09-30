# Databricks notebook source
# MAGIC %md
# MAGIC # Scenario: Build a Customer Service agent app
# MAGIC Functionalities:
# MAGIC 1. Look up customer service requests using SQL function tools
# MAGIC 3. Calculate transactions via a python tool
# MAGIC 4. Do Q&A on unstructured product documentation in a vector store via a RAG agent
# MAGIC 5. Do Q&A on structured tables via a Genie agent
# MAGIC
# MAGIC ![](../graph.png)
# MAGIC
# MAGIC -----------------------
# MAGIC ## Part 1: Create tools
# MAGIC ### 1.1 Create SQL Functions and register as UC Functions
# MAGIC Create queries that access data critical to steps in the customer service workflow for processing a return.
# MAGIC   1. `get_return_policy` -> return policy
# MAGIC   2. `get_request_history` -> number of requests per issue category for that customer
# MAGIC   3. `extract_product` -> product in issue_description (uses `ai_extract`)
# MAGIC
# MAGIC While you can create tools coding them in LangChain/LangGraph, they may not be easily discovered for re-use. Instead, register them as [Unity Catalog Functions](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-tool#unity-catalog-function-tools-vs-agent-code-tools)
# MAGIC
# MAGIC |Tools as...|Pros & Cons|
# MAGIC |------|-----------|
# MAGIC |Unity Catalog functions|Creates a central registry for tools that can be governed like other Unity Catalog objects|
# MAGIC ||Grants easier discoverability and reuse|
# MAGIC ||Examples: SQL functions, python functions
# MAGIC |Agent code tools|Defined in the AI agent's code|
# MAGIC ||Useful for calling REST APIs, using arbitrary code, or executing low-latency tools|
# MAGIC ||Lacks built-in governance and discoverability of functions|
# MAGIC ||Examples: GenieAgent, Retriever (both can also be registered as UC function)

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC # Restart to load the packages into the Python environment
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze

# COMMAND ----------

# DBTITLE 1,Parameter Configs
import os

# Catalog and schema have been automatically created thanks to lab environment
catalog_name = "yen_training"
schema_name = "agents"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Customer Service Requests Processing Workflow
# MAGIC
# MAGIC Below is a structured outline of the **key steps** a customer service agent would typically follow when **processing a return**. This workflow ensures consistency and clarity across your support team.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. UC SQL function: Retrieve Company Policies
# MAGIC - **Action**: Access the internal knowledge base or policy documents related to returns, refunds, and exchanges.  
# MAGIC - **Why**: Verifying you’re in compliance with company guidelines prevents potential errors and conflicts.
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Create function to retrieve return policy
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.get_return_policy()
# MAGIC RETURNS TABLE (policy STRING, policy_details STRING, last_updated DATE)
# MAGIC COMMENT 'Returns the details of the Return Policy'
# MAGIC LANGUAGE SQL
# MAGIC RETURN 
# MAGIC     SELECT policy, policy_details, last_updated 
# MAGIC     FROM retail_prod.agents.policies
# MAGIC     WHERE policy = 'Return Policy'
# MAGIC LIMIT 1;

# COMMAND ----------

# DBTITLE 1,Test function to retrieve return policy
# MAGIC %sql
# MAGIC select * from ${catalog_name}.${schema_name}.get_return_policy()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 2. UC SQL function: Look Up the Order History by name
# MAGIC - **Action**: Query your order management system or customer database using the customer's name.  
# MAGIC - **Why**: Reviewing past purchases, return patterns, and any specific notes helps you determine appropriate next steps (e.g., confirm eligibility for return).
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Create function that retrieves order history based on userID
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.get_requests_history(user_name STRING)
# MAGIC RETURNS TABLE (requests INT, issue_category STRING)
# MAGIC COMMENT "This takes a customer's name as an input and returns the number of requests per issue category"
# MAGIC LANGUAGE SQL
# MAGIC RETURN 
# MAGIC     SELECT count(*) as requests, issue_category
# MAGIC     FROM retail_prod.agents.cust_service_data 
# MAGIC     WHERE name = user_name
# MAGIC     GROUP BY issue_category;

# COMMAND ----------

# DBTITLE 1,Test function that retrieves order history based on userID
# MAGIC %sql
# MAGIC select * from ${catalog_name}.${schema_name}.get_requests_history('Tina Daugherty')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Extract product name from `issue_description`
# MAGIC Use [AI functions](https://docs.databricks.com/aws/en/large-language-models/ai-functions) to do batch (column-wise) LLM inferencing.<br>
# MAGIC The base AI function is [`ai_query`](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query) where you can specify an LLM endpoint and your prompt.<br>
# MAGIC Here, we use [`ai_extract`](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_extract) to extract named entities such as the product name and make it a UC function tool
# MAGIC ---

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.extract_product(text STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns the product mentioned in issue_description'
# MAGIC LANGUAGE SQL
# MAGIC RETURN ai_extract(text, array('product')).product

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW product_table AS
# MAGIC SELECT 
# MAGIC   * , 
# MAGIC   ${catalog_name}.${schema_name}.extract_product(issue_description) as product 
# MAGIC FROM ${catalog_name}.${schema_name}.cust_service_data
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM product_table;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. UC Python function to call external API

# COMMAND ----------

!curl "https://www.saferproducts.gov/RestWebServices/Recall?ProductName=Nuby%20Stroller%20Fans&format=json"

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.get_recall_api(product_name STRING)
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns product recalls from the Consumer Product Safety Commissions'
# MAGIC LANGUAGE PYTHON
# MAGIC AS $$
# MAGIC import requests
# MAGIC
# MAGIC if product_name is None:
# MAGIC     return None
# MAGIC
# MAGIC elif product_name.lower() == 'product':
# MAGIC     return None
# MAGIC
# MAGIC elif len(product_name)==0:
# MAGIC     return None
# MAGIC
# MAGIC else:
# MAGIC     url = "https://www.saferproducts.gov/RestWebServices/Recall"
# MAGIC     headers = {
# MAGIC         'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
# MAGIC         }
# MAGIC     data = {
# MAGIC         "ProductName": product_name,
# MAGIC         "format": "json"
# MAGIC         }
# MAGIC     response = requests.get(url, params=data, headers=headers)
# MAGIC
# MAGIC     if response.status_code == 200:
# MAGIC         data = response.json()
# MAGIC         if len(data)>0:
# MAGIC             try:
# MAGIC                 return data[0].get('Remedies')[0].get('Name')
# MAGIC             except:
# MAGIC                 return None
# MAGIC         else:
# MAGIC             return None
# MAGIC
# MAGIC     else: # API request failed
# MAGIC         return None
# MAGIC $$;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   * , 
# MAGIC   ${catalog_name}.${schema_name}.get_recall_api(product) as recall 
# MAGIC FROM product_table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Tools on External MCP Server (Github) 

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP CONNECTION IF EXISTS yen_github_conn;
# MAGIC CREATE CONNECTION yen_github_conn 
# MAGIC TYPE HTTP
# MAGIC OPTIONS (
# MAGIC   host 'https://api.github.com',
# MAGIC   port 443,
# MAGIC   base_path '/',
# MAGIC   bearer_token '<your_github_token>'
# MAGIC )
# MAGIC COMMENT 'Create connection with external Github MCP server'
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test connection
# MAGIC SELECT http_request(
# MAGIC   conn => 'yen_github_conn',
# MAGIC   method => 'GET',
# MAGIC   path => concat('/users/', 'yenlow', '/repos'),
# MAGIC   headers => map(
# MAGIC     'Accept', "application/vnd.github+json"
# MAGIC   )
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION ${catalog_name}.${schema_name}.list_repos(
# MAGIC   user STRING  COMMENT 'GitHub username'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC COMMENT 'Returns JSON array of public repos for a user'
# MAGIC RETURN http_request(
# MAGIC   conn => 'yen_github_conn',
# MAGIC   method => 'GET',
# MAGIC   path => concat('/users/', user, '/repos'),
# MAGIC   headers => map('Accept', "application/vnd.github+json")
# MAGIC ).text;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ${catalog_name}.${schema_name}.list_repos('yenlow');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Check out the SQL functions in UC 

# COMMAND ----------

# DBTITLE 1,Let's take a look at our created functions
from IPython.display import display, HTML

# Retrieve the Databricks host URL
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Create HTML link to created functions
html_link = f'<a href="https://{workspace_url}/explore/data/functions/{catalog_name}/{schema_name}/get_recall_api" target="_blank">Go to Unity Catalog to see Registered Functions</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP FUNCTION ${catalog_name}.${schema_name}.list_my_repos;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Test the tools in AI Playground
# MAGIC - Bind the tools to a LLM that can reason and plan which tool to use
# MAGIC - Dive deeper into the agent’s performance by exploring MLflow traces.
# MAGIC
# MAGIC The AI Playground can be found on the left navigation bar under 'Machine Learning' or you can use the link created below

# COMMAND ----------


