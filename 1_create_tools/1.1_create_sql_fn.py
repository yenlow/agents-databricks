"""Create SQL functions and register as UC Functions.

SQL definitions live in sql/create_*.sql. Use deploy_functions.py to deploy them:

    uv run python 1_create_tools/deploy_functions.py --warehouse-id <id>

Functions deployed:
  1. get_return_policy   - Returns the company return policy
  2. get_requests_history - Customer request history by issue category
  3. extract_product     - Extract product name from issue description via ai_extract
  4. get_recall_api      - Python UDF calling Consumer Product Safety Commission API
  5. list_repos          - GitHub repos for a user via HTTP connection
"""
