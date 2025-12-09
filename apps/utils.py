import os
import requests
import json
from databricks.sdk import WorkspaceClient


def get_user_info_from_headers(headers: dict) -> dict:
    """Extract user info from HTTP headers (framework-agnostic).

    Args:
        headers: Dictionary of HTTP headers

    Returns:
        Dictionary with user_name, user_email, and user_id
    """
    if headers is None:
        headers = {}
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )


def get_user_info():
    """Get user info from Streamlit context headers.

    This function is Streamlit-specific. For Chainlit, use get_user_info_from_headers.
    """
    try:
        import streamlit as st
        headers = st.context.headers
        return get_user_info_from_headers(headers)
    except ImportError:
        # Streamlit not available, return empty dict
        return dict(user_name=None, user_email=None, user_id=None)


def ask_agent(input_dict: dict, w: WorkspaceClient = None) -> requests.models.Response:
    # Example input_dict:
    # input_dict = {
    #     "input": [
    #         {"role": "user", "content": "What is the latest customer service request?"}
    #     ],
    #     "custom_inputs": {"thread_id": "1001"},
    # }
    url = f'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/{os.getenv("SERVING_ENDPOINT")}/invocations'
    if w is None:
        w = WorkspaceClient()
    headers = w.config.authenticate()
    response = requests.post(
        headers=headers,
        url=url,
        json=input_dict,
    )
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response


def ask_agent_mlflowclient(input_dict: dict, client) -> dict:
    # returns response.json()
    return client.predict(endpoint=os.getenv("SERVING_ENDPOINT"), inputs=input_dict)


def extract_text_content(response_json):
    # Extract text content from the response (equivalent to jq extraction)
    # jq -r '.output[] | select(.type == "message") | .content[] | .text'
    text_contents = []
    for output_item in response_json.get("output", []):
        if output_item.get("type") == "message":
            for content_item in output_item.get("content", []):
                if "text" in content_item:
                    text_contents.append(content_item["text"])
    return text_contents
