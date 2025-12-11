import os
import sys
import logging
import streamlit as st
from uuid import uuid4
from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import get_user_info, ask_agent, ask_agent_mlflowclient, extract_text_content

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT")
assert SERVING_ENDPOINT, (
    "Unable to determine serving endpoint to use for chatbot app. If developing locally, "
    "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
    "deploying to a Databricks app, include a serving endpoint resource named "
    "'serving_endpoint' with CAN_QUERY permissions, as described in "
    "https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app#deploy-the-databricks-app"
)

w = WorkspaceClient()
client = get_deploy_client("databricks")
user_info = get_user_info()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.sidebar.title(
    "🛠️ AATWW: Agent and the Whole Works"
)
st.sidebar.markdown("""E2E Multi-Agent Supervisor with:
- AI/BI Genie (customer requests)
- Vector Search (product documents)
- SQL/Python functions
- external MCP (Github tools)
- custom MCP (weather/news tools hosted on Apps)
- external API (Product recalls)
- LakeBase (memory)
- Databricks Apps""")

# Initialize chat history
if "user_id" not in st.session_state:
    st.session_state.user_id = user_info.get("user_id")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())


# Add reset button
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid4())
    st.rerun()

# Persist chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["input"][0]["role"]):
        st.markdown(message["input"][0]["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    with st.spinner("Thinking..."):
        # Add user message to chat history
        # If using requests, input_dict is expected
        st.session_state.messages.append(
            {
                "input": [{"role": "user", "content": prompt}],
                "custom_inputs": {"thread_id": st.session_state.thread_id},
                "databricks_options": {"return_trace": True},
            }
        )

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            tool_call = None
            # Query the Databricks serving endpoint
            messages = st.session_state.messages[-1]
            print(messages)
            response_json = ask_agent_mlflowclient(
                input_dict=messages, client=client
            )
            text_contents = extract_text_content(response_json)
            custom_outputs = response_json.get("custom_outputs", {})
            tool_call = response_json.get("output", {})[0].get("name", '').replace("transfer_to_", "")
            if len(text_contents) > 0:
                assistant_response = text_contents[0]
            else:
                assistant_response = "No response returned. Try again"
            st.markdown(assistant_response)


        # Add assistant response to chat history
        st.session_state.messages.append(
            {
                "input": [{"role": "assistant", "content": assistant_response}],
                "custom_inputs": custom_outputs,
                "databricks_options": {"return_trace": True},
            }
        )
