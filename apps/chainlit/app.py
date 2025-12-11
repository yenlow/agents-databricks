import os
import sys
import logging
import chainlit as cl
from uuid import uuid4
from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import ask_agent_mlflowclient, extract_text_content, get_user_info_from_headers

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

# Initialize clients
w = WorkspaceClient()
client = get_deploy_client("databricks")


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with user info and thread ID."""
    # Get user info from request headers if available
    user_info = {}
    try:
        # Try to get headers from Chainlit context
        headers = cl.context.session.http_referer if hasattr(cl.context.session, 'http_referer') else {}
        user_info = get_user_info_from_headers(headers)
    except Exception as e:
        logger.warning(f"Could not get user info from headers: {e}")
        user_info = {"user_name": None, "user_email": None, "user_id": None}

    # Initialize session variables
    cl.user_session.set("user_id", user_info.get("user_id"))
    cl.user_session.set("messages", [])
    cl.user_session.set("thread_id", str(uuid4()))

    # Send welcome message
    await cl.Message(
        content="""Welcome to **AATWW: Agent and the Whole Works**

E2E Multi-Agent Supervisor with:
- AI/BI Genie (customer requests)
- Vector Search (product documents)
- SQL/Python functions
- External MCP (Github tools)
- Custom MCP (weather/news tools hosted on Apps)
- External API (Product recalls)
- LakeBase (memory)
- Databricks Apps

Type your question below to get started!"""
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages."""
    prompt = message.content

    # Get session variables
    messages = cl.user_session.get("messages", [])
    thread_id = cl.user_session.get("thread_id")

    # Build the input message for the agent
    input_message = {
        "input": [{"role": "user", "content": prompt}],
        "custom_inputs": {"thread_id": thread_id},
        "databricks_options": {"return_trace": True},
    }

    # Add user message to history
    messages.append(input_message)
    cl.user_session.set("messages", messages)

    # Create a placeholder message for streaming effect
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Query the Databricks serving endpoint
        response_json = ask_agent_mlflowclient(
            input_dict=input_message,
            client=client
        )

        # Extract text content from response
        text_contents = extract_text_content(response_json)
        custom_outputs = response_json.get("custom_outputs", {})

        # Get the assistant response
        if len(text_contents) > 0:
            assistant_response = text_contents[0]
        else:
            assistant_response = "No response returned. Try again"

        # Update the message with the response
        msg.content = assistant_response
        await msg.update()

        # Add assistant response to history
        messages.append({
            "input": [{"role": "assistant", "content": assistant_response}],
            "custom_inputs": custom_outputs,
            "databricks_options": {"return_trace": True},
        })
        cl.user_session.set("messages", messages)

    except Exception as e:
        logger.error(f"Error calling agent: {e}")
        msg.content = f"Error: {str(e)}"
        await msg.update()


@cl.action_callback("reset_chat")
async def on_reset_chat(action: cl.Action):
    """Handle reset chat action."""
    # Clear messages and generate new thread ID
    cl.user_session.set("messages", [])
    cl.user_session.set("thread_id", str(uuid4()))

    await cl.Message(content="Chat has been reset. You can start a new conversation.").send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Handle chat resume for persistent sessions."""
    # Restore session state from thread if available
    if thread.get("metadata"):
        cl.user_session.set("messages", thread["metadata"].get("messages", []))
        cl.user_session.set("thread_id", thread["metadata"].get("thread_id", str(uuid4())))
