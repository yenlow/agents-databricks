# this script wires up LLM + MCP + tooling, enabling an agent that can take real actions on Databricks, not just generate text.

import asyncio

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient, DatabricksOAuthClientProvider
from langchain_core.tools import BaseTool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client as connect
from pydantic import create_model

nest_asyncio.apply()

###############################################################################
## Configure MCP Servers for your agent
##
## This section sets up server connections so your agent can retrieve data or take actions.

## There are three connection types:
## 1. Managed MCP servers — fully managed by Databricks (no setup required)
## 2. External MCP servers — hosted outside Databricks but proxied through a
##    Managed MCP server proxy (some setup required)
## 3. Custom MCP servers — MCP servers hosted as Databricks Apps (OAuth setup required)
##
## Note: External MCP servers get added to the "managed" URL list
## because their proxy endpoints are managed by Databricks.
###############################################################################

# ---------------------------------------------------------------------------
# Managed MCP Server — simplest setup
# ---------------------------------------------------------------------------
# Databricks manages this connection automatically using your workspace settings
# and Personal Access Token (PAT) authentication.

workspace_client = WorkspaceClient()

host = workspace_client.config.host

# Managed MCP Servers URLS (includes both fully managed and proxied external MCP)
# - If you're using an external MCP server, create a UC connection and flag it
#   as an MCP connection. This reveals a proxy endpoint.
# - Add that proxy endpoint URL to this list.

MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/functions/system/ai",  # Default managed MCP endpoint
    # Example for external MCP:
    # "https://<workspace-hostname>/api/2.0/mcp/external/{connection_name}"
]

# ---------------------------------------------------------------------------
# Custom MCP Server — hosted as a Databricks App
# ---------------------------------------------------------------------------
# Use this if you’re running your own MCP server in Databricks.
# These require OAuth with a service principal for machine-to-machine (M2M) auth.
#
# Custom MCP Servers — add URLs below (not managed or proxied by Databricks)
CUSTOM_MCP_SERVER_URLS = [
    "https://mcp-nitin-1444828305810485.aws.databricksapps.com/mcp"
]


#####################
## MCP Tool Creation
#####################


# Define a custom LangChain tool that wraps functionality for calling MCP servers
class MCPTool(BaseTool):
    """Custom LangChain tool that wraps MCP server functionality"""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type,
        server_url: str,
        ws: WorkspaceClient,
        is_custom: bool = False,
    ):
        # Initialize the tool
        super().__init__(name=name, description=description, args_schema=args_schema)
        # Store custom attributes: MCP server URL, Databricks workspace client, and whether the tool is for a custom server
        object.__setattr__(self, "server_url", server_url)
        object.__setattr__(self, "workspace_client", ws)
        object.__setattr__(self, "is_custom", is_custom)

    def _run(self, **kwargs) -> str:
        """Execute the MCP tool"""
        if self.is_custom:
            # Use the async method for custom MCP servers (OAuth required)
            return asyncio.run(self._run_custom_async(**kwargs))
        else:
            # Use managed MCP server via synchronous call
            mcp_client = DatabricksMCPClient(
                server_url=self.server_url, workspace_client=self.workspace_client
            )
            response = mcp_client.call_tool(self.name, kwargs)
            return "".join([c.text for c in response.content])

    async def _run_custom_async(self, **kwargs) -> str:
        """Execute custom MCP tool asynchronously"""
        async with connect(
            self.server_url, auth=DatabricksOAuthClientProvider(self.workspace_client)
        ) as (
            read_stream,
            write_stream,
            _,
        ):
            # Create an async session with the server and call the tool
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(self.name, kwargs)
                return "".join([c.text for c in response.content])


# Retrieve tool definitions from a custom MCP server (OAuth required)
async def get_custom_mcp_tools(ws: WorkspaceClient, server_url: str):
    """Get tools from a custom MCP server using OAuth"""
    async with connect(server_url, auth=DatabricksOAuthClientProvider(ws)) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            return tools_response.tools


# Retrieve tool definitions from a managed MCP server
def get_managed_mcp_tools(ws: WorkspaceClient, server_url: str):
    """Get tools from a managed MCP server"""
    mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
    return mcp_client.list_tools()


# Convert an MCP tool definition into a LangChain-compatible tool
def create_langchain_tool_from_mcp(
    mcp_tool, server_url: str, ws: WorkspaceClient, is_custom: bool = False
):
    """Create a LangChain tool from an MCP tool definition"""
    schema = mcp_tool.inputSchema.copy()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Map JSON schema types to Python types for input validation
    TYPE_MAPPING = {"integer": int, "number": float, "boolean": bool}
    field_definitions = {}
    for field_name, field_info in properties.items():
        field_type_str = field_info.get("type", "string")
        field_type = TYPE_MAPPING.get(field_type_str, str)

        if field_name in required:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (field_type, None)

    # Dynamically create a Pydantic schema for the tool's input arguments
    args_schema = create_model(f"{mcp_tool.name}Args", **field_definitions)

    # Return a configured MCPTool instance
    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"Tool: {mcp_tool.name}",
        args_schema=args_schema,
        server_url=server_url,
        ws=ws,
        is_custom=is_custom,
    )


# Gather all tools from managed and custom MCP servers into a single list
async def create_mcp_tools(
    ws: WorkspaceClient,
    managed_server_urls: list[str] = None,
    custom_server_urls: list[str] = None,
) -> list[MCPTool]:
    """Create LangChain tools from both managed and custom MCP servers"""
    tools = []

    if managed_server_urls:
        # Load managed MCP tools
        for server_url in managed_server_urls:
            try:
                mcp_tools = get_managed_mcp_tools(ws, server_url)
                for mcp_tool in mcp_tools:
                    tool = create_langchain_tool_from_mcp(
                        mcp_tool, server_url, ws, is_custom=False
                    )
                    tools.append(tool)
            except Exception as e:
                print(f"Error loading tools from managed server {server_url}: {e}")

    if custom_server_urls:
        # Load custom MCP tools (async)
        for server_url in custom_server_urls:
            try:
                mcp_tools = await get_custom_mcp_tools(ws, server_url)
                for mcp_tool in mcp_tools:
                    tool = create_langchain_tool_from_mcp(
                        mcp_tool, server_url, ws, is_custom=True
                    )
                    tools.append(tool)
            except Exception as e:
                print(f"Error loading tools from custom server {server_url}: {e}")

    return tools
