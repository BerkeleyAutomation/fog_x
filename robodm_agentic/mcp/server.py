"""Model Context Protocol (MCP) server for RoboDM agentic framework.

The Model Context Protocol is a standardized way for AI assistants to connect
with external data sources and tools. This server exposes RoboDM functionality
as MCP tools and resources.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    # Create mock types for when MCP is not available
    class MockTypes:
        class Tool:
            def __init__(self, name: str, description: str, inputSchema: Dict):
                self.name = name
                self.description = description
                self.inputSchema = inputSchema

        class Resource:
            def __init__(self, uri: str, name: str, description: str, mimeType: str = "text/plain"):
                self.uri = uri
                self.name = name
                self.description = description
                self.mimeType = mimeType

        class TextContent:
            def __init__(self, type: str, text: str):
                self.type = type
                self.text = text

        class CallToolResult:
            def __init__(self, content: List):
                self.content = content

        class ListResourcesResult:
            def __init__(self, resources: List):
                self.resources = resources

        class ReadResourceResult:
            def __init__(self, contents: List):
                self.contents = contents

    types = MockTypes()

from ..core.robodm_interface import RoboDMInterface


class RoboDMMCPServer:
    """MCP Server for RoboDM trajectory data access."""

    def __init__(self, robodm_interface: RoboDMInterface, server_name: str = "robodm-agentic"):
        """Initialize MCP server.

        Args:
            robodm_interface: RoboDM interface instance
            server_name: Name of the MCP server
        """
        self.robodm = robodm_interface
        self.server_name = server_name

        if not HAS_MCP:
            logger.warning("MCP package not available. Install with: pip install mcp")
            self.server = None
        else:
            self.server = Server(server_name)
            self._register_handlers()

    def _register_handlers(self):
        """Register MCP handlers for tools and resources."""
        if not self.server:
            return

        # Register tool handlers
        @self.server.list_tools()
        async def handle_list_tools() -> List['types.Tool']:
            """List available RoboDM tools."""
            tool_schemas = self.robodm.get_tool_schemas()
            return [
                types.Tool(
                    name=schema["name"],
                    description=schema["description"],
                    inputSchema=schema["parameters"],
                )
                for schema in tool_schemas
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> 'types.CallToolResult':
            """Handle tool calls."""
            try:
                if not hasattr(self.robodm, name):
                    raise ValueError(f"Unknown tool: {name}")

                method = getattr(self.robodm, name)
                result = method(**arguments)

                # Special handling for frame data for MCP
                if name == "get_trajectory_frames":
                    result = {
                        "trajectory_id": arguments.get("trajectory_id"),
                        "num_frames": len(result),
                        "frame_info": f"Found {len(result)} frames.",
                    }
                else:
                    # Serialize any numpy arrays for JSON transmission
                    result = self._serialize_data(result)

                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps(result, indent=2, default=str),
                        )
                    ]
                )

            except Exception as e:
                logger.error(f"Tool call failed: {name} - {e}")
                return types.CallToolResult(
                    content=[
                        types.TextContent(type="text", text=f"Error: {str(e)}")
                    ]
                )

        # Register resource handlers
        @self.server.list_resources()
        async def handle_list_resources() -> 'types.ListResourcesResult':
            """List available RoboDM resources."""
            resources = []

            try:
                # Get all trajectories as resources
                trajectories = self.robodm.get_all_trajectories()

                for traj_id in trajectories[:50]:  # Limit to first 50 for performance
                    resources.append(
                        types.Resource(
                            uri=f"robodm://trajectory/{traj_id}",
                            name=f"Trajectory {traj_id}",
                            description=f"RoboDM trajectory data for {traj_id}",
                            mimeType="application/json"
                        )
                    )

                # Add summary resource
                resources.append(
                    types.Resource(
                        uri="robodm://summary",
                        name="Database Summary",
                        description="Summary of the RoboDM database",
                        mimeType="application/json"
                    )
                )

            except Exception as e:
                logger.error(f"Error listing resources: {e}")

            return types.ListResourcesResult(resources=resources)

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> 'types.ReadResourceResult':
            """Read a specific resource."""
            try:
                if uri.startswith("robodm://trajectory/"):
                    # Extract trajectory ID from URI
                    traj_id = uri.split("/")[-1]
                    metadata = self.robodm.get_trajectory_metadata(traj_id)

                    content = types.TextContent(
                        type="text",
                        text=json.dumps(metadata, indent=2, default=str)
                    )

                elif uri == "robodm://summary":
                    # Provide database summary
                    all_trajectories = self.robodm.get_all_trajectories()
                    summary = {
                        "total_trajectories": len(all_trajectories),
                        "sample_trajectories": all_trajectories[:10],
                        "available_functions": list(self.robodm.get_available_functions().keys())
                    }

                    content = types.TextContent(
                        type="text",
                        text=json.dumps(summary, indent=2)
                    )

                else:
                    raise ValueError(f"Unknown resource URI: {uri}")

                return types.ReadResourceResult(contents=[content])

            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                error_content = types.TextContent(
                    type="text",
                    text=f"Error reading resource: {str(e)}"
                )
                return types.ReadResourceResult(contents=[error_content])

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON transmission."""
        if hasattr(data, 'tolist'):  # numpy array
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        else:
            return data

    async def run_stdio(self):
        """Run the MCP server over stdio."""
        if not self.server:
            raise RuntimeError("MCP server not available. Install mcp package.")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

    async def run_websocket(self, host: str = "localhost", port: int = 8000):
        """Run the MCP server over WebSocket."""
        if not self.server:
            raise RuntimeError("MCP server not available. Install mcp package.")

        # WebSocket server implementation would go here
        # This is a placeholder for now
        logger.info(f"WebSocket MCP server would run on ws://{host}:{port}")
        raise NotImplementedError("WebSocket server not yet implemented")

    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the MCP server."""
        return {
            "name": self.server_name,
            "mcp_available": HAS_MCP,
            "num_trajectories": len(self.robodm.get_all_trajectories()) if self.robodm else 0,
            "available_tools": list(self.robodm.get_available_functions().keys()) if self.robodm else [],
        }
