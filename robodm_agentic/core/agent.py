"""Main agentic interface for RoboDM trajectory querying."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from robodm_agentic.clients.llm_client import LLMClient
from robodm_agentic.clients.vlm_client import VLMClient
from robodm_agentic.core.robodm_interface import RoboDMInterface

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of an agentic query."""
    query: str
    tool_call: Dict[str, Any]
    tool_result: Any
    frames: List[Any]
    answer: str
    success: bool
    error: Optional[str] = None


class RoboDMAgent:
    """Main agent for querying RoboDM trajectories using natural language."""

    def __init__(self,
                 robodm_interface: RoboDMInterface,
                 llm_client: Optional[LLMClient] = None,
                 vlm_client: Optional[VLMClient] = None,
                 enable_vision: bool = True):
        """Initialize the RoboDM agent.

        Args:
            robodm_interface: Interface to RoboDM data
            llm_client: LLM client for tool-call generation
            vlm_client: VLM client for visual analysis
            enable_vision: Whether to enable vision analysis
        """
        self.robodm = robodm_interface
        self.llm = llm_client or LLMClient()
        self.vlm = vlm_client if enable_vision else None
        if enable_vision and vlm_client is None:
            try:
                self.vlm = VLMClient()
            except Exception as e:
                logger.warning(f"Could not initialize VLM client: {e}")
                self.vlm = None

        self.logger = logging.getLogger(__name__)
        self.enable_vision = enable_vision and self.vlm is not None

    async def query(self, user_query: str, include_frames: bool = None) -> QueryResult:
        """Process a natural language query about trajectories.

        Args:
            user_query: Natural language query
            include_frames: Whether to extract and analyze frames (auto-detect if None)

        Returns:
            QueryResult with all execution details and answer
        """
        self.logger.info(f"Processing query: {user_query}")

        if include_frames is None:
            include_frames = self._should_include_frames(user_query)

        try:
            # Step 1: Generate a tool call from the LLM
            self.logger.info("Generating tool call with LLM...")
            tools = self.robodm.get_tool_schemas()
            tool_call = await self.llm.generate_tool_call(user_query, tools)

            # Step 2: Execute the tool call
            self.logger.info(f"Executing tool call: {tool_call}")
            tool_result, success, error = self._execute_tool_call(tool_call)

            # Step 3: Extract frames if needed
            frames = []
            if include_frames and self.enable_vision and success:
                frames = self._extract_frames_from_result(tool_result)

            # Step 4: Generate the final answer
            if self.enable_vision and self.vlm and (frames or success):
                self.logger.info("Analyzing with VLM...")
                answer = await self.vlm.analyze_and_answer(user_query, frames, tool_result)
            else:
                answer = self._generate_text_answer(user_query, tool_result, success, error)

            return QueryResult(
                query=user_query,
                tool_call=tool_call,
                tool_result=tool_result,
                frames=frames,
                answer=answer,
                success=success,
                error=error
            )

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return QueryResult(
                query=user_query,
                tool_call={},
                tool_result=None,
                frames=[],
                answer=f"Query failed: {str(e)}",
                success=False,
                error=str(e)
            )

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[Any, bool, Optional[str]]:
        """Safely execute the tool call against the RoboDM interface."""
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})

        if not tool_name or not hasattr(self.robodm, tool_name):
            error_msg = f"Invalid tool name: {tool_name}"
            self.logger.error(error_msg)
            return None, False, error_msg

        try:
            method = getattr(self.robodm, tool_name)
            result = method(**arguments)
            return result, True, None
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e}"
            self.logger.error(error_msg)
            return None, False, error_msg

    def _should_include_frames(self, query: str) -> bool:
        """Determine if query likely needs visual analysis."""
        vision_keywords = [
            'frame', 'image', 'visual', 'see', 'look', 'appearance',
            'color', 'object', 'robot', 'action', 'movement', 'gesture',
            'hidden', 'occluded', 'view', 'camera', 'scene', 'environment'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in vision_keywords)

    def _extract_frames_from_result(self, tool_result: Any) -> List[Any]:
        """Extract visual frames from the result of a tool call."""
        frames = []
        if not tool_result:
            return frames

        # If the result is a list of trajectory IDs, get frames from them
        if isinstance(tool_result, list) and all(isinstance(i, str) for i in tool_result):
            sample_trajectories = tool_result[:3]  # Limit to first 3 for performance
            for traj_id in sample_trajectories:
                try:
                    traj_frames = self.robodm.get_trajectory_frames(traj_id)
                    if traj_frames:
                        # Take a few frames from each trajectory
                        sample_frames = traj_frames[:5] if len(traj_frames) > 5 else traj_frames
                        frames.extend(sample_frames)
                except Exception as e:
                    self.logger.warning(f"Could not get frames from {traj_id}: {e}")
        # If the result itself is a list of frames
        elif isinstance(tool_result, list) and tool_result and not isinstance(tool_result[0], str):
            frames = tool_result

        return frames

    def _generate_text_answer(self, query: str, tool_result: Any, success: bool, error: Optional[str]) -> str:
        """Generate text-only answer from the tool result."""
        if not success:
            return f"Query execution failed: {error or 'Unknown error'}"

        answer_parts = []

        if tool_result is not None:
            if isinstance(tool_result, list):
                answer_parts.append(f"Found {len(tool_result)} items matching your query.")
                if tool_result and len(tool_result) <= 10:
                    # Truncate long item representations
                    items_str = ", ".join(str(item)[:100] for item in tool_result)
                    answer_parts.append("Items: " + items_str)
            elif isinstance(tool_result, (int, float, str)):
                answer_parts.append(f"Result: {tool_result}")
            elif isinstance(tool_result, dict):
                 answer_parts.append(f"Result: {json.dumps(tool_result, indent=2)}")
            else:
                answer_parts.append(f"Query executed successfully.")
        else:
            answer_parts.append("Query executed successfully with no specific output.")

        return "\n".join(answer_parts)

    async def batch_query(self, queries: List[str]) -> List[QueryResult]:
        """Process multiple queries in parallel."""
        self.logger.info(f"Processing {len(queries)} queries in batch")

        tasks = [self.query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(QueryResult(
                    query=queries[i],
                    tool_call={},
                    tool_result=None,
                    frames=[],
                    answer=f"Batch query failed: {result}",
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    async def test_setup(self) -> Dict[str, bool]:
        """Test all components to ensure they're working."""
        results = {
            'robodm_interface': False,
            'llm_client': False,
            'vlm_client': False,
        }

        # Test RoboDM interface
        try:
            trajectories = self.robodm.get_all_trajectories()
            results['robodm_interface'] = len(trajectories) >= 0
        except Exception as e:
            self.logger.error(f"RoboDM interface test failed: {e}")

        # Test LLM client
        try:
            results['llm_client'] = await self.llm.test_connection()
        except Exception as e:
            self.logger.error(f"LLM client test failed: {e}")

        # Test VLM client
        if self.vlm:
            try:
                results['vlm_client'] = await self.vlm.test_connection()
            except Exception as e:
                self.logger.error(f"VLM client test failed: {e}")

        return results

    def close(self):
        """Close all resources."""
        if hasattr(self.robodm, 'close_all'):
            self.robodm.close_all()
