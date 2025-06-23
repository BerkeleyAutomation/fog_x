"""LLM client for function calling using various models."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class LLMClient:
    """Client for interacting with Language Models for tool calling."""

    def __init__(self,
                 model: str = "qwen2.5:7b",
                 provider: str = "ollama",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Initialize LLM client.

        Args:
            model: Model name/identifier
            provider: "ollama", "openai", or "anthropic"
            api_key: API key for hosted providers
            base_url: Base URL for API (for custom endpoints)
        """
        self.model = model
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url

        # Initialize client based on provider
        if self.provider == "ollama":
            if not HAS_OLLAMA:
                raise ImportError("ollama package not installed. Run: pip install ollama")
            self.client = ollama.Client(host=base_url) if base_url else ollama.Client()
        elif self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def generate_tool_call(self, user_query: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a tool call based on the user query and available tools."""

        system_prompt = self._build_tool_prompt(tools)

        try:
            if self.provider == "ollama":
                response = await self._call_ollama(system_prompt, user_query)
            elif self.provider == "openai":
                response = await self._call_openai(system_prompt, user_query)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            return self._extract_json(response)

        except Exception as e:
            logger.error(f"Error generating tool call: {e}")
            return {
                "tool_name": "error",
                "arguments": {"message": f"Failed to generate tool call: {e}"}
            }

    def _build_tool_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Build system prompt with available tools and instructions."""

        tools_doc = json.dumps(tools, indent=2)

        return f"""You are an expert at calling functions to answer user questions.
Given a user query, select the best function from the following list and return a JSON object with the function name and arguments.

Available Functions:
{tools_doc}

IMPORTANT RULES:
1. Respond with a single JSON object in the format: {{"tool_name": "<function_name>", "arguments": {{...}}}}
2. Do not include any other text, explanations, or markdown formatting.
3. If the user query doesn't seem to map to any function, you can use the "error" tool with a message.

Example Query: "how many trajectories failed?"
Example Response:
{{
  "tool_name": "count_trajectories",
  "arguments": {{
    "filter_query": {{"metadata.status": "failed"}}
  }}
}}

Example Query: "show me one trajectory"
Example Response:
{{
  "tool_name": "sample_trajectories",
  "arguments": {{
    "n": 1
  }}
}}
"""

    async def _call_ollama(self, system_prompt: str, user_query: str) -> str:
        """Call Ollama API."""
        try:
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                options={"temperature": 0.0} # For reproducibility
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    async def _call_openai(self, system_prompt: str, user_query: str) -> str:
        """Call OpenAI-compatible API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0,
                response_format={"type": "json_object"} # Use JSON mode if available
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON object from LLM response."""
        try:
            # Find the start and end of the JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from LLM response: {response}. Error: {e}")
            return {
                "tool_name": "error",
                "arguments": {"message": "Failed to parse JSON response from LLM."}
            }

    async def test_connection(self) -> bool:
        """Test if the LLM connection is working."""
        try:
            test_response = await self.generate_tool_call(
                "test connection",
                [{"name": "test_function", "description": "A test function", "parameters": {}}]
            )
            return "tool_name" in test_response
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
