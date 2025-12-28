import logging
import asyncio
from typing import List, Optional, Union, Any, Dict
from google import genai
from google.genai import types
from .models import GeminiResponse
from .storage import BaseStorage, FileStorage
from .media import MediaManager

logger = logging.getLogger("gsdk")

class GeminiSDK:
    def __init__(
        self,
        api_keys: List[str],
        model_name: str = "gemini-flash-latest", 
        system_instruction: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        use_search: bool = True,
        max_retries: Optional[int] = None,
        retry_delay: float = 5.0,
        **generation_config
    ):
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.model_name = model_name
        self.storage = storage or FileStorage()
        self.system_instruction = system_instruction
        self.use_search = use_search
        self.max_retries = max_retries or (len(api_keys) * 3)
        self.retry_delay = retry_delay
        self.base_gen_config = generation_config
        self._init_client()

    def _init_client(self):
        # Using v1beta for tool support
        logger.info(f"Initializing client with key #{self.current_key_idx}")
        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_idx],
            http_options={'api_version': 'v1beta'}
        )
        self.media = MediaManager(self.client)

    def _prepare_config(self, tools: Optional[List[Any]] = None, **kwargs):
        """Merges global and request-specific configs. Fixes 400 error."""
        all_tools = []
        
        # If custom tools are provided, we temporarily disable search 
        # to prevent "Tool use unsupported" conflict in some regions/models.
        if tools:
            all_tools.extend(tools)
        elif self.use_search:
            all_tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Merge generation config
        merged_config = self.base_gen_config.copy()
        merged_config.update(kwargs)
        
        # Ensure we don't pass empty tools list
        tools_param = all_tools if all_tools else None

        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=tools_param,
            **merged_config
        )

    async def _rotate_key(self):
        if len(self.api_keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            logger.warning(f"Rotating to key #{self.current_key_idx}")
        else:
            await asyncio.sleep(self.retry_delay)
        self._init_client()

    async def ask(self, session_id: str, content: Any, tools: Optional[List[Any]] = None, _retry_count: int = 0, **kwargs) -> GeminiResponse:
        history = self.storage.get(session_id)
        user_parts = [types.Part.from_text(text=content)] if isinstance(content, str) else content
        
        try:
            current_message = types.Content(role="user", parts=user_parts)
            full_contents = history + [current_message]
            config = self._prepare_config(tools=tools, **kwargs)

            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=full_contents, config=config
            )

            if response.candidates:
                model_content = response.candidates[0].content
                if not model_content.role: model_content.role = "model"
                self.storage.set(session_id, full_contents + [model_content])

            return self._parse_res(response)

        except Exception as e:
            if _retry_count < self.max_retries and any(x in str(e) for x in ["429", "403", "503", "500", "quota"]):
                await self._rotate_key()
                return await self.ask(session_id, content, tools, _retry_count + 1, **kwargs)
            raise e

    async def ask_stream(self, session_id: str, content: Any, tools: Optional[List[Any]] = None, _retry_count: int = 0, **kwargs):
        history = self.storage.get(session_id)
        user_parts = [types.Part.from_text(text=content)] if isinstance(content, str) else content
        
        try:
            current_message = types.Content(role="user", parts=user_parts)
            full_contents = history + [current_message]
            config = self._prepare_config(tools=tools, **kwargs)

            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model_name, contents=full_contents, config=config
            )

            full_text = ""
            tool_calls = []

            async for chunk in response_stream:
                if chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if part.text:
                            full_text += part.text
                            yield part.text
                        if part.call:
                            tool_calls.append(part.call)
                            yield part.call # Yielding ToolCall object

            if full_text or tool_calls:
                parts = []
                if full_text: parts.append(types.Part.from_text(text=full_text))
                for call in tool_calls: parts.append(types.Part(call=call))
                model_content = types.Content(role="model", parts=parts)
                self.storage.set(session_id, full_contents + [model_content])

        except Exception as e:
            if _retry_count < self.max_retries and any(x in str(e) for x in ["429", "403", "503", "500", "quota"]):
                await self._rotate_key()
                async for chunk in self.ask_stream(session_id, content, tools, _retry_count + 1, **kwargs):
                    yield chunk
            else:
                raise e

    def _parse_res(self, raw) -> GeminiResponse:
        text, tool_calls = "", []
        if raw.candidates:
            for part in raw.candidates[0].content.parts:
                if part.text: text += part.text
                if part.call: tool_calls.append(part.call)
        return GeminiResponse(text=text, tool_calls=tool_calls, raw=raw)