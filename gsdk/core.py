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
        version = 'v1beta' 
        logger.info(f"Initializing client with key #{self.current_key_idx}")
        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_idx],
            http_options={'api_version': version}
        )
        self.media = MediaManager(self.client)

    def _prepare_config(self, tools: Optional[List[Any]] = None, **kwargs):
        """Internal: Merges global and request-specific configs."""
        all_tools = []
        if self.use_search:
            all_tools.append(types.Tool(google_search=types.GoogleSearch()))
        if tools:
            all_tools.extend(tools)

        merged_config = self.base_gen_config.copy()
        merged_config.update(kwargs)
        
        return types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=all_tools if all_tools else None,
            **merged_config
        )

    async def ask(self, session_id: str, content: Any, tools: Optional[List[Any]] = None, _retry_count: int = 0, **kwargs) -> GeminiResponse:
        if _retry_count >= self.max_retries:
             raise Exception("Max retries reached.")

        history = self.storage.get(session_id)
        user_parts = [types.Part.from_text(text=content)] if isinstance(content, str) else content
        
        try:
            current_message = types.Content(role="user", parts=user_parts)
            full_contents = history + [current_message]
            config = self._prepare_config(tools=tools, **kwargs)

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_contents,
                config=config
            )

            if response.candidates:
                model_content = response.candidates[0].content
                if not model_content.role: model_content.role = "model"
                self.storage.set(session_id, full_contents + [model_content])

            return self._parse_res(response)

        except Exception as e:
            return await self._handle_error(e, session_id, content, tools, _retry_count, is_stream=False, **kwargs)

    async def ask_stream(self, session_id: str, content: Any, tools: Optional[List[Any]] = None, _retry_count: int = 0, **kwargs):
        history = self.storage.get(session_id)
        user_parts = [types.Part.from_text(text=content)] if isinstance(content, str) else content
        
        try:
            current_message = types.Content(role="user", parts=user_parts)
            full_contents = history + [current_message]
            config = self._prepare_config(tools=tools, **kwargs)

            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=full_contents,
                config=config
            )

            full_text = ""
            tool_calls = []

            async for chunk in response_stream:
                if chunk.candidates:
                    part = chunk.candidates[0].content.parts[0]
                    if part.text:
                        full_text += part.text
                        yield part.text
                    if part.call:
                        tool_calls.append(part.call)
                        yield part.call

            if full_text or tool_calls:
                # Save both text and tool calls to history
                parts = []
                if full_text: parts.append(types.Part.from_text(text=full_text))
                for call in tool_calls: parts.append(types.Part(call=call))
                
                model_content = types.Content(role="model", parts=parts)
                self.storage.set(session_id, full_contents + [model_content])

        except Exception as e:
            # For stream errors, rotation logic is handled inside handle_error
            async for res in self._handle_error(e, session_id, content, tools, _retry_count, is_stream=True, **kwargs):
                yield res

    async def _handle_error(self, e, session_id, content, tools, _retry_count, is_stream=False, **kwargs):
        error_msg = str(e).lower()
        retryable = ["429", "403", "503", "500", "quota", "exhausted"]
        
        if any(code in error_msg for code in retryable) and _retry_count < self.max_retries:
            if len(self.api_keys) > 1:
                self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                logger.warning(f"Rotating to key #{self.current_key_idx}")
            else:
                await asyncio.sleep(self.retry_delay)
            
            self._init_client()
            if is_stream:
                async for chunk in self.ask_stream(session_id, content, tools, _retry_count + 1, **kwargs):
                    yield chunk
            else:
                return await self.ask(session_id, content, tools, _retry_count + 1, **kwargs)
        raise e

    def _parse_res(self, raw) -> GeminiResponse:
        text = ""
        tool_calls = []
        if raw.candidates:
            for part in raw.candidates[0].content.parts:
                if part.text: text += part.text
                if part.call: tool_calls.append(part.call)

        sources = []
        try:
            gm = raw.candidates[0].grounding_metadata
            if gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if chunk.web: sources.append(f"{chunk.web.title}: {chunk.web.uri}")
        except: pass

        return GeminiResponse(text=text, tool_calls=tool_calls, sources=sources, raw=raw)