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
        model_name: str = "gemini-2.0-flash-exp", 
        system_instruction: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        use_search: bool = True,
        max_retries: Optional[int] = None,
        retry_delay: float = 5.0,
        **global_config
    ):
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.model_name = model_name
        self.storage = storage or FileStorage()
        self.system_instruction = system_instruction
        self.use_search = use_search
        self.max_retries = max_retries or (len(api_keys) * 3)
        self.retry_delay = retry_delay
        self.global_config = global_config
        self._init_client()

    def _init_client(self):
        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_idx],
            http_options={'api_version': 'v1beta'}
        )
        self.media = MediaManager(self.client)

    def _prepare_config(self, tools: Optional[List[Any]] = None, **kwargs):
        final_tools = tools if tools else ([types.Tool(google_search=types.GoogleSearch())] if self.use_search else None)
        conf = self.global_config.copy()
        conf.update(kwargs)
        return types.GenerateContentConfig(system_instruction=self.system_instruction, tools=final_tools, **conf)

    async def _rotate_key(self):
        if len(self.api_keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        else:
            await asyncio.sleep(self.retry_delay)
        self._init_client()

    async def _summarize_context(self, session_id: str):
        history = self.storage.get(session_id)
        if len(history) < 6: return
        try:
            res = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=f"Summarize this: {str(history[:-2])}")])]
            )
            new_hist = [types.Content(role="user", parts=[types.Part.from_text(text=f"[Summary]: {res.text}")]), *history[-2:]]
            self.storage.set(session_id, new_hist)
        except: pass

    async def ask(self, session_id: str, content: Any, tools: Optional[List[Any]] = None, _retry: int = 0, **kwargs):
        history = self.storage.get(session_id)
        parts = [types.Part.from_text(text=content)] if isinstance(content, str) else (content if isinstance(content, list) else [content])
        try:
            full_contents = history + [types.Content(role="user", parts=parts)]
            response = await self.client.aio.models.generate_content(model=self.model_name, contents=full_contents, config=self._prepare_config(tools, **kwargs))
            if response.candidates:
                mc = response.candidates[0].content
                if not mc.role: mc.role = "model"
                self.storage.set(session_id, full_contents + [mc])
            return self._parse_res(response)
        except Exception as e:
            if "context" in str(e).lower(): await self._summarize_context(session_id); return await self.ask(session_id, content, tools, _retry, **kwargs)
            if _retry < self.max_retries: await self._rotate_key(); return await self.ask(session_id, content, tools, _retry + 1, **kwargs)
            raise e

    async def ask_stream(self, session_id: str, content: Any, tools: Optional[List[Any]] = None, _retry: int = 0, **kwargs):
        history = self.storage.get(session_id)
        parts = [types.Part.from_text(text=content)] if isinstance(content, str) else (content if isinstance(content, list) else [content])
        try:
            full_contents = history + [types.Content(role="user", parts=parts)]
            response_stream = await self.client.aio.models.generate_content_stream(model=self.model_name, contents=full_contents, config=self._prepare_config(tools, **kwargs))
            txt, col = "", []
            async for chunk in response_stream:
                if chunk.candidates:
                    for p in chunk.candidates[0].content.parts:
                        if p.text: txt += p.text; yield p.text
                        if p.function_call: col.append(p); yield p.function_call
            if txt or col:
                h_parts = [types.Part.from_text(text=txt)] if txt else []
                h_parts.extend(col)
                self.storage.set(session_id, full_contents + [types.Content(role="model", parts=h_parts)])
        except Exception as e:
            if "context" in str(e).lower(): await self._summarize_context(session_id); async for c in self.ask_stream(session_id, content, tools, _retry, **kwargs): yield c
            elif _retry < self.max_retries: await self._rotate_key(); async for c in self.ask_stream(session_id, content, tools, _retry + 1, **kwargs): yield c
            else: raise e

    def _parse_res(self, raw) -> GeminiResponse:
        res = GeminiResponse(raw=raw)
        if raw.candidates:
            c = raw.candidates[0]
            for p in c.content.parts:
                if p.text: res.text += p.text
                if p.function_call: res.tool_calls.append(p.function_call)
                if p.inline_data: res.audio_bytes = p.inline_data.data
        return res