import logging
import asyncio
from typing import List, Optional, Union, Any, Dict
from google import genai
from google.genai import types
from .models import GeminiResponse
from .storage import BaseStorage, FileStorage
from .media import MediaManager

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GeminiSDK")

class GeminiSDK:
    def __init__(
        self,
        api_keys: List[str],
        model_name: str = "gemini-3-flash-preview", 
        system_instruction: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        use_search: bool = True
    ):
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.model_name = model_name
        self.storage = storage or FileStorage()
        self.system_instruction = system_instruction
        self.use_search = use_search
        self._init_client()

    def _init_client(self):
        version = 'v1beta' 

        logger.info(f"Initializing client with key #{self.current_key_idx} (API: {version})")

        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_idx],
            http_options={'api_version': version}
        )
        self.media = MediaManager(self.client)

        tools = [types.Tool(google_search=types.GoogleSearch())] if self.use_search else None

        self.config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=tools,
            temperature=0.7
        )

        tools = [types.Tool(google_search=types.GoogleSearch())] if self.use_search else None

        self.config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=tools,
            temperature=0.7
        )

    async def ask(self, session_id: str, content: Any, _retry_depth: int = 0, **kwargs) -> GeminiResponse:

        max_retries = len(self.api_keys) * 3

        if _retry_depth > max_retries:
             raise Exception(f"❌ Gave up after {_retry_depth} tries. All keys/IP blocked.")

        history = self.storage.get(session_id)

        if isinstance(content, str):
            user_parts = [types.Part.from_text(text=content)]
        else:
            user_parts = content if isinstance(content, list) else [content]

        try:
            current_message = types.Content(role="user", parts=user_parts)
            full_contents = history + [current_message]

            current_config = self.config
            if kwargs:
                config_dict = self.config.model_dump()
                config_dict.update(kwargs)
                current_config = types.GenerateContentConfig(**config_dict)

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_contents,
                config=current_config
            )

            if response.candidates and response.candidates[0].content:
                model_content = response.candidates[0].content
                if not model_content.role: model_content.role = "model"
                self.storage.set(session_id, full_contents + [model_content])

            return self._parse_res(response)

        except Exception as e:
            error_str = str(e)

            # Ловим 429, 403, 503
            if any(code in error_str for code in ["429", "403", "503"]):
                import asyncio

                # Если ключей много
                if len(self.api_keys) > 1:
                    logger.warning(f"⚠️ Key #{self.current_key_idx} blocked. Rotating in 5s...")

                    # МЕНЯЕМ КЛЮЧ
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                    self._init_client()

                    # ЖДЕМ 5 СЕКУНД (чтобы IP остыл)
                    await asyncio.sleep(5) 

                    return await self.ask(session_id, content, _retry_depth=_retry_depth + 1, **kwargs)

                else:
                    wait_time = 5 * (_retry_depth + 1)
                    logger.warning(f"⚠️ Limit hit. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    return await self.ask(session_id, content, _retry_depth=_retry_depth + 1, **kwargs)

            logger.error(f"❌ Critical API Error: {error_str}")
            raise e

    def _parse_res(self, raw) -> GeminiResponse:
        try:
            text = raw.text or ""
        except:
            text = "".join([p.text for p in raw.candidates[0].content.parts if p.text])

        sources = []
        try:
            gm = raw.candidates[0].grounding_metadata
            if gm.search_entry_point:
                sources.append(gm.search_entry_point.rendered_content)
            if gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if chunk.web:
                        sources.append(f"{chunk.web.title}: {chunk.web.uri}")
        except: pass

        return GeminiResponse(text=text, sources=sources, raw=raw)