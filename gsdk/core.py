import logging
import asyncio
from typing import List, Optional, Union, Any, Dict
from google import genai
from google.genai import types
from .models import GeminiResponse
from .storage import BaseStorage, FileStorage
from .media import MediaManager

# Unified logger for the entire SDK
logger = logging.getLogger("gsdk")

class GeminiSDK:
    def __init__(
        self,
        api_keys: List[str],
        model_name: str = "gemini-3-flash-preview", 
        system_instruction: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        use_search: bool = True,
        max_retries: Optional[int] = None,
        retry_delay: float = 5.0,
        **generation_config
    ):
        """
        Initialize GeminiSDK with full flexibility.
        
        :param api_keys: List of Google AI Studio API keys for rotation.
        :param model_name: Name of the model (e.g., 'gemini-3-flash-preview' or 'gemini-flash-latest').
        :param system_instruction: Global system prompt.
        :param storage: Custom storage instance (defaults to FileStorage).
        :param use_search: Enable/Disable Google Search grounding.
        :param max_retries: Attempts before giving up. Defaults to (keys count * 3).
        :param retry_delay: Delay in seconds between retries.
        :param generation_config: Global model parameters like temperature, top_p, max_output_tokens, etc.
        """
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.model_name = model_name
        self.storage = storage or FileStorage()
        self.system_instruction = system_instruction
        self.use_search = use_search
        
        self.max_retries = max_retries or (len(api_keys) * 3)
        self.retry_delay = retry_delay
        
        # Default generation parameters
        self.base_gen_config = generation_config
        self._init_client()

    def _init_client(self):
        """Re-initializes the GenAI client (used during key rotation)."""
        version = 'v1beta' 
        logger.info(f"Connecting to Gemini API with key index #{self.current_key_idx}")

        self.client = genai.Client(
            api_key=self.api_keys[self.current_key_idx],
            http_options={'api_version': version}
        )
        self.media = MediaManager(self.client)

        # Build tools list
        tools = [types.Tool(google_search=types.GoogleSearch())] if self.use_search else None

        # Build initial config
        self.config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=tools,
            **self.base_gen_config
        )

    async def ask(self, session_id: str, content: Any, _retry_count: int = 0, **request_kwargs) -> GeminiResponse:
        """
        Send a request to the model with session persistence and automatic retries.
        
        :param session_id: Unique ID for the conversation.
        :param content: String or list of Parts (text, image, file).
        :param request_kwargs: Override global generation config for this specific request.
        """
        if _retry_count >= self.max_retries:
             logger.error(f"Request failed after {self.max_retries} attempts.")
             raise Exception("Max retries reached. All API keys might be exhausted or IP blocked.")

        history = self.storage.get(session_id)

        # Convert input to appropriate Part objects
        if isinstance(content, str):
            user_parts = [types.Part.from_text(text=content)]
        elif isinstance(content, list):
            user_parts = content
        else:
            user_parts = [content]

        try:
            current_message = types.Content(role="user", parts=user_parts)
            full_contents = history + [current_message]

            # Merge global config with request-specific overrides
            merged_config_dict = self.config.model_dump()
            if request_kwargs:
                merged_config_dict.update(request_kwargs)
            
            # Clean up None values to avoid API errors
            final_config = types.GenerateContentConfig(**{k: v for k, v in merged_config_dict.items() if v is not None})

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_contents,
                config=final_config
            )

            # Update session history
            if response.candidates and response.candidates[0].content:
                model_content = response.candidates[0].content
                if not model_content.role: 
                    model_content.role = "model"
                self.storage.set(session_id, full_contents + [model_content])

            return self._parse_res(response)

        except Exception as e:
            error_msg = str(e).lower()
            # Retryable error codes
            if any(code in error_msg for code in ["429", "403", "503", "500", "quota", "exhausted"]):
                
                if len(self.api_keys) > 1:
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                    logger.warning(f"Rate limit hit. Rotating to key #{self.current_key_idx}...")
                else:
                    logger.warning(f"Rate limit hit. Sleeping for {self.retry_delay}s...")

                self._init_client()
                await asyncio.sleep(self.retry_delay)
                return await self.ask(session_id, content, _retry_count=_retry_count + 1, **request_kwargs)

            logger.error(f"Critical Gemini API Error: {e}")
            raise e

    async def ask_stream(self, session_id: str, content: Any, _retry_count: int = 0, **request_kwargs):
            """
            Send a request and yield response chunks in real-time.
            Automatically updates history once the stream is finished.
            """
            if _retry_count >= self.max_retries:
                 logger.error(f"Stream request failed after {self.max_retries} attempts.")
                 raise Exception("Max retries reached during streaming request.")

            history = self.storage.get(session_id)

            if isinstance(content, str):
                user_parts = [types.Part.from_text(text=content)]
            elif isinstance(content, list):
                user_parts = content
            else:
                user_parts = [content]

            try:
                current_message = types.Content(role="user", parts=user_parts)
                full_contents = history + [current_message]

                # Merge configs
                merged_config_dict = self.config.model_dump()
                if request_kwargs:
                    merged_config_dict.update(request_kwargs)
                
                final_config = types.GenerateContentConfig(**{k: v for k, v in merged_config_dict.items() if v is not None})

                # Start the stream
                response_stream = await self.client.aio.models.generate_content_stream(
                    model=self.model_name,
                    contents=full_contents,
                    config=final_config
                )

                full_text = ""
                async for chunk in response_stream:
                    chunk_text = chunk.text or ""
                    full_text += chunk_text
                    yield chunk_text # Yielding text chunk to the user

                # After stream finishes, save to history
                if full_text:
                    model_content = types.Content(role="model", parts=[types.Part.from_text(text=full_text)])
                    self.storage.set(session_id, full_contents + [model_content])

            except Exception as e:
                error_msg = str(e).lower()
                if any(code in error_msg for code in ["429", "403", "503", "500", "quota"]):
                    if len(self.api_keys) > 1:
                        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                        logger.warning(f"Rate limit during stream start. Rotating to key #{self.current_key_idx}...")
                    else:
                        logger.warning(f"Rate limit during stream start. Sleeping {self.retry_delay}s...")

                    self._init_client()
                    await asyncio.sleep(self.retry_delay)
                    
                    # Recursively call the same generator
                    async for chunk in self.ask_stream(session_id, content, _retry_count=_retry_count + 1, **request_kwargs):
                        yield chunk
                    return

                logger.error(f"Critical Stream Error: {e}")
                raise e

    def _parse_res(self, raw) -> GeminiResponse:
        """Parses the raw response into a GeminiResponse object with text and sources."""
        text = ""
        try:
            text = raw.text
        except:
            if raw.candidates and raw.candidates[0].content.parts:
                text = "".join([p.text for p in raw.candidates[0].content.parts if p.text])

        sources = []
        try:
            # Extract Google Search Grounding metadata
            gm = raw.candidates[0].grounding_metadata
            if gm.search_entry_point:
                sources.append(gm.search_entry_point.rendered_content)
            if gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if chunk.web:
                        sources.append(f"{chunk.web.title}: {chunk.web.uri}")
        except: 
            pass

        return GeminiResponse(text=text, sources=sources, raw=raw)