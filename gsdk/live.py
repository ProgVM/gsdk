import logging
from google import genai

logger = logging.getLogger("gsdk.live")

class GeminiLive:
    """Multimodal Live API for real-time interaction (WebSockets)."""
    def __init__(self, api_key: str, model_name: str = "gemini-flash-latest"):
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        self.model_name = model_name

    async def start_session(self):
        """Context manager for the live connection."""
        logger.info(f"Opening Live session with model: {self.model_name}")
        async with self.client.aio.live.connect(model=self.model_name) as session:
            yield session