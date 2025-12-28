from google import genai

class GeminiLive:
    """New Live Multimodal API (WebSockets/Bidi)"""
    def __init__(self, api_key: str, model_name: str = "gemini-3.0-flash"):
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self.model_name = model_name

    async def start_session(self):
        """Context manager for live connection."""
        async with self.client.aio.live.connect(model=self.model_name) as session:
            yield session