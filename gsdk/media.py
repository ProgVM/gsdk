import mimetypes
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger("gsdk.media")

class MediaManager:
    """Handles asynchronous file uploads to Gemini's file service."""
    def __init__(self, client: genai.Client):
        self.client = client

    async def upload_file(self, file_path: str, mime_type: str = None):
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not mime_type:
            mime_type, _ = mimetypes.guess_type(path_obj)

        logger.info(f"Uploading file '{path_obj.name}'...")
        uploaded_file = await self.client.aio.files.upload(
            path=file_path,
            config={'mime_type': mime_type} if mime_type else None
        )
        logger.info(f"Upload complete. URI: {uploaded_file.uri}")
        return uploaded_file