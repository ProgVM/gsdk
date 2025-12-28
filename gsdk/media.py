import mimetypes
from pathlib import Path
from google import genai

class MediaManager:
    def __init__(self, client: genai.Client):
        self.client = client

    async def upload_file(self, file_path: str, mime_type: str = None):
        """
        Загружает файл на серверы Google.
        Возвращает объект файла, который можно передать в sdk.ask().
        """
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not mime_type:
            mime_type, _ = mimetypes.guess_type(path_obj)

        uploaded_file = await self.client.aio.files.upload(
            path=file_path,
            config={'mime_type': mime_type} if mime_type else None
        )

        return uploaded_file