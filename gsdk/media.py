import time
from google import genai
from google.genai import types

class MediaManager:
    def __init__(self, client: genai.Client):
        self.client = client

    async def upload(self, path: str, mime_type: str = None):
        file = self.client.files.upload(path=path, config=types.UploadFileConfig(mime_type=mime_type))

        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = self.client.files.get(name=file.name)

        if file.state.name == "FAILED":
            raise Exception(f"File processing failed: {file.name}")

        return file