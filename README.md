# gsdk 🚀

A lightweight Python library for the Google Gemini API.

## 📦 Installation
```bash
pip install gsdk
```

🚀 Quick Start
```python
import asyncio
from gsdk import GeminiSDK

async def main():
    sdk = GeminiSDK(
        api_keys=["YOUR_API_KEY"],
        model_name="gemini-3-flash-preview"
    )

    response = await sdk.ask("session_1", "Hello! Who are you?")
    print(f"AI: {response.text}")

asyncio.run(main())
```

🛠 Project Structure

- `gsdk.core` — Main SDK class and logic.

- `gsdk.media` — File uploads (images, video).

- `gsdk.storage` — Session persistence.

- `gsdk.live` — Real-time Multimodal API.


⚠️ Requirements

- Python 3.10+

- `google-genai` library (installed automatically)