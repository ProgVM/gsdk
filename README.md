# gsdk 🚀

**gsdk** (Gemini SDK) is a lightweight, high-performance Python wrapper for the **Google Gemini API** (built on the modern `google-genai`). It is designed for production use, offering automatic key rotation, session persistence, and full flexibility for model configuration.

[![PyPI version](https://img.shields.io/pypi/v/gsdk.svg)](https://pypi.org/project/gsdk/)
[![Python versions](https://img.shields.io/pypi/pyversions/gsdk.svg)](https://pypi.org/project/gsdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🔑 **Smart Key Rotation**: Automatically switch between multiple API keys when you hit rate limits (429/403).
- 🔄 **Configurable Retries**: Set custom retry counts and delays to ensure stability.
- 💾 **Session Persistence**: Built-in support for **File** and **Redis** storage to maintain conversation history.
- ⚙️ **Full Flexibility**: Pass any generation parameter (`temperature`, `top_p`, `max_tokens`) globally or per request.
- 🔍 **Google Search Grounding**: Easily enable real-time web search for the model.
- 🎙️ **Live Multimodal**: Support for the Gemini Real-time (WebSockets) API.
- 📁 **Media Support**: Simplified async file uploads for images, video, and audio.

---

## 📦 Installation

```bash
pip install gsdk
```

Make sure you have the requirements installed:
```bash
pip install google-genai redis
```

---

## 🚀 Quick Start

```python
import asyncio
import logging
from gsdk import GeminiSDK

# Enable logging to see key rotation and API status
logging.basicConfig(level=logging.INFO)

async def main():
    # Initialize the SDK
    sdk = GeminiSDK(
        api_keys=["YOUR_API_KEY_1", "YOUR_API_KEY_2"],
        model_name="gemini-flash-latest",
        temperature=0.7 # Global setting
    )

    # Simple text request
    response = await sdk.ask("session_id_1", "Hello! Who are you?")
    print(f"AI: {response.text}")

    # Request with parameter override
    response = await sdk.ask(
        "session_id_1", 
        "Tell me a very short story.", 
        temperature=0.1,  # Overrides global 0.7
        max_output_tokens=100
    )
    print(f"AI: {response.text}")

asyncio.run(main())
```

---

## 🛠 Advanced Usage

### 1. Key Rotation & Retry Logic
Perfect for staying within free-tier limits. If one key hits a 429 error, **gsdk** waits and switches to the next one.

```python
sdk = GeminiSDK(
    api_keys=["KEY_1", "KEY_2", "KEY_3"],
    max_retries=10,
    retry_delay=5.0  # Seconds to wait before retry
)
```

### 2. Redis Storage (for Production)
Use Redis to share chat history across multiple server instances.

```python
from gsdk.storage import RedisStorage

redis_store = RedisStorage(host='localhost', port=6379, db=0)
sdk = GeminiSDK(api_keys=["..."], storage=redis_store)
```

### 3. Media Uploads (Multimodal)
```python
# Upload a file (Image/Video/PDF)
uploaded_file = await sdk.media.upload_file("path/to/image.jpg")

# Ask a question about the file
response = await sdk.ask("session_2", [uploaded_file, "What is in this image?"])
print(response.text)
```

### 4. Google Search Grounding
Enable the model to search the web for up-to-date information.

```python
sdk = GeminiSDK(api_keys=["..."], use_search=True)

response = await sdk.ask("news_chat", "What is the price of Bitcoin today?")
print(f"Source info: {response.sources}")
```

### 5. Real-time Live API
```python
from gsdk import GeminiLive

live = GeminiLive(api_key="YOUR_KEY")

async def run_live():
    async with live.start_session() as session:
        await session.send("Hello!", end_of_turn=True)
        async for message in session.receive():
            print(message)
```

---

## 📖 API Reference

### `GeminiSDK` Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_keys` | `List[str]` | Required | List of Google AI Studio API keys. |
| `model_name` | `str` | `gemini-3-flash-preview` | The Gemini model to use. |
| `system_instruction` | `str` | `None` | The system prompt for the model. |
| `storage` | `BaseStorage` | `FileStorage` | Storage engine for sessions. |
| `use_search` | `bool` | `True` | Enable Google Search grounding. |
| `max_retries` | `int` | `keys * 3` | Total retries before raising an error. |
| `retry_delay` | `float` | `5.0` | Seconds to wait between retries. |
| `**generation_config`| `kwargs` | `None` | Any parameter supported by Gemini (temperature, top_p, etc.). |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Developed with ❤️ for the Gemini Community.**