# gsdk 🚀

**gsdk** (Gemini SDK) is a lightweight, high-performance Python wrapper for the **Google Gemini API** (built on the modern `google-genai`). It is designed for production use, offering automatic key rotation, session persistence, and real-time streaming.

[![PyPI version](https://img.shields.io/pypi/v/gsdk.svg)](https://pypi.org/project/gsdk/)
[![Python versions](https://img.shields.io/pypi/pyversions/gsdk.svg)](https://pypi.org/project/gsdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🔑 **Smart Key Rotation**: Automatically switch between multiple API keys when you hit rate limits (429/403).
- 🌊 **Streaming Support**: Real-time response generation with automatic history updates.
- 🔄 **Configurable Retries**: Set custom retry counts and delays for maximum stability.
- 💾 **Session Persistence**: Built-in support for **File** and **Redis** storage.
- ⚙️ **Full Flexibility**: Pass any generation parameter (`temperature`, `top_p`, `max_tokens`) globally or per request.
- 🔍 **Google Search Grounding**: Integrated real-time web search capabilities.
- 📁 **Media Support**: Simplified async file uploads for multimodal tasks.

---

## 📦 Installation

```bash
pip install gsdk
```

---

## 🚀 Quick Start

### Basic Chat
```python
import asyncio
from gsdk import GeminiSDK

async def main():
    sdk = GeminiSDK(api_keys=["YOUR_API_KEY"], model_name="gemini-flash-latest")

    response = await sdk.ask("session_1", "Hello! Who are you?")
    print(f"AI: {response.text}")

asyncio.run(main())
```

### 🌊 Real-time Streaming
Perfect for chat interfaces where you want to show text as it is generated.

```python
async def stream_example():
    sdk = GeminiSDK(api_keys=["KEY_1", "KEY_2"])
    
    print("AI: ", end="", flush=True)
    async for chunk in sdk.ask_stream("session_1", "Write a long poem about coding."):
        print(chunk, end="", flush=True)

asyncio.run(stream_example())
```

---

## 🛠 Advanced Usage

### 1. Production Storage (Redis)
Share session history across multiple workers or servers.

```python
from gsdk.storage import RedisStorage

storage = RedisStorage(host='localhost', port=6379)
sdk = GeminiSDK(api_keys=["..."], storage=storage)
```

### 2. Handling Images and Files
```python
# Upload image or document
media = await sdk.media.upload_file("chart.png")

# Multimodal request
response = await sdk.ask("session_2", [media, "Analyze this chart."])
print(response.text)
```

### 3. Google Search Grounding
```python
sdk = GeminiSDK(api_keys=["..."], use_search=True)

response = await sdk.ask("news", "What's happening in AI today?")
print(f"Sources used: {response.sources}")
```

---

## 📖 API Reference

### `GeminiSDK` Methods

| Method | Description |
|--------|-------------|
| `ask(session_id, content, **kwargs)` | Sends a message and returns a full `GeminiResponse`. |
| `ask_stream(session_id, content, **kwargs)` | **Async Generator**. Yields text chunks and saves history after completion. |
| `media.upload_file(path)` | Uploads a file to Google servers for multimodal input. |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_keys` | `List[str]` | Required | List of keys for rotation. |
| `model_name` | `str` | `gemini-3-flash-preview` | The Gemini model version. |
| `storage` | `BaseStorage` | `FileStorage` | How to store conversation history. |
| `max_retries` | `int` | `keys * 3` | Total attempts for failed requests. |
| `retry_delay` | `float` | `5.0` | Seconds to wait after a rate limit. |
| `**gen_config` | `kwargs` | `None` | Global defaults for `temperature`, `top_p`, etc. |

---

## 🤝 Contributing
1. Fork the Project.
2. Create your Feature Branch.
3. Commit your Changes.
4. Push to the Branch.
5. Open a Pull Request.

## 📜 License
Distributed under the MIT License.

---
**gsdk** — Powering the next generation of Gemini applications. 🚀