from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

@dataclass
class GeminiResponse:
    text: str = ""
    images: List[bytes] = field(default_factory=list)
    audio: List[bytes] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    raw: Any = None

    def __str__(self):
        return self.text