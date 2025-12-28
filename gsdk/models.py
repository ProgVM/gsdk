from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

@dataclass
class GeminiResponse:
    text: str = ""
    sources: List[str] = field(default_factory=list)
    tool_calls: List[Any] = field(default_factory=list)
    raw: Any = None

    def __str__(self):
        return self.text