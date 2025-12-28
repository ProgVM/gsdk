import pickle
import os
from abc import ABC, abstractmethod
from typing import List, Any
from google.genai import types

class BaseStorage(ABC):
    @abstractmethod
    def get(self, session_id: str) -> List[Any]:
        pass

    @abstractmethod
    def set(self, session_id: str, history: List[Any]):
        pass

    @abstractmethod
    def delete(self, session_id: str):
        pass

class FileStorage(BaseStorage):
    def __init__(self, path: str = "sessions"):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

    def _get_path(self, session_id: str) -> str:
        return os.path.join(self.path, f"{session_id}.bin")

    def get(self, session_id: str) -> List[Any]:
        filepath = self._get_path(session_id)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return []
        return []

    def set(self, session_id: str, history: List[Any]):
        filepath = self._get_path(session_id)
        with open(filepath, "wb") as f:
            pickle.dump(history, f)

    def delete(self, session_id: str):
        filepath = self._get_path(session_id)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass