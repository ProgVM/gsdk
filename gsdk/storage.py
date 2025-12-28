import pickle
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Any

logger = logging.getLogger("gsdk.storage")

class BaseStorage(ABC):
    @abstractmethod
    def get(self, session_id: str) -> List[Any]: pass
    @abstractmethod
    def set(self, session_id: str, history: List[Any]): pass
    @abstractmethod
    def delete(self, session_id: str): pass

class FileStorage(BaseStorage):
    """Pickle-based storage for local development."""
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
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                return []
        return []

    def set(self, session_id: str, history: List[Any]):
        filepath = self._get_path(session_id)
        try:
            with open(filepath, "wb") as f:
                pickle.dump(history, f)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

    def delete(self, session_id: str):
        filepath = self._get_path(session_id)
        if os.path.exists(filepath):
            os.remove(filepath)

class RedisStorage(BaseStorage):
    """High-performance storage for production using Redis."""
    def __init__(self, host='localhost', port=6379, db=0, password=None, prefix="gsdk:"):
        try:
            import redis
            self.r = redis.Redis(host=host, port=port, db=db, password=password)
            self.prefix = prefix
        except ImportError:
            raise ImportError("Please install 'redis' to use RedisStorage.")

    def get(self, session_id: str) -> List[Any]:
        data = self.r.get(f"{self.prefix}{session_id}")
        return pickle.loads(data) if data else []

    def set(self, session_id: str, history: List[Any]):
        self.r.set(f"{self.prefix}{session_id}", pickle.dumps(history))

    def delete(self, session_id: str):
        self.r.delete(f"{self.prefix}{session_id}")