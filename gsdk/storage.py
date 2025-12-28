import pickle
import os
from typing import List

class BaseStorage:
    def get(self, sid: str) -> List: raise NotImplementedError
    def set(self, sid: str, history: List): raise NotImplementedError

class FileStorage(BaseStorage):
    def __init__(self, dir: str = "sessions"):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)

    def _path(self, sid: str): return os.path.join(self.dir, f"{sid}.bin")

    def get(self, sid: str):
        if os.path.exists(self._path(sid)):
            with open(self._path(sid), 'rb') as f:
                return pickle.load(f)
        return []

    def set(self, sid: str, history: List):
        with open(self._path(sid), 'wb') as f:
            pickle.dump(history, f)