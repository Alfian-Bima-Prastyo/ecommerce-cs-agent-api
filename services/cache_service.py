import time
from typing import Any

class CacheService:
    def __init__(self, ttl_seconds: int = 300):  # default 5 menit
        self._store: dict = {}
        self._ttl = ttl_seconds

    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self._ttl

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value, timestamp = self._store[key]
        if self._is_expired(timestamp):
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time())

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> dict:
        valid = sum(
            1 for _, (_, ts) in self._store.items()
            if not self._is_expired(ts)
        )
        return {"total_keys": len(self._store), "valid_keys": valid}

_cache = CacheService(ttl_seconds=300)

def get_cache() -> CacheService:
    return _cache