"""
data/cache.py — in-run memoization to avoid duplicate yfinance calls.

The RunCache is instantiated once in scanner.py and passed to analytics modules.
Not thread-safe (scanner is single-threaded).
"""


class RunCache:
    """Simple in-memory cache keyed by (name, *args).

    Call cache.get_or_fetch(key, fn, *args) to retrieve cached or freshly
    fetched data.  Not thread-safe (scanner is single-threaded).
    """

    def __init__(self) -> None:
        self._store: dict = {}
        self._hits: int = 0
        self._misses: int = 0

    def get_or_fetch(self, key: str, fn, *args, **kwargs):
        """Return cached value for *key*, or call fn(*args, **kwargs) and cache it.

        The cache key is *key* as provided — callers are responsible for making
        keys unique across different functions (e.g. include the function name).
        """
        if key in self._store:
            self._hits += 1
            return self._store[key]

        self._misses += 1
        value = fn(*args, **kwargs)
        self._store[key] = value
        return value

    def clear(self) -> None:
        """Evict all cached entries and reset hit/miss counters."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Return {"hits": int, "misses": int}."""
        return {"hits": self._hits, "misses": self._misses}
