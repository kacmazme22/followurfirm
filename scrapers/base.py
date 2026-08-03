"""
Abstract base class for all scrapers (KAP, Bigpara, Google News RSS, ...).

Design goals, per project instructions:

  - Politeness (rate-limit delay + User-Agent) is centralized here so no
    scraper subclass has to reimplement delay/header logic.
  - `run()` never raises. A scraper failing must not take down the whole
    pipeline (one source crashing shouldn't stop the others). Callers get a
    `ScraperResult` with either items or an error string, never an exception.
  - No retry logic here, intentionally. Per project instruction: single
    attempt, log + skip on failure. `tenacity` is reserved for a narrower use
    elsewhere, not this file.
  - In-run caching only (dict[str, httpx.Response] per instance) — avoids
    hitting the same URL twice within one pipeline run. Cross-run caching
    belongs in utils/cache.py (not written yet, not imported here).
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel, Field

from config.settings import PolitenessConfig, TickerConfig
from scrapers.models import RawScrapedItem

logger = logging.getLogger(__name__)


class ScraperResult(BaseModel):
    """What `BaseScraper.run()` returns. `main.py` reads `.error` to feed
    `DigestRun.add_error()` and `.items` to feed the normalization step —
    scrapers never raise past this boundary."""

    items: list[RawScrapedItem] = Field(default_factory=list)
    error: str | None = None


class BaseScraper(ABC):
    """Common scraping behavior: politeness, in-run cache, graceful errors.

    Subclasses only implement `fetch_raw()` — the source-specific fetch and
    parse logic — and call `self._get(url)` instead of hitting httpx
    directly, so every request gets rate-limited, gets the configured
    User-Agent, and is deduped against the in-run cache automatically.
    """

    def __init__(
        self,
        ticker: TickerConfig,
        politeness: PolitenessConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.ticker = ticker
        self.politeness = politeness
        self._injected_client = http_client
        self._client: httpx.AsyncClient | None = http_client
        self._cache: dict[str, httpx.Response] = {}

    @abstractmethod
    async def fetch_raw(self) -> list[RawScrapedItem]:
        """Source-specific fetch + parse logic. Implemented by subclasses.
        Use `self._get(url)` for any HTTP request so politeness and the
        in-run cache apply."""

    async def run(self) -> ScraperResult:
        """Runs `fetch_raw()`, never raises. Any exception is caught, logged
        at ERROR level, and turned into a `ScraperResult(error=...)` so a
        broken source degrades the digest instead of stopping the pipeline."""
        owns_client = self._client is None
        if owns_client:
            self._client = httpx.AsyncClient()
        try:
            items = await self.fetch_raw()
            return ScraperResult(items=items)
        except Exception as exc:  # intentional: never propagate past run()
            logger.error(
                "%s failed for ticker %s: %s",
                type(self).__name__,
                self.ticker.symbol,
                exc,
                exc_info=True,
            )
            return ScraperResult(error=str(exc))
        finally:
            if owns_client and self._client is not None:
                await self._client.aclose()
                self._client = self._injected_client

    async def _get(self, url: str, **kwargs) -> httpx.Response:
        """GET with politeness (delay + User-Agent) and in-run caching.
        A repeated request to the same URL within a run returns the cached
        response instead of hitting the network again."""
        return await self._request("GET", url, cache_key=url, **kwargs)

    async def _post(self, url: str, *, json: dict | None = None, **kwargs) -> httpx.Response:
        """POST with politeness (delay + User-Agent) and in-run caching. The
        cache key includes the JSON body since, unlike GET, the same URL can
        carry different requests depending on payload."""
        cache_key = f"POST:{url}:{json}"
        return await self._request("POST", url, cache_key=cache_key, json=json, **kwargs)

    async def _request(self, method: str, url: str, *, cache_key: str, **kwargs) -> httpx.Response:
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._client is None:
            raise RuntimeError("_request() called outside of run() — no HTTP client set up.")

        await self._sleep_politely()

        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("User-Agent", self.politeness.user_agent)

        response = await self._client.request(method, url, headers=headers, **kwargs)
        self._cache[cache_key] = response
        return response

    async def _sleep_politely(self) -> None:
        delay = random.uniform(self.politeness.min_delay_seconds, self.politeness.max_delay_seconds)
        await asyncio.sleep(delay)
