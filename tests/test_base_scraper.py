"""
Manual verification for BaseScraper (scrapers/base.py).

No real network calls: httpx.MockTransport stands in for the network so this
covers the success path, the in-run cache, and the "never raise" error path.

pytest / pytest-asyncio aren't in requirements.txt yet, so this is written as
a plain asyncio script rather than a pytest suite — run directly:

    python -m tests.test_base_scraper
"""

from __future__ import annotations

import asyncio

import httpx

from config.constants import SourceType
from config.settings import PolitenessConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem

FAST_POLITENESS = PolitenessConfig(min_delay_seconds=0.0, max_delay_seconds=0.0)
DUMMY_TICKER = TickerConfig(symbol="THYAO", name="Türk Hava Yolları")


class _DummySuccessScraper(BaseScraper):
    """Calls the same URL twice to prove the in-run cache dedupes requests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests_seen: list[str] = []

    async def fetch_raw(self) -> list[RawScrapedItem]:
        url = "https://example.com/thyao-news"
        first = await self._get(url)
        second = await self._get(url)  # should hit the cache, not the network
        assert first is second, "expected the second _get() to return the cached response"

        return [
            RawScrapedItem(
                source=SourceType.BIGPARA,
                ticker=self.ticker.symbol,
                raw_title=first.json()["title"],
            )
        ]


class _DummyFailingScraper(BaseScraper):
    async def fetch_raw(self) -> list[RawScrapedItem]:
        raise ValueError("simulated parse failure")


def _mock_transport(request_log: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(str(request.url))
        assert request.headers["User-Agent"] == FAST_POLITENESS.user_agent
        return httpx.Response(200, json={"title": "Uçak filosuna 3 yeni uçak katıldı"})

    return httpx.MockTransport(handler)


async def test_success_path_and_cache() -> None:
    request_log: list[str] = []
    client = httpx.AsyncClient(transport=_mock_transport(request_log))
    scraper = _DummySuccessScraper(DUMMY_TICKER, FAST_POLITENESS, http_client=client)

    result = await scraper.run()
    await client.aclose()

    assert result.error is None
    assert len(result.items) == 1
    assert result.items[0].raw_title == "Uçak filosuna 3 yeni uçak katıldı"
    assert len(request_log) == 1, f"expected exactly 1 network call (cache should dedupe), got {len(request_log)}"
    print("[OK] success path + in-run cache dedupe")


async def test_error_path_never_raises() -> None:
    scraper = _DummyFailingScraper(DUMMY_TICKER, FAST_POLITENESS)  # no injected client -> owns its own
    result = await scraper.run()

    assert result.items == []
    assert result.error == "simulated parse failure"
    print("[OK] error path caught, logged, and returned as ScraperResult.error")


async def main() -> None:
    await test_success_path_and_cache()
    await test_error_path_never_raises()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
