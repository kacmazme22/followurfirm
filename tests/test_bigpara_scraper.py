"""
Manual verification for BigparaScraper — hits the real Bigpara stock-news
page (no mocking: the page structure needed to be seen live, same as
Google News). Single request, real politeness delay via BaseScraper._get().

Run directly:

    python -m tests.test_bigpara_scraper
"""

from __future__ import annotations

import asyncio

from config.settings import BigparaSourceConfig, PolitenessConfig, TickerConfig
from scrapers.bigpara_scraper import BigparaScraper

THYAO = TickerConfig(symbol="THYAO", name="Türk Hava Yolları")


async def main() -> None:
    scraper = BigparaScraper(
        ticker=THYAO,
        source_config=BigparaSourceConfig(
            base_url="https://bigpara.hurriyet.com.tr",
            politeness=PolitenessConfig(),
        ),
    )
    result = await scraper.run()

    assert result.error is None, f"expected no error, got: {result.error}"
    assert len(result.items) > 0, "expected at least one real news item for THYAO"

    print(f"Fetched {len(result.items)} items for {THYAO.symbol}\n")
    for item in result.items[:5]:
        print(f"- title: {item.raw_title}")
        print(f"  url: {item.raw_url}")
        print(f"  published_at: {item.raw_published_at}")
        print()

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
