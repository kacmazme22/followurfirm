"""
Manual verification for GoogleNewsScraper — hits the real Google News RSS
feed (no mocking, since the point is to prove real parsing against the real
feed format). Single request, real politeness delay applied via
BaseScraper._get().

Run directly:

    python -m tests.test_google_news_scraper
"""

from __future__ import annotations

import asyncio
import logging

from urllib.parse import quote

import feedparser

from config.settings import GoogleNewsRssConfig, PolitenessConfig, TickerConfig
from scrapers.google_news_scraper import GoogleNewsScraper

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

THYAO = TickerConfig(symbol="THYAO", name="Türk Hava Yolları")


async def main() -> None:
    scraper = GoogleNewsScraper(
        ticker=THYAO,
        source_config=GoogleNewsRssConfig(),
        politeness=PolitenessConfig(),
    )
    result = await scraper.run()

    assert result.error is None, f"expected no error, got: {result.error}"
    assert len(result.items) > 0, "expected at least one real news item for THYAO"

    # Re-fetch the raw feed size directly (scraper.run() only logs the
    # before/after count, doesn't return it) so the filtering effect is
    # visible in this test's own output too.
    query = scraper.source_config.query_template_tr.format(company_name=THYAO.name, ticker=THYAO.symbol)
    raw_feed = feedparser.parse(
        f"https://news.google.com/rss/search?q={quote(query)}&hl=tr&gl=TR&ceid=TR:tr"
    )
    print(f"{len(raw_feed.entries)} ham Google News item -> {len(result.items)} tanesi 36 saat içinde\n")

    print(f"Fetched {len(result.items)} items for {THYAO.symbol}\n")
    for item in result.items[:5]:
        print(f"- title: {item.raw_title}")
        print(f"  url: {item.raw_url}")
        print(f"  published_at: {item.raw_published_at}")
        print(f"  source_name: {item.extra.get('source_name')}")
        print()

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
