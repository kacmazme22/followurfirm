"""
Manual verification for KapScraper — hits the real KAP byCriteria JSON API
(no mocking: the point is to prove the endpoint and field mapping against
real data). Single POST per ticker, real politeness delay applied via
BaseScraper._post().

Run directly:

    python -m tests.test_kap_scraper
"""

from __future__ import annotations

import asyncio

import scrapers.kap_scraper as kap_scraper_module
from config.settings import KapSourceConfig, PolitenessConfig, TickerConfig
from scrapers.kap_scraper import KapScraper

FAST_POLITENESS = PolitenessConfig(min_delay_seconds=0.2, max_delay_seconds=0.5)
KAP_SOURCE_CONFIG = KapSourceConfig(
    base_url="https://www.kap.org.tr",
    disclosure_endpoint="/tr/api/disclosure/members/byCriteria",
    max_items_per_ticker=10,
    politeness=FAST_POLITENESS,
)


async def test_default_lookback_never_errors() -> None:
    """The default LOOKBACK_DAYS=2 window may legitimately yield 0 items for
    a quiet ticker on a quiet day — that's not an error. This just proves
    the request succeeds and the contract (ScraperResult, no exception) holds."""
    for symbol, name in [("THYAO", "Türk Hava Yolları"), ("ASTOR", "Astor Enerji")]:
        scraper = KapScraper(TickerConfig(symbol=symbol, name=name), KAP_SOURCE_CONFIG)
        result = await scraper.run()
        assert result.error is None, f"{symbol}: unexpected error {result.error}"
        print(f"[OK] {symbol}: {len(result.items)} item(s) in the default {kap_scraper_module.LOOKBACK_DAYS}-day window")


async def test_field_mapping_against_real_disclosure() -> None:
    """GARAN had real disclosures in the last 7 days (checked live beforehand).
    Widen the lookback just for this test to get real data and verify the
    subject/summary/url field mapping end to end."""
    original_lookback = kap_scraper_module.LOOKBACK_DAYS
    kap_scraper_module.LOOKBACK_DAYS = 7
    try:
        scraper = KapScraper(TickerConfig(symbol="GARAN", name="Garanti BBVA"), KAP_SOURCE_CONFIG)
        result = await scraper.run()
    finally:
        kap_scraper_module.LOOKBACK_DAYS = original_lookback

    assert result.error is None, f"unexpected error: {result.error}"
    assert len(result.items) > 0, "expected at least one real GARAN disclosure in a 7-day window"

    item = result.items[0]
    assert item.ticker == "GARAN"
    assert item.raw_title == item.raw_disclosure_type, "raw_title should mirror `subject`, same as raw_disclosure_type"
    assert item.raw_url.startswith("https://www.kap.org.tr/tr/Bildirim/")
    assert item.raw_body_snippet, "raw_body_snippet should carry `summary`, distinct from the title"

    print(f"\nFetched {len(result.items)} real GARAN disclosure(s) (7-day window):\n")
    for it in result.items:
        print(f"- title (subject):     {it.raw_title}")
        print(f"  body (summary):      {it.raw_body_snippet}")
        print(f"  disclosure_type:     {it.raw_disclosure_type}")
        print(f"  published_at:        {it.raw_published_at}")
        print(f"  url:                 {it.raw_url}")
        print(f"  extra:               {it.extra}")
        print()


async def main() -> None:
    await test_default_lookback_never_errors()
    await test_field_mapping_against_real_disclosure()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
