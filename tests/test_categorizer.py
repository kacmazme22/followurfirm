"""
Manual verification for nlp/categorizer.py.

Two parts:
  1. Synthetic edge cases (no network): boilerplate skipping, date parsing
     success/graceful-failure.
  2. Real data (network): pulls real RawScrapedItems from KAP (GARAN) and
     Bigpara (THYAO) — same scrapers already verified in
     tests/test_kap_scraper.py / tests/test_bigpara_scraper.py — and checks
     they land in the expected categories.

Run directly:

    python -m tests.test_categorizer
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import scrapers.kap_scraper as kap_scraper_module
from config.constants import NewsCategory, SourceType
from config.settings import BigparaSourceConfig, KapSourceConfig, PolitenessConfig, TickerConfig
from nlp.categorizer import categorize, categorize_batch
from scrapers.bigpara_scraper import BigparaScraper
from scrapers.kap_scraper import KapScraper
from scrapers.models import RawScrapedItem


def test_boilerplate_is_skipped() -> None:
    noisy = RawScrapedItem(
        source=SourceType.KAP,
        ticker="GARAN",
        raw_title="Yukarıdaki açıklamalarımızın gerçeğe uygun olduğunu beyan ederiz.",
    )
    assert categorize(noisy) is None
    assert categorize_batch([noisy]) == []
    print("[OK] boilerplate title -> skipped entirely")


def test_date_parsing() -> None:
    kap_style = RawScrapedItem(
        source=SourceType.KAP, ticker="GARAN", raw_title="Finansal Rapor",
        raw_published_at="30.07.2026 08:00:49",
    )
    item = categorize(kap_style)
    assert item is not None
    assert item.published_at == datetime(2026, 7, 30, 8, 0, 49)
    assert item.category == NewsCategory.FINANCIALS

    relative_style = RawScrapedItem(
        source=SourceType.BIGPARA, ticker="THYAO", raw_title="Bir haber başlığı",
        raw_published_at="2 sa önce",
    )
    item2 = categorize(relative_style)
    assert item2 is not None
    assert item2.published_at is None, "relative time strings should not raise, just parse to None"
    print("[OK] KAP absolute date parses; Bigpara relative time gracefully becomes None")


async def test_real_kap_and_bigpara_data() -> None:
    garan = TickerConfig(symbol="GARAN", name="Garanti BBVA")
    thyao = TickerConfig(symbol="THYAO", name="Türk Hava Yolları")

    # Widen KAP's lookback like tests/test_kap_scraper.py does, to guarantee
    # real disclosures for GARAN regardless of how quiet the last 2 days were.
    original_lookback = kap_scraper_module.LOOKBACK_DAYS
    kap_scraper_module.LOOKBACK_DAYS = 7
    try:
        kap_scraper = KapScraper(
            garan,
            KapSourceConfig(
                base_url="https://www.kap.org.tr",
                disclosure_endpoint="/tr/api/disclosure/members/byCriteria",
                max_items_per_ticker=10,
                politeness=PolitenessConfig(),
            ),
        )
        kap_result = await kap_scraper.run()
    finally:
        kap_scraper_module.LOOKBACK_DAYS = original_lookback

    bigpara_scraper = BigparaScraper(
        thyao,
        BigparaSourceConfig(base_url="https://bigpara.hurriyet.com.tr", politeness=PolitenessConfig()),
    )
    bigpara_result = await bigpara_scraper.run()

    assert kap_result.error is None, f"KAP fetch failed: {kap_result.error}"
    assert bigpara_result.error is None, f"Bigpara fetch failed: {bigpara_result.error}"
    assert kap_result.items, "expected real GARAN disclosures to categorize against"
    assert bigpara_result.items, "expected real THYAO bigpara items to categorize against"

    kap_news = categorize_batch(kap_result.items)
    bigpara_news = categorize_batch(bigpara_result.items)

    print("\nKAP (GARAN) categorization:\n")
    for item in kap_news:
        print(f"- [{item.category.value}] {item.disclosure_type_raw!r} -> published_at={item.published_at}")

    print("\nBigpara (THYAO) categorization (first 5):\n")
    for item in bigpara_news[:5]:
        print(f"- [{item.category.value}] {item.title[:70]!r}")

    financial_items = [
        i for i in kap_news if i.disclosure_type_raw and "finansal rapor" in i.disclosure_type_raw.lower()
    ]
    assert financial_items, "expected at least one real 'Finansal Rapor' disclosure"
    assert all(i.category == NewsCategory.FINANCIALS for i in financial_items), "'finansal rapor' must map to FINANCIALS"

    ozel_durum_items = [
        i for i in kap_news if i.disclosure_type_raw and "özel durum açıklaması" in i.disclosure_type_raw.lower()
    ]
    assert ozel_durum_items, "expected at least one real 'Özel Durum Açıklaması' disclosure"
    assert all(i.category == NewsCategory.KAP_MATERIAL for i in ozel_durum_items)

    no_match_items = [
        i for i in kap_news if i.disclosure_type_raw and "sorumluluk beyan" in i.disclosure_type_raw.lower()
    ]
    assert no_match_items, "expected at least one real 'Sorumluluk Beyanı' disclosure (no keyword hit -> default)"
    assert all(i.category == NewsCategory.GENERAL_SECTOR for i in no_match_items)

    print("\n[OK] real KAP data: 'finansal rapor'->FINANCIALS, 'özel durum açıklaması'->KAP_MATERIAL, no-match->GENERAL_SECTOR")


async def main() -> None:
    test_boilerplate_is_skipped()
    test_date_parsing()
    await test_real_kap_and_bigpara_data()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
