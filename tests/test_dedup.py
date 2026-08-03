"""
Manual verification for nlp/dedup.py.

Synthetic cases (deterministic, no network) cover the actual contract:
priority ordering, KAP-link preservation, and tie-breaking by first-seen.
A real-data smoke test at the end runs the full scrape -> categorize ->
dedup pipeline against live KAP + Bigpara data (same scrapers verified in
earlier tests) just to prove it doesn't crash on real input.

Run directly:

    python -m tests.test_dedup
"""

from __future__ import annotations

import asyncio

from config.constants import SourceType
from config.settings import BigparaSourceConfig, KapSourceConfig, PolitenessConfig, TickerConfig
from nlp.categorizer import categorize_batch
from nlp.dedup import deduplicate
from scrapers.bigpara_scraper import BigparaScraper
from scrapers.kap_scraper import KapScraper
from scrapers.models import NewsItem


def _item(source: SourceType, ticker: str, title: str, url: str | None = None) -> NewsItem:
    return NewsItem(ticker=ticker, title=title, url=url, source=source)


def test_exact_hash_priority() -> None:
    """Same URL (-> same content_hash) from KAP and Google News: Google News
    must survive, and the KAP URL must be preserved on it."""
    kap_item = _item(SourceType.KAP, "THYAO", "Bilanço açıklandı", url="https://www.kap.org.tr/tr/Bildirim/111")
    same_url_but_different_source = kap_item.model_copy(update={"source": SourceType.GOOGLE_NEWS})

    result = deduplicate([kap_item, same_url_but_different_source])

    assert len(result) == 1
    assert result[0].source == SourceType.GOOGLE_NEWS
    assert str(result[0].related_kap_url) == "https://www.kap.org.tr/tr/Bildirim/111"
    print("[OK] exact-hash dedup: GOOGLE_NEWS survives over KAP, KAP link preserved")


def test_fuzzy_title_priority_bigpara_over_kap() -> None:
    """KAP and Bigpara report the same event with different URLs and
    slightly different wording -> Bigpara (higher priority) must survive,
    KAP's URL must land in related_kap_url."""
    kap_item = _item(
        SourceType.KAP, "GARAN", "Garanti BBVA Finansal Rapor Açıklaması",
        url="https://www.kap.org.tr/tr/Bildirim/222",
    )
    bigpara_item = _item(
        SourceType.BIGPARA, "GARAN", "Garanti BBVA Finansal Rapor Açıklaması Hk.",
        url="https://bigpara.hurriyet.com.tr/haberler/garan-rapor_ID999/",
    )

    result = deduplicate([kap_item, bigpara_item])

    assert len(result) == 1, f"expected the two to fuzzy-match into one, got {len(result)}"
    assert result[0].source == SourceType.BIGPARA
    assert str(result[0].related_kap_url) == "https://www.kap.org.tr/tr/Bildirim/222"
    print("[OK] fuzzy-title dedup: BIGPARA survives over KAP, KAP link preserved on the winner")


def test_tie_priority_keeps_first_seen() -> None:
    first = _item(SourceType.GOOGLE_NEWS, "EREGL", "Ereğli Demir Çelik üretim rekoru kırdı", url="https://a.example/1")
    second = _item(SourceType.GOOGLE_NEWS, "EREGL", "Ereğli Demir Çelik'ten üretimde rekor kırıldı", url="https://a.example/2")

    result = deduplicate([first, second])

    assert len(result) == 1
    assert str(result[0].url) == "https://a.example/1", "equal priority -> first-seen item should survive"
    print("[OK] equal-priority fuzzy match: first-seen item kept")


def test_different_tickers_never_merge() -> None:
    a = _item(SourceType.KAP, "THYAO", "Finansal Rapor", url="https://www.kap.org.tr/tr/Bildirim/1")
    b = _item(SourceType.BIGPARA, "GARAN", "Finansal Rapor", url="https://bigpara.hurriyet.com.tr/x")

    result = deduplicate([a, b])

    assert len(result) == 2, "fuzzy stage must not merge across different tickers even with identical titles"
    print("[OK] identical titles on different tickers are not merged")


async def test_real_pipeline_smoke() -> None:
    """Full scrape -> categorize -> dedup on live KAP + Bigpara data for
    GARAN. Not asserting a specific dedup outcome (today's real disclosures
    may or may not overlap with today's real Bigpara headlines) — just
    proving the pipeline runs end to end on real input without crashing,
    and that dedup never increases the item count."""
    garan = TickerConfig(symbol="GARAN", name="Garanti BBVA")

    import scrapers.kap_scraper as kap_scraper_module
    original_lookback = kap_scraper_module.LOOKBACK_DAYS
    kap_scraper_module.LOOKBACK_DAYS = 7
    try:
        kap_result = await KapScraper(
            garan,
            KapSourceConfig(
                base_url="https://www.kap.org.tr",
                disclosure_endpoint="/tr/api/disclosure/members/byCriteria",
                max_items_per_ticker=10,
                politeness=PolitenessConfig(),
            ),
        ).run()
    finally:
        kap_scraper_module.LOOKBACK_DAYS = original_lookback

    bigpara_result = await BigparaScraper(
        garan,
        BigparaSourceConfig(base_url="https://bigpara.hurriyet.com.tr", politeness=PolitenessConfig()),
    ).run()

    assert kap_result.error is None and bigpara_result.error is None

    news_items = categorize_batch(kap_result.items + bigpara_result.items)
    before_count = len(news_items)
    deduped = deduplicate(news_items)

    assert len(deduped) <= before_count
    print(f"\n[OK] real pipeline: {before_count} categorized items -> {len(deduped)} after dedup")
    kap_links_preserved = [i for i in deduped if i.related_kap_url is not None]
    print(f"     items carrying a preserved related_kap_url: {len(kap_links_preserved)}")


async def main() -> None:
    test_exact_hash_priority()
    test_fuzzy_title_priority_bigpara_over_kap()
    test_tie_priority_keeps_first_seen()
    test_different_tickers_never_merge()
    await test_real_pipeline_smoke()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
