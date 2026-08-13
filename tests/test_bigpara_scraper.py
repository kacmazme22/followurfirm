"""
Manual verification for BigparaScraper — hits the real Bigpara stock-news
page (no mocking: the page structure needed to be seen live, same as
Google News). Single request, real politeness delay via BaseScraper._get().

Run directly:

    python -m tests.test_bigpara_scraper
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from config.constants import NEWS_LOOKBACK_HOURS
from config.settings import BigparaSourceConfig, PolitenessConfig, TickerConfig
from scrapers.bigpara_scraper import (
    NEWS_CARD_SELECTOR,
    TITLE_SELECTOR,
    BigparaScraper,
    _parse_relative_turkish_time,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

THYAO = TickerConfig(symbol="THYAO", name="Türk Hava Yolları")


def test_parse_relative_turkish_time() -> None:
    now = datetime(2026, 8, 13, 12, 0, 0)

    assert _parse_relative_turkish_time("3 dk önce", now=now) == now - timedelta(minutes=3)
    assert _parse_relative_turkish_time("5 dakika önce", now=now) == now - timedelta(minutes=5)
    assert _parse_relative_turkish_time("2 sa önce", now=now) == now - timedelta(hours=2)
    assert _parse_relative_turkish_time("18 saat önce", now=now) == now - timedelta(hours=18)
    assert _parse_relative_turkish_time("1 gün önce", now=now) == now - timedelta(days=1)
    assert _parse_relative_turkish_time("1 ay önce", now=now) == now - timedelta(days=30)
    assert _parse_relative_turkish_time("3 ay önce", now=now) == now - timedelta(days=90)
    assert _parse_relative_turkish_time("1 yıl önce", now=now) == now - timedelta(days=365)
    assert _parse_relative_turkish_time("Aracı Kurum Haberleri ・ 4 sa önce", now=now) == now - timedelta(hours=4)
    assert _parse_relative_turkish_time("Kap Haberleri ･ 2 ay önce YUNSA -", now=now) == now - timedelta(days=60)

    # Unknown/unparseable formats -> None (caller must NOT treat as stale)
    assert _parse_relative_turkish_time(None) is None
    assert _parse_relative_turkish_time("") is None
    assert _parse_relative_turkish_time("dün") is None
    assert _parse_relative_turkish_time("13.08.2026 tarihinde") is None

    print("[OK] _parse_relative_turkish_time: dk/dakika/sa/saat/gün kalıpları doğru parse edildi, bilinmeyen format None döndü")


async def main() -> None:
    test_parse_relative_turkish_time()

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

    # Re-fetch the raw page directly (scraper.run() only logs the
    # before/after count) so the filtering effect — and specifically that
    # the oldest relative-time cards are gone — is visible in this test too.
    url = f"{scraper.source_config.base_url}/borsa/hisse-fiyatlari/{THYAO.symbol.lower()}-detay/hisse-haberleri/"
    async with scraper._client_session():
        raw_response = await scraper._get(url)
    soup = BeautifulSoup(raw_response.text, "html.parser")
    raw_cards = soup.select(NEWS_CARD_SELECTOR)

    cutoff = datetime.now() - timedelta(hours=NEWS_LOOKBACK_HOURS)
    surviving_urls = {item.raw_url for item in result.items}

    stale_but_present = 0
    for card in raw_cards:
        title_link = card.select_one(TITLE_SELECTOR)
        if title_link is None or not title_link.get("href"):
            continue
        info_el = card.select_one("div.news-card__info")
        raw_time = scraper._extract_relative_time(info_el) if info_el else None
        parsed = _parse_relative_turkish_time(raw_time)
        card_url = urljoin(scraper.source_config.base_url, title_link["href"])
        if parsed is not None and parsed < cutoff:
            assert card_url not in surviving_urls, (
                f"stale item ({raw_time}) should have been filtered out but is still present: {card_url}"
            )
            stale_but_present += 1

    print(f"{len(raw_cards)} ham Bigpara item -> {len(result.items)} tanesi {NEWS_LOOKBACK_HOURS} saat içinde")
    print(f"[OK] {stale_but_present} eski (36 saatten önceki) item doğrulandı, hiçbiri sonuçta yok\n")

    print(f"Fetched {len(result.items)} items for {THYAO.symbol}\n")
    for item in result.items[:5]:
        print(f"- title: {item.raw_title}")
        print(f"  url: {item.raw_url}")
        print(f"  published_at: {item.raw_published_at}")
        print()

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
