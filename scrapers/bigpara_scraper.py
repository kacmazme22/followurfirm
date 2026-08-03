"""
Bigpara ticker-news scraper. Unlike KAP, Bigpara's stock-news page is plain
server-rendered HTML (verified: no Next.js/Nuxt/React markers, no
`__NEXT_DATA__`/`__NUXT__`/`data-reactroot` in the response) — `httpx` +
`beautifulsoup4` is enough, no JS rendering needed.

URL pattern: `/borsa/hisse-fiyatlari/{ticker}-detay/hisse-haberleri/`. Bigpara
also has slugs with the company name baked in (e.g.
`thyao-turk-hava-yollari-detay`), but the bare-ticker form
(`thyao-detay`) 200s and serves byte-identical news cards — confirmed with a
live request — so no ticker->slug lookup table is needed.
"""

from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup

from config.constants import SourceType
from config.settings import BigparaSourceConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem

# Card structure (confirmed via a live page fetch):
#   div.news-card-container--card-listing
#     div.news-card
#       a.news-card__title      <- headline text + href (relative)
#       div.news-card__info     <- "Aracı Kurum Haberleri ･ 2 sa önce"
NEWS_CARD_SELECTOR = "div.news-card-container--card-listing div.news-card"
TITLE_SELECTOR = "a.news-card__title"
INFO_SELECTOR = "div.news-card__info"
INFO_SEPARATOR = "･"


class BigparaScraper(BaseScraper):
    """Fetches and parses the Bigpara stock-news page for one ticker."""

    def __init__(
        self,
        ticker: TickerConfig,
        source_config: BigparaSourceConfig,
        http_client=None,
    ) -> None:
        super().__init__(ticker, source_config.politeness, http_client)
        self.source_config = source_config

    async def fetch_raw(self) -> list[RawScrapedItem]:
        url = f"{self.source_config.base_url}/borsa/hisse-fiyatlari/{self.ticker.symbol.lower()}-detay/hisse-haberleri/"
        response = await self._get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        items: list[RawScrapedItem] = []
        for card in soup.select(NEWS_CARD_SELECTOR):
            title_link = card.select_one(TITLE_SELECTOR)
            if title_link is None or not title_link.get("href"):
                continue

            info_el = card.select_one(INFO_SELECTOR)
            items.append(
                RawScrapedItem(
                    source=SourceType.BIGPARA,
                    ticker=self.ticker.symbol,
                    raw_title=title_link.get_text(strip=True),
                    raw_url=urljoin(self.source_config.base_url, title_link["href"]),
                    raw_published_at=self._extract_relative_time(info_el) if info_el else None,
                )
            )
        # No news cards on the page is a normal outcome, not an error.
        return items

    @staticmethod
    def _extract_relative_time(info_el) -> str | None:
        """`news-card__info` reads like "Aracı Kurum Haberleri ･ 2 sa önce" —
        keep the trailing relative-time segment as-is (raw string, whatever
        format it's in), fall back to the full text if the separator isn't
        there."""
        text = info_el.get_text(" ", strip=True)
        if INFO_SEPARATOR in text:
            return text.rsplit(INFO_SEPARATOR, 1)[-1].strip() or None
        return text or None
