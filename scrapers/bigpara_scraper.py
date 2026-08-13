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

Time-window filtering: Bigpara exposes no absolute timestamp, only a Turkish
relative-time string ("2 sa önce", "1 gün önce") — `_parse_relative_turkish_time()`
converts that to an approximate datetime, which is then compared against
NEWS_LOOKBACK_HOURS the same way Google News's scraper does. If the string
doesn't match a known pattern (format changed, unexpected unit, etc.), the
item is kept rather than dropped — an unparseable date is not evidence the
article is stale, and silently losing a genuinely current item is worse than
occasionally keeping one old one.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from config.constants import NEWS_LOOKBACK_HOURS, SourceType
from config.settings import BigparaSourceConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem

logger = logging.getLogger(__name__)

# Card structure (confirmed via a live page fetch):
#   div.news-card-container--card-listing
#     div.news-card
#       a.news-card__title      <- headline text + href (relative)
#       div.news-card__info     <- "Aracı Kurum Haberleri ･ 2 sa önce"
NEWS_CARD_SELECTOR = "div.news-card-container--card-listing div.news-card"
TITLE_SELECTOR = "a.news-card__title"
INFO_SELECTOR = "div.news-card__info"
INFO_SEPARATOR = "･"

# Matches "3 dk önce", "2 saat önce", "1 gün önce", etc. Unit spelled out
# ("dakika"/"saat") or abbreviated ("dk"/"sa") — both observed live.
_RELATIVE_TIME_RE = re.compile(r"(\d+)\s*(dk|dakika|sa|saat|gün)\s*önce", re.IGNORECASE)

_UNIT_TO_TIMEDELTA_ARG = {
    "dk": "minutes",
    "dakika": "minutes",
    "sa": "hours",
    "saat": "hours",
    "gün": "days",
}


def _parse_relative_turkish_time(text: str | None, now: datetime | None = None) -> datetime | None:
    """"2 sa önce" -> datetime ~2 hours before `now` (defaults to
    datetime.now()). Returns None if `text` doesn't match a known pattern —
    callers must treat None as "unknown, don't filter it out", not "stale"."""
    if not text:
        return None
    match = _RELATIVE_TIME_RE.search(text)
    if not match:
        return None
    unit_arg = _UNIT_TO_TIMEDELTA_ARG.get(match.group(2).lower())
    if unit_arg is None:
        return None
    return (now or datetime.now()) - timedelta(**{unit_arg: int(match.group(1))})


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

        cutoff = datetime.now() - timedelta(hours=NEWS_LOOKBACK_HOURS)

        cards = soup.select(NEWS_CARD_SELECTOR)
        items: list[RawScrapedItem] = []
        skipped = 0
        for card in cards:
            title_link = card.select_one(TITLE_SELECTOR)
            if title_link is None or not title_link.get("href"):
                continue

            info_el = card.select_one(INFO_SELECTOR)
            raw_published_at = self._extract_relative_time(info_el) if info_el else None

            parsed_time = _parse_relative_turkish_time(raw_published_at)
            if parsed_time is not None and parsed_time < cutoff:
                skipped += 1
                continue

            items.append(
                RawScrapedItem(
                    source=SourceType.BIGPARA,
                    ticker=self.ticker.symbol,
                    raw_title=title_link.get_text(strip=True),
                    raw_url=urljoin(self.source_config.base_url, title_link["href"]),
                    raw_published_at=raw_published_at,
                )
            )

        logger.info(
            "%s: %d item, zaman filtresiyle %d tanesi elendi",
            self.ticker.symbol, len(cards), skipped,
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
