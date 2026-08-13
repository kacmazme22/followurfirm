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
relative-time string ("2 sa önce", "1 gün önce", "3 ay önce") —
`_parse_relative_turkish_time()` converts that to an approximate datetime,
which is then compared against NEWS_LOOKBACK_HOURS the same way Google
News's scraper does. If the string doesn't match a known pattern (format
changed, unexpected unit, etc.), the item is kept rather than dropped — an
unparseable date is not evidence the article is stale, and silently losing a
genuinely current item is worse than occasionally keeping one old one.

Bug fixed 2026-08-13: the initial version only recognized dk/dakika/sa/saat/
gün units. Live YUNSA data showed most of a ticker's news cards are actually
"1 ay önce" / "3 ay önce" (months old) — a real, common case for
lower-volume tickers, not a rare edge case — so those items hit the "can't
parse, keep it" fallback above and sailed straight through the filter
unfiltered (17 of YUNSA's 18 raw cards survived a 36h window). "ay" (~30
days) and "yıl" (~365 days, seen in older KAP-relay history) are now
recognized too.

Article-body enrichment: after time-filtering, the MAX_ARTICLES_TO_ENRICH_PER_TICKER
most recent surviving items get their raw_body_snippet filled in with the
real article body (utils/article_fetcher.py) — Bigpara's news-card list view
never exposes anything but a title, confirmed by inspecting the live DOM.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from config.constants import MAX_ARTICLES_TO_ENRICH_PER_TICKER, NEWS_LOOKBACK_HOURS, SourceType
from config.settings import BigparaSourceConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem
from utils.article_fetcher import fetch_article_body

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

# Matches "3 dk önce", "2 saat önce", "1 gün önce", "3 ay önce", "1 yıl
# önce", etc. Unit spelled out ("dakika"/"saat") or abbreviated ("dk"/"sa")
# — both observed live, as are all of gün/ay/yıl.
_RELATIVE_TIME_RE = re.compile(r"(\d+)\s*(dk|dakika|sa|saat|gün|ay|yıl)\s*önce", re.IGNORECASE)

# Units with an exact timedelta() equivalent.
_UNIT_TO_TIMEDELTA_ARG = {
    "dk": "minutes",
    "dakika": "minutes",
    "sa": "hours",
    "saat": "hours",
    "gün": "days",
}

# "ay"/"yıl" have no fixed length — approximated in days, which is more than
# precise enough for a 36h-scale staleness filter.
_UNIT_TO_APPROX_DAYS = {
    "ay": 30,
    "yıl": 365,
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
    amount = int(match.group(1))
    unit = match.group(2).lower()
    now = now or datetime.now()
    if unit in _UNIT_TO_TIMEDELTA_ARG:
        return now - timedelta(**{_UNIT_TO_TIMEDELTA_ARG[unit]: amount})
    if unit in _UNIT_TO_APPROX_DAYS:
        return now - timedelta(days=_UNIT_TO_APPROX_DAYS[unit] * amount)
    return None


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

        await self._enrich_with_article_bodies(items)
        # No news cards on the page is a normal outcome, not an error.
        return items

    async def _enrich_with_article_bodies(self, items: list[RawScrapedItem]) -> None:
        """Mutates the newest MAX_ARTICLES_TO_ENRICH_PER_TICKER items in
        place, filling raw_body_snippet with real article text where
        possible. `items` is already newest-first (card order on the page),
        so a plain slice picks the most recent ones."""
        enriched = 0
        failed = 0
        for item in items[:MAX_ARTICLES_TO_ENRICH_PER_TICKER]:
            if not item.raw_url:
                continue
            body = await fetch_article_body(self, item.raw_url, is_google_news=False)
            if body:
                item.raw_body_snippet = body
                enriched += 1
            else:
                failed += 1

        logger.info(
            "%s: %d haber zenginleştirildi, %d'si başarısız/boş döndü",
            self.ticker.symbol, enriched, failed,
        )

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
