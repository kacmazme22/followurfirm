"""
Google News RSS scraper — the lowest-risk source: no JS rendering, no
undocumented endpoints, just a public RSS feed parsed with `feedparser`.

Query building uses `sources.google_news_rss.query_template_tr` from
config.yaml (default `"{company_name} {ticker} hisse"`), filled in with the
ticker's `name`/`symbol` from `TickerConfig` — no hardcoded query strings.

Time-window filtering: unlike KAP (which filters by date range server-side,
see kap_scraper.py's LOOKBACK_DAYS), Google News RSS returns whatever it
returns with no date parameter to constrain it — so filtering happens here,
client-side, using `entry.published_parsed` (feedparser's own normalized
`time.struct_time`, already UTC — not the raw `published` string). Items
older than NEWS_LOOKBACK_HOURS are dropped before ever becoming a
RawScrapedItem, so they never reach categorize()/dedup()/the LLM.

Article-body enrichment: after time-filtering, the MAX_ARTICLES_TO_ENRICH_PER_TICKER
most recent surviving items get their raw_body_snippet filled in with the
real article body (utils/article_fetcher.py) instead of staying title-only
— see that module's docstring for why this exists and what it costs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import feedparser

from config.constants import MAX_ARTICLES_TO_ENRICH_PER_TICKER, NEWS_LOOKBACK_HOURS, SourceType
from config.settings import GoogleNewsRssConfig, PolitenessConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem
from utils.article_fetcher import fetch_article_body

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"

# Google News RSS titles are consistently "Headline - Source Name"; the
# separator it uses to append the source name at the end of the title.
_TITLE_SOURCE_SEPARATOR = " - "


class GoogleNewsScraper(BaseScraper):
    """Fetches and parses the Google News RSS feed for one ticker."""

    def __init__(
        self,
        ticker: TickerConfig,
        source_config: GoogleNewsRssConfig,
        politeness: PolitenessConfig,
        http_client=None,
    ) -> None:
        super().__init__(ticker, politeness, http_client)
        self.source_config = source_config

    async def fetch_raw(self) -> list[RawScrapedItem]:
        query = self.source_config.query_template_tr.format(
            company_name=self.ticker.name, ticker=self.ticker.symbol
        )
        url = (
            f"{GOOGLE_NEWS_RSS_BASE}?q={quote(query)}"
            f"&hl={self.source_config.language}"
            f"&gl={self.source_config.country}"
            f"&ceid={self.source_config.country}:{self.source_config.language}"
        )

        response = await self._get(url)
        feed = feedparser.parse(response.text)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=NEWS_LOOKBACK_HOURS)

        items: list[RawScrapedItem] = []
        skipped = 0
        for entry in feed.entries:
            published_parsed = entry.get("published_parsed")
            if published_parsed is not None:
                # feedparser normalizes *_parsed fields to UTC already, so
                # this struct_time can be read directly as a UTC timestamp.
                published_at = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                if published_at < cutoff:
                    skipped += 1
                    continue

            title, source_name = self._split_title_and_source(entry.get("title", ""))
            items.append(
                RawScrapedItem(
                    source=SourceType.GOOGLE_NEWS,
                    ticker=self.ticker.symbol,
                    raw_title=title,
                    raw_url=entry.get("link"),
                    raw_published_at=entry.get("published"),
                    extra={"source_name": source_name} if source_name else {},
                )
            )

        logger.info(
            "%s: %d item, zaman filtresiyle %d tanesi elendi",
            self.ticker.symbol, len(feed.entries), skipped,
        )

        await self._enrich_with_article_bodies(items)
        # An empty feed (0 results) is a normal outcome, not an error —
        # returning an empty list here, no exception.
        return items

    async def _enrich_with_article_bodies(self, items: list[RawScrapedItem]) -> None:
        """Mutates the newest MAX_ARTICLES_TO_ENRICH_PER_TICKER items in
        place, filling raw_body_snippet with real article text where
        possible. `items` is already newest-first (feed order), so a plain
        slice picks the most recent ones."""
        enriched = 0
        failed = 0
        for item in items[:MAX_ARTICLES_TO_ENRICH_PER_TICKER]:
            if not item.raw_url:
                continue
            body = await fetch_article_body(self, item.raw_url, is_google_news=True)
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
    def _split_title_and_source(raw_title: str) -> tuple[str, str | None]:
        """Splits "Headline - Source Name" from the right, since a headline
        itself may legitimately contain " - "."""
        if _TITLE_SOURCE_SEPARATOR not in raw_title:
            return raw_title.strip(), None
        title, source_name = raw_title.rsplit(_TITLE_SOURCE_SEPARATOR, 1)
        return title.strip(), source_name.strip() or None
