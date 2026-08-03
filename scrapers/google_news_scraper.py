"""
Google News RSS scraper — the lowest-risk source: no JS rendering, no
undocumented endpoints, just a public RSS feed parsed with `feedparser`.

Query building uses `sources.google_news_rss.query_template_tr` from
config.yaml (default `"{company_name} {ticker} hisse"`), filled in with the
ticker's `name`/`symbol` from `TickerConfig` — no hardcoded query strings.
"""

from __future__ import annotations

from urllib.parse import quote

import feedparser

from config.constants import SourceType
from config.settings import GoogleNewsRssConfig, PolitenessConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem

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

        items: list[RawScrapedItem] = []
        for entry in feed.entries:
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
        # An empty feed (0 results) is a normal outcome, not an error —
        # returning an empty list here, no exception.
        return items

    @staticmethod
    def _split_title_and_source(raw_title: str) -> tuple[str, str | None]:
        """Splits "Headline - Source Name" from the right, since a headline
        itself may legitimately contain " - "."""
        if _TITLE_SOURCE_SEPARATOR not in raw_title:
            return raw_title.strip(), None
        title, source_name = raw_title.rsplit(_TITLE_SOURCE_SEPARATOR, 1)
        return title.strip(), source_name.strip() or None
