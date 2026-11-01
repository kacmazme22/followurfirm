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
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import feedparser
from rapidfuzz import fuzz

from config.constants import MAX_ARTICLES_TO_ENRICH_PER_TICKER, SourceType
from config.settings import GoogleNewsRssConfig, PolitenessConfig, TickerConfig
from scrapers.base import BaseScraper
from nlp.relevance import filter_relevant
from scrapers.models import NewsItem, RawScrapedItem
from utils.memory import kap_lookback_days, news_lookback_hours
from utils.article_fetcher import StaleArticle, decode_google_urls, fetch_article_body, is_refusing, site_of

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"
# Looking for an unreadable story on other outlets (find_text_elsewhere).
MAX_ALT_LOOKUPS_PER_TICKER = 10
FOUND, UNREADABLE, NO_RECENT_COPY = "found", "unreadable", "no_recent_copy"
ALT_QUERY_WORDS = 8
ALT_TITLE_SIMILARITY = 80
ALT_CANDIDATES = 4

# Google News RSS titles are consistently "Headline - Source Name"; the
# separator it uses to append the source name at the end of the title.
_TITLE_SOURCE_SEPARATOR = " - "


def _published_sort_key(item: RawScrapedItem) -> float:
    try:
        return parsedate_to_datetime(item.raw_published_at).timestamp() if item.raw_published_at else 0.0
    except (TypeError, ValueError):
        return 0.0


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
        # Google's own "when:" filter must not be narrower than the window
        # since the last mail (utils/memory.py).
        query = re.sub(r"when:\d+d", f"when:{kap_lookback_days()}d", query)
        url = (
            f"{GOOGLE_NEWS_RSS_BASE}?q={quote(query)}"
            f"&hl={self.source_config.language}"
            f"&gl={self.source_config.country}"
            f"&ceid={self.source_config.country}:{self.source_config.language}"
        )

        response = await self._get(url)
        feed = feedparser.parse(response.text)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=news_lookback_hours())

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

        items = filter_relevant(items, self.ticker)
        # Google orders the feed by relevance; the article texts (and the
        # page-date check against re-dated old pages) go to the newest first.
        items.sort(key=_published_sort_key, reverse=True)
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
        targets = [item for item in items[:MAX_ARTICLES_TO_ENRICH_PER_TICKER] if item.raw_url]
        decoded = await decode_google_urls([item.raw_url for item in targets])
        for item, publisher_url in zip(targets, decoded):
            if not publisher_url:
                failed += 1
                continue
            # The publisher's link instead of a 250-character Google token:
            # a better link in the mail, and the same story from another
            # feed now has the same URL for dedup and the digest memory.
            item.raw_url = publisher_url
            try:
                body = await fetch_article_body(self, publisher_url, is_google_news=False, check_age=True)
            except StaleArticle:
                items.remove(item)
                continue
            if body:
                item.raw_body_snippet = body
                enriched += 1
            else:
                failed += 1

        logger.info(
            "%s: %d haber zenginleştirildi, %d'si başarısız/boş döndü",
            self.ticker.symbol, enriched, failed,
        )

    async def find_text_elsewhere(self, items: list[NewsItem]) -> None:
        """A headline whose own page couldn't be read (blocked site, no
        text) is searched again on Google News: the same story is nearly
        always carried by several outlets, and the first one that can be
        read gives the text and the link. The bare headline is the last
        resort (user's request, 2026-11-01)."""
        targets = [i for i in items if not i.body_snippet and i.source != SourceType.KAP][:MAX_ALT_LOOKUPS_PER_TICKER]
        if not targets:
            return
        found = stale = 0
        try:
            async with self._client_session():
                for item in targets:
                    outcome = await self._read_elsewhere(item)
                    found += outcome == FOUND
                    if outcome == NO_RECENT_COPY and item.source == SourceType.GOOGLE_NEWS:
                        # Not even the story itself is in Google's last 7
                        # days: an old page the feed re-dated ("Akbank
                        # Düşünce Kulübü 10 yaşında", 2026-11-01, from a site
                        # that refused the page-date check).
                        items.remove(item)
                        stale += 1
                        logger.info("%s: son 7 günde hiçbir kanalda yok, eski haber sayılıp çıkarıldı: %s",
                                    self.ticker.symbol, item.title[:100])
        except Exception as exc:  # optional enrichment: never fails the ticker
            logger.warning("%s: başka kanalda arama durdu: %s", self.ticker.symbol, exc)
        logger.info(
            "%s: metni alınamayan %d haberin %d tanesi başka bir kanalda bulundu, %d tanesi eski haber olarak çıkarıldı",
            self.ticker.symbol, len(targets), found, stale,
        )

    async def _read_elsewhere(self, item: NewsItem) -> str:
        words = re.sub(r"[^\w\s%,.]", " ", item.title).split()[:ALT_QUERY_WORDS]
        query = f"{' '.join(words)} when:7d"
        url = (
            f"{GOOGLE_NEWS_RSS_BASE}?q={quote(query)}&hl={self.source_config.language}"
            f"&gl={self.source_config.country}&ceid={self.source_config.country}:{self.source_config.language}"
        )
        response = await self._get(url)
        candidates = []
        for entry in feedparser.parse(response.text).entries:
            title, _ = self._split_title_and_source(entry.get("title", ""))
            if entry.get("link") and fuzz.token_set_ratio(title, item.title) >= ALT_TITLE_SIMILARITY:
                candidates.append(entry["link"])
            if len(candidates) >= ALT_CANDIDATES:
                break
        if not candidates:
            return NO_RECENT_COPY
        decoded = [u for u in await decode_google_urls(candidates) if u]
        # Copies on the item's own site don't count: Google re-dates a
        # site's old pages in the search too (haber7's 2016 Akbank stories
        # came back as "recent" copies of themselves, 2026-11-01). A story
        # no other outlet carries is treated as having no recent copy.
        own_site = site_of(str(item.url)) if "news.google.com" not in str(item.url) else None
        independent = [u for u in decoded if site_of(u) != own_site]
        if not independent:
            return NO_RECENT_COPY
        for publisher_url in independent:
            if publisher_url == str(item.url) or is_refusing(publisher_url):
                continue
            try:
                body = await fetch_article_body(self, publisher_url, is_google_news=False, check_age=True)
            except StaleArticle:
                continue
            if body:
                item.body_snippet = body[:2000]
                item.url = publisher_url
                return FOUND
        return UNREADABLE

    @staticmethod
    def _split_title_and_source(raw_title: str) -> tuple[str, str | None]:
        """Splits "Headline - Source Name" from the right, since a headline
        itself may legitimately contain " - "."""
        if _TITLE_SOURCE_SEPARATOR not in raw_title:
            return raw_title.strip(), None
        title, source_name = raw_title.rsplit(_TITLE_SOURCE_SEPARATOR, 1)
        return title.strip(), source_name.strip() or None
