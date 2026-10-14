"""
Bing News RSS scraper: the source that actually gets article text.

Why it exists (2026-10-08..14 production logs): Google News headlines arrive
fine, but every one of their article bodies failed — googlenewsdecoder
returned "no URL" for all 26 attempts, so the LLM only ever saw headlines.
That is why the digest said "faiz kararı beklentileri netleşti" without the
expectation, or "GUBRF temettü dağıtacak" without the amount and date.

Bing's RSS links are `bing.com/news/apiclick.aspx?...&url=<real article>`,
so the publisher URL can be read straight from the query string — no
reverse-engineered decoder — and each item also carries a short description,
which is kept as the body when the article page itself can't be fetched.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, quote, urlparse

import feedparser
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from config.constants import MAX_ARTICLES_TO_ENRICH_PER_TICKER, SourceType
from config.settings import BingNewsConfig, PolitenessConfig, TickerConfig
from nlp.relevance import filter_relevant
from scrapers.base import BaseScraper
from scrapers.models import NewsItem, RawScrapedItem
from utils.memory import news_lookback_hours
from utils.article_fetcher import fetch_article_body

logger = logging.getLogger(__name__)

BING_NEWS_RSS_BASE = "https://www.bing.com/news/search"


def real_article_url(link: str | None) -> str | None:
    """bing.com/news/apiclick.aspx?...&url=https%3a%2f%2f... -> the article URL;
    anything else is returned unchanged."""
    if not link:
        return None
    parsed = urlparse(link)
    if "bing.com" in parsed.netloc:
        target = parse_qs(parsed.query).get("url")
        if target and target[0].startswith("http"):
            return target[0]
    return link


# Newest first: by relevance, 10-12 of a company's 12 results were older
# than the look-back window (2026-10-14).
SORT_BY_DATE = "&qft=" + quote('sortbydate="1"')

# Google News headlines arrive without text (its links can't be decoded);
# the same story is usually on Bing too, with the publisher URL.
TITLE_MATCH_SIMILARITY = 85
MAX_BODY_LOOKUPS_PER_TICKER = 5


def feed_url(query: str, config: BingNewsConfig, newest_first: bool = True) -> str:
    url = f"{BING_NEWS_RSS_BASE}?q={quote(query)}&format=rss&setlang={config.language}&cc={config.country}"
    return url + SORT_BY_DATE if newest_first else url


def description_text(entry) -> str | None:
    """RSS description, HTML stripped; None when it's just a stub."""
    raw = entry.get("description") or entry.get("summary") or ""
    text = " ".join(BeautifulSoup(raw, "html.parser").get_text(" ").split())
    return text if len(text) >= 40 else None


class BingNewsScraper(BaseScraper):
    """Fetches the Bing News RSS feed for one ticker."""

    def __init__(
        self,
        ticker: TickerConfig,
        source_config: BingNewsConfig,
        politeness: PolitenessConfig,
        http_client=None,
    ) -> None:
        super().__init__(ticker, politeness, http_client)
        self.source_config = source_config

    async def fetch_raw(self) -> list[RawScrapedItem]:
        query = self.source_config.query_template.format(company_name=self.ticker.name, ticker=self.ticker.symbol)
        response = await self._get(feed_url(query, self.source_config))
        feed = feedparser.parse(response.text)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=news_lookback_hours())
        items: list[RawScrapedItem] = []
        skipped = 0
        for entry in feed.entries:
            published = entry.get("published_parsed")
            if published is not None and datetime(*published[:6], tzinfo=timezone.utc) < cutoff:
                skipped += 1
                continue
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            items.append(
                RawScrapedItem(
                    source=SourceType.BING_NEWS,
                    ticker=self.ticker.symbol,
                    raw_title=title,
                    raw_url=real_article_url(entry.get("link")),
                    raw_published_at=entry.get("published"),
                    raw_body_snippet=description_text(entry),
                )
            )

        logger.info("%s: Bing %d item, zaman filtresiyle %d tanesi elendi", self.ticker.symbol, len(feed.entries), skipped)
        items = filter_relevant(items, self.ticker)
        await self._enrich_with_article_bodies(items)
        return items

    async def fill_headline_only(self, items: list[NewsItem]) -> None:
        """For headline-only items (Google News), finds the same headline on
        Bing and takes its publisher link, description and article text."""
        targets = [i for i in items if not i.body_snippet and i.source == SourceType.GOOGLE_NEWS]
        if not targets:
            return
        try:
            async with self._client_session():
                found = await self._lookup(targets[:MAX_BODY_LOOKUPS_PER_TICKER])
        except Exception as exc:  # optional enrichment: never fails the ticker
            logger.warning("%s: Bing başlık araması durdu: %s", self.ticker.symbol, exc)
            return
        logger.info(
            "%s: yalnızca başlık olan %d Google haberinin %d tanesine Bing'den metin bulundu",
            self.ticker.symbol, len(targets), found,
        )

    async def _lookup(self, targets: list[NewsItem]) -> int:
        found = 0
        for item in targets:
            try:
                response = await self._get(feed_url(f'"{item.title}"', self.source_config, newest_first=False))
            except Exception as exc:
                logger.info("Bing başlık araması başarısız (%s): %s", item.title, exc)
                continue
            entries = feedparser.parse(response.text).entries
            best = max(
                entries, key=lambda e: fuzz.token_set_ratio(e.get("title", ""), item.title), default=None
            )
            if best is None or fuzz.token_set_ratio(best.get("title", ""), item.title) < TITLE_MATCH_SIMILARITY:
                continue
            url = real_article_url(best.get("link"))
            body = description_text(best)
            if url:
                article = await fetch_article_body(self, url, is_google_news=False)
                if article and len(article) > len(body or ""):
                    body = article
                item.url = url
            if body:
                item.body_snippet = body[:2000]
                found += 1
        return found

    async def _enrich_with_article_bodies(self, items: list[RawScrapedItem]) -> None:
        """The RSS description stays as the body unless the article page
        yields something longer."""
        enriched = failed = 0
        for item in items[:MAX_ARTICLES_TO_ENRICH_PER_TICKER]:
            if not item.raw_url:
                continue
            body = await fetch_article_body(self, item.raw_url, is_google_news=False)
            if body and len(body) > len(item.raw_body_snippet or ""):
                item.raw_body_snippet = body
                enriched += 1
            else:
                failed += 1
        logger.info(
            "%s: Bing %d makale metni alındı, %d tanesinde yalnızca RSS özeti kaldı",
            self.ticker.symbol, enriched, failed,
        )
