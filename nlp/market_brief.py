"""
"Piyasa Gündemi": the market-wide box at the top of the digest (policy
rates, inflation prints, BIST index changes, regulation, big global moves).

Headlines only, from a handful of Google News RSS queries
(config.yaml sources.market_brief). Article bodies aren't fetched: Google
News body enrichment has never worked in CI (see utils/article_fetcher.py)
and the LLM only needs the headline to write a one-line agenda item. The
same market-noise rules as the company feeds drop technical analysis and
price-move headlines before the LLM sees them.

Never raises: any failure just means no market box today.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import feedparser
import httpx
from rapidfuzz import fuzz

from config.constants import SourceType
from config.settings import AppConfig
from nlp.providers.base import SummarizerProvider
from nlp.relevance import is_market_noise, is_multi_ticker_list
from scrapers.google_news_scraper import GOOGLE_NEWS_RSS_BASE, GoogleNewsScraper
from scrapers.models import NewsItem, SynthesizedSection

logger = logging.getLogger(__name__)

LOOKBACK_HOURS = 24
# Same story from several outlets: "TCMB faizi yüzde 38'e indirdi" vs
# "Merkez Bankası faizi 300 baz puan indirdi - X".
DUPLICATE_TITLE_SCORE = 85
QUERY_SPACING_SECONDS = 2.0
MAX_LINKS_PER_ITEM = 2


async def _get_feed(client: httpx.AsyncClient, url: str, query: str) -> httpx.Response | None:
    """Google News answered back-to-back queries with 503s and timeouts on
    2026-10-05; space the queries out and retry each once."""
    for attempt in (1, 2):
        await asyncio.sleep(QUERY_SPACING_SECONDS * attempt)
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            logger.warning(
                "Piyasa gündemi sorgusu başarısız (%s, deneme %d/2): %s: %s", query, attempt, type(exc).__name__, exc
            )
    return None


async def fetch_market_headlines(settings: AppConfig) -> list[NewsItem]:
    config = settings.yaml.sources.market_brief
    rss = settings.yaml.sources.google_news_rss
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    headlines: list[NewsItem] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=15.0), follow_redirects=True) as client:
        for query in config.queries:
            url = (
                f"{GOOGLE_NEWS_RSS_BASE}?q={quote(query)}"
                f"&hl={rss.language}&gl={rss.country}&ceid={rss.country}:{rss.language}"
            )
            response = await _get_feed(client, url, query)
            if response is None:
                continue

            for entry in feedparser.parse(response.text).entries:
                published = entry.get("published_parsed")
                if published is not None and datetime(*published[:6], tzinfo=timezone.utc) < cutoff:
                    continue
                title, _ = GoogleNewsScraper._split_title_and_source(entry.get("title", ""))
                if not title or is_market_noise(title) or is_multi_ticker_list(title):
                    continue
                if any(fuzz.token_set_ratio(title, h.title) >= DUPLICATE_TITLE_SCORE for h in headlines):
                    continue
                headlines.append(
                    NewsItem(ticker="PIYASA", title=title, url=entry.get("link"), source=SourceType.GOOGLE_NEWS)
                )

    logger.info("Piyasa gündemi: %d başlık toplandı", len(headlines))
    return headlines[: config.max_headlines]


async def build_market_brief(settings: AppConfig, provider: SummarizerProvider) -> list[SynthesizedSection]:
    if not settings.yaml.sources.market_brief.enabled:
        return []
    try:
        headlines = await fetch_market_headlines(settings)
        if not headlines:
            return []
        sections = await provider.summarize_market(headlines)
        # One market item came back with six near-identical Google links.
        for section in sections:
            section.source_urls = section.source_urls[:MAX_LINKS_PER_ITEM]
        return sections
    except Exception as exc:  # the box is optional; never sink the digest
        logger.error("Piyasa gündemi oluşturulamadı: %s: %s", type(exc).__name__, exc, exc_info=True)
        return []
