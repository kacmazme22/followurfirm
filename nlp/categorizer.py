"""
Tier-1 rule-based categorizer: RawScrapedItem -> NewsItem, using the keyword
map in config/constants.py. Zero LLM calls — this is the always-on pass;
nlp/providers/ (Tier 2) only kicks in when this leaves too many items
unclassified, per config.ai_synthesis.llm_trigger.
"""

from __future__ import annotations

from datetime import datetime

from dateutil import parser as dateutil_parser

from config.constants import (
    BOILERPLATE_NOISE_PATTERNS,
    DEFAULT_CATEGORY,
    KAP_KEYWORD_CATEGORY_MAP,
    NewsCategory,
)
from scrapers.models import NewsItem, RawScrapedItem


def categorize(raw_item: RawScrapedItem) -> NewsItem | None:
    """Converts one RawScrapedItem into a NewsItem with a rule-based
    category. Returns None if the title matches a known boilerplate/noise
    pattern (KAP auto-generated disclaimer footers) — those are skipped
    entirely rather than assigned any category."""
    if _is_boilerplate(raw_item.raw_title):
        return None

    return NewsItem(
        ticker=raw_item.ticker,
        title=raw_item.raw_title,
        url=raw_item.raw_url,
        source=raw_item.source,
        published_at=_parse_published_at(raw_item.raw_published_at),
        body_snippet=raw_item.raw_body_snippet,
        category=_determine_category(raw_item),
        disclosure_type_raw=raw_item.raw_disclosure_type,
    )


def categorize_batch(raw_items: list[RawScrapedItem]) -> list[NewsItem]:
    """Runs categorize() over a list, dropping boilerplate items (None)."""
    result: list[NewsItem] = []
    for raw in raw_items:
        item = categorize(raw)
        if item is not None:
            result.append(item)
    return result


def _is_boilerplate(title: str) -> bool:
    lowered = title.lower()
    return any(pattern in lowered for pattern in BOILERPLATE_NOISE_PATTERNS)


def _determine_category(raw_item: RawScrapedItem) -> NewsCategory:
    """KAP items carry `raw_disclosure_type` — check that first since it's
    the cleaner, purpose-built signal. Falls back to the title for
    non-KAP sources (Bigpara, Google News) that have no disclosure type."""
    for text in (raw_item.raw_disclosure_type, raw_item.raw_title):
        if not text:
            continue
        category = _match_keyword_category(text)
        if category is not None:
            return category
    return DEFAULT_CATEGORY


def _match_keyword_category(text: str) -> NewsCategory | None:
    lowered = text.lower()
    for keyword, category in KAP_KEYWORD_CATEGORY_MAP.items():
        if keyword in lowered:
            return category
    return None


def _parse_published_at(raw_published_at: str | None) -> datetime | None:
    """Best-effort parse. KAP's "DD.MM.YYYY HH:MM:SS" parses fine with
    dayfirst=True; Bigpara's relative strings ("2 sa önce") and anything
    else unparseable is left as None rather than raising."""
    if not raw_published_at:
        return None
    try:
        return dateutil_parser.parse(raw_published_at, dayfirst=True)
    except (ValueError, OverflowError, TypeError):
        return None
