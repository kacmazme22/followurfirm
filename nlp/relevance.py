"""
Rule-based relevance filter, run on RawScrapedItems before categorization.

Why it exists (diagnosed from the 2026-10-03 production run): Bigpara's
per-ticker news page is not actually per-ticker. Alongside real company news
it lists market-wide items that merely *mention* the ticker somewhere in the
body — broker morning bulletins ("Analiz: Günlük Bülten - Ziraat Yatırım")
and multi-stock KAP relays ("ESCOM, AKYHO, HATSN, KOCMT, NATEN, ..."). With
AI_PROVIDER=noop those reached the email verbatim, with a trafilatura dump
of a whole-market data table as their "summary".

Two rules, applied to non-KAP sources only (KAP items are already matched
to the ticker by KAP's own stockCodes field, see kap_scraper.py):

  1. Multi-stock lists: a title containing a run of MULTI_TICKER_RUN or more
     ticker-shaped codes in a row is a market-wide list, never company news.
  2. Mentions: the title must name the ticker or the company (see
     _company_keywords) — the company being somewhere in a bulletin's body
     isn't enough.

Pure string rules, zero network/LLM cost, and deliberately conservative:
dropping a real headline that names neither the company nor its ticker is
an accepted trade-off against mailing a market-data dump every morning.
"""

from __future__ import annotations

import logging
import re

from config.constants import SourceType
from config.settings import TickerConfig
from scrapers.models import RawScrapedItem

logger = logging.getLogger(__name__)

MULTI_TICKER_RUN = 4

# 4-6 uppercase letters/digits as a standalone token: BIST ticker shape.
_TICKER_TOKEN = r"[A-ZÇĞİÖŞÜ0-9]{4,6}"
_MULTI_TICKER_RE = re.compile(
    rf"(?<![\wÇĞİÖŞÜçğıöşü]){_TICKER_TOKEN}(?:[\s,/;|-]+{_TICKER_TOKEN}){{{MULTI_TICKER_RUN - 1},}}(?![\wÇĞİÖŞÜçğıöşü])"
)

# Words too generic to identify a company on their own.
_GENERIC_NAME_WORDS = {
    "a.ş.", "aş", "a.s.", "as", "t.a.ş.", "holding", "sanayi", "ve", "ticaret",
    "fabrikaları", "teknoloji", "bankası", "yatırım", "enerji", "grup", "group",
}


def tr_lower(text: str) -> str:
    """Turkish-aware lowercase: plain str.lower() turns 'İ' into 'i̇' (i plus
    a combining dot) and 'I' into 'i' instead of 'ı', so 'KOÇ HOLDİNG' and
    'Koç Holding' wouldn't compare equal."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def _company_keywords(ticker: TickerConfig) -> set[str]:
    """Words that identify the company in a headline: the ticker symbol, the
    full company name, and the first distinctive word of the name
    ("Koç Holding" -> "koç", "Gübre Fabrikaları" -> "gübre")."""
    keywords = {tr_lower(ticker.symbol), tr_lower(ticker.name)}
    for word in tr_lower(ticker.name).split():
        if word not in _GENERIC_NAME_WORDS and len(word) >= 3:
            keywords.add(word)
            break
    return keywords


def _mentions_company(title: str, keywords: set[str]) -> bool:
    lowered = tr_lower(title)
    return any(
        re.search(rf"(?<![\wçğıöşü]){re.escape(k)}", lowered) for k in keywords
    )


def is_multi_ticker_list(title: str) -> bool:
    return _MULTI_TICKER_RE.search(title) is not None


def filter_relevant(items: list[RawScrapedItem], ticker: TickerConfig) -> list[RawScrapedItem]:
    keywords = _company_keywords(ticker)
    kept: list[RawScrapedItem] = []
    dropped_multi = dropped_unrelated = 0

    for item in items:
        if item.source == SourceType.KAP:
            kept.append(item)
        elif is_multi_ticker_list(item.raw_title):
            dropped_multi += 1
        elif not _mentions_company(item.raw_title, keywords):
            dropped_unrelated += 1
        else:
            kept.append(item)

    if dropped_multi or dropped_unrelated:
        logger.info(
            "%s: alaka filtresi %d çok-hisseli liste, %d şirketi anmayan haber eledi",
            ticker.symbol, dropped_multi, dropped_unrelated,
        )
    return kept
