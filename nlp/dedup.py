"""
Cross-source dedup for NewsItem lists. Two stages, both keeping the
higher-`SOURCE_PRIORITY` item (config/constants.py) and, on a tie, the item
seen first in the input order:

  1. Exact hash: items sharing `content_hash` (same URL, or same
     ticker+title when there's no URL — see NewsItem._make_hash) collapse to
     one.
  2. Fuzzy title: within the same ticker, titles above
     `similarity_threshold` (rapidfuzz token_sort_ratio) collapse to one —
     catches the same story reported by two sources with different URLs and
     slightly different wording.

Whenever a KAP item loses to a non-KAP item in either stage, its URL isn't
just discarded: it's carried onto the surviving item's `related_kap_url` so
the email can still link to the original disclosure.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from config.constants import SOURCE_PRIORITY, SourceType
from scrapers.models import NewsItem

DEFAULT_SIMILARITY_THRESHOLD = 88


def deduplicate(items: list[NewsItem], similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD) -> list[NewsItem]:
    exact_deduped = _dedup_exact_hash(items)
    return _dedup_fuzzy_title(exact_deduped, similarity_threshold)


def _priority(item: NewsItem) -> int:
    return SOURCE_PRIORITY.get(item.source, 0)


def _carry_kap_link(winner: NewsItem, loser: NewsItem) -> NewsItem:
    """Returns `winner`, updated with `loser`'s KAP link if `loser` is the
    one being dropped and `winner` doesn't already have one recorded."""
    if winner.related_kap_url is not None:
        return winner
    if loser.source == SourceType.KAP:
        return winner.model_copy(update={"related_kap_url": loser.url})
    if loser.related_kap_url is not None:
        return winner.model_copy(update={"related_kap_url": loser.related_kap_url})
    return winner


def _resolve_duplicate(a: NewsItem, b: NewsItem) -> NewsItem:
    """`a` is the item already kept, `b` is the new candidate found to be a
    duplicate of it. Returns whichever should survive, with the other's KAP
    link (if any) carried onto it. Ties keep `a` (first-seen)."""
    winner, loser = (b, a) if _priority(b) > _priority(a) else (a, b)
    return _carry_kap_link(winner, loser)


def _dedup_exact_hash(items: list[NewsItem]) -> list[NewsItem]:
    result: list[NewsItem] = []
    hash_to_index: dict[str, int] = {}

    for item in items:
        existing_index = hash_to_index.get(item.content_hash)
        if existing_index is None:
            hash_to_index[item.content_hash] = len(result)
            result.append(item)
        else:
            result[existing_index] = _resolve_duplicate(result[existing_index], item)

    return result


def _dedup_fuzzy_title(items: list[NewsItem], similarity_threshold: int) -> list[NewsItem]:
    kept: list[NewsItem] = []

    for item in items:
        match_index = next(
            (
                i
                for i, existing in enumerate(kept)
                if existing.ticker == item.ticker
                and fuzz.token_sort_ratio(existing.title, item.title) >= similarity_threshold
            ),
            None,
        )
        if match_index is None:
            kept.append(item)
        else:
            kept[match_index] = _resolve_duplicate(kept[match_index], item)

    return kept
