"""
Audit trail in the workflow log: what went into the LLM and which inputs
each digest line was written from.

Why: on 2026-10-06 the market box said "BIST 100 ... yeni zirve kaydetti",
which wasn't true, and there was no way to tell from the log which headline
(if any) it came from — the headlines sent to Groq were never logged. With
this, every line of the email can be traced back to its source titles in
the Actions log, which is also the only place they can be read from outside
GitHub's runners.
"""

from __future__ import annotations

import logging

from scrapers.models import NewsItem, SynthesizedSection

logger = logging.getLogger("audit")

BODY_PREVIEW_CHARS = 160


# KAP filings (and Bigpara's copies of them) are logged in full: on
# 2026-10-06 the same Akbank bond filing came out as "50,0 mn USD" in one run
# and "5,0 mn USD, EUR" in the next, and a 160-char preview couldn't show
# which the source actually said.
FULL_BODY_URL_PATHS = ("/kap-haberleri/", "kap.org.tr")


def log_inputs(label: str, items: list[NewsItem]) -> None:
    for i, item in enumerate(items, start=1):
        full = item.source.value == "kap" or any(p in str(item.url or "") for p in FULL_BODY_URL_PATHS)
        limit = None if full else BODY_PREVIEW_CHARS
        body = " ".join(item.body_snippet.split())[:limit] if item.body_snippet else "(yalnızca başlık)"
        logger.info("[%s] girdi %d (%s): %s | %s", label, i, item.source.value, item.title, body)


def log_outputs(label: str, sections: list[SynthesizedSection], items: list[NewsItem]) -> None:
    titles_by_url: dict[str, str] = {}
    for item in items:
        for url in (item.url, getattr(item, "related_kap_url", None)):
            if url:
                titles_by_url.setdefault(str(url), item.title)
    for section in sections:
        cited = [titles_by_url.get(str(url), str(url)) for url in section.source_urls]
        logger.info(
            "[%s] çıktı «%s» <- %s", label, section.subheading, " || ".join(cited) if cited else "(kaynak yok)"
        )
