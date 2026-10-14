"""
Digest memory: what earlier digests already reported, so a story is neither
repeated the next morning nor missed after a day without a mail.

Before this, every run looked back a fixed 36 hours (KAP: 2 days). A filing
from yesterday afternoon was therefore in two consecutive mails, and a day
without a mail (KAP unreachable, a dropped trigger) silently lost whatever
fell outside the 36-hour window.

Now:
  - the look-back window runs from the last successful send (plus a margin),
    at least NEWS_LOOKBACK_HOURS and at most MAX_LOOKBACK_HOURS;
  - items whose link (or, for news articles, whose title) was in a mail of
    the last KEEP_DAYS days are dropped before the LLM sees them.

The file lives in .digest-memory/ and is carried between runs by the
workflow's Actions cache; only a run that actually sent the mail saves it,
so a dry run never marks anything as reported.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rapidfuzz import fuzz

from config.constants import NEWS_LOOKBACK_HOURS, SourceType
from scrapers.models import DigestRun, NewsItem

logger = logging.getLogger(__name__)

MEMORY_PATH = Path(".digest-memory/memory.json")
KEEP_DAYS = 14
MARGIN_HOURS = 6
MAX_LOOKBACK_HOURS = 7 * 24
TITLE_SIMILARITY = 90
MIN_TITLE_WORDS = 5

# KAP form titles are the same every time ("Pay Dışında Sermaye Piyasası
# Aracı İşlemlerine İlişkin Bildirim (Faiz İçeren)"), on KAP and in Google's
# and Bigpara's copies alike; for these only the link identifies the filing.
_FORM_TITLE = re.compile(r"(bildirim|açıklama(sı)?|ilanı|belgesi|takvim)\)*\s*$", re.IGNORECASE)

_lookback_hours: int = NEWS_LOOKBACK_HOURS


def news_lookback_hours() -> int:
    """The window every scraper uses for this run (set by load_memory)."""
    return _lookback_hours


def kap_lookback_days() -> int:
    return max(2, math.ceil(_lookback_hours / 24))


def _normalize_url(url: str) -> str:
    url = url.strip().lower().split("#")[0]
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url.rstrip("/")


def _title_key(title: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", title.lower()).split())


def _title_is_specific(item: NewsItem) -> bool:
    return (
        item.source != SourceType.KAP
        and "/kap-haberleri/" not in str(item.url or "")
        and not _FORM_TITLE.search(item.title)
        and len(item.title.split()) >= MIN_TITLE_WORDS
    )


class DigestMemory:
    def __init__(self, last_sent_at: datetime | None = None, entries: list[dict] | None = None) -> None:
        self.last_sent_at = last_sent_at
        self.entries = entries or []
        # IR page URL -> document links seen on it (scrapers/ir_scraper.py)
        self.ir_seen: dict[str, list[str]] = {}
        self._inputs: dict[str, list[NewsItem]] = {}

    def note_inputs(self, symbol: str, items: list[NewsItem]) -> None:
        """The items this run's LLM call for `symbol` was given."""
        self._inputs[symbol] = items

    # --- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path: Path = MEMORY_PATH) -> "DigestMemory":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.info("Hafıza: kayıt yok (ilk çalışma ya da cache bulunamadı)")
            return cls()
        except (OSError, ValueError) as exc:
            logger.warning("Hafıza okunamadı, boş başlanıyor: %s", exc)
            return cls()
        last = data.get("last_sent_at")
        memory = cls(datetime.fromisoformat(last) if last else None, data.get("entries", []))
        memory.ir_seen = data.get("ir_seen", {})
        logger.info("Hafıza: son gönderim %s, %d kayıt", memory.last_sent_at, len(memory.entries))
        return memory

    def save(self, path: Path = MEMORY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_sent_at": self.last_sent_at.isoformat() if self.last_sent_at else None,
            "entries": self.entries,
            "ir_seen": self.ir_seen,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- window ----------------------------------------------------------

    def lookback_hours(self, now: datetime | None = None) -> int:
        if self.last_sent_at is None:
            return NEWS_LOOKBACK_HOURS
        now = now or datetime.now(timezone.utc)
        since = (now - self.last_sent_at).total_seconds() / 3600 + MARGIN_HOURS
        return int(min(MAX_LOOKBACK_HOURS, max(NEWS_LOOKBACK_HOURS, math.ceil(since))))

    # --- filtering -------------------------------------------------------

    def _seen(self, item: NewsItem) -> dict | None:
        urls = {_normalize_url(str(u)) for u in (item.url, item.related_kap_url) if u}
        for entry in self.entries:
            if entry.get("url") and _normalize_url(entry["url"]) in urls:
                return entry
        if not _title_is_specific(item):
            return None
        key = _title_key(item.title)
        for entry in self.entries:
            if entry.get("ticker") == item.ticker and entry.get("title_specific") and \
                    fuzz.token_sort_ratio(_title_key(entry["title"]), key) >= TITLE_SIMILARITY:
                return entry
        return None

    def drop_reported(self, items: list[NewsItem], symbol: str) -> list[NewsItem]:
        kept = []
        for item in items:
            entry = self._seen(item)
            if entry is None:
                kept.append(item)
            else:
                logger.info("%s: hafıza: %s tarihli bültende vardı, elendi: %s", symbol, entry.get("sent_on"), item.title)
        return kept

    # --- recording -------------------------------------------------------

    def record(self, digest_run: DigestRun, now: datetime | None = None) -> None:
        """Remembers every input item a section of the mail was written from
        (an item the model skipped stays eligible: tomorrow it may come with
        the text it lacked today), then prunes entries older than KEEP_DAYS."""
        now = now or datetime.now(timezone.utc)
        sent_on = now.date().isoformat()
        for report in digest_run.company_reports:
            cited = {str(u) for _, sections in report.ordered_sections() for s in sections for u in s.source_urls}
            for item in self._inputs.get(report.ticker, []):
                urls = [str(u) for u in (item.url, item.related_kap_url) if u]
                if not any(u in cited for u in urls):
                    continue
                for url in urls:
                    self.entries.append({
                        "ticker": item.ticker, "url": url, "title": item.title,
                        "title_specific": _title_is_specific(item), "sent_on": sent_on,
                    })
        cutoff = (now - timedelta(days=KEEP_DAYS)).date().isoformat()
        self.entries = [e for e in self.entries if e.get("sent_on", "") >= cutoff]
        self.last_sent_at = now


def load_memory(enabled: bool = True) -> DigestMemory:
    """Loads the memory and sets this run's look-back window from it."""
    global _lookback_hours
    memory = DigestMemory.load() if enabled else DigestMemory()
    if not enabled:
        logger.info("Hafıza bu çalışmada kapalı (IGNORE_MEMORY)")
    _lookback_hours = memory.lookback_hours()
    logger.info("Haber penceresi: son %d saat (KAP: %d gün)", _lookback_hours, kap_lookback_days())
    return memory
