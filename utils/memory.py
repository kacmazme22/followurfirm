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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rapidfuzz import fuzz

from config.constants import NEWS_LOOKBACK_HOURS, SourceType
from scrapers.models import DigestRun, NewsItem

logger = logging.getLogger(__name__)

MEMORY_PATH = Path(".digest-memory/memory.json")
KEEP_DAYS = 14
MARGIN_HOURS = 6
MAX_LOOKBACK_HOURS = 7 * 24
REPORTED_DAYS = 7
REPORTED_CHARS = 220
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
        # IR page URL -> document links seen on it, and ticker -> the IR
        # pages found for it by search (scrapers/ir_scraper.py)
        self.ir_seen: dict[str, list[str]] = {}
        self.ir_pages: dict[str, dict] = {}
        # What the mail said (ticker or "PIYASA", date, label, text): the
        # same event reaches us again through another source, so links and
        # titles alone don't stop it (see previously_reported).
        self.reported: list[dict] = []
        # Dated company events announced in earlier mails (results dates).
        self.calendar: list[dict] = []
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
        memory.ir_pages = data.get("ir_pages", {})
        memory.reported = data.get("reported", [])
        memory.calendar = data.get("calendar", [])
        logger.info("Hafıza: son gönderim %s, %d kayıt", memory.last_sent_at, len(memory.entries))
        return memory

    def save(self, path: Path = MEMORY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_sent_at": self.last_sent_at.isoformat() if self.last_sent_at else None,
            "entries": self.entries,
            "ir_seen": self.ir_seen,
            "ir_pages": self.ir_pages,
            "reported": self.reported,
            "calendar": self.calendar,
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

    # --- what was already said --------------------------------------------

    def previously_reported(self, key: str, today: date | None = None) -> list[str]:
        """Lines for the prompt: what the last REPORTED_DAYS mails said about
        `key` (a ticker or "PIYASA"). On 2026-10-24..26 GUBRF's 7,25 mlr TL
        investment and SAHOL's CarrefourSA payment were written three days
        running, each time from a different news site."""
        today = today or datetime.now(timezone.utc).date()
        cutoff = (today - timedelta(days=REPORTED_DAYS)).isoformat()
        return [
            f"{date.fromisoformat(e['date']).strftime('%d.%m')} {e['label']}: {e['text']}"
            for e in self.reported
            if e.get("key") == key and cutoff <= e.get("date", "") < today.isoformat()
        ]

    def upcoming(self, today: date | None = None, days: int = 7) -> list[dict]:
        """Company events from earlier mails falling in the next `days`."""
        today = today or datetime.now(timezone.utc).date()
        end = (today + timedelta(days=days)).isoformat()
        seen, result = set(), []
        for event in sorted(self.calendar, key=lambda e: e["date"]):
            key = (event["ticker"], event["date"])
            if today.isoformat() <= event["date"] <= end and key not in seen:
                seen.add(key)
                result.append(event)
        return result

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
        self._record_texts(digest_run, sent_on)
        self.reported = [e for e in self.reported if e.get("date", "") >= cutoff]
        self.calendar = [e for e in self.calendar if e.get("date", "") >= sent_on]
        self.last_sent_at = now

    def _record_texts(self, digest_run: DigestRun, sent_on: str) -> None:
        def add(key: str, label: str, text: str) -> None:
            self.reported.append({"key": key, "date": sent_on, "label": label, "text": " ".join(text.split())[:REPORTED_CHARS]})

        for section in digest_run.market_brief:
            add("PIYASA", section.subheading, section.narrative)
        for report in digest_run.company_reports:
            for _, sections in report.ordered_sections():
                for section in sections:
                    add(report.ticker, section.subheading, section.narrative)
                    for event_date in _scheduled_dates(section.narrative, sent_on):
                        self.calendar.append({
                            "ticker": report.ticker, "date": event_date, "label": section.subheading,
                            "text": " ".join(section.narrative.split())[:160],
                        })


_MONTH_NUMBERS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
}
_TEXT_DATE = re.compile(r"\b(\d{1,2}) (" + "|".join(_MONTH_NUMBERS) + r")\w*(?: (\d{4}))?", re.IGNORECASE)
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
# Only announcements of something still to come: a results date, a general
# meeting, a dividend payment ("27 Ekim'de açıklayacak", "ödenecek").
_FUTURE_EVENT = re.compile(
    r"açıklayacak|açıklanacak|yayımlanacak|paylaşılacak|ödenecek|ödeyecek|toplanacak|yapılacak|planlan", re.IGNORECASE
)


def _scheduled_dates(text: str, sent_on: str) -> list[str]:
    """Future dates in a sentence announcing an upcoming event."""
    year = int(sent_on[:4])
    dates = []
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if not _FUTURE_EVENT.search(sentence):
            continue
        for day, month, yr in _TEXT_DATE.findall(sentence):
            dates.append((int(yr or year), _MONTH_NUMBERS[month.lower().replace("İ", "i")], int(day)))
        for day, month, yr in _NUMERIC_DATE.findall(sentence):
            dates.append((int(yr), int(month), int(day)))
    result = []
    for y, m, d in dates:
        try:
            iso = date(y, m, d).isoformat()
        except ValueError:
            continue
        if iso > sent_on and iso not in result:
            result.append(iso)
    return result


def load_memory(enabled: bool = True) -> DigestMemory:
    """Loads the memory and sets this run's look-back window from it."""
    global _lookback_hours
    memory = DigestMemory.load() if enabled else DigestMemory()
    if not enabled:
        logger.info("Hafıza bu çalışmada kapalı (IGNORE_MEMORY)")
    _lookback_hours = memory.lookback_hours()
    logger.info("Haber penceresi: son %d saat (KAP: %d gün)", _lookback_hours, kap_lookback_days())
    return memory
