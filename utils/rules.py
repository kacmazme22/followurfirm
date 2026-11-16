"""
Digest rules from the 2-15 November review, each behind a switch in
utils/experiments.py. Every rule reports what it did (or, in shadow mode,
would have done) through experiments.note(), which ends up in the mail's
"Deneme kayıtları" box.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from rapidfuzz import fuzz

from scrapers.models import NewsItem, SynthesizedCompanyReport, SynthesizedSection
from utils import experiments
from utils.factcheck import source_numbers, unsourced

# --- buyback_weekly ---------------------------------------------------------

BUYBACK_QUIET_DAYS = 6
_BUYBACK_LABEL = re.compile(r"geri al[ıi]m", re.IGNORECASE)
_DECISION_DATE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b")
_BIG_AMOUNT = re.compile(r"\d+(?:,\d+)?\s?(?:mlr|milyar)\b", re.IGNORECASE)


def _is_buyback(section: SynthesizedSection) -> bool:
    return bool(_BUYBACK_LABEL.search(section.subheading))


def _earlier_buybacks(memory, key: str, today: date) -> list[dict]:
    cutoff = (today - timedelta(days=BUYBACK_QUIET_DAYS)).isoformat()
    return [
        e for e in memory.reported
        if e.get("key") == key and _BUYBACK_LABEL.search(e.get("label", ""))
        and cutoff <= e.get("date", "") < today.isoformat()
    ]


def _program_changed(text: str, earlier: list[dict]) -> bool:
    """A new board decision date or a new billion-scale ceiling means the
    programme itself changed; daily trades only bring share counts."""
    before = " ".join(e["text"] for e in earlier)
    dates = set(_DECISION_DATE.findall(text)) - set(_DECISION_DATE.findall(before))
    amounts = {a.replace(" ", "") for a in _BIG_AMOUNT.findall(text)} - {
        a.replace(" ", "") for a in _BIG_AMOUNT.findall(before)
    }
    # Trade dates also look like dates: only count ones next to "karar".
    decision_dates = [d for d in dates if re.search(rf"{re.escape(d)}[^.]{{0,40}}karar|karar[^.]{{0,40}}{re.escape(d)}", text)]
    return bool(decision_dates or amounts)


def buyback_weekly(report: SynthesizedCompanyReport, memory, today: date) -> None:
    """Daily buyback trades went out as an item on 2, 5, 6, 11 and 12 Nov
    (GUBRF; three times as the subject). The user: "kaçıncı bu aynı
    başlık". A buyback item within BUYBACK_QUIET_DAYS of an earlier one is
    held back unless the programme itself changed."""
    if not experiments.active("buyback_weekly"):
        return
    earlier = _earlier_buybacks(memory, report.ticker, today)
    if not earlier:
        return
    for category, sections in list(report.sections_by_category.items()):
        kept = []
        for section in sections:
            if _is_buyback(section) and not _program_changed(section.narrative, earlier):
                experiments.note(
                    "buyback_weekly", report.ticker,
                    f"{earlier[-1]['date']} tarihinde yazılmış geri alımın günlük işlemi bültene alınmadı: {section.narrative[:160]}",
                )
                if experiments.applies("buyback_weekly"):
                    continue
            kept.append(section)
        if kept:
            report.sections_by_category[category] = kept
        else:
            del report.sections_by_category[category]


def recently_reported_buyback(memory, key: str, today: date) -> bool:
    return bool(_earlier_buybacks(memory, key, today))


# --- headline candidates and checks ---------------------------------------

_LOW_VALUE_LABELS = {"röportaj", "aracı kurum işlemleri", "özel emir işlemi", "sosyal sorumluluk", "pay dönüşümü"}
_LOW_VALUE_TEXT = re.compile(
    r"personel al|işe al[ıi]m|konferans|cop\d+|ödül|etkinlik|sponsor|zirve|panel|yaşında|film", re.IGNORECASE
)


def is_low_value_for_headline(section: SynthesizedSection) -> bool:
    """Kept in the mail (the user finds interviews and such worth reading)
    but never the subject: on 8 Nov "500 büyük şirket" opened the mail."""
    label = section.subheading.lower().replace("i̇", "i").rstrip(".")
    return label in _LOW_VALUE_LABELS or bool(_LOW_VALUE_TEXT.search(section.narrative[:300]))


_SUBJECT_TICKER = re.compile(r"^([A-ZÇĞİÖŞÜ]{4,6})\s*[:—-]")


def headline_from_own_items(headline: str, texts_by_key: dict[str, str]) -> bool:
    """True when the subject's figures appear in the items of the company it
    names. On 15 Nov "GUBRF: 177
    mlr TL ulaşım yatırımı" took a figure from the market box's road news."""
    match = _SUBJECT_TICKER.match(headline)
    if not match or match.group(1) not in texts_by_key:
        return True  # names no followed company ("TCMB: ..."): the whole mail is the source (check_line)
    key = match.group(1)
    own = texts_by_key[key]
    ok = not unsourced(headline, source_numbers(own))
    if not ok:
        experiments.note("headline_same_ticker", key, f"Konu satırının rakamı {key} maddelerinde yok: «{headline}»")
    return ok or not experiments.applies("headline_same_ticker")


# --- headline_only_dedup -------------------------------------------------------

_KEEP_HEADLINE = re.compile(r"analiz|rapor|hedef fiyat|yatırımcı ilişkileri sayfasında|sunum", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def drop_headline_copies(items: list[NewsItem], key: str) -> list[NewsItem]:
    """A bare headline that is a copy of an item which has text adds
    nothing: KCHOL had 63 inputs on 11 Nov, 40 of them bare headlines,
    mostly the same results story. Strict on purpose (checked on 14 days of
    logs: 14 of 160 bare headlines matched, all true copies): 5+ words,
    80+ similarity, no figure of its own, and never broker reports or IR
    documents, whose titles look alike across brokers."""
    if not experiments.active("headline_only_dedup"):
        return items
    with_text = [i for i in items if i.body_snippet and len(i.title.split()) >= 5]
    kept = []
    for item in items:
        copy_of = None
        if not item.body_snippet and len(item.title.split()) >= 5 and not _KEEP_HEADLINE.search(item.title):
            figures = set(_NUMBER.findall(item.title))
            for other in with_text:
                if fuzz.token_set_ratio(item.title.lower(), other.title.lower()) >= 80 and figures <= set(_NUMBER.findall(other.title)):
                    copy_of = other
                    break
        if copy_of is not None:
            experiments.note("headline_only_dedup", key, f"«{item.title[:90]}» = «{copy_of.title[:90]}»")
            if experiments.applies("headline_only_dedup"):
                continue
        kept.append(item)
    return kept


# --- same_day_merge ---------------------------------------------------------------

def same_day_merge(report: SynthesizedCompanyReport) -> None:
    """One event written as several items on one day under different labels
    (SAHOL 13 Nov: "6 mlr dolar elektrifikasyon" as Yatırım, Röportaj and
    Enerji dönüşümü). Merged only when the texts are alike AND share an
    amount, so two different deals on the same day stay apart."""
    if not experiments.active("same_day_merge"):
        return
    from utils.formatting import _append_paragraphs

    flat = [(category, s) for category, sections in report.ordered_sections() for s in sections]
    removed: set[int] = set()
    for i, (_, first) in enumerate(flat):
        if id(first) in removed:
            continue
        for _, second in flat[i + 1:]:
            if id(second) in removed:
                continue
            similar = fuzz.token_set_ratio(first.narrative.lower(), second.narrative.lower()) >= 75
            shared = not unsourced(second.narrative, source_numbers(first.narrative)) and bool(
                re.search(r"\d", second.narrative)
            )
            if not (similar and shared):
                continue
            experiments.note(
                "same_day_merge", report.ticker,
                f"«{first.subheading}» ile «{second.subheading}» aynı olay: {second.narrative[:120]}",
            )
            if experiments.applies("same_day_merge"):
                first.narrative = _append_paragraphs(first.narrative, second.narrative)
                first.source_urls = list(first.source_urls) + [u for u in second.source_urls if u not in first.source_urls]
                removed.add(id(second))
    if removed:
        report.sections_by_category = {
            category: kept
            for category, sections in report.ordered_sections()
            if (kept := [s for s in sections if id(s) not in removed])
        }


# --- kap_form_tail ----------------------------------------------------------------

_TRANSACTION_FORM = re.compile(r"geri al[ıi]n|pay al[ıi]m sat[ıi]m", re.IGNORECASE)
FORM_HEAD_CHARS = 800


def form_text(title: str, text: str, limit: int) -> str:
    """Buyback and share-transaction forms list trades oldest first, so the
    first `limit` characters held last year's rows: on 11 Nov GLYHO's form
    was cut before its 10 Nov trade and the mail gave a 2025 date (the
    user: "Son işlem 10 Kasım 2026'da gerçekleşmiş"). Such forms keep their
    head (the programme) and their tail (the latest trades)."""
    if len(text) <= limit or not _TRANSACTION_FORM.search(title) or not experiments.active("kap_form_tail"):
        return text[:limit]
    combined = f"{text[:FORM_HEAD_CHARS]} … {text[-(limit - FORM_HEAD_CHARS - 3):]}"
    experiments.note("kap_form_tail", "KAP", f"«{title[:80]}» formunun son {limit - FORM_HEAD_CHARS - 3} karakteri de okundu ({len(text)} karakterlik form)")
    return combined if experiments.applies("kap_form_tail") else text[:limit]


# --- calendar_near_days -----------------------------------------------------------

def calendar_events(events: list[dict], today: date) -> list[dict]:
    """`events`: the next 7 days. The "Şirket takvimi" line used to show
    them all every day, so YEOTK's 13 Nov payment stood in the box from 6
    to 13 Nov. Now: today and tomorrow; on Mondays the whole week."""
    if not experiments.active("calendar_near_days") or today.weekday() == 0:
        return events
    last = (today + timedelta(days=1)).isoformat()
    for event in events:
        if event["date"] > last:
            experiments.note("calendar_near_days", event["ticker"], f"{event['date']} tarihli «{event['label']}» takvimde henüz gösterilmedi")
    return [e for e in events if e["date"] <= last] if experiments.applies("calendar_near_days") else events


# --- llm_repeat_check ---------------------------------------------------------------

async def llm_repeat_check(report: SynthesizedCompanyReport, memory, provider) -> None:
    """Asks the model whether today's items repeat earlier mails. KCHOL's
    3rd-quarter profit went out on 10, 11, 12 and 14 Nov reworded each
    time, which the text-similarity check (find_repeat) can't see."""
    if not experiments.active("llm_repeat_check"):
        return
    earlier = memory.previously_reported(report.ticker)
    flat = [s for _, sections in report.ordered_sections() for s in sections]
    if not earlier or not flat:
        return
    pairs = await provider.find_same_events([f"{s.subheading}: {s.narrative}" for s in flat], earlier)
    drop = set()
    for today_index, earlier_index in pairs:
        section = flat[today_index]
        experiments.note(
            "llm_repeat_check", report.ticker,
            f"«{section.subheading}: {section.narrative[:120]}» önceki bültendeki «{earlier[earlier_index][:120]}» ile aynı olay",
        )
        drop.add(id(section))
    if drop and experiments.applies("llm_repeat_check"):
        report.sections_by_category = {
            category: kept
            for category, sections in report.ordered_sections()
            if (kept := [s for s in sections if id(s) not in drop])
        }
