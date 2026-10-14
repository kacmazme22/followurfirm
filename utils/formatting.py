"""
Deterministic clean-up of LLM output, applied after synthesis.

Both fixes come from the 2026-10-14 mail, and both are things a prompt only
gets right most of the time:
  - three separate "Borçlanma." rows and two "Dava." rows for Akbank instead
    of one heading each with the events as paragraphs below it;
  - "30,000,000,000 TRY", "173,751,000 TRY" (English digit grouping, ISO
    currency code) and "05 Oct 2026" in Turkish prose.
"""

from __future__ import annotations

import re

from scrapers.models import SynthesizedCompanyReport, SynthesizedSection

PARAGRAPH_SEPARATOR = "\n\n"

# 30,000,000,000 (English grouping) or 1.713.897.558 (Turkish grouping):
# 7+ digit integers with a consistent 3-digit group separator. Dates like
# 05.04.2028 don't match (their groups aren't 3 digits).
_GROUPED_NUMBER = re.compile(r"(?<![\d.,])(\d{1,3}(?:([.,])\d{3})(?:\2\d{3})+)(?![\d]|[.,]\d)")
_MONTHS_EN_TR = {
    "Jan": "Ocak", "Feb": "Şubat", "Mar": "Mart", "Apr": "Nisan", "May": "Mayıs", "Jun": "Haziran",
    "Jul": "Temmuz", "Aug": "Ağustos", "Sep": "Eylül", "Oct": "Ekim", "Nov": "Kasım", "Dec": "Aralık",
}
_EN_DATE = re.compile(r"\b(\d{1,2}) (" + "|".join(_MONTHS_EN_TR) + r")[a-z]* (\d{4})\b")


def _abbreviate(value: int) -> str:
    for size, unit in ((1_000_000_000, "mlr"), (1_000_000, "mn")):
        if value >= size:
            number = f"{value / size:.2f}".rstrip("0").rstrip(".").replace(".", ",")
            return f"{number} {unit}"
    return f"{value:,}".replace(",", ".")


# "1.713,9 mn" / "1,713,9 mn": the model's own half-conversion of a 10-digit
# amount (2026-10-14 subject line).
_THOUSANDS_OF_MILLIONS = re.compile(r"(?<![\d.,])(\d{1,3})[.,](\d{3})(?:,(\d+))? ?(mn|milyon)\b")


def turkish_numbers(text: str) -> str:
    """1.713.897.558 TL -> 1,71 mlr TL; 30,000,000,000 TRY -> 30 mlr TL;
    1.713,9 mn TL -> 1,71 mlr TL; 05 Oct 2026 -> 5 Ekim 2026."""
    def replace(match: re.Match) -> str:
        value = int(re.sub(r"[.,]", "", match.group(1)))
        return _abbreviate(value) if value >= 1_000_000 else match.group(1)

    def millions(match: re.Match) -> str:
        value = float(f"{match.group(1)}{match.group(2)}.{match.group(3) or 0}") * 1_000_000
        return _abbreviate(int(value))

    text = _GROUPED_NUMBER.sub(replace, text)
    text = _THOUSANDS_OF_MILLIONS.sub(millions, text)
    text = re.sub(r"\bTRY\b", "TL", text)
    text = _EN_DATE.sub(lambda m: f"{int(m.group(1))} {_MONTHS_EN_TR[m.group(2)]} {m.group(3)}", text)
    return text


def _label_key(label: str) -> str:
    return " ".join(label.lower().replace("i̇", "i").rstrip(".").split())


def merge_same_label(sections: list[SynthesizedSection]) -> list[SynthesizedSection]:
    """Sections with the same label become one: narratives as paragraphs,
    links merged (order kept)."""
    merged: dict[str, SynthesizedSection] = {}
    for section in sections:
        key = _label_key(section.subheading)
        if key not in merged:
            merged[key] = section.model_copy()
            continue
        target = merged[key]
        urls = list(target.source_urls) + [u for u in section.source_urls if u not in target.source_urls]
        target.narrative = target.narrative + PARAGRAPH_SEPARATOR + section.narrative
        target.source_urls = urls
    return list(merged.values())


def polish_sections(sections: list[SynthesizedSection]) -> list[SynthesizedSection]:
    result = merge_same_label(sections)
    for section in result:
        section.narrative = turkish_numbers(section.narrative)
        section.subheading = turkish_numbers(section.subheading)
    return result


def polish_report(report: SynthesizedCompanyReport) -> SynthesizedCompanyReport:
    """Same label across categories (e.g. "Dava" filed under two buckets)
    is merged into the first category it appears in."""
    seen: dict[str, SynthesizedSection] = {}
    new_sections: dict = {}
    for category, sections in report.ordered_sections():
        kept = []
        for section in polish_sections(sections):
            key = _label_key(section.subheading)
            if key in seen:
                first = seen[key]
                first.narrative += PARAGRAPH_SEPARATOR + section.narrative
                first.source_urls = list(first.source_urls) + [u for u in section.source_urls if u not in first.source_urls]
                continue
            seen[key] = section
            kept.append(section)
        if kept:
            new_sections[category] = kept
    report.sections_by_category = new_sections
    if report.summary:
        report.summary = turkish_numbers(report.summary)
    return report
