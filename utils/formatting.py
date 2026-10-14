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
_EN_DATE = re.compile(r"\b(\d{1,2}) (" + "|".join(_MONTHS_EN_TR) + r")[a-z]*\.?( \d{4})?\b")


def _abbreviate(value: int) -> str:
    for size, unit in ((1_000_000_000, "mlr"), (1_000_000, "mn")):
        if value >= size:
            number = f"{value / size:.2f}".rstrip("0").rstrip(".").replace(".", ",")
            return f"{number} {unit}"
    return f"{value:,}".replace(",", ".")


# "1.713,9 mn" / "1,713,9 mn": the model's own half-conversion of a 10-digit
# amount (2026-10-14 subject line).
_THOUSANDS_OF_MILLIONS = re.compile(r"(?<![\d.,])(\d{1,3})[.,](\d{3})(?:,(\d+))? ?(mn|milyon)\b")


# Things the model was told not to write and wrote anyway (2026-10-14):
# "ISIN TRSAKBKA2716" in a bond item, "(rakamlar haberde yer almıyor)".
_ISIN = re.compile(r"[;,]?\s*ISIN(?: kodu)?:?\s*[A-Z]{2}[A-Z0-9]{9}\d")
_META_REMARK = re.compile(
    r"\s*\([^()]*(?:yer almıyor|yer almamakta|detaylandırılmamış|belirtilmemiş|belirtilmiyor|haberde yok|başlık)[^()]*\)"
)
# Clauses pointing at the source instead of saying something: "; ceza KAP
# bildiriminde yer alıyor", "; tarih henüz belirtilmedi" (2026-10-14).
_META_CLAUSE = re.compile(
    r"[;,]\s*[^;,.]*?(?:KAP bildiriminde (?:yer alıyor|yer almaktadır|bulunuyor)|henüz belirtilmedi|belirtilmedi)(?=[.;]|$)"
)

# One label per kind of event, so that merge_same_label can put e.g. a bond
# sale and two issuance permits under one "Borçlanma" heading; the model
# called them "Tahvil İhracı" and "İzin" in the same report.
_LABEL_ALIASES = (
    (re.compile(r"tahvil|bono|ihraç|borçlanma|sukuk|eurobond|kira sertifika"), "Borçlanma"),
    (re.compile(r"ceza|soruşturma|dava|uzlaşma"), "Dava"),
    (re.compile(r"bilanço tarihi|finansal takvim"), "Bilanço tarihi"),
    (re.compile(r"analist|aracı kurum raporu|hedef fiyat"), "Analist raporu"),
)
# A generic label over a plainly borrowing narrative ("Yatırım: ... tahvil
# sattı", gpt-oss-20b on 2026-10-14).
_GENERIC_LABELS = {"yatırım", "finansman", "sermaye piyasası", "diğer", "genel"}
_BORROWING_TEXT = re.compile(r"tahvil|bono|borçlanma aracı|ihraç tavanı|sukuk|eurobond")
# "Finansal sonuç: ... bilanço tarihini açıkladı" is a date, not a result.
_RESULTS_DATE = re.compile(r"(bilanço|sonuç)\w* (açıklama )?tarih|açıklanacağı|açıklayacağı")
_BROKER_LABEL = re.compile(r"yatırım|menkul|securities|research")
_REPORT_WORDS = re.compile(r"rapor|bülten|tahmin|hedef fiyat|tavsiye|analiz")


def canonical_label(label: str, narrative: str) -> str:
    key = _label_key(label)
    if _BROKER_LABEL.search(key) and _REPORT_WORDS.search(narrative.lower()):
        return "Analist raporu"
    for pattern, canonical in _LABEL_ALIASES:
        if pattern.search(key):
            return canonical
    if key in _GENERIC_LABELS and _BORROWING_TEXT.search(narrative.lower()):
        return "Borçlanma"
    if key.startswith("finansal sonuç") and _RESULTS_DATE.search(narrative.lower()):
        return "Bilanço tarihi"
    if key.rstrip(".") in ("izin", "onay", "spk onayı", "spk izni") and re.search(r"ihraç|borçlanma", narrative.lower()):
        return "Borçlanma"
    return label


# The same remark as a whole sentence: "Brüt temettü tutarı hisse başına
# açıklanmış ancak miktar haberde yer almıyor." (GUBRF, 2026-10-14).
_META_SENTENCE = re.compile(
    r"\s*[^.()]*(?:haberde yer almıyor|haberde yer almamakta|haberde belirtilmemiş|detaylandırılmamış)[^.()]*\.(?=\s|$)"
)


def clean_prose(text: str) -> str:
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    text = _ISIN.sub("", text)
    text = _META_REMARK.sub("", text)
    text = _META_CLAUSE.sub("", text)
    text = re.sub(r"(\d) ?mrn\b", r"\1 mlr", text)  # "4 mrn USD"
    return _META_SENTENCE.sub("", text).strip()


# The fallback model (gpt-oss-20b, 2026-10-14) wrote "12 118,29 puan",
# "8 430 000 000 TL" and "22 %": space-grouped digits and a trailing
# percent sign.
_SPACE_GROUPED = re.compile(r"(?<![\d.,])(\d{1,3}(?: \d{3})+)(?![\d])(?= ?(?:,\d|TL|mn|mlr|puan|lira|dolar|USD|EUR|\$|€|adet|ton|MW|\)|\.|,|;|$))")
_PERCENT_AFTER = re.compile(r"(?<![%\d])(\d+(?:,\d+)?) ?%(?!\d)")


def turkish_numbers(text: str) -> str:
    """1.713.897.558 TL -> 1,71 mlr TL; 30,000,000,000 TRY -> 30 mlr TL;
    1.713,9 mn TL -> 1,71 mlr TL; 05 Oct 2026 -> 5 Ekim 2026."""
    def replace(match: re.Match) -> str:
        value = int(re.sub(r"[.,]", "", match.group(1)))
        return _abbreviate(value) if value >= 1_000_000 else match.group(1)

    def millions(match: re.Match) -> str:
        value = float(f"{match.group(1)}{match.group(2)}.{match.group(3) or 0}") * 1_000_000
        return _abbreviate(int(value))

    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    text = _SPACE_GROUPED.sub(lambda m: m.group(1).replace(" ", "."), text)
    text = _PERCENT_AFTER.sub(r"%\1", text)
    # kuruş on a million-plus amount ("9.132.064.896,755 TL") don't matter
    text = re.sub(r"(?<![\d.,])(\d{1,3}(?:\.\d{3}){2,}),\d+", r"\1", text)
    text = _GROUPED_NUMBER.sub(replace, text)
    text = _THOUSANDS_OF_MILLIONS.sub(millions, text)
    text = re.sub(r"\bTRY\b", "TL", text)
    text = _EN_DATE.sub(lambda m: f"{int(m.group(1))} {_MONTHS_EN_TR[m.group(2)]}{m.group(3) or ''}", text)
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
    for section in sections:
        section.subheading = canonical_label(section.subheading, section.narrative)
    result = merge_same_label(sections)
    for section in result:
        section.narrative = turkish_numbers(clean_prose(section.narrative))
        section.subheading = turkish_numbers(section.subheading)
    return result


_YEAR_RANGE = re.compile(r"\b(20\d\d)\s*[-‑–]\s*(20\d\d)\b")


def fix_year_ranges(text: str, source_text: str) -> str:
    """"2026‑2027 üçüncü çeyrek" where only 2026 is in the sources -> "2026
    üçüncü çeyrek"; a range both of whose years are sourced is kept."""
    known = set(re.findall(r"\b20\d\d\b", source_text))

    def replace(match: re.Match) -> str:
        first, second = match.group(1), match.group(2)
        if first == second:  # "2026‑2026 3. çeyrek"
            return first
        if first in known and second not in known:
            return first
        if second in known and first not in known:
            return second
        return match.group(0)

    return _YEAR_RANGE.sub(replace, text)


# A paragraph whose subject is plainly another label's: on 2026-10-14 the
# Rekabet Kurulu fine was written as the first paragraph of "Borçlanma".
_PARAGRAPH_KIND = (
    ("Dava", re.compile(r"rekabet kurulu|idari para cezası|soruşturma|dava")),
    ("Borçlanma", re.compile(r"tahvil|bono|borçlanma aracı|ihraç tavanı|ihraç belgesi|sukuk|eurobond")),
)


def _paragraph_kind(paragraph: str) -> str | None:
    kinds = [label for label, pattern in _PARAGRAPH_KIND if pattern.search(paragraph.lower())]
    return kinds[0] if len(kinds) == 1 else None


def refile_paragraphs(sections: list[SynthesizedSection]) -> list[SynthesizedSection]:
    """Moves paragraphs of a "Dava"/"Borçlanma" section that are clearly about
    the other one into a section of that label (created if missing)."""
    by_label = {s.subheading: s for s in sections}
    for section in list(sections):
        if section.subheading not in ("Dava", "Borçlanma"):
            continue
        paragraphs = section.narrative.split(PARAGRAPH_SEPARATOR)
        stay, move = [], []
        for paragraph in paragraphs:
            kind = _paragraph_kind(paragraph)
            (move if kind and kind != section.subheading else stay).append(paragraph)
        if not move or not stay:
            continue
        section.narrative = PARAGRAPH_SEPARATOR.join(stay)
        other = "Dava" if section.subheading == "Borçlanma" else "Borçlanma"
        if other in by_label:
            by_label[other].narrative += PARAGRAPH_SEPARATOR + PARAGRAPH_SEPARATOR.join(move)
        else:
            new = SynthesizedSection(subheading=other, narrative=PARAGRAPH_SEPARATOR.join(move), source_urls=[])
            sections.insert(sections.index(section), new)
            by_label[other] = new
    return sections


def polish_report(report: SynthesizedCompanyReport, source_text: str | None = None) -> SynthesizedCompanyReport:
    """Same label across categories (e.g. "Dava" filed under two buckets)
    is merged into the first category it appears in."""
    seen: dict[str, SynthesizedSection] = {}
    new_sections: dict = {}
    for category, sections in report.ordered_sections():
        kept = []
        for section in refile_paragraphs(polish_sections(sections)):
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
        report.summary = turkish_numbers(clean_prose(report.summary))
    if source_text is not None:
        for sections in new_sections.values():
            for section in sections:
                section.narrative = fix_year_ranges(section.narrative, source_text)
        if report.summary:
            report.summary = fix_year_ranges(report.summary, source_text)
    return report
