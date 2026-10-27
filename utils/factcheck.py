"""
Every amount and rate in the mail must be traceable to the sources.

On 2026-10-24 the subject and the SAHOL section said Sabancı "sold
CarrefourSA for 325 mn USD"; no input contained 325 (Sabancı in fact paid
the buyer about 92 mn USD). The prompt already says "never invent a
figure"; this is the check that doesn't depend on the model obeying it.

A sentence whose amount/rate can't be found among the source numbers is
removed (a section left empty is dropped) and logged. Matching allows for
rounding and unit changes: "1,71 mlr TL" matches "1.713.897.558 TL",
"92 mn dolar" matches "91,95 mn USD", "7,25 milyar" matches "7 milyar 252
milyon 152 bin 706 TL".
"""

from __future__ import annotations

import logging
import re

from scrapers.models import SynthesizedSection

logger = logging.getLogger("audit")

TOLERANCE = 0.015  # relative: rounding to 2-3 significant digits

_SCALES = {
    "bin": 1e3,
    "mn": 1e6, "mln": 1e6, "milyon": 1e6, "million": 1e6,
    "mlr": 1e9, "mlrd": 1e9, "mlyr": 1e9, "milyar": 1e9, "billion": 1e9,
    "trilyon": 1e12,
}
_NUMBER = re.compile(
    r"(?<![\w.,])(%\s?)?(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?![\d])"
    r"(?:\s?(%))?(?:\s?(bin|mn|mln|milyon|million|mlr|mlrd|mlyr|milyar|billion|trilyon)\b)?",
    re.IGNORECASE,
)
_MONTHS = "ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık"
# Things that look like numbers but are dates, periods, times or ordinals.
_NOT_AMOUNTS = re.compile(
    rf"\b\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}\b"          # 06.12.2027
    rf"|\b\d{{1,2}}\s+(?:{_MONTHS})\w*(?:\s+\d{{4}})?"    # 16 Ekim (2026)
    rf"|\b(?:{_MONTHS})\w*\s+\d{{4}}"                      # Ekim 2026
    rf"|\b\d{{1,2}}[.:]\d{{2}}\b(?=\s*(?:'|’)?(?:de|da|te|ta)?\b)"  # 14.00
    rf"|\b\d[ÇQ]\d{{2}}\b|\b\d\.\s*çeyrek|\b\d+\.\s"       # 3Ç26, 3. çeyrek, ordinals
    rf"|\b(?:19|20)\d{{2}}\b",                             # years (checked elsewhere)
    re.IGNORECASE,
)
_COMPOUND = re.compile(
    r"(\d+(?:,\d+)?)\s+(milyar|milyon|bin)(?:\s+(\d+(?:,\d+)?)\s+(milyon|bin))?(?:\s+(\d+(?:,\d+)?)\s+(bin))?(?:\s+(\d+))?",
    re.IGNORECASE,
)


def _value(integer: str, fraction: str | None) -> float:
    return float(f"{integer.replace('.', '')}.{fraction or 0}")


# Reuters writes "14.1 MLYR TL": a decimal point, not Turkish grouping.
_DOT_DECIMAL = re.compile(r"(?<![\d.,])(\d{1,3})\.(\d{1,2})(?![\d.])")


def _normalize(text: str) -> str:
    """"MİLYON" lowercases to "mi̇lyon" (i + combining dot) in Python."""
    return _DOT_DECIMAL.sub(r"\1,\2", text.replace("İ", "i"))


def _collapse_compounds(text: str) -> str:
    """"5 milyar 102 milyon 800 bin TL" -> "5102,8 mn TL", one figure."""
    def total(match: re.Match) -> str:
        groups = match.groups()
        value = 0.0
        for amount, unit in zip(groups[0:6:2], groups[1:6:2]):
            if amount and unit:
                value += float(amount.replace(",", ".")) * _SCALES[unit.lower()]
        if groups[6]:
            value += float(groups[6])
        return f"{value / 1e6:.6f}".rstrip("0").rstrip(".").replace(".", ",") + " mn"

    return _COMPOUND.sub(lambda m: total(m) if m.group(3) else m.group(0), text)


def _amounts(text: str, checkable_only: bool) -> list[tuple[str, float]]:
    """(matched text, value) for the numbers in `text`. With
    checkable_only, only figures worth checking: with a unit or %, with
    decimals, or of 1000 and more."""
    cleaned = _NOT_AMOUNTS.sub(" ", _collapse_compounds(_normalize(text)))
    result = []
    for match in _NUMBER.finditer(cleaned):
        pct_before, integer, fraction, pct_after, unit = match.groups()
        value = _value(integer, fraction)
        scaled = value * _SCALES.get((unit or "").lower(), 1)
        if checkable_only and not (unit or pct_before or pct_after or fraction or value >= 1000):
            continue
        result.append((match.group(0).strip(), scaled))
    return result


def source_numbers(source_text: str) -> set[float]:
    numbers: set[float] = set()
    for _, value in _amounts(source_text, checkable_only=False):
        numbers.add(value)
    for match in _NUMBER.finditer(_normalize(source_text)):  # raw values too, dates included
        numbers.add(_value(match.group(2), match.group(3)))
    return numbers


def _known(value: float, numbers: set[float]) -> bool:
    return any(abs(value - n) <= TOLERANCE * max(abs(n), 1e-9) for n in numbers)


def unsourced(text: str, numbers: set[float]) -> list[str]:
    return [shown for shown, value in _amounts(text, checkable_only=True) if not _known(value, numbers)]


_BREAK = re.compile(r"\.\s+(?=[A-ZÇĞİÖŞÜ0-9%])")
_SENTENCE_TAIL = re.compile(r"(?:[a-zçğıöşü0-9%)”\"']|[A-ZÇĞİÖŞÜ]{2})$")


def _sentences(paragraph: str) -> list[str]:
    """Split at ". " before a capital, but not after an abbreviation like
    "A.Ş." (single capitals); "USD. %57" and "TL. Şirket" do split."""
    parts, start = [], 0
    for match in _BREAK.finditer(paragraph):
        if _SENTENCE_TAIL.search(paragraph[start:match.start()]):
            parts.append(paragraph[start:match.start()])
            start = match.end()
    parts.append(paragraph[start:])
    return parts


def drop_unsourced(text: str, numbers: set[float], label: str) -> str:
    """`text` without the sentences carrying an unsourced figure."""
    paragraphs = []
    for paragraph in text.split("\n\n"):
        kept = []
        for position, sentence in enumerate(_sentences(paragraph)):
            bad = unsourced(sentence, numbers)
            if bad and position == 0:
                # The rest of the paragraph leans on its first sentence: on
                # 2026-10-27 dropping it left "Kredilerin maliyetleri ..."
                # with no loan in sight.
                logger.warning("[%s] kaynakta olmayan rakam (%s), paragraf çıkarıldı: %s", label, ", ".join(bad), paragraph)
                kept = []
                break
            if bad:
                logger.warning("[%s] kaynakta olmayan rakam (%s), cümle çıkarıldı: %s", label, ", ".join(bad), sentence)
            else:
                kept.append(sentence)
        if kept:
            joined = ". ".join(s.rstrip(".") for s in kept)
            paragraphs.append(joined if joined.endswith((".", "!", "?")) else joined + ".")
    return "\n\n".join(paragraphs)


def check_sections(sections: list[SynthesizedSection], source_text: str, label: str) -> list[SynthesizedSection]:
    numbers = source_numbers(source_text)
    kept = []
    for section in sections:
        narrative = drop_unsourced(section.narrative, numbers, f"{label} {section.subheading}")
        if narrative.strip(" ."):
            section.narrative = narrative
            kept.append(section)
        else:
            logger.warning("[%s] «%s» bölümü kaynakta olmayan rakamlar yüzünden çıkarıldı", label, section.subheading)
    return kept


def check_line(text: str | None, source_text: str, label: str) -> str | None:
    """A one-line text (Kısaca, subject) is kept whole or dropped."""
    if not text:
        return text
    bad = unsourced(text, source_numbers(source_text))
    if bad:
        logger.warning("[%s] kaynakta olmayan rakam (%s), satır çıkarıldı: %s", label, ", ".join(bad), text)
        return None
    return text
