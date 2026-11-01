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
    "mlr": 1e9, "mlrd": 1e9, "mlyr": 1e9, "milyar": 1e9, "billion": 1e9, "bn": 1e9,
    "trilyon": 1e12,
}
_NUMBER = re.compile(
    r"(?<![\w.,])(%\s?|yüzde\s)?(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?![\d])"
    r"(?:\s?(%))?(?:\s?(bin|mn|mln|milyon|million|mlr|mlrd|mlyr|milyar|billion|bn|trilyon)\b)?",
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


class SourceNumbers:
    """The numbers in the sources, in two pools. `amounts`: figures carrying
    a unit (scaled: "1,71 mlr" -> 1.71e9) and long grouped numbers
    ("1.713.897.558"). `raw`: every number as written. An output figure
    with a unit is matched against `amounts` only: "1,71 mn TL" for a
    source "1,71 mlr TL" passed on 2026-10-29 because some other number in
    a bank's financial statement happened to be near 1,71 million."""

    def __init__(self, amounts: set[float], raw: set[float], with_units: set[tuple[float, float]] | None = None) -> None:
        self.amounts = amounts
        self.raw = raw
        # (number as written, unit scale) for figures written with a unit word
        self.with_units = with_units or set()


def _amounts(text: str, checkable_only: bool, with_mantissa: bool = False) -> list:
    """(matched text, value, has_unit) for the numbers in `text`. With
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
        has_unit = bool(unit) or (integer.count(".") >= 2)  # "1.713.897.558" is a full amount
        if with_mantissa:
            result.append((match.group(0).strip(), scaled, has_unit, value, _SCALES.get((unit or "").lower())))
        else:
            result.append((match.group(0).strip(), scaled, has_unit))
    return result


def source_numbers(source_text: str) -> SourceNumbers:
    amounts: set[float] = set()
    raw: set[float] = set()
    with_units: set[tuple[float, float]] = set()
    for _, value, has_unit, mantissa, scale in _amounts(source_text, checkable_only=False, with_mantissa=True):
        (amounts if has_unit else raw).add(value)
        if scale:
            with_units.add((mantissa, scale))
    for match in _NUMBER.finditer(_normalize(source_text)):  # raw values too, dates included
        raw.add(_value(match.group(2), match.group(3)))
    return SourceNumbers(amounts, raw, with_units)


def _known(value: float, numbers: set[float]) -> bool:
    return any(abs(value - n) <= TOLERANCE * max(abs(n), 1e-9) for n in numbers)


def _wrong_unit(mantissa: float, scale: float | None, numbers: SourceNumbers) -> bool:
    """"1,71 mn" where the sources only say "1,71 mlr": same figure, the
    unit changed (2026-10-29, Rekabet Kurulu fine)."""
    if not scale:
        return False
    same = [s for m, s in numbers.with_units if abs(m - mantissa) <= 0.005 * max(m, 1e-9)]
    return bool(same) and scale not in same


def unsourced(text: str, numbers: SourceNumbers) -> list[str]:
    result = []
    for shown, value, has_unit, mantissa, scale in _amounts(text, checkable_only=True, with_mantissa=True):
        pool = numbers.amounts if has_unit else numbers.raw | numbers.amounts
        # "11,253 m²" for a source "11.253 m²": a thousands separator written
        # as a comma, not a decimal (GLYHO, 2026-11-01).
        alternatives = [value, value * 1000] if re.fullmatch(r"\d{1,3},\d{3}", shown) else [value]
        if _wrong_unit(mantissa, scale, numbers) or not any(_known(v, pool) for v in alternatives):
            result.append(shown)
    return result


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


def drop_unsourced(text: str, numbers: SourceNumbers, label: str) -> str:
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
