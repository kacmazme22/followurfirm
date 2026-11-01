"""
Fetches a real article body (not just a headline) for a scraped item, via
trafilatura content extraction — used by google_news_scraper.py and
bigpara_scraper.py to enrich a bounded number of items per ticker
(MAX_ARTICLES_TO_ENRICH_PER_TICKER, config/constants.py) with genuine
multi-sentence content instead of a bare title. Diagnosed 2026-08-13: LLM
synthesis was producing generic filler sentences because Bigpara/Google
News never supplied a real body_snippet, only raw_title — KAP was the only
source with real summary text.

Google News RSS links are opaque redirect tokens
(news.google.com/rss/articles/...) that no longer resolve via a plain HTTP
redirect (verified live) — they require decoding via `googlenewsdecoder`,
which reverse-engineers Google's internal batchexecute endpoint. Same risk
class as KAP's undocumented API: could break silently if Google changes the
token format. Bigpara URLs are already real article URLs, no decode needed.

Never raises: any failure (decode, fetch, or empty extraction) returns
None, so callers can safely keep body_snippet=None and fall back to
title-only rendering, exactly like before this module existed.
"""

from __future__ import annotations

import asyncio
import logging
import re

import trafilatura
from bs4 import BeautifulSoup

from scrapers.base import BaseScraper
from utils.formatting import turkish_numbers

logger = logging.getLogger(__name__)

# Imported defensively: googlenewsdecoder is a small reverse-engineering
# package with fragile transitive deps (2026-10-04: selectolax 1.0 removed a
# backend it imports, and the module-level ImportError killed main.py before
# a single ticker ran). Without it we only lose Google News body enrichment —
# headlines still flow — so a broken install must never take the digest down.
try:
    from googlenewsdecoder import gnewsdecoder
except Exception as _import_exc:  # ImportError, or anything its own imports raise
    gnewsdecoder = None
    logger.warning("googlenewsdecoder yüklenemedi, Google News gövde zenginleştirmesi kapalı: %s", _import_exc)

# NewsItem.body_snippet (scrapers/models.py) caps at max_length=2000 —
# trimmed well under that so truncation never collides with the field's own
# validation limit.
MAX_BODY_SNIPPET_CHARS = 1500

# What's left after cleaning must be at least this long to count as an
# article; shorter means trafilatura only found site chrome.
MIN_BODY_CHARS = 120

# Site chrome trafilatura picked up instead of the article (2026-10-03 mail:
# every KCHOL Bigpara item's "summary" was the Hürriyet footer). A line
# containing any of these is dropped.
_CHROME_MARKERS = (
    "en çok arananlar",
    "copyright",
    "kullanım koşulları",
    "gizlilik politika",
    "login olduğunuz",
    "sıralamayı değiştirmek",
    "bist 100 dolar euro",
    "bildirimler bildirimler",
)

# Everything from here on is the broker/portal legal disclaimer
# ("Burada yer alan yatırım bilgi, yorum ve tavsiyeleri...").
_DISCLAIMER_START = re.compile(r"burada yer alan yatırım bilgi|yatırım danışmanlığı kapsamında değildir", re.IGNORECASE)


# Bot-protection interstitials (Cloudflare and the like) come back as HTTP
# 200; on 2026-10-14 one reached the model as a GUBRF dividend "article".
_CHALLENGE_MARKERS = (
    "checking your browser", "enable javascript", "supports javascript", "just a moment",
    "verify you are human", "attention required",
)


def clean_article_text(text: str) -> str | None:
    """Drops site-chrome lines and the trailing legal disclaimer; None if
    nothing article-like is left."""
    if any(marker in text[:600].lower() for marker in _CHALLENGE_MARKERS):
        return None
    match = _DISCLAIMER_START.search(text)
    if match:
        text = text[: match.start()]
    lines = [
        line.strip() for line in text.splitlines()
        if line.strip() and not any(marker in line.lower() for marker in _CHROME_MARKERS)
    ]
    cleaned = "\n".join(lines).replace("*", "").strip()
    return cleaned if len(cleaned) >= MIN_BODY_CHARS else None


# KAP form rows that say nothing about the event itself.
_FORM_NOISE_LABELS = (
    "güncelleme mi", "düzeltme mi", "ertelenmiş", "daha önce yapılan açıklamanın tarihi",
    "ilgili şirketler", "ilgili fonlar", "hak sahiplerinin belirlendiği", "derecelendirme notunu veren",
)
_XBRL_CELL = re.compile(r"^[a-z]+_[A-Za-z0-9]+\|")
_LANGUAGE_HEADERS = {"türkçe", "ingilizce", "english", "turkish"}
_EMPTY_VALUES = {"", "-", "[]", "yok", "hayır (no)", "hayır"}


def _turkish_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.select('[class*="content-en"]'):
        tag.decompose()
    return soup


_TEXT_BLOCK_ID = re.compile(r"^[a-z]+_\w*TextBlock\|")


def extract_text_blocks(html: str) -> list[str]:
    """The free-text parts of a KAP form ("oda_ExplanationTextBlock|" and the
    like), read element by element up to the next XBRL id. Özel Durum
    filings are almost nothing but this block — on 2026-10-14 the Akbank
    Rekabet fine filing had no other form field, fell through to plain page
    extraction and reached the model as KAP's search-box help text followed
    by "oda_..." ids and every label in two languages."""
    lines = [" ".join(t.split()) for t in _turkish_soup(html).get_text("\n").splitlines()]
    lines = [t for t in lines if t]
    blocks: list[str] = []
    current: list[str] | None = None
    for line in lines:
        if _XBRL_CELL.match(line):
            if current:
                blocks.append(" ".join(current))
            current = [] if _TEXT_BLOCK_ID.match(line) else None
            rest = line.split("|", 1)[1].strip()
            if current is not None and rest:
                current.append(rest)
            continue
        if current is None:
            continue
        if not current and len(line) <= 40 and not re.search(r"[\d.]", line):
            continue  # the block's own label ("Açıklamalar")
        current.append(line)
    if current:
        blocks.append(" ".join(current))
    unique: list[str] = []
    for block in blocks:
        if len(block) >= 20 and block not in unique:
            unique.append(block)
    return unique


def extract_form_lines(html: str) -> list[str]:
    """KAP disclosures (on kap.org.tr and on Bigpara's kap-haberleri copies)
    are label/value forms. Flattened to plain text they read "Satışı
    Gerçekleştirilen Nominal Tutar 50.000.000 ... Döviz Cinsi EUR", and the
    model attached values to the wrong labels: one Akbank bond issue came out
    as "50 mn USD", "5 mn" and "4 mlr USD" in three runs (2026-10-06); it was
    50 mn EUR under a 4 bn USD programme. Each table row becomes one
    "Etiket: Değer" line instead; English duplicates (KAP's content-en
    cells) and procedural form fields are dropped."""
    soup = _turkish_soup(html)
    lines: list[str] = []
    for row in soup.find_all("tr"):
        cells = [" ".join(c.get_text(" ").split()) for c in row.find_all(["td", "th"], recursive=False)]
        # kap.org.tr prefixes each row with its XBRL element name
        # ("oda_UpdateAnnouncementFlag|"); the human label is the next cell.
        cells = [c for c in cells if c and not _XBRL_CELL.match(c)]
        if len(cells) == 1 and cells[0].strip("[]").lower() in _LANGUAGE_HEADERS:
            continue
        if len(cells) == 1 and len(cells[0]) <= 80:
            # Section header row ("İlgili İhraç Tavanı Bilgileri"): keeps the
            # programme ceiling apart from the amount actually sold.
            header = f"[{cells[0]}]"
            if lines and lines[-1].startswith("["):
                lines[-1] = header  # header with no fields under it
            elif header not in lines:
                lines.append(header)
            continue
        if len(cells) < 2:
            continue
        label, values = cells[0], [v for v in cells[1:] if v.lower() not in _EMPTY_VALUES]
        if not values or any(noise in label.lower() for noise in _FORM_NOISE_LABELS):
            continue
        line = f"{label}: {' | '.join(values)}"
        if line not in lines:
            lines.append(line)
    while lines and lines[-1].startswith("["):  # trailing headers with no fields
        lines.pop()
    return lines


def _longest_paragraphs(text: str, limit: int = 2) -> list[str]:
    """The free-text explanation ("Açıklamalar") of a form page: its longest
    sentence-like lines, since the rest of the extracted text is the
    flattened form again."""
    paragraphs = [p.strip() for p in text.splitlines() if len(p.strip()) >= 120 and "." in p]
    return sorted(paragraphs, key=len, reverse=True)[:limit]


def structured_text(html: str, url: str) -> str | None:
    """Form rows (if the page is a form) plus its explanation paragraphs;
    otherwise the plain article text."""
    try:
        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=False) or ""
    except Exception as exc:
        logger.info("trafilatura extraction failed for %s: %s", url, exc)
        extracted = ""
    form = extract_form_lines(html)
    blocks = extract_text_blocks(html)
    if blocks or sum(1 for line in form if not line.startswith("[")) >= 3:
        logger.info("Form yapısı: %s -> %d satır, %d açıklama", url, len(form), len(blocks))
        explanations = blocks or _longest_paragraphs(extracted)
        if all(line.startswith("[") for line in form):
            form = []  # section headers with no fields under them
        # Amounts are normalised here, before the model sees them: given
        # "Tutar: 30.000.000.000" it wrote "30 mn TL" (2026-10-14).
        parts = [turkish_numbers(line) for line in form] + [f"Açıklama: {turkish_numbers(p)}" for p in explanations]
        return "\n".join(parts)
    return clean_article_text(extracted) if extracted else None


def _decoded_url(result: dict) -> str | None:
    """googlenewsdecoder 0.2 answers {"success": True, "decoded_url": ...};
    0.1 answered {"status": True, ...}. The check only knew "status", so
    from the upgrade on every successfully decoded link was thrown away
    and logged as "returned no URL ... None" (None: a success carries no
    message) — every Google News item stayed a bare headline from
    2026-10-05 to 2026-11-01."""
    if not isinstance(result, dict):
        return None
    if not (result.get("success") or result.get("status")):
        return None
    url = result.get("decoded_url")
    return url if isinstance(url, str) and url.startswith("http") else None


async def decode_google_urls(urls: list[str]) -> list[str | None]:
    """Publisher URLs for Google News links (None where it fails), decoded
    in one batch: one page fetch per link, then a single POST to Google."""
    if gnewsdecoder is None or not urls:
        return [None] * len(urls)
    try:
        results = await asyncio.to_thread(gnewsdecoder, list(urls), interval=1)
    except Exception as exc:
        logger.info("Google News URL decode failed: %s: %s", type(exc).__name__, exc)
        return [None] * len(urls)
    decoded = [_decoded_url(r) for r in results]
    for url, result, ok in zip(urls, results, decoded):
        if not ok:
            logger.info("Google News URL çözülemedi: %s: %s", url[:90], (result or {}).get("message"))
    return decoded


# Publisher sites answer a browser better than a bot: 403s (nefes.com.tr,
# memurpostasi.com) and 429s (haber7.com, 11 articles on 2026-11-01) once
# Google links started resolving.
ARTICLE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.6",
}
RATE_LIMIT_WAIT_SECONDS = 8


async def _get_article(get, url: str):
    """GET with browser headers; on 429 wait (Retry-After, at most 20 s) and
    try once more."""
    response = await get(url, headers=dict(ARTICLE_HEADERS))
    if response.status_code == 429:
        try:
            wait = min(20, int(response.headers.get("retry-after", RATE_LIMIT_WAIT_SECONDS)))
        except ValueError:
            wait = RATE_LIMIT_WAIT_SECONDS
        logger.info("429 (%s), %d sn bekleyip tekrar deneniyor", url[:80], wait)
        await asyncio.sleep(wait)
        response = await get(url, headers=dict(ARTICLE_HEADERS))
    return response


async def fetch_text(client, url: str) -> str | None:
    """Article text for `url` with a plain httpx client (no scraper)."""
    try:
        response = await _get_article(client.get, url)
        response.raise_for_status()
    except Exception as exc:
        logger.info("Article fetch failed for %s: %s", url, exc)
        return None
    text = structured_text(response.text, str(response.url))
    return text[:MAX_BODY_SNIPPET_CHARS] if text else None


async def fetch_article_body(scraper: BaseScraper, url: str, is_google_news: bool) -> str | None:
    """Fetches `url` (decoding it first if it's a Google News redirect
    token) and extracts the real article body via trafilatura. Uses
    `scraper._get()` so the fetch gets the same politeness delay, User-Agent,
    and in-run caching as every other request that scraper makes. Returns
    None on any failure at any stage — decode, fetch, or empty extraction —
    never raises."""
    target_url = url

    if is_google_news:
        if gnewsdecoder is None:
            return None
        target_url = (await decode_google_urls([url]))[0]
        if not target_url:
            return None

    try:
        response = await _get_article(scraper._get, target_url)
    except Exception as exc:
        logger.info("Article fetch failed for %s: %s", target_url, exc)
        return None

    text = structured_text(response.text, target_url)
    if not text:
        logger.info("No article text from %s (HTTP %s)", target_url, response.status_code)
        return None
    return text[:MAX_BODY_SNIPPET_CHARS]
