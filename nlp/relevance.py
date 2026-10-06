"""
Rule-based relevance filter, run on RawScrapedItems before categorization.

Why it exists (diagnosed from the 2026-10-03 production run): Bigpara's
per-ticker news page is not actually per-ticker. Alongside real company news
it lists market-wide items that merely *mention* the ticker somewhere in the
body — broker morning bulletins ("Analiz: Günlük Bülten - Ziraat Yatırım")
and multi-stock KAP relays ("ESCOM, AKYHO, HATSN, KOCMT, NATEN, ..."). With
AI_PROVIDER=noop those reached the email verbatim, with a trafilatura dump
of a whole-market data table as their "summary".

Three rules, applied to non-KAP sources only (KAP items are already matched
to the ticker by KAP's own stockCodes field, see kap_scraper.py):

  1. Multi-stock lists: a title containing a run of MULTI_TICKER_RUN or more
     ticker-shaped codes in a row is a market-wide list, never company news.
  2. Market noise: technical analysis (support/resistance, moving averages,
     RSI...), daily price-move reports ("hisseleri yüzde 3 yükseldi") and
     trading-flow data (volume, short selling, broker distribution). The
     digest is for company fundamentals and events — the reader explicitly
     doesn't want price levels, and a price chart already shows them.
  3. Mentions: the title must name the ticker or the company (see
     _company_keywords) — the company being somewhere in a bulletin's body
     isn't enough.

Pure string rules, zero network/LLM cost, and deliberately conservative:
dropping a real headline that names neither the company nor its ticker is
an accepted trade-off against mailing a market-data dump every morning.
"""

from __future__ import annotations

import logging
import re

from config.constants import SourceType
from config.settings import TickerConfig
from scrapers.models import RawScrapedItem

logger = logging.getLogger(__name__)

MULTI_TICKER_RUN = 4

# 4-6 uppercase letters/digits as a standalone token: BIST ticker shape.
_TICKER_TOKEN = r"[A-ZÇĞİÖŞÜ0-9]{4,6}"
_MULTI_TICKER_RE = re.compile(
    rf"(?<![\wÇĞİÖŞÜçğıöşü]){_TICKER_TOKEN}(?:[\s,/;|-]+{_TICKER_TOKEN}){{{MULTI_TICKER_RUN - 1},}}(?![\wÇĞİÖŞÜçğıöşü])"
)

# Matched against the tr_lower()ed title. Phrases, not single words, so a
# real event headline isn't caught by accident ("destek" alone would also
# match "devlet desteği aldı", hence "destek seviye").
_MARKET_NOISE_PATTERNS = [
    re.compile(p) for p in (
        r"teknik (analiz|görünüm|yorum|değerlendirme|seviye)",
        r"destek (seviye|bölge|nokta)",
        r"direnç",
        r"hareketli ortalama",
        # tr_lower() turns the acronym "RSI" into "rsı", hence rs[iı].
        r"\b(rs[iı]|macd|fibonacci|bollinger)\b",
        r"aşırı (alım|satım)",
        r"formasyon",
        r"al[- ]sat seviye",
        r"hedef seviye",
        r"stop[- ]?loss",
        r"açığa satış",
        r"işlem hacmi",
        r"\btakas\b",
        r"aracı kurum dağılımı",
        r"en çok (alan|satan|işlem gören|yükselen|düşen|değer kaybeden|değer kazanan)",
        r"günün (en|kazanan|kaybeden)",
        r"(günlük|sabah|piyasa) (bülten|not|yorum)",
        # Order-flow, not business news: "KCHOL'de iki kurumdan yüklü satış",
        # broker model-portfolio reweightings (2026-10-05 dry run).
        r"yüklü (alım|satış)",
        r"kurum(dan|lardan) (gelen )?(alım|satış)",
        r"(model|döngüsel) portföy",
        r"portföy ağırlı",
        # Quote/forum pages Google News indexes as "news": "YEO TEKNOLOJI
        # ENERJI (YEOTK) Hisse Senedi", "YEOTK Hisse Yorumları".
        r"hisse senedi$",
        r"hisse (yorumları|fiyatı|grafiği|detay)",
        r"güncel yorumlar",
        r"canlı grafik",
        # Price-limit trading halts: a price event, not company news.
        r"devre kesici",
        # Bigpara relays of market-infrastructure notices that list the
        # ticker among others (MKK share-type conversions, BIST index lists).
        r"merkezi kayıt kuruluşu",
        r"borsa istanbul a\.ş",
        r"pay endeksleri",
        # Price-move reports: "Akbank hisseleri yüzde 3 yükseldi", "KCHOL
        # payları %2 değer kaybetti". Requires the hisse/pay subject so an
        # earnings headline ("net kârı yüzde 30 arttı") isn't dropped.
        r"(hisse|pay)(ler|lar)?(i|ı|si|sı)?\b.{0,40}(yüzde|%)\s?[\d,.]+.{0,25}"
        r"(yüksel|düş|geril|arttı|artış göster|değer kazan|değer kaybet|prim yap|sert)",
        r"(hisse|pay)(ler|lar)?(i|ı|si|sı)?\b.{0,30}(rekor kır|zirve|taban|tavan)",
    )
]

# Words too generic to identify a company on their own.
_GENERIC_NAME_WORDS = {
    "a.ş.", "aş", "a.s.", "as", "t.a.ş.", "holding", "sanayi", "ve", "ticaret",
    "fabrikaları", "teknoloji", "bankası", "yatırım", "enerji", "grup", "group",
}


def clean_title(title: str) -> str:
    """Bigpara wraps tickers in asterisks and relayed headlines in a ticker
    prefix: "***KCHOL*** (*Koç Holding YKB Vekili Ali Koç: ...)" and
    "***ESCOM* *AKYHO* *HATSN*...". Strip the asterisks (they also hid
    multi-stock lists from _MULTI_TICKER_RE) and unwrap the parenthesised
    headline so the reader sees "Koç Holding YKB Vekili Ali Koç: ..."."""
    text = re.sub(r"\s+", " ", title.replace("*", " ")).strip()
    wrapped = re.match(rf"^{_TICKER_TOKEN}\s*\((.+?)\)?$", text)
    if wrapped and len(wrapped.group(1)) > 15:
        text = wrapped.group(1).strip()
    return text


def tr_lower(text: str) -> str:
    """Turkish-aware lowercase: plain str.lower() turns 'İ' into 'i̇' (i plus
    a combining dot) and 'I' into 'i' instead of 'ı', so 'KOÇ HOLDİNG' and
    'Koç Holding' wouldn't compare equal."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def _company_keywords(ticker: TickerConfig) -> set[str]:
    """Words that identify the company in a headline: the ticker symbol, the
    full company name, and the first distinctive word of the name
    ("Koç Holding" -> "koç", "Gübre Fabrikaları" -> "gübre")."""
    keywords = {tr_lower(ticker.symbol), tr_lower(ticker.name)}
    for word in tr_lower(ticker.name).split():
        if word not in _GENERIC_NAME_WORDS and len(word) >= 3:
            keywords.add(word)
            break
    return keywords


def _mentions_company(title: str, keywords: set[str]) -> bool:
    lowered = tr_lower(title)
    return any(
        re.search(rf"(?<![\wçğıöşü]){re.escape(k)}", lowered) for k in keywords
    )


def is_multi_ticker_list(title: str) -> bool:
    return _MULTI_TICKER_RE.search(title) is not None


def is_market_noise(title: str, url: str | None = None) -> bool:
    # Bigpara files broker morning notes under ".../araci-kurum-raporlari/
    # analiz-gunluk-bulten-...": market-wide by construction.
    if url and "/analiz-" in url:
        return True
    lowered = tr_lower(title)
    return any(p.search(lowered) for p in _MARKET_NOISE_PATTERNS)


def filter_relevant(items: list[RawScrapedItem], ticker: TickerConfig) -> list[RawScrapedItem]:
    keywords = _company_keywords(ticker)
    kept: list[RawScrapedItem] = []
    dropped_multi = dropped_noise = dropped_unrelated = 0

    for item in items:
        if item.source != SourceType.KAP:
            item.raw_title = clean_title(item.raw_title)

        # Google News relays of KAP filings ("KAP GÜBRE FABRİKALARI T.A.Ş. GUBRF
        # Genel Kurul İşlemlerine İlişkin Bildirim") are a bare filing type
        # with no body; the filing itself comes from KAP or Bigpara's mirror.
        if item.source == SourceType.GOOGLE_NEWS and tr_lower(item.raw_title).startswith("kap "):
            dropped_unrelated += 1
            continue

        if item.source == SourceType.KAP:
            kept.append(item)
        elif is_multi_ticker_list(item.raw_title):
            dropped_multi += 1
        elif is_market_noise(item.raw_title, item.raw_url):
            dropped_noise += 1
        elif not _mentions_company(item.raw_title, keywords):
            dropped_unrelated += 1
        else:
            kept.append(item)

    if dropped_multi or dropped_noise or dropped_unrelated:
        logger.info(
            "%s: alaka filtresi %d çok-hisseli liste, %d teknik/fiyat haberi, %d şirketi anmayan haber eledi",
            ticker.symbol, dropped_multi, dropped_noise, dropped_unrelated,
        )
    return kept
