"""
Static, rarely-changing constants: category enums and the keyword/regex maps
that power Tier-1 rule-based categorization (nlp/categorizer.py).

Keeping these here — rather than inline in the categorizer — means updating
KAP terminology or adding a new news-source heuristic never requires touching
categorization logic itself.
"""

from __future__ import annotations

from enum import Enum


class NewsCategory(str, Enum):
    """The four fixed subheadings every ticker's digest section is grouped into."""

    NEW_BUSINESS = "yeni_is_iliskileri"        # Yeni İş İlişkileri & İhaleler
    FINANCIALS = "finansal_sonuclar"           # Finansal Sonuçlar & Bilanço
    KAP_MATERIAL = "onemli_kap_aciklamalari"   # Önemli KAP Açıklamaları
    GENERAL_SECTOR = "sektorel_genel_haberler" # Sektörel ve Genel Haberler

    @property
    def display_name_tr(self) -> str:
        return {
            NewsCategory.NEW_BUSINESS: "Yeni İş İlişkileri & İhaleler",
            NewsCategory.FINANCIALS: "Finansal Sonuçlar & Bilanço",
            NewsCategory.KAP_MATERIAL: "Önemli KAP Açıklamaları",
            NewsCategory.GENERAL_SECTOR: "Sektörel ve Genel Haberler",
        }[self]


class SourceType(str, Enum):
    KAP = "kap"
    BIGPARA = "bigpara"
    GOOGLE_NEWS = "google_news"
    COMPANY_IR = "company_ir"
    WEB_SEARCH_FALLBACK = "web_search_fallback"
    # Phase 2
    SEC_EDGAR = "sec_edgar"
    YAHOO_FINANCE = "yahoo_finance"
    FINVIZ = "finviz"
    PR_NEWSWIRE = "pr_newswire"


# ---------------------------------------------------------------------------
# KAP disclosure-type -> category keyword map.
#
# KAP disclosures carry a Turkish "bildirim türü" (disclosure type) field in
# the raw scrape. This dict maps *substrings* (lowercased, checked with `in`)
# of that field, plus free-text headline keywords, to a NewsCategory. Order
# matters: more specific keywords should be checked before generic ones —
# the categorizer iterates this in insertion order and takes the first hit.
# ---------------------------------------------------------------------------
KAP_KEYWORD_CATEGORY_MAP: dict[str, NewsCategory] = {
    # Checked first: buyback headlines often also contain "satın al" ("geri
    # alım kapsamında pay satın alındı"), which would otherwise land them in
    # NEW_BUSINESS below.
    "geri alım": NewsCategory.KAP_MATERIAL,
    "geri alınan pay": NewsCategory.KAP_MATERIAL,

    # New business / tenders / contracts
    "ihale": NewsCategory.NEW_BUSINESS,
    "sipariş": NewsCategory.NEW_BUSINESS,
    "satın al": NewsCategory.NEW_BUSINESS,
    "devral": NewsCategory.NEW_BUSINESS,
    "ihracat": NewsCategory.NEW_BUSINESS,
    "yatırım kararı": NewsCategory.NEW_BUSINESS,
    "yeni tesis": NewsCategory.NEW_BUSINESS,
    "fabrika": NewsCategory.NEW_BUSINESS,
    "lisans": NewsCategory.NEW_BUSINESS,
    "sözleşme": NewsCategory.NEW_BUSINESS,
    "satış anlaşması": NewsCategory.NEW_BUSINESS,
    "iş birliği": NewsCategory.NEW_BUSINESS,
    "işbirliği": NewsCategory.NEW_BUSINESS,
    "yatırım teşvik": NewsCategory.NEW_BUSINESS,
    "kapasite artış": NewsCategory.NEW_BUSINESS,
    "yeni pazar": NewsCategory.NEW_BUSINESS,
    "distribütör": NewsCategory.NEW_BUSINESS,
    "tedarik": NewsCategory.NEW_BUSINESS,

    # Financial statements / earnings
    "finansal rapor": NewsCategory.FINANCIALS,
    "finansal tablo": NewsCategory.FINANCIALS,
    "bilanço": NewsCategory.FINANCIALS,
    "bağımsız denetim": NewsCategory.FINANCIALS,
    "kar payı": NewsCategory.FINANCIALS,
    "kâr payı": NewsCategory.FINANCIALS,
    "temettü": NewsCategory.FINANCIALS,
    "faaliyet raporu": NewsCategory.FINANCIALS,
    "net kar": NewsCategory.FINANCIALS,
    "net kâr": NewsCategory.FINANCIALS,
    "ebitda": NewsCategory.FINANCIALS,
    "ciro": NewsCategory.FINANCIALS,
    "çeyrek": NewsCategory.FINANCIALS,
    "bilanço beklenti": NewsCategory.FINANCIALS,

    # Material KAP disclosures / corporate actions
    "özel durum açıklaması": NewsCategory.KAP_MATERIAL,
    "yönetim kurulu karar": NewsCategory.KAP_MATERIAL,
    "genel kurul": NewsCategory.KAP_MATERIAL,
    "sermaye artır": NewsCategory.KAP_MATERIAL,
    "bedelsiz": NewsCategory.KAP_MATERIAL,
    "geri alım": NewsCategory.KAP_MATERIAL,
    "pay alım": NewsCategory.KAP_MATERIAL,
    "birleşme": NewsCategory.KAP_MATERIAL,
    "bölünme": NewsCategory.KAP_MATERIAL,
    "halka arz": NewsCategory.KAP_MATERIAL,
    "ortaklık yapısı": NewsCategory.KAP_MATERIAL,
    "yönetim değişikliği": NewsCategory.KAP_MATERIAL,
    "istifa": NewsCategory.KAP_MATERIAL,
    "atama": NewsCategory.KAP_MATERIAL,
    "genel müdür": NewsCategory.KAP_MATERIAL,
    "borçlanma aracı": NewsCategory.KAP_MATERIAL,
    "tahvil": NewsCategory.KAP_MATERIAL,
    "sendikasyon": NewsCategory.KAP_MATERIAL,
    "kredi derecelendirme": NewsCategory.KAP_MATERIAL,
    "pay satış": NewsCategory.KAP_MATERIAL,
    "dava": NewsCategory.KAP_MATERIAL,

    # Fallback bucket — general / sector news picks up anything unmatched above
    # (this key is intentionally never matched directly; categorizer.py treats
    # "no keyword hit" as GENERAL_SECTOR by default).
}

# Words that, if present in a headline, strongly suggest the item is
# *duplicated boilerplate* (e.g. KAP auto-generated disclaimer footers) and
# should be down-weighted or stripped before dedup/summarization.
BOILERPLATE_NOISE_PATTERNS: list[str] = [
    "yukarıdaki açıklamalarımızın",
    "sermaye piyasası kurulu",
    "özel durum açıklamalarımız",
]

DEFAULT_CATEGORY = NewsCategory.GENERAL_SECTOR

# ---------------------------------------------------------------------------
# Time-window filtering for sources that have no server-side date filter of
# their own (Google News RSS, Bigpara) — unlike KAP, which filters by date
# range in the request itself (scrapers/kap_scraper.py's LOOKBACK_DAYS).
# 36h = today fully covered, plus a 12h safety margin against the midnight
# boundary (same reasoning KAP's LOOKBACK_DAYS=2 uses, just in hours since
# these two sources expose finer-grained timestamps than KAP's date-only
# window).
# ---------------------------------------------------------------------------
NEWS_LOOKBACK_HOURS = 36

# ---------------------------------------------------------------------------
# Article-body enrichment (utils/article_fetcher.py): Google News and Bigpara
# list views only ever expose a headline, never real article content (see
# the 2026-08-13 diagnosis — this was why LLM synthesis produced generic
# filler instead of analyzing anything). Enrichment fetches the item's own
# page and extracts the real body via trafilatura, but costs an extra
# request (+ a Google News redirect-token decode) per item, so it's bounded
# to the N most recent time-filtered items per ticker per source rather than
# applied to every item.
# ---------------------------------------------------------------------------
MAX_ARTICLES_TO_ENRICH_PER_TICKER = 8

# ---------------------------------------------------------------------------
# Cross-source dedup priority (nlp/dedup.py): when the same story is reported
# by more than one source, the higher-priority source wins and is kept.
# GOOGLE_NEWS/BIGPARA outrank KAP because they're readable news write-ups,
# while a raw KAP disclosure is terser — but the KAP link is never dropped
# outright, it's preserved on the surviving item via NewsItem.related_kap_url.
# ---------------------------------------------------------------------------
SOURCE_PRIORITY: dict[SourceType, int] = {
    SourceType.GOOGLE_NEWS: 2,
    SourceType.BIGPARA: 2,
    SourceType.KAP: 1,
    SourceType.COMPANY_IR: 0,
    SourceType.WEB_SEARCH_FALLBACK: 0,
    SourceType.SEC_EDGAR: 0,
    SourceType.YAHOO_FINANCE: 0,
    SourceType.FINVIZ: 0,
    SourceType.PR_NEWSWIRE: 0,
}
