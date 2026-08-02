"""
Shared Pydantic data models for the scraping layer.

Two-tier model design, deliberately:

  RawScrapedItem  -- exactly what a scraper pulled off the page/feed, with
                     minimal transformation. Kept around for debugging and
                     so a scraper bug doesn't corrupt data other modules rely
                     on. Never passed to the categorizer or summarizer.

  NewsItem        -- normalized, validated, ticker-tagged item. This is the
                     shared contract every scraper must produce (via a
                     `to_news_item()` conversion in each scraper module).
                     dedup.py, categorizer.py, and the email renderer all
                     operate exclusively on NewsItem.

Keeping these separate means fuzzy-matching/dedup logic in dedup.py never has
to defensively guess whether a title has already been cleaned, whitespace
collapsed, etc. — by the time something is a NewsItem, that's guaranteed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from config.constants import DEFAULT_CATEGORY, NewsCategory, SourceType


# ---------------------------------------------------------------------------
# Raw layer
# ---------------------------------------------------------------------------

class RawScrapedItem(BaseModel):
    """
    Minimally-processed output straight from a scraper. Intentionally loose:
    fields that might be missing/malformed on a given source are Optional
    rather than required, because scrapers must degrade gracefully (per
    project instruction #2) rather than raising on a single bad item.
    """

    source: SourceType
    ticker: str
    raw_title: str
    raw_url: str | None = None
    raw_published_at: str | None = None  # unparsed date string, source-specific format
    raw_body_snippet: str | None = None
    raw_disclosure_type: str | None = None  # KAP "bildirim türü" field, if applicable
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict = Field(default_factory=dict)  # escape hatch for source-specific quirks


# ---------------------------------------------------------------------------
# Normalized layer
# ---------------------------------------------------------------------------

class NewsItem(BaseModel):
    """
    The canonical, validated unit of news that flows through dedup,
    categorization, summarization, and rendering.
    """

    ticker: str = Field(..., min_length=1, max_length=12)
    title: str = Field(..., min_length=1)
    url: HttpUrl | None = None
    source: SourceType
    published_at: datetime | None = None
    body_snippet: str | None = Field(default=None, max_length=2000)
    category: NewsCategory = DEFAULT_CATEGORY
    disclosure_type_raw: str | None = None

    # Populated by dedup.py; not set at construction time.
    content_hash: str | None = None

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        return " ".join(v.split())  # collapse whitespace/newlines

    @model_validator(mode="after")
    def _compute_content_hash(self) -> "NewsItem":
        if self.content_hash is None:
            self.content_hash = self._make_hash()
        return self

    def _make_hash(self) -> str:
        """
        URL-based hash when a URL exists (normalized to strip tracking
        params/fragments), else a title+ticker fallback hash. This is the
        first-pass, cheap dedup signal; dedup.py layers fuzzy title matching
        on top for cross-source duplicates that don't share a URL.
        """
        if self.url:
            normalized = self._normalize_url(str(self.url))
            basis = f"url:{normalized}"
        else:
            basis = f"title:{self.ticker}:{self.title.lower()}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_url(url: str) -> str:
        parts = urlsplit(url)
        # Strip query string and fragment — utm_* and session params are the
        # most common cause of otherwise-identical URLs hashing differently.
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


class KapDisclosure(NewsItem):
    """
    KAP-specific specialization of NewsItem. Adds fields unique to KAP
    disclosures that the email template may want to surface (e.g. a direct
    "view on KAP" link distinct from a generic source URL).
    """

    disclosure_id: str | None = None  # KAP's internal disclosure reference number
    company_kap_code: str | None = None

    @model_validator(mode="after")
    def _force_kap_source(self) -> "KapDisclosure":
        object.__setattr__(self, "source", SourceType.KAP)
        return self


class CompanyReport(BaseModel):
    """
    One ticker's fully-assembled section of the digest: all its NewsItems,
    already deduped and categorized, grouped by category for direct template
    consumption.
    """

    ticker: str
    company_name: str
    items_by_category: dict[NewsCategory, list[NewsItem]] = Field(default_factory=dict)
    total_items: int = 0
    sources_used: set[SourceType] = Field(default_factory=set)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_item(self, item: NewsItem) -> None:
        self.items_by_category.setdefault(item.category, []).append(item)
        self.total_items += 1
        self.sources_used.add(item.source)

    def is_empty(self) -> bool:
        return self.total_items == 0

    def ordered_categories(self) -> list[tuple[NewsCategory, list[NewsItem]]]:
        """Returns categories in the fixed display order defined by the enum,
        skipping empty ones — matches project instruction section 2's ordering."""
        order = [
            NewsCategory.NEW_BUSINESS,
            NewsCategory.FINANCIALS,
            NewsCategory.KAP_MATERIAL,
            NewsCategory.GENERAL_SECTOR,
        ]
        return [(cat, self.items_by_category[cat]) for cat in order if self.items_by_category.get(cat)]


class DigestRun(BaseModel):
    """Top-level container for one full pipeline execution — what gets
    handed to the email renderer."""

    run_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    company_reports: list[CompanyReport] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)  # graceful-degradation log surfaced in footer/logs

    def add_error(self, source: SourceType | str, ticker: str, message: str) -> None:
        source_label = source.value if isinstance(source, SourceType) else source
        self.errors.append(f"[{source_label}] {ticker}: {message}")

    @property
    def has_any_content(self) -> bool:
        return any(not r.is_empty() for r in self.company_reports)
