"""
KAP disclosure scraper — talks to KAP's internal JSON API directly. No HTML
parsing and no JS rendering needed: KAP's public pages are a Next.js SPA with
no server-rendered disclosure data, but its `/tr/api/disclosure/members/byCriteria`
endpoint (undocumented, discovered via third-party notes and independently
verified with a live POST) returns the same data as plain JSON.

Field mapping decisions:
  - `subject` -> raw_title AND raw_disclosure_type: KAP's own general
    category label for the disclosure (e.g. "Tertip İhraç Belgesi", "Pay
    Alım Satım Bildirimi") — exactly the free text KAP_KEYWORD_CATEGORY_MAP
    matches against, and what a reader should see first.
  - `summary` -> raw_body_snippet: the disclosure's specific detail text,
    shown after the general category/title.
  - `disclosureClass` (ODA/DKB/DG/FR/...) is intentionally NOT filtered
    here — what each code means isn't fully mapped yet. Everything passes
    through; add an exclude list to config/constants.py later if some
    classes turn out to be noise.
"""

from __future__ import annotations

from datetime import date, timedelta

from config.constants import SourceType
from config.settings import KapSourceConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem

DISCLOSURE_REFERER = "https://www.kap.org.tr/tr/bildirim-sorgu"
DISCLOSURE_DETAIL_URL_TEMPLATE = "https://www.kap.org.tr/tr/Bildirim/{disclosure_index}"

# Disclosures can land just after midnight, so the query window is wider than
# a strict "yesterday->today". This is safe because NewsItem's content_hash
# dedup (scrapers/models.py) already collapses the same disclosure showing up
# in two consecutive runs — a wide window + dedup beats a tight window that
# risks silently dropping a disclosure at the day boundary.
LOOKBACK_DAYS = 2


class KapScraper(BaseScraper):
    """Fetches recent disclosures from KAP's byCriteria JSON API and filters
    them down to this scraper's ticker."""

    def __init__(
        self,
        ticker: TickerConfig,
        source_config: KapSourceConfig,
        http_client=None,
    ) -> None:
        super().__init__(ticker, source_config.politeness, http_client)
        self.source_config = source_config

    async def fetch_raw(self) -> list[RawScrapedItem]:
        today = date.today()
        from_date = today - timedelta(days=LOOKBACK_DAYS)

        url = f"{self.source_config.base_url}{self.source_config.disclosure_endpoint}"
        body = {
            "fromDate": from_date.isoformat(),
            "toDate": today.isoformat(),
            "mkkMemberOidList": [],
            "subjectList": [],
        }

        response = await self._post(url, json=body, headers={"Referer": DISCLOSURE_REFERER})
        disclosures = response.json()

        items: list[RawScrapedItem] = []
        for disclosure in disclosures:
            if self.ticker.symbol not in self._ticker_codes(disclosure):
                continue

            items.append(
                RawScrapedItem(
                    source=SourceType.KAP,
                    ticker=self.ticker.symbol,
                    raw_title=disclosure.get("subject") or disclosure.get("summary") or "",
                    raw_url=self._disclosure_url(disclosure.get("disclosureIndex")),
                    raw_published_at=disclosure.get("publishDate"),
                    raw_body_snippet=disclosure.get("summary"),
                    raw_disclosure_type=disclosure.get("subject"),
                    extra={
                        "disclosure_index": disclosure.get("disclosureIndex"),
                        "disclosure_class": disclosure.get("disclosureClass"),
                        "kap_title": disclosure.get("kapTitle"),
                    },
                )
            )
            if len(items) >= self.source_config.max_items_per_ticker:
                break

        return items

    @staticmethod
    def _ticker_codes(disclosure: dict) -> set[str]:
        """`stockCodes` is a comma-separated string (or null) of tickers the
        disclosure relates to."""
        raw = disclosure.get("stockCodes") or ""
        return {code.strip() for code in raw.split(",") if code.strip()}

    @staticmethod
    def _disclosure_url(disclosure_index: int | None) -> str | None:
        if disclosure_index is None:
            return None
        return DISCLOSURE_DETAIL_URL_TEMPLATE.format(disclosure_index=disclosure_index)
