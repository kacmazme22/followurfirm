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

Two ways to use this scraper:
  - `fetch_raw()` / `run()`: the normal BaseScraper contract, one HTTP
    request for `self.ticker` only.
  - `fetch_all_raw(tickers)`: bypasses per-ticker filtering entirely and
    makes exactly ONE request covering every company (mkkMemberOidList=[]),
    then groups client-side by `stockCodes`. main.py uses this instead of
    instantiating one KapScraper per ticker, since N per-ticker instances
    would otherwise each make the identical POST — BaseScraper's in-run
    cache is per-instance, so it can't dedupe the request across them.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from config.constants import SourceType
from config.settings import KapSourceConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem

logger = logging.getLogger(__name__)

DISCLOSURE_REFERER = "https://www.kap.org.tr/tr/bildirim-sorgu"
DISCLOSURE_DETAIL_URL_TEMPLATE = "https://www.kap.org.tr/tr/Bildirim/{disclosure_index}"

# Disclosures can land just after midnight, so the query window is wider than
# a strict "yesterday->today". This is safe because NewsItem's content_hash
# dedup (scrapers/models.py) already collapses the same disclosure showing up
# in two consecutive runs — a wide window + dedup beats a tight window that
# risks silently dropping a disclosure at the day boundary.
LOOKBACK_DAYS = 2


class KapScraper(BaseScraper):
    """Fetches recent disclosures from KAP's byCriteria JSON API."""

    def __init__(
        self,
        ticker: TickerConfig,
        source_config: KapSourceConfig,
        http_client=None,
    ) -> None:
        super().__init__(ticker, source_config.politeness, http_client)
        self.source_config = source_config

    async def fetch_raw(self) -> list[RawScrapedItem]:
        disclosures = await self._fetch_disclosures_json()
        return self._items_for_ticker(disclosures, self.ticker.symbol)

    async def fetch_all_raw(self, tickers: list[TickerConfig]) -> dict[str, list[RawScrapedItem]]:
        """Single POST covering every company, grouped client-side into a
        dict keyed by ticker symbol. Unlike fetch_raw()/run(), this ignores
        `self.ticker` entirely (it's only a label on this instance, e.g. for
        logging elsewhere) and is NOT wrapped in BaseScraper.run()'s
        try/except — callers should wrap this call themselves, the same way
        fetch_raw() is unprotected until run() wraps it."""
        async with self._client_session():
            disclosures = await self._fetch_disclosures_json()
        grouped = {ticker.symbol: self._items_for_ticker(disclosures, ticker.symbol) for ticker in tickers}
        logger.info(
            "KAP: toplam %d bildirim, takip edilenler: %s",
            len(disclosures), {symbol: len(items) for symbol, items in grouped.items()},
        )
        return grouped

    async def _fetch_disclosures_json(self) -> list[dict]:
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
        return response.json()

    def _items_for_ticker(self, disclosures: list[dict], ticker_symbol: str) -> list[RawScrapedItem]:
        items: list[RawScrapedItem] = []
        for disclosure in disclosures:
            if ticker_symbol not in self._ticker_codes(disclosure):
                continue

            items.append(
                RawScrapedItem(
                    source=SourceType.KAP,
                    ticker=ticker_symbol,
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
