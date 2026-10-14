"""
Company investor-relations pages: new presentations and reports.

KAP carries most of what a company publishes, but not everything: corporate
presentations, results decks and ESG reports often appear only on the IR
site. Each configured page is read and its document links compared with
the ones seen on earlier runs (kept in the digest memory); a link that
wasn't there before becomes an item ("YEO Teknoloji yatırımcı ilişkileri
sayfasında yeni doküman: ...") with the link, for the analyst/IR section.

The first time a page is read, everything on it is only recorded, not
reported; a page that suddenly shows many "new" links (a site redesign)
is re-recorded the same way instead of flooding the mail.

Pages verified from a GitHub runner on 2026-10-14 (scripts/probe_sources.py):
Akbank and YEO list their documents in plain HTML. Koç Holding's
"Sunumlar ve Bültenler" builds its list with JavaScript (no document links
in the HTML) and gubretas.com.tr fails certificate verification, so those
two rely on KAP for now.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

from config.constants import SourceType
from config.settings import PolitenessConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.models import RawScrapedItem
from utils.memory import DigestMemory

logger = logging.getLogger(__name__)

MAX_NEW_PER_PAGE = 5
_DOCUMENT_LINK = re.compile(r"\.pdf(\?|$)|/detay/|/yil-liste/", re.IGNORECASE)
_ENGLISH_PATH = re.compile(r"/en/|_en\.pdf|-en\.pdf|english", re.IGNORECASE)
# Investor documents, not the site's news or product pages.
_DOCUMENT_WORDS = re.compile(
    r"sunum|rapor|sonu[cç]|finansal|faaliyet|tablo|dipnot|bdr|beklenti|presentation|report|results", re.IGNORECASE
)


def document_links(html: str, page_url: str) -> dict[str, str]:
    """Absolute document URL -> human label, in page order."""
    soup = BeautifulSoup(html, "html.parser")
    links: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"].strip())
        if not _DOCUMENT_LINK.search(href) or _ENGLISH_PATH.search(href):
            continue
        label = " ".join(anchor.get_text(" ").split()).removesuffix(" İndir").strip()
        if not label or len(label) < 4:
            label = _label_from_filename(href)
        if not _DOCUMENT_WORDS.search(f"{label} {unquote(href)}"):
            continue
        if href not in links or len(label) > len(links[href]):
            links[href] = label
    return links


def _label_from_filename(url: str) -> str:
    name = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[-_]+", " ", name).split())


class IrPageScraper(BaseScraper):
    def __init__(
        self,
        ticker: TickerConfig,
        page_urls: list[str],
        memory: DigestMemory,
        politeness: PolitenessConfig,
        http_client=None,
    ) -> None:
        super().__init__(ticker, politeness, http_client)
        self.page_urls = page_urls
        self.memory = memory

    async def fetch_raw(self) -> list[RawScrapedItem]:
        items: list[RawScrapedItem] = []
        for page_url in self.page_urls:
            try:
                response = await self._get(page_url)
            except Exception as exc:
                logger.info("%s: IR sayfası okunamadı (%s): %s", self.ticker.symbol, page_url, exc)
                continue
            links = document_links(response.text, str(response.url))
            items += self._new_documents(page_url, links)
        return items

    def _new_documents(self, page_url: str, links: dict[str, str]) -> list[RawScrapedItem]:
        seen = self.memory.ir_seen.get(page_url)
        self.memory.ir_seen[page_url] = sorted(set(links) | set(seen or []))
        if seen is None:
            logger.info("%s: IR sayfası ilk kez okundu, %d doküman kaydedildi: %s", self.ticker.symbol, len(links), page_url)
            return []
        new = [url for url in links if url not in seen]
        if len(new) > MAX_NEW_PER_PAGE:
            logger.warning(
                "%s: IR sayfasında %d yeni link (site değişmiş olabilir), bildirilmeden kaydedildi: %s",
                self.ticker.symbol, len(new), page_url,
            )
            return []
        logger.info("%s: IR sayfası %d doküman, %d yeni: %s", self.ticker.symbol, len(links), len(new), page_url)
        return [
            RawScrapedItem(
                source=SourceType.COMPANY_IR,
                ticker=self.ticker.symbol,
                raw_title=f"{self.ticker.name} yatırımcı ilişkileri sayfasında yeni doküman: {links[url]}",
                raw_url=url,
                raw_disclosure_type="Yatırımcı sunumu",
            )
            for url in new
        ]
