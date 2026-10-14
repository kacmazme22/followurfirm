"""
Company investor-relations pages: new presentations and reports.

KAP carries most of what a company publishes, but not everything: corporate
presentations, results decks and ESG reports often appear only on the IR
site. Nothing here is configured per company: for every tracked ticker

  1. the IR site is found from the company's website as listed on its KAP
     company page (linked from any of its filings): the site's "Yatırımcı
     İlişkileri" link, on the same site or on a separate domain carrying
     the company name (akbankinvestorrelations.com). The KAP page comes
     from a filing's link, or from KAP's list of BIST companies. (Web
     search was tried first: from GitHub's runners Bing's RSS answered
     with unrelated results and DuckDuckGo with a bot check.)
  2. on that page, links to document listings ("Sunumlar", "Finansal
     Raporlar", "Yayınlar") on the same site are followed one level down;
  3. the document links (PDFs, report pages) on those pages are compared
     with the ones seen on earlier runs, kept in the digest memory; a new one
     becomes an item for the analyst/IR section, with its link.

What was found is kept in the memory too and searched again after
REDISCOVER_DAYS (sooner when nothing was found), so a run normally costs
one request per watched page. `sources.company_ir_pages.urls` in
config.yaml can still pin pages for a ticker; it overrides the search.

The first time a page is read everything on it is only recorded, not
reported; a page that suddenly shows many "new" links (a site redesign) is
re-recorded the same way instead of flooding the mail.

Seen from a GitHub runner on 2026-10-14: Akbank and YEO list documents in
plain HTML; Koç Holding's list is built by JavaScript and gubretas.com.tr
fails certificate verification, so for those two the pages yield nothing
(logged) and KAP remains the source.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, timedelta
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

from config.constants import SourceType
from config.settings import PolitenessConfig, TickerConfig
from scrapers.base import BaseScraper
from scrapers.kap_scraper import BROWSER_HEADERS as KAP_HEADERS
from scrapers.kap_scraper import DETAIL_HOSTS as KAP_HOSTS
from scrapers.models import RawScrapedItem
from utils.memory import DigestMemory

logger = logging.getLogger(__name__)

MAX_NEW_PER_PAGE = 5
MAX_LISTING_PAGES = 4
REDISCOVER_DAYS = 30
RETRY_DISCOVERY_DAYS = 7

KAP_COMPANY_LIST = "/tr/bist-sirketler"

_DOCUMENT_LINK = re.compile(r"\.pdf(\?|$)|/detay/|/yil-liste/", re.IGNORECASE)
_ENGLISH_PATH = re.compile(r"/en/|_en\.pdf|-en\.pdf|english", re.IGNORECASE)
# Investor documents, not the site's news or product pages.
_DOCUMENT_WORDS = re.compile(
    r"sunum|rapor|sonu[cç]|finansal|faaliyet|tablo|dipnot|bdr|beklenti|presentation|report|results", re.IGNORECASE
)
# Links on an IR home page that lead to document listings.
_LISTING_WORDS = re.compile(
    r"sunum|rapor|yay[ıi]n|finansal|sonu[cç]|presentation|report|publication", re.IGNORECASE
)
# Words of a company name that say nothing about which company it is.
_GENERIC_NAME_WORDS = {
    "turk", "turkiye", "anonim", "sirketi", "holding", "teknoloji", "fabrikalari", "sanayi", "ticaret",
    "bankasi", "enerji", "yatirim", "grup", "group", "as", "aş", "tas",
}


def _ascii(text: str) -> str:
    text = text.replace("ı", "i").replace("İ", "i")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def name_tokens(ticker: TickerConfig) -> list[str]:
    """Distinctive parts of the company name (and the ticker) to look for in
    a domain: "Gübre Fabrikaları" -> ["gubre", "gubrf"]."""
    words = re.findall(r"[a-z0-9]+", _ascii(ticker.name))
    tokens = [w for w in words if len(w) >= 3 and w not in _GENERIC_NAME_WORDS]
    return tokens + [ticker.symbol.lower()]


def is_company_site(url: str, ticker: TickerConfig) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    return any(token in host for token in name_tokens(ticker))


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


def listing_pages(html: str, page_url: str) -> list[str]:
    """Same-site links from an IR home page to its document listings."""
    host = urlparse(page_url).netloc
    soup = BeautifulSoup(html, "html.parser")
    pages: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"].strip()).split("#")[0]
        label = " ".join(anchor.get_text(" ").split())
        if urlparse(href).netloc != host or _DOCUMENT_LINK.search(href) or _ENGLISH_PATH.search(href):
            continue
        if _LISTING_WORDS.search(f"{label} {unquote(urlparse(href).path)}") and href not in pages and href != page_url:
            pages.append(href)
    return pages[:MAX_LISTING_PAGES]


def _label_from_filename(url: str) -> str:
    name = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1])
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    return " ".join(re.sub(r"[-_]+", " ", name).split())


# KAP writes the site as "http://www.akbank.com" for some companies and as a
# bare "www.koc.com.tr" for others (YEOTK, GUBRF, KCHOL on 2026-10-14).
_URL_IN_TEXT = re.compile(r"(?:https?://|(?<![\w./])www\.)[\w.-]+\.[a-z]{2,}(?:/[\w./%-]*)?", re.IGNORECASE)
_SKIP_HOSTS = ("kap.org.tr", "mkk.com.tr", "borsaistanbul.com", "spk.gov.tr", "google", "facebook", "twitter",
               "linkedin", "instagram", "youtube", "w3.org", "schema.org")


def company_website(html: str, ticker: TickerConfig) -> str | None:
    """The company's site among the URLs on its KAP page ("İnternet Adresi")."""
    for url in _URL_IN_TEXT.findall(html):
        if not url.lower().startswith("http"):
            url = f"https://{url}"
        if any(skip in urlparse(url).netloc.lower() for skip in _SKIP_HOSTS):
            continue
        if is_company_site(url, ticker):
            return f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
    return None


_IR_LABEL = re.compile(r"yat[ıi]r[ıi]mc[ıi] ili[şs]kileri|investor relations", re.IGNORECASE)
_IR_PATH = re.compile(r"yatirimci|investor", re.IGNORECASE)


def ir_link_on_site(html: str, page_url: str, ticker: TickerConfig) -> str | None:
    """The "Yatırımcı İlişkileri" link on a company site: same site, or a
    separate IR domain that carries the company name
    (akbankinvestorrelations.com). A link labelled "Yatırımcı İlişkileri"
    wins over one that only has "investor" in its address; akbank.com also
    links its brokerage, yatirim.akbank.com, which must not be taken."""
    soup = BeautifulSoup(html, "html.parser")
    by_label, by_path = [], []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"].strip()).split("#")[0]
        if not href.startswith("http") or _ENGLISH_PATH.search(href):
            continue
        if urlparse(href).netloc != urlparse(page_url).netloc and not is_company_site(href, ticker):
            continue
        label = " ".join(anchor.get_text(" ").split())
        if _IR_LABEL.search(label):
            by_label.append(href)
        elif _IR_PATH.search(unquote(urlparse(href).netloc + urlparse(href).path)):
            by_path.append(href)
    return (by_label + by_path or [None])[0]


_KAP_COMPANY_PATH = re.compile(r"/tr/sirket-bilgileri/(?:ozet|genel)/[\w-]+")


def kap_company_paths(html: str, symbol: str, limit: int = 3) -> list[str]:
    """Company page links in KAP's company list, nearest to `symbol` first.
    The list also links auditors next to each company (2026-10-14: a
    ±400-character window picked "Güney Bağımsız Denetim" for KCHOL), so
    callers confirm each candidate page carries the symbol."""
    symbol_re = re.compile(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])")
    positions = [m.start() for m in symbol_re.finditer(html)]
    if not positions:
        return []
    distances: dict[str, int] = {}
    for match in _KAP_COMPANY_PATH.finditer(html):
        distance = min(abs(match.start() - p) for p in positions)
        path = match.group(0)
        distances[path] = min(distance, distances.get(path, distance))
    return sorted(distances, key=distances.get)[:limit]


class IrPageScraper(BaseScraper):
    def __init__(
        self,
        ticker: TickerConfig,
        memory: DigestMemory,
        politeness: PolitenessConfig,
        pinned_pages: list[str] | None = None,
        kap_company_url: str | None = None,
        http_client=None,
    ) -> None:
        super().__init__(ticker, politeness, http_client)
        self.memory = memory
        self.pinned_pages = pinned_pages or []
        self.kap_company_url = kap_company_url

    async def fetch_raw(self) -> list[RawScrapedItem]:
        pages = self.pinned_pages or await self._watched_pages()
        items: list[RawScrapedItem] = []
        for page_url in pages:
            html = await self._html(page_url)
            if html is not None:
                items += self._new_documents(page_url, document_links(html, page_url))
        return items

    # --- discovery -------------------------------------------------------

    async def _watched_pages(self) -> list[str]:
        known = self.memory.ir_pages.get(self.ticker.symbol)
        if known and date.fromisoformat(known["checked_on"]) > date.today() - timedelta(
            days=REDISCOVER_DAYS if known["pages"] else RETRY_DISCOVERY_DAYS
        ):
            return known["pages"]
        pages = await self._discover()
        if pages or self.kap_company_url:
            self.memory.ir_pages[self.ticker.symbol] = {"pages": pages, "checked_on": date.today().isoformat()}
        # else: no KAP filing to start from today; tried again on the next run
        return pages

    async def _discover(self) -> list[str]:
        home = await self._find_ir_home()
        if home is None:
            logger.info("%s: yatırımcı ilişkileri sayfası bulunamadı", self.ticker.symbol)
            return []
        html = await self._html(home)
        if html is None:
            return []
        pages = [home] + listing_pages(html, home)
        logger.info("%s: IR sayfaları bulundu: %s", self.ticker.symbol, ", ".join(pages))
        return pages

    async def _find_ir_home(self) -> str | None:
        website = await self._website_from_kap()
        if website:
            html = await self._html(website)
            if html is not None:
                ir_link = ir_link_on_site(html, website, self.ticker)
                logger.info("%s: KAP'taki web sitesi %s, IR bağlantısı: %s", self.ticker.symbol, website, ir_link)
                return ir_link or website
        return None

    async def _website_from_kap(self) -> str | None:
        """The company's own site, as listed on its KAP company page (linked
        from its filings, or found in KAP's list of BIST companies)."""
        candidates = [self.kap_company_url] if self.kap_company_url else await self._kap_company_urls_from_list()
        for url in candidates:
            try:
                response = await self._get(url, headers=dict(KAP_HEADERS))
                response.raise_for_status()
            except Exception as exc:
                logger.info("%s: KAP şirket sayfası okunamadı (%s): %s", self.ticker.symbol, url, exc)
                continue
            if self.ticker.symbol not in response.text:
                continue  # another company's page (an auditor listed next to it)
            website = company_website(response.text, self.ticker)
            logger.info("%s: KAP şirket sayfası adayı %s, web sitesi: %s", self.ticker.symbol, url, website)
            if website:
                self.kap_company_url = url
                return website
        logger.info("%s: KAP şirket sayfası bulunamadı (%d aday)", self.ticker.symbol, len(candidates))
        return None

    async def _kap_company_urls_from_list(self) -> list[str]:
        for host in KAP_HOSTS:
            try:
                response = await self._get(f"{host}{KAP_COMPANY_LIST}", headers=dict(KAP_HEADERS))
                response.raise_for_status()
            except Exception as exc:
                logger.info("KAP şirket listesi okunamadı (%s): %s", host, exc)
                continue
            return [f"{host}{path}" for path in kap_company_paths(response.text, self.ticker.symbol)]
        return []

    async def _html(self, url: str) -> str | None:
        try:
            response = await self._get(url)
            response.raise_for_status()
        except Exception as exc:
            logger.info("%s: IR sayfası okunamadı (%s): %s: %s", self.ticker.symbol, url, type(exc).__name__, exc)
            return None
        return response.text

    # --- new documents ---------------------------------------------------

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
