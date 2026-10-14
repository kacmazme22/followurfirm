"""One log line per candidate source page: final URL after redirects, HTTP
status, and the PDF / announcement links found on it. Run from the workflow
because only GitHub's runners can reach these sites."""

import asyncio
import re

import httpx
from bs4 import BeautifulSoup

CANDIDATES = [
    "https://www.akbankinvestorrelations.com/tr",
    "https://www.akbank.com/tr-tr/yatirimci-iliskileri",
    "https://www.koc.com.tr/yatirimci-iliskileri/sunumlar-ve-bultenler/yatirimci-sunumlari",
    "https://www.koc.com.tr/yatirimci-iliskileri",
    "https://www.gubretas.com.tr/yatirimci-iliskileri",
    "https://www.gubretas.com.tr/tr/yatirimci-iliskileri",
    "https://www.yeo.com.tr/yatirimci-iliskileri",
    "https://www.yeotk.com.tr/yatirimci-iliskileri",
    "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin",
    "https://www.tcmb.gov.tr",
    "https://data.tuik.gov.tr/Kategori/GetKategori?p=Enflasyon-ve-Fiyat-106",
    "https://www.tuik.gov.tr",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}


async def probe(client: httpx.AsyncClient, url: str) -> None:
    try:
        response = await client.get(url)
    except Exception as exc:
        print(f"PROBE {url} -> {type(exc).__name__}: {exc}")
        return
    soup = BeautifulSoup(response.text, "html.parser")
    links = [(a.get_text(" ", strip=True)[:60], a["href"]) for a in soup.find_all("a", href=True)]
    pdfs = [link for link in links if re.search(r"\.pdf|download|sunum|presentation", link[1], re.I)]
    title = soup.title.get_text(strip=True)[:80] if soup.title else ""
    print(f"PROBE {url} -> {response.status_code} {response.url} | title={title!r} | {len(links)} link, {len(pdfs)} pdf/sunum")
    for text, href in pdfs[:8]:
        print(f"PROBE   {text!r} {href[:160]}")


async def main() -> None:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=HEADERS) as client:
        for url in CANDIDATES:
            await probe(client, url)


asyncio.run(main())
