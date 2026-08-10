"""
Renders templates/email_base.html against a synthetic DigestRun so the
output can be visually inspected. Builds CompanyReports (raw NewsItems) the
same way the real pipeline does, then runs them through NoopProvider to get
the SynthesizedCompanyReport shape the template actually consumes — DigestRun
holds SynthesizedCompanyReport now, not CompanyReport (see main.py /
nlp/providers/).

Covers: multiple tickers, multiple categories, an item with related_kap_url,
an empty ticker, and a footer error.

Run directly:

    python -m tests.test_email_template

Writes rendered HTML to tests/output/sample_digest.html (gitignored/scratch,
not meant to be committed — see .gitignore).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from config.constants import NewsCategory, SourceType
from nlp.providers.noop_provider import NoopProvider
from scrapers.models import CompanyReport, DigestRun, NewsItem
from templates.styles import template_colors

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
OUTPUT_PATH = Path(__file__).parent / "output" / "sample_digest.html"


def _build_sample_reports() -> list[CompanyReport]:
    thyao = CompanyReport(ticker="THYAO", company_name="Türk Hava Yolları")
    thyao.add_item(
        NewsItem(
            ticker="THYAO",
            title="THY 2. Çeyrek Finansal Sonuçları Açıklandı",
            url="https://www.kap.org.tr/tr/Bildirim/1641971",
            source=SourceType.KAP,
            category=NewsCategory.FINANCIALS,
            body_snippet=(
                "Türk Hava Yolları, 2. çeyrekte net kar beklentilerin üzerinde gerçekleşti. "
                "Yolcu doluluk oranı geçen yıla göre arttı ve kargo gelirlerinde de belirgin bir yükseliş görüldü."
            ),
        )
    )
    thyao.add_item(
        NewsItem(
            ticker="THYAO",
            title="THY'den Yeni Uçak Alım Anlaşması",
            url="https://bigpara.hurriyet.com.tr/haberler/thy-ucak-alim_ID999/",
            related_kap_url="https://www.kap.org.tr/tr/Bildirim/1641972",
            source=SourceType.BIGPARA,
            category=NewsCategory.NEW_BUSINESS,
            body_snippet="Filoya 10 yeni geniş gövde uçak katılacak, teslimatlar önümüzdeki yıl başlayacak.",
        )
    )
    thyao.add_item(
        NewsItem(
            ticker="THYAO",
            title="Yönetim Kurulu Kararı Hk.",
            source=SourceType.KAP,
            category=NewsCategory.KAP_MATERIAL,
            body_snippet=None,
        )
    )

    garan = CompanyReport(ticker="GARAN", company_name="Garanti BBVA")
    garan.add_item(
        NewsItem(
            ticker="GARAN",
            title="Bankacılık Sektöründe Faiz Görünümü",
            url="https://news.google.com/rss/articles/example",
            source=SourceType.GOOGLE_NEWS,
            category=NewsCategory.GENERAL_SECTOR,
            body_snippet="Sektör genelinde faiz indirimi beklentileri güçlendi, bankacılık hisseleri hareketlendi.",
        )
    )

    astor = CompanyReport(ticker="ASTOR", company_name="Astor Enerji")  # deliberately empty

    return [thyao, garan, astor]


async def main() -> None:
    provider = NoopProvider()
    synthesized_reports = [await provider.summarize_company_report(r) for r in _build_sample_reports()]

    digest_run = DigestRun(run_date=datetime(2026, 8, 3, 8, 0, 0))
    digest_run.company_reports = synthesized_reports
    digest_run.add_error(SourceType.BIGPARA, "EREGL", "timeout after 15s")

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("email_base.html")
    html = template.render(digest_run=digest_run, colors=template_colors())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")

    assert "THYAO" in html
    assert "GARAN" in html
    assert "ASTOR" in html
    assert "Bugün ASTOR için yeni bir gelişme yok." in html
    assert "https://www.kap.org.tr/tr/Bildirim/1641972" in html, "related_kap_url should render as a source link"
    assert "timeout after 15s" in html, "footer should surface the error"
    assert "EREGL" in html

    print(f"[OK] rendered {len(html)} chars -> {OUTPUT_PATH}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
