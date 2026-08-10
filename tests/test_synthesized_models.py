"""
Manual verification for SynthesizedSection/SynthesizedCompanyReport
(scrapers/models.py). Purely synthetic — no LLM call exists yet, this task
is data structure only.

Run directly:

    python -m tests.test_synthesized_models
"""

from __future__ import annotations

from config.constants import NewsCategory
from scrapers.models import SynthesizedCompanyReport, SynthesizedSection


def main() -> None:
    report = SynthesizedCompanyReport(
        ticker="THYAO",
        company_name="Türk Hava Yolları",
        sections_by_category={
            NewsCategory.KAP_MATERIAL: [
                SynthesizedSection(
                    subheading="Özel Durum Açıklaması",
                    narrative="THY, filo genişletme kararını KAP üzerinden duyurdu.",
                    source_urls=["https://www.kap.org.tr/tr/Bildirim/1641971"],
                ),
            ],
            NewsCategory.FINANCIALS: [
                SynthesizedSection(
                    subheading="2. Çeyrek Sonuçları",
                    narrative="Net kar beklentilerin üzerinde gerçekleşti.",
                    source_urls=[
                        "https://www.kap.org.tr/tr/Bildirim/1639061",
                        "https://bigpara.hurriyet.com.tr/haberler/thy-bilanco_ID1/",
                    ],
                ),
                SynthesizedSection(
                    subheading="Analist Yorumları",
                    narrative="Aracı kurumlar hedef fiyatları yukarı revize etti.",
                ),
            ],
            # NEW_BUSINESS and GENERAL_SECTOR deliberately absent -> must be skipped.
        },
    )

    ordered = report.ordered_sections()
    categories_in_order = [cat for cat, _ in ordered]

    assert categories_in_order == [NewsCategory.FINANCIALS, NewsCategory.KAP_MATERIAL], (
        f"expected fixed order (FINANCIALS before KAP_MATERIAL, empty categories skipped), got {categories_in_order}"
    )
    print(f"[OK] ordered_sections() category order: {[c.value for c in categories_in_order]}")

    financials_sections = dict(ordered)[NewsCategory.FINANCIALS]
    assert len(financials_sections) == 2
    assert financials_sections[0].subheading == "2. Çeyrek Sonuçları"
    assert len(financials_sections[0].source_urls) == 2
    print("[OK] section contents (subheading, narrative, source_urls) preserved correctly")

    assert not report.is_empty()
    print("[OK] is_empty() == False for a report with sections")

    empty_report = SynthesizedCompanyReport(ticker="ASTOR", company_name="Astor Enerji")
    assert empty_report.is_empty()
    assert empty_report.ordered_sections() == []
    print("[OK] is_empty() == True and ordered_sections() == [] for an empty report")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
