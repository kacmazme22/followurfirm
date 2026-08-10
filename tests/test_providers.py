"""
Manual verification for nlp/providers/{base,noop_provider,factory}.py.

Run directly:

    python -m tests.test_providers
"""

from __future__ import annotations

import asyncio

from config.constants import NewsCategory, SourceType
from config.settings import AppConfig, RawYamlConfig, Settings, SourcesConfig
from config.settings import BigparaSourceConfig, KapSourceConfig
from nlp.providers.factory import get_provider
from nlp.providers.groq_provider import GroqProvider
from nlp.providers.noop_provider import NoopProvider
from scrapers.models import CompanyReport, NewsItem, SynthesizedCompanyReport


def _sample_app_config(ai_provider: str) -> AppConfig:
    yaml_config = RawYamlConfig(
        tickers={"bist": [], "global": []},
        recipients=["you@example.com"],
        sources=SourcesConfig(
            kap=KapSourceConfig(base_url="https://www.kap.org.tr", disclosure_endpoint="/tr/api/disclosure/members/byCriteria"),
            bigpara=BigparaSourceConfig(base_url="https://bigpara.hurriyet.com.tr"),
        ),
    )
    return AppConfig(yaml=yaml_config, env=Settings(ai_provider=ai_provider, groq_api_key="dummy-key-for-construction-only"))


def test_factory_returns_noop_provider() -> None:
    provider = get_provider(_sample_app_config("noop"))
    assert isinstance(provider, NoopProvider)
    print("[OK] get_provider(noop) -> NoopProvider")


def test_factory_returns_groq_provider() -> None:
    provider = get_provider(_sample_app_config("groq"))
    assert isinstance(provider, GroqProvider)
    print("[OK] get_provider(groq) -> GroqProvider (no real API call made, just construction)")


def _thyao_report_with_two_items() -> CompanyReport:
    report = CompanyReport(ticker="THYAO", company_name="Türk Hava Yolları")
    report.add_item(
        NewsItem(
            ticker="THYAO",
            title="THY 2. çeyrek bilanço açıkladı",
            url="https://www.kap.org.tr/tr/Bildirim/1",
            related_kap_url="https://www.kap.org.tr/tr/Bildirim/2",
            source=SourceType.KAP,
            category=NewsCategory.FINANCIALS,
            body_snippet="Net kar beklentileri aştı.",
        )
    )
    report.add_item(
        NewsItem(
            ticker="THYAO",
            title="X" * 150,  # deliberately over SynthesizedSection.subheading's max_length=100
            source=SourceType.BIGPARA,
            category=NewsCategory.FINANCIALS,
            body_snippet=None,  # no snippet -> narrative should fall back to title
        )
    )
    return report


async def test_noop_converts_to_synthesized_report() -> None:
    provider = NoopProvider()
    result = await provider.summarize_company_report(_thyao_report_with_two_items())

    assert isinstance(result, SynthesizedCompanyReport)
    financials = dict(result.ordered_sections())[NewsCategory.FINANCIALS]
    assert len(financials) == 2

    first = financials[0]
    assert first.subheading == "THY 2. çeyrek bilanço açıkladı"
    assert first.narrative == "Net kar beklentileri aştı."
    assert [str(u) for u in first.source_urls] == [
        "https://www.kap.org.tr/tr/Bildirim/1",
        "https://www.kap.org.tr/tr/Bildirim/2",
    ], "url then related_kap_url, in that order"
    print("[OK] NoopProvider: item -> SynthesizedSection, url+related_kap_url both preserved")

    second = financials[1]
    assert len(second.subheading) <= 100, "subheading must respect max_length=100"
    assert second.subheading.endswith("…"), "over-length title should be truncated with an ellipsis"
    assert second.narrative == "X" * 150, "narrative has no length cap, falls back to full title when no snippet"
    print("[OK] NoopProvider: over-length title truncated for subheading, narrative falls back to title")


async def main() -> None:
    test_factory_returns_noop_provider()
    test_factory_returns_groq_provider()
    await test_noop_converts_to_synthesized_report()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
