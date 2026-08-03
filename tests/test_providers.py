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
from nlp.providers.noop_provider import NoopProvider
from scrapers.models import CompanyReport, NewsItem


def _sample_app_config(ai_provider: str) -> AppConfig:
    yaml_config = RawYamlConfig(
        tickers={"bist": [], "global": []},
        recipients=["you@example.com"],
        sources=SourcesConfig(
            kap=KapSourceConfig(base_url="https://www.kap.org.tr", disclosure_endpoint="/tr/api/disclosure/members/byCriteria"),
            bigpara=BigparaSourceConfig(base_url="https://bigpara.hurriyet.com.tr"),
        ),
    )
    return AppConfig(yaml=yaml_config, env=Settings(ai_provider=ai_provider))


def test_factory_returns_noop_provider() -> None:
    provider = get_provider(_sample_app_config("noop"))
    assert isinstance(provider, NoopProvider)
    print("[OK] get_provider(noop) -> NoopProvider")


def test_factory_groq_not_implemented() -> None:
    try:
        get_provider(_sample_app_config("groq"))
    except NotImplementedError as exc:
        assert "Groq" in str(exc)
        print(f"[OK] get_provider(groq) raises NotImplementedError: {exc}")
    else:
        raise AssertionError("expected NotImplementedError for ai_provider='groq'")


async def test_noop_leaves_report_unchanged() -> None:
    report = CompanyReport(ticker="THYAO", company_name="Türk Hava Yolları")
    report.add_item(
        NewsItem(
            ticker="THYAO",
            title="THY 2. çeyrek bilanço açıkladı",
            source=SourceType.KAP,
            category=NewsCategory.FINANCIALS,
            body_snippet="Net kar beklentileri aştı.",
        )
    )

    provider = NoopProvider()
    result = await provider.summarize_company_report(report)

    assert result is report, "noop should return the exact same object, not a copy"
    assert result.total_items == 1
    assert result.items_by_category[NewsCategory.FINANCIALS][0].body_snippet == "Net kar beklentileri aştı."
    print("[OK] NoopProvider.summarize_company_report leaves the report byte-for-byte unchanged")


async def main() -> None:
    test_factory_returns_noop_provider()
    test_factory_groq_not_implemented()
    await test_noop_leaves_report_unchanged()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
