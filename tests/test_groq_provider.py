"""
Manual verification for nlp/providers/groq_provider.py — real Groq API calls
against real THYAO data (all categories, not just Finansal). Requires
GROQ_API_KEY in .env.

Two scenarios:
  1. Success: GroqProvider(settings) (default max_tokens=6000) against a
     real THYAO CompanyReport built the same way main.py does.
  2. Deliberate failure + fallback: GroqProvider(settings, max_tokens=50) —
     too low to finish a JSON section, so at least one category should hit
     finish_reason != "stop" and raise RuntimeError. Then replicates
     main.py's _build_company_report() fallback (catch -> NoopProvider ->
     digest_run.add_error()) to prove the safety net actually works.

Run directly:

    python -m tests.test_groq_provider
"""

from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import scrapers.kap_scraper as kap_scraper_module
from config.settings import PolitenessConfig, get_settings
from nlp.categorizer import categorize_batch
from nlp.dedup import deduplicate
from nlp.providers.groq_provider import GroqProvider
from nlp.providers.noop_provider import NoopProvider
from scrapers.bigpara_scraper import BigparaScraper
from scrapers.google_news_scraper import GoogleNewsScraper
from scrapers.kap_scraper import KapScraper
from scrapers.models import CompanyReport, DigestRun


async def _real_thyao_company_report() -> CompanyReport:
    """Builds a real THYAO CompanyReport the same way main.py's
    _build_company_report() does (KAP widened to 7 days like earlier tests,
    to guarantee non-trivial data across categories, not just Google News)."""
    settings = get_settings()
    thyao = next(t for t in settings.active_bist_tickers if t.symbol == "THYAO")

    original_lookback = kap_scraper_module.LOOKBACK_DAYS
    kap_scraper_module.LOOKBACK_DAYS = 7
    try:
        kap_result = await KapScraper(thyao, settings.yaml.sources.kap).run()
    finally:
        kap_scraper_module.LOOKBACK_DAYS = original_lookback

    bigpara_result = await BigparaScraper(thyao, settings.yaml.sources.bigpara).run()
    google_result = await GoogleNewsScraper(
        thyao, settings.yaml.sources.google_news_rss, PolitenessConfig()
    ).run()

    raw_items = kap_result.items + bigpara_result.items + google_result.items
    news_items = deduplicate(categorize_batch(raw_items))

    report = CompanyReport(ticker=thyao.symbol, company_name=thyao.name)
    for item in news_items:
        report.add_item(item)
    return report


async def test_groq_provider_success() -> None:
    settings = get_settings()
    report = await _real_thyao_company_report()
    print(f"THYAO CompanyReport: {report.total_items} item(s) across {len(report.items_by_category)} category(ies)\n")

    provider = GroqProvider(settings)
    result = await provider.summarize_company_report(report)

    for category, sections in result.ordered_sections():
        print(f"--- {category.display_name_tr} ({len(sections)} section(s)) ---")
        for section in sections:
            print(f"  [{section.subheading}]")
            print(f"  {section.narrative}")
            print(f"  kaynaklar: {[str(u) for u in section.source_urls]}")
            print()

    assert not result.is_empty(), "expected at least one synthesized section from real THYAO data"
    print("[OK] GroqProvider produced a valid SynthesizedCompanyReport from real THYAO data\n")


async def test_groq_provider_failure_and_fallback() -> None:
    settings = get_settings()
    report = await _real_thyao_company_report()

    broken_provider = GroqProvider(settings, max_tokens=50)  # deliberately too low to finish

    caught: Exception | None = None
    try:
        await broken_provider.summarize_company_report(report)
    except RuntimeError as exc:
        caught = exc

    assert caught is not None, "expected max_tokens=50 to trigger a RuntimeError (finish_reason != 'stop')"
    print(f"[OK] GroqProvider(max_tokens=50) raised RuntimeError as expected: {caught}\n")

    # Replicates main.py's _build_company_report() fallback path exactly.
    digest_run = DigestRun()
    try:
        result = await broken_provider.summarize_company_report(report)
    except RuntimeError as exc:
        digest_run.add_error("llm", report.ticker, f"LLM sentezi başarısız oldu ({exc}), ham liste gösteriliyor")
        result = await NoopProvider().summarize_company_report(report)

    assert not result.is_empty()
    print(f"digest_run.errors: {digest_run.errors}")
    print("[OK] main.py-style fallback: NoopProvider filled in after GroqProvider failed, error recorded on digest_run\n")


async def main() -> None:
    await test_groq_provider_success()
    await test_groq_provider_failure_and_fallback()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
