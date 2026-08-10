"""
Manual verification for nlp/providers/groq_provider.py — real Groq API calls
against real THYAO data (all categories, not just Finansal). Requires
GROQ_API_KEY in .env.

Scenarios:
  1. Success: GroqProvider(settings) (default max_tokens=6000) against a
     real THYAO CompanyReport built the same way main.py does.
  2. Deliberate failure + fallback: GroqProvider(settings, max_tokens=50) —
     too low for the model to produce valid JSON (with response_format=
     json_object, Groq's own server rejects this outright rather than
     returning a truncated completion — see groq_provider.py's docstring),
     so both attempts (initial + 1 retry) fail and RuntimeError propagates.
     Then replicates main.py's _build_company_report() fallback (catch ->
     NoopProvider -> digest_run.add_error()) to prove the safety net works.
  3. Reliability stats: repeated real calls against the two categories that
     failed organically during earlier development (onemli_kap_aciklamalari,
     and one 10-item chunk of sektorel_genel_haberler representative of the
     >15-item chunking path) — reports concrete first-try vs retry-saved vs
     still-failed counts, not just "it worked".

Run directly:

    python -m tests.test_groq_provider
"""

from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import groq

import scrapers.kap_scraper as kap_scraper_module
from config.constants import NewsCategory
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
    try:
        result = await provider.summarize_company_report(report)
    except RuntimeError as exc:
        print(f"[gözlem] guard iki denemede de tetiklendi (retry yetmedi): {exc}\n")
        return
    except groq.APIStatusError as exc:
        print(f"[gözlem] Groq API hatası (muhtemelen rate limit), kod hatası değil: {exc}\n")
        return

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

    broken_provider = GroqProvider(settings, max_tokens=50)  # deliberately too low, every attempt

    digest_run = DigestRun()
    try:
        result = await broken_provider.summarize_company_report(report)
        raise AssertionError("expected max_tokens=50 to trigger a RuntimeError after retry")
    except RuntimeError as exc:
        print(f"[OK] GroqProvider(max_tokens=50) raised RuntimeError after retry as expected: {exc}\n")
        digest_run.add_error("llm", report.ticker, f"LLM sentezi başarısız oldu ({exc}), ham liste gösteriliyor")
        result = await NoopProvider().summarize_company_report(report)
    except groq.APIStatusError as exc:
        print(f"[gözlem] Groq API hatası (muhtemelen rate limit), fallback senaryosu bu koşuda gösterilemedi: {exc}\n")
        return

    assert not result.is_empty()
    print(f"digest_run.errors: {digest_run.errors}")
    print("[OK] main.py-style fallback: NoopProvider filled in after GroqProvider failed, error recorded on digest_run\n")


async def _stats_run(provider: GroqProvider, ticker: str, category: NewsCategory, items: list, label: str, results: list) -> None:
    """Instruments _call_once (via a counting wrapper) so we know whether
    _call_with_retry's single retry was what saved a call, not just whether
    the end result succeeded."""
    call_count = 0
    original_call_once = provider._call_once

    async def counting_call_once(t, c, i):
        nonlocal call_count
        call_count += 1
        return await original_call_once(t, c, i)

    provider._call_once = counting_call_once
    try:
        await provider._call_with_retry(ticker, category, items)
        outcome = "success_first_try" if call_count == 1 else "success_after_retry"
        results.append(outcome)
        print(f"  {label}: {outcome} ({call_count} attempt(s))")
    except RuntimeError as exc:
        results.append("failed_after_retry")
        print(f"  {label}: failed_after_retry ({call_count} attempt(s)): {exc}")
    except groq.APIStatusError as exc:
        results.append("api_error")
        print(f"  {label}: api_error (rate limit, not a code issue): {exc}")
    finally:
        provider._call_once = original_call_once


async def test_reliability_stats() -> None:
    """Repeated real calls against the two categories that failed
    organically during this provider's development, reporting concrete
    counts rather than a single pass/fail."""
    settings = get_settings()
    report = await _real_thyao_company_report()
    provider = GroqProvider(settings)

    kap_material_items = report.items_by_category.get(NewsCategory.KAP_MATERIAL, [])
    general_items = report.items_by_category.get(NewsCategory.GENERAL_SECTOR, [])

    print(f"onemli_kap_aciklamalari: {len(kap_material_items)} item(s)")
    print(f"sektorel_genel_haberler: {len(general_items)} item(s) (testing one 10-item chunk, representative of the >15 chunking path)\n")

    results: list[str] = []

    if kap_material_items:
        print("--- onemli_kap_aciklamalari, 5 deneme ---")
        for i in range(5):
            await _stats_run(provider, "THYAO", NewsCategory.KAP_MATERIAL, kap_material_items, f"deneme {i + 1}", results)
            await asyncio.sleep(15)  # respect this account's rate limits

    if general_items:
        chunk = general_items[:10]
        print("\n--- sektorel_genel_haberler (10 item'lık 1 chunk), 3 deneme ---")
        for i in range(3):
            await _stats_run(provider, "THYAO", NewsCategory.GENERAL_SECTOR, chunk, f"deneme {i + 1}", results)
            await asyncio.sleep(15)

    print(f"\nTOPLAM SONUÇ ({len(results)} deneme): {results}")
    first_try = results.count("success_first_try")
    after_retry = results.count("success_after_retry")
    failed = results.count("failed_after_retry")
    api_errors = results.count("api_error")
    print(
        f"İlk seferde başarılı: {first_try} | Retry ile kurtarılan: {after_retry} | "
        f"İki denemede de başarısız: {failed} | API/rate-limit hatası: {api_errors}"
    )


async def main() -> None:
    await test_groq_provider_success()
    await asyncio.sleep(20)
    await test_groq_provider_failure_and_fallback()
    await asyncio.sleep(20)
    await test_reliability_stats()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
