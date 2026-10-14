"""
Pipeline orchestrator: scrape -> categorize -> dedup -> (noop-by-default) LLM
polish -> render HTML -> email. Running this module writes the rendered
digest to output/digest_{date}.html AND calls send_digest_email() — which is
itself a no-op against the real network as long as DRY_RUN=true (see
utils/email_sender.py and README.md).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date
from pathlib import Path

import httpx
from jinja2 import Environment, FileSystemLoader

from config.constants import SourceType
from config.settings import AppConfig, PolitenessConfig, TickerConfig, get_settings
from nlp.categorizer import categorize_batch
from nlp.dedup import deduplicate
from nlp.relevance import tr_lower
from nlp.market_brief import build_market_brief
from nlp.providers.factory import get_provider
from nlp.providers.noop_provider import NoopProvider
from scrapers.bigpara_scraper import BigparaScraper
from scrapers.bing_news_scraper import BingNewsScraper
from scrapers.google_news_scraper import GoogleNewsScraper
from scrapers.ir_scraper import IrPageScraper
from scrapers.kap_scraper import KapScraper
from scrapers.models import CompanyReport, DigestRun, RawScrapedItem, SynthesizedCompanyReport
from templates.styles import template_colors
from utils.audit import log_inputs, log_outputs
from utils.formatting import polish_report, turkish_numbers
from utils.email_sender import build_digest_subject, send_digest_email
from utils.memory import DigestMemory, load_memory

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR = Path(__file__).parent / "output"

# GoogleNewsRssConfig (config/settings.py) carries no politeness sub-config
# of its own (unlike kap/bigpara), so it uses the library default here.
_GOOGLE_NEWS_POLITENESS = PolitenessConfig()

# KAP is reached from GitHub's US-hosted runners; httpx's default 5s connect
# timeout and a single attempt left a whole day without KAP items on one
# ConnectTimeout (2026-10-05) while the same request took ~5s on 2026-10-03.
# This one request carries every company's disclosures, so it's worth a
# longer timeout and a couple of retries.
KAP_TIMEOUT = httpx.Timeout(30.0, connect=20.0)
KAP_ATTEMPTS = 3
KAP_RETRY_DELAY_SECONDS = 10

# Only used to construct a KapScraper instance for the bulk fetch below.
# fetch_all_raw() ignores this ticker entirely — it's just a label, not a
# filter (see scrapers/kap_scraper.py).
_KAP_BULK_PLACEHOLDER_TICKER = TickerConfig(symbol="ALL_BIST", name="Tüm BIST şirketleri")


async def fetch_kap_once_and_group(
    tickers: list[TickerConfig], settings: AppConfig, digest_run: DigestRun
) -> dict[str, list[RawScrapedItem]]:
    """One POST to KAP covering every company, grouped client-side by
    ticker — see KapScraper.fetch_all_raw() for why this replaces N
    per-ticker KAP calls. Never raises: a KAP-wide failure is logged and
    recorded on digest_run, and the pipeline continues with no KAP items
    for any ticker rather than stopping entirely."""
    last_exc: Exception | None = None
    for attempt in range(1, KAP_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=KAP_TIMEOUT, follow_redirects=True) as client:
                scraper = KapScraper(_KAP_BULK_PLACEHOLDER_TICKER, settings.yaml.sources.kap, http_client=client)
                return await scraper.fetch_all_raw(tickers)
        except Exception as exc:
            last_exc = exc
            logger.warning("KAP deneme %d/%d başarısız: %s: %s", attempt, KAP_ATTEMPTS, type(exc).__name__, exc)
            if attempt < KAP_ATTEMPTS:
                await asyncio.sleep(KAP_RETRY_DELAY_SECONDS * attempt)

    logger.error("KAP bulk fetch failed", exc_info=last_exc)
    # str(httpx.ConnectTimeout()) is empty, which put a bare "[kap] ALL: "
    # line in the email (2026-10-05); name the exception type instead.
    digest_run.add_error(
        SourceType.KAP, "ALL",
        f"KAP'a {KAP_ATTEMPTS} denemede ulaşılamadı ({type(last_exc).__name__}); bugünkü bültende KAP bildirimleri yok.",
    )
    return {}


def _drop_mirrored_filings(raw_items: list[RawScrapedItem], symbol: str) -> list[RawScrapedItem]:
    """Bigpara's kap-haberleri page is a copy of a KAP filing ("AKBNK AKBANK
    T.A.Ş. (Özel Durum Açıklaması (Genel))"). When KAP's own detail page was
    read, the copy only adds a duplicate item and eats the LLM's text budget."""
    kap_subjects = {tr_lower(i.raw_title) for i in raw_items if i.source == SourceType.KAP and i.raw_body_snippet}
    if not kap_subjects:
        return raw_items
    kept = [
        i for i in raw_items
        if not (i.raw_url and "/kap-haberleri/" in i.raw_url and any(s in tr_lower(i.raw_title) for s in kap_subjects))
    ]
    if len(kept) < len(raw_items):
        logger.info("%s: KAP'tan okunan %d bildirimin Bigpara kopyası elendi", symbol, len(raw_items) - len(kept))
    return kept


async def _build_company_report(
    ticker: TickerConfig,
    kap_raw_by_ticker: dict[str, list[RawScrapedItem]],
    settings: AppConfig,
    digest_run: DigestRun,
    memory: DigestMemory,
) -> SynthesizedCompanyReport:
    raw_items: list[RawScrapedItem] = list(kap_raw_by_ticker.get(ticker.symbol, []))

    bigpara_result = await BigparaScraper(ticker, settings.yaml.sources.bigpara).run()
    raw_items += bigpara_result.items
    if bigpara_result.error:
        digest_run.add_error(SourceType.BIGPARA, ticker.symbol, bigpara_result.error)

    # Before Google: on a duplicate story the first-seen copy is kept, and
    # Bing's copy carries article text where Google's is a bare headline.
    if settings.yaml.sources.bing_news.enabled:
        bing_result = await BingNewsScraper(ticker, settings.yaml.sources.bing_news, _GOOGLE_NEWS_POLITENESS).run()
        raw_items += bing_result.items
        if bing_result.error:
            digest_run.add_error(SourceType.BING_NEWS, ticker.symbol, bing_result.error)

    google_result = await GoogleNewsScraper(
        ticker, settings.yaml.sources.google_news_rss, _GOOGLE_NEWS_POLITENESS
    ).run()
    raw_items += google_result.items
    if google_result.error:
        digest_run.add_error(SourceType.GOOGLE_NEWS, ticker.symbol, google_result.error)

    ir_pages = settings.yaml.sources.company_ir_pages
    if ir_pages.enabled:
        ir_result = await IrPageScraper(
            ticker, memory, _GOOGLE_NEWS_POLITENESS, pinned_pages=ir_pages.urls.get(ticker.symbol)
        ).run()
        raw_items += ir_result.items
        if ir_result.error:
            digest_run.add_error(SourceType.COMPANY_IR, ticker.symbol, ir_result.error)

    raw_items = _drop_mirrored_filings(raw_items, ticker.symbol)
    news_items = categorize_batch(raw_items)  # boilerplate items already dropped (None -> skipped)
    news_items = deduplicate(news_items)
    if settings.yaml.sources.bing_news.enabled:
        await BingNewsScraper(ticker, settings.yaml.sources.bing_news, _GOOGLE_NEWS_POLITENESS).fill_headline_only(news_items)
    news_items = memory.drop_reported(news_items, ticker.symbol)

    report = CompanyReport(ticker=ticker.symbol, company_name=ticker.name)
    for item in news_items:
        report.add_item(item)
    logger.info(
        "%s: %d haber (%s) özetlemeye gidiyor",
        ticker.symbol, report.total_items,
        ", ".join(f"{cat.value}={len(items)}" for cat, items in report.ordered_categories()) or "boş",
    )

    provider = get_provider(settings)
    sent_items = [item for _, items in report.ordered_categories() for item in items]
    memory.note_inputs(ticker.symbol, sent_items)
    log_inputs(ticker.symbol, sent_items)
    try:
        source_text = " ".join(f"{i.title} {i.body_snippet or ''}" for i in sent_items)
        synthesized = polish_report(await provider.summarize_company_report(report), source_text)
        log_outputs(ticker.symbol, [s for _, secs in synthesized.ordered_sections() for s in secs], sent_items)
        return synthesized
    except RuntimeError as exc:
        # LLM synthesis failed (bad/truncated response, invalid JSON, schema
        # mismatch — see GroqProvider's hallucination guards). Fall back to
        # the raw-item rendering instead of losing the ticker's digest
        # section entirely, and surface why in the footer via digest_run.errors.
        logger.error("LLM synthesis failed for %s: %s", ticker.symbol, exc, exc_info=True)
        digest_run.add_error(
            "llm", ticker.symbol, f"LLM sentezi başarısız oldu ({exc}), ham liste gösteriliyor"
        )
        return await NoopProvider().summarize_company_report(report)


class KapUnavailable(Exception):
    """KAP couldn't be reached and this run was told to require it.

    KAP answers some GitHub runner IPs and not others (unreachable on the
    2026-10-06 07:00 run, fine the same evening). The early external
    trigger runs with REQUIRE_KAP=true: rather than mailing a KAP-less
    digest it exits without sending, so the once-per-day marker isn't set
    and the next trigger (on a different runner) gets another chance; the
    last trigger of the morning runs without it and always sends."""


async def run_pipeline(memory: DigestMemory | None = None) -> DigestRun:
    """`memory` is filled in as a side effect: run_pipeline only filters with
    it; the caller records and saves it once the mail has gone out."""
    settings = get_settings()
    digest_run = DigestRun()
    memory = memory if memory is not None else load_memory(enabled=False)

    kap_raw_by_ticker = await fetch_kap_once_and_group(settings.active_bist_tickers, settings, digest_run)
    if not kap_raw_by_ticker and os.environ.get("REQUIRE_KAP", "").lower() == "true":
        raise KapUnavailable()
    digest_run.market_brief = await build_market_brief(settings, get_provider(settings))

    for ticker in settings.active_bist_tickers:
        try:
            report = await _build_company_report(ticker, kap_raw_by_ticker, settings, digest_run, memory)
            digest_run.company_reports.append(report)
        except Exception as exc:  # one ticker blowing up must not sink the rest
            logger.error("Ticker %s failed entirely: %s", ticker.symbol, exc, exc_info=True)
            digest_run.add_error("pipeline", ticker.symbol, str(exc))

    digest_run.headline = await get_provider(settings).pick_headline(_headline_candidates(digest_run))
    if digest_run.headline:
        digest_run.headline = turkish_numbers(digest_run.headline)
    logger.info("Konu başlığı: %s", digest_run.headline)
    return digest_run


def _headline_candidates(digest_run: DigestRun) -> list[str]:
    candidates = []
    for report in digest_run.company_reports:
        for _, sections in report.ordered_sections():
            candidates += [f"{report.ticker} — {s.subheading}: {s.narrative.split(chr(10))[0][:220]}" for s in sections]
    candidates += [f"Piyasa — {s.subheading}: {s.narrative[:200]}" for s in digest_run.market_brief]
    return candidates


def render_digest_text(digest_run: DigestRun) -> str:
    """Plain-text twin of the HTML digest. Sent as the email's text/plain
    alternative (HTML-only mail scores worse with spam filters) and printed
    in the workflow log, so a run's content can be read straight from the
    Actions page without downloading the artifact."""
    lines = [f"FollowUrFirm Digest — {digest_run.run_date.strftime('%d.%m.%Y %H:%M')} UTC", ""]

    def add_section(section) -> None:
        if section.narrative != section.subheading:
            lines.append(f"* {section.subheading}: {section.narrative}".replace("\n\n", "\n  "))
        else:
            lines.append(f"* {section.subheading}")
        lines.extend(f"  -> {url}" for url in section.source_urls)

    if digest_run.market_brief:
        lines.append("== Piyasa Gündemi ==")
        for section in digest_run.market_brief:
            add_section(section)
        lines.append("")

    for report in digest_run.company_reports:
        lines.append(f"== {report.ticker} — {report.company_name} ==")
        if report.is_empty():
            lines += ["Bugün yeni bir gelişme yok.", ""]
            continue
        if report.summary:
            lines.append(f"Kısaca: {report.summary}")
        for _, sections in report.ordered_sections():
            for section in sections:
                add_section(section)
        lines.append("")
    if digest_run.errors:
        lines.append("Uyarılar:")
        lines += [f"- {error}" for error in digest_run.errors]
    return "\n".join(lines)


def render_digest_html(digest_run: DigestRun) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("email_base.html")
    return template.render(digest_run=digest_run, colors=template_colors())


if __name__ == "__main__":
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.yaml.logging.get("level", "INFO"), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    memory = load_memory(enabled=os.environ.get("IGNORE_MEMORY", "").lower() != "true")
    try:
        result = asyncio.run(run_pipeline(memory))
    except KapUnavailable:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "SKIPPED_NO_KAP").write_text("KAP unreachable; left for the next trigger.\n", encoding="utf-8")
        print("KAP'a ulaşılamadı ve REQUIRE_KAP=true: mail gönderilmedi, sonraki tetikleme deneyecek.")
        raise SystemExit(0)
    rendered_html = render_digest_html(result)
    rendered_text = render_digest_text(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"digest_{date.today().isoformat()}.html"
    output_path.write_text(rendered_html, encoding="utf-8")
    output_path.with_suffix(".txt").write_text(rendered_text, encoding="utf-8")
    print(f"Digest written to {output_path}")

    subject = build_digest_subject(settings, headline=result.headline)
    send_digest_email(rendered_html, subject, settings.yaml.recipients, settings, text_content=rendered_text)

    # Written on every run; the workflow only caches it after a real send.
    memory.record(result)
    memory.save()
