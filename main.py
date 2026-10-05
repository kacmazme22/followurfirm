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
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from config.constants import SourceType
from config.settings import AppConfig, PolitenessConfig, TickerConfig, get_settings
from nlp.categorizer import categorize_batch
from nlp.dedup import deduplicate
from nlp.providers.factory import get_provider
from nlp.providers.noop_provider import NoopProvider
from scrapers.bigpara_scraper import BigparaScraper
from scrapers.google_news_scraper import GoogleNewsScraper
from scrapers.kap_scraper import KapScraper
from scrapers.models import CompanyReport, DigestRun, RawScrapedItem, SynthesizedCompanyReport
from templates.styles import template_colors
from utils.email_sender import build_digest_subject, send_digest_email

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR = Path(__file__).parent / "output"

# GoogleNewsRssConfig (config/settings.py) carries no politeness sub-config
# of its own (unlike kap/bigpara), so it uses the library default here.
_GOOGLE_NEWS_POLITENESS = PolitenessConfig()

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
    scraper = KapScraper(_KAP_BULK_PLACEHOLDER_TICKER, settings.yaml.sources.kap)
    try:
        return await scraper.fetch_all_raw(tickers)
    except Exception as exc:
        logger.error("KAP bulk fetch failed: %s", exc, exc_info=True)
        digest_run.add_error(SourceType.KAP, "ALL", str(exc))
        return {}


async def _build_company_report(
    ticker: TickerConfig,
    kap_raw_by_ticker: dict[str, list[RawScrapedItem]],
    settings: AppConfig,
    digest_run: DigestRun,
) -> SynthesizedCompanyReport:
    raw_items: list[RawScrapedItem] = list(kap_raw_by_ticker.get(ticker.symbol, []))

    bigpara_result = await BigparaScraper(ticker, settings.yaml.sources.bigpara).run()
    raw_items += bigpara_result.items
    if bigpara_result.error:
        digest_run.add_error(SourceType.BIGPARA, ticker.symbol, bigpara_result.error)

    google_result = await GoogleNewsScraper(
        ticker, settings.yaml.sources.google_news_rss, _GOOGLE_NEWS_POLITENESS
    ).run()
    raw_items += google_result.items
    if google_result.error:
        digest_run.add_error(SourceType.GOOGLE_NEWS, ticker.symbol, google_result.error)

    news_items = categorize_batch(raw_items)  # boilerplate items already dropped (None -> skipped)
    news_items = deduplicate(news_items)

    report = CompanyReport(ticker=ticker.symbol, company_name=ticker.name)
    for item in news_items:
        report.add_item(item)
    logger.info(
        "%s: %d haber (%s) özetlemeye gidiyor",
        ticker.symbol, report.total_items,
        ", ".join(f"{cat.value}={len(items)}" for cat, items in report.ordered_categories()) or "boş",
    )

    provider = get_provider(settings)
    try:
        return await provider.summarize_company_report(report)
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


async def run_pipeline() -> DigestRun:
    settings = get_settings()
    digest_run = DigestRun()

    kap_raw_by_ticker = await fetch_kap_once_and_group(settings.active_bist_tickers, settings, digest_run)

    for ticker in settings.active_bist_tickers:
        try:
            report = await _build_company_report(ticker, kap_raw_by_ticker, settings, digest_run)
            digest_run.company_reports.append(report)
        except Exception as exc:  # one ticker blowing up must not sink the rest
            logger.error("Ticker %s failed entirely: %s", ticker.symbol, exc, exc_info=True)
            digest_run.add_error("pipeline", ticker.symbol, str(exc))

    return digest_run


def render_digest_text(digest_run: DigestRun) -> str:
    """Plain-text twin of the HTML digest. Sent as the email's text/plain
    alternative (HTML-only mail scores worse with spam filters) and printed
    in the workflow log, so a run's content can be read straight from the
    Actions page without downloading the artifact."""
    lines = [f"FollowUrFirm Digest — {digest_run.run_date.strftime('%d.%m.%Y %H:%M')} UTC", ""]
    for report in digest_run.company_reports:
        lines.append(f"== {report.ticker} — {report.company_name} ==")
        if report.is_empty():
            lines += ["Bugün yeni bir gelişme yok.", ""]
            continue
        for category, sections in report.ordered_sections():
            lines.append(f"[{category.display_name_tr}]")
            for section in sections:
                lines.append(f"* {section.subheading}")
                if section.narrative != section.subheading:
                    lines.append(f"  {section.narrative}")
                lines += [f"  -> {url}" for url in section.source_urls]
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

    result = asyncio.run(run_pipeline())
    rendered_html = render_digest_html(result)
    rendered_text = render_digest_text(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"digest_{date.today().isoformat()}.html"
    output_path.write_text(rendered_html, encoding="utf-8")
    output_path.with_suffix(".txt").write_text(rendered_text, encoding="utf-8")
    print(f"Digest written to {output_path}")

    subject = build_digest_subject(settings)
    send_digest_email(rendered_html, subject, settings.yaml.recipients, settings, text_content=rendered_text)
