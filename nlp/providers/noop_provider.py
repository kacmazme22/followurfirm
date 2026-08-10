"""
The default provider (AI_PROVIDER=noop): no LLM call. Converts CompanyReport
-> SynthesizedCompanyReport by schema alone — each NewsItem becomes its own
SynthesizedSection (subheading=title, narrative=body_snippet or title,
source_urls=[url, related_kap_url if any]). No merging, no rewriting: this
is what GroqProvider falls back to when the LLM call fails (see
GroqProvider's docstring and main.py's fallback wiring), so the digest still
renders — just as a plain item list instead of synthesized prose.
"""

from __future__ import annotations

from nlp.providers.base import SummarizerProvider
from scrapers.models import CompanyReport, SynthesizedCompanyReport, SynthesizedSection

# SynthesizedSection.subheading has max_length=100 (scrapers/models.py) but
# NewsItem.title has no such cap — real headlines (e.g. multi-ticker broker
# bulletins) regularly exceed it, so it's truncated here rather than letting
# pydantic validation fail on a perfectly normal title.
_SUBHEADING_MAX_LENGTH = 100


class NoopProvider(SummarizerProvider):
    async def summarize_company_report(self, report: CompanyReport) -> SynthesizedCompanyReport:
        sections_by_category = {
            category: [self._to_section(item) for item in items]
            for category, items in report.ordered_categories()
        }
        return SynthesizedCompanyReport(
            ticker=report.ticker,
            company_name=report.company_name,
            sections_by_category=sections_by_category,
        )

    @staticmethod
    def _to_section(item) -> SynthesizedSection:
        source_urls = []
        if item.url:
            source_urls.append(item.url)
        if item.related_kap_url:
            source_urls.append(item.related_kap_url)

        subheading = item.title
        if len(subheading) > _SUBHEADING_MAX_LENGTH:
            subheading = subheading[: _SUBHEADING_MAX_LENGTH - 1].rstrip() + "…"

        return SynthesizedSection(
            subheading=subheading,
            narrative=item.body_snippet or item.title,
            source_urls=source_urls,
        )
