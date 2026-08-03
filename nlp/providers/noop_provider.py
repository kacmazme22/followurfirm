"""
The default provider (AI_PROVIDER=noop): a genuine no-op, not a stub. No LLM
call, no string manipulation — nlp/categorizer.py has already populated
body_snippet correctly, so there's nothing here to do.
"""

from __future__ import annotations

from nlp.providers.base import SummarizerProvider
from scrapers.models import CompanyReport


class NoopProvider(SummarizerProvider):
    async def summarize_company_report(self, report: CompanyReport) -> CompanyReport:
        return report
