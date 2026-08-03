"""
Provider-agnostic interface for Tier-2 LLM synthesis. A provider only polishes
text (summarizing/cleaning up body_snippet content) — it never touches
categorization or dedup decisions, those are settled by nlp/categorizer.py
and nlp/dedup.py before a CompanyReport ever reaches a provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scrapers.models import CompanyReport


class SummarizerProvider(ABC):
    """Implemented by each AI provider (Groq, noop, ...) selected via
    `AI_PROVIDER` (config/settings.py) and instantiated by
    nlp/providers/factory.py."""

    @abstractmethod
    async def summarize_company_report(self, report: CompanyReport) -> CompanyReport:
        """When needed, uses an LLM to shorten/clean up items' body_snippet
        text. Does NOT change category or dedup decisions — text polish
        only."""
