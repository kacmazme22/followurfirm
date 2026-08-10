"""
Provider-agnostic interface for Tier-2 LLM synthesis. A provider turns a
CompanyReport's raw, deduped NewsItems into a SynthesizedCompanyReport —
dynamic LLM-authored subheadings + narrative prose grouped under the fixed
NewsCategory buckets. Category/dedup decisions are already settled by
nlp/categorizer.py and nlp/dedup.py before a CompanyReport ever reaches a
provider; a provider only decides how to narrate what's already there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scrapers.models import CompanyReport, SynthesizedCompanyReport


class SummarizerProvider(ABC):
    """Implemented by each AI provider (Groq, noop, ...) selected via
    `AI_PROVIDER` (config/settings.py) and instantiated by
    nlp/providers/factory.py."""

    @abstractmethod
    async def summarize_company_report(self, report: CompanyReport) -> SynthesizedCompanyReport:
        """Converts a CompanyReport (raw NewsItems, already categorized and
        deduped) into a SynthesizedCompanyReport (LLM-authored subheadings +
        narrative). Implementations may raise RuntimeError on failure —
        callers (main.py) are expected to catch it and fall back to
        NoopProvider for that ticker rather than have the pipeline crash."""
