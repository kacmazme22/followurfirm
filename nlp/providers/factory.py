"""
Single entrypoint for selecting a SummarizerProvider based on
settings.env.ai_provider (config/settings.py). Nothing else in the codebase
should instantiate a provider directly.
"""

from __future__ import annotations

from config.settings import AppConfig
from nlp.providers.base import SummarizerProvider
from nlp.providers.groq_provider import GroqProvider
from nlp.providers.noop_provider import NoopProvider


def get_provider(settings: AppConfig) -> SummarizerProvider:
    if settings.env.ai_provider == "noop":
        return NoopProvider()
    if settings.env.ai_provider == "groq":
        return GroqProvider(settings)
    raise NotImplementedError(f"Unknown AI_PROVIDER: {settings.env.ai_provider!r}")
