"""
Central settings object for FollowUrFirm.

Design goal: nothing else in the codebase should call `os.getenv(...)` or
`yaml.safe_load(...)` directly. Every module imports `get_settings()` and
receives one fully-typed, validated `Settings` instance. This means:

  - Swapping the AI provider (Groq -> something else) is a single env var
    change (`AI_PROVIDER`), never a code change.
  - Adding a new source toggle or ticker only touches config.yaml.
  - Secrets never leak into version control (they live in .env, gitignored).

`Settings` is a lru_cache'd singleton via `get_settings()` so we parse
config.yaml exactly once per process, but tests can call
`get_settings.cache_clear()` to force a reload with monkeypatched env vars.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, EmailStr, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_YAML_PATH = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Sub-models mirroring config.yaml structure
# ---------------------------------------------------------------------------

class TickerConfig(BaseModel):
    symbol: str
    name: str
    enabled: bool = True


class PolitenessConfig(BaseModel):
    min_delay_seconds: float = 1.5
    max_delay_seconds: float = 4.0
    user_agent: str = "Mozilla/5.0 (compatible; FollowUrFirmBot/1.0)"


class KapSourceConfig(BaseModel):
    enabled: bool = True
    base_url: str
    disclosure_endpoint: str
    max_items_per_ticker: int = 10
    politeness: PolitenessConfig = Field(default_factory=PolitenessConfig)


class BigparaSourceConfig(BaseModel):
    enabled: bool = True
    base_url: str
    politeness: PolitenessConfig = Field(default_factory=PolitenessConfig)


class GoogleNewsRssConfig(BaseModel):
    enabled: bool = True
    query_template_tr: str = "{company_name} {ticker} hisse"
    language: str = "tr"
    country: str = "TR"


class CompanyIrPagesConfig(BaseModel):
    enabled: bool = False
    urls: dict[str, str] = Field(default_factory=dict)


class WebSearchFallbackConfig(BaseModel):
    enabled: bool = True
    trigger_if_items_below: int = 2


class GenericPhase2SourceConfig(BaseModel):
    enabled: bool = False
    base_url: str | None = None


class SourcesConfig(BaseModel):
    kap: KapSourceConfig
    bigpara: BigparaSourceConfig
    google_news_rss: GoogleNewsRssConfig = Field(default_factory=GoogleNewsRssConfig)
    company_ir_pages: CompanyIrPagesConfig = Field(default_factory=CompanyIrPagesConfig)
    web_search_fallback: WebSearchFallbackConfig = Field(default_factory=WebSearchFallbackConfig)
    sec_edgar: GenericPhase2SourceConfig = Field(default_factory=GenericPhase2SourceConfig)
    yahoo_finance: GenericPhase2SourceConfig = Field(default_factory=GenericPhase2SourceConfig)
    finviz: GenericPhase2SourceConfig = Field(default_factory=GenericPhase2SourceConfig)
    pr_newswire: GenericPhase2SourceConfig = Field(default_factory=GenericPhase2SourceConfig)


class LlmTriggerConfig(BaseModel):
    min_items_to_justify_call: int = 3


class AiSynthesisConfig(BaseModel):
    rule_based_categorization: bool = True
    llm_synthesis_enabled: bool = True
    llm_trigger: LlmTriggerConfig = Field(default_factory=LlmTriggerConfig)


class EmailContentConfig(BaseModel):
    subject_template: str = "FollowUrFirm Günlük Bülten — {date}"
    from_display_name: str = "FollowUrFirm Digest"


class ScheduleConfig(BaseModel):
    cron_utc: str = "0 5 * * *"
    timezone_display: str = "Europe/Istanbul"


class TickersConfig(BaseModel):
    bist: list[TickerConfig] = Field(default_factory=list)
    global_: list[TickerConfig] = Field(default_factory=list, alias="global")

    model_config = {"populate_by_name": True}


class RawYamlConfig(BaseModel):
    """Mirrors config.yaml 1:1 before merging with env-derived Settings."""

    market_phase: Literal["bist", "global", "both"] = "bist"
    tickers: TickersConfig
    recipients: list[EmailStr]
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    sources: SourcesConfig
    ai_synthesis: AiSynthesisConfig = Field(default_factory=AiSynthesisConfig)
    email: EmailContentConfig = Field(default_factory=EmailContentConfig)
    cache: dict = Field(default_factory=lambda: {"enabled": True})
    logging: dict = Field(default_factory=lambda: {"level": "INFO", "json_format": False})


# ---------------------------------------------------------------------------
# Env-driven settings (secrets + provider selection)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Env-backed settings. Values here either ARE secrets, or control behavior
    that should be flippable without editing YAML (e.g. in a GitHub Actions
    `env:` block via repo secrets).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- AI provider selection ---
    # "groq" | "noop" | any future free-tier provider key registered in
    # nlp/providers/factory.py. Never set this to a metered provider.
    ai_provider: Literal["groq", "noop"] = "noop"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"

    # --- SMTP / email delivery ---
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None  # Gmail App Password, NOT the account password

    # --- Misc / runtime ---
    environment: Literal["local", "ci"] = "local"
    dry_run: bool = False  # if True, render email but do not send

    @field_validator("groq_api_key")
    @classmethod
    def _warn_if_groq_selected_without_key(cls, v, info):
        # Validation-time cross-field checks are limited in pydantic v2 without
        # a model_validator; the authoritative check happens in
        # get_settings() below where both fields are available together.
        return v


class AppConfig(BaseModel):
    """
    The single object the rest of the app imports. Combines the typed YAML
    config with env-derived Settings, and exposes a few convenience
    properties so callers don't need to know which half of the merge a given
    value came from.
    """

    yaml: RawYamlConfig
    env: Settings

    @property
    def active_bist_tickers(self) -> list[TickerConfig]:
        return [t for t in self.yaml.tickers.bist if t.enabled]

    @property
    def active_global_tickers(self) -> list[TickerConfig]:
        return [t for t in self.yaml.tickers.global_ if t.enabled]

    @property
    def all_active_tickers(self) -> list[TickerConfig]:
        return self.active_bist_tickers + self.active_global_tickers

    def validate_provider_credentials(self) -> None:
        """
        Call this once at pipeline startup. Raises early and loudly if the
        selected AI provider is missing required credentials, rather than
        failing deep inside nlp/providers/groq_provider.py mid-run.
        """
        if self.env.ai_provider == "groq" and not self.env.groq_api_key:
            raise RuntimeError(
                "AI_PROVIDER=groq but GROQ_API_KEY is not set. "
                "Set it in .env (local) or as a GitHub Actions secret (CI), "
                "or switch AI_PROVIDER=noop to disable LLM synthesis entirely."
            )
        if self.env.smtp_username and not self.env.smtp_password:
            raise RuntimeError(
                "SMTP_USERNAME is set but SMTP_PASSWORD is missing. "
                "For Gmail, generate an App Password (not your account password): "
                "https://myaccount.google.com/apppasswords"
            )


@lru_cache
def get_settings(config_path: Path | str = CONFIG_YAML_PATH) -> AppConfig:
    """
    Single entrypoint for the entire app. Cached so config.yaml is parsed
    once per process. Tests / scripts that need to reload with different
    env vars should call `get_settings.cache_clear()` first.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Copy config/config.yaml.example or restore it before running the pipeline."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    yaml_config = RawYamlConfig.model_validate(raw)
    env_settings = Settings()  # reads from process env + .env file

    app_config = AppConfig(yaml=yaml_config, env=env_settings)
    app_config.validate_provider_credentials()
    return app_config


if __name__ == "__main__":
    # Quick sanity check: `python -m config.settings`
    cfg = get_settings()
    print(f"Market phase: {cfg.yaml.market_phase}")
    print(f"Active BIST tickers: {[t.symbol for t in cfg.active_bist_tickers]}")
    print(f"AI provider: {cfg.env.ai_provider}")
    print(f"Recipients: {cfg.yaml.recipients}")
