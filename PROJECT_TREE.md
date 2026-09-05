# FollowUrFirm — Folder Structure

```
followurfirm/
├── config/
│   ├── __init__.py
│   ├── settings.py          # pydantic-settings: env-driven, typed config object
│   ├── config.yaml          # human-edited: tracked tickers, recipients, source toggles
│   └── constants.py         # KAP disclosure-type keyword maps, category enums
│
├── scrapers/
│   ├── __init__.py
│   ├── base.py               # BaseScraper ABC: fetch(), parse(), politeness helpers
│   ├── kap_scraper.py         # KAP disclosure listing scraper (undocumented endpoint)
│   ├── bigpara_scraper.py     # Bigpara ticker news scraper
│   ├── google_news_scraper.py # Google News RSS per-ticker (feedparser based)
│   └── models.py              # Pydantic: RawScrapedItem, NewsItem, KapDisclosure, CompanyReport
│
├── nlp/
│   ├── __init__.py
│   ├── categorizer.py        # Rule-based keyword/regex categorization (Tier 1, no LLM)
│   ├── dedup.py               # Fuzzy title matching + URL hashing for duplicate filtering
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py            # SummarizerProvider interface (abstract)
│   │   ├── groq_provider.py   # Groq free-tier implementation
│   │   ├── noop_provider.py   # Fallback: truncation-based "summarizer", zero LLM calls
│   │   └── factory.py         # get_provider() -> reads env var, returns concrete instance
│   └── prompts.py             # Turkish summarization prompt templates
│
├── templates/
│   ├── email_base.html        # Jinja2 master layout (KapMail-style HTML email)
│   ├── partials/
│   │   ├── ticker_card.html   # Per-company section card
│   │   ├── category_badge.html
│   │   └── footer.html
│   └── styles.py              # Inline-CSS constants (email-safe, no external stylesheet)
│
├── utils/
│   ├── __init__.py
│   ├── logging_setup.py       # Structured logging, graceful-degradation friendly
│   ├── email_sender.py        # smtplib wrapper, Gmail App Password auth
│   ├── cache.py                # In-run response cache (avoid duplicate hits per run)
│   └── rate_limit.py           # Randomized delay / politeness helper
│
├── tests/
│   └── (unit tests mirroring package layout)
│
├── .github/
│   └── workflows/
│       └── daily_digest.yml    # Cron-triggered GitHub Actions workflow
│
├── main.py                     # Pipeline orchestrator (entrypoint)
├── requirements.txt
├── .env.example
└── README.md
```

## Design rationale

- **`config/settings.py` vs `config/config.yaml`**: secrets and env-dependent
  toggles (API keys, SMTP creds, which LLM provider is active) live in
  environment variables loaded via `pydantic-settings`. Ticker lists, email
  recipients, and per-source enable/disable flags — things a non-developer
  might tweak — live in `config.yaml`. `settings.py` merges both into one
  typed `Settings` object so the rest of the codebase never touches raw env
  vars or raw YAML directly.

- **`nlp/providers/`**: isolated behind a `SummarizerProvider` ABC so that
  swapping Groq for another free-tier provider (or turning off LLM synthesis
  entirely and relying purely on rule-based categorization) is a one-line env
  var change (`AI_PROVIDER=groq|noop|<future>`), never a code change in
  `categorizer.py` or `main.py`.

- **`scrapers/models.py` centralizes Pydantic schemas** rather than scattering
  them per-scraper, because `NewsItem` is a shared contract that every
  scraper must emit into, and `dedup.py` / `categorizer.py` both operate on
  that shared shape regardless of source.
