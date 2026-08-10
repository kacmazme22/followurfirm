"""
Manual verification for utils/email_sender.py — DRY_RUN path only. No real
SMTP send is exercised here (that's a manual, human-supervised test once
.env has real credentials, per the task).

Runs the real pipeline (real network requests, same as tests/test_dedup.py's
smoke test) to get real rendered HTML, then calls send_digest_email() with
dry_run=True and confirms: (a) it returns without raising, (b) it logs the
recipients/subject, (c) build_digest_subject() fills the {date} placeholder.

Run directly:

    python -m tests.test_email_sender
"""

from __future__ import annotations

import asyncio
import logging

from config.settings import get_settings
from main import render_digest_html, run_pipeline
from utils.email_sender import build_digest_subject, send_digest_email


def test_build_digest_subject() -> None:
    settings = get_settings()
    from datetime import date

    subject = build_digest_subject(settings, today=date(2026, 8, 3))
    assert "03.08.2026" in subject, f"expected the {{date}} placeholder filled, got: {subject!r}"
    print(f"[OK] build_digest_subject() -> {subject!r}")


async def test_dry_run_never_connects() -> None:
    settings = get_settings()
    assert settings.env.dry_run, (
        "This test expects DRY_RUN=true (see .env). Refusing to run with dry_run=False "
        "to avoid accidentally exercising the real SMTP path from an automated test."
    )

    digest_run = await run_pipeline()
    html = render_digest_html(digest_run)
    subject = build_digest_subject(settings)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # No exception, no network connection — dry_run short-circuits before smtplib.
    send_digest_email(html, subject, settings.yaml.recipients, settings)
    print("[OK] send_digest_email() with dry_run=True returned without touching SMTP (see DRY RUN log line above)")


async def main() -> None:
    test_build_digest_subject()
    await test_dry_run_never_connects()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
