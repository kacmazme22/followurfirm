"""
Sends the rendered digest HTML via SMTP (Gmail App Password by default,
per .env.example). `DRY_RUN=true` is a hard short-circuit before any network
connection — the safety valve that lets main.py be run repeatedly while
testing without ever actually emailing anyone.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config.settings import AppConfig

logger = logging.getLogger(__name__)


def build_digest_subject(settings: AppConfig, today: date | None = None) -> str:
    """Fills settings.yaml.email.subject_template's {date} placeholder."""
    today = today or date.today()
    return settings.yaml.email.subject_template.format(date=today.strftime("%d.%m.%Y"))


def send_digest_email(
    html_content: str,
    subject: str,
    recipients: list[str],
    settings: AppConfig,
    text_content: str | None = None,
) -> None:
    """Sends `html_content` as an HTML email. Raises RuntimeError (chained
    from the underlying smtplib/socket exception) on any connection or auth
    failure — never swallowed here, since whether that's worth a retry or
    just a log entry is main.py's call, not this function's."""
    if settings.env.dry_run:
        logger.info("DRY RUN: e-posta gönderilecekti, alıcılar: %s | konu: %r", recipients, subject)
        return

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"{settings.yaml.email.from_display_name} <{settings.env.smtp_username}>"
    message["To"] = ", ".join(recipients)
    # multipart/alternative: clients show the LAST part they can render, so
    # plain text goes first and HTML wins wherever HTML is supported.
    if text_content:
        message.attach(MIMEText(text_content, "plain", "utf-8"))
    message.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(settings.env.smtp_host, settings.env.smtp_port, timeout=30) as server:
            server.starttls()
            server.login(settings.env.smtp_username, settings.env.smtp_password)
            server.sendmail(settings.env.smtp_username, recipients, message.as_string())
    except Exception as exc:
        logger.error(
            "SMTP gönderimi başarısız (%s:%s): %s",
            settings.env.smtp_host, settings.env.smtp_port, exc,
            exc_info=True,
        )
        raise RuntimeError(
            f"E-posta gönderilemedi ({settings.env.smtp_host}:{settings.env.smtp_port}): {exc}"
        ) from exc

    logger.info("E-posta gönderildi: alıcılar=%s konu=%r", recipients, subject)
