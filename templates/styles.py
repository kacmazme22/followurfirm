"""
Inline-CSS color/font constants for the HTML email template
(templates/email_base.html). Single source of truth — the template imports
these via a Jinja2 `colors` context variable rather than hardcoding hex
values, so a palette change only touches this file.

Colors are per the project's KapMail-style palette, except BADGE_GENERAL
(General/Sector category) which isn't specified there — a neutral slate tone
was chosen to sit visually behind the three "meaningful" category colors.
"""

from __future__ import annotations

PRIMARY_DARK = "#0F172A"
BORDER_GREY = "#E2E8F0"
TEXT_GREY = "#64748B"

BADGE_NEW_BUSINESS = {"text": "#059669", "bg": "#ECFDF5"}  # green — new business/tenders
BADGE_KAP = {"text": "#2563EB", "bg": "#EFF6FF"}  # blue — material KAP disclosures
BADGE_FINANCIAL = {"text": "#D97706", "bg": "#FEF3C7"}  # amber — financials/earnings
BADGE_GENERAL = {"text": "#475569", "bg": "#F1F5F9"}  # neutral slate — general/sector news

FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# Keyed by NewsCategory.value (config/constants.py) so the template can look
# up a badge with `category_badges[item.category.value]` without an if/elif
# chain per category.
CATEGORY_BADGES = {
    "yeni_is_iliskileri": BADGE_NEW_BUSINESS,
    "finansal_sonuclar": BADGE_FINANCIAL,
    "onemli_kap_aciklamalari": BADGE_KAP,
    "sektorel_genel_haberler": BADGE_GENERAL,
}


def template_colors() -> dict:
    """Everything the Jinja2 template needs under one `colors` context var."""
    return {
        "PRIMARY_DARK": PRIMARY_DARK,
        "BORDER_GREY": BORDER_GREY,
        "TEXT_GREY": TEXT_GREY,
        "BADGE_NEW_BUSINESS": BADGE_NEW_BUSINESS,
        "BADGE_KAP": BADGE_KAP,
        "BADGE_FINANCIAL": BADGE_FINANCIAL,
        "BADGE_GENERAL": BADGE_GENERAL,
        "FONT_STACK": FONT_STACK,
        "CATEGORY_BADGES": CATEGORY_BADGES,
    }
