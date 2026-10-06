"""
Inline-CSS color/font constants for the HTML email template
(templates/email_base.html). Single source of truth — the template imports
these via a Jinja2 `colors` context variable rather than hardcoding hex
values, so a palette change only touches this file.

Palette and category-color mapping match the KapMail reference template
(email_template.py) the user supplied — a warm cream/sepia "editorial" look
rather than the previous slate/blue SaaS-dashboard look.
"""

from __future__ import annotations

from config.constants import NewsCategory

C = {
    "cream": "#FAF8F5",
    "parchment": "#F0EBE3",
    "border": "#D8CFC4",
    "sand": "#E8E0D6",
    "ink": "#1C1410",
    "sepia": "#5C4A3A",
    "cognac": "#8B6248",
    "bottle": "#2D5A40",
    "navy": "#1A2B4A",
    "amber": "#8B5E1A",
    "plum": "#5E3A5C",
}

SERIF = "Georgia,'Times New Roman',Times,serif"
SANS = "-apple-system,'Helvetica Neue',Arial,sans-serif"

# Keyed by NewsCategory.value (config/constants.py) so the template can look
# up a color with `colors.CATEGORY_COLORS[category.value]` without an
# if/elif chain per category. Mapping confirmed with the user.
CATEGORY_COLORS = {
    NewsCategory.NEW_BUSINESS.value: C["bottle"],
    NewsCategory.FINANCIALS.value: C["amber"],
    NewsCategory.KAP_MATERIAL.value: C["navy"],
    NewsCategory.ANALYST_IR.value: C["plum"],
    NewsCategory.GENERAL_SECTOR.value: C["sepia"],
}


def template_colors() -> dict:
    """Everything the Jinja2 template needs under one `colors` context var."""
    return {
        "C": C,
        "SERIF": SERIF,
        "SANS": SANS,
        "CATEGORY_COLORS": CATEGORY_COLORS,
    }
