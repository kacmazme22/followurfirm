"""
Trial switches for new digest rules, and the record of what they did.

Every rule added after the 2-15 November review runs behind a switch in
config.yaml (`experiments:`): "on" applies it, "shadow" only decides,
"off" skips it. Either way, what the rule did — or would have done — is
noted here and shown at the end of the mail ("Deneme kayıtları"), each
note with "doğru · yanlış" links. A rule that proves itself is switched on
for good; one that misfires is fixed or removed (user's request,
2026-11-16).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("experiments")

_modes: dict[str, str] = {}
_notes: list[dict] = []


def configure(modes: dict[str, str]) -> None:
    _modes.clear()
    _modes.update(modes)
    _notes.clear()


def mode(name: str) -> str:
    return _modes.get(name, "off")


def active(name: str) -> bool:
    """The rule should run (decide); whether it may change anything is
    `applies(name)`."""
    return mode(name) in ("on", "shadow")


def applies(name: str) -> bool:
    return mode(name) == "on"


def note(name: str, key: str, text: str) -> None:
    applied = applies(name)
    _notes.append({"rule": name, "key": key, "text": text, "applied": applied})
    logger.info("[deneme:%s:%s] %s: %s", name, "uygulandı" if applied else "uygulanmadı", key, text)


def notes() -> list[dict]:
    return list(_notes)


RULE_NAMES = {
    "buyback_weekly": "Geri alım haftalık",
    "headline_same_ticker": "Konu satırı rakam kontrolü",
    "kap_form_tail": "KAP formu son kısmı",
    "calendar_near_days": "Takvim bugün/yarın",
    "headline_only_dedup": "Kopya başlık eleme",
    "no_headline_low_value": "Konu satırı dışı maddeler",
    "llm_repeat_check": "Tekrar kontrolü (model)",
    "same_day_merge": "Aynı gün birleştirme",
}
