"""
"Yanlış / gereksiz" links under every item of the mail.

A click opens a pre-filled mail from the reader to their own address,
subject "FollowUrFirm geri bildirim | <tür> | <tarih> | <hisse> | <başlık>",
with the item's text in the body. The messages can then be read (Gmail
search: subject:"FollowUrFirm geri bildirim") and fixed without anyone
having to describe the example again or take screenshots.
"""

from __future__ import annotations

from urllib.parse import quote

SUBJECT_PREFIX = "FollowUrFirm geri bildirim"
KINDS = {"yanlis": "yanlış", "gereksiz": "gereksiz", "not": "not"}
MAX_BODY_CHARS = 600


def feedback_link(to: str | None, kind: str, day: str, key: str, label: str = "", text: str = "") -> str | None:
    if not to:
        return None
    subject = " | ".join(p for p in (SUBJECT_PREFIX, KINDS[kind], day, key, label) if p)
    body = (f"Madde: {' '.join(text.split())[:MAX_BODY_CHARS]}\n\n" if text else "") + "Notun (isteğe bağlı): "
    return f"mailto:{quote(to, safe='@')}?subject={quote(subject)}&body={quote(body)}"
