"""
Text of a PDF attachment, for KAP filings whose content is the attachment
("Yatırımcı sunumumuz ekte sunulmuştur"). KAP's download endpoint returns
the file behind a Java serialization header, so everything before "%PDF" is
cut off first (found in the user's KapMail project).
"""

from __future__ import annotations

import io
import logging
import re

logger = logging.getLogger(__name__)

MAX_PAGES = 6
MAX_CHARS = 1500


def pdf_text(content: bytes, max_pages: int = MAX_PAGES, max_chars: int = MAX_CHARS) -> str | None:
    start = content.find(b"%PDF")
    if start < 0:
        return None
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content[start:]))
        parts = []
        for page in reader.pages[:max_pages]:
            parts.append(page.extract_text() or "")
            if sum(len(p) for p in parts) >= max_chars:
                break
    except Exception as exc:  # malformed or encrypted attachment: skip it
        logger.info("PDF okunamadı: %s: %s", type(exc).__name__, exc)
        return None
    text = " ".join(" ".join(parts).split())
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)  # words hyphenated across lines
    return text[:max_chars] or None
