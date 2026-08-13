"""
Real Groq-backed SummarizerProvider. Makes one Groq call per NewsCategory
chunk (see CHUNK_SIZE below) rather than one giant per-ticker call — keeps
each prompt small and focused, and means a schema/parsing problem in one
chunk doesn't corrupt a JSON blob that also held many other sections.

Reliability measures, all verified against real Groq responses during manual
testing (openai/gpt-oss-120b, THYAO's real categories):
  - response_format={"type": "json_object"} — Groq's JSON mode, documented
    as supported on all models (console.groq.com/docs/structured-outputs).
    Doesn't guarantee schema match, but does guarantee syntactically valid
    JSON, which is exactly the failure mode observed in testing (missing
    commas at finish_reason="stop" — a real, recurring issue with this
    model, not an edge case).
  - One retry on any guard failure (truncation, invalid JSON, schema
    mismatch, or a Groq API-level rejection) before giving up — cheap
    insurance against the same non-deterministic slip happening twice in a
    row.
  - Category chunking: categories with more than CHUNK_THRESHOLD items are
    split into CHUNK_SIZE-item chunks (or CATEGORY_CHUNK_SIZE_OVERRIDES'ta
    varsa daha küçük bir boyuta), each synthesized separately and their
    SynthesizedSection lists concatenated. Multiple "section groups" under
    one category is fine — ordered_sections() already lists per-category,
    it doesn't care how many sections came from how many calls.

Hallucination guards (unchanged from the original implementation):
  - finish_reason != "stop" -> RuntimeError
  - invalid JSON -> RuntimeError
  - JSON that doesn't validate against SynthesizedSection -> RuntimeError
  - JSON that IS valid but isn't shaped like {"sections": [...]} (json_object
    mode guarantees syntax, not schema — observed in testing: the model can
    return a bare array instead) -> RuntimeError
  - a Groq API-level error -> RuntimeError, EXCEPT one specific case (see
    below): HTTP 413 "request too large" (code=rate_limit_exceeded) gets its
    own guard, _PayloadTooLargeError, because retrying it with the same
    max_tokens deterministically reproduces the same 413 — observed live
    against real THYAO/EREGL/GARAN/ASTOR data (2026-08-13 run: 9/26 calls
    failed this way, all via the generic retry-with-same-size path, which
    never helped). _call_with_retry now retries a _PayloadTooLargeError with
    max_tokens reduced by PAYLOAD_TOO_LARGE_RETRY_FACTOR instead of the
    unmodified size — not a full fix (chunk item count also drives request
    size), but closes the "guaranteed to fail twice" blind spot.
These are retried once (see above), and if the retry also fails, the
RuntimeError propagates out of summarize_company_report() uncaught — main.py
catches it and falls back to NoopProvider for that ticker.
"""

from __future__ import annotations

import json
import logging

import groq
from groq import AsyncGroq
from pydantic import ValidationError

from config.constants import NewsCategory
from config.settings import AppConfig
from nlp.providers.base import SummarizerProvider
from scrapers.models import CompanyReport, NewsItem, SynthesizedCompanyReport, SynthesizedSection

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 6000
MAX_ATTEMPTS = 2  # 1 initial try + 1 retry

# On a 413 "request too large" guard failure, the retry uses
# max_tokens * this factor instead of the original value (e.g. 6000 -> 3900).
# Retrying with an unmodified max_tokens against this specific error is
# pointless — the request is the same size, so Groq rejects it the same way.
PAYLOAD_TOO_LARGE_RETRY_FACTOR = 0.65

# A category with more items than this is split into CHUNK_SIZE-item chunks,
# each sent as its own Groq call — keeps individual prompts/responses small
# (less chance of a truncated or malformed JSON blob) and bounds how much
# work a single failed call can lose.
CHUNK_THRESHOLD = 15
CHUNK_SIZE = 10

# sektorel_genel_haberler ran into Groq's 413 "request too large" far more
# often than any other category during live testing (2026-08-13: 3 of 4
# tickers fell back to NoopProvider, all triggered by this category) — its
# items tend to carry longer body_snippets, so the same CHUNK_SIZE produces
# a noticeably bigger prompt. Smaller chunks for this category specifically,
# rather than lowering CHUNK_SIZE globally and paying the extra-calls cost
# for categories that were never a problem.
CATEGORY_CHUNK_SIZE_OVERRIDES: dict[NewsCategory, int] = {
    NewsCategory.GENERAL_SECTOR: 7,
}


class _PayloadTooLargeError(RuntimeError):
    """Groq rejected the request as too large for the account's TPM budget
    (HTTP 413, body.error.code == "rate_limit_exceeded", message mentions
    "too large"). Subclasses RuntimeError so it's still caught by any
    existing `except RuntimeError` (main.py's fallback-to-noop included);
    it exists only so _call_with_retry can react differently — see
    PAYLOAD_TOO_LARGE_RETRY_FACTOR above."""


def _is_request_too_large(exc: groq.APIStatusError) -> bool:
    if exc.status_code != 413:
        return False
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    return error.get("code") == "rate_limit_exceeded" and "too large" in (error.get("message") or "").lower()

SYSTEM_PROMPT = """Sen bir finansal haber editörüsün. Sana bir hisse senedi için toplanmış ham haber başlıkları, özetleri ve linkleri verilecek. Görevin:
1. Aynı olayı anlatan haberleri birleştirip TEK bir anlatıya dönüştürmek.
2. Önemsiz/tekrarcı haberleri atlamak, sadece bilgi değeri olanları kullanmak.
3. Kendi mantıklı alt-başlıklarını üretmek — sabit bir liste yok, sen karar ver.
4. Her paragrafı hangi haberlerin desteklediğini URL olarak belirtmek.
5. Türkçe, akıcı, gazetecilik diliyle yaz.

KRİTİK KURALLAR (ihlal edilemez):
- SADECE sana verilen ham metinde geçen isim, kurum, rakam ve olayları kullan. Hiçbir ek bilgi, kurum ismi, kişi ismi veya rakam UYDURMA.
- Eğer bir rakam (yüzde, tutar vb.) kaynak metinde net olarak belirtilmemişse, o rakamı ASLA tahmini/placeholder olarak yazma (örn. "%X artış" gibi ifadeler YASAK) — bunun yerine o detayı tamamen atla veya belirsiz bırak ("bir miktar artış" gibi genel ifade kullan).
- Kaynak metinde adı geçmeyen hiçbir aracı kurum, analist veya şirket ismini ekleme.

Çıktıyı SADECE şu JSON formatında ver, başka hiçbir metin ekleme:
{"sections": [{"subheading": "...", "narrative": "...", "source_urls": ["...", "..."]}]}"""


class GroqProvider(SummarizerProvider):
    def __init__(self, settings: AppConfig, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._client = AsyncGroq(api_key=settings.env.groq_api_key)
        self._model = settings.env.groq_model
        self._max_tokens = max_tokens

    async def summarize_company_report(self, report: CompanyReport) -> SynthesizedCompanyReport:
        sections_by_category: dict[NewsCategory, list[SynthesizedSection]] = {}
        for category, items in report.ordered_categories():
            sections_by_category[category] = await self._synthesize_category(report.ticker, category, items)

        return SynthesizedCompanyReport(
            ticker=report.ticker,
            company_name=report.company_name,
            sections_by_category=sections_by_category,
        )

    async def _synthesize_category(
        self, ticker: str, category: NewsCategory, items: list[NewsItem]
    ) -> list[SynthesizedSection]:
        chunk_size = CATEGORY_CHUNK_SIZE_OVERRIDES.get(category, CHUNK_SIZE)
        chunks = (
            [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
            if len(items) > CHUNK_THRESHOLD
            else [items]
        )

        sections: list[SynthesizedSection] = []
        for chunk in chunks:
            sections.extend(await self._call_with_retry(ticker, category, chunk))
        return sections

    async def _call_with_retry(
        self, ticker: str, category: NewsCategory, items: list[NewsItem]
    ) -> list[SynthesizedSection]:
        last_error: RuntimeError | None = None
        max_tokens = self._max_tokens
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._call_once(ticker, category, items, max_tokens=max_tokens)
            except _PayloadTooLargeError as exc:
                last_error = exc
                reduced = max(1, int(max_tokens * PAYLOAD_TOO_LARGE_RETRY_FACTOR))
                logger.warning(
                    "Groq %s/%s deneme %d/%d: istek çok büyük (413), max_tokens %d -> %d ile %s",
                    ticker, category.value, attempt, MAX_ATTEMPTS, max_tokens, reduced,
                    "tekrar deneniyor" if attempt < MAX_ATTEMPTS else "vazgeçiliyor",
                )
                max_tokens = reduced
            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    "Groq %s/%s deneme %d/%d başarısız, %s",
                    ticker, category.value, attempt, MAX_ATTEMPTS,
                    "tekrar deneniyor" if attempt < MAX_ATTEMPTS else "vazgeçiliyor",
                )
        raise last_error

    async def _call_once(
        self, ticker: str, category: NewsCategory, items: list[NewsItem], max_tokens: int | None = None
    ) -> list[SynthesizedSection]:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_user_prompt(ticker, category, items)},
                ],
            )
        except groq.APIStatusError as exc:
            if _is_request_too_large(exc):
                raise _PayloadTooLargeError(f"Groq {ticker}/{category.value}: istek çok büyük (413): {exc}") from exc
            # Observed in testing: with response_format=json_object, a
            # request whose max_tokens is too tight for the model to finish
            # valid JSON often doesn't even come back as a truncated
            # completion — Groq's own server-side JSON validator rejects it
            # outright (400 json_validate_failed). Same underlying problem
            # as finish_reason != "stop" below (not enough budget to answer),
            # so it gets the same treatment: RuntimeError, let main.py fall
            # back to NoopProvider rather than this bubbling up as a raw
            # APIStatusError main.py isn't watching for.
            raise RuntimeError(f"Groq {ticker}/{category.value}: API hatası: {exc}") from exc

        choice = completion.choices[0]

        if choice.finish_reason != "stop":
            raise RuntimeError(
                f"Groq {ticker}/{category.value}: finish_reason={choice.finish_reason!r} "
                f"(beklenen 'stop') — kesik/yarım çıktı kullanılmadı."
            )

        try:
            parsed = json.loads(choice.message.content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Groq {ticker}/{category.value}: geçersiz JSON döndü: {exc}") from exc

        # response_format={"type": "json_object"} only guarantees *syntactically*
        # valid JSON, not that it matches our schema — the model can (and, in
        # testing, did) return a bare array or an object without a "sections"
        # key. Guard the shape explicitly rather than letting AttributeError/
        # TypeError leak past this method uncaught.
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sections"), list):
            raise RuntimeError(
                f"Groq {ticker}/{category.value}: JSON beklenen {{'sections': [...]}} yapısında değil "
                f"(üst seviye tip: {type(parsed).__name__})"
            )

        try:
            return [SynthesizedSection.model_validate(section) for section in parsed["sections"]]
        except ValidationError as exc:
            raise RuntimeError(f"Groq {ticker}/{category.value}: JSON, SynthesizedSection şemasına uymuyor: {exc}") from exc

    @staticmethod
    def _build_user_prompt(ticker: str, category: NewsCategory, items: list[NewsItem]) -> str:
        lines = [f"Ticker: {ticker}", f"Kategori: {category.display_name_tr}", "", "Ham haberler:"]
        for i, item in enumerate(items, start=1):
            lines.append(f"{i}. Başlık: {item.title}")
            if item.body_snippet:
                lines.append(f"   Özet: {item.body_snippet}")
            if item.url:
                lines.append(f"   URL: {item.url}")
            lines.append("")
        return "\n".join(lines)
