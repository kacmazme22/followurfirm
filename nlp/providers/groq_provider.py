"""
Real Groq-backed SummarizerProvider. Makes one Groq call per NewsCategory
(not one giant per-ticker call) — keeps each prompt small and focused, and
means a schema/parsing problem in one category's response doesn't corrupt a
JSON blob that also held three other categories' worth of sections.

Hallucination guards (verified against real Groq responses during manual
comparison testing — see the openai/gpt-oss-120b vs qwen/qwen3.6-27b
comparison this provider is built from):
  - finish_reason != "stop" -> RuntimeError (never use a truncated JSON blob,
    a truncated JSON parse "succeeding" on a partial object would silently
    drop sections rather than fail loudly).
  - invalid JSON -> RuntimeError.
  - JSON that doesn't validate against SynthesizedSection -> RuntimeError.

None of these are caught here — they propagate out of
summarize_company_report() so main.py can catch them and fall back to
NoopProvider for that ticker (see main.py's _build_company_report), rather
than this module silently swallowing a bad LLM response.
"""

from __future__ import annotations

import json

from groq import AsyncGroq
from pydantic import ValidationError

from config.constants import NewsCategory
from config.settings import AppConfig
from nlp.providers.base import SummarizerProvider
from scrapers.models import CompanyReport, NewsItem, SynthesizedCompanyReport, SynthesizedSection

DEFAULT_MAX_TOKENS = 6000

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
        completion = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(ticker, category, items)},
            ],
        )
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

        try:
            return [SynthesizedSection.model_validate(section) for section in parsed.get("sections", [])]
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
