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
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import NamedTuple

import groq
from groq import AsyncGroq
from pydantic import ValidationError

from config.constants import NewsCategory
from config.settings import AppConfig
from nlp.providers.base import SummarizerProvider
from scrapers.models import CompanyReport, NewsItem, SynthesizedCompanyReport, SynthesizedSection

logger = logging.getLogger(__name__)

# Groq's free tier caps tokens per minute at 8000 for this model and counts
# the *requested* max_tokens against it, not just what's generated: a 6000
# budget plus a ~3.4k-token prompt was rejected outright (413, "Requested
# 9435", 2026-10-05). Answers are now a few short bullets, so a much smaller
# budget leaves room for the article text that actually matters.
DEFAULT_MAX_TOKENS = 2500
MARKET_MAX_TOKENS = 1500

# Article text sent per call, shared across the items that have a body, so
# a busy day trims each body instead of tipping the request over the limit.
BODY_BUDGET_CHARS = 9000
MAX_BODY_CHARS_PER_ITEM = 1500
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

def _today_line() -> str:
    """The model has no idea what day it is: on 2026-10-06 it reported a
    30 September general assembly as upcoming and an old June rate decision
    as news. Every prompt now starts with the date."""
    return f"Bugünün tarihi: {datetime.now(ZoneInfo('Europe/Istanbul')).strftime('%d.%m.%Y')}"


def _body_cap(items: list[NewsItem]) -> int:
    with_body = sum(1 for item in items if item.body_snippet) or 1
    return min(MAX_BODY_CHARS_PER_ITEM, BODY_BUDGET_CHARS // with_body)


def _item_urls(item: NewsItem) -> list[str]:
    return [str(url) for url in (item.url, getattr(item, "related_kap_url", None)) if url]


def _resolve_source_ids(section: object, sources: list[NewsItem]) -> object:
    """Replaces the model's 1-based "source_ids" with the cited items' URLs
    (deduplicated, order kept). Out-of-range or non-integer ids are ignored:
    a bad citation costs a link, not the whole section."""
    if not isinstance(section, dict) or "source_ids" not in section:
        return section
    urls: list[str] = []
    for raw_id in section.pop("source_ids") or []:
        try:
            index = int(raw_id) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(sources):
            urls.extend(u for u in _item_urls(sources[index]) if u not in urls)
    section.setdefault("source_urls", urls)
    return section


def _parse_category(code: str | None) -> NewsCategory:
    """Model-chosen category code -> NewsCategory; anything unrecognised
    lands in the general bucket rather than failing the whole call."""
    try:
        return NewsCategory(code)
    except ValueError:
        return NewsCategory.GENERAL_SECTOR


SYSTEM_PROMPT = """Sen bir aracı kurum araştırma bülteninin editörüsün. Okuyucu uzun vadeli bir yatırımcı; sabah birkaç dakikada takip ettiği şirkette ne olduğunu öğrenmek istiyor. Sana bir hisse için numaralı ham haberler verilecek. Aynı olayı anlatan haberleri TEK maddede birleştir, bilgi değeri olmayanları atla.

DAHİL ET: sözleşme/sipariş/ihale, yatırım/tesis/kapasite, satın alma/birleşme/varlık satışı, ihracat/yeni pazar, finansal sonuçlar, temettü, pay geri alımı, sermaye artırımı/bedelsiz, borçlanma/tahvil/kredi, yönetim ve ortaklık yapısı değişikliği, içeriden pay alım-satımı, dava/ceza/regülasyon kararı, derecelendirme notu, patent/lisans, şirketin açıkladığı hedefler, aracı kurum hedef fiyat/tavsiye değişikliği (rakamıyla).

DIŞLA: teknik analiz ve fiyat seviyeleri, günlük fiyat/hacim/açığa satış verisi, blok alım-satım akışı, model portföy ağırlıkları; uyum raporları, yönetim kurulu toplantı/katılım istatistikleri, komite listeleri, form alanları ("güncelleme mi, düzeltme mi, ertelenmiş mi"), sorumluluk beyanları; somut karar veya rakam içermeyen yönetici röportaj/konferans sözleri; çok şirketli raporlarda diğer şirketlerin bilgileri.

ANALİST RAPORLARI VE YATIRIMCI SUNUMLARI (okuyucu için çok değerli, asla atlama):
- Metin varsa: kurumu, hedef fiyatı (öncekiyle birlikte), tavsiyeyi, tahmin değişikliklerini ve raporun/sunumun ana tezini 2-5 cümlede özetle.
- Yalnızca başlık varsa: kim neyi yayımladı, tek cümle; başlıkta olmayan rakam ya da tez EKLEME. Ör.: "Ak Yatırım, Akbank için yeni bir şirket raporu yayımladı." Okuyucu linkten açacak.
- subheading: "Analist raporu", "Yatırımcı sunumu", "Toplantı notu" veya "Hedef fiyat".

YAZIM:
- subheading: 1-3 kelimelik olay etiketi ("Yeni iş", "Geri alım", "Temettü", "Varlık satışı", "Satın alma", "Yatırım", "Finansal sonuç", "Borçlanma", "Yönetim", "Ortaklık yapısı", "Patent", "Analist görüşü", "Dava").
- narrative: uzunluğu haberdeki bilgiye göre belirle; kısa yazmak için bilgi atma, uzatmak için dolgu ekleme. Basit bir olay (tek atama, tek patent) 1-2 cümle; çok parçalı bir olay (finansal sonuç, büyük sözleşme, birleşme, analist raporu, geri alım programı) yatırımcının ihtiyaç duyduğu tüm rakamlarla 3-5 cümle. Önce olay ve büyüklüğü (tutar, adet, oran, fiyat aralığı, karşı taraf, tarih), sonra kritik detaylar (vade, finansman, devreye alma, kârın kullanımı, öncekiyle karşılaştırma). Örnek: "Azerenerji (Azerbaycan) ile 250 MWh batarya depolama tesisi için 58,6 mn $'lık EPC-F sözleşmesi imzalandı. Finansman 4 yıl geri ödemeli, tesis 1 yıl içinde devreye girecek."
- Okuyucu linke tıklamadan anlamalı: haberin özündeki rakamı (hedef fiyat ve önceki değeri, tavsiye, tutar, oran) mutlaka yaz. "Yeni hedef fiyatını duyurdu" gibi rakamsız bir cümle YASAK.
- Ham metni kopyalama; bildirim dilini ("Şirketimiz", "kamuoyuna duyurulur") ve içi boş yorumları ("dikkat çekecek", "önem taşımaktadır", "olumlu etki") yazma.
- source_ids: maddeyi destekleyen haberlerin numaraları (ör. [1, 3]). Metne URL yazma.

RAKAM DOĞRULUĞU (en önemli kural):
- SADECE kaynakta geçen isim, kurum ve rakamları kullan; hiçbir şey uydurma, tahmin etme.
- Rakamı kaynaktaki birimiyle aktar. Kısaltma yalnızca kesinse: 58.600.000 -> 58,6 mn; 1.500.000.000 -> 1,5 mlr. Birim belirsizse (milyon mu milyar mı) kaynaktaki yazımı aynen kullan; aynı tutarı iki farklı birimle ASLA yazma.
- Bir tarihin ne olduğu (ihraç, vade, ödeme) kaynakta açık değilse o tarihi yazma.
- Tarihleri kullanıcı mesajındaki "Bugünün tarihi" ile karşılaştır: geçmişteki bir olayı (yapılmış genel kurul, ödenmiş temettü) gelecek zamanla YAZMA; tescil/sonuç bildirimini "yapılacak" diye sunma.
- Borçlanma bildirimlerinde ihraç TAVANI/limiti ile fiilen satılan (nominal) tutarı ayır; "ihraç etti" diye yalnızca satılan tutarı yaz, tavanı ancak ayrıca belirt. Yönetim kurulu/SPK onay tarihleri gibi süreç tarihlerini yazma.
- Bir haberin özü bir rakamsa (tutar, oran) ve kaynakta o rakam yoksa, o maddeyi hiç yazma. İSTİSNALAR (rakam olmasa da HER ZAMAN yaz): (1) analist raporu, yatırımcı sunumu, analist/yatırımcı toplantısı ve toplantı notları (aşağıya bak); (2) kaynağı "kap" olan önemli bildirimler (borçlanma/tahvil ihracı, sözleşme, yatırım, satın alma/satış, temettü, geri alım, sermaye artırımı, yönetim/ortaklık değişikliği, dava): detay yoksa tek cümleyle ne olduğunu yaz, ör. "Yurtdışı piyasalarda tahvil ihracı yaptı; tutar ve vade KAP bildiriminde." Okuyucu linkten açar. "(metin yok, yalnızca başlık)" işaretli haberlerde bilgi sadece başlıktan ibarettir; başlıkta olmayan hiçbir şeyi yazma.
- Hiçbir haber kriterlere uymuyorsa {"sections": []} döndür; boş bölüm dolgu metinden iyidir.

Çıktıyı SADECE JSON olarak ver ("category" ve "kisaca" alanlarını yalnızca kullanıcı mesajı isterse ekle):
{"kisaca": "...", "sections": [{"category": "...", "subheading": "...", "narrative": "...", "source_ids": [1]}]}"""

MARKET_SYSTEM_PROMPT = """Sen Borsa İstanbul odaklı bir sabah bülteninin editörüsün. Sana son 24 saatin piyasa haberlerinin BAŞLIKLARI ve linkleri verilecek. Bültenin en üstündeki "Piyasa Gündemi" kutusunu yaz: piyasayı etkileyebilecek en önemli 3-5 gelişme.

ÖNCELİK SIRASI: TCMB/Fed faiz kararları ve açıklamaları, enflasyon ve diğer makro veriler (TÜİK, büyüme, cari denge, işsizlik), BIST endeks değişiklikleri ve Borsa İstanbul kararları, vergi/regülasyon/SPK kararları, kur ve küresel piyasalarda büyük hareketler, bugün açıklanacak önemli veriler.

KURALLAR:
- SADECE başlıklarda geçen bilgi ve rakamları kullan; rakam UYDURMA. Başlıkta rakam yoksa rakamsız yaz.
- Teknik analiz, destek/direnç, tek bir hissenin günlük fiyat hareketi, "günün en çok yükselenleri", sıradan günlük endeks/altın/döviz fiyat hareketleri gibi maddeleri ALMA (yalnızca rekor veya olağanüstü bir hareketse, rakamıyla yaz).
- Rakamı başlıkta nasıl geçiyorsa öyle yaz; "12.4xx" gibi yer tutucu veya yuvarlatılmış rakam YASAK. Rakam yoksa rakamsız yaz.
- Başlık geçmiş bir dönemi anlatıyorsa (bugünün tarihinden önceki bir ayın kararı vb.) onu bugünün haberi gibi YAZMA.
- Enflasyon maddesi yazıyorsan, açıklanan ayın AYLIK oranını yıllık oranın yanında ver, ör. "Eylül: aylık %2,1, yıllık %29,73 (ENAG yıllık %46,61)". Aylık oran başlıklarda yoksa yalnızca var olanları yaz, uydurma.
- Beklenti haberleri değerlidir: aracı kurumların yaklaşan karar/veri için tahminlerini (ör. "Citi ve Commerzbank 22 Ekim PPK'sında 100 bp indirim bekliyor") tarih ve rakamıyla yaz.
- "Zirve", "rekor", "tarihi seviye" gibi nitelemeleri YALNIZCA başlıkta aynen geçiyorsa kullan; "gün içi en yüksek" ile "yeni zirve" aynı şey değildir. Başlıkta olmayan zaman ifadesi ("haftanın ilk yarısında", "bugün") ekleme.
- Somut bir karar, veri veya olay içermeyen yorum/analiz başlıklarını ("... üzerine analiz gündemde", "uzmanlar değerlendirdi") madde yapma. 3 güçlü madde, 5 zayıf maddeden iyidir; uygun madde yoksa boş liste döndür.
- "-ebilir/-abilir" ile biten tahmin cümleleri ("olumlu duyarlılık yaratabilir", "likiditeyi artırabilir") YASAK; sadece ne olduğunu yaz.
- Aynı gelişmeyi anlatan başlıkları tek maddede birleştir.
- subheading: 1-3 kelimelik etiket: "Faiz", "Enflasyon", "Makro veri", "Endeks değişikliği" (YALNIZCA endekse giren/çıkan şirketler, MSCI/FTSE kararları gibi bileşim değişiklikleri için), "Dünkü seans", "Küresel", "Regülasyon".
- Dünkü seansın kapanışı en fazla TEK maddede, başlıktaki rakamlarla yazılabilir ("Dünkü seans: BIST 100 %0,56 düşüşle 12.374 puanda kapandı; bankacılık yükseldi."). Açılış, gün içi ve "yatay seyir" başlıklarından madde yapma. narrative: 1-2 kısa cümle; mümkünse piyasa için anlamını kaynaktaki bilgiyle söyle, spekülasyon yapma.
- source_ids: maddeyi en iyi destekleyen EN FAZLA 2 başlığın numarası (ör. [2, 5]).
- Önemli bir gelişme yoksa {"sections": []} döndür.

Çıktıyı SADECE şu JSON formatında ver:
{"sections": [{"subheading": "...", "narrative": "...", "source_ids": [1]}]}"""


class _Parsed(NamedTuple):
    """One Groq response: (category code or None, section) pairs plus the
    optional top-level "kisaca" lede."""

    sections: list[tuple[str | None, SynthesizedSection]]
    summary: str | None


class GroqProvider(SummarizerProvider):
    def __init__(self, settings: AppConfig, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._client = AsyncGroq(api_key=settings.env.groq_api_key)
        self._model = settings.env.groq_model
        self._max_tokens = max_tokens

    async def summarize_company_report(self, report: CompanyReport) -> SynthesizedCompanyReport:
        categories = report.ordered_categories()
        total_items = sum(len(items) for _, items in categories)

        sections_by_category: dict[NewsCategory, list[SynthesizedSection]] = {}
        summary: str | None = None
        if total_items <= CHUNK_THRESHOLD:
            sections_by_category, summary = await self._synthesize_whole_report(report.ticker, categories)
        else:
            for category, items in categories:
                sections_by_category[category] = await self._synthesize_category(report.ticker, category, items)

        return SynthesizedCompanyReport(
            ticker=report.ticker,
            company_name=report.company_name,
            summary=summary,
            sections_by_category=sections_by_category,
        )

    async def summarize_market(self, headlines: list[NewsItem]) -> list[SynthesizedSection]:
        lines = [_today_line(), "Son 24 saatin piyasa başlıkları:", ""]
        lines += [f"{i}. {item.title}" for i, item in enumerate(headlines, start=1)]
        parsed = await self._call_with_retry(
            "PIYASA", "piyasa_gundemi", "\n".join(lines), headlines,
            system_prompt=MARKET_SYSTEM_PROMPT, max_tokens=MARKET_MAX_TOKENS,
        )
        return [section for _, section in parsed.sections]

    async def _synthesize_whole_report(
        self, ticker: str, categories: list[tuple[NewsCategory, list[NewsItem]]]
    ) -> tuple[dict[NewsCategory, list[SynthesizedSection]], str | None]:
        """One call for all of a ticker's items. Per-category calls can't see
        each other, so one broker report filed under both Finansal Sonuçlar
        and Sektörel came back as three overlapping sections (AKBNK/HSBC,
        2026-10-05 dry run). Here the model sees everything at once and puts
        each story in exactly one category."""
        prompt, sources = self._build_report_prompt(ticker, categories)
        result: dict[NewsCategory, list[SynthesizedSection]] = {}
        parsed = await self._call_with_retry(ticker, "tum_kategoriler", prompt, sources)
        for category_code, section in parsed.sections:
            category = _parse_category(category_code)
            result.setdefault(category, []).append(section)
        return result, parsed.summary

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
            prompt = self._build_user_prompt(ticker, category, chunk)
            parsed = await self._call_with_retry(ticker, category.value, prompt, chunk)
            sections.extend(section for _, section in parsed.sections)
        return sections

    async def _call_with_retry(
        self, ticker: str, label: str, user_prompt: str, sources: list[NewsItem],
        system_prompt: str = SYSTEM_PROMPT, max_tokens: int | None = None,
    ) -> _Parsed:
        """`sources` are the prompt's numbered items in order; the model cites
        them by number (source_ids) and the numbers are mapped back to URLs
        here — Google News links are ~250 characters each, and having the
        model read and then echo them back cost more tokens than the news."""
        last_error: RuntimeError | None = None
        max_tokens = max_tokens or self._max_tokens
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._call_once(ticker, label, user_prompt, sources, system_prompt, max_tokens=max_tokens)
            except _PayloadTooLargeError as exc:
                last_error = exc
                reduced = max(1, int(max_tokens * PAYLOAD_TOO_LARGE_RETRY_FACTOR))
                logger.warning(
                    "Groq %s/%s deneme %d/%d: istek çok büyük (413), max_tokens %d -> %d ile %s",
                    ticker, label, attempt, MAX_ATTEMPTS, max_tokens, reduced,
                    "tekrar deneniyor" if attempt < MAX_ATTEMPTS else "vazgeçiliyor",
                )
                max_tokens = reduced
            except RuntimeError as exc:
                last_error = exc
                logger.warning(
                    "Groq %s/%s deneme %d/%d başarısız, %s",
                    ticker, label, attempt, MAX_ATTEMPTS,
                    "tekrar deneniyor" if attempt < MAX_ATTEMPTS else "vazgeçiliyor",
                )
        raise last_error

    async def _call_once(
        self, ticker: str, label: str, user_prompt: str, sources: list[NewsItem] | None = None,
        system_prompt: str = SYSTEM_PROMPT, max_tokens: int | None = None,
    ) -> _Parsed:
        """Returns (category code or None, section) pairs: the whole-report
        prompt asks for a "category" field per section, the per-category
        prompt doesn't and its caller ignores it."""
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except groq.APIStatusError as exc:
            if _is_request_too_large(exc):
                raise _PayloadTooLargeError(f"Groq {ticker}/{label}: istek çok büyük (413): {exc}") from exc
            # Observed in testing: with response_format=json_object, a
            # request whose max_tokens is too tight for the model to finish
            # valid JSON often doesn't even come back as a truncated
            # completion — Groq's own server-side JSON validator rejects it
            # outright (400 json_validate_failed). Same underlying problem
            # as finish_reason != "stop" below (not enough budget to answer),
            # so it gets the same treatment: RuntimeError, let main.py fall
            # back to NoopProvider rather than this bubbling up as a raw
            # APIStatusError main.py isn't watching for.
            raise RuntimeError(f"Groq {ticker}/{label}: API hatası: {exc}") from exc

        choice = completion.choices[0]

        if choice.finish_reason != "stop":
            raise RuntimeError(
                f"Groq {ticker}/{label}: finish_reason={choice.finish_reason!r} "
                f"(beklenen 'stop') — kesik/yarım çıktı kullanılmadı."
            )

        try:
            parsed = json.loads(choice.message.content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Groq {ticker}/{label}: geçersiz JSON döndü: {exc}") from exc

        # response_format={"type": "json_object"} only guarantees *syntactically*
        # valid JSON, not that it matches our schema — the model can (and, in
        # testing, did) return a bare array or an object without a "sections"
        # key. Guard the shape explicitly rather than letting AttributeError/
        # TypeError leak past this method uncaught.
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sections"), list):
            raise RuntimeError(
                f"Groq {ticker}/{label}: JSON beklenen {{'sections': [...]}} yapısında değil "
                f"(üst seviye tip: {type(parsed).__name__})"
            )

        try:
            sections = [
                (
                    section.pop("category", None) if isinstance(section, dict) else None,
                    SynthesizedSection.model_validate(_resolve_source_ids(section, sources or [])),
                )
                for section in parsed["sections"]
            ]
        except ValidationError as exc:
            raise RuntimeError(f"Groq {ticker}/{label}: JSON, SynthesizedSection şemasına uymuyor: {exc}") from exc

        summary = parsed.get("kisaca")
        return _Parsed(sections, summary.strip() if isinstance(summary, str) and summary.strip() else None)

    @staticmethod
    def _build_report_prompt(
        ticker: str, categories: list[tuple[NewsCategory, list[NewsItem]]]
    ) -> tuple[str, list[NewsItem]]:
        codes = ", ".join(f'"{c.value}" ({c.display_name_tr})' for c in NewsCategory)
        lines = [
            _today_line(),
            f"Ticker: {ticker}",
            "",
            "Bu hissenin TÜM haberleri aşağıda, kural tabanlı ön-kategorileriyle birlikte veriliyor.",
            f"Her section'a bir \"category\" alanı ekle; değeri şunlardan biri olmalı: {codes}.",
            "Ön-kategori bir ipucudur; haberin içeriğine göre daha uygun kategoriyi seçebilirsin.",
            "Aynı olay/rapor birden fazla haberde veya ön-kategoride geçse bile TEK section yaz ve tek kategoriye koy.",
            "Ayrıca üst seviyede bir \"kisaca\" alanı yaz: günün en önemli gelişmesini anlatan TEK cümle (en fazla ~25 kelime). Hiç section yoksa \"kisaca\" boş string olsun.",
            "",
            "Ham haberler:",
        ]
        sources = [item for _, items in categories for item in items]
        body_cap = _body_cap(sources)
        hints = [category.value for category, items in categories for _ in items]
        for i, (item, hint) in enumerate(zip(sources, hints), start=1):
            lines.append(f"{i}. [ön-kategori: {hint}] Başlık: {item.title}")
            lines.append(
                f"   Metin: {item.body_snippet[:body_cap]}" if item.body_snippet else "   (metin yok, yalnızca başlık)"
            )
            lines.append("")
        return "\n".join(lines), sources

    @staticmethod
    def _build_user_prompt(ticker: str, category: NewsCategory, items: list[NewsItem]) -> str:
        lines = [_today_line(), f"Ticker: {ticker}", f"Kategori: {category.display_name_tr}", "", "Ham haberler:"]
        body_cap = _body_cap(items)
        for i, item in enumerate(items, start=1):
            lines.append(f"{i}. Başlık: {item.title}")
            lines.append(
                f"   Metin: {item.body_snippet[:body_cap]}" if item.body_snippet else "   (metin yok, yalnızca başlık)"
            )
            lines.append("")
        return "\n".join(lines)
