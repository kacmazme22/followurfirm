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
from typing import NamedTuple

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

def _parse_category(code: str | None) -> NewsCategory:
    """Model-chosen category code -> NewsCategory; anything unrecognised
    lands in the general bucket rather than failing the whole call."""
    try:
        return NewsCategory(code)
    except ValueError:
        return NewsCategory.GENERAL_SECTOR


SYSTEM_PROMPT = """Sen bir aracı kurum araştırma bülteninin editörüsün. Okuyucu, hisselerini takip eden uzun vadeli bir yatırımcı; sabah birkaç dakikada şirketlerinde ne olduğunu öğrenmek istiyor. Sana bir hisse için toplanmış ham haberler, özetleri ve linkleri verilecek. Görevin:
1. Aynı olayı anlatan haberleri birleştirip TEK maddeye dönüştürmek.
2. Önemsiz/tekrarcı haberleri atlamak, sadece yatırımcıya bilgi veren olayları kullanmak.
3. Her maddeyi hangi haberlerin desteklediğini URL olarak belirtmek.

YAZIM FORMATI (kısa ve yoğun — ham metni ASLA kopyalama, işle):
- subheading: 1-3 kelimelik olay etiketi, ör. "Yeni iş", "Geri alım", "Temettü", "Varlık satışı", "Satın alma", "Yatırım", "Finansal sonuç", "Borçlanma", "Yönetim", "Ortaklık yapısı", "Patent", "Analist görüşü", "Dava".
- narrative: EN FAZLA 2 cümle, toplam ~45 kelime. Önce olay ve büyüklüğü (tutar, adet, oran, fiyat aralığı, karşı taraf, tarih), sonra varsa tek bir kritik detay (vade, finansman, devreye alma, kârın kullanımı). Örnek:
  "Azerenerji (Azerbaycan) ile 250 MWh batarya depolama tesisi için 58,6 mn $'lık EPC-F sözleşmesi imzalandı. Finansman 4 yıl geri ödemeli, tesis 1 yıl içinde devreye girecek."
  "5 Ekim'de 4,76-4,82 TL aralığından 456.517 pay geri alındı; toplam geri alınan pay 1,60 mn adede (%0,142) ulaştı."
- Büyük tutarları kısalt: 58.600.000 dolar -> 58,6 mn $; 1.500.000.000 TL -> 1,5 mlr TL.
- "Şirketimiz", "kamuoyuna duyurulur", "olumlu etki" gibi bildirim dilini ve form alanlarını (güncelleme mi, düzeltme mi, ertelenmiş mi) yazma.

KRİTİK KURALLAR (ihlal edilemez):
- SADECE sana verilen ham metinde geçen isim, kurum, rakam ve olayları kullan. Hiçbir ek bilgi, kurum ismi, kişi ismi veya rakam UYDURMA.
- Eğer bir rakam (yüzde, tutar vb.) kaynak metinde net olarak belirtilmemişse, o rakamı ASLA tahmini/placeholder olarak yazma (örn. "%X artış" gibi ifadeler YASAK) — bunun yerine o detayı tamamen atla veya belirsiz bırak ("bir miktar artış" gibi genel ifade kullan).
- Kaynak metinde adı geçmeyen hiçbir aracı kurum, analist veya şirket ismini ekleme.

İÇERİK FİLTRESİ:
- DIŞLA: günlük açılış-kapanış fiyatı, işlem hacmi, açığa satış hacmi/oranı gibi anlık/ham piyasa verileri. Bu tür haberler yatırımcının kendi grafik ekranlarından takip ettiği "an itibariyle fotoğraf" niteliğindedir; bültende tekrarına gerek yoktur. Ham veri maddeleri, tek başına yorum veya bağlam içermiyorsa, asla section'a çevrilmemelidir.
- DIŞLA: sorumluluk beyanı, yasal uyum beyanı ve benzeri prosedürel/hukuki formalite açıklamaları. Özellikle "belgenin doğruluğu/sorumluluğu beyan edilmiştir" türü kalıp metinler yatırım kararına girdi üretmiyorsa tamamen atlanmalıdır.
- DIŞLA: birden fazla farklı hisseye, endekse veya yatırım fikrine ait karışık istatistik tabloları / teknik takip listeleri. Eğer kaynak metni birden fazla ticker kodu, BIST100/BIST50 gibi endeks satırları ve çok sayıda fiyat/ölçü sütununu aynı anda içeriyorsa (örn. "MIATK EREGL ALTNY AKBNK ..." veya "BIST100 / BIST50 / AEFES / AKBNK / KCHOL..." gibi karışık tablo pasajları), bu tür içerik TAMAMEN ATLANMALIDIR. Bu, tek hisseye özel ve anlamlı bir haber değil, çok-hisseli toplu veri dökümüdür.
- DIŞLA: teknik analiz ve fiyat seviyeleri — destek/direnç, hareketli ortalama, RSI/MACD, formasyon, al-sat seviyesi, "hisse yüzde X yükseldi/düştü" gibi günlük fiyat hareketi haberleri. Okuyucu bunları istemiyor; bülten bir uzun vadeli yatırımcı içindir, fiyat grafiği yerine şirketin işini anlatır.
- DAHİL ET (asıl amaç budur): şirketin işine ve değerine dair olaylar — yeni sözleşme/sipariş/ihale, yatırım, tesis/kapasite, satın alma/birleşme/ortaklık, ihracat ve yeni pazar, finansal sonuçlar (gelir, kâr, marj ve bunların değişimi), temettü, pay geri alımı, sermaye artırımı/bedelsiz, borçlanma/kredi/tahvil ihracı, yönetim ve ortaklık yapısı değişikliği, içeriden (yönetici/ana ortak) pay alım-satımı, dava/ceza/regülasyon kararları, kredi derecelendirme notu, aracı kurumların hedef fiyat ve tavsiye değişiklikleri (tek cümleyle, kurum adı kaynakta geçiyorsa), şirketin kendi açıkladığı hedef ve beklentiler, sektörü doğrudan etkileyen düzenlemeler.
- Her section'da önce NE oldu, sonra kaynakta dayanağı varsa yatırımcı için önemini (büyüklüğü, ciroya/kâra etkisi, takvimi) yaz. Kaynakta dayanağı yoksa önem cümlesi YAZMA: "dikkat çekecek", "potansiyel etkileri olabilir", "önem taşımaktadır", "güçlendirilmesi açısından" gibi içi boş kapanış cümleleri YASAK.
- DIŞLA: prosedürel/periyodik KAP içeriği — kurumsal yönetim veya sürdürülebilirlik uyum raporları, yönetim kurulu toplantı/katılım istatistikleri, komite listeleri, "bu açıklama düzeltme/erteleme değildir" gibi form alanları. Bir atama/karar haberi varsa sadece kimin hangi göreve geldiğini yaz, mevzuat madde numaralarını yazma.
- DIŞLA: blok/kurumsal alım-satım akışı ("X kurumdan yüklü satış"), aracı kurumların model portföy ağırlık değişiklikleri ve takas/aracı kurum dağılımı — bunlar şirketin işine dair değil, hisse akışına dair.
- DIŞLA: yöneticilerin konferans/röportaj sözleri ve genel görüşleri (ör. "yapay zekâ büyük fırsatlar getiriyor") — şirkete dair somut bir karar, rakam veya hedef içermiyorsa.
- Birden çok şirketi kapsayan bir rapordan (ör. sektör raporu) sadece bu hisseye dair kısmı al; diğer şirketlerin hedef fiyat ve tavsiyelerini yazma.
- Eğer bir madde sadece fiyat, hacim, oran, günlük değişim, açığa satış miktarı gibi ham veriden oluşuyorsa, onu hiçbir şekilde section yapma; tamamen atla. Eğer aynı madde yorum/bağlam da taşıyorsa, yalnızca şirketin işine dair kısmını koru, fiyat ve teknik verileri sil.
- Hiçbir madde bu kriterleri karşılamıyorsa {"sections": []} döndür — boş bölüm, dolgu metinden iyidir.
- Bu kurallar, "yalnızca ham metinde geçen bilgiyi kullan, uydurma" kuralıyla çelişmez. Hedef, LLM'in kaynak metnindeki var olan ama atladığı detayları ortaya çıkarmasıdır; rakamı, tutarı veya tarafı kaynakta olmayan şeylerden uydurmak değil.

BİRLEŞTİRME KURALI (tekrarları tek anlatıda topla, bilgi silmeden):
- Birleştirme SADECE aynı olayı/konuyu anlatan maddeler için geçerlidir.
- Farklı konulardaki maddeleri asla silme veya tek maddeye indirmeye çalışma — her BAĞIMSIZ konu kendi section'ını hak eder.
- Birleştirme = aynı bilgiyi tekrar etmemek demektir, bilgi kaybetmek değildir.
- Aynı temel konu/olayı anlatan birden fazla ham haberi (farklı kaynaklardan gelse bile) TEK bir section içinde birleştir, ama farklı konuya ait maddeler birbirine karıştırılmamalıdır.
- Kredi kullanımı gibi aynı işlemin farklı parçalarını (tutar, para birimi, amaç, proje) tek birleşik maddede toplamayı tercih et; ancak ayrı ve bağımsız konular için ayrı section bırak.
- Finansal sonuçlar için tek ve mutlak "en fazla 1 section" kuralı yoktur. Sadece aynı döneme ait aynı olay/konu varsa tek section içinde birleştirilir; farklı bağımsız finansal olaylar ayrı section olarak kalır.

URL YAZIM KURALI:
- source_urls alanı DIŞINDA, narrative metnine çıplak URL veya parantez içinde link YAZMA. Kaynaklar sadece source_urls listesinde yer almalıdır.

KAP AÇIKLAMALARINI ÖZETLERKEN DERİNLİK KURALI:
- KAP açıklamalarını asla "şirket KAP üzerinden bir açıklama paylaştı" gibi yüzeysel şekilde geçme.
- Kaynak metinde somut rakam, taraf, tutar, tarih, amaç, kur, proje, kurum veya koşul varsa, yatırımcı için en önemli olanları (2 cümle sınırı içinde) mutlaka kullan.
- Örnek: bir kredi anlaşması haberinde sadece "kredi kullanıldı" demek yeterli değil; kredinin tutarını, para birimini, hangi projeye/amaca tahsis edildiğini, hangi kurumdan alındığını mutlaka belirt.
- Eğer kaynakta bu detaylar varsa, LLM bu detayları atlamamalı; fakat kaynakta yoksa, hiç bir detay eklememeli, sadece genel cümleyle yetinmelidir.

Çıktıyı SADECE şu JSON formatında ver, başka hiçbir metin ekleme ("category" ve "kisaca" alanlarını yalnızca kullanıcı mesajı isterse ekle):
{"kisaca": "...", "sections": [{"category": "...", "subheading": "...", "narrative": "...", "source_urls": ["...", "..."]}]}"""

MARKET_SYSTEM_PROMPT = """Sen Borsa İstanbul odaklı bir sabah bülteninin editörüsün. Sana son 24 saatin piyasa haberlerinin BAŞLIKLARI ve linkleri verilecek. Bültenin en üstündeki "Piyasa Gündemi" kutusunu yaz: piyasayı etkileyebilecek en önemli 3-5 gelişme.

ÖNCELİK SIRASI: TCMB/Fed faiz kararları ve açıklamaları, enflasyon ve diğer makro veriler (TÜİK, büyüme, cari denge, işsizlik), BIST endeks değişiklikleri ve Borsa İstanbul kararları, vergi/regülasyon/SPK kararları, kur ve küresel piyasalarda büyük hareketler, bugün açıklanacak önemli veriler.

KURALLAR:
- SADECE başlıklarda geçen bilgi ve rakamları kullan; rakam UYDURMA. Başlıkta rakam yoksa rakamsız yaz.
- Teknik analiz, destek/direnç, tek bir hissenin günlük fiyat hareketi, "günün en çok yükselenleri" gibi maddeleri ALMA.
- Aynı gelişmeyi anlatan başlıkları tek maddede birleştir.
- subheading: 1-3 kelimelik etiket ("Faiz", "Enflasyon", "Endeks değişikliği", "Küresel", "Regülasyon"). narrative: 1-2 kısa cümle; mümkünse piyasa için anlamını kaynaktaki bilgiyle söyle, spekülasyon yapma.
- Önemli bir gelişme yoksa {"sections": []} döndür.

Çıktıyı SADECE şu JSON formatında ver:
{"sections": [{"subheading": "...", "narrative": "...", "source_urls": ["..."]}]}"""


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
        lines = ["Son 24 saatin piyasa başlıkları:", ""]
        for i, item in enumerate(headlines, start=1):
            lines.append(f"{i}. {item.title}")
            if item.url:
                lines.append(f"   URL: {item.url}")
        parsed = await self._call_with_retry("PIYASA", "piyasa_gundemi", "\n".join(lines), MARKET_SYSTEM_PROMPT)
        return [section for _, section in parsed.sections]

    async def _synthesize_whole_report(
        self, ticker: str, categories: list[tuple[NewsCategory, list[NewsItem]]]
    ) -> tuple[dict[NewsCategory, list[SynthesizedSection]], str | None]:
        """One call for all of a ticker's items. Per-category calls can't see
        each other, so one broker report filed under both Finansal Sonuçlar
        and Sektörel came back as three overlapping sections (AKBNK/HSBC,
        2026-10-05 dry run). Here the model sees everything at once and puts
        each story in exactly one category."""
        prompt = self._build_report_prompt(ticker, categories)
        result: dict[NewsCategory, list[SynthesizedSection]] = {}
        parsed = await self._call_with_retry(ticker, "tum_kategoriler", prompt)
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
            parsed = await self._call_with_retry(ticker, category.value, prompt)
            sections.extend(section for _, section in parsed.sections)
        return sections

    async def _call_with_retry(
        self, ticker: str, label: str, user_prompt: str, system_prompt: str = SYSTEM_PROMPT
    ) -> _Parsed:
        last_error: RuntimeError | None = None
        max_tokens = self._max_tokens
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await self._call_once(ticker, label, user_prompt, system_prompt, max_tokens=max_tokens)
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
        self, ticker: str, label: str, user_prompt: str, system_prompt: str = SYSTEM_PROMPT,
        max_tokens: int | None = None,
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
                    SynthesizedSection.model_validate(section),
                )
                for section in parsed["sections"]
            ]
        except ValidationError as exc:
            raise RuntimeError(f"Groq {ticker}/{label}: JSON, SynthesizedSection şemasına uymuyor: {exc}") from exc

        summary = parsed.get("kisaca")
        return _Parsed(sections, summary.strip() if isinstance(summary, str) and summary.strip() else None)

    @staticmethod
    def _build_report_prompt(ticker: str, categories: list[tuple[NewsCategory, list[NewsItem]]]) -> str:
        codes = ", ".join(f'"{c.value}" ({c.display_name_tr})' for c in NewsCategory)
        lines = [
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
        i = 0
        for category, items in categories:
            for item in items:
                i += 1
                lines.append(f"{i}. [ön-kategori: {category.value}] Başlık: {item.title}")
                if item.body_snippet:
                    lines.append(f"   Özet: {item.body_snippet}")
                if item.url:
                    lines.append(f"   URL: {item.url}")
                lines.append("")
        return "\n".join(lines)

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
