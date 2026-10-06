"""Offline tests for GroqProvider's call layout, with a fake Groq client in
place of the network: a small report goes out as one whole-report call whose
sections are grouped by the model-chosen category; a large one falls back to
per-category calls."""

import asyncio
import json
from types import SimpleNamespace

from config.constants import NewsCategory, SourceType
from nlp.providers.groq_provider import CHUNK_THRESHOLD, GroqProvider
from scrapers.models import CompanyReport, NewsItem


class _FakeCompletions:
    def __init__(self, sections):
        self.sections = sections
        self.prompts: list[str] = []

    async def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][1]["content"])
        content = json.dumps({"sections": self.sections})
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])


def _provider(sections):
    provider = GroqProvider.__new__(GroqProvider)
    completions = _FakeCompletions(sections)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider._model = "fake"
    provider._max_tokens = 1000
    return provider, completions


def _report(n_financial: int, n_general: int) -> CompanyReport:
    report = CompanyReport(ticker="AKBNK", company_name="Akbank")
    for i in range(n_financial):
        report.add_item(NewsItem(ticker="AKBNK", title=f"Bilanço {i}", source=SourceType.BIGPARA, category=NewsCategory.FINANCIALS))
    for i in range(n_general):
        report.add_item(NewsItem(ticker="AKBNK", title=f"Haber {i}", source=SourceType.BIGPARA, category=NewsCategory.GENERAL_SECTOR))
    return report


def test_small_report_is_one_call_grouped_by_model_category():
    provider, completions = _provider([
        {"category": "finansal_sonuclar", "subheading": "HSBC raporu", "narrative": "x"},
        {"category": "uydurma_kategori", "subheading": "Diğer", "narrative": "y"},
    ])
    result = asyncio.run(provider.summarize_company_report(_report(1, 2)))

    assert len(completions.prompts) == 1
    assert "ön-kategori: finansal_sonuclar" in completions.prompts[0]
    assert [s.subheading for s in result.sections_by_category[NewsCategory.FINANCIALS]] == ["HSBC raporu"]
    # Unknown category codes fall back to the general bucket.
    assert [s.subheading for s in result.sections_by_category[NewsCategory.GENERAL_SECTOR]] == ["Diğer"]


def test_large_report_falls_back_to_per_category_calls():
    provider, completions = _provider([{"subheading": "Bölüm", "narrative": "x"}])
    asyncio.run(provider.summarize_company_report(_report(2, CHUNK_THRESHOLD)))

    assert len(completions.prompts) > 1
    assert all("ön-kategori" not in p for p in completions.prompts)


def test_kisaca_is_read_from_whole_report_response():
    provider, completions = _provider([{"category": "yeni_is_iliskileri", "subheading": "Yeni iş", "narrative": "x"}])
    completions_create = completions.create

    async def create_with_kisaca(**kwargs):
        response = await completions_create(**kwargs)
        payload = json.loads(response.choices[0].message.content)
        payload["kisaca"] = "  58,6 mn $'lık sözleşme imzalandı.  "
        response.choices[0].message.content = json.dumps(payload)
        return response

    completions.create = create_with_kisaca
    result = asyncio.run(provider.summarize_company_report(_report(1, 0)))
    assert result.summary == "58,6 mn $'lık sözleşme imzalandı."


def test_source_ids_are_mapped_back_to_urls():
    provider, completions = _provider([
        {"category": "finansal_sonuclar", "subheading": "Temettü", "narrative": "x", "source_ids": [2, 2, 99, "a"]},
    ])
    report = CompanyReport(ticker="GUBRF", company_name="Gübretaş")
    report.add_item(NewsItem(ticker="GUBRF", title="A", source=SourceType.BIGPARA, url="https://a.example/1", category=NewsCategory.FINANCIALS))
    report.add_item(NewsItem(ticker="GUBRF", title="B", source=SourceType.KAP, url="https://www.kap.org.tr/tr/Bildirim/2", category=NewsCategory.FINANCIALS))
    result = asyncio.run(provider.summarize_company_report(report))

    section = result.sections_by_category[NewsCategory.FINANCIALS][0]
    assert [str(u) for u in section.source_urls] == ["https://www.kap.org.tr/tr/Bildirim/2"]
    # URLs are no longer sent to the model at all.
    assert "https://" not in completions.prompts[0]


def test_headline_only_items_are_marked_for_the_model():
    provider, completions = _provider([])
    asyncio.run(provider.summarize_company_report(_report(1, 0)))
    assert "(metin yok, yalnızca başlık)" in completions.prompts[0]


def test_audit_log_traces_each_section_to_its_source_titles(caplog):
    from utils.audit import log_inputs, log_outputs
    from scrapers.models import SynthesizedSection

    items = [
        NewsItem(ticker="PIYASA", title="Borsa güne yükselişle başladı", source=SourceType.GOOGLE_NEWS, url="https://n.example/1"),
        NewsItem(ticker="PIYASA", title="TCMB faizi sabit tuttu", source=SourceType.GOOGLE_NEWS, url="https://n.example/2"),
    ]
    section = SynthesizedSection(subheading="Faiz", narrative="x", source_urls=["https://n.example/2"])
    with caplog.at_level("INFO", logger="audit"):
        log_inputs("PIYASA", items)
        log_outputs("PIYASA", [section], items)
    assert "girdi 1 (google_news): Borsa güne yükselişle başladı | (yalnızca başlık)" in caplog.text
    assert "çıktı «Faiz» <- TCMB faizi sabit tuttu" in caplog.text
