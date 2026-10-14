from datetime import datetime, timedelta, timezone

from config.constants import NEWS_LOOKBACK_HOURS, SourceType
from scrapers.models import DigestRun, NewsItem, SynthesizedCompanyReport, SynthesizedSection
from utils.memory import MAX_LOOKBACK_HOURS, DigestMemory

NOW = datetime(2026, 10, 15, 4, 0, tzinfo=timezone.utc)


def _item(title, url, source=SourceType.BING_NEWS, ticker="AKBNK"):
    return NewsItem(ticker=ticker, title=title, url=url, source=source)


def _sent(memory, items, cited):
    report = SynthesizedCompanyReport(ticker="AKBNK", company_name="Akbank")
    memory.note_inputs("AKBNK", items)
    section = SynthesizedSection(subheading="Dava", narrative="...", source_urls=cited)
    report.sections_by_category = {items[0].category: [section]}
    run = DigestRun(company_reports=[report])
    memory.record(run, now=NOW)


def test_window_follows_last_send():
    assert DigestMemory().lookback_hours(NOW) == NEWS_LOOKBACK_HOURS
    assert DigestMemory(NOW - timedelta(hours=24)).lookback_hours(NOW) == NEWS_LOOKBACK_HOURS
    assert DigestMemory(NOW - timedelta(days=3)).lookback_hours(NOW) == 72 + 6
    assert DigestMemory(NOW - timedelta(days=30)).lookback_hours(NOW) == MAX_LOOKBACK_HOURS


def test_reported_items_are_dropped_next_day():
    memory = DigestMemory()
    fine = _item("Akbank'a Rekabet Kurulu'ndan 1,7 milyar TL idari para cezası", "https://www.a.com/haber/1")
    skipped = _item("Akbank hisseleri günü yükselişle kapattı bugün", "https://www.a.com/haber/2")
    _sent(memory, [fine, skipped], ["https://www.a.com/haber/1"])
    assert memory.last_sent_at == NOW

    tomorrow = [
        _item("Akbank'a Rekabet Kurulu'ndan 1,7 milyar TL idari para cezası", "https://b.com/x"),  # same story, other site
        _item("Akbank hisseleri günü yükselişle kapattı bugün", "https://www.a.com/haber/2"),  # not in the mail: kept
        _item("Akbank yeni bir tahvil ihracı yaptı ve 500 milyon dolar topladı", "https://a.com/haber/3"),
    ]
    kept = memory.drop_reported(tomorrow, "AKBNK")
    assert [str(i.url) for i in kept] == ["https://www.a.com/haber/2", "https://a.com/haber/3"]


def test_kap_form_titles_match_only_by_link():
    memory = DigestMemory()
    title = "Pay Dışında Sermaye Piyasası Aracı İşlemlerine İlişkin Bildirim (Faiz İçeren)"
    first = _item(title, "https://www.kap.org.tr/tr/Bildirim/1", source=SourceType.KAP)
    _sent(memory, [first], ["https://www.kap.org.tr/tr/Bildirim/1"])
    again = _item(title, "https://kap.org.tr/tr/Bildirim/1/", source=SourceType.KAP)
    new_filing = _item(title, "https://www.kap.org.tr/tr/Bildirim/2", source=SourceType.KAP)
    assert memory.drop_reported([again, new_filing], "AKBNK") == [new_filing]


def test_old_entries_pruned_and_round_trip(tmp_path):
    memory = DigestMemory(entries=[{"ticker": "AKBNK", "url": "https://x.com/old", "title": "t", "sent_on": "2026-09-01"}])
    item = _item("Akbank'a Rekabet Kurulu'ndan 1,7 milyar TL idari para cezası", "https://www.a.com/haber/1")
    _sent(memory, [item], ["https://www.a.com/haber/1"])
    path = tmp_path / "m.json"
    memory.save(path)
    loaded = DigestMemory.load(path)
    assert loaded.last_sent_at == NOW
    assert [e["url"] for e in loaded.entries] == ["https://www.a.com/haber/1"]
