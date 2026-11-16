"""The trial rules (utils/rules.py) on cases from the 2-15 Nov mails, in
both "on" and "shadow" mode."""

from datetime import date

import pytest

from config.constants import SourceType
from scrapers.models import NewsItem, SynthesizedCompanyReport, SynthesizedSection
from utils import experiments, rules
from utils.memory import DigestMemory

TODAY = date(2026, 11, 12)  # a Thursday


def _report(ticker: str, *sections: tuple[str, str]) -> SynthesizedCompanyReport:
    from config.constants import NewsCategory

    category = next(iter(NewsCategory))
    return SynthesizedCompanyReport(
        ticker=ticker, company_name=ticker,
        sections_by_category={category: [SynthesizedSection(subheading=h, narrative=n) for h, n in sections]},
    )


def _texts(report):
    return [s.narrative for _, secs in report.ordered_sections() for s in secs]


def _memory(*entries):
    memory = DigestMemory()
    memory.reported = [dict(e) for e in entries]
    return memory


@pytest.fixture(autouse=True)
def _reset():
    experiments.configure({})
    yield
    experiments.configure({})


GUBRF_EARLIER = {"key": "GUBRF", "date": "2026-11-11", "label": "Geri alım",
                 "text": "10 Kasım'da 7,40 TL'den 250 bin pay geri aldı; program 4 mlr TL'lik."}


@pytest.mark.parametrize("mode, kept", [("on", 0), ("shadow", 1), ("off", 1)])
def test_daily_buyback_trade_held_back_within_the_week(mode, kept):
    experiments.configure({"buyback_weekly": mode})
    report = _report("GUBRF", ("Geri alım", "11 Kasım'da 7,45 TL'den 300 bin pay geri aldı."))
    rules.buyback_weekly(report, _memory(GUBRF_EARLIER), TODAY)
    assert len(_texts(report)) == kept
    assert len(experiments.notes()) == (0 if mode == "off" else 1)


def test_buyback_programme_change_still_goes_out():
    experiments.configure({"buyback_weekly": "on"})
    report = _report("GUBRF", ("Geri alım", "Yönetim kurulu 12.11.2026 tarihli kararla 6 mlr TL'lik yeni geri alım programı açıkladı."))
    rules.buyback_weekly(report, _memory(GUBRF_EARLIER), TODAY)
    assert len(_texts(report)) == 1


def test_first_buyback_of_the_week_goes_out():
    experiments.configure({"buyback_weekly": "on"})
    old = dict(GUBRF_EARLIER, date="2026-11-02")
    report = _report("GUBRF", ("Geri alım", "11 Kasım'da 7,45 TL'den 300 bin pay geri aldı."))
    rules.buyback_weekly(report, _memory(old), TODAY)
    assert len(_texts(report)) == 1


def test_headline_figure_from_another_box_is_refused():
    """15 Nov: "GUBRF: 177 mlr TL ulaşım yatırımı" — 177 mlr was in the
    market box (roads), not in GUBRF's items."""
    experiments.configure({"headline_same_ticker": "on"})
    texts = {"GUBRF": "Geri alım 300 bin pay 7,45 TL", "PIYASA": "Ulaşım 177 mlr TL yatırım"}
    assert not rules.headline_from_own_items("GUBRF: 177 mlr TL ulaşım yatırımı", texts)
    assert rules.headline_from_own_items("GUBRF: 300 bin pay geri aldı", texts)
    assert rules.headline_from_own_items("TCMB: 177 mlr TL yatırım", texts)  # names no followed company


def test_headline_check_in_shadow_mode_only_notes():
    experiments.configure({"headline_same_ticker": "shadow"})
    texts = {"GUBRF": "Geri alım", "PIYASA": "177 mlr TL"}
    assert rules.headline_from_own_items("GUBRF: 177 mlr TL ulaşım yatırımı", texts)
    assert experiments.notes()[0]["applied"] is False


def _item(title, body=None):
    return NewsItem(ticker="KCHOL", title=title, source=SourceType.GOOGLE_NEWS, body_snippet=body)


def test_bare_headline_copy_dropped_but_not_broker_reports_or_new_figures():
    experiments.configure({"headline_only_dedup": "on"})
    items = [
        _item("Koç Holding üçüncü çeyrekte 12,4 milyar TL net kâr açıkladı", "Koç Holding ..."),
        _item("Koç Holding üçüncü çeyrekte 12,4 milyar TL net kâr elde etti"),     # copy
        _item("Koç Holding üçüncü çeyrekte 13 milyar TL net kâr açıkladı"),          # its own figure
        _item("Koç Holding üçüncü çeyrek kâr analiz raporu yayımlandı"),             # broker report
        _item("Koç kâr"),                                                            # too short
    ]
    kept = rules.drop_headline_copies(items, "KCHOL")
    assert [i.title for i in kept] == [items[0].title, items[2].title, items[3].title, items[4].title]


def test_low_value_items_never_the_subject():
    assert rules.is_low_value_for_headline(SynthesizedSection(subheading="Röportaj", narrative="CEO konuştu."))
    assert rules.is_low_value_for_headline(SynthesizedSection(subheading="Etkinlik", narrative="COP31 konferansına katıldı."))
    assert not rules.is_low_value_for_headline(SynthesizedSection(subheading="Yeni iş", narrative="120 mn TL'lik sözleşme imzaladı."))


def test_long_buyback_form_keeps_its_latest_rows():
    experiments.configure({"kap_form_tail": "on"})
    form = "Program: 4 mlr TL. " + " ".join(f"{d:02d}.10.2025 100 pay;" for d in range(1, 29)) * 5 + " 10.11.2026 250.000 pay 7,45 TL"
    text = rules.form_text("Geri Alınan Paylara İlişkin Bildirim", form, 1900)
    assert len(text) <= 1900 and text.startswith("Program") and text.endswith("7,45 TL")
    assert rules.form_text("Özel Durum Açıklaması", form, 1900) == form[:1900]


def test_calendar_shows_today_and_tomorrow_except_mondays():
    experiments.configure({"calendar_near_days": "on"})
    events = [{"ticker": "YEOTK", "date": d, "label": "Temettü", "text": ""} for d in ("2026-11-12", "2026-11-13", "2026-11-16")]
    assert [e["date"] for e in rules.calendar_events(events, TODAY)] == ["2026-11-12", "2026-11-13"]
    assert len(rules.calendar_events(events, date(2026, 11, 9))) == 3  # Monday: the week


def test_same_day_merge_needs_similar_text_and_a_shared_amount():
    experiments.configure({"same_day_merge": "on"})
    report = _report(
        "SAHOL",
        ("Yatırım", "Sabancı Holding enerji dönüşümü için 6 mlr dolar elektrifikasyon yatırımı planlıyor."),
        ("Röportaj", "Sabancı Holding CEO'su 6 mlr dolar elektrifikasyon yatırımı planlandığını söyledi."),
        ("Satın alma", "Sabancı Holding 120 mn dolara bir yazılım şirketi aldı."),
    )
    rules.same_day_merge(report)
    assert len(_texts(report)) == 2


class _FakeProvider:
    async def find_same_events(self, today, earlier):
        return [(0, 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, kept", [("on", 0), ("shadow", 1)])
async def test_llm_repeat_check(mode, kept):
    experiments.configure({"llm_repeat_check": mode})
    memory = _memory({"key": "KCHOL", "date": "2026-11-11", "label": "Finansal sonuç", "text": "3. çeyrekte 12,4 mlr TL kâr"})
    report = _report("KCHOL", ("Finansal sonuç", "Üçüncü çeyrek net kârı 12,4 mlr TL oldu."))
    await rules.llm_repeat_check(report, memory, _FakeProvider())
    assert len(_texts(report)) == kept


def test_news_announced_at_an_event_can_still_be_the_subject():
    """16 Nov dry run: a 50 mlr TL green-finance programme presented at
    COP31 was kept out of the subject because of the word "COP31"."""
    news = SynthesizedSection(subheading="Yeni ürün", narrative="Akbank COP31 konferansında sanayiye 50 milyar TL yeşil finansman programı tanıttı.")
    assert not rules.is_low_value_for_headline(news)


def test_same_day_merge_ignores_items_without_a_common_amount():
    experiments.configure({"same_day_merge": "on"})
    report = _report(
        "AKBNK",
        ("Röportaj", "Akbank/Özben: Akbank olarak 193 tane yenilenebilir elektrik santralının finansmanını sağladık."),
        ("Röportaj", "Akbank/Özben: 2,5 trilyon TL sürdürülebilirlik finansmanını 2030'a kadar yükseltiriz."),
    )
    rules.same_day_merge(report)
    assert len(_texts(report)) == 2


def test_notes_are_grouped_per_rule_and_company():
    experiments.configure({"no_headline_low_value": "on"})
    for text in ("a", "b", "c"):
        experiments.note("no_headline_low_value", "AKBNK", text)
    assert [n["text"] for n in experiments.notes()] == ["a · b · c"]
