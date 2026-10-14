"""Offline tests for the rule-based relevance filter and Tier-1 categorizer:
pure string rules, no network, so they run in plain `pytest`."""

import pytest

from config.constants import NewsCategory
from nlp.categorizer import _match_keyword_category
from nlp.relevance import is_market_noise, is_multi_ticker_list


@pytest.mark.parametrize("title", [
    "AKBNK teknik analiz: destek seviyesi 58 TL",
    "Akbank hisseleri yüzde 3,2 yükseldi",
    "KCHOL payları %2 değer kaybetti",
    "Koç Holding hissesi rekor kırdı",
    "Akbank direnç bölgesine yaklaştı",
    "KCHOL hareketli ortalama üzerinde",
    "Gübretaş açığa satış verileri",
    "Günün en çok yükselen hisseleri arasında GUBRF",
    "AKBNK RSI aşırı alımda",
])
def test_market_noise_dropped(title):
    assert is_market_noise(title)


@pytest.mark.parametrize("title", [
    "Akbank net kârı yüzde 30 arttı",
    "Akbank payları yüzde 100 bedelsiz sermaye artırımı",
    "Koç Holding payların yüzde 2'sine kadar geri alım yapacak",
    "Gübretaş yeni tesis için devlet desteği aldı",
    "Takasbank ile anlaşma imzalandı",
    "KCHOL hedef fiyatını 320 TL'ye yükseltti",
    "Gübretaş satışları yüzde 15 arttı",
])
def test_company_news_kept(title):
    assert not is_market_noise(title)


def test_multi_ticker_list():
    assert is_multi_ticker_list("ESCOM AKYHO HATSN KOCMT NATEN pay işlemleri")
    assert not is_multi_ticker_list("AKBNK sendikasyon kredisi yeniledi")


@pytest.mark.parametrize("text, expected", [
    ("İhracat anlaşması imzalandı", NewsCategory.NEW_BUSINESS),
    ("İSTİFA", NewsCategory.KAP_MATERIAL),
    ("Geri alım programı kapsamında pay satın alındı", NewsCategory.KAP_MATERIAL),
    ("Koç Holding İtalya'da şirket satın aldı", NewsCategory.NEW_BUSINESS),
    ("Gübretaş üçüncü çeyrek net kârı açıklandı", NewsCategory.FINANCIALS),
])
def test_keyword_category(text, expected):
    assert _match_keyword_category(text) == expected


# --- From the 2026-10-03 mail the user flagged ("bu olmamalıydı") ----------

from nlp.relevance import clean_title  # noqa: E402
from utils.article_fetcher import clean_article_text  # noqa: E402
from nlp.providers.noop_provider import _lead  # noqa: E402


def test_clean_title_strips_bigpara_asterisks_and_wrapper():
    assert clean_title("***KCHOL*** (*Koç Holding YKB Vekili Ali Koç: YZ doğru kullanılırsa büyük fırsatlar getiriyor)") == (
        "Koç Holding YKB Vekili Ali Koç: YZ doğru kullanılırsa büyük fırsatlar getiriyor"
    )
    # Asterisks used to hide this list from the multi-ticker rule.
    assert is_multi_ticker_list(clean_title("***ESCOM* *AKYHO* *HATSN* *KOCMT* *NATEN* *AYDEM*..."))


@pytest.mark.parametrize("title", [
    "YEO TEKNOLOJI ENERJI (YEOTK) Hisse Senedi",
    "YEOTK Hisse Yorumları (YEO TEKNOLOJI ENERJI Güncel Yorumlar)",
    "YEOTK MERKEZİ KAYIT KURULUŞU A.Ş. (Borsada İşlem Gören Tipe Dönüşüm Duyurusu)",
])
def test_quote_pages_and_infra_notices_dropped(title):
    assert is_market_noise(title)


def test_bigpara_analysis_bulletin_dropped_by_url():
    assert is_market_noise("Analiz: Günlük Bülten", "https://www.bigpara.com/haberler/araci-kurum-raporlari/analiz-gunluk-bulten-ziraat-yatirim_ID1/")


def test_article_cleaning_drops_site_chrome_and_disclaimer():
    chrome = ("Bildirimler Bildirimler En Çok Arananlar © Copyright 2026 - Hürriyet Gazetecilik Matbaacılık A.Ş "
              "Kullanım Koşulları. BIST 100 Dolar Euro Altın Petrol Tahvil")
    assert clean_article_text(chrome) is None

    article = (
        "En Çok Arananlar Bigpara\n"
        "Gübretaş Maden Yatırımları Genel Kurulu, 1,5 milyar TL kârın en geç 9 Ekim 2026 tarihini geçmemek üzere "
        "dağıtılması için Yönetim Kuruluna yetki verdi.\n"
        "ForInvest Haber Burada yer alan yatırım bilgi, yorum ve tavsiyeleri yatırım danışmanlığı kapsamında değildir."
    )
    cleaned = clean_article_text(article)
    assert cleaned.startswith("Gübretaş Maden")
    assert "danışmanlığı" not in cleaned and "En Çok Arananlar" not in cleaned


def test_noop_fallback_keeps_only_the_lead():
    body = "Birinci cümle burada bitiyor. " * 30
    lead = _lead(body)
    assert len(lead) <= 280 and lead.endswith(".")


@pytest.mark.parametrize("text", [
    "Koç Holding 2Ç26 yatırımcı sunumu yayımlandı",
    "Akbank analist toplantısı notları",
    "HSBC Akbank hedef fiyatını 87 TL'ye indirdi",
])
def test_analyst_and_ir_material_has_its_own_category(text):
    assert _match_keyword_category(text) == NewsCategory.ANALYST_IR


def test_broker_report_url_goes_to_analyst_category():
    from nlp.categorizer import categorize
    from scrapers.models import RawScrapedItem
    from config.constants import SourceType

    item = categorize(RawScrapedItem(
        source=SourceType.BIGPARA, ticker="AKBNK", raw_title="AKBNK YKBNK GARAN düzeltme",
        raw_url="https://www.bigpara.com/haberler/araci-kurum-raporlari/akbnk-ykbnk-garan-duzeltme-hsbc-t_ID1/",
    ))
    assert item.category == NewsCategory.ANALYST_IR


# --- From the 2026-10-06 audit log -------------------------------------------

from datetime import datetime as _dt  # noqa: E402


@pytest.mark.parametrize("title, dropped", [
    ("MERKEZ BANKASI HAZİRAN AYI FAİZ KARARI 2026 SON DAKİKA (PPK AÇIKLAMASI)", True),
    ("FED FAİZ KARARI EKİM AYI TOPLANTI TARİHİ | FED faiz kararı ne zaman açıklanacak?", True),
    ("TÜİK eylül enflasyonunu %29,73; ENAG %46,61 olarak açıkladı", False),
    ("Citi'den Merkez Bankası için faiz tahmini: Ekim ayında 100 baz puanlık indirim bekliyor", False),
])
def test_market_seo_pages_and_stale_months_dropped(title, dropped):
    from nlp.market_brief import _is_not_news
    assert _is_not_news(title, _dt(2026, 10, 6)) is dropped


def test_kap_registration_notice_dropped_by_summary():
    from nlp.categorizer import categorize
    from scrapers.models import RawScrapedItem
    from config.constants import SourceType

    assert categorize(RawScrapedItem(
        source=SourceType.KAP, ticker="GUBRF", raw_title="Genel Kurul İşlemlerine İlişkin Bildirim",
        raw_body_snippet="Genel Kurul Kararlarının Tescili Hk.",
    )) is None


@pytest.mark.parametrize("title", [
    "Altın düşerken beklenmedik gelişme! Fed'in faiz beklentisi değişti",
    "Altında 7 Ekim alarmı",
])
def test_market_clickbait_dropped(title):
    from nlp.market_brief import _is_not_news
    assert _is_not_news(title, _dt(2026, 10, 6))


def test_google_news_kap_relays_dropped():
    from nlp.relevance import filter_relevant
    from config.settings import TickerConfig
    from scrapers.models import RawScrapedItem
    from config.constants import SourceType

    relay = RawScrapedItem(source=SourceType.GOOGLE_NEWS, ticker="GUBRF",
                           raw_title="KAP GÜBRE FABRİKALARI T.A.Ş. GUBRF Genel Kurul İşlemlerine İlişkin Bildirim")
    news = RawScrapedItem(source=SourceType.GOOGLE_NEWS, ticker="GUBRF", raw_title="Gübretaş yeni tesis yatırımı açıkladı")
    kept = filter_relevant([relay, news], TickerConfig(symbol="GUBRF", name="Gübre Fabrikaları"))
    assert [i.raw_title for i in kept] == ["Gübretaş yeni tesis yatırımı açıkladı"]


def test_daily_close_is_not_an_index_change():
    from nlp.market_brief import _fix_index_change_labels
    from scrapers.models import NewsItem, SynthesizedSection
    from config.constants import SourceType

    close = NewsItem(ticker="PIYASA", title="Borsa günü düşüşle tamamladı: BIST 100 12.374 puanda",
                     source=SourceType.GOOGLE_NEWS, url="https://n.example/1")
    msci = NewsItem(ticker="PIYASA", title="MSCI, iki Türk şirketini endekse dahil etti",
                    source=SourceType.GOOGLE_NEWS, url="https://n.example/2")
    wrong = SynthesizedSection(subheading="Endeks değişikliği", narrative="x", source_urls=["https://n.example/1"])
    right = SynthesizedSection(subheading="Endeks değişikliği", narrative="y", source_urls=["https://n.example/2"])
    _fix_index_change_labels([wrong, right], [close, msci])
    assert wrong.subheading == "Dünkü seans"
    assert right.subheading == "Endeks değişikliği"


def test_circuit_breaker_is_noise():
    assert is_market_noise("YEOTK BORSA İSTANBUL BISTECH DEVRE KESİCİ UYGULAMASI (Pay Bazında Devre Kesici Bildirimi)")


# --- KAP forms as "Etiket: Değer" lines (2026-10-06 bond mix-up) -------------

_KAP_FORM_HTML = """
<html><body>
<table>
 <tr><td>Yapılan Açıklama Güncelleme mi?</td><td>Evet (Yes)</td></tr>
 <tr><td>İlgili Şirketler</td><td>[]</td></tr>
 <tr><td colspan="2">İlgili İhraç Tavanı Bilgileri</td></tr>
 <tr><td>Para Birimi</td><td>USD</td></tr>
 <tr><td>Tutar</td><td>4.000.000.000</td></tr>
 <tr><td colspan="2">İhraç Edilecek Sermaye Piyasası Aracı Bilgileri</td></tr>
 <tr><td>Satışı Gerçekleştirilen Nominal Tutar</td><td>50.000.000</td></tr>
 <tr><td class="content-en">Nominal Amount Sold</td><td class="content-en">50.000.000</td></tr>
 <tr><td>Döviz Cinsi</td><td>EUR</td></tr>
 <tr><td>Vade Tarihi</td><td>05.04.2028</td></tr>
</table>
<p>Bankamızca yurt dışında ihraç edilen tahvilin satışı tamamlanmış olup, işlemin detayları yukarıdaki tabloda yer almaktadır ve kamuoyuna duyurulur.</p>
</body></html>
"""


def test_kap_form_rows_become_label_value_lines():
    from utils.article_fetcher import extract_form_lines

    lines = extract_form_lines(_KAP_FORM_HTML)
    assert "Satışı Gerçekleştirilen Nominal Tutar: 50.000.000" in lines
    assert "Döviz Cinsi: EUR" in lines
    # The ceiling's currency stays under its own section header.
    assert lines.index("[İlgili İhraç Tavanı Bilgileri]") < lines.index("Para Birimi: USD") < lines.index(
        "[İhraç Edilecek Sermaye Piyasası Aracı Bilgileri]") < lines.index("Döviz Cinsi: EUR")
    assert not any("Güncelleme" in l or "İlgili Şirketler" in l or "Nominal Amount" in l for l in lines)


@pytest.mark.parametrize("title", [
    "KCHOL'de Global & BofA'dan Yüksek Hacimli Alımlar",
    "BofA borsada alıma döndü: YKBNK ve KCHOL listenin başında",
    "Emanet Koç Holding hisselerinde 1,4 milyarlık sır! 36 yıl sonra ortaya çıktı",
    "Koç Holding hisselerinin peşinde 33 yıl! Servetin değeri...",
])
def test_broker_flow_and_wealth_trivia_dropped(title):
    assert is_market_noise(title)


def test_numbers_and_labels_are_cleaned_up():
    from utils.formatting import polish_sections, turkish_numbers
    from scrapers.models import SynthesizedSection

    assert turkish_numbers("30,000,000,000 TRY tavan, 173,751,000 TRY satıldı") == "30 mlr TL tavan, 173,75 mn TL satıldı"
    assert turkish_numbers("1.713.897.558 TL ceza, vade 05.04.2028") == "1,71 mlr TL ceza, vade 05.04.2028"
    assert turkish_numbers("05 Oct 2026 tarihinde") == "5 Ekim 2026 tarihinde"

    merged = polish_sections([
        SynthesizedSection(subheading="Borçlanma", narrative="A", source_urls=["https://a.example/1"]),
        SynthesizedSection(subheading="Dava", narrative="B"),
        SynthesizedSection(subheading="borçlanma.", narrative="C", source_urls=["https://a.example/2"]),
    ])
    assert [s.subheading for s in merged] == ["Borçlanma", "Dava"]
    assert merged[0].narrative == "A\n\nC" and len(merged[0].source_urls) == 2


def test_kap_xbrl_rows_use_the_human_label():
    from utils.article_fetcher import extract_form_lines

    html = """<table>
      <tr><td>oda_UpdateAnnouncementFlag|</td><td>Yapılan Açıklama Güncelleme mi?</td><td>Evet (Yes)</td></tr>
      <tr><td>oda_ExplanationTextBlock|</td><td>Açıklamalar</td><td>Rekabet Kurulu soruşturması uzlaşmayla sonuçlanmıştır.</td></tr>
      <tr><td>[Türkçe]</td></tr><tr><td>Türkçe</td></tr>
    </table>"""
    assert extract_form_lines(html) == ["Açıklamalar: Rekabet Kurulu soruşturması uzlaşmayla sonuçlanmıştır."]


def test_bigpara_copy_of_a_read_kap_filing_is_dropped():
    from main import _drop_mirrored_filings
    from scrapers.models import RawScrapedItem
    from config.constants import SourceType

    kap = RawScrapedItem(source=SourceType.KAP, ticker="AKBNK", raw_title="Özel Durum Açıklaması (Genel)", raw_body_snippet="Açıklamalar: ...")
    mirror = RawScrapedItem(source=SourceType.BIGPARA, ticker="AKBNK", raw_title="AKBNK AKBANK T.A.Ş. (Özel Durum Açıklaması (Genel))",
                            raw_url="https://www.bigpara.com/haberler/kap-haberleri/akbnk-x_ID1/")
    news = RawScrapedItem(source=SourceType.BIGPARA, ticker="AKBNK", raw_title="Akbank'a yeni kredi", raw_url="https://www.bigpara.com/haberler/x")
    assert _drop_mirrored_filings([kap, mirror, news], "AKBNK") == [kap, news]


def test_kap_text_block_without_other_fields():
    """Özel Durum filing: only flags and the explanation block; the block
    must reach the model, not KAP's search help text or XBRL ids."""
    from utils.article_fetcher import structured_text

    html = """<html><body><div class="search-help">() Aşağıdaki öneriler, sadece endeks/pazar/dönem bilgisi girildiğinde listelenmektedir.</div>
    <table>
    <tr><td>oda_DelayedAnnouncementFlag|</td><td>Yapılan Açıklama Ertelenmiş Bir Açıklama mı?</td><td>Hayır (No)</td></tr>
    <tr><td>oda_ExplanationSection|</td><td>Açıklamalar<div class="content-en">Explanations</div></td></tr>
    </table>
    <div><span>oda_ExplanationTextBlock|</span><div><p>Rekabet Kurulu soruşturması 1.713.897.558 TL idari para cezası ile uzlaşmayla sonuçlanmıştır.</p>
    <div class="content-en"><p>The investigation was settled.</p></div></div></div>
    </body></html>"""
    text = structured_text(html, "https://www.kap.org.tr/tr/Bildirim/1")
    assert text == "Açıklama: Rekabet Kurulu soruşturması 1,71 mlr TL idari para cezası ile uzlaşmayla sonuçlanmıştır."


def test_polish_canonical_labels_isin_and_dates():
    from scrapers.models import SynthesizedSection
    from utils.formatting import polish_sections

    sections = [
        SynthesizedSection(subheading="Tahvil İhracı", narrative="173,75 mn TL tahvil sattı, vade 6 Dec 2027, ISIN TRSAKBKA2716.", source_urls=[]),
        SynthesizedSection(subheading="İzin", narrative="SPK ihraç izni verdi.", source_urls=[]),
        SynthesizedSection(subheading="İzin", narrative="Bakanlık tesis için çevre izni verdi.", source_urls=[]),
        SynthesizedSection(subheading="İş Yatırım", narrative="İş Yatırım 3Ç kar tahmini raporu yayımladı (rakamlar haberde yer almıyor).", source_urls=[]),
        SynthesizedSection(subheading="Ceza", narrative="1,71 mlr TL ceza ödendi.", source_urls=[]),
    ]
    out = {s.subheading: s.narrative for s in polish_sections(sections)}
    assert out["Borçlanma"] == "173,75 mn TL tahvil sattı, vade 6 Aralık 2027.\n\nSPK ihraç izni verdi."
    assert out["Analist raporu"] == "İş Yatırım 3Ç kar tahmini raporu yayımladı."
    assert out["Dava"] == "1,71 mlr TL ceza ödendi."
    assert out["İzin"] == "Bakanlık tesis için çevre izni verdi."  # not a borrowing permit


async def test_bing_fills_headline_only_google_items():
    from config.constants import SourceType
    from config.settings import BingNewsConfig, PolitenessConfig, TickerConfig
    from scrapers.bing_news_scraper import BingNewsScraper
    from scrapers.models import NewsItem

    rss = """<rss><channel><item><title>Gübre Fabrikaları hisse başına 3,81 TL brüt temettü ödeyecek</title>
    <link>https://www.bing.com/news/apiclick.aspx?ref=x&amp;url=https%3a%2f%2fwww.ornek.com%2fgubrf-temettu</link>
    <description>Gübre Fabrikaları 16 Ekim'de hisse başına 3,81 TL brüt nakit temettü ödeyecek.</description></item></channel></rss>"""

    class Response:
        text = rss
        status_code = 200

    scraper = BingNewsScraper(TickerConfig(symbol="GUBRF", name="Gübre Fabrikaları"), BingNewsConfig(), PolitenessConfig())

    async def fake_get(url, **kwargs):
        if "bing.com" in url:
            return Response()
        raise RuntimeError("no article fetch in tests")

    scraper._get = fake_get
    item = NewsItem(ticker="GUBRF", title="Gübre Fabrikaları hisse başına 3,81 TL brüt temettü ödeyecek",
                    url="https://news.google.com/rss/articles/abc", source=SourceType.GOOGLE_NEWS)
    await scraper.fill_headline_only([item])
    assert str(item.url) == "https://www.ornek.com/gubrf-temettu"
    assert "3,81 TL" in item.body_snippet


def test_ir_document_links_and_first_read_seeding():
    from config.settings import PolitenessConfig, TickerConfig
    from scrapers.ir_scraper import IrPageScraper, document_links
    from utils.memory import DigestMemory

    html = """<a href="/tr/images/pdf/akbank_corporate_presentation.pdf">Kurumsal sunum (Corporate presentation) İndir</a>
    <a href="/tr/finansal-raporlar/detay/2C26-Ozet-Sunum/597/1459/0">2Ç26 Konsolide Finansal Sonuçlarına İlişkin Özet Sunum</a>
    <a href="/en/images/pdf/x.pdf">English deck</a><a href="/tr/kariyer">Kariyer</a>
    <a href="https://yeo.com.tr/wp-content/uploads/2026/08/Yeo-Teknoloji-30.06.2026_BDR-4.pdf"></a>"""
    links = document_links(html, "https://www.akbankinvestorrelations.com/tr")
    assert list(links.values()) == [
        "Kurumsal sunum (Corporate presentation)",
        "2Ç26 Konsolide Finansal Sonuçlarına İlişkin Özet Sunum",
        "Yeo Teknoloji 30.06.2026 BDR 4",
    ]

    memory = DigestMemory()
    scraper = IrPageScraper(TickerConfig(symbol="AKBNK", name="Akbank"), memory, PolitenessConfig())
    first = dict(list(links.items())[:2])
    assert scraper._new_documents("p", first) == []  # first read: recorded only
    items = scraper._new_documents("p", links)
    assert [i.raw_url for i in items] == ["https://yeo.com.tr/wp-content/uploads/2026/08/Yeo-Teknoloji-30.06.2026_BDR-4.pdf"]
    assert items[0].raw_title.startswith("Akbank yatırımcı ilişkileri sayfasında yeni doküman")


def test_results_date_label_and_same_year_range():
    from scrapers.models import SynthesizedSection
    from utils.formatting import fix_year_ranges, polish_sections

    out = polish_sections([
        SynthesizedSection(subheading="Finansal sonuç", narrative="Akbank 3. çeyrek bilanço tarihini açıkladı.", source_urls=[]),
        SynthesizedSection(subheading="Bilanço tarihi", narrative="Bilanço 27 Ekim'de açıklanacak.", source_urls=[]),
    ])
    assert [s.subheading for s in out] == ["Bilanço tarihi"]
    assert fix_year_ranges("2026‑2026 3. çeyrek", "") == "2026 3. çeyrek"


def test_ir_discovery_helpers():
    from config.settings import TickerConfig
    from scrapers.ir_scraper import is_company_site, listing_pages, name_tokens, search_result_urls

    gubrf = TickerConfig(symbol="GUBRF", name="Gübre Fabrikaları")
    assert name_tokens(gubrf) == ["gubre", "gubrf"]
    assert is_company_site("https://www.gubretas.com.tr/yatirimci-iliskileri", gubrf)
    assert not is_company_site("https://www.kap.org.tr/tr/sirket-bilgileri/ozet/123", gubrf)
    assert is_company_site("https://www.koc.com.tr/yatirimci-iliskileri", TickerConfig(symbol="KCHOL", name="Koç Holding"))
    assert not is_company_site("https://tr.investing.com/equities/koc-holding", TickerConfig(symbol="KCHOL", name="Koç Holding"))

    rss = """<rss><channel><item><title>Akbank Yatırımcı İlişkileri</title><link>https://www.akbankinvestorrelations.com/tr</link></item>
    <item><title>Akbank</title><link>https://www.investing.com/akbank</link></item></channel></rss>"""
    assert search_result_urls(rss) == ["https://www.akbankinvestorrelations.com/tr", "https://www.investing.com/akbank"]
    ddg = '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fyeo.com.tr%2Fyatirimci-iliskileri-birimi%2F&rut=x">YEO</a>'
    assert search_result_urls(ddg) == ["https://yeo.com.tr/yatirimci-iliskileri-birimi/"]

    home = """<a href="/tr/yayinlar/sunumlar/Kurumsal-sunum/314/0/0">Kurumsal sunum</a>
    <a href="/tr/kariyer">Kariyer</a><a href="https://twitter.com/akbank">Sunumlar</a>
    <a href="/tr/images/pdf/x.pdf">Sunum</a><a href="/en/presentations">Presentations</a>"""
    assert listing_pages(home, "https://www.akbankinvestorrelations.com/tr") == [
        "https://www.akbankinvestorrelations.com/tr/yayinlar/sunumlar/Kurumsal-sunum/314/0/0"
    ]


def test_ir_home_from_kap_company_page():
    from config.settings import TickerConfig
    from scrapers.ir_scraper import company_website, ir_link_on_site

    akbank = TickerConfig(symbol="AKBNK", name="Akbank")
    kap_page = '<script>{"web":"https://www.kap.org.tr/tr/x","internetAdresi":"http://www.akbank.com"}</script>'
    assert company_website(kap_page, akbank) == "http://www.akbank.com/"
    site = """<a href="/tr/bireysel">Bireysel</a>
    <a href="https://www.akbankinvestorrelations.com/tr">Yatırımcı İlişkileri</a>"""
    assert ir_link_on_site(site, "https://www.akbank.com/", akbank) == "https://www.akbankinvestorrelations.com/tr"
    koc = TickerConfig(symbol="KCHOL", name="Koç Holding")
    assert ir_link_on_site('<a href="/yatirimci-iliskileri">Yatırımcı İlişkileri</a>', "https://www.koc.com.tr/", koc) == \
        "https://www.koc.com.tr/yatirimci-iliskileri"
    assert ir_link_on_site('<a href="https://investing.com/x">Investor</a>', "https://www.koc.com.tr/", koc) is None


def test_fine_paragraph_moved_out_of_borrowing():
    from scrapers.models import SynthesizedSection
    from utils.formatting import refile_paragraphs

    sections = refile_paragraphs([SynthesizedSection(
        subheading="Borçlanma",
        narrative="Rekabet Kurulu soruşturmasında 1,71 mlr TL idari para cezası ödendi.\n\nSPK TL borçlanma aracı ihraç yetkisi verdi.",
        source_urls=[],
    )])
    assert [(s.subheading, s.narrative) for s in sections] == [
        ("Dava", "Rekabet Kurulu soruşturmasında 1,71 mlr TL idari para cezası ödendi."),
        ("Borçlanma", "SPK TL borçlanma aracı ihraç yetkisi verdi."),
    ]
