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
