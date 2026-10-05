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
