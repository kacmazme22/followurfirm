# FollowUrFirm

Otomatik, AI-destekli günlük finansal bülten sistemi. BIST hisselerini
(THYAO, EREGL, GARAN, ASTOR ...) izler, KAP açıklamalarını ve haber
kaynaklarını toplayıp kategorize edilmiş bir HTML e-posta bülteni üretir.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env dosyasını doldurun (SMTP App Password, opsiyonel Groq key)

# config/config.yaml içinde takip edilen tickerlar ve alıcı listesini düzenleyin

python -m config.settings   # sanity check: config doğru okunuyor mu?
python main.py              # scrape -> categorize -> dedup -> render, output/digest_{tarih}.html yazar (email gönderimi henüz yok)
```

## Maliyet ilkesi

Bu proje **sıfır ücretli servis** ilkesiyle inşa edilir:
- Kategorizasyon önce kural-tabanlı (bkz. `config/constants.py`), LLM'e
  yalnızca gerçekten gerektiğinde başvurulur.
- LLM gerektiğinde yalnızca free-tier sağlayıcılar (`AI_PROVIDER=groq`)
  kullanılır — asla token-başına ücretlendirilen bir key ile değil.
- E-posta gönderimi mevcut ücretsiz Gmail/SMTP hesabı üzerinden yapılır.

Detaylı mimari ve klasör yapısı için `PROJECT_TREE.md` dosyasına bakın.

## Proje Durumu (son güncelleme: 2026-08-03)

### Tamamlanan (test edilmiş, gerçek ağ istekleriyle doğrulanmış)

- **`scrapers/base.py`** — `BaseScraper` ABC: politeness (rate-limit + User-Agent), in-run cache, `run()` hiç exception fırlatmıyor (`ScraperResult{items, error}` dönüyor). `_get`/`_post` artık `follow_redirects=True` kullanıyor (bkz. "Bilinen gotcha'lar").
- **`scrapers/kap_scraper.py`** — KAP'ın undocumented JSON API'si (`/tr/api/disclosure/members/byCriteria`, POST). İki kullanım şekli: `fetch_raw()` (tek ticker, `run()` üzerinden) ve `fetch_all_raw(tickers)` (tüm şirketleri **tek POST**'ta çekip client-side `stockCodes`'a göre gruplar — `main.py` bunu kullanıyor, ticker başına ayrı istek atmıyor).
- **`scrapers/bigpara_scraper.py`** — Bigpara hisse-haberleri sayfası, düz server-rendered HTML (JS-render yok), `BeautifulSoup` ile parse. URL: `{base_url}/borsa/hisse-fiyatlari/{ticker}-detay/hisse-haberleri/`.
- **`scrapers/google_news_scraper.py`** — Google News RSS (`feedparser`), ticker+şirket adıyla query kuruyor.
- **`nlp/categorizer.py`** — Kural-tabanlı (LLM yok) kategorizasyon, `KAP_KEYWORD_CATEGORY_MAP` + boilerplate filtresi.
- **`nlp/dedup.py`** — Cross-source dedup (`SOURCE_PRIORITY`: Google News/Bigpara > KAP), elenen KAP linki `NewsItem.related_kap_url`'de korunuyor.
- **`nlp/providers/`** — `SummarizerProvider` arayüzü + `NoopProvider` (gerçek no-op) + `factory.py`. **Groq implement edilmedi** (`AI_PROVIDER=groq` şu an `NotImplementedError` fırlatır — bilinçli olarak, ayrı bir görev).
- **`templates/email_base.html` + `styles.py`** — Jinja2, inline-CSS (email client uyumluluğu için `<style>` bloğu yok), KapMail renk paleti.
- **`main.py`** — Orchestrator: `run_pipeline()` (scrape → categorize → dedup → noop provider) + `render_digest_html()`. Bir ticker tamamen patlarsa diğerlerini etkilemez. **Email gönderimi yok** — sadece `output/digest_{tarih}.html` yazıyor.

### Sırada (henüz yazılmadı)

| Dosya | Not |
|---|---|
| `utils/email_sender.py` | SMTP gönderim (Gmail App Password, `.env`'de zaten alanları var) |
| `.github/workflows/daily_digest.yml` | Cron workflow, `main.py`'yi günlük çalıştıracak |
| `nlp/providers/groq_provider.py` | Bilinçli olarak ertelendi, ayrı görev |
| `utils/cache.py`, `rate_limit.py` | `PROJECT_TREE.md`'de var ama henüz gerekmedi, `BaseScraper` politeness+cache'i şimdilik yeterli |

### Bilinen gotcha'lar / tuhaflıklar

- **KAP bazı günler 0 sonuç döner, bu normal**: THYAO/EREGL/GARAN/ASTOR'un her gün KAP bildirimi olmuyor. Test/doğrulama gerekiyorsa `scrapers/kap_scraper.py`'deki `LOOKBACK_DAYS` (varsayılan 2) geçici olarak büyütülüp gerçek veri bulunabilir (production değerini değiştirmeden).
- **Bigpara redirect + domain değişikliği**: `bigpara.hurriyet.com.tr` bazı isteklerde 301 ile (aynı URL'ye, muhtemelen cookie-handshake) veya `www.bigpara.com`'a yönlendirebiliyor. `BaseScraper` artık `follow_redirects=True` kullandığı için bu sorunsuz çözülüyor — ama linkler bazen `bigpara.com` domain'inde çıkabilir, bu bir hata değil.
- **`follow_redirects=True` fix'i (2026-08-03)**: Daha önce `BaseScraper._request()` redirect'leri takip etmiyordu, bu yüzden Bigpara sessizce 0 item + `error=None` dönüyordu (hata yok ama veri de yok — fark edilmesi zor bir "sessiz veri kaybı" idi). Artık merkezi olarak düzeltildi, tüm scraper'lara otomatik yayıldı.
- **Windows terminalinde Türkçe karakterler bozuk görünebilir** (`T�RK` gibi) — bu sadece konsol code page sorunu, gerçek veri (dosyaya yazıldığında/UTF-8 okunduğunda) doğru. Birden fazla kez codepoint seviyesinde doğrulandı.
- **`GoogleNewsRssConfig`'in kendi `politeness` alanı yok** (`config/settings.py`) — `main.py` şu an kütüphane varsayılanı `PolitenessConfig()`'i kullanıyor, şema değişikliği yapılmadı.

### Nasıl devam edilir

Yeni bir oturumda önce bu bölümü oku, sonra `git log --oneline` ile son commit'lere bak. Her görev kendi commit'ine sahip (bkz. commit mesajları — `feat(scrapers): ...`, `feat(nlp): ...` gibi), büyük bir "tek seferde her şey" commit'i yok.
