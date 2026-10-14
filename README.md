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

# Takip edilen tickerları `config/config.yaml` içindeki `tickers` bölümünden,
# alıcı listesini de aynı dosyadaki `recipients` bölümünden düzenleyin.

python -m config.settings   # sanity check: config doğru okunuyor mu?
DRY_RUN=true python main.py # scrape -> categorize -> dedup -> özet -> output/digest_{tarih}.html, mail göndermez
```

## Sabah bülteninin zamanında gelmesi

Workflow yalnızca dışarıdan tetikleniyor. GitHub'ın kendi zamanlayıcısı
(`schedule:`) 4-7 saat gecikmeyle çalıştığı için Ekim 2026'da kaldırıldı.
Tetikleyiciler cron-job.org'da:

| Saat (TR) | Gövde | Ne yapar |
|---|---|---|
| 07:00 | `{"ref":"master","inputs":{"dry_run":"false","ai_provider":"groq","require_kap":"true"}}` | KAP'a ulaşamazsa mail göndermez, 07:30'a bırakır |
| 07:30 | `{"ref":"master","inputs":{"dry_run":"false","ai_provider":"groq"}}` | 07:00 göndermediyse her durumda gönderir |

İstek: `POST https://api.github.com/repos/kacmazme22/followurfirm/actions/workflows/daily_digest.yml/dispatches`,
header'lar `Authorization: Bearer <token>` (fine-grained, yalnızca bu depo,
**Actions: Read and write**) ve `Accept: application/vnd.github+json`.

**Günde bir kez** kuralı: maili ilk gönderen çalışma bunu Actions cache'ine
yazar, aynı gün sonraki tetiklemeler birkaç saniyede çıkar. Elle `force`
seçeneğiyle yeniden gönderilebilir.

## Bülten hafızası

`utils/memory.py`. Her gerçek gönderimden sonra, bültendeki maddelerin
kaynak linkleri ve başlıkları `.digest-memory/memory.json` dosyasına yazılır
ve Actions cache'inde saklanır (14 gün).

- Ertesi gün aynı link ya da aynı haber başlığı (başka sitede olsa bile)
  modele gönderilmez. KAP form başlıkları her seferinde aynı olduğu için
  KAP bildirimleri yalnızca linkle eşleştirilir.
- Haber penceresi sabit 36 saat değil, **son gönderimden bu yana + 6 saat**
  (en az 36 saat, en fazla 7 gün). Mail gitmeyen bir günün haberleri
  ertesi güne kalır.
- Dry run hafızayı okur ama kaydetmez. Daha önce gönderilmiş haberleri de
  görmek için `ignore_memory` seçeneği kullanılır.

## Logdan inceleme

Her çalışmanın Actions logunda `audit` satırları var: modele giden her
haber (`girdi`) ve bültendeki her maddenin hangi haberlerden yazıldığı
(`çıktı`). Elenen haberler de gerekçesiyle loglanır (alaka filtresi,
hafıza). Bülteni log üzerinden kaynağına kadar takip etmek mümkündür.

## Maliyet ilkesi

Bu proje **sıfır ücretli servis** ilkesiyle inşa edilir:
- Kategorizasyon önce kural-tabanlı (bkz. `config/constants.py`), LLM'e
  yalnızca gerçekten gerektiğinde başvurulur.
- LLM gerektiğinde yalnızca free-tier sağlayıcılar (`AI_PROVIDER=groq`)
  kullanılır — asla token-başına ücretlendirilen bir key ile değil.
- E-posta gönderimi mevcut ücretsiz Gmail/SMTP hesabı üzerinden yapılır.

Detaylı mimari ve klasör yapısı için `PROJECT_TREE.md` dosyasına bakın.

## Proje Durumu (son güncelleme: Ekim 2026)

Kaynaklar: KAP (JSON API + bildirim detay sayfası, formlar kodla
"Etiket: Değer" satırlarına çevrilir), Bigpara, Bing News RSS (makale
metni), Google News RSS (yalnızca başlık). Özetleme Groq (ücretsiz katman).
Açık işler GitHub issue'larında takip ediliyor.

### Bilinen gotcha'lar / tuhaflıklar

- **KAP bazı günler 0 sonuç döner, bu normal**: THYAO/EREGL/GARAN/ASTOR'un her gün KAP bildirimi olmuyor. Test/doğrulama gerekiyorsa `scrapers/kap_scraper.py`'deki `LOOKBACK_DAYS` (varsayılan 2) geçici olarak büyütülüp gerçek veri bulunabilir (production değerini değiştirmeden).
- **Bigpara redirect + domain değişikliği**: `bigpara.hurriyet.com.tr` bazı isteklerde 301 ile (aynı URL'ye, muhtemelen cookie-handshake) veya `www.bigpara.com`'a yönlendirebiliyor. `BaseScraper` artık `follow_redirects=True` kullandığı için bu sorunsuz çözülüyor — ama linkler bazen `bigpara.com` domain'inde çıkabilir, bu bir hata değil.
- **`follow_redirects=True` fix'i (2026-08-03)**: Daha önce `BaseScraper._request()` redirect'leri takip etmiyordu, bu yüzden Bigpara sessizce 0 item + `error=None` dönüyordu (hata yok ama veri de yok — fark edilmesi zor bir "sessiz veri kaybı" idi). Artık merkezi olarak düzeltildi, tüm scraper'lara otomatik yayıldı.
- **Windows terminalinde Türkçe karakterler bozuk görünebilir** (`T�RK` gibi) — bu sadece konsol code page sorunu, gerçek veri (dosyaya yazıldığında/UTF-8 okunduğunda) doğru. Birden fazla kez codepoint seviyesinde doğrulandı.
- **`GoogleNewsRssConfig`'in kendi `politeness` alanı yok** (`config/settings.py`) — `main.py` şu an kütüphane varsayılanı `PolitenessConfig()`'i kullanıyor, şema değişikliği yapılmadı.
- **Zamanlama**: `config/config.yaml`'daki `schedule` alanı sadece bilgi amaçlıdır, hiçbir şeyi tetiklemez. Bülten saati cron-job.org'daki işlerden gelir.

### Nasıl devam edilir

Yeni bir oturumda önce bu bölümü oku, sonra `git log --oneline` ile son commit'lere bak. Her görev kendi commit'ine sahip (bkz. commit mesajları — `feat(scrapers): ...`, `feat(nlp): ...` gibi), büyük bir "tek seferde her şey" commit'i yok.
