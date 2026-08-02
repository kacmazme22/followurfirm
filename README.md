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
python main.py              # tam pipeline'ı çalıştırır (henüz iskelet halinde)
```

## Maliyet ilkesi

Bu proje **sıfır ücretli servis** ilkesiyle inşa edilir:
- Kategorizasyon önce kural-tabanlı (bkz. `config/constants.py`), LLM'e
  yalnızca gerçekten gerektiğinde başvurulur.
- LLM gerektiğinde yalnızca free-tier sağlayıcılar (`AI_PROVIDER=groq`)
  kullanılır — asla token-başına ücretlendirilen bir key ile değil.
- E-posta gönderimi mevcut ücretsiz Gmail/SMTP hesabı üzerinden yapılır.

Detaylı mimari ve klasör yapısı için `PROJECT_TREE.md` dosyasına bakın.
