# Airport Noise — Edge Device Sistemi

Mevcut projenin yanına eklenen 3 dosya. GUI'ya dokunmadan çalışır.

## Dosyalar

| Dosya | Görev | Port |
|---|---|---|
| `firmware.py` | Mikrofon dinleme + BEATs inference + kayıt + telemetry | — |
| `web_server.py` | Lokal web dashboard (tablo + ses oynatıcı) | 5000 |
| `center_server.py` | Merkez sunucu simülasyonu | 8080 |

---

## Kurulum

```bash
pip install fastapi uvicorn requests
# Diğer bağımlılıklar (torch, torchaudio, sounddevice) zaten mevcut
```

---

## Çalıştırma (3 ayrı terminal)

**Terminal 1 — Merkez sunucu (önce başlat)**
```bash
python center_server.py
# → http://localhost:8080
```

**Terminal 2 — Lokal dashboard**
```bash
python web_server.py
# → http://localhost:5000
```

**Terminal 3 — Firmware**
```bash
# Proje kökünde çalıştır (BEATs.py ile aynı klasör)
python firmware.py

# Seçenekler:
python firmware.py --no-telemetry          # merkeze gönderme
python firmware.py --cpu                   # GPU yoksa
python firmware.py --list-devices          # mevcut mikrofonları listele
python firmware.py --device 1              # belirli mikrofon indeksi
python firmware.py --model-dir D:\models   # varsayılan
```

---

## Klasör Yapısı (otomatik oluşur)

```
Airport_Noise\
├── firmware.py
├── web_server.py
├── center_server.py
│
├── recordings\          ← firmware'in kaydettiği .wav dosyaları
│   └── rec_2026-06-11_13-15-00_AIRCRAFT.wav
│
├── device_data.db       ← lokal SQLite (firmware + web_server okur)
├── center_data.db       ← merkez SQLite (center_server yazar)
├── center_recordings\   ← merkeze gelen .wav kopyaları
│
└── firmware.log         ← firmware çalışma logu
```

---

## Nasıl Çalışır

```
[Mikrofon]
    │  sounddevice — 5s pencere, 1s hop
    ▼
[MicrophoneWorker Thread]
    │  audio_queue'ya koyar
    ▼
[InferenceWorker Thread]
    │  BEATs encoder (frozen) → MLP → softmax
    │  Majority voting (son 5 tahmin)
    │
    ├── AMBIENT / UNKNOWN → sessizce geç
    │
    └── Anlamlı ses (AIRCRAFT/SPEECH/TRAFFIC/WIND/OTHER)
            │
            ├── recordings/ → .wav kaydet
            ├── device_data.db → satır ekle
            └── HTTP POST → center_server (port 8080)

[web_server — port 5000]
    └── device_data.db okur → HTML tablo + ses oynatıcı

[center_server — port 8080]
    └── POST alır → center_data.db yazar → dashboard
```

---

## Parametreler (firmware.py)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--model-dir` | `D:\models` | BEATs checkpoint klasörü |
| `--beats-ckpt` | `BEATs_iter3_plus_AS2M.pt` | Encoder dosya adı |
| `--mlp-ckpt` | `beats_mlp.pt` | MLP dosya adı |
| `--audio-dir` | `recordings` | .wav kayıt klasörü |
| `--db-path` | `device_data.db` | Lokal veritabanı |
| `--center-url` | `http://localhost:8080/api/telemetry` | Merkez endpoint |
| `--no-telemetry` | — | Merkeze gönderimi kapat |
| `--device` | sistem varsayılanı | sounddevice cihaz indeksi |
| `--cpu` | — | GPU yerine CPU kullan |
| `--list-devices` | — | Mikrofonları listele ve çık |

---

## Notlar

- `firmware.py` proje kökünde çalıştırılmalı (`BEATs.py` ile aynı klasör)
- `CONFIDENCE_THR = 0.45` — noise_detector.py ile aynı eşik
- `MAJORITY_LEN = 5` — son 5 tahminin çoğunluğu (kararlılık için)
- Bir olay kaydedildikten sonra majority buffer temizlenir (aynı ses tekrar tetiklemez)
- `firmware.log` dosyası tüm çalışma geçmişini tutar