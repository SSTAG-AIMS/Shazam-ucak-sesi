"""
╔══════════════════════════════════════════════════════════════════╗
║         Dataset Builder v4  —  OTHER sınıfı + GENERIC_AUDIO    ║
║   ESC-50  +  AeroSonicDB  +  env_audio  +  GENERIC_AUDIO_CLASSIFIER ║
╚══════════════════════════════════════════════════════════════════╝

v3'den fark:
  - OTHER sınıfı eklendi (6. sınıf)
  - C:\\GENERIC_AUDIO_CLASSIFIER klasöründen WAV dosyaları taranır
  - Klasör → sınıf eşleme tablosuna göre etiketleme yapılır
  - manifest_v4.csv çıktısı (manifest_v3.csv'ye dokunulmaz)
  - Sınıf dağılımı OTHER dahil terminale yazdırılır

Sınıflar: AIRCRAFT | AMBIENT | SPEECH | TRAFFIC | WIND | OTHER

Çalıştırma sırası:
  python env_audio_processor.py      → env_clips/ üretir   (zaten yapıldıysa gerek yok)
  python dataset_builder_v4.py       → manifest_v4.csv üretir
  python train_efficientnet.py       → MANIFEST_CSV'yi v4'e güncelle
"""

import os
import warnings
from pathlib import Path
from collections import Counter

import csv
import numpy as np
import pandas as pd
import librosa
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR  —  v3 ile senkronize
# ================================================================

AIRPLANE_PATH   = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\Dataset_Airplane"
ESC50_PATH      = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\Dataset_ESC50"
PROJECT_ROOT    = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
GENERIC_ROOT    = r"C:\Downloads_2\DATASET"          # YENİ veri seti
CACHE_DIR       = os.path.join(PROJECT_ROOT, "cache")
MANIFEST_OUT    = os.path.join(PROJECT_ROOT, "cache", "manifest_v4.csv")   # v3'e dokunmaz

# Canlı mikrofon klipler (D diskinde)
LIVE_CLIPS_DIR     = r"C:\Airport_Live_Clips"
APPROVED_MANIFEST  = os.path.join(LIVE_CLIPS_DIR, "approved_manifest.csv")
MANIFEST_OUT_V5    = os.path.join(PROJECT_ROOT, "cache", "manifest_v5.csv")

# Ses parametreleri — train_model ve noise_detector ile AYNI olmalı
SR        = 22050
CLIP_DUR  = 5.0
HOP_DUR   = 2.5
N_MFCC    = 40
N_FFT     = 2048
HOP_FFT   = 512

# ESC-50 kategori → etiket (v3 ile aynı)
ESC50_LABEL_MAP = {
    "airplane":      "AIRCRAFT",
    "helicopter":    "AIRCRAFT",
    "car_horn":      "TRAFFIC",
    "engine":        "TRAFFIC",
    "train":         "TRAFFIC",
    "siren":         "TRAFFIC",
    "wind":          "WIND",
    "rain":          "WIND",
    "thunderstorm":  "WIND",
    "sea_waves":     "WIND",
    "clapping":      "SPEECH",
    "laughing":      "SPEECH",
    "crying_baby":   "SPEECH",
    "crowd":         "SPEECH",
    "footsteps":     "SPEECH",
}

# AMBIENT env_audio manifest dosyası
ENV_MANIFEST = os.path.join(AIRPLANE_PATH, "env_audio_manifest.csv")

# ----------------------------------------------------------------
# GENERIC_AUDIO_CLASSIFIER  —  Klasör → Sınıf eşleme
# ----------------------------------------------------------------
# Alt klasör adı (küçük harf karşılaştırma yapılır)  →  hedef sınıf
GENERIC_LABEL_MAP = {
    # Vehicles
    "airplane":   "AIRCRAFT",
    "helicopter": "AIRCRAFT",
    "car":        "TRAFFIC",
    "bus":        "TRAFFIC",
    "truck":      "TRAFFIC",
    "bike":       "TRAFFIC",
    "bicycle":    "TRAFFIC",
    "train":      "TRAFFIC",
    # Environment
    "traffic":    "TRAFFIC",
    "wind":       "WIND",
    "crowd":      "SPEECH",
    "rainfall":   "AMBIENT",
    "office":     "AMBIENT",
    "military":   "OTHER",
    # Animals → OTHER  (tüm alt klasörler)
    "cats":       "OTHER",
    "cat":        "OTHER",
    "dogs":       "OTHER",
    "dog":        "OTHER", 
    "elephant":   "OTHER",
    "horse":      "OTHER",
    "lions":      "OTHER",
    "lion":       "OTHER",
    # Birds → OTHER  (tüm alt klasörler)
    "crows":      "OTHER",
    "crow":       "OTHER",
    "parrot":     "OTHER",
    "peacock":    "OTHER",
    "sparrow":    "OTHER",
}


# ================================================================
# 🔊  VERİ YÜKLEME  —  v3'ten gelen fonksiyonlar (değişmedi)
# ================================================================

def load_esc50_records() -> list[dict]:
    csv_path = os.path.join(ESC50_PATH, "esc50.csv")
    if not os.path.exists(csv_path):
        print(f"[⚠ ] ESC-50 CSV bulunamadı: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    records, missing = [], 0

    for _, row in df.iterrows():
        cat = str(row.get("category", ""))
        if cat not in ESC50_LABEL_MAP:
            continue
        fname = row["filename"]
        candidates = [
            os.path.join(ESC50_PATH, "audio", fname),
            os.path.join(ESC50_PATH, "audio", "audio", fname),
            os.path.join(ESC50_PATH, "audio", "audio", "44100", fname),
            os.path.join(ESC50_PATH, "audio", "audio", "16000", fname),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            missing += 1
            continue
        records.append({"path": path, "label": ESC50_LABEL_MAP[cat], "source": "ESC50"})

    print(f"[ESC-50]           {len(records):5d} örnek  (bulunamayan: {missing})")
    return records


def load_aerosonic_records() -> list[dict]:
    roots = [
        os.path.join(AIRPLANE_PATH, "audio", "audio"),
        os.path.join(AIRPLANE_PATH, "audio"),
    ]
    audio_root = next((p for p in roots if os.path.isdir(p)), None)
    if audio_root is None:
        print(f"[⚠ ] AeroSonicDB audio bulunamadı")
        return []

    records = []
    for root, _, files in os.walk(audio_root):
        for f in files:
            if f.lower().endswith(".wav"):
                records.append({
                    "path":   os.path.join(root, f),
                    "label":  "AIRCRAFT",
                    "source": "AeroSonic",
                })

    print(f"[AeroSonicDB]      {len(records):5d} örnek  (tümü → AIRCRAFT)")
    return records


def load_ambient_records() -> list[dict]:
    if not os.path.exists(ENV_MANIFEST):
        print(f"[⚠ ] AMBIENT manifest bulunamadı: {ENV_MANIFEST}")
        print(f"      Önce: python env_audio_processor.py")
        return []

    df = pd.read_csv(ENV_MANIFEST)
    records = []
    for _, row in df.iterrows():
        if os.path.exists(str(row["path"])):
            records.append({
                "path":   row["path"],
                "label":  "AMBIENT",
                "source": "AeroSonicDB_env",
            })

    print(f"[env_audio]        {len(records):5d} AMBIENT örnek yüklendi")
    return records


# ================================================================
# 🆕  GENERIC_AUDIO_CLASSIFIER  —  YENİ
# ================================================================

def load_generic_records() -> list[dict]:
    """
    C:\\GENERIC_AUDIO_CLASSIFIER altındaki tüm WAV dosyalarını tara.
    Her dosyanın doğrudan üst klasör adını GENERIC_LABEL_MAP ile eşle.
    Eşleşmeyen klasörler atlanır ve raporlanır.

    Klasör yapısı:
        GENERIC_AUDIO_CLASSIFIER/
        ├── Animals/CATS/*.wav   → OTHER
        ├── Birds/CROWS/*.wav    → OTHER
        ├── Environment/WIND/*.wav → WIND
        └── Vehicles/airplane/*.wav → AIRCRAFT
    """
    if not os.path.isdir(GENERIC_ROOT):
        print(f"[⚠ ] GENERIC_AUDIO_CLASSIFIER bulunamadı: {GENERIC_ROOT}")
        return []

    records    = []
    skipped    = Counter()    # eşleşmeyen klasör adları
    per_label  = Counter()    # sınıf başına dosya sayısı

    for root, _, files in os.walk(GENERIC_ROOT):
        wav_files = [f for f in files if f.lower().endswith(".wav")]
        if not wav_files:
            continue

        # En alt klasör adını al (büyük/küçük harf duyarsız)
        folder_name = Path(root).name.lower()
        label = GENERIC_LABEL_MAP.get(folder_name)

        if label is None:
            skipped[folder_name] += len(wav_files)
            continue

        for f in wav_files:
            full_path = os.path.join(root, f)
            records.append({
                "path":   full_path,
                "label":  label,
                "source": "GENERIC_AUDIO",
            })
            per_label[label] += 1

    # Özet
    total = sum(per_label.values())
    print(f"[GENERIC_AUDIO]    {total:5d} örnek yüklendi")
    for lbl, cnt in sorted(per_label.items()):
        print(f"  ↳ {lbl:10s}: {cnt:5d}")

    if skipped:
        print(f"  [ATLANAN klasörler — GENERIC_LABEL_MAP'te yok]")
        for folder, cnt in sorted(skipped.items()):
            print(f"    '{folder}' → {cnt} dosya atlandı")

    return records

def load_live_records() -> list[dict]:
    """
    C:\\Airport_Live_Clips\\approved_manifest.csv içindeki
    onaylanmış canlı mikrofon kliplerini yükler.
    """
    if not os.path.exists(APPROVED_MANIFEST):
        print(f"[live_clips]       Approved manifest bulunamadı: {APPROVED_MANIFEST}")
        print(f"                   GUI'den klip toplayıp onayladıktan sonra çalıştır.")
        return []

    records = []
    missing = 0
    with open(APPROVED_MANIFEST, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "approved":
                continue
            path = row.get("clip_path", "")
            if not os.path.exists(path):
                missing += 1
                continue
            records.append({
                "path":   path,
                "label":  row["corrected_label"],
                "source": "LIVE_MIC",
            })

    print(f"[live_clips]       {len(records):5d} onaylı klip  "
          f"(bulunamayan: {missing})")

    # Sınıf dağılımı
    dist = Counter(r["label"] for r in records)
    for lbl, cnt in sorted(dist.items()):
        print(f"  ↳ {lbl:10s}: {cnt:4d}")

    return records

# ================================================================
# 📊  DAĞILIM RAPORU
# ================================================================

def show_distribution(records: list[dict]):
    dist = Counter(r["label"] for r in records)
    print("\n── Sınıf Dağılımı (6 Sınıf) ────────────────────────────")
    CLASS_ORDER = ["AIRCRAFT", "AMBIENT", "SPEECH", "TRAFFIC", "WIND", "OTHER"]
    for lbl in CLASS_ORDER:
        cnt = dist.get(lbl, 0)
        bar = "█" * (cnt // 20)
        print(f"  {lbl:10s} {cnt:6d}  {bar}")
    # Beklenmeyen sınıflar varsa göster
    for lbl, cnt in sorted(dist.items()):
        if lbl not in CLASS_ORDER:
            print(f"  {lbl:10s} {cnt:6d}  [!BEKLENMEDİK]")
    print(f"\n  TOPLAM: {sum(dist.values())}\n")


# ================================================================
# 🎛️  ÖZELLİK ÇIKARIM  —  v3 ile AYNI (264 boyut)
# ================================================================

def load_audio_fixed(path: str, sr: int = SR,
                     duration: float = CLIP_DUR) -> np.ndarray | None:
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=duration + 0.5)
        target = int(sr * duration)
        if len(y) >= target:
            start = (len(y) - target) // 2
            y = y[start:start + target]
        else:
            y = np.pad(y, (0, target - len(y)))
        # RMS Normalizasyonu — mikrofon seviye farkını dengele
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 1e-8:
            y = y * (0.1 / rms)
        y = np.clip(y, -1.0, 1.0)

        return y.astype(np.float32)
    
    except Exception as e:
        print(f"  [!] {os.path.basename(path)}: {e}")
        return None


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """264 boyutlu özellik vektörü — v3 ile birebir aynı."""
    features = []

    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=N_FFT, hop_length=HOP_FFT)
    d_mfcc  = librosa.feature.delta(mfcc)
    d2_mfcc = librosa.feature.delta(mfcc, order=2)
    for m in [mfcc, d_mfcc, d2_mfcc]:
        features.extend([m.mean(axis=1), m.std(axis=1)])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_FFT)
    features.append(chroma.mean(axis=1))

    sc  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sf  = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_FFT)
    for feat in [sc, sb, sr_, sf]:
        features.extend([feat.mean(axis=1), feat.std(axis=1)])

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_FFT)
    rms = librosa.feature.rms(y=y, hop_length=HOP_FFT)
    for feat in [zcr, rms]:
        features.extend([feat.mean(axis=1), feat.std(axis=1)])

    return np.concatenate(features, axis=0).astype(np.float32)


def build_feature_matrix(records: list[dict],
                          cache_path: str | None = None) -> tuple:
    """Özellik matrisi oluştur. Önbellekten yükler veya hesaplar."""
    if cache_path and os.path.exists(cache_path):
        print(f"[Önbellekten] {cache_path}")
        d = joblib.load(cache_path)
        return d["X"], d["y"], d["paths"]

    X, y, paths = [], [], []
    for rec in tqdm(records, ncols=80, desc="  Özellik"):
        audio = load_audio_fixed(rec["path"])
        if audio is None:
            continue
        feat = extract_features(audio)
        X.append(feat)
        y.append(rec["label"])
        paths.append(rec["path"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump({"X": X, "y": y, "paths": paths}, cache_path)
        print(f"  → Önbellek kaydedildi: {cache_path}")

    print(f"  X.shape = {X.shape}")
    return X, y, paths


# ================================================================
# 🏃  ANA
# ================================================================

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("  Dataset Builder  —  AMBIENT + OTHER + GENERIC + LIVE")
    print("=" * 60)

    # ── Temel veri seti (v4 ile aynı) ────────────────────────────
    base_records = (
        load_esc50_records()     +
        load_aerosonic_records() +
        load_ambient_records()   +
        load_generic_records()
    )

    # ── Canlı mikrofon klipler ────────────────────────────────────
    live_records = load_live_records()

    records = base_records + live_records

    if not records:
        print("[HATA] Hiç veri yüklenemedi.")
        raise SystemExit(1)

    show_distribution(records)

    # YENİ
    if live_records:
        cache_path = os.path.join(CACHE_DIR, "features_v5.pkl")
    else:
        cache_path = os.path.join(CACHE_DIR, "features_v4.pkl")
    X, y, paths = build_feature_matrix(records, cache_path=cache_path)

    # ── Manifest: live var mı yok mu? ────────────────────────────
    manifest = pd.DataFrame({"path": paths, "label": y})

    if live_records:
        manifest.to_csv(MANIFEST_OUT_V5, index=False)
        print(f"\n  Live klip içeriyor → manifest_v5.csv kaydedildi")
        print(f"  Manifest: {MANIFEST_OUT_V5}")
        print(f"\n  ✅ train_efficientnet.py içindeki MANIFEST_CSV'yi")
        print(f"     '{MANIFEST_OUT_V5}' olarak güncelle, ardından eğit.")
    else:
        manifest.to_csv(MANIFEST_OUT, index=False)
        print(f"\n  Live klip yok → manifest_v4.csv kaydedildi (değişmedi)")
        print(f"  Manifest: {MANIFEST_OUT}")