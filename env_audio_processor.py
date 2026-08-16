"""
╔══════════════════════════════════════════════════════════════╗
║          AeroSonicDB  env_audio  →  AMBIENT Klipleri        ║
║          v2 — Segment-Index Tabanlı Filtre + SVM Doğrulama  ║
╚══════════════════════════════════════════════════════════════╝

Değişiklikler (v1 → v2):
  - parse_annotations() tamamen kaldırıldı (zaman tabanlı, çalışmıyordu)
  - load_segment_flags() eklendi → environment_class_mappings.csv'yi
    global segment indexi üzerinden okur (hata payı sıfır)
  - extract_ambient_clips() artık global_seg_offset + seg_flags kullanır
  - İsteğe bağlı SVM ikinci katman doğrulama eklendi (USE_SVM_VALIDATION)

CSV Formatı (environment_class_mappings.csv):
  720 satır × 6 sütun
  Satır 0: "0,1,2,3,4,5" → gerçek başlık, header=0 ile okunur
  Col 0, Col 1 → uçak indikatörü  (0=temiz, 1=uçak, "ignore"=kirli)
  Col 2-5       → ortam/rüzgar/trafik/konuşma (bu script için kullanılmaz)

Çalıştırma:
  python env_audio_processor.py           # segment filtresi ile
  python env_audio_processor.py --svm     # + SVM doğrulama katmanı
  python env_audio_processor.py --inspect # CSV'yi incele, klip çıkarma

Çıktı:
  Dataset_Airplane/env_clips/   ← temiz 5 sn WAV klipleri
  env_audio_manifest.csv        ← hangi dosyadan, kaç klip çıktı

Gereksinimler:
  pip install librosa soundfile pandas numpy tqdm scikit-learn
"""

import os
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

AIRPLANE_PATH  = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\Dataset_Airplane"
OUTPUT_DIR     = os.path.join(AIRPLANE_PATH, "env_clips")
MANIFEST_PATH  = os.path.join(AIRPLANE_PATH, "env_audio_manifest.csv")

SR             = 22050
CLIP_DURATION  = 5.0     # klip uzunluğu (sn)
HOP_DURATION   = 2.5     # hop uzunluğu  (sn) — CSV ile birebir eşleşmeli!
MIN_SILENCE_DB = -65.0   # bu dBFS altındaki klipler sessiz → atla

# ── SVM İkinci Katman Doğrulama ──────────────────────────────────
# True yapılırsa: temiz etiketli ama gerçekte uçak sesi içeren
# klipleri SVM ile tespit eder (annotation kaçıranları yakalar).
# ⚠ SVM 264-boyut özellik vektörü dataset_builder_v3.py ile AYNI olmalı.
USE_SVM_VALIDATION = False
SVM_MODEL_DIR  = os.path.join(
    r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise", "models"
)
SVM_CONF_THRESHOLD = 0.85   # AIRCRAFT olasılığı bu eşiğin üzerindeyse at

# ── Özellik Parametreleri (dataset_builder_v3.py ile eşleşmeli) ──
N_MFCC   = 40
N_FFT    = 2048
HOP_FFT  = 512

# ================================================================
# 📂  DOSYA KEŞFİ
# ================================================================

def find_env_audio_files(base: str) -> list[str]:
    """env_audio/ altındaki tüm WAV dosyalarını sıralı döndür."""
    candidates = [
        os.path.join(base, "env_audio", "env_audio"),
        os.path.join(base, "env_audio"),
    ]
    audio_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if audio_dir is None:
        print(f"[⚠ ] env_audio klasörü bulunamadı: {base}")
        return []

    files = sorted([
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(".wav")
    ])
    print(f"[env_audio] {len(files)} WAV dosyası bulundu: {audio_dir}")
    return files


def find_annotation_csv(base: str) -> str | None:
    """environment_class_mappings.csv dosyasını bul."""
    candidates = [
        os.path.join(base, "environment_class_mappings.csv"),
        os.path.join(base, "environment_mappings_raw.csv"),
    ]
    found = [p for p in candidates if os.path.exists(p)]
    if not found:
        print("[⚠ ] Annotation CSV bulunamadı — filtre devre dışı")
        return None
    print(f"[Annotation] {found[0]} kullanılıyor")
    return found[0]


# ================================================================
# 📋  SEGMENT FLAG OKUMA  (v2 — index tabanlı)
# ================================================================

def load_segment_flags(csv_path: str) -> np.ndarray:
    """
    environment_class_mappings.csv'yi global segment indexi olarak yükle.

    CSV formatı:
      Satır 0  → başlık: "0,1,2,3,4,5"
      Satır 1+ → her biri bir 2.5s hop penceresi
      Col 0, Col 1 → 0:temiz | 1:uçak | "ignore":kirli

    Returns:
      np.ndarray[bool] — True = temiz (AMBIENT'e dahil edilebilir)
    """
    try:
        # header=0: ilk satır sütun adı olarak yorumlanır
        df = pd.read_csv(csv_path, header=0, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]

        col0 = df.columns[0]   # uçak indikatörü 1
        col1 = df.columns[1]   # uçak indikatörü 2

        def _is_clean(val: str) -> bool:
            """0 → temiz; 1 veya "ignore" → kirli."""
            return str(val).strip().lower() == "0"

        flags = np.array(
            [_is_clean(row[col0]) and _is_clean(row[col1])
             for _, row in df.iterrows()],
            dtype=bool
        )

        n_total = len(flags)
        n_clean = int(flags.sum())
        n_dirty = n_total - n_clean
        print(f"[SegmentFlags] {n_total} segment okundu")
        print(f"               Temiz : {n_clean:4d}  (%{100*n_clean/n_total:.0f})")
        print(f"               Kirli : {n_dirty:4d}  (%{100*n_dirty/n_total:.0f})")
        return flags

    except Exception as e:
        print(f"[SegmentFlags] CSV okunamadı: {e}")
        print("[SegmentFlags] ⚠  Tüm segmentler temiz kabul ediliyor (filtre yok)")
        # Fallback: sınırsız True array → eski davranış
        return np.ones(999_999, dtype=bool)


# ================================================================
# 🤖  SVM DOĞRULAMA KATMANI  (isteğe bağlı)
# ================================================================

def load_svm_model() -> tuple | None:
    """
    SVM modeli ve label encoder'ı yükle.
    USE_SVM_VALIDATION=False ise None döndürür.
    """
    if not USE_SVM_VALIDATION:
        return None

    try:
        import pickle
        model_path   = os.path.join(SVM_MODEL_DIR, "best_model.pkl")
        encoder_path = os.path.join(SVM_MODEL_DIR, "label_encoder.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        # AIRCRAFT sınıf indexini bul
        aircraft_idx = list(encoder.classes_).index("AIRCRAFT")
        print(f"[SVM] Model yüklendi. AIRCRAFT idx={aircraft_idx}")
        return model, encoder, aircraft_idx

    except Exception as e:
        print(f"[SVM] Model yüklenemedi: {e} — SVM doğrulama atlanıyor")
        return None


def extract_features_for_svm(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    264-boyut özellik vektörü çıkar.

    ⚠  dataset_builder_v3.py'deki extract_features() ile BİREBİR AYNI olmalı!
       Aşağıdaki uygulama standart bir 264-boyut pipeline'ı temsil eder;
       kendi builder'ınla uyuşmuyorsa bu fonksiyonu override edin.

    Boyut dağılımı (264):
      MFCC         40 mean + 40 std = 80
      MFCC delta   40 mean + 40 std = 80
      MFCC delta2  40 mean + 40 std = 80
      Spectral     centroid/bw/rolloff/contrast × mean+std = 24 (12×2)
                   → toplam 264
    """
    # MFCC
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                      n_fft=N_FFT, hop_length=HOP_FFT)
    mfcc_d    = librosa.feature.delta(mfcc)
    mfcc_d2   = librosa.feature.delta(mfcc, order=2)

    feats = []
    for mat in [mfcc, mfcc_d, mfcc_d2]:
        feats.extend(mat.mean(axis=1).tolist())
        feats.extend(mat.std(axis=1).tolist())

    # Spectral features (12 adet × 2 istatistik = 24)
    spec_feats = {
        "centroid"  : librosa.feature.spectral_centroid(y=y, sr=sr,
                          n_fft=N_FFT, hop_length=HOP_FFT)[0],
        "bandwidth" : librosa.feature.spectral_bandwidth(y=y, sr=sr,
                          n_fft=N_FFT, hop_length=HOP_FFT)[0],
        "rolloff"   : librosa.feature.spectral_rolloff(y=y, sr=sr,
                          n_fft=N_FFT, hop_length=HOP_FFT)[0],
        "flatness"  : librosa.feature.spectral_flatness(y=y,
                          n_fft=N_FFT, hop_length=HOP_FFT)[0],
        "zcr"       : librosa.feature.zero_crossing_rate(y,
                          hop_length=HOP_FFT)[0],
        "rms"       : librosa.feature.rms(y=y, hop_length=HOP_FFT)[0],
    }
    for arr in spec_feats.values():
        feats.append(float(arr.mean()))
        feats.append(float(arr.std()))

    vec = np.array(feats, dtype=np.float32)

    # Boyut garantisi: eksikse sıfır doldur, fazlaysa kırp
    if len(vec) < 264:
        vec = np.pad(vec, (0, 264 - len(vec)))
    elif len(vec) > 264:
        vec = vec[:264]

    return vec


def svm_predicts_aircraft(chunk: np.ndarray, svm_bundle: tuple) -> bool:
    """
    SVM bu klibi AIRCRAFT olarak mı sınıflandırıyor?
    svm_bundle: (model, encoder, aircraft_idx)
    """
    model, encoder, aircraft_idx = svm_bundle
    feat = extract_features_for_svm(chunk).reshape(1, -1)

    try:
        proba = model.predict_proba(feat)[0]
        return bool(proba[aircraft_idx] >= SVM_CONF_THRESHOLD)
    except AttributeError:
        # predict_proba yoksa (linear SVC vb.) predict kullan
        pred = model.predict(feat)[0]
        pred_label = encoder.inverse_transform([pred])[0]
        return pred_label == "AIRCRAFT"


# ================================================================
# 🔇  YARDIMCI
# ================================================================

def rms_db(y: np.ndarray) -> float:
    """RMS ses seviyesi (dBFS)."""
    rms = np.sqrt(np.mean(y ** 2))
    return 20 * np.log10(rms + 1e-10)


# ================================================================
# ✂️  KLİP ÇIKARIM  (v2 — index tabanlı)
# ================================================================

def extract_ambient_clips(
    audio_path:       str,
    seg_flags:        np.ndarray,   # global segment flags (bool)
    global_seg_offset: int,          # bu dosya için CSV başlangıç satırı
    output_dir:       str,
    file_index:       int,
    svm_bundle:       tuple | None = None,
    sr:               int   = SR,
    clip_dur:         float = CLIP_DURATION,
    hop_dur:          float = HOP_DURATION,
) -> tuple[list[str], int]:
    """
    Bir env_audio dosyasından temiz AMBIENT klipleri çıkar.

    Algoritma:
      1. Dosyayı yükle
      2. hop_dur adımlarla ilerle (CSV ile birebir eşleşir)
      3. global_seg_offset + local_idx → seg_flags'e bak
         • False (kirli)  → atla
         • True  (temiz)  → 5 sn klip oluştur
      4. Sessiz → atla
      5. SVM_VALIDATION açıksa → SVM ile çift kontrol
      6. Kaydet

    Returns:
      (saved_paths, segments_consumed)
        segments_consumed: bu dosyada kaç hop/segment işlendi
        (global_seg_idx güncellemesi için)
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"  [!] Yüklenemedi {os.path.basename(audio_path)}: {e}")
        return [], 0

    clip_samples = int(clip_dur * sr)
    hop_samples  = int(hop_dur * sr)

    saved_paths       = []
    clip_idx          = 0
    segments_consumed = 0   # bu dosyadaki toplam hop sayısı

    start_sample = 0
    while start_sample + clip_samples <= len(y):
        global_idx = global_seg_offset + segments_consumed

        # ── Sınır kontrolü ──────────────────────────────────────
        if global_idx >= len(seg_flags):
            segments_consumed += 1
            start_sample += hop_samples
            continue
        
        is_clean = seg_flags[global_idx]
        # ── Ana filtre: segment flag ─────────────────────────────
        if is_clean:
            chunk = y[start_sample: start_sample + clip_samples]

            # Sessizlik kontrolü
            if rms_db(chunk) > MIN_SILENCE_DB:

                # İsteğe bağlı SVM doğrulama
                svm_rejected = False
                if svm_bundle is not None:
                    svm_rejected = svm_predicts_aircraft(chunk, svm_bundle)

                if not svm_rejected:
                    out_name = f"ambient_{file_index:04d}_{clip_idx:04d}.wav"
                    out_path = os.path.join(output_dir, out_name)
                    sf.write(out_path, chunk, sr, subtype="PCM_16")
                    saved_paths.append(out_path)
                    clip_idx += 1
        
        segments_consumed += 1
        start_sample      += hop_samples
        
    return saved_paths, segments_consumed


# ================================================================
# 🚀  ANA FONKSİYON
# ================================================================

def process_env_audio(use_svm: bool = False) -> pd.DataFrame:
    """
    Tüm env_audio dosyalarını işle, AMBIENT kliplerini kaydet.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dosyaları bul
    audio_files = find_env_audio_files(AIRPLANE_PATH)
    if not audio_files:
        print("[HATA] İşlenecek env_audio dosyası bulunamadı.")
        return pd.DataFrame()

    # 2. Segment flags yükle
    csv_path  = find_annotation_csv(AIRPLANE_PATH)
    seg_flags = load_segment_flags(csv_path) if csv_path else np.ones(999_999, dtype=bool)

    # 3. SVM modeli yükle (isteğe bağlı)
    global USE_SVM_VALIDATION
    if use_svm:
        USE_SVM_VALIDATION = True
    svm_bundle = load_svm_model()

    # 4. Özet
    print(f"\n[İşleme] {len(audio_files)} dosya → {OUTPUT_DIR}")
    print(f"         Klip: {CLIP_DURATION}sn | Hop: {HOP_DURATION}sn | "
          f"Min dB: {MIN_SILENCE_DB}")
    print(f"         SVM doğrulama: {'AÇIK' if svm_bundle else 'KAPALI'}\n")

    # SONRA — dosya başına CSV dilimi (DOĞRU)
    rows_per_file = len(seg_flags) // len(audio_files)
    print(f"[Mapping] {len(seg_flags)} satır ÷ {len(audio_files)} dosya = "
          f"{rows_per_file} satır/dosya  "
          f"(her dosyada ilk {rows_per_file * HOP_DURATION / 60:.1f} dk annotated)\n")

    manifest_rows = []
    total_clips   = 0

    for i, audio_path in enumerate(tqdm(audio_files, ncols=80, desc="  env_audio")):
        fname = os.path.basename(audio_path)

        # Bu dosyanın CSV dilimi: [i*rows_per_file : (i+1)*rows_per_file]
        file_flags = seg_flags[i * rows_per_file : (i + 1) * rows_per_file]

        clips, consumed = extract_ambient_clips(
            audio_path        = audio_path,
            seg_flags         = file_flags,   # ← dosyaya özel dilim
            global_seg_offset = 0,            # ← dilim zaten doğru başlıyor
            output_dir        = OUTPUT_DIR,
            file_index        = i,
            svm_bundle        = svm_bundle,
        )

        total_clips += len(clips)

        for cp in clips:
            manifest_rows.append({
                "path"  : cp,
                "label" : "AMBIENT",
                "source": "AeroSonicDB_env",
                "origin": fname,
            })

        n_annotated = int(file_flags.sum())
        tqdm.write(
            f"  {fname:25s} → {len(clips):3d} klip  "
            f"(annotated temiz: {n_annotated}/{len(file_flags)}, "
            f"tüketilen: {consumed})"
        )

    # 5. Manifest kaydet
    df = pd.DataFrame(manifest_rows)
    df.to_csv(MANIFEST_PATH, index=False)

    print("\n" + "=" * 60)
    print(f"  TAMAMLANDI")
    print(f"  Toplam AMBIENT klip : {total_clips}")
    print(f"  CSV kapsamı         : {len(seg_flags)} satır ({rows_per_file}/dosya)")
    print(f"  Çıktı klasörü       : {OUTPUT_DIR}")
    print(f"  Manifest            : {MANIFEST_PATH}")
    print("=" * 60)

    if not df.empty:
        print(f"\n  Ortalama klip/dosya : {total_clips / max(len(audio_files), 1):.1f}")
        print(f"  Toplam veri süresi  ≈ {total_clips * CLIP_DURATION / 60:.1f} dakika")
        print(f"\n  ⚠  Sonraki adım: cache/ sil → dataset_builder_v3.py çalıştır")

    return df


# ================================================================
# 🔍  HIZLI KONTROL
# ================================================================

def inspect_csv():
    """
    CSV formatını ve flag dağılımını ekrana yazdır.
    """
    csv_path = find_annotation_csv(AIRPLANE_PATH)
    if csv_path is None:
        return

    df = pd.read_csv(csv_path, header=0, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"\n{'='*55}")
    print(f"  CSV: {csv_path}")
    print(f"  {len(df)} satır × {len(df.columns)} sütun")
    print(f"  Sütunlar: {list(df.columns)}")
    print(f"\n  İlk 8 satır:")
    print(df.head(8).to_string())

    print(f"\n  Sütun değer dağılımı:")
    for col in df.columns:
        counts = df[col].value_counts()
        print(f"    Col '{col}': {counts.to_dict()}")

    # Flag istatistikleri
    col0 = df.columns[0]
    col1 = df.columns[1]
    clean_mask = (df[col0].str.strip() == "0") & (df[col1].str.strip() == "0")
    n_clean = clean_mask.sum()
    n_total = len(df)
    print(f"\n  Temiz segment (col0=0 ve col1=0) : {n_clean} / {n_total} "
          f"(%{100*n_clean/n_total:.0f})")
    print(f"  Kirli segment                     : {n_total-n_clean} / {n_total} "
          f"(%{100*(n_total-n_clean)/n_total:.0f})")
    print(f"{'='*55}")


# ================================================================
# 🏃  ÇALIŞTIR
# ================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="env_audio → AMBIENT klip çıkarıcı v2")
    p.add_argument("--inspect", action="store_true",
                   help="CSV formatını incele, klip çıkarma")
    p.add_argument("--svm", action="store_true",
                   help="SVM ikinci katman doğrulamayı etkinleştir")
    args = p.parse_args()

    if args.inspect:
        inspect_csv()
    else:
        manifest = process_env_audio(use_svm=args.svm)
        if not manifest.empty:
            print(f"\nÖnizleme (ilk 5 satır):")
            print(manifest.head().to_string())
