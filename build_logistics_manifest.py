"""
╔══════════════════════════════════════════════════════════════╗
║   LOGISTICS Windowing  —  manifest_v6.csv Oluşturucu        ║
╚══════════════════════════════════════════════════════════════╝

Ne yapar:
  1. Self_Data/LOGISTICS altındaki ham .mp3/.wav dosyalarını okur
  2. Her dosyayı çakışmasız 5sn'lik parçalara (chunk) böler
     (5sn'den kısa kalan kuyruk atılır, pad edilmez)
  3. Parçaları Self_Data/LOGISTICS/chunks/ altına .wav olarak kaydeder
     (22050 Hz, mono, PCM_16 — train_beats.py ile birebir aynı format)
  4. path,label kolonlarıyla cache/manifest_v6.csv'yi oluşturur
     (manifest_v4.csv / manifest_v5.csv'ye DOKUNULMAZ)

Not:
  - Sessizlik filtresi yok — 3 kaynak dosya da baştan sona aktif
    motor sesi içeriyor (RMS -11/-24 dB aralığında, hiç sessiz yok).
  - train_beats.py içindeki MANIFEST_CSV seçimi şu an sadece v5/v4'e
    bakıyor; manifest_v6.csv'yi kullanmak için o dosyada ayrıca
    güncelleme yapılması gerekecek (bu script onu değiştirmez).

Çalıştırma:
  python build_logistics_manifest.py
"""

import os
import warnings

import librosa
import soundfile as sf
import pandas as pd

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT  = r"C:\Airport_Noise_Detection-main"
RAW_DIR       = os.path.join(PROJECT_ROOT, "Self_Data", "LOGISTICS")
CHUNKS_DIR    = os.path.join(RAW_DIR, "chunks")
MANIFEST_OUT  = os.path.join(PROJECT_ROOT, "cache", "manifest_v6.csv")

# Ses parametreleri — train_beats.py / noise_detector.py ile AYNI olmalı
SR       = 22050
CLIP_DUR = 5.0
LABEL    = "LOGISTICS"

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac")


# ================================================================
# 📂  DOSYA KEŞFİ
# ================================================================

def find_raw_files(raw_dir: str) -> list:
    """Self_Data/LOGISTICS altındaki ham ses dosyalarını bul (chunks/ hariç)."""
    if not os.path.isdir(raw_dir):
        print(f"[HATA] Klasör bulunamadı: {raw_dir}")
        return []

    files = sorted([
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f))
        and f.lower().endswith(AUDIO_EXTENSIONS)
    ])
    print(f"[Bulundu] {len(files)} ham ses dosyası:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    return files


# ================================================================
# ✂️  CHUNK'LAMA
# ================================================================

def chunk_file(path: str, sr: int = SR, clip_dur: float = CLIP_DUR) -> list:
    """
    Bir ses dosyasını çakışmasız 5sn'lik parçalara böl.
    5sn'den kısa kalan kuyruk atılır (pad edilmez).
    """
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"  [!] Yüklenemedi {os.path.basename(path)}: {e}")
        return []

    clip_samples = int(clip_dur * sr)
    n_chunks = len(y) // clip_samples

    chunks = [y[i * clip_samples:(i + 1) * clip_samples] for i in range(n_chunks)]

    dropped_sec = (len(y) - n_chunks * clip_samples) / sr
    print(f"  {os.path.basename(path):50s}  {n_chunks:2d} chunk  "
          f"(kuyrukta atılan: {dropped_sec:.2f}sn)")
    return chunks


# ================================================================
# 🏃  ANA
# ================================================================

def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)

    print("=" * 60)
    print("  LOGISTICS Windowing  —  5sn çakışmasız chunk")
    print("=" * 60)

    raw_files = find_raw_files(RAW_DIR)
    if not raw_files:
        print("[HATA] Self_Data/LOGISTICS altında ses dosyası bulunamadı.")
        raise SystemExit(1)

    print(f"\n[İşleniyor]")
    manifest_rows = []
    counter = 1

    for raw_path in raw_files:
        chunks = chunk_file(raw_path)
        for chunk in chunks:
            out_name = f"logistics_{counter:04d}.wav"
            out_path = os.path.join(CHUNKS_DIR, out_name)
            sf.write(out_path, chunk, SR, subtype="PCM_16")
            manifest_rows.append({"path": out_path, "label": LABEL})
            counter += 1

    df = pd.DataFrame(manifest_rows)
    df.to_csv(MANIFEST_OUT, index=False)

    print(f"\n{'=' * 60}")
    print(f"  TAMAMLANDI")
    print(f"  Toplam chunk : {len(df)}")
    print(f"  Chunk klasörü: {CHUNKS_DIR}")
    print(f"  Manifest     : {MANIFEST_OUT}")
    print(f"{'=' * 60}")
    print(f"\n  Sonraki adım: train_beats.py içindeki MANIFEST_CSV seçimine")
    print(f"  manifest_v6.csv'yi eklemek ve BEATS_CLASSES listesine")
    print(f"  'LOGISTICS' sınıfını dahil etmek gerekecek (7 sınıf).")


if __name__ == "__main__":
    main()