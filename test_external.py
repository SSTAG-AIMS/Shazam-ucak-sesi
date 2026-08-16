"""
╔══════════════════════════════════════════════════════════════╗
║         External Test  —  AirportCNN + EfficientNet-B0      ║
║   Dış dataset üzerinde model performans değerlendirmesi      ║
╚══════════════════════════════════════════════════════════════╝

Klasör yapısı:
  Airport_Noise/
  ├── test_external.py
  ├── models/
  │   ├── best_cnn.pt
  │   ├── cnn_label_encoder.pkl
  │   ├── best_efficientnet.pt              (opsiyonel)
  │   └── efficientnet_label_encoder.pkl    (opsiyonel)
  └── Test_Folder/
      ├── aircraft-test/    .wav dosyaları  (uçak, test split)
      ├── aircraft-train/   .wav dosyaları  (uçak, train split)
      ├── negative-test/    .wav dosyaları  (negatif, test split)
      ├── negative-train/   .wav dosyaları  (negatif, train split)
      └── labels.csv        filename, class, duration, sample_rate, dtype, split

Dosya tarama stratejisi:
  1. Dört klasör fiziksel olarak taranır
     (aircraft-test, aircraft-train, negative-test, negative-train).
  2. CSV'de eşleşme varsa label CSV'den alınır.
  3. CSV'de eşleşme yoksa label klasör adından çıkarsanır
     ("aircraft-*" → "aircraft", "negative-*" → "negative").
  4. CSV'de olup fiziksel olarak bulunmayan dosyalar sessizce atlanır.

Sınıf eşleştirmesi (5-sınıf → 2-sınıf):
  "AIRCRAFT" → "aircraft"
  diğer 4    → "negative"

Confidence threshold:
  AIRCRAFT softmax olasılığı eşiğin üstündeyse "aircraft" denir.
  Varsayılan: 0.13

Çalıştırma:
  python test_external.py
  python test_external.py --threshold 0.3    (eşiği düşür, recall artır)
  python test_external.py --threshold 0.7    (eşiği yükselt, precision artır)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio.transforms as TA
import torchvision.transforms as TV
import torchvision.models as tvm

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, recall_score, precision_score
)

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT   = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
TEST_FOLDER    = os.path.join(PROJECT_ROOT, "Test_Folder")
LABELS_CSV     = os.path.join(TEST_FOLDER, "labels.csv")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
OUT_DIR        = os.path.join(PROJECT_ROOT, "outputs", "external_test")

SR       = 22050
N_MELS   = 128
N_FFT    = 2048
HOP_FFT  = 512
CLIP_DUR = 5.0
HOP_DUR  = 2.5

CONF_THRESHOLD = 0.58

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

AIRCRAFT_CLASS = "AIRCRAFT"

# Hangi klasörler taranacak ve her birinin varsayılan etiketi
SCAN_SUBDIRS = {
    "aircraft-test":  "aircraft",
    "aircraft-train": "aircraft",
    "negative-test":  "negative",
    "negative-train": "negative",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# 🏗️  MODEL MİMARİLERİ
# ================================================================

class AirportCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(256, n_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class EfficientNetAirport(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        bb = tvm.efficientnet_b0(weights=None)
        self.features   = bb.features
        self.avgpool    = bb.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(1280, 512),
            nn.ReLU(True), nn.Dropout(0.3), nn.Linear(512, n_classes),
        )
    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


# ================================================================
# 📂  DOSYA TARAMA
# ================================================================

def scan_files(test_folder: str, labels_csv: str) -> pd.DataFrame:
    """
    Dört klasörü (aircraft-test/train + negative-test/train) fiziksel tara.
    CSV eşleşmesi varsa label CSV'den al.
    CSV'de yoksa klasör adından çıkarsa (aircraft-* → aircraft, negative-* → negative).
    """
    csv_map = {}
    if os.path.exists(labels_csv):
        df_csv = pd.read_csv(labels_csv)
        for _, row in df_csv.iterrows():
            csv_map[row["filename"]] = row
        print(f"  CSV: {len(df_csv)} kayıt okundu.")
    else:
        print(f"  ⚠  labels.csv bulunamadı → etiket klasör adından çıkarılacak.")

    records = []
    for subdir, fallback_label in SCAN_SUBDIRS.items():
        folder = os.path.join(test_folder, subdir)
        if not os.path.isdir(folder):
            print(f"  ⚠  Klasör bulunamadı: {folder}")
            continue
        wavs = sorted(Path(folder).glob("*.wav"))
        print(f"  {subdir}: {len(wavs)} .wav")
        for wp in wavs:
            fname = wp.name
            row   = csv_map.get(fname)
            if row is not None:
                label = str(row["class"]).lower().strip()
            else:
                # CSV'de yoksa klasör adından çıkarsa
                label = fallback_label
            records.append({
                "filepath":   str(wp),
                "filename":   fname,
                "true_label": label,
                "duration":   row["duration"] if row is not None else "?",
                "split":      subdir,
                "in_csv":     fname in csv_map,
            })

    return pd.DataFrame(records)


# ================================================================
# 🔊  SES İŞLEME
# ================================================================

def load_audio(path: str) -> np.ndarray:
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
        return y.astype(np.float32)
    except Exception as e:
        print(f"  [HATA] {path}: {e}")
        return np.zeros(int(SR * CLIP_DUR), dtype=np.float32)


def extract_windows(y: np.ndarray) -> list:
    clip = int(SR * CLIP_DUR)
    hop  = int(SR * HOP_DUR)
    if len(y) <= clip:
        return [np.pad(y, (0, clip - len(y)))]
    wins, s = [], 0
    while s + clip <= len(y):
        wins.append(y[s: s + clip]); s += hop
    if s < len(y):
        last = y[s:]
        wins.append(np.pad(last, (0, clip - len(last))))
    return wins


# ================================================================
# 🤖  MODEL YÜKLEYİCİLER
# ================================================================

def load_cnn():
    p = os.path.join(MODELS_DIR, "best_cnn.pt")
    if not os.path.exists(p):
        print("  [CNN] best_cnn.pt bulunamadı → atlanıyor"); return None, None
    ckpt = torch.load(p, map_location=device, weights_only=False)
    m = AirportCNN(ckpt.get("n_classes", 5))
    m.load_state_dict(ckpt["model_state"]); m.eval().to(device)
    le = joblib.load(os.path.join(MODELS_DIR, "cnn_label_encoder.pkl"))
    print(f"  [CNN] ✅ ep{ckpt['epoch']}  val_F1={ckpt.get('val_f1',0):.4f}"
          f"  sınıflar={list(le.classes_)}")
    return m, le


def load_efficientnet():
    p = os.path.join(MODELS_DIR, "best_efficientnet.pt")
    if not os.path.exists(p):
        print("  [EfficientNet] best_efficientnet.pt bulunamadı → atlanıyor")
        return None, None
    ckpt = torch.load(p, map_location=device, weights_only=False)
    m = EfficientNetAirport(ckpt.get("n_classes", 5))
    m.load_state_dict(ckpt["model_state"]); m.eval().to(device)
    le = joblib.load(os.path.join(MODELS_DIR, "efficientnet_label_encoder.pkl"))
    print(f"  [EfficientNet] ✅ {ckpt.get('phase','?')} ep{ckpt['epoch']}"
          f"  val_F1={ckpt.get('val_f1',0):.4f}  sınıflar={list(le.classes_)}")
    return m, le


# ================================================================
# 🔢  TAHMİN
# ================================================================

def predict_file(y, model, le, model_type, mel_tf, db_tf,
                 resize_fn=None, norm_fn=None, threshold=0.5):
    windows = extract_windows(y)
    ac_idx  = list(le.classes_).index(AIRCRAFT_CLASS)
    all_probs = []

    with torch.no_grad():
        for win in windows:
            y_t  = torch.FloatTensor(win).unsqueeze(0)
            if model_type == "cnn":
                inp = db_tf(mel_tf(y_t)).unsqueeze(0).to(device)
            else:
                spec = db_tf(mel_tf(y_t))
                mn, mx = spec.min(), spec.max()
                spec = (spec - mn) / (mx - mn + 1e-8)
                inp  = norm_fn(resize_fn(spec.repeat(3,1,1))).unsqueeze(0).to(device)
            probs = torch.softmax(model(inp), dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    avg     = np.mean(all_probs, axis=0)
    ac_prob = float(avg[ac_idx])
    pred_2  = "aircraft" if ac_prob >= threshold else "negative"
    pred_5  = le.classes_[int(np.argmax(avg))]
    return pred_5, pred_2, ac_prob, avg


# ================================================================
# 📊  GÖRSELLEŞTIRME
# ================================================================

def plot_confusion_matrices(y_true_k, predictions_k, labels_2, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n   = len(predictions_k)
    if n == 0: return
    fig = plt.figure(figsize=(9 * n, 6))
    for i, (name, preds) in enumerate(predictions_k.items()):
        ax     = fig.add_subplot(1, n, i + 1)
        cm     = confusion_matrix(y_true_k, preds, labels=labels_2, normalize="true")
        cm_raw = confusion_matrix(y_true_k, preds, labels=labels_2)
        annots = np.empty_like(cm, dtype=object)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annots[r,c] = f"{cm[r,c]:.2f}\n({cm_raw[r,c]})"
        sns.heatmap(cm, annot=annots, fmt="", cmap="Blues",
                    xticklabels=labels_2, yticklabels=labels_2, ax=ax,
                    linewidths=0.5, vmin=0, vmax=1)
        f1m = f1_score(y_true_k, preds, average="macro",
                       labels=labels_2, zero_division=0)
        acc = accuracy_score(y_true_k, preds)
        ax.set_title(f"{name}\nF1 Macro: {f1m:.4f}  |  Acc: {acc:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    plt.suptitle("Confusion Matrix — Dış Dataset Testi",
                 fontsize=14, fontweight="bold", y=1.02)
    out = os.path.join(save_dir, "confusion_matrix_external.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


def plot_5class_distribution(predictions_5, y_true, save_dir):
    """Tahminlerin hangi 5-sınıfa düştüğünü gerçek etikete göre gösterir."""
    os.makedirs(save_dir, exist_ok=True)
    n   = len(predictions_5)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1: axes = [axes]

    all_5classes = ["AIRCRAFT", "AMBIENT", "SPEECH", "TRAFFIC", "WIND"]
    pal = {"aircraft": "#2196F3", "negative": "#F44336", "unknown": "#9E9E9E"}

    for ax, (name, preds5) in zip(axes, predictions_5.items()):
        true_groups = sorted(set(y_true))
        bottom = np.zeros(len(all_5classes))
        for tg in true_groups:
            mask  = [t == tg for t in y_true]
            sub   = [p for p, m in zip(preds5, mask) if m]
            counts = Counter(sub)
            vals  = np.array([counts.get(c, 0) for c in all_5classes], dtype=float)
            bars  = ax.bar(range(len(all_5classes)), vals, bottom=bottom,
                           label=f"Gerçek: {tg}",
                           color=pal.get(tg, "#9E9E9E"),
                           edgecolor="white", linewidth=0.7)
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(xi, b + v/2, str(int(v)), ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(range(len(all_5classes)))
        ax.set_xticklabels(all_5classes, fontsize=10)
        ax.set_ylabel("Dosya Sayısı")
        ax.set_title(f"{name} — 5-Sınıf Tahmin Dağılımı\n"
                     "(Negatif örnekler hangi sınıfa gidiyor?)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "5class_distribution.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def plot_confidence_distribution(conf_data, y_true, threshold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n   = len(conf_data)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1: axes = [axes]

    pal = {"aircraft": "#2196F3", "negative": "#F44336", "unknown": "#9E9E9E"}
    for ax, (name, confs) in zip(axes, conf_data.items()):
        confs  = np.array(confs)
        labels = np.array(y_true)
        for lbl, color in pal.items():
            mask = labels == lbl
            if mask.sum() > 0:
                ax.hist(confs[mask], bins=20, alpha=0.7, color=color,
                        label=f"{lbl} (n={mask.sum()})",
                        density=True, edgecolor="white")
        ax.axvline(threshold, color="black", linestyle="--",
                   linewidth=2, label=f"Eşik: {threshold}")
        ax.set_xlabel("AIRCRAFT Softmax Olasılığı", fontsize=11)
        ax.set_ylabel("Yoğunluk")
        ax.set_title(f"{name} — AIRCRAFT Güven Dağılımı",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1)

    plt.tight_layout()
    out = os.path.join(save_dir, "confidence_distribution.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def plot_threshold_sweep(conf_data, y_true, save_dir):
    """En iyi eşik değerini bulmak için Recall/Precision/F1 süpürmesi."""
    os.makedirs(save_dir, exist_ok=True)
    known_mask = [t != "unknown" for t in y_true]
    y_known    = [t for t, m in zip(y_true, known_mask) if m]
    if len(y_known) == 0 or len(set(y_known)) < 2:
        print("  ⚠  Threshold sweep: tek sınıf veya sıfır etiket → atlanıyor")
        return

    thresholds = np.linspace(0.05, 0.95, 40)
    n   = len(conf_data)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1: axes = [axes]

    for ax, (name, confs) in zip(axes, conf_data.items()):
        confs_k = np.array([c for c, m in zip(confs, known_mask) if m])
        recs, precs, f1s = [], [], []
        for thr in thresholds:
            pk = ["aircraft" if c >= thr else "negative" for c in confs_k]
            recs.append(recall_score(y_known, pk, pos_label="aircraft",
                                     zero_division=0))
            precs.append(precision_score(y_known, pk, pos_label="aircraft",
                                         zero_division=0))
            f1s.append(f1_score(y_known, pk, pos_label="aircraft",
                                zero_division=0))

        ax.plot(thresholds, recs,  label="Recall",    color="#2196F3", linewidth=2)
        ax.plot(thresholds, precs, label="Precision", color="#F44336", linewidth=2)
        ax.plot(thresholds, f1s,   label="F1",        color="#4CAF50", linewidth=2)
        best_thr = thresholds[np.argmax(f1s)]
        best_f1  = max(f1s)
        ax.axvline(best_thr, color="gray", linestyle="--",
                   linewidth=1.5, label=f"En iyi eşik: {best_thr:.2f}")
        ax.scatter([best_thr], [best_f1], color="green", zorder=5, s=80)
        ax.set_xlabel("Eşik Değeri"); ax.set_ylabel("Skor")
        ax.set_title(f"{name} — Eşik Analizi\n"
                     f"En iyi F1={best_f1:.4f} @ threshold={best_thr:.2f}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        print(f"  [{name}] Önerilen eşik: {best_thr:.2f}  →  F1={best_f1:.4f}")

    plt.tight_layout()
    out = os.path.join(save_dir, "threshold_sweep.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ================================================================
# 📝  RAPOR
# ================================================================

def write_report(y_true, predictions, conf_data, threshold, save_dir, n_files):
    os.makedirs(save_dir, exist_ok=True)
    known_mask = [t != "unknown" for t in y_true]
    y_known    = [t for t, m in zip(y_true, known_mask) if m]
    labels_ev  = sorted(set(y_known)) if y_known else ["aircraft", "negative"]

    lines = ["=" * 65,
             "  DIŞ DATASET TEST RAPORU",
             f"  Taranan dosya       : {n_files}",
             f"  Etiketli            : {len(y_known)}",
             f"  Etiketsiz (unknown) : {len(y_true)-len(y_known)}",
             f"  Confidence eşiği    : {threshold}",
             f"  Sınıf dağılımı      : {dict(Counter(y_true))}",
             "=" * 65]

    for name, preds in predictions.items():
        preds_k = [p for p, m in zip(preds, known_mask) if m]
        confs   = np.array(conf_data[name])

        lines += [f"\n{'─'*65}", f"  {name}", f"{'─'*65}"]

        if preds_k and y_known:
            acc  = accuracy_score(y_known, preds_k)
            f1m  = f1_score(y_known, preds_k, average="macro",
                            labels=labels_ev, zero_division=0)
            prec = precision_score(y_known, preds_k, pos_label="aircraft",
                                   zero_division=0) if "aircraft" in labels_ev else 0
            rec  = recall_score(y_known, preds_k, pos_label="aircraft",
                                zero_division=0) if "aircraft" in labels_ev else 0
            lines += [f"  Değerlendirilen dosya : {len(preds_k)}",
                      f"  Accuracy              : {acc:.4f}  ({acc:.1%})",
                      f"  F1 Macro              : {f1m:.4f}",
                      f"  Aircraft Precision    : {prec:.4f}",
                      f"  Aircraft Recall       : {rec:.4f}",
                      "",
                      classification_report(y_known, preds_k,
                                            labels=labels_ev,
                                            target_names=labels_ev, digits=3)]

        lines += [f"  AIRCRAFT Güven — tüm {len(confs)} dosya:",
                  f"    Ortalama : {confs.mean():.4f}",
                  f"    Medyan   : {float(np.median(confs)):.4f}",
                  f"    Min/Max  : {confs.min():.4f} / {confs.max():.4f}",
                  f"    ≥{threshold} tahmin : {(confs>=threshold).sum()}"
                  f"/{len(confs)} ({(confs>=threshold).mean():.1%})"]

    # Özet karşılaştırma
    if len(predictions) > 1 and y_known:
        lines += ["\n" + "=" * 65, "  KARŞILAŞTIRMA",
                  f"  {'Model':20s}  {'Acc':>7}  {'F1 Mac':>7}  "
                  f"{'Precision':>9}  {'Recall':>7}", "  "+"─"*55]
        for name, preds in predictions.items():
            preds_k = [p for p, m in zip(preds, known_mask) if m]
            if not preds_k: continue
            acc  = accuracy_score(y_known, preds_k)
            f1m  = f1_score(y_known, preds_k, average="macro",
                            labels=labels_ev, zero_division=0)
            prec = precision_score(y_known, preds_k, pos_label="aircraft",
                                   zero_division=0) if "aircraft" in labels_ev else 0
            rec  = recall_score(y_known, preds_k, pos_label="aircraft",
                                zero_division=0) if "aircraft" in labels_ev else 0
            lines.append(f"  {name:20s}  {acc:7.4f}  {f1m:7.4f}  "
                         f"{prec:9.4f}  {rec:7.4f}")
        lines += ["=" * 65]

    report = "\n".join(lines)
    print("\n" + report)
    out = os.path.join(save_dir, "external_test_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  → {out}")


# ================================================================
# 🏃  ANA PIPELINE
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=CONF_THRESHOLD,
                        help="AIRCRAFT softmax eşiği (varsayılan: 0.13)")
    args = parser.parse_args()
    threshold = args.threshold

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  External Test  —  AirportCNN + EfficientNet-B0")
    print(f"  Device: {device}  |  Eşik: {threshold}")
    print("=" * 65)

    # ── 1. Dosya tarama ───────────────────────────────────────────
    print("\n── Dosya Tarama ─────────────────────────────────────────")
    df = scan_files(TEST_FOLDER, LABELS_CSV)
    if len(df) == 0:
        print(f"[HATA] .wav bulunamadı: {TEST_FOLDER}"); raise SystemExit(1)

    print(f"\n  Toplam .wav  : {len(df)}")
    print(f"  CSV'de olan  : {df['in_csv'].sum()}")
    print(f"  CSV'de yok   : {(~df['in_csv']).sum()} (klasör adından etiket alındı)")
    print(f"  Ground truth : {dict(Counter(df['true_label']))}")

    # ── 2. Modeller ───────────────────────────────────────────────
    print("\n── Modeller ─────────────────────────────────────────────")
    mel_tf    = TA.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                                   hop_length=HOP_FFT, n_mels=N_MELS)
    db_tf     = TA.AmplitudeToDB(top_db=80)
    resize_fn = TV.Resize((224, 224), antialias=True)
    norm_fn   = TV.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    cnn_model, cnn_le = load_cnn()
    eff_model, eff_le = load_efficientnet()
    if cnn_model is None and eff_model is None:
        print("[HATA] Hiçbir model yok."); raise SystemExit(1)

    # ── 3. Tahminler ──────────────────────────────────────────────
    print(f"\n── Tahminler ({len(df)} dosya) ───────────────────────────")
    y_true      = []
    cnn_p2, cnn_p5, cnn_c = [], [], []
    eff_p2, eff_p5, eff_c = [], [], []
    rows_out = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  İşleniyor", ncols=72):
        y     = load_audio(row["filepath"])
        truth = row["true_label"]
        y_true.append(truth)
        n_win = len(extract_windows(y))

        out_row = {"filename": row["filename"], "split": row["split"],
                   "true_label": truth, "n_windows": n_win,
                   "in_csv": row["in_csv"]}

        if cnn_model is not None:
            p5, p2, conf, _ = predict_file(
                y, cnn_model, cnn_le, "cnn", mel_tf, db_tf,
                threshold=threshold)
            cnn_p5.append(p5); cnn_p2.append(p2); cnn_c.append(conf)
            out_row.update({"cnn_5class": p5, "cnn_2class": p2,
                            "cnn_aircraft_prob": round(conf, 4),
                            "cnn_correct": (p2==truth) if truth!="unknown" else None})

        if eff_model is not None:
            p5, p2, conf, _ = predict_file(
                y, eff_model, eff_le, "efficientnet", mel_tf, db_tf,
                resize_fn, norm_fn, threshold=threshold)
            eff_p5.append(p5); eff_p2.append(p2); eff_c.append(conf)
            out_row.update({"eff_5class": p5, "eff_2class": p2,
                            "eff_aircraft_prob": round(conf, 4),
                            "eff_correct": (p2==truth) if truth!="unknown" else None})

        rows_out.append(out_row)

    # Per-file CSV
    pd.DataFrame(rows_out).to_csv(
        os.path.join(OUT_DIR, "per_file_results.csv"),
        index=False, encoding="utf-8-sig")
    print(f"  → {os.path.join(OUT_DIR, 'per_file_results.csv')}")

    # Sonuçları topla
    predictions   = {}
    predictions_5 = {}
    conf_data     = {}
    if cnn_model:
        predictions["AirportCNN"]   = cnn_p2
        predictions_5["AirportCNN"] = cnn_p5
        conf_data["AirportCNN"]     = cnn_c
    if eff_model:
        predictions["EfficientNet"]   = eff_p2
        predictions_5["EfficientNet"] = eff_p5
        conf_data["EfficientNet"]     = eff_c

    # ── 4. Terminal özet ──────────────────────────────────────────
    print("\n" + "="*65)
    known_mask = [t != "unknown" for t in y_true]
    y_known    = [t for t, m in zip(y_true, known_mask) if m]
    for name, preds in predictions.items():
        preds_k = [p for p, m in zip(preds, known_mask) if m]
        confs   = np.array(conf_data[name])
        print(f"\n  {name}")
        if preds_k and y_known:
            acc = accuracy_score(y_known, preds_k)
            f1m = f1_score(y_known, preds_k, average="macro", zero_division=0)
            print(f"    Doğru  : {sum(p==t for p,t in zip(preds_k,y_known))}"
                  f"/{len(y_known)} ({acc:.1%})")
            print(f"    F1 Mac : {f1m:.4f}")
        print(f"    AIRCRAFT ort. güven: {confs.mean():.4f}")
        print(f"    ≥{threshold} aircraft: {(confs>=threshold).sum()}/{len(confs)}")
        print(f"    5-sınıf: "+"  ".join(f"{k}:{v}"
              for k,v in Counter(predictions_5[name]).most_common()))

    # ── 5. Grafikler ──────────────────────────────────────────────
    print("\n── Grafikler ────────────────────────────────────────────")
    labels_2 = sorted(set(y_true) - {"unknown"}) or ["aircraft", "negative"]
    if y_known and len(set(y_known)) > 0:
        plot_confusion_matrices(
            y_known,
            {k: [p for p,m in zip(v,known_mask) if m]
             for k,v in predictions.items()},
            labels_2, OUT_DIR)
    plot_5class_distribution(predictions_5, y_true, OUT_DIR)
    plot_confidence_distribution(conf_data, y_true, threshold, OUT_DIR)
    if y_known and len(set(y_known)) >= 2:
        plot_threshold_sweep(conf_data, y_true, OUT_DIR)

    write_report(y_true, predictions, conf_data, threshold, OUT_DIR, len(df))

    print(f"\n  ✅ Tamamlandı → {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()