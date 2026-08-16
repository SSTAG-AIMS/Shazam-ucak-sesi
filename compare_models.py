"""
╔══════════════════════════════════════════════════════════════╗
║         Compare Models  —  SVM vs CNN vs EfficientNet       ║
║   Aynı test seti üzerinde yan yana karşılaştırma            ║
╚══════════════════════════════════════════════════════════════╝

Modeller (var olanlar otomatik algılanır, yoksa atlanır):
  models/best_model.pkl            → SVM
  models/best_cnn.pt               → AirportCNN
  models/best_efficientnet.pt      → EfficientNet-B0

Çıktılar (outputs/comparison/):
  comparison_report.txt
  confusion_matrices.png           → yan yana CM (tüm modeller)
  f1_comparison.png                → per-class F1 bar chart
  recall_comparison.png            → per-class Recall bar chart  (YENİ)
"""

import os
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio.transforms as TA
import torchvision.transforms as TV
import torchvision.models as tvm

from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR  —  train_*.py dosyaları ile AYNI
# ================================================================

PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
CACHE_PATH   = os.path.join(PROJECT_ROOT, "cache", "features_v3.pkl")
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "cache", "manifest_v3.csv")
OUT_DIR      = os.path.join(PROJECT_ROOT, "outputs", "comparison")

TEST_SIZE   = 0.15
RANDOM_SEED = 42

SR       = 22050
N_MELS   = 128
N_FFT    = 2048
HOP_FFT  = 512
DURATION = 5.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ================================================================
# 🏗️  MODEL MİMARİLERİ
# ================================================================

class AirportCNN(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EfficientNetAirport(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        backbone        = tvm.efficientnet_b0(weights=None)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ================================================================
# 📥  VERİ — Test seti
# ================================================================

def get_test_split_svm():
    data = joblib.load(CACHE_PATH)
    X, y_raw, paths = data["X"], data["y"], data["paths"]
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    
    # --- FIX: Filter out labels the encoder doesn't know ---
    known_labels = set(le.classes_)
    mask = np.array([label in known_labels for label in y_raw])
    
    X = X[mask]
    y_raw = y_raw[mask]
    paths = [p for i, p in enumerate(paths) if mask[i]]
    # -------------------------------------------------------

    y_enc = le.transform(y_raw)

    _, X_te, _, y_te, _, paths_te = train_test_split(
        X, y_enc, paths,
        test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_SEED
    )
    return X_te, y_te, paths_te, le, list(le.classes_)


def get_test_split_manifest():
    """
    manifest_v3.csv → CNN / EfficientNet test verisi.
    Aynı seed ve TEST_SIZE → aynı dosya seti.
    """
    df = pd.read_csv(MANIFEST_CSV)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    _, test_df = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["label_enc"], random_state=RANDOM_SEED
    )
    return test_df.to_dict("records"), le, list(le.classes_)


# ================================================================
# 🔊  SES YÜKLE
# ================================================================

def load_audio_fixed(path: str) -> np.ndarray:
    target = int(SR * DURATION)
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=DURATION + 0.5)
    except Exception:
        return np.zeros(target, dtype=np.float32)
    if len(y) >= target:
        start = (len(y) - target) // 2
        y = y[start: start + target]
    else:
        y = np.pad(y, (0, target - len(y)))
    return y.astype(np.float32)


# ================================================================
# 🤖  SVM TAHMİN
# ================================================================

def predict_svm(X_te: np.ndarray) -> np.ndarray:
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    return model.predict(X_te)


# ================================================================
# 🧠  CNN TAHMİN
# ================================================================

def predict_cnn(records: list, label_names: list) -> np.ndarray:
    cnn_path = os.path.join(MODELS_DIR, "best_cnn.pt")
    ckpt     = torch.load(cnn_path, map_location="cpu", weights_only=False)
    model    = AirportCNN(n_classes=ckpt.get("n_classes", len(label_names)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mel_tf = TA.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                                hop_length=HOP_FFT, n_mels=N_MELS)
    db_tf  = TA.AmplitudeToDB(top_db=80)
    preds  = []

    with torch.no_grad():
        for rec in tqdm(records, desc="  CNN tahmin", ncols=70):
            y    = load_audio_fixed(rec["path"])
            y_t  = torch.FloatTensor(y).unsqueeze(0)
            spec = db_tf(mel_tf(y_t)).unsqueeze(0)   # (1,1,N_MELS,T)
            preds.append(int(model(spec).argmax(1).item()))

    return np.array(preds)


# ================================================================
# ⚡  EfficientNet TAHMİN
# ================================================================

def predict_efficientnet(records: list, label_names: list) -> np.ndarray:
    eff_path = os.path.join(MODELS_DIR, "best_efficientnet.pt")
    ckpt     = torch.load(eff_path, map_location="cpu", weights_only=False)
    model    = EfficientNetAirport(n_classes=ckpt.get("n_classes", len(label_names)))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mel_tf   = TA.MelSpectrogram(sample_rate=SR, n_fft=N_FFT,
                                  hop_length=HOP_FFT, n_mels=N_MELS)
    db_tf    = TA.AmplitudeToDB(top_db=80)
    resize   = TV.Resize((224, 224), antialias=True)
    norm     = TV.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    preds    = []

    with torch.no_grad():
        for rec in tqdm(records, desc="  EfficientNet tahmin", ncols=70):
            y    = load_audio_fixed(rec["path"])
            y_t  = torch.FloatTensor(y).unsqueeze(0)
            spec = db_tf(mel_tf(y_t))                # (1, N_MELS, T)

            # [0,1] normalize
            s_min, s_max = spec.min(), spec.max()
            if s_max > s_min:
                spec = (spec - s_min) / (s_max - s_min)
            else:
                spec = torch.zeros_like(spec)

            spec_rgb = spec.repeat(3, 1, 1)          # (3, N_MELS, T)
            spec_rgb = norm(resize(spec_rgb))         # (3, 224, 224)
            out      = model(spec_rgb.unsqueeze(0))   # (1, n_classes)
            preds.append(int(out.argmax(1).item()))

    return np.array(preds)


# ================================================================
# 📊  GÖRSELLEŞTIRME
# ================================================================

def _color_map():
    return {
        "SVM":          "#4C72B0",
        "AirportCNN":   "#DD8452",
        "EfficientNet": "#55A868",
    }


def plot_confusion_matrices(y_true, predictions: dict, labels, save_dir):
    """predictions = {"SVM": arr, "AirportCNN": arr, ...}"""
    os.makedirs(save_dir, exist_ok=True)
    n   = len(predictions)
    fig = plt.figure(figsize=(9 * n, 7))
    gs  = gridspec.GridSpec(1, n, figure=fig, wspace=0.35)

    for idx, (name, preds) in enumerate(predictions.items()):
        cm = confusion_matrix(y_true, preds, normalize="true")
        ax = fig.add_subplot(gs[idx])
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    linewidths=0.5, linecolor="#E0E0E0")
        f1m = f1_score(y_true, preds, average="macro", zero_division=0)
        ax.set_title(f"{name}\nF1 Macro: {f1m:.4f}",
                     fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Confusion Matrix Karşılaştırması",
                 fontsize=15, fontweight="bold", y=1.02)
    out = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {out}")


def _bar_comparison(y_true, predictions: dict, labels, metric, save_dir):
    """
    metric: "f1" veya "recall"
    Her sınıf için grouped bar chart.
    """
    os.makedirs(save_dir, exist_ok=True)
    colors = _color_map()
    names  = list(predictions.keys())
    x      = np.arange(len(labels))
    width  = 0.8 / len(names)
    offsets = np.linspace(-(len(names)-1)/2, (len(names)-1)/2, len(names)) * width

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(names):
        preds = predictions[name]
        if metric == "f1":
            scores = f1_score(y_true, preds, average=None,
                              labels=range(len(labels)), zero_division=0)
            macro  = f1_score(y_true, preds, average="macro", zero_division=0)
            ylabel = "F1 Score"
            title  = "Per-Class F1 Karşılaştırması"
            fname  = "f1_comparison.png"
        else:
            scores = recall_score(y_true, preds, average=None,
                                  labels=range(len(labels)), zero_division=0)
            macro  = recall_score(y_true, preds, average="macro", zero_division=0)
            ylabel = "Recall"
            title  = "Per-Class Recall Karşılaştırması"
            fname  = "recall_comparison.png"

        bars = ax.bar(x + offsets[i], scores, width,
                      label=f"{name} (macro {macro:.3f})",
                      color=colors.get(name, f"C{i}"),
                      edgecolor="white", linewidth=0.7)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(0.858, color="gray", linestyle="--",
               linewidth=1, alpha=0.6, label="Hedef: 0.858")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def write_report(y_true, predictions: dict, labels, save_dir):
    lines = []
    lines.append("=" * 65)
    lines.append("  MODEL KARŞILAŞTIRMA RAPORU")
    lines.append(f"  Modeller: {', '.join(predictions.keys())}")
    lines.append("=" * 65)

    summary_rows = []
    for name, preds in predictions.items():
        acc  = accuracy_score(y_true, preds)
        f1m  = f1_score(y_true, preds, average="macro",    zero_division=0)
        f1w  = f1_score(y_true, preds, average="weighted", zero_division=0)
        lines.append(f"\n{'─'*65}")
        lines.append(f"  {name}")
        lines.append(f"{'─'*65}")
        lines.append(f"  Accuracy   : {acc:.4f}  ({acc:.1%})")
        lines.append(f"  F1 Macro   : {f1m:.4f}  {'✅' if f1m >= 0.858 else '⚠️'}")
        lines.append(f"  F1 Weighted: {f1w:.4f}")
        lines.append("")
        lines.append(classification_report(y_true, preds,
                                            target_names=labels, digits=3))
        summary_rows.append((name, acc, f1m, f1w))

    # Özet tablo
    lines.append("=" * 65)
    lines.append("  ÖZET TABLO")
    lines.append(f"  {'Model':20s}  {'Accuracy':>9}  {'F1 Macro':>9}  {'F1 Wtd':>9}")
    lines.append("  " + "─" * 50)
    for name, acc, f1m, f1w in summary_rows:
        lines.append(f"  {name:20s}  {acc:9.4f}  {f1m:9.4f}  {f1w:9.4f}")

    # Kazanan
    best_name = max(predictions, key=lambda n: f1_score(
        y_true, predictions[n], average="macro", zero_division=0))
    lines.append("")
    lines.append(f"  🏆  KAZANAN: {best_name}")
    lines.append("=" * 65)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    out = os.path.join(save_dir, "comparison_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n  → {out}")


# ================================================================
# 🏃  ANA
# ================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  Compare Models  —  SVM vs CNN vs EfficientNet-B0")
    print("=" * 65)

    # ── Test seti ────────────────────────────────────────────────
    # Manifest tabanlı (CNN + EfficientNet için)
    test_records, le_manifest, labels = get_test_split_manifest()
    y_te = np.array([r["label_enc"] for r in test_records])
    paths_te = [r["path"] for r in test_records]

    print(f"\n  Test: {len(y_te)} örnek  |  Sınıflar: {labels}")
    dist = Counter(labels[i] for i in y_te)
    for lbl, cnt in sorted(dist.items()):
        print(f"    {lbl:10s} {cnt:4d}")

    predictions = {}

    # ── SVM ──────────────────────────────────────────────────────
    svm_path = os.path.join(MODELS_DIR, "best_model.pkl")
    cache_ok = os.path.exists(CACHE_PATH)
    svm_ok   = os.path.exists(svm_path) and os.path.exists(
                os.path.join(MODELS_DIR, "label_encoder.pkl"))

    if svm_ok and cache_ok:
        print("\n  [SVM] Tahminler hesaplanıyor...")
        X_te_svm, y_te_svm, _, le_svm, svm_labels = get_test_split_svm()
        svm_raw = predict_svm(X_te_svm)
        # SVM le → manifest le eşleştirmesi (sınıf sırası farklı olabilir)
        remap = {le_svm.transform([c])[0]: le_manifest.transform([c])[0]
                 for c in svm_labels if c in labels}
        svm_pred = np.array([remap.get(p, p) for p in svm_raw])
        # SVM test seti manifest test setiyle tam aynı değil (farklı pipeline);
        # F1'i kendi y_te üzerinden al
        svm_pred_aligned = predict_svm_on_paths(paths_te, le_svm, le_manifest, labels)
        if svm_pred_aligned is not None:
            predictions["SVM"] = svm_pred_aligned
        else:
            # Fallback: SVM kendi test seti üzerinde çalışır, y_te yerine y_te_svm kullan
            print("  [SVM] Manifest tabanlı hizalama kullanılıyor.")
            # SVM'in kendi test seti farklı olabilir; bunu uyarı olarak göster
            print("  ⚠  SVM features_v3.pkl üzerinden değerlendirildi "
                  "(manifest ile tam hizalı olmayabilir).")
            svm_f1 = f1_score(y_te_svm, svm_raw, average="macro", zero_division=0)
            print(f"  SVM F1 Macro (kendi test): {svm_f1:.4f}")
    else:
        if not svm_ok:
            print("\n  [SVM] ⚠  best_model.pkl veya label_encoder.pkl yok → atlanıyor")
        if not cache_ok:
            print("\n  [SVM] ⚠  features_v3.pkl yok → atlanıyor")

    # ── CNN ──────────────────────────────────────────────────────
    cnn_path = os.path.join(MODELS_DIR, "best_cnn.pt")
    if os.path.exists(cnn_path):
        print("\n  [CNN] Tahminler hesaplanıyor...")
        cnn_pred = predict_cnn(test_records, labels)
        predictions["AirportCNN"] = cnn_pred
        print(f"  CNN F1 Macro: {f1_score(y_te, cnn_pred, average='macro'):.4f}")
    else:
        print(f"\n  [CNN] ⚠  best_cnn.pt yok → atlanıyor")

    # ── EfficientNet ─────────────────────────────────────────────
    eff_path = os.path.join(MODELS_DIR, "best_efficientnet.pt")
    if os.path.exists(eff_path):
        print("\n  [EfficientNet] Tahminler hesaplanıyor...")
        eff_pred = predict_efficientnet(test_records, labels)
        predictions["EfficientNet"] = eff_pred
        print(f"  EfficientNet F1 Macro: "
              f"{f1_score(y_te, eff_pred, average='macro'):.4f}")
    else:
        print(f"\n  [EfficientNet] ⚠  best_efficientnet.pt yok → atlanıyor")

    if not predictions:
        print("\n  [HATA] Hiçbir model bulunamadı. Eğitim çalıştırılmış mı?")
        raise SystemExit(1)

    # ── Tek model varsa doğrudan rapor ───────────────────────────
    if len(predictions) == 1:
        name, preds = list(predictions.items())[0]
        print(f"\n  Sadece {name} bulundu. Tek model raporu:")
        print(classification_report(y_te, preds, target_names=labels, digits=3))
        plot_confusion_matrices(y_te, predictions, labels, OUT_DIR)
        return

    # ── Karşılaştırma ─────────────────────────────────────────────
    write_report(y_te, predictions, labels, OUT_DIR)
    plot_confusion_matrices(y_te, predictions, labels, OUT_DIR)
    _bar_comparison(y_te, predictions, labels, "f1",     OUT_DIR)
    _bar_comparison(y_te, predictions, labels, "recall", OUT_DIR)

    print(f"\n  ✅ Tamamlandı. Çıktılar: {OUT_DIR}")


def predict_svm_on_paths(paths: list, le_svm, le_manifest, labels):
    """
    SVM için features_v3.pkl'deki path eşleşmesine göre tahmin üret.
    Manifest test dosyaları ile SVM cache'ini hizala.
    Eşleşme bulunamazsa None döner.
    """
    try:
        data     = joblib.load(CACHE_PATH)
        all_paths = data["paths"]
        X_all    = data["X"]
        y_all    = data["y"]
        model    = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))

        path_to_idx = {p: i for i, p in enumerate(all_paths)}
        indices = [path_to_idx[p] for p in paths if p in path_to_idx]

        if len(indices) < len(paths) * 0.5:
            return None  # yarısından azı eşleşti, güvenilmez

        X_sub   = X_all[indices]
        preds   = model.predict(X_sub)

        # SVM le → manifest le dönüştür
        remap = {}
        for c in le_svm.classes_:
            if c in labels:
                remap[le_svm.transform([c])[0]] = le_manifest.transform([c])[0]
        preds = np.array([remap.get(p, p) for p in preds])
        return preds

    except Exception as e:
        print(f"  [SVM hizalama hatası]: {e}")
        return None


if __name__ == "__main__":
    main()
