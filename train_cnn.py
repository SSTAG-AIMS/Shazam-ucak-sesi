"""
╔══════════════════════════════════════════════════════════════╗
║              Train CNN  —  5 Sınıf  (v2 — Gelişmiş İzleme) ║
║   Mel Spectrogram + SpecAugment + AirportCNN                ║
╚══════════════════════════════════════════════════════════════╝

Sıralama:
  1. python dataset_builder_v3.py     → manifest_v3.csv üretir
  2. python train_cnn.py              → best_cnn.pt üretir

Yenilikler (v2):
  - Ctrl+C → anlık confusion matrix + per-class rapor kaydedilir
  - Her epoch: terminalde per-class F1 + Recall satırı
  - History'de LR + per-class F1 takibi
  - 4 panel grafik: Loss / Acc / F1 Macro / LR
  - Per-class F1 over epochs ayrı grafik
  - Her CHECKPOINT_EVERY epoch'ta ara CM kaydedilir

Gereksinimler:
  pip install torch torchaudio librosa scikit-learn imbalanced-learn
              joblib numpy pandas matplotlib seaborn tqdm
"""

import os
import sys
import time
import signal
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")          # GUI olmadan kayıt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, recall_score)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "cache",  "manifest_v3.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "training_cnn")

# Mel Spectrogram parametreleri
SR        = 22050
N_MELS    = 128
N_FFT     = 2048
HOP_FFT   = 512
DURATION  = 5.0          # saniye

# Eğitim hiper-parametreleri
EPOCHS        = 50
PATIENCE      = 10
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15
RANDOM_SEED   = 42

# Her N epoch'ta ara confusion matrix kaydet (0 = devre dışı)
CHECKPOINT_EVERY = 0

# noise_detector.py ile senkronize — DOKUNMA
MANUAL_CLASS_WEIGHTS = {
    "AIRCRAFT": 0.5,
    "SPEECH":   2.0,
    "TRAFFIC":  2.5,
    "WIND":     2.5,
    "AMBIENT":  2.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# 🔊  DATASET
# ================================================================

class SpecAugment(nn.Module):
    """Mel spektrogramı maskele (Park et al. 2019)."""
    def __init__(self, time_mask: int = 30, freq_mask: int = 15,
                 n_time: int = 2, n_freq: int = 2):
        super().__init__()
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask) for _ in range(n_time)
        ])
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask) for _ in range(n_freq)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.time_masks:
            x = m(x)
        for m in self.freq_masks:
            x = m(x)
        return x


class MelSpectrogramDataset(Dataset):
    def __init__(self, records: list, augment: bool = False):
        self.records    = records
        self.augment    = augment
        self.target_len = int(SR * DURATION)
        self.mel_tf    = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT,
            hop_length=HOP_FFT, n_mels=N_MELS
        )
        self.db_tf     = T.AmplitudeToDB(top_db=80)
        self.spec_aug  = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            y, _ = librosa.load(rec["path"], sr=SR, mono=True,
                                 duration=DURATION + 0.5)
        except Exception:
            y = np.zeros(self.target_len, dtype=np.float32)

        if len(y) >= self.target_len:
            start = (len(y) - self.target_len) // 2
            y = y[start: start + self.target_len]
        else:
            y = np.pad(y, (0, self.target_len - len(y)))

        y_tensor = torch.FloatTensor(y).unsqueeze(0)
        spec     = self.mel_tf(y_tensor)
        spec     = self.db_tf(spec)
        if self.augment and self.spec_aug is not None:
            spec = self.spec_aug(spec)
        return spec, rec["label_enc"]


# ================================================================
# 🏗️  MODEL
# ================================================================

class AirportCNN(nn.Module):
    """
    3 Conv bloğu + AdaptiveAvgPool → Classifier
    Giriş: (B, 1, N_MELS, T) = (B, 1, 128, ~216)
    """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ================================================================
# 🏋  EĞİTİM & DEĞERLENDİRME
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(specs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        out  = model(specs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        preds       = out.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def per_class_f1_recall(y_true, y_pred, label_names):
    """Her sınıf için F1 ve Recall döndür."""
    f1s     = f1_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return dict(zip(label_names, f1s)), dict(zip(label_names, recalls))


def print_per_class(f1_dict, recall_dict, epoch_label=""):
    """Terminale per-class F1 + Recall tablosu yaz."""
    header = f"  ── Per-Class {epoch_label} ──────────────────────────────"
    print(header)
    print(f"  {'Sınıf':12s}  {'F1':>6}  {'Recall':>7}")
    for cls in sorted(f1_dict.keys()):
        f1  = f1_dict[cls]
        rec = recall_dict[cls]
        bar = "▓" * int(f1 * 10)
        warn = " ⚠" if rec < 0.5 else ""
        print(f"  {cls:12s}  {f1:6.4f}  {rec:7.4f}  {bar}{warn}")
    print()


# ================================================================
# 📊  PLOT FONKSİYONLARI
# ================================================================

def plot_training_curves(history: dict, save_dir: str, suffix: str = ""):
    """4 panel: Loss / Acc / F1 Macro / LR"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    ax.plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], label="Train", color="#2196F3")
    ax.plot(epochs, history["val_acc"],   label="Val",   color="#F44336")
    ax.set_title("Accuracy"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # F1 Macro
    ax = axes[1, 0]
    ax.plot(epochs, history["val_f1"], label="Val F1 Macro", color="#4CAF50", linewidth=2)
    ax.axhline(y=0.858, color="orange", linestyle="--", linewidth=1.5, label="Hedef: 0.858")
    best_f1  = max(history["val_f1"])
    best_ep  = history["val_f1"].index(best_f1) + 1
    ax.scatter([best_ep], [best_f1], color="red", zorder=5)
    ax.annotate(f"  Best: {best_f1:.4f} (ep{best_ep})", xy=(best_ep, best_f1),
                fontsize=8, color="red")
    ax.set_title("F1 Macro (Val)"); ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    # LR
    ax = axes[1, 1]
    if history.get("lr"):
        ax.plot(epochs, history["lr"], color="#9C27B0", linewidth=1.5)
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    plt.suptitle(f"CNN Eğitim Eğrileri{suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"training_curves_cnn{suffix}.png"
    out   = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → Kaydedildi: {out}")


def plot_per_class_f1(history: dict, label_names: list, save_dir: str, suffix: str = ""):
    """Her sınıf için F1 over epochs."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, cls in enumerate(label_names):
        key = f"val_f1_{cls}"
        if key in history and history[key]:
            ax.plot(epochs, history[key], label=cls, color=colors[i % len(colors)],
                    linewidth=1.8)

    ax.axhline(y=0.858, color="gray", linestyle="--", linewidth=1, label="Hedef: 0.858")
    ax.set_title("Per-Class F1 (Val)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"per_class_f1{suffix}.png"
    out   = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → Kaydedildi: {out}")


def plot_confusion_matrix(y_true, y_pred, label_names, save_dir, suffix: str = ""):
    os.makedirs(save_dir, exist_ok=True)
    cm     = confusion_matrix(y_true, y_pred, normalize="true")
    cm_raw = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title in zip(axes,
                                [cm_raw, cm],
                                ["Ham Sayılar", "Normalize (satır %)"]):
        fmt = "d" if data is cm_raw else ".2f"
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.tick_params(axis="x", rotation=45)

    # AMBIENT recall özellikle vurgula
    ambient_idx = label_names.index("AMBIENT") if "AMBIENT" in label_names else None
    if ambient_idx is not None:
        ambient_recall = cm[ambient_idx, ambient_idx]
        fig.text(0.5, 0.01,
                 f"AMBIENT Recall: {ambient_recall:.3f}{'  ✅' if ambient_recall >= 0.7 else '  ⚠️ DÜŞÜK'}",
                 ha="center", fontsize=11,
                 color="green" if ambient_recall >= 0.7 else "red")

    plt.suptitle(f"Confusion Matrix — AirportCNN{suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = f"confusion_matrix_cnn{suffix}.png"
    out   = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → Kaydedildi: {out}")


def save_interrupt_report(model, val_loader, criterion, device,
                           label_names, history, plots_dir, epoch):
    """Ctrl+C anında çağrılır: CM + rapor + grafikler kaydedilir."""
    print("\n\n  ⚡  Ctrl+C yakalandı — anlık rapor kaydediliyor...")
    suffix = f"_interrupt_ep{epoch}"

    _, _, preds, true = evaluate(model, val_loader, criterion, device)
    f1_dict, rec_dict = per_class_f1_recall(true, preds, label_names)
    f1_mac = f1_score(true, preds, average="macro", zero_division=0)

    print(f"\n  Val F1 Macro (epoch {epoch}): {f1_mac:.4f}")
    print_per_class(f1_dict, rec_dict, epoch_label=f"(Epoch {epoch})")
    print(classification_report(true, preds, target_names=label_names, digits=3))

    if len(history["train_loss"]) > 0:
        plot_training_curves(history, plots_dir, suffix=suffix)
        plot_per_class_f1(history, label_names, plots_dir, suffix=suffix)
    plot_confusion_matrix(true, preds, label_names, plots_dir, suffix=suffix)

    print(f"\n  Raporlar kaydedildi → {plots_dir}")


# ================================================================
# 🏃  ANA PIPELINE
# ================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    print("=" * 60)
    print("  Train CNN  —  Mel Spectrogram + AirportCNN  (v2)")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── 1. Manifest yükle ────────────────────────────────────────
    if not os.path.exists(MANIFEST_CSV):
        print(f"[HATA] Manifest bulunamadı: {MANIFEST_CSV}")
        print(f"       Önce çalıştır: python dataset_builder_v3.py")
        raise SystemExit(1)

    df = pd.read_csv(MANIFEST_CSV)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"\n  Toplam örnek: {len(df)}")

    dist = Counter(df["label"])
    print("\n── Sınıf Dağılımı ──────────────────────────────────────")
    for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 20)
        print(f"  {lbl:10s} {cnt:5d}  {bar}")

    # ── 2. Label encode ──────────────────────────────────────────
    le     = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    labels = list(le.classes_)
    print(f"\n  Sınıflar: {labels}")

    # ── 3. Eğitim / Test bölünmesi ───────────────────────────────
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["label_enc"], random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val_df["label_enc"], random_state=RANDOM_SEED
    )
    print(f"  Eğitim: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # Val/Test sınıf dağılımı bilgi amaçlı
    print("\n── Val Sınıf Dağılımı ──────────────────────────────────")
    for lbl, cnt in sorted(Counter(val_df["label"]).items()):
        print(f"  {lbl:10s} {cnt:4d}")

    # ── 4. Dataset & DataLoader ──────────────────────────────────
    train_records = train_df.to_dict("records")
    val_records   = val_df.to_dict("records")
    test_records  = test_df.to_dict("records")

    train_ds = MelSpectrogramDataset(train_records, augment=True)
    val_ds   = MelSpectrogramDataset(val_records,   augment=False)
    test_ds  = MelSpectrogramDataset(test_records,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # ── 5. Model, loss, optimizer ────────────────────────────────
    model = AirportCNN(n_classes=len(labels)).to(device)
    print(f"\n  Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")

    weights = torch.FloatTensor([
        MANUAL_CLASS_WEIGHTS[labels[i]] for i in range(len(labels))
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # ── 6. Eğitim döngüsü ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EĞİTİM BAŞLADI  |  Durdurmak için: Ctrl+C")
    print("=" * 60)

    # Başlık: temel metrikler
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>6}  "
          f"{'VaLoss':>8}  {'VaAcc':>6}  {'F1Mac':>6}  "
          f"{'LR':>10}  {'Süre':>5}  {'Best':>5}")
    print("  " + "─" * 70)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_f1":     [], "lr":        [],
    }
    # Per-class F1 history
    for cls in labels:
        history[f"val_f1_{cls}"] = []

    best_val_f1  = -1.0
    patience_cnt = 0
    best_model_path = os.path.join(MODELS_DIR, "best_cnn.pt")
    current_epoch   = 0

    # ── Ctrl+C handler ──────────────────────────────────────────
    interrupted = False

    try:
        for epoch in range(1, EPOCHS + 1):
            current_epoch = epoch
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, va_preds, va_true = evaluate(
                model, val_loader, criterion, device)

            va_f1    = f1_score(va_true, va_preds, average="macro", zero_division=0)
            cur_lr   = optimizer.param_groups[0]["lr"]
            scheduler.step()
            elapsed  = time.time() - t0

            # Per-class metrikleri hesapla
            f1_dict, rec_dict = per_class_f1_recall(va_true, va_preds, labels)

            # History güncelle
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            history["val_f1"].append(va_f1)
            history["lr"].append(cur_lr)
            for cls in labels:
                history[f"val_f1_{cls}"].append(f1_dict.get(cls, 0.0))

            # Best flag
            is_best = va_f1 > best_val_f1
            best_marker = "★" if is_best else ""

            print(f"  {epoch:3d}  {tr_loss:8.4f}  {tr_acc:6.4f}  "
                  f"{va_loss:8.4f}  {va_acc:6.4f}  {va_f1:6.4f}  "
                  f"{cur_lr:10.2e}  {elapsed:4.0f}s  {best_marker}")

            # Per-class satırı — her epoch
            recall_str = "  Recall → " + "  ".join(
                f"{cls[:4]}:{rec_dict[cls]:.2f}" for cls in labels
            )
            f1_str     = "  F1     → " + "  ".join(
                f"{cls[:4]}:{f1_dict[cls]:.2f}" for cls in labels
            )
            print(recall_str)
            print(f1_str)
            print()

            # Early stopping + best model kaydı
            if is_best:
                best_val_f1 = va_f1
                patience_cnt = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "label_names": labels,
                    "n_classes":   len(labels),
                    "val_f1":      best_val_f1,
                }, best_model_path)
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  ⏹  Early stopping (patience={PATIENCE})")
                    break

            # Ara checkpoint: her CHECKPOINT_EVERY epoch'ta CM kaydet
            if CHECKPOINT_EVERY > 0 and epoch % CHECKPOINT_EVERY == 0:
                print(f"  [Checkpoint ep{epoch}] Confusion matrix kaydediliyor...")
                plot_confusion_matrix(va_true, va_preds, labels, PLOTS_DIR,
                                      suffix=f"_val_ep{epoch:03d}")

    except KeyboardInterrupt:
        interrupted = True
        save_interrupt_report(
            model, val_loader, criterion, device,
            labels, history, PLOTS_DIR, current_epoch
        )
        # Mevcut modeli geçici olarak da kaydet
        interrupt_path = os.path.join(MODELS_DIR, f"interrupt_ep{current_epoch}.pt")
        torch.save({
            "epoch":       current_epoch,
            "model_state": model.state_dict(),
            "label_names": labels,
            "n_classes":   len(labels),
            "val_f1":      history["val_f1"][-1] if history["val_f1"] else 0.0,
        }, interrupt_path)
        print(f"  Mevcut model kaydedildi → {interrupt_path}")
        print("  (best_cnn.pt ayrıca korunuyor)\n")

    if interrupted:
        print("  Ctrl+C ile sonlandırıldı. Grafik ve raporlar kaydedildi.")
        return

    # ── 7. Test değerlendirme ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EN İYİ MODEL YÜKLENİYOR...")
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Epoch {ckpt['epoch']}  |  Val F1: {ckpt['val_f1']:.4f}")

    print("\n" + "=" * 60)
    print("  TEST SONUÇLARI")
    print("=" * 60)
    _, te_acc, te_preds, te_true = evaluate(model, test_loader, criterion, device)
    te_f1_mac = f1_score(te_true, te_preds, average="macro", zero_division=0)
    te_f1_wt  = f1_score(te_true, te_preds, average="weighted", zero_division=0)
    te_f1_dict, te_rec_dict = per_class_f1_recall(te_true, te_preds, labels)

    print(f"\n  Accuracy   : {te_acc:.4f}  ({te_acc:.1%})")
    print(f"  F1 Macro   : {te_f1_mac:.4f}   ← ASIL METRİK")
    print(f"  F1 Weighted: {te_f1_wt:.4f}")
    hedef_gec = "✅  HEDEF AŞILDI" if te_f1_mac >= 0.858 else "⚠️  HEDEF'E ULAŞILAMADI"
    print(f"  Hedef 0.858: {hedef_gec}")
    print()
    print_per_class(te_f1_dict, te_rec_dict, epoch_label="(Test)")
    print(classification_report(te_true, te_preds, target_names=labels, digits=3))

    # ── 8. Grafikler ──────────────────────────────────────────────
    plot_training_curves(history, PLOTS_DIR)
    plot_per_class_f1(history, labels, PLOTS_DIR)
    plot_confusion_matrix(te_true, te_preds, labels, PLOTS_DIR, suffix="_test_final")

    # Val confusion matrix da kaydet (son epoch val)
    _, _, va_preds_fin, va_true_fin = evaluate(model, val_loader, criterion, device)
    plot_confusion_matrix(va_true_fin, va_preds_fin, labels, PLOTS_DIR,
                          suffix="_val_final")

    # ── 9. Metadata kaydet ───────────────────────────────────────
    joblib.dump(le, os.path.join(MODELS_DIR, "cnn_label_encoder.pkl"))
    joblib.dump({
        "model_name":    "AirportCNN",
        "test_accuracy": float(te_acc),
        "f1_macro":      float(te_f1_mac),
        "label_names":   labels,
        "sr":            SR,
        "n_mels":        N_MELS,
        "n_fft":         N_FFT,
        "hop_fft":       HOP_FFT,
        "duration":      DURATION,
        "best_epoch":    int(ckpt["epoch"]),
        "version":       "v2",
    }, os.path.join(MODELS_DIR, "cnn_meta.pkl"))

    print("=" * 60)
    print("  🎉  CNN EĞİTİMİ TAMAMLANDI")
    print(f"     Model    : AirportCNN v2")
    print(f"     F1 Macro : {te_f1_mac:.1%}")
    print(f"     Epoch    : {ckpt['epoch']}")
    print(f"     Kayıt    : {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()