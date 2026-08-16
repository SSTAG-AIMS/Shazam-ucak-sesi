"""
╔══════════════════════════════════════════════════════════════╗
║       Train EfficientNet-B0  —  5 Sınıf Transfer Learning  ║
║   Mel Spectrogram → RGB → EfficientNet-B0 (ImageNet)        ║
╚══════════════════════════════════════════════════════════════╝

Sıralama:
  1. python dataset_builder_v3.py      → manifest_v3.csv (zaten var)
  2. python train_efficientnet.py      → best_efficientnet.pt üretir

Strateji  —  2 Aşamalı Eğitim:
  Aşama 1 — Freeze (FREEZE_EPOCHS epoch)
    EfficientNet backbone tamamen dondurulur.
    Sadece yeni classifier head eğitilir.
    LR: HEAD_LR
    Amaç: head'i makul başlangıca getir, backbone bozulmasın.

  Aşama 2 — Fine-tune (FINETUNE_EPOCHS epoch)
    Backbone'un son 3 bloğu (features[6..8]) açılır.
    Backbone ağırlıkları: BACKBONE_LR (çok düşük)
    Head ağırlıkları: HEAD_LR / 5
    Amaç: backbone'u havalimanı seslerine adapte et.

Mel → RGB dönüşümü:
  Mel spectrogram (1, 128, ~216) dB ölçeğinde üretilir.
  [0,1] normalize edilir → 3 kanala kopyalanır (R=G=B).
  224×224 resize → ImageNet mean/std normalize.
  Bilgi kaybı SIFIR. Upscale yönünde minimal interpolasyon.

Çıktılar:
  models/best_efficientnet.pt
  models/efficientnet_label_encoder.pkl
  models/efficientnet_meta.pkl
  outputs/training_efficientnet/

Gereksinimler:
  pip install torch torchaudio torchvision librosa
              scikit-learn joblib numpy pandas matplotlib seaborn tqdm
"""

import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as TA
import torchvision.transforms as TV
import torchvision.models as tvm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, recall_score)

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "cache", "manifest_v5.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "training_efficientnet")

# Mel Spectrogram — train_cnn.py ile AYNI
SR       = 22050
N_MELS   = 128
N_FFT    = 2048
HOP_FFT  = 512
DURATION = 5.0

# Veri bölünmesi — train_cnn.py ile AYNI (aynı test seti)
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
RANDOM_SEED = 42

# ── Aşama 1: Freeze ──────────────────────────────────────────
FREEZE_EPOCHS = 15
HEAD_LR       = 1e-3
BATCH_SIZE    = 32

# ── Aşama 2: Fine-tune ───────────────────────────────────────
FINETUNE_EPOCHS  = 40
BACKBONE_LR      = 1e-4   # backbone için çok düşük LR — kritik
FINETUNE_HEAD_LR = 2e-4   # head için de biraz düşür
WEIGHT_DECAY     = 1e-4

# Early stopping — her iki aşama için ayrı
PATIENCE = 10

# Ara confusion matrix (0 = devre dışı)
CHECKPOINT_EVERY = 5

# ImageNet normalize (EfficientNet bu değerleri bekliyor)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Sınıf ağırlıkları — train_cnn.py ile AYNI, DOKUNMA
MANUAL_CLASS_WEIGHTS = {
    "AIRCRAFT": 1.5,
    "SPEECH":   2.0,
    "TRAFFIC":  1.5,
    "WIND":     2.5,
    "AMBIENT":  2.0,
    "OTHER":    1.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# 🔊  DATASET
# ================================================================

class MelRGBDataset(Dataset):
    """
    Mel spectrogram → dB → normalize [0,1] → 3 kanal → 224x224 resize
    → ImageNet normalize → EfficientNet giriş formatı.

    Augmentation: SpecAugment (time + freq masking) eğitimde aktif.
    """

    def __init__(self, records: list, augment: bool = False):
        self.records    = records
        self.augment    = augment
        self.target_len = int(SR * DURATION)

        # Mel + dB dönüşümleri
        self.mel_tf = TA.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT,
            hop_length=HOP_FFT, n_mels=N_MELS
        )
        self.db_tf = TA.AmplitudeToDB(top_db=80)

        # SpecAugment
        if augment:
            self.time_masks = nn.ModuleList([
                TA.TimeMasking(time_mask_param=30) for _ in range(2)
            ])
            self.freq_masks = nn.ModuleList([
                TA.FrequencyMasking(freq_mask_param=15) for _ in range(2)
            ])
        else:
            self.time_masks = []
            self.freq_masks = []

        # EfficientNet giriş dönüşümleri
        self.to_rgb = TV.Compose([
            TV.Resize((224, 224), antialias=True),
            TV.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Ses yükle
        try:
            y, _ = librosa.load(rec["path"], sr=SR, mono=True,
                                 duration=DURATION + 0.5)
        except Exception:
            y = np.zeros(self.target_len, dtype=np.float32)

        # Sabit uzunluk (merkez-kırp veya sıfır-doldur)
        if len(y) >= self.target_len:
            start = (len(y) - self.target_len) // 2
            y = y[start: start + self.target_len]
        else:
            y = np.pad(y, (0, self.target_len - len(y)))

        y_t  = torch.FloatTensor(y).unsqueeze(0)     # (1, samples)
        spec = self.mel_tf(y_t)                       # (1, N_MELS, T)
        spec = self.db_tf(spec)                       # dB ölçeği

        # SpecAugment (eğitimde)
        if self.augment:
            for m in self.time_masks:
                spec = m(spec)
            for m in self.freq_masks:
                spec = m(spec)

         # CMVN — her frekans bandını standardize et, mikrofon kanal etkisini azalt
        mean = spec.mean(dim=-1, keepdim=True)
        std  = spec.std(dim=-1, keepdim=True) + 1e-8
        spec = (spec - mean) / std
        # [0, 1] aralığına çek
        s_min = spec.min()
        s_max = spec.max()
        if s_max > s_min:
            spec = (spec - s_min) / (s_max - s_min)
        else:
            spec = torch.zeros_like(spec)

        # 3 kanala kopyala: (1, H, W) → (3, H, W)
        spec_rgb = spec.repeat(3, 1, 1)               # (3, N_MELS, T)

        # Resize 224x224 + ImageNet normalize
        spec_rgb = self.to_rgb(spec_rgb)              # (3, 224, 224)

        return spec_rgb, rec["label_enc"]


# ================================================================
# 🏗️  MODEL
# ================================================================

class EfficientNetAirport(nn.Module):
    """
    EfficientNet-B0 backbone (ImageNet pretrained) +
    yeni classifier head (5 sınıf için).

    Mimari:
      backbone.features[0..5]  → Phase 1'de tamamen dondurulur
      backbone.features[6..8]  → Phase 2'de açılır (fine-tune)
      classifier               → her zaman eğitilir
    """

    def __init__(self, n_classes: int = 5):
        super().__init__()

        weights  = tvm.EfficientNet_B0_Weights.DEFAULT
        backbone = tvm.efficientnet_b0(weights=weights)

        # Features bloklarını al (8 blok + head conv)
        self.features   = backbone.features      # ModuleList benzeri Sequential
        self.avgpool    = backbone.avgpool        # AdaptiveAvgPool2d

        # EfficientNet-B0 classifier giriş boyutu: 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def freeze_backbone(self):
        """Aşama 1: tüm features dondurulur."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("  [Freeze] Backbone tamamen donduruldu.")
        self._print_trainable()

    def unfreeze_last_blocks(self, n_blocks: int = 3):
        """
        Aşama 2: features'ın son n_blocks bloğunu aç.
        EfficientNet-B0: features[0..8], son 3 → [6, 7, 8]
        """
        total = len(self.features)
        for i, block in enumerate(self.features):
            unfreeze = (i >= total - n_blocks)
            for param in block.parameters():
                param.requires_grad = unfreeze

        print(f"  [Unfreeze] Son {n_blocks} blok açıldı "
              f"(features[{total-n_blocks}..{total-1}]).")
        self._print_trainable()

    def _print_trainable(self):
        total   = sum(p.numel() for p in self.parameters())
        trainbl = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Eğitilebilir: {trainbl:,} / {total:,} parametre "
              f"({trainbl/total:.1%})")


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


def per_class_metrics(y_true, y_pred, label_names):
    f1s     = f1_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return dict(zip(label_names, f1s)), dict(zip(label_names, recalls))


def print_per_class(f1_dict, rec_dict, tag=""):
    print(f"  ── Per-Class {tag} ─────────────────────────────────")
    print(f"  {'Sınıf':12s}  {'F1':>6}  {'Recall':>7}")
    for cls in sorted(f1_dict):
        warn = " ⚠" if rec_dict[cls] < 0.5 else ""
        bar  = "▓" * int(f1_dict[cls] * 10)
        print(f"  {cls:12s}  {f1_dict[cls]:6.4f}  {rec_dict[cls]:7.4f}  {bar}{warn}")
    print()


# ================================================================
# 📊  PLOT FONKSİYONLARI
# ================================================================

def plot_training_curves(history: dict, phase: str, save_dir: str, suffix: str = ""):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    axes[0, 0].plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_acc"], label="Train", color="#2196F3")
    axes[0, 1].plot(epochs, history["val_acc"],   label="Val",   color="#F44336")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["val_f1"], color="#4CAF50", linewidth=2, label="Val F1 Macro")
    ax.axhline(0.858, color="orange", linestyle="--", linewidth=1.5, label="Hedef: 0.858")
    if history["val_f1"]:
        best_f1 = max(history["val_f1"])
        best_ep = history["val_f1"].index(best_f1) + 1
        ax.scatter([best_ep], [best_f1], color="red", zorder=5)
        ax.annotate(f" Best: {best_f1:.4f} (ep{best_ep})",
                    xy=(best_ep, best_f1), fontsize=8, color="red")
    ax.set_title("F1 Macro (Val)"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if history.get("lr_head"):
        ax.plot(epochs, history["lr_head"], label="Head LR",     color="#9C27B0")
    if history.get("lr_backbone"):
        non_zero = [(e, v) for e, v in
                    zip(epochs, history["lr_backbone"]) if v > 0]
        if non_zero:
            ep_bb, lr_bb = zip(*non_zero)
            ax.plot(ep_bb, lr_bb, label="Backbone LR", color="#FF9800",
                    linestyle="--")
    ax.set_title("Learning Rate"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(f"EfficientNet-B0 Eğitim — {phase}{suffix}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"training_curves_{phase.lower().replace(' ', '_')}{suffix}.png"
    out   = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def plot_per_class_f1_curve(history: dict, label_names: list,
                             phase: str, save_dir: str, suffix: str = ""):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    colors = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cls in enumerate(label_names):
        key = f"val_f1_{cls}"
        if key in history and history[key]:
            ax.plot(epochs, history[key], label=cls,
                    color=colors[i % len(colors)], linewidth=1.8)

    ax.axhline(0.858, color="gray", linestyle="--", linewidth=1, label="Hedef")
    ax.set_title(f"Per-Class F1 (Val) — {phase}{suffix}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"per_class_f1_{phase.lower().replace(' ', '_')}{suffix}.png"
    out   = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def plot_confusion_matrix(y_true, y_pred, label_names,
                           save_dir: str, suffix: str = ""):
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

    # AMBIENT recall özellikle işaretle
    if "AMBIENT" in label_names:
        ai = label_names.index("AMBIENT")
        amb_rec = cm[ai, ai]
        fig.text(0.5, 0.01,
                 f"AMBIENT Recall: {amb_rec:.3f}"
                 f"{'  ✅' if amb_rec >= 0.7 else '  ⚠️ DÜŞÜK'}",
                 ha="center", fontsize=11,
                 color="green" if amb_rec >= 0.7 else "red")

    plt.suptitle(f"Confusion Matrix — EfficientNet-B0{suffix}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = f"confusion_matrix{suffix}.png"
    out   = os.path.join(save_dir, fname)
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def save_mel_samples(val_ds, label_names, save_dir, n_per_class=2):
    """
    Mel görselini RGB olarak kaydet — kalite kontrol için.
    Her sınıftan n_per_class örnek.
    """
    os.makedirs(save_dir, exist_ok=True)
    from collections import defaultdict
    seen = defaultdict(int)
    samples = []
    for rec in val_ds.records:
        cls = label_names[rec["label_enc"]]
        if seen[cls] < n_per_class:
            samples.append(rec)
            seen[cls] += 1
        if all(v >= n_per_class for v in seen.values()):
            break

    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, rec in zip(axes, samples):
        spec_rgb, lbl = val_ds[val_ds.records.index(rec)]
        # ImageNet denormalize → görüntü
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img  = (spec_rgb * std + mean).clamp(0, 1)
        ax.imshow(img.permute(1, 2, 0).numpy(), aspect="auto", origin="lower")
        ax.set_title(f"{label_names[lbl]}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Mel RGB Örnekler — Kalite Kontrol", fontsize=11)
    plt.tight_layout()
    out = os.path.join(save_dir, "mel_rgb_quality_check.png")
    plt.savefig(out, dpi=120); plt.close()
    print(f"  → Mel kalite kontrol: {out}")


# ================================================================
# 🔄  AŞAMA EĞİTİM DÖNGÜSÜ
# ================================================================

def run_phase(phase_name, model, train_loader, val_loader,
              criterion, optimizer, scheduler, n_epochs,
              label_names, best_model_path, plots_dir,
              global_best_f1, history):
    """
    Tek bir eğitim aşaması (Freeze veya Fine-tune).
    global_best_f1: önceki aşamadan gelen en iyi F1 (Fine-tune için)
    Döndürür: güncellenmiş global_best_f1
    """
    patience_cnt   = 0
    current_epoch  = 0

    print(f"\n{'='*60}")
    print(f"  AŞAMA: {phase_name}  ({n_epochs} epoch)")
    print(f"{'='*60}")
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>6}  "
          f"{'VaLoss':>8}  {'VaAcc':>6}  {'F1Mac':>6}  {'Süre':>5}  {'Best'}")
    print("  " + "─" * 62)

    try:
        for epoch in range(1, n_epochs + 1):
            current_epoch = epoch
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, va_preds, va_true = evaluate(
                model, val_loader, criterion, device)
            va_f1 = f1_score(va_true, va_preds, average="macro", zero_division=0)

            # LR kayıt
            lr_head     = optimizer.param_groups[-1]["lr"]   # head grubu
            lr_backbone = (optimizer.param_groups[0]["lr"]
                           if len(optimizer.param_groups) > 1 else 0.0)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            is_best = va_f1 > global_best_f1
            mark    = "★" if is_best else ""

            print(f"  {epoch:3d}  {tr_loss:8.4f}  {tr_acc:6.4f}  "
                  f"{va_loss:8.4f}  {va_acc:6.4f}  {va_f1:6.4f}  "
                  f"{elapsed:4.0f}s  {mark}")

            # Per-class
            f1_dict, rec_dict = per_class_metrics(va_true, va_preds, label_names)
            recall_str = "  Recall → " + "  ".join(
                f"{c[:4]}:{rec_dict[c]:.2f}" for c in label_names)
            f1_str     = "  F1     → " + "  ".join(
                f"{c[:4]}:{f1_dict[c]:.2f}" for c in label_names)
            print(recall_str)
            print(f1_str)
            print()

            # History güncelle
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            history["val_f1"].append(va_f1)
            history["lr_head"].append(lr_head)
            history["lr_backbone"].append(lr_backbone)
            for cls in label_names:
                history[f"val_f1_{cls}"].append(f1_dict.get(cls, 0.0))

            # Checkpoint
            if is_best:
                global_best_f1 = va_f1
                patience_cnt   = 0
                torch.save({
                    "epoch":       epoch,
                    "phase":       phase_name,
                    "model_state": model.state_dict(),
                    "label_names": label_names,
                    "n_classes":   len(label_names),
                    "val_f1":      global_best_f1,
                }, best_model_path)
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  ⏹  Early stopping (patience={PATIENCE})")
                    break

            # Ara CM
            if CHECKPOINT_EVERY > 0 and epoch % CHECKPOINT_EVERY == 0:
                plot_confusion_matrix(
                    va_true, va_preds, label_names, plots_dir,
                    suffix=f"_val_{phase_name.split()[0].lower()}_ep{epoch:03d}")

    except KeyboardInterrupt:
        print(f"\n\n  ⚡  Ctrl+C — {phase_name} anlık rapor kaydediliyor...")
        _, _, preds, true = evaluate(model, val_loader, criterion, device)
        f1m = f1_score(true, preds, average="macro", zero_division=0)
        f1d, rcd = per_class_metrics(true, preds, label_names)
        print(f"  Val F1 Macro ({phase_name}, ep {current_epoch}): {f1m:.4f}")
        print_per_class(f1d, rcd, tag=f"Ctrl+C ep{current_epoch}")
        print(classification_report(true, preds, target_names=label_names, digits=3))

        suffix = f"_interrupt_{phase_name.split()[0].lower()}_ep{current_epoch}"
        if len(history["train_loss"]) > 0:
            plot_training_curves(history, phase_name, plots_dir, suffix=suffix)
            plot_per_class_f1_curve(history, label_names, phase_name,
                                    plots_dir, suffix=suffix)
        plot_confusion_matrix(true, preds, label_names, plots_dir, suffix=suffix)

        intr_path = os.path.join(
            MODELS_DIR, f"efficientnet_interrupt_{phase_name.split()[0].lower()}"
                        f"_ep{current_epoch}.pt")
        torch.save({
            "epoch": current_epoch, "phase": phase_name,
            "model_state": model.state_dict(),
            "label_names": label_names, "n_classes": len(label_names),
            "val_f1": history["val_f1"][-1] if history["val_f1"] else 0.0,
        }, intr_path)
        print(f"  Model kaydedildi → {intr_path}")
        print("  best_efficientnet.pt korunuyor.\n")
        sys.exit(0)

    return global_best_f1


# ================================================================
# 🏃  ANA PIPELINE
# ================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    print("=" * 60)
    print("  Train EfficientNet-B0  —  5 Sınıf Transfer Learning")
    print(f"  Device: {device}")
    print("=" * 60)

    # ── 1. Manifest yükle ────────────────────────────────────────
    if not os.path.exists(MANIFEST_CSV):
        print(f"[HATA] Manifest yok: {MANIFEST_CSV}")
        print("       Önce: python dataset_builder_v3.py")
        raise SystemExit(1)

    df = pd.read_csv(MANIFEST_CSV)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"\n  Toplam örnek: {len(df)}")

    dist = Counter(df["label"])
    print("\n── Sınıf Dağılımı ──────────────────────────────────────")
    for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl:10s} {cnt:5d}  {'█' * (cnt // 20)}")

    # ── 2. Label encode ──────────────────────────────────────────
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    labels = list(le.classes_)
    print(f"\n  Sınıflar: {labels}")

    # ── 3. Bölünme — train_cnn.py ile AYNI seed ve boyutlar ──────
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["label_enc"], random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val_df["label_enc"], random_state=RANDOM_SEED
    )
    print(f"  Eğitim: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # Val dağılımı
    print("\n── Val Sınıf Dağılımı ──────────────────────────────────")
    for lbl, cnt in sorted(Counter(val_df["label"]).items()):
        print(f"  {lbl:10s} {cnt:4d}")

    # ── 4. Dataset & DataLoader ──────────────────────────────────
    train_records = train_df.to_dict("records")
    val_records   = val_df.to_dict("records")
    test_records  = test_df.to_dict("records")

    train_ds = MelRGBDataset(train_records, augment=True)
    val_ds   = MelRGBDataset(val_records,   augment=False)
    test_ds  = MelRGBDataset(test_records,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"),
                          )
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
                          )
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
                          )

    # ── 5. Mel kalite kontrol görseli ────────────────────────────
    print("\n  Mel RGB kalite kontrol görseli oluşturuluyor...")
    save_mel_samples(val_ds, labels, PLOTS_DIR, n_per_class=2)

    # ── 6. Model ─────────────────────────────────────────────────
    model = EfficientNetAirport(n_classes=len(labels)).to(device)

    weights = torch.FloatTensor([
        MANUAL_CLASS_WEIGHTS[labels[i]] for i in range(len(labels))
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_model_path = os.path.join(MODELS_DIR, "best_efficientnet.pt")
    global_best_f1  = -1.0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_f1":     [], "lr_head":   [], "lr_backbone": [],
    }
    for cls in labels:
        history[f"val_f1_{cls}"] = []

    # ════════════════════════════════════════════════════════════
    # AŞAMA 1 — FREEZE: sadece classifier head eğitilir
    # ════════════════════════════════════════════════════════════
    model.freeze_backbone()

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=FREEZE_EPOCHS, eta_min=1e-5
    )

    global_best_f1 = run_phase(
        phase_name="Aşama 1 Freeze",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        n_epochs=FREEZE_EPOCHS,
        label_names=labels,
        best_model_path=best_model_path,
        plots_dir=PLOTS_DIR,
        global_best_f1=global_best_f1,
        history=history,
    )

    # Aşama 1 sonunda grafik kaydet
    plot_training_curves(history, "Aşama 1 Freeze", PLOTS_DIR,
                         suffix=f"_after_phase1")
    plot_per_class_f1_curve(history, labels, "Aşama 1 Freeze", PLOTS_DIR,
                            suffix=f"_after_phase1")

    # ════════════════════════════════════════════════════════════
    # AŞAMA 2 — FINE-TUNE: son 3 blok açılır, farklı LR
    # ════════════════════════════════════════════════════════════
    model.unfreeze_last_blocks(n_blocks=3)

    # En iyi Aşama 1 modelini yükle — fine-tune onun üzerinden başlasın
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"\n  Aşama 1 best modeli yüklendi (ep {ckpt['epoch']}, "
              f"F1 {ckpt['val_f1']:.4f})")

    # İki farklı param grubu — farklı LR
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params     = list(model.classifier.parameters())

    optimizer_p2 = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": FINETUNE_HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=FINETUNE_EPOCHS, eta_min=1e-6
    )

    global_best_f1 = run_phase(
        phase_name="Aşama 2 FineTune",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        n_epochs=FINETUNE_EPOCHS,
        label_names=labels,
        best_model_path=best_model_path,
        plots_dir=PLOTS_DIR,
        global_best_f1=global_best_f1,
        history=history,
    )

    # ── 7. Test değerlendirme — en iyi model ─────────────────────
    print("\n" + "=" * 60)
    print("  EN İYİ MODEL YÜKLENİYOR (test)...")
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  {ckpt['phase']}  |  Epoch {ckpt['epoch']}  |  "
          f"Val F1: {ckpt['val_f1']:.4f}")

    print("\n" + "=" * 60)
    print("  TEST SONUÇLARI")
    print("=" * 60)
    _, te_acc, te_preds, te_true = evaluate(model, test_loader, criterion, device)
    te_f1_mac = f1_score(te_true, te_preds, average="macro", zero_division=0)
    te_f1_wt  = f1_score(te_true, te_preds, average="weighted", zero_division=0)
    te_f1d, te_rcd = per_class_metrics(te_true, te_preds, labels)

    print(f"\n  Accuracy   : {te_acc:.4f}  ({te_acc:.1%})")
    print(f"  F1 Macro   : {te_f1_mac:.4f}   ← ASIL METRİK")
    print(f"  F1 Weighted: {te_f1_wt:.4f}")
    flag = "✅ HEDEF AŞILDI" if te_f1_mac >= 0.858 else "⚠️ HEDEF'E ULAŞILAMADI"
    print(f"  Hedef 0.858: {flag}")
    print()
    print_per_class(te_f1d, te_rcd, tag="(Test)")
    print(classification_report(te_true, te_preds, target_names=labels, digits=3))

    # ── 8. Final grafikler ────────────────────────────────────────
    plot_training_curves(history, "Tüm Aşamalar", PLOTS_DIR)
    plot_per_class_f1_curve(history, labels, "Tüm Aşamalar", PLOTS_DIR)
    plot_confusion_matrix(te_true, te_preds, labels, PLOTS_DIR,
                          suffix="_test_final")
    _, _, va_preds_fin, va_true_fin = evaluate(model, val_loader, criterion, device)
    plot_confusion_matrix(va_true_fin, va_preds_fin, labels, PLOTS_DIR,
                          suffix="_val_final")

    # ── 9. Metadata kaydet ───────────────────────────────────────
    joblib.dump(le, os.path.join(MODELS_DIR, "efficientnet_label_encoder.pkl"))
    joblib.dump({
        "model_name":    "EfficientNet-B0",
        "test_accuracy": float(te_acc),
        "f1_macro":      float(te_f1_mac),
        "label_names":   labels,
        "sr":            SR,
        "n_mels":        N_MELS,
        "n_fft":         N_FFT,
        "hop_fft":       HOP_FFT,
        "duration":      DURATION,
        "best_epoch":    int(ckpt["epoch"]),
        "best_phase":    ckpt["phase"],
        "version":       "v1",
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std":  IMAGENET_STD,
    }, os.path.join(MODELS_DIR, "efficientnet_meta.pkl"))

    print("=" * 60)
    print("  🎉  EfficientNet-B0 EĞİTİMİ TAMAMLANDI")
    print(f"     Model    : EfficientNet-B0")
    print(f"     F1 Macro : {te_f1_mac:.1%}")
    print(f"     Epoch    : {ckpt['epoch']}  ({ckpt['phase']})")
    print(f"     Kayıt    : {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()