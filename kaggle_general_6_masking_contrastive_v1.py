"""Six-category AST experiment: masking, contrastive, or their hybrid.

Run one mode at a time by setting GENERAL_EXPERIMENT_MODE to one of:
  masking_only
  contrastive_only
  masking_contrastive_hybrid
  masking_contrastive_sampler_only

All modes use the same fixed manifest, initialization, sampler, validation
selection rule, and independent test set. This makes the comparison controlled.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import shutil
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    ASTModel,
    AutoFeatureExtractor,
    get_cosine_schedule_with_warmup,
)


SEED = 42
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 10
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
LABELS = ["AIRCRAFT", "AMBIENT", "OTHER", "SPEECH", "TRAFFIC", "WIND"]
MAX_EPOCHS = 24
PATIENCE = 6
BATCH_SIZE = 4
ACCUMULATION = 2
CONTRASTIVE_WEIGHT = 0.12
TEMPERATURE = 0.10

RUN_MODE = os.environ.get("GENERAL_EXPERIMENT_MODE", "masking_only").strip().lower()
MODE_CONFIG = {
    "masking_only": {"masking": True, "contrastive": False, "loss_class_weight": True},
    "contrastive_only": {"masking": False, "contrastive": True, "loss_class_weight": True},
    "masking_contrastive_hybrid": {"masking": True, "contrastive": True, "loss_class_weight": True},
    # Controlled imbalance ablation: the sampler already increases rare-class
    # exposure. Removing the second correction in CE is expected to reduce
    # false WIND predictions without removing WIND from training.
    "masking_contrastive_sampler_only": {
        "masking": True,
        "contrastive": True,
        "loss_class_weight": False,
    },
}
if RUN_MODE not in MODE_CONFIG:
    raise ValueError(f"Gecersiz GENERAL_EXPERIMENT_MODE: {RUN_MODE}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    raise RuntimeError("Kaggle GPU acilmadi. Session options bolumunden GPU T4 secin.")

OUTPUT_STEM = f"general_6_{RUN_MODE}_v1"
OUTPUT = Path("/kaggle/working") / OUTPUT_STEM
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def locate_dataset() -> Path:
    manifests = list(Path("/kaggle/input").rglob("GENERAL_CATEGORY_6CLASS_V1/manifest.csv"))
    if not manifests:
        raise FileNotFoundError(
            "GENERAL_CATEGORY_6CLASS_V1 Kaggle Input olarak eklenmedi."
        )
    return manifests[0].parent


DATASET_ROOT = locate_dataset()
manifest = pd.read_csv(DATASET_ROOT / "manifest.csv")
required_columns = {"path", "label", "split", "source_recording_id", "sha256"}
missing = required_columns - set(manifest.columns)
if missing:
    raise RuntimeError(f"Manifest kolonlari eksik: {sorted(missing)}")

manifest["path"] = manifest["path"].map(lambda item: str(DATASET_ROOT / str(item)))
manifest["target"] = manifest["label"].map({label: index for index, label in enumerate(LABELS)})
if manifest["target"].isna().any():
    unknown = sorted(manifest.loc[manifest["target"].isna(), "label"].unique())
    raise RuntimeError(f"Bilinmeyen etiketler: {unknown}")
manifest["target"] = manifest["target"].astype(int)

train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
validation_frame = manifest[manifest["split"] == "validation"].reset_index(drop=True)
test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)
if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
    raise RuntimeError("Train, validation veya test bolumlerinden biri bos.")

train_ids = set(train_frame["source_recording_id"].astype(str))
validation_ids = set(validation_frame["source_recording_id"].astype(str))
test_ids = set(test_frame["source_recording_id"].astype(str))
if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
    raise RuntimeError("source_recording_id sizintisi bulundu.")

print("GPU:", torch.cuda.get_device_name(0))
print("Deney:", RUN_MODE, MODE_CONFIG[RUN_MODE])
print("Veri kok dizini:", DATASET_ROOT)
print(pd.crosstab(manifest["label"], manifest["split"]).to_string())

extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.nan_to_num(audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1e-6:
        audio = audio / max(peak, 0.10)
    return audio


def random_crop(audio: np.ndarray) -> np.ndarray:
    size = SAMPLE_RATE * WINDOW_SECONDS
    if len(audio) <= size:
        return np.pad(audio, (0, size - len(audio))).astype(np.float32)
    start = random.randint(0, len(audio) - size)
    return audio[start : start + size].astype(np.float32)


def fixed_crop(audio: np.ndarray, crop_index: int) -> np.ndarray:
    size = SAMPLE_RATE * WINDOW_SECONDS
    if len(audio) <= size:
        return np.pad(audio, (0, size - len(audio))).astype(np.float32)
    maximum = len(audio) - size
    starts = [0, maximum // 4, maximum // 2, 3 * maximum // 4, maximum]
    return audio[starts[crop_index] : starts[crop_index] + size].astype(np.float32)


def waveform_augment(audio: np.ndarray) -> np.ndarray:
    output = audio.copy()
    output *= random.uniform(0.75, 1.20)
    if random.random() < 0.45:
        rms = float(np.sqrt(np.mean(output**2) + 1e-8))
        noise = np.random.randn(len(output)).astype(np.float32)
        output += noise * rms * random.uniform(0.003, 0.025)
    if random.random() < 0.30 and len(output):
        output = np.roll(output, random.randint(-SAMPLE_RATE // 4, SAMPLE_RATE // 4))
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def extract_values(audio: np.ndarray) -> torch.Tensor:
    return extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )["input_values"][0]


class PairedAudioDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = str(row["path"])
        if path not in self.cache:
            self.cache[path] = load_audio(path)
        audio = self.cache[path]
        first = extract_values(waveform_augment(random_crop(audio)))
        second = extract_values(waveform_augment(random_crop(audio)))
        return first, second, int(row["target"])


train_counts = train_frame["target"].value_counts().to_dict()
sample_weights = [
    min(5.0, math.sqrt(len(train_frame) / (len(LABELS) * train_counts[int(target)])))
    for target in train_frame["target"]
]
sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(train_frame),
    replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)
train_loader = DataLoader(
    PairedAudioDataset(train_frame),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

counts = torch.bincount(
    torch.tensor(train_frame["target"].to_numpy()), minlength=len(LABELS)
).float()
class_weights = torch.sqrt(counts.sum() / counts.clamp_min(1))
class_weights = (class_weights / class_weights.mean()).clamp(0.4, 4.0).to(DEVICE)
if not MODE_CONFIG[RUN_MODE]["loss_class_weight"]:
    class_weights = torch.ones_like(class_weights)


class CosineClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len(LABELS), 768) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(
            F.normalize(features, dim=1), F.normalize(self.weight, dim=1)
        ) * self.log_scale.exp().clamp(5.0, 40.0)


class GeneralCategoryAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ASTModel.from_pretrained(
            BASE_MODEL, attn_implementation="eager"
        )
        self.classifier = CosineClassifier()
        self.projector = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    @staticmethod
    def spectrogram_mask(values: torch.Tensor) -> torch.Tensor:
        values = values.clone()
        time_bins, frequency_bins = values.shape[1], values.shape[2]
        for row in range(len(values)):
            if torch.rand((), device=values.device) < 0.55:
                width = int(torch.randint(8, 33, (), device=values.device))
                start = int(
                    torch.randint(0, max(1, time_bins - width), (), device=values.device)
                )
                values[row, start : start + width, :] = 0
            if torch.rand((), device=values.device) < 0.45:
                width = int(torch.randint(4, 13, (), device=values.device))
                start = int(
                    torch.randint(
                        0, max(1, frequency_bins - width), (), device=values.device
                    )
                )
                values[row, :, start : start + width] = 0
        return values

    def forward(self, values: torch.Tensor, use_masking: bool):
        if self.training and use_masking:
            values = self.spectrogram_mask(values)
        hidden = self.backbone(input_values=values).last_hidden_state
        features = (hidden[:, 0] + hidden[:, 1]) / 2
        logits = self.classifier(features)
        projection = F.normalize(self.projector(features), dim=1)
        return logits, projection


def supervised_contrastive(
    first: torch.Tensor, second: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    features = torch.cat([first, second], dim=0).float()
    repeated_targets = torch.cat([targets, targets], dim=0)
    similarity = features @ features.T / TEMPERATURE
    self_mask = torch.eye(
        len(features), dtype=torch.bool, device=features.device
    )
    positive = repeated_targets[:, None].eq(repeated_targets[None, :]) & ~self_mask
    similarity = similarity.masked_fill(self_mask, -torch.inf)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_log_probability = log_probability.masked_fill(~positive, 0.0)
    return -positive_log_probability.sum(1).div(
        positive.sum(1).clamp_min(1)
    ).mean()


@torch.no_grad()
def evaluate(model: GeneralCategoryAST, frame: pd.DataFrame):
    model.eval()
    truths, probabilities = [], []
    for index, row in frame.iterrows():
        audio = load_audio(str(row["path"]))
        values = torch.stack(
            [extract_values(fixed_crop(audio, crop)) for crop in range(5)]
        ).to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits, _ = model(values, use_masking=False)
        probabilities.append(torch.softmax(logits, dim=1).mean(0).cpu().numpy())
        truths.append(int(row["target"]))
        if (index + 1) % 100 == 0:
            print(f"  Degerlendirme {index + 1}/{len(frame)}")
    probabilities = np.stack(probabilities)
    truths = np.asarray(truths)
    predictions = probabilities.argmax(axis=1)
    return truths, predictions, probabilities


config = MODE_CONFIG[RUN_MODE]
model = GeneralCategoryAST().to(DEVICE)
blocks = list(model.backbone.encoder.layer)
for parameter in model.backbone.parameters():
    parameter.requires_grad = False

optimizer = torch.optim.AdamW(
    [
        {"params": model.classifier.parameters(), "lr": 7e-4},
        {"params": model.projector.parameters(), "lr": 5e-4},
        {"params": model.backbone.parameters(), "lr": 1.5e-5},
    ],
    weight_decay=0.02,
)
total_updates = math.ceil(len(train_loader) / ACCUMULATION) * MAX_EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, int(total_updates * 0.08)),
    num_training_steps=total_updates,
)
scaler = torch.amp.GradScaler("cuda")

best_validation_f1 = -1.0
best_epoch = 0
best_state = None
stale_epochs = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    open_blocks = 0 if epoch <= 3 else 2 if epoch <= 8 else 4
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if open_blocks:
        for block in blocks[-open_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.backbone.layernorm.parameters():
            parameter.requires_grad = True

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    for step, (first, second, targets) in enumerate(train_loader, 1):
        first = first.to(DEVICE, non_blocking=True)
        second = second.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits_first, projection_first = model(first, config["masking"])
            logits_second, projection_second = model(second, config["masking"])
            classification_loss = (
                F.cross_entropy(
                    logits_first,
                    targets,
                    weight=class_weights,
                    label_smoothing=0.03,
                )
                + F.cross_entropy(
                    logits_second,
                    targets,
                    weight=class_weights,
                    label_smoothing=0.03,
                )
            ) / 2
            if config["contrastive"]:
                contrastive_loss = supervised_contrastive(
                    projection_first, projection_second, targets
                )
            else:
                contrastive_loss = classification_loss.new_zeros(())
            loss = (
                classification_loss + CONTRASTIVE_WEIGHT * contrastive_loss
            ) / ACCUMULATION

        scaler.scale(loss).backward()
        if step % ACCUMULATION == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.item()) * ACCUMULATION * len(targets)

    validation_truth, validation_prediction, _ = evaluate(
        model, validation_frame
    )
    validation_accuracy = accuracy_score(
        validation_truth, validation_prediction
    )
    validation_macro_f1 = f1_score(
        validation_truth,
        validation_prediction,
        average="macro",
        zero_division=0,
    )
    epoch_result = {
        "epoch": epoch,
        "open_blocks": open_blocks,
        "train_loss": running_loss / len(train_frame),
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
    }
    history.append(epoch_result)
    print(
        f"{RUN_MODE} epoch={epoch:02d}/{MAX_EPOCHS} "
        f"loss={epoch_result['train_loss']:.4f} blocks={open_blocks} "
        f"valAcc={validation_accuracy:.4f} valF1={validation_macro_f1:.4f}"
    )
    if validation_macro_f1 > best_validation_f1 + 1e-5:
        best_validation_f1 = float(validation_macro_f1)
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        stale_epochs = 0
        print("  -> Yeni en iyi model")
    else:
        stale_epochs += 1
    if epoch >= 10 and stale_epochs >= PATIENCE:
        print("Erken durdurma")
        break

if best_state is None:
    raise RuntimeError("En iyi model kaydedilemedi.")

model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(OUTPUT / "training_history.csv", index=False)
test_truth, test_prediction, test_probability = evaluate(model, test_frame)

test_accuracy = accuracy_score(test_truth, test_prediction)
test_macro_f1 = f1_score(
    test_truth, test_prediction, average="macro", zero_division=0
)
report_text = classification_report(
    test_truth,
    test_prediction,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    zero_division=0,
)
(OUTPUT / "classification_report.txt").write_text(report_text, encoding="utf-8")

matrix = confusion_matrix(
    test_truth, test_prediction, labels=list(range(len(LABELS)))
)
figure, axis = plt.subplots(figsize=(8, 7))
image = axis.imshow(matrix, cmap="Blues")
axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
axis.set_yticks(range(len(LABELS)), LABELS)
axis.set_xlabel("Tahmin")
axis.set_ylabel("Gercek")
axis.set_title(f"Genel kategori - {RUN_MODE}")
for row in range(len(LABELS)):
    for column in range(len(LABELS)):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
figure.colorbar(image, ax=axis)
figure.tight_layout()
figure.savefig(OUTPUT / "confusion_matrix.png", dpi=180)
plt.close(figure)

predictions = test_frame[
    ["path", "label", "source_recording_id", "dataset", "subtype"]
].copy()
predictions["prediction"] = [LABELS[index] for index in test_prediction]
predictions["confidence"] = test_probability.max(axis=1)
predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)

cpu_state = {
    key: value.detach().cpu().contiguous() for key, value in best_state.items()
}
save_file(cpu_state, str(OUTPUT / "model.safetensors"))

report = {
    "experiment": RUN_MODE,
    "labels": LABELS,
    "masking": config["masking"],
    "contrastive": config["contrastive"],
    "loss_class_weight": config["loss_class_weight"],
    "imbalance_strategy": (
        "sqrt_inverse_sampler_plus_sqrt_inverse_loss"
        if config["loss_class_weight"]
        else "sqrt_inverse_sampler_only"
    ),
    "best_epoch": best_epoch,
    "best_validation_macro_f1": best_validation_f1,
    "independent_test_accuracy": test_accuracy,
    "independent_test_macro_f1": test_macro_f1,
    "train_count": len(train_frame),
    "validation_count": len(validation_frame),
    "independent_test_count": len(test_frame),
    "selection_rule": "Best epoch selected only with validation Macro-F1",
}
(OUTPUT / "independent_test_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
(OUTPUT / "config.json").write_text(
    json.dumps(
        {
            "architecture": "ASTModel + CosineClassifier + contrastive projector",
            "base_model": BASE_MODEL,
            "labels": LABELS,
            "mode": RUN_MODE,
            "sample_rate": SAMPLE_RATE,
            "evaluation_windows": 5,
            "contrastive_weight": CONTRASTIVE_WEIGHT if config["contrastive"] else 0,
            "temperature": TEMPERATURE,
            "loss_class_weight": config["loss_class_weight"],
        },
        indent=2,
    ),
    encoding="utf-8",
)
shutil.make_archive(str(Path("/kaggle/working") / OUTPUT_STEM), "zip", OUTPUT)

print("\nNIHAI RAPOR")
print(json.dumps(report, indent=2))
print(report_text)
print(f"ZIP: /kaggle/working/{OUTPUT_STEM}.zip")

# The three controlled modes can be run sequentially in the same notebook.
# Release the previous mode's GPU tensors and persistent DataLoader workers.
del model, best_state, cpu_state, train_loader
gc.collect()
torch.cuda.empty_cache()
