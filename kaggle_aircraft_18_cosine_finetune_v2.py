"""V2: leakage-free 18-class AST partial fine-tuning with cosine classifier.

Only labels with at least three independent physical airframes are eligible.
The classifier is warmed up first, then 2/4/6 AST blocks are opened gradually.
Light spectrogram masking and five-window inference improve robustness.
"""
from __future__ import annotations

import copy
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoFeatureExtractor, ASTModel, get_cosine_schedule_with_warmup


SEED = 42
SR = 16000
SECONDS = 10
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
MAX_EPOCHS = 36
PATIENCE = 9
BATCH_SIZE = 6
ACCUMULATION = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working/aircraft_18_cosine_finetune_v2")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if DEVICE.type != "cuda":
    raise RuntimeError("Kaggle GPU acilmadi")


def locate_dataset() -> Path:
    found = list(Path("/kaggle/input").rglob("AIRCRAFT_AST_41CLASS_V1/manifest.csv"))
    if not found:
        raise FileNotFoundError("AIRCRAFT_AST_41CLASS_V1 Add Input ile eklenmedi")
    return found[0].parent


root = locate_dataset()
OUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(root / "manifest.csv")
raw["physical_airframe_id"] = raw.physical_airframe_id.fillna("").astype(str)
raw["path"] = raw.path.map(lambda value: str(root / str(value)))

coverage = (
    raw.groupby("label")
       .agg(recordings=("path", "nunique"), physical_airframes=("physical_airframe_id", "nunique"))
       .reset_index()
)
supported_labels = sorted(
    coverage.loc[coverage.physical_airframes >= 3, "label"].astype(str).tolist()
)
unsupported_labels = sorted(
    coverage.loc[coverage.physical_airframes < 3, "label"].astype(str).tolist()
)
if len(supported_labels) != 18:
    print(f"UYARI: Beklenen 18 yerine {len(supported_labels)} desteklenen sinif bulundu")

df = raw[raw.label.isin(supported_labels)].copy()
label2id = {label: index for index, label in enumerate(supported_labels)}
id2label = {index: label for label, index in label2id.items()}
df["target"] = df.label.map(label2id)

# Physical-airframe-disjoint split. The same aircraft cannot leak across sets.
rng = random.Random(SEED)
parts = []
for label, group in df.groupby("label"):
    ids = sorted(group.physical_airframe_id.unique())
    rng.shuffle(ids)
    if len(ids) >= 5:
        test_count = max(1, round(len(ids) * 0.15))
        val_count = max(1, round(len(ids) * 0.15))
    else:
        test_count = val_count = 1
    test_ids = set(ids[:test_count])
    validation_ids = set(ids[test_count:test_count + val_count])
    current = group.copy()
    current["split"] = current.physical_airframe_id.map(
        lambda item: "test" if item in test_ids else "validation" if item in validation_ids else "train"
    )
    parts.append(current)

df = pd.concat(parts, ignore_index=True)
df.to_csv(OUT / "split_manifest.csv", index=False)
coverage.to_csv(OUT / "coverage_report.csv", index=False)
(OUT / "scope.json").write_text(json.dumps({
    "supported_labels": supported_labels,
    "unsupported_labels": unsupported_labels,
    "policy": "Unsupported labels remain available to Shazam but ML returns UNKNOWN when unmatched.",
}, indent=2), encoding="utf-8")

for label, group in df.groupby("label"):
    sets = {name: set(part.physical_airframe_id) for name, part in group.groupby("split")}
    assert sets.get("train", set()).isdisjoint(sets.get("validation", set()))
    assert sets.get("train", set()).isdisjoint(sets.get("test", set()))
    assert sets.get("validation", set()).isdisjoint(sets.get("test", set()))

print("GPU:", torch.cuda.get_device_name(0))
print("Desteklenen sinif:", len(supported_labels), supported_labels)
print("Shazam/UNKNOWN kapsami:", len(unsupported_labels), unsupported_labels)
print(pd.crosstab(df.label, df.split).to_string())

extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    audio = np.nan_to_num(audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / max(peak, 0.10)
    return audio


def crop_audio(audio: np.ndarray, mode: str, crop_index: int = 0) -> np.ndarray:
    required = SR * SECONDS
    if len(audio) <= required:
        return np.pad(audio, (0, required - len(audio))).astype(np.float32)
    if mode == "train":
        start = random.randint(0, len(audio) - required)
    else:
        maximum = len(audio) - required
        positions = [0, maximum // 4, maximum // 2, (3 * maximum) // 4, maximum]
        start = positions[crop_index]
    return audio[start:start + required].astype(np.float32)


class TrainDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        audio = crop_audio(load_audio(row.path), "train")
        # Mild waveform augmentation; label identity is never synthesized.
        if random.random() < 0.50:
            gain = 10 ** (random.uniform(-4.0, 4.0) / 20.0)
            audio = audio * gain
        if random.random() < 0.35:
            noise_scale = random.uniform(0.0002, 0.002)
            audio = audio + np.random.normal(0, noise_scale, size=audio.shape).astype(np.float32)
        values = extractor(audio, sampling_rate=SR, return_tensors="pt")["input_values"][0]
        return values, int(row.target)


train_df = df[df.split == "train"].reset_index(drop=True)
validation_df = df[df.split == "validation"].reset_index(drop=True)
test_df = df[df.split == "test"].reset_index(drop=True)

counts = train_df.target.value_counts().to_dict()
sample_weights = [
    min(5.0, math.sqrt(len(train_df) / (len(supported_labels) * counts[int(target)])))
    for target in train_df.target
]
sampler = WeightedRandomSampler(
    sample_weights, num_samples=len(train_df), replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)
train_loader = DataLoader(
    TrainDataset(train_df), batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=2, pin_memory=True, persistent_workers=True,
)


class CosineClassifier(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(classes, 768) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, features):
        cosine = F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))
        return cosine * self.log_scale.exp().clamp(5.0, 40.0)


class AircraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ASTModel.from_pretrained(BASE_MODEL, attn_implementation="eager")
        self.head = CosineClassifier(len(supported_labels))

    def forward(self, input_values, apply_specaugment=False):
        if apply_specaugment and self.training:
            # AST input is a time x frequency spectrogram. Masking short stripes
            # prevents memorising a single event without changing the label.
            input_values = input_values.clone()
            time_bins, frequency_bins = input_values.shape[1], input_values.shape[2]
            for row in range(len(input_values)):
                if torch.rand((), device=input_values.device) < 0.55:
                    width = int(torch.randint(8, 33, (), device=input_values.device))
                    start = int(torch.randint(0, max(1, time_bins - width), (), device=input_values.device))
                    input_values[row, start:start + width, :] = 0
                if torch.rand((), device=input_values.device) < 0.45:
                    width = int(torch.randint(4, 13, (), device=input_values.device))
                    start = int(torch.randint(0, max(1, frequency_bins - width), (), device=input_values.device))
                    input_values[row, :, start:start + width] = 0
        hidden = self.backbone(input_values=input_values).last_hidden_state
        pooled = (hidden[:, 0] + hidden[:, 1]) / 2.0
        return self.head(pooled)


model = AircraftModel().to(DEVICE)
for parameter in model.backbone.parameters():
    parameter.requires_grad = False
for parameter in model.head.parameters():
    parameter.requires_grad = True

blocks = model.backbone.encoder.layer
assert len(blocks) == 12


def set_open_blocks(number: int):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    if number > 0:
        for block in blocks[-number:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.backbone.layernorm.parameters():
            parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True


# All prospective parameters are registered now; requires_grad controls the schedule.
optimizer = torch.optim.AdamW([
    {"params": model.head.parameters(), "lr": 5e-4, "weight_decay": 2e-3},
    {"params": blocks[-6:].parameters(), "lr": 1e-5, "weight_decay": 1e-2},
    {"params": model.backbone.layernorm.parameters(), "lr": 2e-5, "weight_decay": 1e-2},
])
steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, steps_per_epoch * 2),
    num_training_steps=steps_per_epoch * MAX_EPOCHS,
)
scaler = torch.amp.GradScaler("cuda")

class_counts = torch.bincount(
    torch.tensor(train_df.target.to_numpy()), minlength=len(supported_labels)
).float()
class_weights = torch.sqrt(class_counts.sum() / class_counts.clamp_min(1.0))
class_weights = (class_weights / class_weights.mean()).clamp(0.40, 4.0).to(DEVICE)


@torch.no_grad()
def evaluate(frame: pd.DataFrame):
    model.eval()
    probabilities = []
    truths = []
    for _, row in frame.iterrows():
        crop_values = []
        audio = load_audio(row.path)
        for crop_index in range(5):
            crop = crop_audio(audio, "evaluation", crop_index)
            crop_values.append(
                extractor(crop, sampling_rate=SR, return_tensors="pt")["input_values"][0]
            )
        values = torch.stack(crop_values).to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(values)
        probability = torch.softmax(logits, dim=1).mean(dim=0)
        probabilities.append(probability.cpu().numpy())
        truths.append(int(row.target))
    probabilities = np.stack(probabilities)
    truths = np.asarray(truths)
    predictions = probabilities.argmax(axis=1)
    return truths, predictions, probabilities


def airframe_metrics(frame, truth, prediction):
    evidence = frame[["physical_airframe_id"]].copy()
    evidence["truth"] = truth
    evidence["prediction"] = prediction
    air_truth, air_prediction = [], []
    for _, group in evidence.groupby("physical_airframe_id"):
        air_truth.append(Counter(group.truth).most_common(1)[0][0])
        air_prediction.append(Counter(group.prediction).most_common(1)[0][0])
    return (
        accuracy_score(air_truth, air_prediction),
        f1_score(air_truth, air_prediction, average="macro", zero_division=0),
    )


best_score = -1.0
best_epoch = 0
best_state = None
history = []
no_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    open_blocks = 0 if epoch <= 3 else 2 if epoch <= 8 else 4 if epoch <= 18 else 6
    set_open_blocks(open_blocks)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    for step, (values, targets) in enumerate(train_loader, 1):
        values = values.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(values, apply_specaugment=True)
            loss = F.cross_entropy(
                logits, targets, weight=class_weights, label_smoothing=0.04
            ) / ACCUMULATION
        scaler.scale(loss).backward()
        if step % ACCUMULATION == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # If AMP detected overflow it skips optimizer.step; the LR schedule
            # must not advance in that case.
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.item()) * ACCUMULATION * len(targets)

    val_truth, val_prediction, _ = evaluate(validation_df)
    val_accuracy = accuracy_score(val_truth, val_prediction)
    val_macro_f1 = f1_score(val_truth, val_prediction, average="macro", zero_division=0)
    val_air_accuracy, val_air_macro_f1 = airframe_metrics(
        validation_df, val_truth, val_prediction
    )
    history.append({
        "epoch": epoch, "open_blocks": open_blocks,
        "train_loss": running_loss / len(train_df),
        "validation_recording_accuracy": val_accuracy,
        "validation_recording_macro_f1": val_macro_f1,
        "validation_airframe_accuracy": val_air_accuracy,
        "validation_airframe_macro_f1": val_air_macro_f1,
    })
    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} loss={running_loss/len(train_df):.4f} "
        f"blocks={open_blocks} valRecF1={val_macro_f1:.4f} valAirF1={val_air_macro_f1:.4f}"
    )
    if val_air_macro_f1 > best_score + 1e-5:
        best_score = float(val_air_macro_f1)
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        no_improvement = 0
        print("  -> Yeni en iyi model")
    else:
        no_improvement += 1
    if epoch >= 10 and no_improvement >= PATIENCE:
        print("Erken durdurma")
        break

model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)

test_truth, test_prediction, test_probability = evaluate(test_df)
test_accuracy = accuracy_score(test_truth, test_prediction)
test_macro_f1 = f1_score(test_truth, test_prediction, average="macro", zero_division=0)
test_air_accuracy, test_air_macro_f1 = airframe_metrics(test_df, test_truth, test_prediction)

predictions = test_df[["path", "label", "physical_airframe_id"]].copy()
predictions["prediction"] = [id2label[item] for item in test_prediction]
predictions["confidence"] = test_probability.max(axis=1)
predictions.to_csv(OUT / "test_predictions.csv", index=False)

# Safetensors is an inspectable weight file; configuration records the exact architecture.
cpu_state = {key: value.detach().cpu().contiguous() for key, value in best_state.items()}
save_file(cpu_state, str(OUT / "model.safetensors"))
(OUT / "config.json").write_text(json.dumps({
    "architecture": "ASTModel + CosineClassifier",
    "base_model": BASE_MODEL,
    "labels": supported_labels,
    "label2id": label2id,
    "best_epoch": best_epoch,
    "best_validation_airframe_macro_f1": best_score,
    "training_schedule": "epochs 1-3 head; 4-8 last 2; 9-18 last 4; 19+ last 6 blocks",
    "evaluation_windows": 5,
    "regularization": "waveform gain/noise + light time/frequency masking",
}, indent=2), encoding="utf-8")

report = {
    "classes_in_ml_head": len(supported_labels),
    "unsupported_shazam_or_unknown_classes": len(unsupported_labels),
    "best_epoch": best_epoch,
    "best_validation_airframe_macro_f1": best_score,
    "test_recording_accuracy": test_accuracy,
    "test_recording_macro_f1": test_macro_f1,
    "test_airframe_accuracy": test_air_accuracy,
    "test_airframe_macro_f1": test_air_macro_f1,
}
(OUT / "independent_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(OUT / "classification_report.txt").write_text(
    classification_report(test_truth, test_prediction, labels=list(range(len(supported_labels))),
                          target_names=supported_labels, zero_division=0), encoding="utf-8"
)
shutil.make_archive("/kaggle/working/aircraft_18_cosine_finetune_v2", "zip", OUT)
print("\nNIHAI RAPOR")
print(json.dumps(report, indent=2))
print("ZIP: /kaggle/working/aircraft_18_cosine_finetune_v2.zip")
