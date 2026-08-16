"""Train a validation-selected OTHER-vs-rest specialist beside the best 6-class AST.

Required Kaggle inputs:
  - GENERAL_CATEGORY_6CLASS_V1
  - general_6_masking_contrastive_sampler_only_v1.zip (unless still in working)

The independent test split is evaluated only after epoch/combination selection.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import shutil
import zipfile
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import ASTModel, AutoFeatureExtractor, get_cosine_schedule_with_warmup


SEED = 42
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 10
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
LABELS = ["AIRCRAFT", "AMBIENT", "OTHER", "SPEECH", "TRAFFIC", "WIND"]
OTHER_INDEX = LABELS.index("OTHER")
BASE_STEM = os.environ.get(
    "GENERAL_BASE_MODEL_STEM", "general_6_masking_contrastive_sampler_only_v1"
).strip()
OUTPUT_STEM = "general_6_other_specialist_v1"
OUTPUT = Path("/kaggle/working") / OUTPUT_STEM
MAX_EPOCHS = 14
PATIENCE = 5
BATCH_SIZE = 4
ACCUMULATION = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError("Kaggle GPU acilmadi. Session options bolumunden GPU T4 secin.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
OUTPUT.mkdir(parents=True, exist_ok=True)


def locate_dataset() -> Path:
    found = list(Path("/kaggle/input").rglob("GENERAL_CATEGORY_6CLASS_V1/manifest.csv"))
    if not found:
        raise FileNotFoundError("GENERAL_CATEGORY_6CLASS_V1 Kaggle Input olarak eklenmedi.")
    return found[0].parent


def locate_base_model() -> Path:
    roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    for root in roots:
        for weights in root.rglob("model.safetensors"):
            if weights.parent.name == BASE_STEM:
                return weights.parent
            config_path = weights.parent / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    config = {}
                if config.get("mode") == "masking_contrastive_sampler_only":
                    return weights.parent
    archives = []
    for root in roots:
        archives.extend(root.rglob(f"{BASE_STEM}.zip"))
    if not archives:
        raise FileNotFoundError(f"{BASE_STEM}.zip bulunamadi.")
    destination = Path("/kaggle/working/_other_specialist_base")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(destination)
    weights = list(destination.rglob("model.safetensors"))
    if not weights:
        raise RuntimeError("Temel model ZIP'inde model.safetensors yok.")
    return weights[0].parent


DATASET_ROOT = locate_dataset()
BASE_ROOT = locate_base_model()
manifest = pd.read_csv(DATASET_ROOT / "manifest.csv")
manifest["path"] = manifest["path"].map(lambda value: str(DATASET_ROOT / str(value)))
manifest["target"] = manifest["label"].map({label: i for i, label in enumerate(LABELS)}).astype(int)
manifest["other_target"] = (manifest["target"] == OTHER_INDEX).astype(int)
train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
validation_frame = manifest[manifest["split"] == "validation"].reset_index(drop=True)
test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)

print("GPU:", torch.cuda.get_device_name(0))
print("Temel model:", BASE_ROOT)
print("Veri:", DATASET_ROOT)
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


def fixed_crop(audio: np.ndarray, index: int) -> np.ndarray:
    size = SAMPLE_RATE * WINDOW_SECONDS
    if len(audio) <= size:
        return np.pad(audio, (0, size - len(audio))).astype(np.float32)
    maximum = len(audio) - size
    starts = [0, maximum // 4, maximum // 2, 3 * maximum // 4, maximum]
    return audio[starts[index] : starts[index] + size].astype(np.float32)


def augment(audio: np.ndarray) -> np.ndarray:
    output = audio.copy() * random.uniform(0.75, 1.20)
    if random.random() < 0.45:
        rms = float(np.sqrt(np.mean(output**2) + 1e-8))
        output += np.random.randn(len(output)).astype(np.float32) * rms * random.uniform(0.003, 0.025)
    if random.random() < 0.30 and len(output):
        output = np.roll(output, random.randint(-SAMPLE_RATE // 4, SAMPLE_RATE // 4))
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def values(audio: np.ndarray) -> torch.Tensor:
    return extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"][0]


class SpecialistDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = str(row["path"])
        if path not in self.cache:
            self.cache[path] = load_audio(path)
        clip = self.cache[path]
        first = values(augment(random_crop(clip)))
        second = values(augment(random_crop(clip)))
        return first, second, int(row["other_target"])


binary_counts = train_frame["other_target"].value_counts().to_dict()
sample_weights = [1.0 / binary_counts[int(target)] for target in train_frame["other_target"]]
sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(train_frame),
    replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)
loader = DataLoader(
    SpecialistDataset(train_frame),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)


class CosineClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len(LABELS), 768) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, features: torch.Tensor):
        return F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1)) * self.log_scale.exp().clamp(5.0, 40.0)


class OtherSpecialistAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ASTModel.from_pretrained(BASE_MODEL, attn_implementation="eager")
        self.classifier = CosineClassifier()
        self.projector = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 128))
        self.other_head = nn.Sequential(
            nn.LayerNorm(768), nn.Dropout(0.15), nn.Linear(768, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 2)
        )

    @staticmethod
    def mask(inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.clone()
        time_bins, frequency_bins = inputs.shape[1], inputs.shape[2]
        for row in range(len(inputs)):
            if torch.rand((), device=inputs.device) < 0.55:
                width = int(torch.randint(8, 33, (), device=inputs.device))
                start = int(torch.randint(0, max(1, time_bins - width), (), device=inputs.device))
                inputs[row, start : start + width, :] = 0
            if torch.rand((), device=inputs.device) < 0.45:
                width = int(torch.randint(4, 13, (), device=inputs.device))
                start = int(torch.randint(0, max(1, frequency_bins - width), (), device=inputs.device))
                inputs[row, :, start : start + width] = 0
        return inputs

    def forward(self, inputs: torch.Tensor, masking: bool = False):
        if self.training and masking:
            inputs = self.mask(inputs)
        hidden = self.backbone(input_values=inputs).last_hidden_state
        features = (hidden[:, 0] + hidden[:, 1]) / 2
        return self.classifier(features), self.other_head(features)


model = OtherSpecialistAST().to(DEVICE)
base_state = load_file(str(BASE_ROOT / "model.safetensors"), device=str(DEVICE))
missing, unexpected = model.load_state_dict(base_state, strict=False)
allowed_missing = {key for key in model.state_dict() if key.startswith("other_head.")}
if set(missing) != allowed_missing or unexpected:
    raise RuntimeError(f"Temel model uyusmazligi: missing={missing}, unexpected={unexpected}")

for parameter in model.parameters():
    parameter.requires_grad = False
for parameter in model.other_head.parameters():
    parameter.requires_grad = True

blocks = list(model.backbone.encoder.layer)
optimizer = torch.optim.AdamW(
    [
        {"params": model.other_head.parameters(), "lr": 5e-4},
        {"params": model.backbone.parameters(), "lr": 8e-6},
    ],
    weight_decay=0.025,
)
updates = math.ceil(len(loader) / ACCUMULATION) * MAX_EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(updates * 0.08)), updates)
scaler = torch.amp.GradScaler("cuda")


@torch.no_grad()
def evaluate(frame: pd.DataFrame):
    model.eval()
    truths, base_probabilities, other_probabilities = [], [], []
    for index, row in frame.iterrows():
        audio = load_audio(str(row["path"]))
        inputs = torch.stack([values(fixed_crop(audio, crop)) for crop in range(5)]).to(DEVICE)
        with torch.amp.autocast("cuda"):
            base_logits, other_logits = model(inputs, masking=False)
        base_probabilities.append(torch.softmax(base_logits.float(), 1).mean(0).cpu().numpy())
        other_probabilities.append(float(torch.softmax(other_logits.float(), 1)[:, 1].mean().cpu()))
        truths.append(int(row["target"]))
        if (index + 1) % 100 == 0:
            print(f"  Degerlendirme {index + 1}/{len(frame)}")
    return np.asarray(truths), np.stack(base_probabilities), np.asarray(other_probabilities)


def combine(base_probability: np.ndarray, other_probability: np.ndarray, alpha: float, bias: float):
    logits = np.log(np.clip(base_probability, 1e-8, 1.0))
    specialist_log_odds = np.log(np.clip(other_probability, 1e-5, 1 - 1e-5) / np.clip(1 - other_probability, 1e-5, 1))
    logits[:, OTHER_INDEX] += alpha * specialist_log_odds + bias
    return logits.argmax(1)


def select_combination(truth, base_probability, other_probability):
    best = None
    for alpha in np.arange(0.0, 2.01, 0.25):
        for bias in np.arange(-1.5, 1.51, 0.15):
            prediction = combine(base_probability, other_probability, float(alpha), float(bias))
            score = f1_score(truth, prediction, average="macro", zero_division=0)
            candidate = (float(score), -abs(float(bias)), -float(alpha), float(alpha), float(bias))
            if best is None or candidate > best:
                best = candidate
    return {"macro_f1": best[0], "alpha": best[3], "bias": best[4]}


best_state = None
best_epoch = 0
best_validation_f1 = -1.0
best_rule = None
history = []
stale = 0

for epoch in range(1, MAX_EPOCHS + 1):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    open_blocks = 0 if epoch <= 3 else 2
    if open_blocks:
        for block in blocks[-open_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.backbone.layernorm.parameters():
            parameter.requires_grad = True

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for step, (first, second, target) in enumerate(loader, 1):
        first = first.to(DEVICE, non_blocking=True)
        second = second.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda"):
            _, first_logits = model(first, masking=True)
            _, second_logits = model(second, masking=True)
            loss = (
                F.cross_entropy(first_logits, target, label_smoothing=0.03)
                + F.cross_entropy(second_logits, target, label_smoothing=0.03)
            ) / (2 * ACCUMULATION)
        scaler.scale(loss).backward()
        if step % ACCUMULATION == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.item()) * ACCUMULATION * len(target)

    val_truth, val_base, val_other = evaluate(validation_frame)
    base_prediction = val_base.argmax(1)
    base_f1 = f1_score(val_truth, base_prediction, average="macro", zero_division=0)
    rule = select_combination(val_truth, val_base, val_other)
    selected_f1 = max(float(base_f1), float(rule["macro_f1"]))
    use_specialist = bool(rule["macro_f1"] > base_f1 + 1e-5)
    row = {
        "epoch": epoch,
        "open_blocks": open_blocks,
        "loss": loss_sum / len(train_frame),
        "validation_base_macro_f1": float(base_f1),
        "validation_selected_macro_f1": selected_f1,
        "use_specialist": use_specialist,
        "alpha": rule["alpha"] if use_specialist else 0.0,
        "bias": rule["bias"] if use_specialist else 0.0,
    }
    history.append(row)
    print(
        f"epoch={epoch:02d}/{MAX_EPOCHS} loss={row['loss']:.4f} blocks={open_blocks} "
        f"baseF1={base_f1:.4f} selectedF1={selected_f1:.4f} "
        f"specialist={use_specialist} alpha={row['alpha']:.2f} bias={row['bias']:.2f}"
    )
    if selected_f1 > best_validation_f1 + 1e-5:
        best_validation_f1 = selected_f1
        best_epoch = epoch
        best_rule = {
            "use_specialist": bool(use_specialist),
            "alpha": float(row["alpha"]),
            "bias": float(row["bias"]),
        }
        best_state = copy.deepcopy(model.state_dict())
        stale = 0
        print("  -> Yeni en iyi model")
    else:
        stale += 1
    if epoch >= 7 and stale >= PATIENCE:
        print("Erken durdurma")
        break

if best_state is None:
    raise RuntimeError("En iyi model secilemedi.")

model.load_state_dict(best_state)
pd.DataFrame(history).to_csv(OUTPUT / "training_history.csv", index=False)
test_truth, test_base, test_other = evaluate(test_frame)
base_prediction = test_base.argmax(1)
if best_rule["use_specialist"]:
    test_prediction = combine(test_base, test_other, best_rule["alpha"], best_rule["bias"])
else:
    test_prediction = base_prediction

base_metrics = {
    "accuracy": float(accuracy_score(test_truth, base_prediction)),
    "macro_f1": float(f1_score(test_truth, base_prediction, average="macro", zero_division=0)),
}
selected_metrics = {
    "accuracy": float(accuracy_score(test_truth, test_prediction)),
    "macro_f1": float(f1_score(test_truth, test_prediction, average="macro", zero_division=0)),
}
report_text = classification_report(
    test_truth, test_prediction, labels=list(range(len(LABELS))), target_names=LABELS, zero_division=0
)
(OUTPUT / "classification_report.txt").write_text(report_text, encoding="utf-8")

matrix = confusion_matrix(test_truth, test_prediction, labels=list(range(len(LABELS))))
figure, axis = plt.subplots(figsize=(8, 7))
image = axis.imshow(matrix, cmap="Blues")
axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
axis.set_yticks(range(len(LABELS)), LABELS)
axis.set_xlabel("Tahmin")
axis.set_ylabel("Gercek")
axis.set_title("Genel 6 sinif + OTHER uzmani")
for row in range(len(LABELS)):
    for column in range(len(LABELS)):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
figure.colorbar(image, ax=axis)
figure.tight_layout()
figure.savefig(OUTPUT / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close(figure)

predictions = test_frame[["path", "label", "source_recording_id", "dataset", "subtype"]].copy()
predictions["base_prediction"] = [LABELS[i] for i in base_prediction]
predictions["selected_prediction"] = [LABELS[i] for i in test_prediction]
predictions["other_probability"] = test_other
predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)

state = {key: value.detach().cpu().contiguous() for key, value in best_state.items()}
save_file(state, str(OUTPUT / "model.safetensors"))
result = {
    "experiment": "general_6_other_specialist_v1",
    "base_model": BASE_STEM,
    "selection_uses_test_set": False,
    "best_epoch": best_epoch,
    "best_validation_macro_f1": best_validation_f1,
    "selected_rule": best_rule,
    "independent_test_base": base_metrics,
    "independent_test_selected": selected_metrics,
}
(OUTPUT / "independent_test_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
(OUTPUT / "config.json").write_text(
    json.dumps({"architecture": "AST 6-class head + binary OTHER specialist", "labels": LABELS, **result}, indent=2),
    encoding="utf-8",
)
shutil.make_archive(str(OUTPUT), "zip", OUTPUT)

print("\nNIHAI RAPOR")
print(json.dumps(result, indent=2))
print(report_text)
print(f"ZIP: {OUTPUT}.zip")

del model, best_state, state, loader
gc.collect()
torch.cuda.empty_cache()
