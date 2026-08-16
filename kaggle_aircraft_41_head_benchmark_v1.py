"""Compare five classification heads on frozen AST embeddings without test tuning.

The script is intentionally a controlled head experiment.  It uses the exact same
physical-airframe-disjoint split for every candidate, selects the winner only on
validation Macro-F1, and opens the independent test set once at the end.
"""
from __future__ import annotations

import copy
import json
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoFeatureExtractor, ASTModel


SEED = 42
SR = 16000
SECONDS = 10
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working/aircraft_41_head_benchmark_v1")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if DEVICE.type != "cuda":
    raise RuntimeError("Kaggle GPU acilmadi")


def locate_dataset() -> Path:
    candidates = list(Path("/kaggle/input").rglob("AIRCRAFT_AST_41CLASS_V1/manifest.csv"))
    if not candidates:
        raise FileNotFoundError("AIRCRAFT_AST_41CLASS_V1 veri seti Add Input ile eklenmedi")
    return candidates[0].parent


root = locate_dataset()
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(root / "manifest.csv")
df["path"] = df.path.map(lambda value: str(root / str(value)))
df["physical_airframe_id"] = df.physical_airframe_id.fillna("").astype(str)
labels = sorted(df.label.astype(str).unique())
label2id = {label: index for index, label in enumerate(labels)}
id2label = {index: label for label, index in label2id.items()}
df["target"] = df.label.map(label2id)
assert len(labels) == 41, f"41 yerine {len(labels)} sinif bulundu"

# A physical aircraft is never shared by train, validation, and test.
rng = random.Random(SEED)
parts = []
coverage = []
for label, group in df.groupby("label"):
    airframes = sorted(group.physical_airframe_id.unique())
    rng.shuffle(airframes)
    if len(airframes) >= 5:
        test_count = max(1, round(len(airframes) * 0.15))
        val_count = max(1, round(len(airframes) * 0.15))
    elif len(airframes) >= 3:
        test_count = val_count = 1
    else:
        test_count = val_count = 0
    test_ids = set(airframes[:test_count])
    val_ids = set(airframes[test_count:test_count + val_count])
    current = group.copy()
    current["split"] = current.physical_airframe_id.map(
        lambda item: "test" if item in test_ids else "validation" if item in val_ids else "train"
    )
    parts.append(current)
    coverage.append({
        "label": label,
        "recordings": int(len(group)),
        "physical_airframes": int(len(airframes)),
        "independent_test_ready": bool(len(airframes) >= 3),
    })

df = pd.concat(parts, ignore_index=True)
df.to_csv(OUT / "split_manifest.csv", index=False)
(OUT / "coverage_report.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
backbone = ASTModel.from_pretrained(BASE_MODEL, attn_implementation="eager").to(DEVICE).eval()
for parameter in backbone.parameters():
    parameter.requires_grad = False


def audio_crops(path: str) -> list[np.ndarray]:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    crop_size = SR * SECONDS
    if len(audio) <= crop_size:
        return [np.pad(audio, (0, crop_size - len(audio))).astype(np.float32)]
    starts = sorted({0, (len(audio) - crop_size) // 2, len(audio) - crop_size})
    return [audio[start:start + crop_size].astype(np.float32) for start in starts]


@torch.no_grad()
def extract_recording(path: str) -> np.ndarray:
    batch = extractor(audio_crops(path), sampling_rate=SR, return_tensors="pt")
    values = batch["input_values"].to(DEVICE)
    hidden = backbone(input_values=values).last_hidden_state
    # AST uses CLS and distillation tokens. Preserve every crop as training evidence.
    pooled = F.normalize((hidden[:, 0] + hidden[:, 1]) / 2.0, dim=1)
    return pooled.cpu().numpy().astype(np.float32)


cache_path = OUT / "ast_crop_embeddings.npz"
recording_embeddings: list[np.ndarray] = []
if cache_path.exists():
    cache = np.load(cache_path, allow_pickle=True)
    recording_embeddings = list(cache["items"])
else:
    for number, path in enumerate(df.path, 1):
        recording_embeddings.append(extract_recording(path))
        if number % 25 == 0 or number == len(df):
            print(f"Embedding {number}/{len(df)}")
    np.savez_compressed(cache_path, items=np.array(recording_embeddings, dtype=object))


def expand_split(split: str):
    indices = np.where(df.split.to_numpy() == split)[0]
    features, targets, owners = [], [], []
    for index in indices:
        for vector in recording_embeddings[index]:
            features.append(vector)
            targets.append(int(df.iloc[index].target))
            owners.append(int(index))
    return (
        torch.tensor(np.stack(features), dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
        np.asarray(owners, dtype=np.int64),
    )


x_train, y_train, train_owner = expand_split("train")
x_val, y_val, val_owner = expand_split("validation")
x_test, y_test, test_owner = expand_split("test")
print("Crop sayilari:", len(x_train), len(x_val), len(x_test))

counts = torch.bincount(y_train, minlength=len(labels)).float()
class_weights = counts.sum() / counts.clamp_min(1.0)
class_weights = torch.sqrt(class_weights / class_weights.mean()).clamp(0.35, 4.0).to(DEVICE)


class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(768, len(labels))

    def forward(self, features, targets=None):
        return self.classifier(features), None


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 384), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(384, 192), nn.GELU(), nn.Dropout(0.20), nn.Linear(192, len(labels)),
        )

    def forward(self, features, targets=None):
        return self.network(features), None


class CosineHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len(labels), 768) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(np.log(16.0), dtype=torch.float32))

    def cosine(self, features):
        return F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))

    def forward(self, features, targets=None):
        return self.cosine(features) * self.log_scale.exp().clamp(5.0, 40.0), None


class ArcFaceHead(CosineHead):
    def __init__(self, margin=0.25):
        super().__init__()
        self.margin = margin

    def forward(self, features, targets=None):
        cosine = self.cosine(features).clamp(-1 + 1e-6, 1 - 1e-6)
        inference_logits = cosine * self.log_scale.exp().clamp(5.0, 40.0)
        if targets is None:
            return inference_logits, None
        adjusted = cosine.clone()
        rows = torch.arange(len(targets), device=targets.device)
        target_cosine = cosine[rows, targets]
        adjusted[rows, targets] = torch.cos(torch.acos(target_cosine) + self.margin)
        train_logits = adjusted * self.log_scale.exp().clamp(5.0, 40.0)
        return inference_logits, train_logits


def aggregate_probabilities(probabilities: np.ndarray, owners: np.ndarray):
    unique = np.unique(owners)
    averaged = np.stack([probabilities[owners == owner].mean(axis=0) for owner in unique])
    truth = df.iloc[unique].target.to_numpy(dtype=np.int64)
    return unique, truth, averaged


@torch.no_grad()
def probabilities(model: nn.Module, features: torch.Tensor, batch_size=256):
    model.eval()
    output = []
    for start in range(0, len(features), batch_size):
        logits, _ = model(features[start:start + batch_size].to(DEVICE), None)
        output.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output)


def train_head(name: str, model: nn.Module):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    best_score = -1.0
    best_state = None
    patience = 25
    no_improvement = 0
    batch_size = 96
    generator = torch.Generator().manual_seed(SEED)

    for epoch in range(1, 251):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(permutation), batch_size):
            chosen = permutation[start:start + batch_size]
            features = x_train[chosen].to(DEVICE)
            targets = y_train[chosen].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            inference_logits, special_logits = model(features, targets)
            logits_for_loss = special_logits if special_logits is not None else inference_logits
            loss = F.cross_entropy(
                logits_for_loss, targets, weight=class_weights, label_smoothing=0.04
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(chosen)

        val_crop_prob = probabilities(model, x_val)
        _, val_truth, val_recording_prob = aggregate_probabilities(val_crop_prob, val_owner)
        val_prediction = val_recording_prob.argmax(axis=1)
        score = f1_score(val_truth, val_prediction, average="macro", zero_division=0)
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name} epoch={epoch:03d} loss={epoch_loss/len(x_train):.4f} valF1={score:.4f}")
        if no_improvement >= patience:
            break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "head": name, "labels": labels}, OUT / f"{name}.pt")
    return model, float(best_score)


candidates = {
    "linear": LinearHead(),
    "mlp": MLPHead(),
    "cosine": CosineHead(),
    "arcface": ArcFaceHead(),
}
trained = {}
validation_results = []
for name, candidate in candidates.items():
    print("\n" + "=" * 70 + f"\nTRAIN HEAD: {name.upper()}\n" + "=" * 70)
    trained[name], score = train_head(name, candidate)
    validation_results.append({"head": name, "validation_macro_f1": score})

# Non-parametric prototype head: centroids are made only from train recordings.
train_recording_vectors = []
train_recording_targets = []
for owner in np.unique(train_owner):
    vectors = x_train[train_owner == owner].numpy()
    train_recording_vectors.append(vectors.mean(axis=0))
    train_recording_targets.append(int(df.iloc[owner].target))
train_recording_vectors = F.normalize(torch.tensor(np.stack(train_recording_vectors)), dim=1)
train_recording_targets = np.asarray(train_recording_targets)
centroids = []
for class_id in range(len(labels)):
    selected = train_recording_vectors[train_recording_targets == class_id]
    centroids.append(F.normalize(selected.mean(dim=0), dim=0))
centroids = torch.stack(centroids).numpy()

def prototype_probability(features, owners, temperature):
    normalized = F.normalize(features, dim=1).numpy()
    logits = normalized @ centroids.T * temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
    return aggregate_probabilities(probs, owners)

best_temperature, best_prototype_score = None, -1.0
for temperature in [5.0, 10.0, 15.0, 20.0, 30.0, 40.0]:
    _, truth, prob = prototype_probability(x_val, val_owner, temperature)
    score = f1_score(truth, prob.argmax(axis=1), average="macro", zero_division=0)
    if score > best_prototype_score:
        best_temperature, best_prototype_score = temperature, float(score)
validation_results.append({
    "head": "prototype", "validation_macro_f1": best_prototype_score,
    "temperature": best_temperature,
})

winner = max(validation_results, key=lambda item: item["validation_macro_f1"])["head"]
print("\nVALIDATION SONUCLARI")
print(json.dumps(validation_results, indent=2))
print("SECILEN HEAD:", winner)

# The independent test is opened once, after the validation winner is fixed.
if winner == "prototype":
    test_indices, test_truth, test_prob = prototype_probability(x_test, test_owner, best_temperature)
else:
    crop_prob = probabilities(trained[winner], x_test)
    test_indices, test_truth, test_prob = aggregate_probabilities(crop_prob, test_owner)
test_prediction = test_prob.argmax(axis=1)

test_records = df.iloc[test_indices][["path", "label", "physical_airframe_id"]].copy()
test_records["truth_id"] = test_truth
test_records["prediction_id"] = test_prediction
test_records["prediction_label"] = [id2label[item] for item in test_prediction]
test_records["confidence"] = test_prob.max(axis=1)
test_records.to_csv(OUT / "test_recording_predictions.csv", index=False)

airframe_truth, airframe_prediction = [], []
for _, group in test_records.groupby("physical_airframe_id"):
    airframe_truth.append(Counter(group.truth_id).most_common(1)[0][0])
    airframe_prediction.append(Counter(group.prediction_id).most_common(1)[0][0])

report = {
    "experiment": "Frozen AST backbone classification-head benchmark",
    "classes_in_output": len(labels),
    "head_candidates": ["linear", "mlp", "cosine", "arcface", "prototype"],
    "selection_metric": "validation recording Macro-F1",
    "validation_results": validation_results,
    "selected_head": winner,
    "test_recording_accuracy": accuracy_score(test_truth, test_prediction),
    "test_recording_macro_f1": f1_score(test_truth, test_prediction, average="macro", zero_division=0),
    "test_airframe_accuracy": accuracy_score(airframe_truth, airframe_prediction),
    "test_airframe_macro_f1": f1_score(
        airframe_truth, airframe_prediction, average="macro", zero_division=0
    ),
    "scope_warning": "23 rare classes lack >=3 independent physical airframes.",
}
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(OUT / "classification_report.txt").write_text(
    classification_report(test_truth, test_prediction, labels=list(range(len(labels))),
                          target_names=labels, zero_division=0), encoding="utf-8"
)
(OUT / "labels.json").write_text(
    json.dumps({"label2id": label2id, "id2label": id2label}, indent=2), encoding="utf-8"
)
shutil.make_archive("/kaggle/working/aircraft_41_head_benchmark_v1", "zip", OUT)
print("\nNIHAI RAPOR")
print(json.dumps(report, indent=2))
print("ZIP: /kaggle/working/aircraft_41_head_benchmark_v1.zip")
