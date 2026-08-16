"""Post-hoc error analysis and validation-only calibration for GENERAL 6 hybrid.

Kaggle inputs required:
  - GENERAL_CATEGORY_6CLASS_V1 dataset
  - general_6_masking_contrastive_hybrid_v1.zip (or extracted directory)

This script does not retrain AST. It reconstructs the saved hybrid model,
evaluates validation/test with the original five-window protocol, learns only
a small temperature/class-bias calibrator on validation, and then reports the
untouched independent-test result for both raw and calibrated decisions.
"""
from __future__ import annotations

import json
import math
import os
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
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import ASTModel, AutoFeatureExtractor


SEED = 42
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 10
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
LABELS = ["AIRCRAFT", "AMBIENT", "OTHER", "SPEECH", "TRAFFIC", "WIND"]
MODEL_STEM = os.environ.get(
    "GENERAL_MODEL_STEM",
    "general_6_masking_contrastive_hybrid_v1",
).strip()
OUTPUT_STEM = f"{MODEL_STEM}_error_analysis"
OUTPUT = Path("/kaggle/working") / OUTPUT_STEM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError("Kaggle GPU acilmadi. Session options bolumunden GPU T4 secin.")

torch.manual_seed(SEED)
np.random.seed(SEED)
OUTPUT.mkdir(parents=True, exist_ok=True)


def locate_dataset() -> Path:
    manifests = list(Path("/kaggle/input").rglob("GENERAL_CATEGORY_6CLASS_V1/manifest.csv"))
    if not manifests:
        raise FileNotFoundError("GENERAL_CATEGORY_6CLASS_V1 Kaggle Input olarak eklenmedi.")
    return manifests[0].parent


def locate_model_dir() -> Path:
    direct = []
    search_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    weight_files = []
    for root in search_roots:
        weight_files.extend(root.rglob("model.safetensors"))
    for weights in weight_files:
        config_path = weights.parent / "config.json"
        if weights.parent.name == MODEL_STEM:
            direct.append(weights.parent)
            continue
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
            if config.get("mode") == "masking_contrastive_hybrid":
                direct.append(weights.parent)
    if direct:
        return direct[0]

    archives = []
    for root in search_roots:
        archives.extend(root.rglob(f"{MODEL_STEM}.zip"))
    if not archives:
        raise FileNotFoundError(
            f"{MODEL_STEM}.zip Kaggle Input olarak eklenmedi. Dünkü hibrit ZIP'i yükleyin."
        )
    extracted = Path("/kaggle/working/_hybrid_model_extract")
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(extracted)
    candidates = list(extracted.rglob("model.safetensors"))
    if not candidates:
        raise RuntimeError("Hibrit ZIP içinde model.safetensors bulunamadi.")
    return candidates[0].parent


DATASET_ROOT = locate_dataset()
MODEL_ROOT = locate_model_dir()
manifest = pd.read_csv(DATASET_ROOT / "manifest.csv")
manifest["path"] = manifest["path"].map(lambda value: str(DATASET_ROOT / str(value)))
manifest["target"] = manifest["label"].map({label: i for i, label in enumerate(LABELS)}).astype(int)
validation_frame = manifest[manifest["split"] == "validation"].reset_index(drop=True)
test_frame = manifest[manifest["split"] == "test"].reset_index(drop=True)

print("GPU:", torch.cuda.get_device_name(0))
print("Veri:", DATASET_ROOT)
print("Hibrit model:", MODEL_ROOT)
print("Validation:", len(validation_frame), "Bagimsiz test:", len(test_frame))

extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.nan_to_num(audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1e-6:
        audio = audio / max(peak, 0.10)
    return audio


def fixed_crop(audio: np.ndarray, crop_index: int) -> np.ndarray:
    size = SAMPLE_RATE * WINDOW_SECONDS
    if len(audio) <= size:
        return np.pad(audio, (0, size - len(audio))).astype(np.float32)
    maximum = len(audio) - size
    starts = [0, maximum // 4, maximum // 2, 3 * maximum // 4, maximum]
    return audio[starts[crop_index] : starts[crop_index] + size].astype(np.float32)


def extract_values(audio: np.ndarray) -> torch.Tensor:
    return extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_values"][0]


class CosineClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len(LABELS), 768) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1)) * self.log_scale.exp().clamp(5.0, 40.0)


class GeneralCategoryAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ASTModel.from_pretrained(BASE_MODEL, attn_implementation="eager")
        self.classifier = CosineClassifier()
        self.projector = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 128)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_values=values).last_hidden_state
        features = (hidden[:, 0] + hidden[:, 1]) / 2
        return self.classifier(features)


model = GeneralCategoryAST().to(DEVICE)
state = load_file(str(MODEL_ROOT / "model.safetensors"), device=str(DEVICE))
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    raise RuntimeError(f"Model state uyusmazligi. Eksik={missing}, fazla={unexpected}")
model.eval()


@torch.no_grad()
def evaluate(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    truths, probabilities = [], []
    for index, row in frame.iterrows():
        audio = load_audio(str(row["path"]))
        values = torch.stack([extract_values(fixed_crop(audio, crop)) for crop in range(5)]).to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(values)
        probabilities.append(torch.softmax(logits.float(), dim=1).mean(0).cpu().numpy())
        truths.append(int(row["target"]))
        if (index + 1) % 100 == 0:
            print(f"  Degerlendirme {index + 1}/{len(frame)}")
    return np.asarray(truths), np.stack(probabilities)


validation_truth, validation_probability = evaluate(validation_frame)
test_truth, test_probability = evaluate(test_frame)


def metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(1)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


# Validation-only post-hoc calibrator. A mild L2 penalty prevents extreme bias
# for WIND, whose validation support is only 13.
validation_log_probability = torch.tensor(
    np.log(np.clip(validation_probability, 1e-8, 1.0)), dtype=torch.float32, device=DEVICE
)
validation_target = torch.tensor(validation_truth, dtype=torch.long, device=DEVICE)
log_temperature = nn.Parameter(torch.zeros((), device=DEVICE))
class_bias = nn.Parameter(torch.zeros(len(LABELS), device=DEVICE))
calibration_optimizer = torch.optim.LBFGS(
    [log_temperature, class_bias], lr=0.25, max_iter=200, line_search_fn="strong_wolfe"
)


def calibration_closure():
    calibration_optimizer.zero_grad()
    temperature = log_temperature.exp().clamp(0.5, 3.0)
    centered_bias = class_bias - class_bias.mean()
    logits = validation_log_probability / temperature + centered_bias
    loss = F.cross_entropy(logits, validation_target) + 0.03 * centered_bias.square().mean()
    loss.backward()
    return loss


calibration_optimizer.step(calibration_closure)
temperature = float(log_temperature.exp().clamp(0.5, 3.0).detach().cpu())
bias = (class_bias - class_bias.mean()).detach().cpu().numpy()


def calibrated_probability(probability: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-8, 1.0)) / temperature + bias
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    return output / output.sum(axis=1, keepdims=True)


validation_raw = metrics(validation_truth, validation_probability)
validation_calibrated_probability = calibrated_probability(validation_probability)
validation_calibrated = metrics(validation_truth, validation_calibrated_probability)

# The test set is not used to select the decision rule.
use_calibration = validation_calibrated["macro_f1"] > validation_raw["macro_f1"] + 1e-6
selected_test_probability = calibrated_probability(test_probability) if use_calibration else test_probability
test_raw = metrics(test_truth, test_probability)
test_selected = metrics(test_truth, selected_test_probability)
test_prediction = selected_test_probability.argmax(1)

report_dict = classification_report(
    test_truth,
    test_prediction,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    zero_division=0,
    output_dict=True,
)
class_metrics = pd.DataFrame(report_dict).T.loc[LABELS].reset_index(names="class")
class_metrics.to_csv(OUTPUT / "class_metrics.csv", index=False)

matrix = confusion_matrix(test_truth, test_prediction, labels=list(range(len(LABELS))) )
row_sum = matrix.sum(axis=1, keepdims=True)
normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum != 0)
pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(OUTPUT / "confusion_counts.csv")
pd.DataFrame(normalized, index=LABELS, columns=LABELS).to_csv(OUTPUT / "confusion_row_normalized.csv")

flows = []
for true_index, true_label in enumerate(LABELS):
    for predicted_index, predicted_label in enumerate(LABELS):
        if true_index != predicted_index and matrix[true_index, predicted_index] > 0:
            flows.append({
                "true_class": true_label,
                "predicted_class": predicted_label,
                "count": int(matrix[true_index, predicted_index]),
                "percent_of_true_class": float(100 * normalized[true_index, predicted_index]),
            })
pd.DataFrame(flows).sort_values("count", ascending=False).to_csv(OUTPUT / "misclassification_flows.csv", index=False)

predictions = test_frame[["path", "label", "source_recording_id", "dataset", "subtype"]].copy()
predictions["prediction"] = [LABELS[index] for index in test_prediction]
predictions["confidence"] = selected_test_probability.max(1)
predictions["correct"] = predictions["label"] == predictions["prediction"]
predictions.to_csv(OUTPUT / "test_predictions_selected.csv", index=False)


def draw_confusion(values: np.ndarray, title: str, filename: str, fmt: str):
    figure, axis = plt.subplots(figsize=(8.5, 7.2))
    image = axis.imshow(values, cmap="Blues", vmin=0)
    axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel("Tahmin")
    axis.set_ylabel("Gercek")
    axis.set_title(title)
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axis.text(column, row, format(values[row, column], fmt), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


draw_confusion(matrix, "Hibrit model - bagimsiz test hata matrisi", "confusion_matrix_counts.png", "d")
draw_confusion(normalized * 100, "Hibrit model - satir normalize hata matrisi (%)", "confusion_matrix_percent.png", ".1f")

summary = {
    "selection_uses_test_set": False,
    "calibration_selected_by_validation": bool(use_calibration),
    "temperature": temperature,
    "class_bias": {label: float(value) for label, value in zip(LABELS, bias)},
    "validation_raw": validation_raw,
    "validation_calibrated": validation_calibrated,
    "independent_test_raw": test_raw,
    "independent_test_selected": test_selected,
}
(OUTPUT / "analysis_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
shutil.make_archive(str(OUTPUT), "zip", OUTPUT)

print("\nANALIZ RAPORU")
print(json.dumps(summary, indent=2))
print("\nSINIF SONUCLARI")
print(class_metrics.to_string(index=False))
print("\nEN BUYUK KARISIKLIKLAR")
flow_frame = pd.DataFrame(flows).sort_values("count", ascending=False)
print(flow_frame.head(12).to_string(index=False))
print(f"\nZIP: {OUTPUT}.zip")
