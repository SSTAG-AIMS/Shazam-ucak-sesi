"""Kaggle-ready AST fine-tuning for the 41 ICAO aircraft types.

Attach AIRCRAFT_AST_41CLASS_V1.zip (or its extracted Kaggle dataset), enable a
GPU and run this file in a notebook with: %run /path/to/this/file.py
"""

from __future__ import annotations

import json, math, random, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoFeatureExtractor, ASTForAudioClassification

SEED = 42
SR = 16000
SECONDS = 10
MAX_SAMPLES = SR * SECONDS
EPOCHS = 25
PATIENCE = 6
BATCH_SIZE = 6
ACCUMULATION = 4
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
WORK = Path("/kaggle/working/aircraft_ast_41_v1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
if DEVICE.type != "cuda": raise RuntimeError("Kaggle GPU acilmadi")


def locate_dataset() -> Path:
    candidates = list(Path("/kaggle/input").rglob("AIRCRAFT_AST_41CLASS_V1/manifest.csv"))
    if candidates:
        return candidates[0].parent
    archives = list(Path("/kaggle/input").rglob("AIRCRAFT_AST_41CLASS_V1.zip"))
    if not archives:
        raise FileNotFoundError("AIRCRAFT_AST_41CLASS_V1 verisi Add Input ile eklenmedi")
    target = Path("/kaggle/working/AIRCRAFT_AST_41CLASS_V1")
    if target.exists(): shutil.rmtree(target)
    with zipfile.ZipFile(archives[0]) as z: z.extractall(target.parent)
    return target


ROOT = locate_dataset()
df = pd.read_csv(ROOT / "manifest.csv")
df["path"] = df["path"].map(lambda p: str(ROOT / str(p)))
df["label"] = df["label"].astype(str)
df["physical_airframe_id"] = df["physical_airframe_id"].fillna("").astype(str)
labels = sorted(df.label.unique())
label2id = {x: i for i, x in enumerate(labels)}
id2label = {i: x for x, i in label2id.items()}
assert len(labels) == 41, f"41 yerine {len(labels)} sinif bulundu"

# Physical-airframe-disjoint split.  Classes with <3 aircraft remain train-only;
# claiming an independent score for those classes would be scientifically false.
rng = random.Random(SEED)
parts, coverage = [], []
for label, group in df.groupby("label"):
    ids = sorted(group.physical_airframe_id.unique()); rng.shuffle(ids)
    if len(ids) >= 5:
        n_test = max(1, round(len(ids) * .15)); n_val = max(1, round(len(ids) * .15))
    elif len(ids) >= 3:
        n_test = n_val = 1
    else:
        n_test = n_val = 0
    test_ids = set(ids[:n_test]); val_ids = set(ids[n_test:n_test+n_val])
    current = group.copy()
    current["split"] = current.physical_airframe_id.map(
        lambda x: "test" if x in test_ids else "validation" if x in val_ids else "train"
    )
    parts.append(current)
    coverage.append({"label": label, "recordings": len(group), "airframes": len(ids),
                     "independent_test_ready": len(ids) >= 3})
df = pd.concat(parts, ignore_index=True)
df["target"] = df.label.map(label2id)

for label, group in df.groupby("label"):
    sets = {s: set(g.physical_airframe_id) for s, g in group.groupby("split")}
    assert sets.get("train", set()).isdisjoint(sets.get("validation", set()))
    assert sets.get("train", set()).isdisjoint(sets.get("test", set()))

WORK.mkdir(parents=True, exist_ok=True)
df.to_csv(WORK / "split_manifest.csv", index=False)
(WORK / "coverage_report.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
print("Kayit:", len(df), "Sinif:", len(labels), "GPU:", torch.cuda.get_device_name(0))
print(pd.crosstab(df.label, df.split).to_string())

extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)


class AudioData(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool):
        self.frame = frame.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        y, _ = librosa.load(row.path, sr=SR, mono=True)
        if len(y) > MAX_SAMPLES:
            start = random.randint(0, len(y)-MAX_SAMPLES) if self.train else (len(y)-MAX_SAMPLES)//2
            y = y[start:start+MAX_SAMPLES]
        else: y = np.pad(y, (0, MAX_SAMPLES-len(y)))
        if self.train:
            y = y * random.uniform(.65, 1.35)
            y = np.roll(y, random.randint(-SR, SR))
            if random.random() < .35:
                y = y + np.random.normal(0, random.uniform(.0002, .002), len(y)).astype(np.float32)
        return y.astype(np.float32), int(row.target), str(row.physical_airframe_id), str(row.path)


def collate(batch):
    audio, target, airframe, path = zip(*batch)
    values = extractor(list(audio), sampling_rate=SR, return_tensors="pt")["input_values"]
    return values, torch.tensor(target), list(airframe), list(path)


train_df = df[df.split == "train"]
val_df = df[df.split == "validation"]
test_df = df[df.split == "test"]
train_counts = Counter(train_df.target)
# Inverse-square-root sampling limits domination by B738 without making a
# one-record class appear as if it had real diversity.
sample_weights = [min(5.0, math.sqrt(len(train_df) / (len(labels)*train_counts[int(t)])))
                  for t in train_df.target]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)
train_loader = DataLoader(AudioData(train_df, True), batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate)
val_loader = DataLoader(AudioData(val_df, False), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate)
test_loader = DataLoader(AudioData(test_df, False), batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True, collate_fn=collate)

model = ASTForAudioClassification.from_pretrained(
    BASE_MODEL, num_labels=len(labels), label2id=label2id, id2label=id2label,
    ignore_mismatched_sizes=True, attn_implementation="eager",
).to(DEVICE)
for p in model.parameters(): p.requires_grad = False
for p in model.classifier.parameters(): p.requires_grad = True

# Effective-number class weights + focal modulation combat long-tail imbalance.
beta = .999
raw = np.array([(1-beta)/(1-beta**train_counts.get(i, 1)) for i in range(len(labels))])
raw = raw / raw.mean(); class_weights = torch.tensor(np.clip(raw, .25, 4.0), dtype=torch.float32, device=DEVICE)


def set_stage(epoch: int):
    # Head warm-up, then gradually open the final 2 and final 4 Transformer blocks.
    blocks = model.audio_spectrogram_transformer.encoder.layer
    open_blocks = 0 if epoch < 2 else 2 if epoch < 5 else 4
    for p in model.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True
    for block in blocks[-open_blocks:] if open_blocks else []:
        for p in block.parameters(): p.requires_grad = True
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            for p in module.parameters(): p.requires_grad = True
    return open_blocks


def focal_loss(logits, target, gamma=1.5):
    ce = F.cross_entropy(logits, target, weight=class_weights, reduction="none", label_smoothing=.03)
    pt = torch.softmax(logits, 1).gather(1, target[:, None]).squeeze(1)
    return (((1-pt)**gamma) * ce).mean()


@torch.no_grad()
def evaluate(loader):
    model.eval(); ys, ps, frames, paths, confidences = [], [], [], [], []
    for x, y, airframe, path in loader:
        prob = torch.softmax(model(input_values=x.to(DEVICE)).logits, 1).cpu()
        pred = prob.argmax(1)
        ys += y.tolist(); ps += pred.tolist(); frames += airframe; paths += path
        confidences += prob.max(1).values.tolist()
    rec_f1 = f1_score(ys, ps, average="macro", zero_division=0) if ys else 0.0
    table = pd.DataFrame({"truth": ys, "prediction": ps, "airframe": frames,
                          "path": paths, "confidence": confidences})
    grouped_true, grouped_pred = [], []
    for _, g in table.groupby("airframe"):
        grouped_true.append(Counter(g.truth).most_common(1)[0][0])
        grouped_pred.append(Counter(g.prediction).most_common(1)[0][0])
    air_f1 = f1_score(grouped_true, grouped_pred, average="macro", zero_division=0) if grouped_true else 0.0
    return rec_f1, air_f1, table, grouped_true, grouped_pred


optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.02)
scaler = torch.amp.GradScaler("cuda")
best, stale, history = -1.0, 0, []
for epoch in range(EPOCHS):
    opened = set_stage(epoch); model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
    for step, (x, y, _, _) in enumerate(train_loader):
        with torch.amp.autocast("cuda"):
            logits = model(input_values=x.to(DEVICE)).logits
            loss = focal_loss(logits, y.to(DEVICE)) / ACCUMULATION
        scaler.scale(loss).backward(); losses.append(float(loss)*ACCUMULATION)
        if (step+1) % ACCUMULATION == 0 or step+1 == len(train_loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    rec_f1, air_f1, _, _, _ = evaluate(val_loader)
    history.append({"epoch": epoch+1, "loss": np.mean(losses), "opened_blocks": opened,
                    "validation_recording_macro_f1": rec_f1, "validation_airframe_macro_f1": air_f1})
    print(f"Epoch {epoch+1:02}/{EPOCHS} loss={np.mean(losses):.4f} blocks={opened} valRecF1={rec_f1:.4f} valAirF1={air_f1:.4f}")
    if air_f1 > best + 1e-4:
        best, stale = air_f1, 0
        model.save_pretrained(WORK / "best_model"); extractor.save_pretrained(WORK / "best_model")
    else:
        stale += 1
        if stale >= PATIENCE: print("Early stopping"); break

pd.DataFrame(history).to_csv(WORK / "training_history.csv", index=False)
model = ASTForAudioClassification.from_pretrained(WORK / "best_model", attn_implementation="eager").to(DEVICE)
rec_f1, air_f1, pred, air_y, air_p = evaluate(test_loader)
pred["truth_label"] = pred.truth.map(id2label); pred["prediction_label"] = pred.prediction.map(id2label)
pred.to_csv(WORK / "test_predictions.csv", index=False)
report = {
    "classes_in_head": len(labels), "best_validation_airframe_macro_f1": best,
    "test_recording_accuracy": accuracy_score(pred.truth, pred.prediction) if len(pred) else None,
    "test_recording_macro_f1": rec_f1,
    "test_airframe_accuracy": accuracy_score(air_y, air_p) if air_y else None,
    "test_airframe_macro_f1": air_f1,
    "important_scope_warning": "Scores cover only classes with >=3 independent physical airframes; coverage_report.json lists unsupported rare classes.",
}
(WORK / "independent_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(WORK / "classification_report.txt").write_text(
    classification_report(pred.truth, pred.prediction, labels=list(range(len(labels))),
                          target_names=labels, zero_division=0), encoding="utf-8")
shutil.make_archive("/kaggle/working/aircraft_ast_41class_v1", "zip", WORK)
print(json.dumps(report, indent=2))
print("ZIP: /kaggle/working/aircraft_ast_41class_v1.zip")
