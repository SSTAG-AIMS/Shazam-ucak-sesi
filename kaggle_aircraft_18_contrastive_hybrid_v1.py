"""Controlled AST experiments: contrastive-only versus masking+contrastive.

The previous aircraft_18_cosine_finetune_v2 result is the masking-only baseline.
This script keeps the exact same physical-airframe-disjoint split and trains two
fresh models from the same AudioSet checkpoint. Test labels never select epochs.
"""
from __future__ import annotations

import copy, json, math, os, random, shutil
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

SEED, SR, SECONDS = 42, 16000, 10
BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
MAX_EPOCHS, PATIENCE = 28, 7
BATCH_SIZE, ACCUMULATION = 4, 2
CONTRASTIVE_WEIGHT, TEMPERATURE = 0.15, 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_MODE = os.environ.get("AIRCRAFT_EXPERIMENT_MODE", "all").strip().lower()
VALID_RUN_MODES = {"contrastive_only", "masking_contrastive_hybrid", "all"}
if RUN_MODE not in VALID_RUN_MODES:
    raise ValueError(
        f"Gecersiz AIRCRAFT_EXPERIMENT_MODE={RUN_MODE!r}. "
        f"Secenekler: {sorted(VALID_RUN_MODES)}"
    )

OUTPUT_STEM = {
    "contrastive_only": "aircraft_18_contrastive_only_v1",
    "masking_contrastive_hybrid": "aircraft_18_masking_contrastive_hybrid_v1",
    "all": "aircraft_18_contrastive_hybrid_v1",
}[RUN_MODE]
OUT = Path("/kaggle/working") / OUTPUT_STEM

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
if DEVICE.type != "cuda": raise RuntimeError("Kaggle GPU acilmadi")


def locate_root():
    found = list(Path("/kaggle/input").rglob("AIRCRAFT_AST_41CLASS_V1/manifest.csv"))
    if not found: raise FileNotFoundError("AIRCRAFT_AST_41CLASS_V1 Add Input ile eklenmedi")
    return found[0].parent


root = locate_root(); OUT.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(root / "manifest.csv")
raw["physical_airframe_id"] = raw.physical_airframe_id.fillna("").astype(str)
raw["path"] = raw.path.map(lambda value: str(root / str(value)))
airframe_counts = raw.groupby("label").physical_airframe_id.nunique()
labels = sorted(airframe_counts[airframe_counts >= 3].index.astype(str))
unsupported = sorted(airframe_counts[airframe_counts < 3].index.astype(str))
label2id = {label: index for index, label in enumerate(labels)}
id2label = {index: label for label, index in label2id.items()}
df = raw[raw.label.isin(labels)].copy(); df["target"] = df.label.map(label2id)

# Identical split rule and seed as masking-only V2.
rng = random.Random(SEED); pieces = []
for label, group in df.groupby("label"):
    ids = sorted(group.physical_airframe_id.unique()); rng.shuffle(ids)
    if len(ids) >= 5:
        nt = max(1, round(len(ids) * .15)); nv = max(1, round(len(ids) * .15))
    else: nt = nv = 1
    test, validation = set(ids[:nt]), set(ids[nt:nt + nv])
    current = group.copy()
    current["split"] = current.physical_airframe_id.map(
        lambda item: "test" if item in test else "validation" if item in validation else "train")
    pieces.append(current)
df = pd.concat(pieces, ignore_index=True); df.to_csv(OUT / "split_manifest.csv", index=False)
print("GPU:", torch.cuda.get_device_name(0)); print("Sinif:", len(labels), labels)
print(pd.crosstab(df.label, df.split).to_string())

extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)


def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True); audio = np.nan_to_num(audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    return audio / max(peak, .10) if peak > 1e-6 else audio


def random_crop(audio):
    size = SR * SECONDS
    if len(audio) <= size: return np.pad(audio, (0, size - len(audio))).astype(np.float32)
    start = random.randint(0, len(audio) - size); return audio[start:start + size].astype(np.float32)


def fixed_crop(audio, index):
    size = SR * SECONDS
    if len(audio) <= size: return np.pad(audio, (0, size - len(audio))).astype(np.float32)
    maximum = len(audio) - size
    starts = [0, maximum // 4, maximum // 2, 3 * maximum // 4, maximum]
    return audio[starts[index]:starts[index] + size].astype(np.float32)


def waveform_augment(audio):
    result = audio.copy() * (10 ** (random.uniform(-4, 4) / 20))
    if random.random() < .55:
        result += np.random.normal(0, random.uniform(.0002, .002), result.shape).astype(np.float32)
    if random.random() < .30:
        result = np.roll(result, random.randint(-800, 800))
    return np.clip(result, -1, 1).astype(np.float32)


class PairDataset(Dataset):
    def __init__(self, frame): self.frame = frame.reset_index(drop=True)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]; audio = load_audio(row.path)
        view1 = waveform_augment(random_crop(audio)); view2 = waveform_augment(random_crop(audio))
        value1 = extractor(view1, sampling_rate=SR, return_tensors="pt")["input_values"][0]
        value2 = extractor(view2, sampling_rate=SR, return_tensors="pt")["input_values"][0]
        return value1, value2, int(row.target)


train_df = df[df.split == "train"].reset_index(drop=True)
val_df = df[df.split == "validation"].reset_index(drop=True)
test_df = df[df.split == "test"].reset_index(drop=True)
train_counts = train_df.target.value_counts().to_dict()
sample_weights = [min(5., math.sqrt(len(train_df)/(len(labels)*train_counts[int(t)]))) for t in train_df.target]
sampler = WeightedRandomSampler(sample_weights, len(train_df), replacement=True,
                                generator=torch.Generator().manual_seed(SEED))
loader = DataLoader(PairDataset(train_df), batch_size=BATCH_SIZE, sampler=sampler,
                    num_workers=2, pin_memory=True, persistent_workers=True)
counts = torch.bincount(torch.tensor(train_df.target.to_numpy()), minlength=len(labels)).float()
class_weights = torch.sqrt(counts.sum()/counts.clamp_min(1)); class_weights = (class_weights/class_weights.mean()).clamp(.4, 4).to(DEVICE)


class CosineHead(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.randn(len(labels), 768)*.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.)))
    def forward(self, features):
        return F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1)) * self.log_scale.exp().clamp(5, 40)


class ContrastiveAST(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone = ASTModel.from_pretrained(BASE_MODEL, attn_implementation="eager")
        self.classifier = CosineHead()
        self.projector = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 128))
    def mask(self, values):
        values = values.clone(); time_bins, freq_bins = values.shape[1], values.shape[2]
        for row in range(len(values)):
            if torch.rand((), device=values.device) < .55:
                width = int(torch.randint(8, 33, (), device=values.device)); start = int(torch.randint(0, max(1,time_bins-width), (), device=values.device))
                values[row, start:start+width, :] = 0
            if torch.rand((), device=values.device) < .45:
                width = int(torch.randint(4, 13, (), device=values.device)); start = int(torch.randint(0, max(1,freq_bins-width), (), device=values.device))
                values[row, :, start:start+width] = 0
        return values
    def forward(self, values, use_masking=False):
        if use_masking and self.training: values = self.mask(values)
        hidden = self.backbone(input_values=values).last_hidden_state
        features = (hidden[:,0] + hidden[:,1]) / 2
        return self.classifier(features), F.normalize(self.projector(features), dim=1)


def supervised_contrastive(z1, z2, targets):
    # Both augmented views of the same recording are guaranteed positives;
    # other recordings with the same label are additional positives.
    # Contrastive logits are intentionally calculated in float32. Under CUDA
    # autocast they would otherwise be float16, where the old -1e9 mask value
    # overflows (the minimum finite float16 value is about -65504).
    features = torch.cat([z1, z2], dim=0).float()
    repeated = torch.cat([targets, targets], dim=0)
    similarity = features @ features.T / TEMPERATURE
    self_mask = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positive = repeated[:,None].eq(repeated[None,:]) & ~self_mask
    similarity = similarity.masked_fill(self_mask, -torch.inf)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_log_probability = log_probability.masked_fill(~positive, 0.0)
    return -positive_log_probability.sum(1).div(positive.sum(1).clamp_min(1)).mean()


@torch.no_grad()
def evaluate(model, frame):
    model.eval(); truths, probabilities = [], []
    for _, row in frame.iterrows():
        audio = load_audio(row.path)
        values = torch.stack([extractor(fixed_crop(audio, i), sampling_rate=SR, return_tensors="pt")["input_values"][0] for i in range(5)]).to(DEVICE)
        with torch.amp.autocast("cuda"): logits, _ = model(values, False)
        probabilities.append(torch.softmax(logits, 1).mean(0).cpu().numpy()); truths.append(int(row.target))
    probabilities = np.stack(probabilities); truths = np.asarray(truths); predictions = probabilities.argmax(1)
    return truths, predictions, probabilities


def airframe_metrics(frame, truth, prediction):
    evidence = frame[["physical_airframe_id"]].copy(); evidence["truth"] = truth; evidence["prediction"] = prediction
    yt, yp = [], []
    for _, group in evidence.groupby("physical_airframe_id"):
        yt.append(Counter(group.truth).most_common(1)[0][0]); yp.append(Counter(group.prediction).most_common(1)[0][0])
    return accuracy_score(yt,yp), f1_score(yt,yp,average="macro",zero_division=0)


def train_experiment(name, use_masking):
    print("\n" + "="*75 + f"\n{name.upper()}\n" + "="*75)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = ContrastiveAST().to(DEVICE); blocks = model.backbone.encoder.layer
    for p in model.backbone.parameters(): p.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 2e-3},
        {"params": model.projector.parameters(), "lr": 4e-4, "weight_decay": 2e-3},
        {"params": blocks[-4:].parameters(), "lr": 1e-5, "weight_decay": 1e-2},
        {"params": model.backbone.layernorm.parameters(), "lr": 2e-5, "weight_decay": 1e-2},
    ])
    total_steps = math.ceil(len(loader)/ACCUMULATION)*MAX_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1,total_steps//12), total_steps)
    scaler = torch.amp.GradScaler("cuda"); best=-1.; best_epoch=0; best_state=None; stale=0; history=[]
    for epoch in range(1, MAX_EPOCHS+1):
        open_blocks = 0 if epoch <= 3 else 2 if epoch <= 8 else 4
        for p in model.backbone.parameters(): p.requires_grad=False
        for block in blocks[-open_blocks:] if open_blocks else []:
            for p in block.parameters(): p.requires_grad=True
        if open_blocks:
            for p in model.backbone.layernorm.parameters(): p.requires_grad=True
        model.train(); optimizer.zero_grad(set_to_none=True); running=0.
        for step,(view1,view2,target) in enumerate(loader,1):
            view1,view2,target=view1.to(DEVICE),view2.to(DEVICE),target.to(DEVICE)
            with torch.amp.autocast("cuda"):
                logits1,z1=model(view1,use_masking); logits2,z2=model(view2,use_masking)
                classification=(F.cross_entropy(logits1,target,weight=class_weights,label_smoothing=.04)+F.cross_entropy(logits2,target,weight=class_weights,label_smoothing=.04))/2
                contrastive=supervised_contrastive(z1,z2,target)
                loss=(classification+CONTRASTIVE_WEIGHT*contrastive)/ACCUMULATION
            scaler.scale(loss).backward()
            if step%ACCUMULATION==0 or step==len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
                old=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=old: scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item())*ACCUMULATION*len(target)
        yt,yp,_=evaluate(model,val_df); recf=f1_score(yt,yp,average="macro",zero_division=0); _,airf=airframe_metrics(val_df,yt,yp)
        history.append({"epoch":epoch,"open_blocks":open_blocks,"loss":running/len(train_df),"val_recording_f1":recf,"val_airframe_f1":airf})
        print(f"{name} epoch={epoch:02d} loss={running/len(train_df):.4f} blocks={open_blocks} valRecF1={recf:.4f} valAirF1={airf:.4f}")
        if airf>best+1e-5: best=float(airf); best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0; print("  -> Yeni en iyi")
        else: stale+=1
        if epoch>=10 and stale>=PATIENCE: print("Erken durdurma"); break
    model.load_state_dict(best_state); pd.DataFrame(history).to_csv(OUT/f"{name}_history.csv",index=False)
    yt,yp,prob=evaluate(model,test_df); air_acc,air_f1=airframe_metrics(test_df,yt,yp)
    cpu_state={k:v.detach().cpu().contiguous() for k,v in best_state.items()}; save_file(cpu_state,str(OUT/f"{name}.safetensors"))
    report={"experiment":name,"masking":use_masking,"contrastive":True,"best_epoch":best_epoch,"best_validation_airframe_macro_f1":best,
            "test_recording_accuracy":accuracy_score(yt,yp),"test_recording_macro_f1":f1_score(yt,yp,average="macro",zero_division=0),
            "test_airframe_accuracy":air_acc,"test_airframe_macro_f1":air_f1}
    (OUT/f"{name}_classification_report.txt").write_text(classification_report(yt,yp,labels=list(range(len(labels))),target_names=labels,zero_division=0),encoding="utf-8")
    del model; torch.cuda.empty_cache(); return report


experiment_plan = {
    "contrastive_only": [("contrastive_only", False)],
    "masking_contrastive_hybrid": [("masking_contrastive_hybrid", True)],
    "all": [
        ("contrastive_only", False),
        ("masking_contrastive_hybrid", True),
    ],
}[RUN_MODE]
results = [train_experiment(name, use_masking) for name, use_masking in experiment_plan]
masking_baseline={"experiment":"masking_only_external_v2","test_recording_accuracy":.6105263157894737,"test_recording_macro_f1":.3648798085383451,
                  "test_airframe_accuracy":.6521739130434783,"test_airframe_macro_f1":.35235690235690237,
                  "note":"Previously completed V2, same seed and split; stored as an external controlled reference."}
final={"run_mode":RUN_MODE,"classes":labels,"unsupported_shazam_or_unknown":unsupported,"masking_only_baseline":masking_baseline,"new_results":results,
       "selection_rule":"Compare independent test Macro-F1 only after each model was selected by validation airframe Macro-F1."}
(OUT/"experiment_comparison.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
shutil.make_archive(str(Path("/kaggle/working") / OUTPUT_STEM), "zip", OUT)
print("\nKARSILASTIRMA\n",json.dumps(final,indent=2))
print(f"ZIP: /kaggle/working/{OUTPUT_STEM}.zip")
