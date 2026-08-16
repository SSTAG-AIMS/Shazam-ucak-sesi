"""Train a four-class aircraft type head on frozen BEATs embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from BEATs import BEATs, BEATsConfig


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "cache" / "aircraft_type_clips_350.csv"
ENCODER_PATH = Path(r"C:\models\BEATs_iter3_plus_AS2M.pt")
CACHE_PATH = PROJECT_ROOT / "models" / "aircraft_type_beats_embeddings.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "aircraft_type_beats.pt"
REPORT_PATH = PROJECT_ROOT / "outputs" / "aircraft_type_beats_report.json"
CLASSES = [
    "AIRBUS_A320",
    "BOEING_737_800",
    "DASH_8_300",
    "DIAMOND_DA42",
    "EMBRAER_E190",
    "FOKKER_100",
    "PILATUS_PC12",
    "SAAB_340",
]
SR = 22050
DURATION = 5.0
TARGET_LEN = int(SR * DURATION)
EMBED_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(
    path: Path, category: str | None = None
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if "label" not in row and "subtype" in row:
            row["label"] = row["subtype"]
    if category:
        rows = [row for row in rows if row.get("category") == category]
    missing = [row["path"] for row in rows if not Path(row["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ses dosyası bulunamadı; ilk dosya: {missing[0]}")
    unknown = sorted({row["label"] for row in rows} - set(CLASSES))
    if unknown:
        raise ValueError(f"Bilinmeyen etiketler: {unknown}")
    return rows


def load_audio(path: str) -> np.ndarray:
    samples, _ = librosa.load(path, sr=SR, mono=True)
    if len(samples) >= TARGET_LEN:
        start = (len(samples) - TARGET_LEN) // 2
        samples = samples[start:start + TARGET_LEN]
    else:
        samples = np.pad(samples, (0, TARGET_LEN - len(samples)))
    return samples.astype(np.float32)


def load_encoder() -> BEATs:
    checkpoint = torch.load(ENCODER_PATH, map_location=DEVICE, weights_only=False)
    encoder = BEATs(BEATsConfig(checkpoint["cfg"]))
    encoder.load_state_dict(checkpoint["model"])
    encoder.eval().to(DEVICE)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


@torch.no_grad()
def extract_embeddings(rows: list[dict[str, str]], batch_size: int) -> np.ndarray:
    encoder = load_encoder()
    embeddings: list[np.ndarray] = []
    started = time.time()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        waves = np.stack([load_audio(row["path"]) for row in batch_rows])
        tensor = torch.from_numpy(waves).to(DEVICE)
        padding = torch.zeros(tensor.shape, dtype=torch.bool, device=DEVICE)
        features, _ = encoder.extract_features(tensor, padding_mask=padding)
        embeddings.append(features.mean(dim=1).cpu().numpy())
        done = min(start + batch_size, len(rows))
        print(f"Embedding: {done}/{len(rows)}", flush=True)
    result = np.vstack(embeddings).astype(np.float32)
    print(f"Embedding tamamlandı: {(time.time() - started) / 60:.1f} dakika")
    return result


def load_or_build_cache(
    rows: list[dict[str, str]],
    batch_size: int,
    rebuild: bool,
) -> np.ndarray:
    paths = np.array([row["path"] for row in rows])
    if CACHE_PATH.exists() and not rebuild:
        cached = np.load(CACHE_PATH, allow_pickle=False)
        if np.array_equal(cached["paths"], paths):
            print(f"Embedding cache kullanılıyor: {CACHE_PATH}")
            return cached["embeddings"].astype(np.float32)
        print("Embedding cache manifest ile uyuşmuyor; yeniden oluşturulacak")
    embeddings = extract_embeddings(rows, batch_size)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, paths=paths, embeddings=embeddings)
    return embeddings


class AircraftTypeHead(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, n_classes),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


@torch.no_grad()
def predict(model: nn.Module, embeddings: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(embeddings), batch_size):
        tensor = torch.from_numpy(embeddings[start:start + batch_size]).to(DEVICE)
        predictions.append(model(tensor).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def main() -> None:
    global CLASSES, CACHE_PATH, MODEL_PATH, REPORT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--category",
        choices=("TRAFFIC", "OTHER"),
        help="Kategori alt tür modeli eğitir; manifest category/subtype alanlarını kullanır.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        default="",
        help="Mevcut modeli ezmeden ayrı çıktı üretmek için ek ad (örn. v2_seed7).",
    )
    parser.add_argument(
        "--focus-classes",
        default="",
        help="Kayıp ağırlığı artırılacak virgülle ayrılmış sınıflar.",
    )
    parser.add_argument(
        "--focus-weight",
        type=float,
        default=1.0,
        help="Odak sınıflarına uygulanacak ek kayıp ağırlığı.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    if args.category:
        category = args.category.upper()
        with args.manifest.open("r", encoding="utf-8", newline="") as stream:
            category_rows = [
                row for row in csv.DictReader(stream)
                if row.get("category") == category
            ]
        if not category_rows:
            raise ValueError(f"Manifestte {category} kaydı bulunamadı")
        CLASSES = sorted({row["subtype"] for row in category_rows})
        stem = category.lower() + "_subtype_beats"
        CACHE_PATH = PROJECT_ROOT / "models" / f"{stem}_embeddings.npz"
        output_stem = stem + (f"_{args.variant}" if args.variant else "")
        MODEL_PATH = PROJECT_ROOT / "models" / f"{output_stem}.pt"
        REPORT_PATH = PROJECT_ROOT / "outputs" / f"{output_stem}_report.json"

    rows = load_manifest(args.manifest, args.category)
    embeddings = load_or_build_cache(rows, args.batch_size, args.rebuild_cache)
    label_to_id = {label: index for index, label in enumerate(CLASSES)}
    labels = np.array([label_to_id[row["label"]] for row in rows], dtype=np.int64)
    split_indices = {
        split: np.array([i for i, row in enumerate(rows) if row["split"] == split])
        for split in ("train", "validation", "test")
    }

    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    counts = Counter(labels[train_idx].tolist())
    max_count = max(counts.values())
    class_weights = torch.tensor(
        [max_count / counts[index] for index in range(len(CLASSES))],
        dtype=torch.float32,
        device=DEVICE,
    )
    focus_classes = {
        value.strip().upper()
        for value in args.focus_classes.split(",")
        if value.strip()
    }
    for class_name in focus_classes:
        if class_name not in CLASSES:
            raise ValueError(f"Bilinmeyen odak sınıfı: {class_name}")
        class_weights[CLASSES.index(class_name)] *= args.focus_weight
    train_data = TensorDataset(
        torch.from_numpy(embeddings[train_idx]),
        torch.from_numpy(labels[train_idx]),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = AircraftTypeHead(n_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_pred = predict(model, embeddings[val_idx])
        val_f1 = f1_score(labels[val_idx], val_pred, average="macro", zero_division=0)
        scheduler.step(val_f1)
        print(
            f"Epoch {epoch:03d}  loss={np.mean(losses):.4f}  val_f1={val_f1:.4f}",
            flush=True,
        )
        if val_f1 > best_f1 + 1e-5:
            best_f1 = float(val_f1)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "embed_dim": EMBED_DIM,
                    "epoch": epoch,
                    "val_f1": best_f1,
                    "class_weights": class_weights.cpu().tolist(),
                    "architecture": "LayerNorm-768_Linear-256_ReLU_Dropout-Linear-4",
                },
                MODEL_PATH,
            )
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Erken durdurma: {epoch}. epoch")
            break

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_pred = predict(model, embeddings[test_idx])
    test_f1 = f1_score(labels[test_idx], test_pred, average="macro", zero_division=0)
    report = classification_report(
        labels[test_idx],
        test_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels[test_idx], test_pred, labels=list(range(len(CLASSES)))
    ).tolist()
    result = {
        "model": str(MODEL_PATH),
        "device": str(DEVICE),
        "best_epoch": best_epoch,
        "validation_f1_macro": best_f1,
        "test_f1_macro": float(test_f1),
        "classes": CLASSES,
        "split_sizes": {name: int(len(indices)) for name, indices in split_indices.items()},
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
