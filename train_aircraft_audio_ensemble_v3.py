"""Train and evaluate BEATs, AST, PANNs, CLAP and fusion subtype heads."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from aircraft_audio_ensemble_v3 import AdvancedAudioBackbones
from train_aircraft_audio_ensemble_v2 import evaluate, load_rows


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "cache" / "aircraft_agent_manifest_v1.csv"
V2_CACHE = ROOT / "models" / "aircraft_audio_foundation_embeddings_v2.npz"
EXTRA_CACHE = ROOT / "models" / "aircraft_audio_ast_panns_embeddings_v3.npz"
MODEL = ROOT / "models" / "aircraft_audio_ensemble_v3.joblib"
REPORT = ROOT / "outputs" / "aircraft_audio_ensemble_v3_report.json"


def extra_embeddings(rows: list[dict], rebuild: bool = False):
    paths = np.asarray([str(Path(row["path"]).resolve()) for row in rows])
    if EXTRA_CACHE.is_file() and not rebuild:
        cached = np.load(EXTRA_CACHE, allow_pickle=False)
        if np.array_equal(cached["paths"], paths):
            print(f"AST/PANNs embedding cache kullanılıyor: {EXTRA_CACHE}")
            return cached["ast"], cached["panns"], cached["row_indices"]

    extractor = AdvancedAudioBackbones()
    ast_parts, panns_parts = [], []
    for index, row in enumerate(rows):
        ast_parts.append(extractor.ast_file(Path(row["path"])))
        print(f"AST embedding: {index + 1}/{len(rows)}", flush=True)
    extractor.ast = None
    extractor.ast_processor = None
    gc.collect()

    for index, row in enumerate(rows):
        panns_parts.append(extractor.panns_file(Path(row["path"])))
        print(f"PANNs embedding: {index + 1}/{len(rows)}", flush=True)
    extractor.panns = None
    gc.collect()

    indices = []
    for index, (ast, panns) in enumerate(zip(ast_parts, panns_parts)):
        count = min(len(ast), len(panns))
        ast_parts[index] = ast[:count]
        panns_parts[index] = panns[:count]
        indices.extend([index] * count)
    ast_all, panns_all = np.vstack(ast_parts), np.vstack(panns_parts)
    row_indices = np.asarray(indices)
    np.savez_compressed(
        EXTRA_CACHE, paths=paths, ast=ast_all, panns=panns_all, row_indices=row_indices,
    )
    return ast_all, panns_all, row_indices


def model_set() -> dict:
    svc = lambda: make_pipeline(
        StandardScaler(),
        SVC(C=4, probability=True, class_weight="balanced", random_state=42),
    )
    return {
        "BEATs-SVM": svc(),
        "AST-SVM": svc(),
        "PANNs-CNN14-SVM": svc(),
        "CLAP-SVM": svc(),
        "Multi-Embedding Fusion": make_pipeline(
            StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
            LogisticRegression(C=1, max_iter=5000, class_weight="balanced", random_state=42),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    rows = load_rows(MANIFEST)

    v2 = np.load(V2_CACHE, allow_pickle=False)
    expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in rows])
    if not np.array_equal(v2["paths"], expected_paths):
        raise RuntimeError("V2 embedding cache manifest ile eşleşmiyor; önce v2 eğitimi çalıştırılmalı")
    beats, clap, base_indices = v2["beats"], v2["clap"], v2["row_indices"]
    ast, panns, extra_indices = extra_embeddings(rows, args.rebuild_cache)
    if not np.array_equal(base_indices, extra_indices):
        raise RuntimeError("Embedding pencere indeksleri eşleşmiyor")

    features = {
        "BEATs-SVM": beats,
        "AST-SVM": ast,
        "PANNs-CNN14-SVM": panns,
        "CLAP-SVM": clap,
    }
    features["Multi-Embedding Fusion"] = np.concatenate(list(features.values()), axis=1)
    labels = np.asarray([rows[int(index)]["label"] for index in base_indices])
    train_mask = np.asarray([rows[int(index)]["split"] == "train" for index in base_indices])
    models = model_set()
    for name, model in models.items():
        print(f"Eğitim: {name}", flush=True)
        model.fit(features[name][train_mask], labels[train_mask])

    reports = {
        split: evaluate(models, rows, base_indices, features, split)
        for split in ("validation", "test")
    }
    joblib.dump({
        "models": models,
        "classes": sorted(set(labels[train_mask])),
        "metadata": {
            "manifest": str(MANIFEST.resolve()),
            "backbones": ["Microsoft BEATs", "MIT AST", "PANNs CNN14", "LAION CLAP"],
            "airframe_disjoint": True,
            "decision": "human_review_evidence",
        },
    }, MODEL)
    REPORT.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: value["ensemble"] for split, value in reports.items()}, indent=2))
    print(f"Model: {MODEL}\nRapor: {REPORT}")


if __name__ == "__main__":
    main()
