"""Batch runner that turns existing manifests into a human-review queue."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from catalog_review import read_jsonl
from dataset_labeling_agent import LabelingAgent, append_jsonl


ROOT = Path(__file__).resolve().parent
DEFAULT_AIRCRAFT_MANIFEST = ROOT / "cache" / "aircraft_type_manifest.csv"
DEFAULT_QUEUE = ROOT / "cache" / "catalog_review_queue_aircraft_v1.jsonl"


def _resolved_audio_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidate = (ROOT / path).resolve()
    if candidate.is_file():
        return candidate
    return (manifest_path.parent / path).resolve()


def load_aircraft_candidates(
    manifest_path: Path,
    allowed_splits: Iterable[str] = ("train",),
) -> list[dict[str, Any]]:
    allowed = {value.strip().lower() for value in allowed_splits}
    candidates = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = str(row.get("split") or "").strip().lower()
            if split not in allowed:
                continue
            audio_path = _resolved_audio_path(str(row["path"]), manifest_path)
            candidates.append(
                {
                    "audio_path": audio_path,
                    "source_recording_id": audio_path.stem,
                    "source_uri": f"doi:{row['source_doi']}#{audio_path.name}",
                    "license_name": row["license"],
                    "category_hint": "AIRCRAFT",
                    "subtype_hint": row["label"],
                    "dataset_split": split,
                    "physical_airframe_id": row.get("hex_id") or "",
                    "session_id": row.get("session") or "",
                    "icao_type": row.get("icao_type") or "",
                }
            )
    return candidates


def select_one_existing_per_subtype(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose one readable source for every subtype without reordering labels."""
    selected: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        subtype = str(candidate.get("subtype_hint") or "").strip().upper()
        if subtype and subtype not in selected and Path(candidate["audio_path"]).is_file():
            selected[subtype] = candidate
    return list(selected.values())


def build_review_queue(
    candidates: Iterable[dict[str, Any]],
    output_path: Path,
    *,
    agent: LabelingAgent | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    agent = agent or LabelingAgent()
    existing = read_jsonl(output_path)
    existing_sources = {str(row.get("source_uri")) for row in existing}
    stats = Counter(existing=len(existing))
    errors: list[dict[str, str]] = []

    for candidate in candidates:
        if limit is not None and stats["added"] >= limit:
            break
        if str(candidate["source_uri"]) in existing_sources:
            stats["skipped_existing"] += 1
            continue
        audio_path = Path(candidate["audio_path"])
        if not audio_path.is_file():
            stats["missing"] += 1
            errors.append({"audio_path": str(audio_path), "error": "FILE_NOT_FOUND"})
            continue
        try:
            record = agent.process(
                audio_path,
                source_recording_id=str(candidate["source_recording_id"]),
                source_uri=str(candidate["source_uri"]),
                license_name=str(candidate["license_name"]),
                category_hint=str(candidate["category_hint"]),
                subtype_hint=str(candidate["subtype_hint"]),
            )
            record.update(
                {
                    "dataset_split": candidate.get("dataset_split"),
                    "manifest_category_hint": candidate.get("category_hint"),
                    "manifest_subtype_hint": candidate.get("subtype_hint"),
                    "physical_airframe_id": candidate.get("physical_airframe_id"),
                    "session_id": candidate.get("session_id"),
                    "icao_type": candidate.get("icao_type"),
                    "catalog_role": "FINGERPRINT_CANDIDATE",
                }
            )
            append_jsonl(record, output_path)
            existing_sources.add(str(candidate["source_uri"]))
            stats["added"] += 1
            if record["agent_action"] == "QUARANTINE":
                stats["quarantined"] += 1
        except Exception as exc:
            stats["failed"] += 1
            errors.append({"audio_path": str(audio_path), "error": str(exc)})

    return {"stats": dict(stats), "errors": errors, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Uçak manifestini insan inceleme kuyruğuna aktar")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_AIRCRAFT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--split", action="append", default=None, help="Varsayılan: train")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--with-multi-models",
        action="store_true",
        help="EfficientNet, CNN, SVM, BEATs ve CLAP kanıtlarını insan kuyruğuna ekle",
    )
    parser.add_argument(
        "--one-per-subtype",
        action="store_true",
        help="Pilot için her uçak alt türünden okunabilir tek kayıt seç",
    )
    args = parser.parse_args()

    splits = args.split or ["train"]
    candidates = load_aircraft_candidates(args.manifest, splits)
    if args.one_per_subtype:
        candidates = select_one_existing_per_subtype(candidates)
    if args.with_multi_models:
        from multi_model_evidence_provider import MultiModelEvidenceProvider

        agent = LabelingAgent(evidence_provider=MultiModelEvidenceProvider())
    else:
        agent = LabelingAgent()
    report = build_review_queue(candidates, args.output, agent=agent, limit=args.limit)
    report["candidate_count"] = len(candidates)
    report["allowed_splits"] = splits
    report["selection"] = "ONE_EXISTING_PER_SUBTYPE" if args.one_per_subtype else "SEQUENTIAL"
    report["evidence_mode"] = (
        "MULTI_MODEL_REVIEW_EVIDENCE_V1" if args.with_multi_models else "METADATA_HINT"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
