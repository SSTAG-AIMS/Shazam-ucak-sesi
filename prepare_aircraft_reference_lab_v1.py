"""Prepare a clean agent-vs-golden-reference aircraft lab."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Self_Data" / "AIRCRAFT_100_REFERENCE_V1"
LAB = ROOT / "Test_Folder" / "AIRCRAFT_REFERENCE_LAB_V1"
SOURCE_URI = "https://zenodo.org/records/10215080"
LICENSE = "CC BY-NC 4.0"
PILOT_MODEL = ROOT / "models" / "aircraft_agent_pilot_v1.joblib"
KNOWN_ICAO = {
    "AIRBUS_A320": "A320", "BOEING_737_800": "B738", "DASH_8_300": "DH8C",
    "DIAMOND_DA42": "DA42", "EMBRAER_E190": "E190", "FOKKER_100": "F100",
    "PILATUS_PC12": "PC12", "SAAB_340": "SF34",
}


def _icao(label: str) -> str:
    if label in KNOWN_ICAO: return KNOWN_ICAO[label]
    parts = label.split("_")
    return parts[1] if len(parts) > 2 and parts[0] == "ICAO" else "UNKNOWN"


def _pilot_classes() -> list[str]:
    if not PILOT_MODEL.is_file(): return sorted(KNOWN_ICAO)
    return [str(item) for item in joblib.load(PILOT_MODEL)["classes"]]


# Public, deterministic class list used by tests and lab tooling.
TYPES = tuple(_pilot_classes())


def prepare_lab(lab: Path = LAB, source: Path = SOURCE) -> dict:
    tests = lab / "TEST_SESLERI"; gold = lab / "ALTIN_REFERANSLAR"; workspace = lab / "workspace"
    # TEST_SESLERI and ALTIN_REFERANSLAR are generated views.  Rebuilding on
    # top of an older catalog used to leave stale files behind; a former test
    # query could therefore also remain in the gold bank.  Never touch the
    # human-decision workspace, but recreate these two generated views.
    for generated in (tests, gold):
        if generated.is_dir():
            shutil.rmtree(generated)
    for directory in (tests, gold, workspace): directory.mkdir(parents=True, exist_ok=True)
    for folder in ("BEKLEYEN", "KABUL_EDILEN", "REDDEDILEN", "EMIN_OLUNAMAYANLAR"):
        (workspace / folder).mkdir(parents=True, exist_ok=True)
    records, gold_records = [], []
    catalog_types = sorted(path.name for path in source.iterdir() if path.is_dir())
    for aircraft_type in catalog_types:
        candidates = sorted((source / aircraft_type).glob("*.wav"))
        if not candidates: continue
        gold_dir = gold / aircraft_type; gold_dir.mkdir(parents=True, exist_ok=True)
        # Keep at least one completely separate query recording. Rich source
        # folders get three references; tiny test fixtures still get one.
        reference_count = min(3, max(0, len(candidates) - 1))
        for reference_index, reference_source in enumerate(candidates[:reference_count], 1):
            gold_target = gold_dir / reference_source.name
            if not gold_target.exists(): shutil.copy2(reference_source, gold_target)
            gold_records.append({
                "aircraft_type": aircraft_type, "icao_type": _icao(aircraft_type),
                "audio_path": str(gold_target.resolve()), "verified": True,
                "reference_index": reference_index,
                "physical_airframe_id": reference_source.stem.split("_")[0].upper(),
                "source_uri": SOURCE_URI, "license": LICENSE,
            })
        # Only types already supported by the generalising agent get blind
        # test queries. All catalogued types still enter the golden bank.
        if aircraft_type not in TYPES or len(candidates) < 2: continue
        test_dir = tests / aircraft_type; test_dir.mkdir(parents=True, exist_ok=True)
        test_source = candidates[reference_count]
        test_target = test_dir / test_source.name
        if not test_target.exists(): shutil.copy2(test_source, test_target)
        records.append({
            "audio_path": str(test_target.resolve()), "aircraft_type": aircraft_type,
            "icao_type": _icao(aircraft_type), "physical_airframe_id": test_source.stem.split("_")[0].upper(),
            "source_uri": SOURCE_URI, "license": LICENSE, "expected_category": "AIRCRAFT",
        })
    manifest = {
        "name": "AIRCRAFT_REFERENCE_LAB_V2", "workflow": "AGENT -> ALTIN_REFERANS -> INSAN -> SHAZAM",
        "production_isolation": True, "sample_count": len(records), "gold_reference_count": len(gold_records),
        "catalogued_subtype_count": len(catalog_types), "agent_subtype_count": len(TYPES),
        "workspace": str(workspace.resolve()), "records": records, "gold_references": gold_records,
    }
    (lab / "test_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (lab / "NASIL_TEST_EDILIR.txt").write_text(
        "python run_aircraft_reference_lab_v1.py\n"
        "TEST_SESLERI: Agenta verilecek bilinmeyen kayıtlar.\n"
        "ALTIN_REFERANSLAR: Her tip için en fazla 3 kaynak-doğrulanmış A/B referansı.\n"
        "workspace/BEKLEYEN: Karar bekleyenler.\nworkspace/KABUL_EDILEN: İnsan onaylılar.\n"
        "workspace/REDDEDILEN: İnsan tarafından reddedilenler.\n"
        "workspace/EMIN_OLUNAMAYANLAR: Dinleyerek kesin karar verilemeyenler.\n"
        "Katalog 41 ADS-B doğrulamalı tipe kadar genişleyebilir; agent yalnızca yeterli farklı fiziksel uçağı olan tipleri tahmin eder.\n"
        "Shazam agent tahmininde veya referans seçiminde kullanılmaz; yalnızca kabulden sonra indekslenir.\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    result = prepare_lab(); print(json.dumps({"lab": str(LAB), "test": result["sample_count"], "gold": result["gold_reference_count"]}, ensure_ascii=False, indent=2))
