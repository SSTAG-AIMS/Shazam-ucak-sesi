"""Provenance-first intake gate for manually collected aircraft references."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse

from dataset_catalog import CatalogValidationError, normalize_label, sha256_file
from dataset_labeling_agent import observe_audio, quality_issues


ROOT = Path(__file__).resolve().parent
DEFAULT_INBOX = ROOT / "Self_Data" / "AIRCRAFT_REFERENCE_INBOX_V1"
DEFAULT_QUEUE = ROOT / "cache" / "aircraft_reference_intake_queue_v1.jsonl"
ALLOWED_LICENSE_PREFIXES = ("CC0", "CC BY", "PUBLIC DOMAIN")


class ReferenceIntakeError(ValueError):
    pass


def validate_source_uri(value: str) -> str:
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https", "doi"}:
        raise ReferenceIntakeError("Kaynak URI http, https veya doi olmalıdır")
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise ReferenceIntakeError("Kaynak URI geçerli bir alan adı içermelidir")
    if parsed.scheme == "doi" and not parsed.path:
        raise ReferenceIntakeError("DOI kaynağı kimlik içermelidir")
    return value


def validate_license(value: str) -> str:
    value = value.strip().upper()
    if not any(value.startswith(prefix) for prefix in ALLOWED_LICENSE_PREFIXES):
        raise ReferenceIntakeError(
            "Lisans açık ve doğrulanabilir olmalıdır: CC0, CC BY, CC BY-NC veya Public Domain"
        )
    return value


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_reference(
    audio_path: Path,
    *,
    aircraft_type: str,
    icao_type: str,
    physical_airframe_id: str,
    source_uri: str,
    license_name: str,
    inbox: Path = DEFAULT_INBOX,
    queue: Path = DEFAULT_QUEUE,
) -> dict:
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise ReferenceIntakeError(f"Ses dosyası bulunamadı: {audio_path}")
    try:
        label = normalize_label(aircraft_type)
        icao = normalize_label(icao_type)
    except CatalogValidationError as exc:
        raise ReferenceIntakeError(str(exc)) from exc
    airframe = physical_airframe_id.strip().upper()
    if not airframe:
        raise ReferenceIntakeError("Fiziksel uçak kimliği/tescil/hex_id zorunludur")
    source_uri = validate_source_uri(source_uri)
    license_name = validate_license(license_name)
    digest = sha256_file(audio_path)
    if any(item.get("sha256") == digest for item in read_jsonl(queue)):
        raise ReferenceIntakeError("Bu ses dosyası bekleyen kuyruğa daha önce eklenmiş")

    observation = observe_audio(audio_path)
    issues = quality_issues(observation)
    destination_dir = inbox / label
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{digest[:12]}_{audio_path.name}"
    if not destination.exists():
        shutil.copy2(audio_path, destination)
    record = {
        "intake_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_uri}#{digest}")),
        "audio_path": str(destination.resolve()),
        "sha256": digest,
        "category": "AIRCRAFT",
        "proposed_subtype": label,
        "icao_type": icao,
        "physical_airframe_id": airframe,
        "source_uri": source_uri,
        "license": license_name,
        "quality": observation.__dict__,
        "quality_issues": issues,
        "intake_status": "QUARANTINE" if issues else "PENDING_HUMAN_REVIEW",
        "fingerprint_indexed": False,
        "human_approved": False,
    }
    queue.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Yeni uçak referansını güvenli bekleme havuzuna alır")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--type", required=True, dest="aircraft_type")
    parser.add_argument("--icao", required=True)
    parser.add_argument("--airframe-id", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    args = parser.parse_args()
    record = stage_reference(
        args.audio, aircraft_type=args.aircraft_type, icao_type=args.icao,
        physical_airframe_id=args.airframe_id, source_uri=args.source_uri,
        license_name=args.license_name, inbox=args.inbox, queue=args.queue,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
