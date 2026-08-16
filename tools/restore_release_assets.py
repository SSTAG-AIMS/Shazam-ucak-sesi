"""Verify and restore GitHub Release ZIP assets into a fresh clone."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(workspace: Path, member_name: str) -> Path:
    destination = (workspace / member_name).resolve()
    workspace_resolved = workspace.resolve()
    try:
        destination.relative_to(workspace_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Guvenli olmayan ZIP yolu: {member_name}") from exc
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset_dir",
        nargs="?",
        type=Path,
        default=Path("release_assets"),
        help="Indirilen ZIP ve manifestlerin bulundugu klasor",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Var olan yerel artefakt dosyalarinin ustune yaz",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    asset_dir = args.asset_dir
    if not asset_dir.is_absolute():
        asset_dir = workspace / asset_dir

    manifest_path = asset_dir / "release_assets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archives = manifest.get("archives", [])
    if not archives:
        raise RuntimeError("Manifestte arsiv kaydi bulunamadi.")

    for record in archives:
        archive_path = asset_dir / record["name"]
        if not archive_path.is_file():
            raise FileNotFoundError(f"Eksik Release dosyasi: {archive_path}")
        actual = sha256_file(archive_path)
        if actual.lower() != str(record["sha256"]).lower():
            raise RuntimeError(f"SHA-256 uyusmuyor: {archive_path.name}")
        print(f"Dogrulandi: {archive_path.name}")

    for record in archives:
        archive_path = asset_dir / record["name"]
        print(f"Aciliyor: {archive_path.name}")
        with zipfile.ZipFile(archive_path) as bundle:
            for member in bundle.infolist():
                destination = safe_destination(workspace, member.filename)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and not args.overwrite:
                    continue
                with bundle.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

    print("Tum Release artefaktlari dogrulandi ve proje klasorlerine yerlestirildi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
