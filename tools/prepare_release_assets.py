"""Prepare large project artefacts as GitHub Release ZIP assets.

Normal Git is intentionally kept source-only. This tool groups models, curated
datasets, independent tests and Shazam state into ZIP files that remain below
GitHub's per-release-asset limit. A SHA-256 manifest is produced beside them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PART_MIB = 1792
DEFAULT_ROOTS = ("models", "Self_Data", "Test_Data", "Test_Folder", "cache")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "assets"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def build_groups(workspace: Path, include_downloads: bool) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []

    for root_name in DEFAULT_ROOTS:
        root = workspace / root_name
        if not root.exists():
            continue

        if root_name in {"Self_Data", "Test_Folder"}:
            root_files = sorted(item for item in root.iterdir() if item.is_file())
            if root_files:
                groups.append((f"{root_name}_metadata", root_files))
            for child in sorted(item for item in root.iterdir() if item.is_dir()):
                groups.append((f"{root_name}_{child.name}", files_under(child)))
        else:
            groups.append((root_name, files_under(root)))

    if include_downloads:
        downloads = workspace / "downloads"
        if downloads.exists():
            root_files = sorted(item for item in downloads.iterdir() if item.is_file())
            if root_files:
                groups.append(("downloads_metadata", root_files))
            for child in sorted(item for item in downloads.iterdir() if item.is_dir()):
                groups.append((f"downloads_{child.name}", files_under(child)))

    return [(name, files) for name, files in groups if files]


def partition(files: list[Path], max_bytes: int) -> list[list[Path]]:
    parts: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0

    for path in files:
        size = path.stat().st_size
        if size > max_bytes:
            raise RuntimeError(
                f"Tek dosya parca sinirini asiyor: {path} "
                f"({size / 1024**3:.2f} GiB)"
            )
        if current and current_size + size > max_bytes:
            parts.append(current)
            current = []
            current_size = 0
        current.append(path)
        current_size += size

    if current:
        parts.append(current)
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release_assets"),
        help="ZIP ve manifest cikti klasoru (varsayilan: release_assets)",
    )
    parser.add_argument(
        "--part-mib",
        type=int,
        default=DEFAULT_PART_MIB,
        help="Bir ZIP parcasi icin en fazla kaynak boyutu MiB",
    )
    parser.add_argument(
        "--include-downloads",
        action="store_true",
        help="Yeniden indirilebilir ham downloads/ aynalarini da paketle",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    output = args.output
    if not output.is_absolute():
        output = workspace / output
    output.mkdir(parents=True, exist_ok=True)

    existing = list(output.glob("*.zip")) + list(output.glob("release_assets_manifest.json"))
    if existing:
        print("Cikti klasorunde eski paketler var. Once baska yere tasiyin:", file=sys.stderr)
        for item in existing:
            print(f"- {item}", file=sys.stderr)
        return 2

    max_bytes = args.part_mib * 1024 * 1024
    if max_bytes >= 2 * 1024**3:
        raise ValueError("GitHub Release icin --part-mib 2048 degerinden kucuk olmali.")

    groups = build_groups(workspace, args.include_downloads)
    selected_bytes = sum(path.stat().st_size for _, files in groups for path in files)
    free_bytes = shutil.disk_usage(output).free
    if free_bytes < selected_bytes + 1024**3:
        raise RuntimeError(
            f"Yetersiz bos alan: en az {(selected_bytes + 1024**3) / 1024**3:.2f} GiB gerekli."
        )

    manifest: dict[str, object] = {
        "format": "airport-noise-detection-github-release-assets-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_roots": list(DEFAULT_ROOTS),
        "raw_downloads_included": bool(args.include_downloads),
        "part_source_limit_mib": args.part_mib,
        "archives": [],
    }

    archive_records: list[dict[str, object]] = []
    for group_name, files in groups:
        parts = partition(files, max_bytes)
        for part_index, part_files in enumerate(parts, start=1):
            suffix = f".part{part_index:02d}" if len(parts) > 1 else ""
            archive_name = f"airport_noise_{safe_name(group_name)}{suffix}.zip"
            archive_path = output / archive_name
            print(f"Olusturuluyor: {archive_name} ({len(part_files)} dosya)")

            # WAV, SQLite and model weights are already dense binary data. Stored
            # ZIP is fast and keeps the 2 GiB ceiling predictable.
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
                for source in part_files:
                    bundle.write(source, source.relative_to(workspace).as_posix())

            if archive_path.stat().st_size >= 2 * 1024**3:
                raise RuntimeError(f"Release dosyasi 2 GiB sinirini asti: {archive_path}")

            archive_records.append(
                {
                    "name": archive_name,
                    "group": group_name,
                    "part": part_index,
                    "parts": len(parts),
                    "file_count": len(part_files),
                    "size_bytes": archive_path.stat().st_size,
                    "sha256": sha256_file(archive_path),
                }
            )

    manifest["archives"] = archive_records
    manifest["archive_count"] = len(archive_records)
    manifest["selected_source_bytes"] = selected_bytes
    manifest_path = output / "release_assets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nTamamlandi: {len(archive_records)} ZIP")
    print(f"Manifest: {manifest_path}")
    print("ZIP dosyalarini ve manifesti ayni GitHub Release'e yukleyin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
