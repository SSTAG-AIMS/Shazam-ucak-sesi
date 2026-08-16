"""Download and verify the official AeroSonicDB v1.1.2 audio archive."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests


URL = "https://zenodo.org/records/10215080/files/audio.zip?download=1"
EXPECTED_MD5 = "77605a8ef12a38a289b63ae3457d326e"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "Self_Data" / "AeroSonicDB_source" / "audio.zip"


def file_md5(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def download(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(URL, headers=headers, stream=True, timeout=(30, 120)) as response:
        if existing and response.status_code != 206:
            existing = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()
        total = existing + int(response.headers.get("content-length", 0))
        mode = "ab" if existing else "wb"
        downloaded = existing
        last_report = 0.0
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    percent = downloaded / total * 100 if total else 0
                    print(
                        f"İndiriliyor: {downloaded / 1024**3:.2f} / "
                        f"{total / 1024**3:.2f} GB (%{percent:.1f})",
                        flush=True,
                    )
                    last_report = now
    partial.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.output.exists():
        download(args.output)
    print("MD5 doğrulanıyor...", flush=True)
    actual = file_md5(args.output)
    if actual != EXPECTED_MD5:
        raise SystemExit(f"MD5 uyuşmadı: beklenen={EXPECTED_MD5}, bulunan={actual}")
    print(f"Tamam: {args.output} ({actual})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"HATA: {exc}", file=sys.stderr, flush=True)
        raise
