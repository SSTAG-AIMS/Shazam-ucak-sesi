"""Discover and stage openly licensed aircraft media from Wikimedia Commons.

Candidates are kept separate from the active Shazam index.  Wikimedia category
membership verifies the documented aircraft type, but not that the aircraft is
audible throughout the file; therefore human/audio-agent review is still needed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Self_Data" / "WIKIMEDIA_AIRCRAFT_CANDIDATES_V1"
DISCOVERY = OUT / "discovery_manifest.jsonl"
STAGED = OUT / "staged_audio_manifest.jsonl"
CHECKPOINT = OUT / "scan_checkpoint.json"
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AirportNoiseDetectionResearch/1.0 (educational acoustic research)"
FFMPEG = Path(
    "C:\\Users\\muham\\AppData\\Local\\Microsoft\\WinGet\\Packages\\"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\"
    r"ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
)

# Local label -> Commons model name, title tokens.  Tokens reject unrelated
# files occasionally placed in broad Wikimedia categories.
TYPE_MAP: dict[str, tuple[str, tuple[str, ...]]] = {
    "AIRBUS_A320": ("Airbus A320", ("a320",)),
    "BOEING_737_800": ("Boeing 737-800", ("737-800", "b738")),
    "DASH_8_300": ("De Havilland Canada Dash 8-300", ("dash 8", "dhc-8", "dh8")),
    "DIAMOND_DA42": ("Diamond DA42", ("da42",)),
    "EMBRAER_E190": ("Embraer E190", ("e190", "erj-190", "embraer 190")),
    "FOKKER_100": ("Fokker 100", ("fokker 100", "f100")),
    "PILATUS_PC12": ("Pilatus PC-12", ("pc-12", "pc12")),
    "SAAB_340": ("Saab 340", ("saab 340", "sf34")),
    "ICAO_A21N_A321_271NX": ("Airbus A321neo", ("a321neo", "a21n")),
    "ICAO_A333_A330_323X": ("Airbus A330-300", ("a330-300", "a333")),
    "ICAO_A359_A350_941": ("Airbus A350-900", ("a350-900", "a359")),
    "ICAO_AC50_500_S": ("Aero Commander 500", ("aero commander 500", "ac50")),
    "ICAO_B190_1900C": ("Beechcraft 1900", ("beechcraft 1900", "b190")),
    "ICAO_B350_B300C": ("Beechcraft King Air 350", ("king air 350", "b350")),
    "ICAO_B38M_737_8_MAX": ("Boeing 737 MAX 8", ("737 max 8", "b38m")),
    "ICAO_B412_412": ("Bell 412", ("bell 412", "b412")),
    "ICAO_B463_BAE_146_300": ("BAe 146-300", ("146-300", "b463")),
    "ICAO_B734_737_476": ("Boeing 737-400", ("737-400", "b734")),
    "ICAO_B737_737_7K2": ("Boeing 737-700", ("737-700", "b737")),
    "ICAO_B77W_777_300ER": ("Boeing 777-300ER", ("777-300", "b77w")),
    "ICAO_BE20_B200": ("Beechcraft King Air 200", ("king air 200", "b200", "be20")),
    "ICAO_BE55_95_B55": ("Beechcraft Baron 55", ("baron 55", "be55")),
    "ICAO_C172_172S": ("Cessna 172", ("cessna 172", "c172")),
    "ICAO_C182_182T": ("Cessna 182", ("cessna 182", "c182")),
    "ICAO_C208_208B": ("Cessna 208 Caravan", ("cessna 208", "caravan", "c208")),
    "ICAO_C25A_525A": ("Cessna CitationJet CJ2", ("citation cj2", "525a", "c25a")),
    "ICAO_C510_510": ("Cessna Citation Mustang", ("citation mustang", "c510")),
    "ICAO_DA40_DA_40": ("Diamond DA40", ("da40",)),
    "ICAO_DH8D_DHC_8_402": ("De Havilland Canada Dash 8 Q400", ("q400", "dh8d")),
    "ICAO_E55P_EMB_505": ("Embraer Phenom 300", ("phenom 300", "e55p")),
    "ICAO_F2TH_FALCON_2000EX": ("Dassault Falcon 2000", ("falcon 2000", "f2th")),
    "ICAO_F70_F28_MK_0070": ("Fokker 70", ("fokker 70", "f70")),
    "ICAO_G150_GULFSTREAM_G150": ("Gulfstream G150", ("gulfstream g150", "g150")),
    "ICAO_GL5T_BD_700_GLOBAL_5000": ("Bombardier Global 5000", ("global 5000", "gl5t")),
    "ICAO_GL7T_BD_700_2A12": ("Bombardier Global 7500", ("global 7500", "gl7t")),
    "ICAO_P28R_PA_28R_200": ("Piper PA-28R", ("pa-28r", "arrow", "p28r")),
    "ICAO_PA31_PA_31": ("Piper PA-31 Navajo", ("pa-31", "navajo")),
    "ICAO_PC24_PC_24": ("Pilatus PC-24", ("pc-24", "pc24")),
    "ICAO_R44_R44_II": ("Robinson R44", ("robinson r44", "r44")),
    "ICAO_SR22": ("Cirrus SR22", ("sr22",)),
    "ICAO_SW4_SA227_DC": ("Fairchild Swearingen Metroliner", ("metroliner", "sa227", "sw4")),
}

ALLOWED_LICENSE_MARKERS = ("cc by", "cc-by", "public domain", "cc0")


def _get(params: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(6):
        response = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=45)
        if response.status_code != 429:
            response.raise_for_status()
            time.sleep(1.25)
            return response.json()
        delay = min(90.0, float(response.headers.get("Retry-After", 10)) + 10 + attempt * 3)
        print(f"[Commons] hız sınırı; {delay:.0f} sn bekleniyor", flush=True)
        time.sleep(delay)
    response.raise_for_status()
    raise RuntimeError("Wikimedia API hız sınırı aşılamadı")


def _category_files(category: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    continuation: dict[str, Any] = {}
    while True:
        payload = _get({
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmtype": "file", "cmlimit": "500",
            **continuation,
        })
        rows.extend(payload.get("query", {}).get("categorymembers", []))
        if "continue" not in payload:
            break
        continuation = payload["continue"]
    return rows


def _video_info(pageids: list[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in range(0, len(pageids), 50):
        payload = _get({
            "action": "query", "format": "json", "prop": "videoinfo",
            "pageids": "|".join(map(str, pageids[start:start + 50])),
            "viprop": "url|size|mime|derivatives|extmetadata",
        })
        for page in payload.get("query", {}).get("pages", {}).values():
            if page.get("videoinfo"):
                output.append({"pageid": page["pageid"], "title": page["title"], **page["videoinfo"][0]})
    return output


def discover() -> list[dict[str, Any]]:
    OUT.mkdir(parents=True, exist_ok=True)
    found: dict[tuple[str, int], dict[str, Any]] = {}
    if DISCOVERY.is_file():
        for line in DISCOVERY.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found[(row["subtype"], int(row["pageid"]))] = row
    scanned = set()
    if CHECKPOINT.is_file():
        scanned = set(json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("scanned_types", []))
    for subtype, (name, tokens) in TYPE_MAP.items():
        if subtype in scanned:
            print(f"[Commons] {subtype}: kontrol noktası mevcut", flush=True)
            continue
        # The dedicated video category is substantially cleaner than broad
        # take-off/landing categories, which mostly contain still images.
        categories = [f"Videos of {name}"]
        members: dict[int, tuple[str, str]] = {}
        for category in categories:
            for item in _category_files(category):
                suffix = Path(item["title"].split(":", 1)[-1]).suffix.lower()
                if suffix in {".webm", ".ogv", ".ogg", ".oga", ".mp4"}:
                    members[int(item["pageid"])] = (item["title"], category)
        for info in _video_info(sorted(members)):
            title_lower = info["title"].lower().replace("_", " ")
            if not any(token in title_lower for token in tokens):
                continue
            metadata = info.get("extmetadata", {})
            license_name = html.unescape(metadata.get("LicenseShortName", {}).get("value", ""))
            if not any(marker in license_name.lower() for marker in ALLOWED_LICENSE_MARKERS):
                continue
            derivatives = [d for d in info.get("derivatives", []) if "audio" in d.get("type", "") or "opus" in d.get("type", "") or "vorbis" in d.get("type", "")]
            derivatives.sort(key=lambda d: (d.get("width", 10_000), d.get("height", 10_000)))
            media_url = derivatives[0].get("src") if derivatives else info.get("url")
            found[(subtype, info["pageid"])] = {
                "category": "AIRCRAFT", "subtype": subtype, "commons_model_name": name,
                "title": info["title"], "pageid": info["pageid"],
                "source_page": info.get("descriptionurl"), "media_url": media_url,
                "original_url": info.get("url"), "duration_seconds": info.get("duration"),
                "license": license_name,
                "artist": re.sub("<[^>]+>", "", html.unescape(metadata.get("Artist", {}).get("value", ""))).strip(),
                "source_category": members[info["pageid"]][1],
                "verification_status": "WIKIMEDIA_CATEGORY_LABEL_PENDING_AUDIO_REVIEW",
            }
        print(f"[Commons] {subtype}: {sum(key[0] == subtype for key in found)} aday", flush=True)
        scanned.add(subtype)
        checkpoint_rows = sorted(found.values(), key=lambda r: (r["subtype"], r["pageid"]))
        with DISCOVERY.open("w", encoding="utf-8", newline="\n") as stream:
            for row in checkpoint_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        CHECKPOINT.write_text(
            json.dumps({"scanned_types": sorted(scanned)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    rows = sorted(found.values(), key=lambda r: (r["subtype"], r["pageid"]))
    with DISCOVERY.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "aircraft_types_requested": len(TYPE_MAP),
        "aircraft_types_with_candidates": len({r["subtype"] for r in rows}),
        "candidate_media_count": len(rows),
        "status": "PENDING_AUDIO_AND_HUMAN_REVIEW_NOT_IN_ACTIVE_SHAZAM",
    }
    (OUT / "discovery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def stage(rows: list[dict[str, Any]], max_per_type: int) -> list[dict[str, Any]]:
    if not FFMPEG.is_file():
        raise FileNotFoundError(f"ffmpeg bulunamadı: {FFMPEG}")
    counts: dict[str, int] = {}
    staged: list[dict[str, Any]] = []
    for row in rows:
        subtype = row["subtype"]
        if counts.get(subtype, 0) >= max_per_type:
            continue
        folder = OUT / "PENDING_AUDIO_REVIEW" / subtype
        folder.mkdir(parents=True, exist_ok=True)
        wav = folder / f"commons_{row['pageid']}.wav"
        command = [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-i", row["media_url"], "-vn", "-ac", "1", "-ar", "22050",
            "-t", "120", "-c:a", "pcm_s16le", str(wav),
        ]
        try:
            subprocess.run(command, check=True, timeout=240)
            if wav.stat().st_size < 44_100:
                wav.unlink(missing_ok=True); continue
        except Exception as exc:
            wav.unlink(missing_ok=True)
            print(f"[Commons] atlandı {row['title']}: {exc}", flush=True)
            continue
        staged.append({**row, "audio_path": str(wav.resolve())})
        counts[subtype] = counts.get(subtype, 0) + 1
    with STAGED.open("w", encoding="utf-8", newline="\n") as stream:
        for row in staged:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-per-type", type=int, default=3)
    args = parser.parse_args()
    rows = discover()
    print(json.dumps({"discovered": len(rows), "types": len({r['subtype'] for r in rows})}, ensure_ascii=False))
    if args.download:
        staged = stage(rows, max(1, args.max_per_type))
        print(json.dumps({"staged": len(staged), "types": len({r['subtype'] for r in staged})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
