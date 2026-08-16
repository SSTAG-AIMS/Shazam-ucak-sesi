from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

import build_aircraft_shazam_3000_v1 as builder


class AircraftShazam3000Tests(unittest.TestCase):
    def test_candidate_pool_supports_target_without_blind_airframes(self) -> None:
        rows = builder._load_source_rows()
        selected = builder._balanced(builder._candidate_segments(rows), builder.TARGET)
        blind = json.loads(
            (builder.ROOT / "Test_Folder/AIRCRAFT_REFERENCE_LAB_V1/test_manifest.json").read_text(
                encoding="utf-8"
            )
        )["records"]
        blind_ids = {str(row["physical_airframe_id"]).upper() for row in blind}
        self.assertEqual(len(selected), 3_000)
        self.assertTrue({row["hex_id"].upper() for row in selected}.isdisjoint(blind_ids))

    def test_built_catalog_has_exactly_3000_real_tracks(self) -> None:
        if not builder.REPORT.is_file():
            self.skipTest("3000-track catalogue has not been built")
        report = json.loads(builder.REPORT.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in builder.MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with sqlite3.connect(builder.DATABASE) as connection:
            tracks = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            fingerprints = connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
        self.assertEqual(report["accepted_clip_count"], 3_000)
        self.assertEqual(len(rows), 3_000)
        self.assertEqual(tracks, 3_000)
        self.assertGreater(fingerprints, 0)
        self.assertEqual(sum(Path(row["audio_path"]).is_file() for row in rows), 3_000)


if __name__ == "__main__":
    unittest.main()
