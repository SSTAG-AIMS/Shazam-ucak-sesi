import csv
import json
import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path

import build_source_verified_shazam_10000_v2 as builder


class SourceVerifiedShazam10000Tests(unittest.TestCase):
    def test_category_mapping_is_deterministic(self):
        self.assertEqual(builder._category_for(["Fixed-wing_aircraft_and_airplane"]), ("AIRCRAFT", "FIXED_WING_AIRCRAFT_AND_AIRPLANE"))
        self.assertEqual(builder._category_for(["Train"]), ("TRAFFIC", "TRAIN"))
        self.assertEqual(builder._category_for(["Speech"]), ("SPEECH", "SPEECH"))
        self.assertEqual(builder._category_for(["Wind"]), ("WIND", "WIND"))
        self.assertEqual(builder._category_for(["Rain"]), ("AMBIENT", "RAIN"))
        self.assertEqual(builder._category_for(["Dog"]), ("OTHER", "DOG"))

    def test_fsd_metadata_has_enough_permitted_records(self):
        info_path = builder.FSD_ROOT / "metadata" / "FSD50K.metadata" / "eval_clips_info_FSD50K.json"
        labels_path = builder.FSD_ROOT / "metadata" / "FSD50K.metadata" / "collection" / "collection_eval.csv"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        with labels_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        eligible = [row for row in rows if info[row["fname"]]["license"] in builder.ALLOWED_FSD_LICENSES]
        self.assertEqual(len(rows), 10_231)
        self.assertEqual(len(eligible), 9_828)

    def test_esc_fold_split_has_no_source_overlap(self):
        accepted, tests = builder._esc_rows()
        self.assertEqual(len(accepted) + len(tests), 2_000)
        self.assertGreaterEqual(len(tests), 400)
        self.assertFalse(
            {row["source_recording_id"] for row in accepted}
            & {row["source_recording_id"] for row in tests}
        )

    def test_parallel_index_builder_writes_real_tracks_and_hashes(self):
        accepted, _ = builder._esc_rows()
        rows = []
        for row in accepted[:2]:
            rows.append({**row, "audio_path": row["path"]})
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "test.sqlite3"
            total = builder._build_index(rows, database)
            self.assertGreater(total, 0)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0], 2)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0], total)

    def test_unique_rows_rejects_same_source_and_same_waveform(self):
        accepted, _ = builder._esc_rows()
        first = accepted[0]
        same_source = {**accepted[1], "source_recording_id": first["source_recording_id"]}
        same_audio = {**accepted[2], "path": first["path"]}
        unique, duplicates, source_ids, hashes = builder._unique_rows([first, same_source, same_audio])
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(duplicates), 2)
        self.assertEqual(len(source_ids), 1)
        self.assertEqual(len(hashes), 1)

    def test_fingerprint_quality_gate_rejects_landmarkless_clip(self):
        silent = next(builder.FSD_ROOT.rglob("118229.wav"))
        valid = builder._esc_rows()[0][0]["path"]
        self.assertFalse(builder._is_fingerprintable({"path": str(silent)}))
        self.assertTrue(builder._is_fingerprintable({"path": valid}))

    def test_built_catalog_integrity_when_present(self):
        manifest = builder.OUTPUT / "index_manifest.json"
        if not manifest.is_file():
            self.skipTest("10.000 kayıtlık katalog henüz üretilmedi")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["accepted_count"], 10_000)
        self.assertEqual(payload["leakage_count"], 0)
        accepted = [json.loads(line) for line in (builder.OUTPUT / "accepted_source_labels.jsonl").read_text(encoding="utf-8").splitlines()]
        tests = [json.loads(line) for line in (builder.OUTPUT / "independent_test.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(accepted), 10_000)
        accepted_paths = {Path(row["audio_path"]).resolve() for row in accepted}
        test_paths = {Path(row["audio_path"]).resolve() for row in tests}
        self.assertEqual(accepted_paths, {path.resolve() for path in (builder.OUTPUT / "ACCEPTED_SOURCE_LABELS").rglob("*.wav")})
        self.assertEqual(test_paths, {path.resolve() for path in (builder.OUTPUT / "INDEPENDENT_TEST").rglob("*.wav")})
        self.assertFalse({row["sha256"] for row in accepted} & {row["sha256"] for row in tests})
        self.assertFalse({row["source_recording_id"] for row in accepted} & {row["source_recording_id"] for row in tests})
        if payload["database_built"]:
            with closing(sqlite3.connect(payload["database"])) as connection:
                track_count = connection.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            self.assertEqual(track_count, 10_000)


if __name__ == "__main__":
    unittest.main()
