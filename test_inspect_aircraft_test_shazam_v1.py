import sqlite3
import tempfile
import unittest
from pathlib import Path

from inspect_aircraft_test_shazam_v1 import inspect_database


class InspectAircraftTestShazamTests(unittest.TestCase):
    def test_reads_tracks_and_fingerprint_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "x.sqlite3"; connection = sqlite3.connect(db)
            connection.executescript("CREATE TABLE tracks(aircraft_type,reference_name,source_path,hash_count); CREATE TABLE fingerprints(hash,track_id,anchor_time);")
            connection.execute("INSERT INTO tracks VALUES (?,?,?,?)", ("A320", "r", "X/KABUL_EDILEN/a.wav", 2))
            connection.executemany("INSERT INTO fingerprints VALUES (?,?,?)", [("a", 1, 0), ("b", 1, 1)])
            connection.commit(); connection.close()
            proof = inspect_database(db)
            self.assertEqual(proof["track_count"], 1); self.assertEqual(proof["fingerprint_count"], 2)
            self.assertTrue(proof["all_sources_are_accepted"])


if __name__ == "__main__":
    unittest.main()
