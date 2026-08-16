import sqlite3
import tempfile
import unittest
from pathlib import Path

from reference_audio_library import ReferenceAudioLibrary


class ReferenceAudioLibraryTests(unittest.TestCase):
    def test_excludes_query_and_its_byte_identical_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = root / "query.wav"
            duplicate = root / "duplicate.wav"
            other = root / "other.wav"
            query.write_bytes(b"same-audio")
            duplicate.write_bytes(b"same-audio")
            other.write_bytes(b"different-audio")
            db = root / "aircraft.sqlite3"
            connection = sqlite3.connect(db)
            try:
                connection.execute(
                    "CREATE TABLE tracks (reference_name TEXT, source_path TEXT, "
                    "hash_count INTEGER, aircraft_type TEXT)"
                )
                connection.executemany(
                    "INSERT INTO tracks VALUES (?, ?, ?, ?)",
                    [("query", str(query), 9, "AIRBUS_A320"),
                     ("duplicate", str(duplicate), 8, "AIRBUS_A320"),
                     ("other", str(other), 7, "AIRBUS_A320")],
                )
                connection.commit()
            finally:
                connection.close()
            library = ReferenceAudioLibrary(aircraft_db=db, category_db=root / "missing.db")
            refs = library.references_for("AIRCRAFT", "AIRBUS_A320", exclude_path=query)
            self.assertEqual([ref.name for ref in refs], ["other"])


if __name__ == "__main__":
    unittest.main()
