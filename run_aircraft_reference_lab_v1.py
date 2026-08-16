"""One-command launcher for the isolated aircraft reference test lab."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from gui_aircraft_reference_intake_v1 import AircraftReferenceIntakeWindow
from prepare_aircraft_reference_lab_v1 import LAB, prepare_lab


def main() -> None:
    prepare_lab()
    workspace = LAB / "workspace"
    app = QApplication(sys.argv); app.setStyle("Fusion")
    window = AircraftReferenceIntakeWindow(
        queue_path=workspace / "intake_queue.jsonl",
        decisions_path=workspace / "intake_decisions.jsonl",
        database_path=workspace / "aircraft_test_fingerprints.sqlite3",
        inbox_path=workspace / "BEKLEYEN",
    )
    window.setWindowTitle("Uçak Referans Laboratuvarı V1 — İZOLE TEST MODU")
    window.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()
