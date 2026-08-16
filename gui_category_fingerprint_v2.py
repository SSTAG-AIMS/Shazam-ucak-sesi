"""Separate launcher for Category Shazam v2."""

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

import gui_main
from gui_category_fingerprint_v1 import (
    ExperimentalClassificationTab,
    ExperimentalSidePanel,
)
from noise_detector_category_fp_v2 import CategoryFingerprintV2System


gui_main.AirportNoiseSystem = CategoryFingerprintV2System
gui_main.SidePanel = ExperimentalSidePanel
gui_main.ClassificationTab = ExperimentalClassificationTab


class ExperimentalV2MainWindow(gui_main.MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "✈  Havalimanı Gürültü Tespit Sistemi  v3.4 — Category Shazam v2"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(gui_main.DARK_STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))
    window = ExperimentalV2MainWindow()
    window.show()
    sys.exit(app.exec())
