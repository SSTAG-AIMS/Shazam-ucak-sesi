"""Separate launcher for the experimental category fingerprint v1 build."""

import sys

import numpy as np
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

import gui_main
from noise_detector_category_fp_v1 import CategoryFingerprintV1System


gui_main.AirportNoiseSystem = CategoryFingerprintV1System


class ExperimentalSidePanel(gui_main.SidePanel):
    """Show which subtype engine produced the final experimental result."""

    def update_stats(self, result):
        super().update_stats(result)
        match = result.get("subtype_match") or {}
        if not match:
            return

        subtype = match.get("subtype", "UNKNOWN")
        confidence = float(match.get("confidence", 0.0)) * 100.0
        method = match.get("method", "")
        if method == "shazam_v1":
            detail = f"%{confidence:.1f} · Parmak izi"
        elif method in {
            "beats_multi_window_vote",
            "beats_multi_window_vote_v2_calibrated",
        }:
            predicted = match.get("predicted_subtype", subtype)
            votes = match.get("vote_counts", {})
            won_votes = int(votes.get(predicted, 0))
            n_windows = int(match.get("n_windows", 0))
            vote_text = f" · {won_votes}/{n_windows}" if n_windows else ""
            method_text = (
                "BEATs v2"
                if method == "beats_multi_window_vote_v2_calibrated"
                else "BEATs"
            )
            detail = f"%{confidence:.1f}{vote_text} · {method_text}"
        else:
            return
        self._set_subtype_display(subtype, detail)


class ExperimentalClassificationTab(gui_main.ClassificationTab):
    """Make a one-window confidence result visible as points."""

    def _draw_confidence(self, result):
        super()._draw_confidence(result)
        probabilities = result.get("frame_probs")
        times = result.get("frame_times")
        if probabilities is None or times is None:
            return
        if len(probabilities) != 1 or len(times) != 1 or not self.fig3.axes:
            return
        values = np.asarray(probabilities[0])
        ax = self.fig3.axes[0]
        for index, class_name in enumerate(result.get("class_names", [])):
            if index >= len(values):
                continue
            ax.scatter(
                [times[0]],
                [values[index]],
                color=gui_main.CLASS_COLORS.get(class_name, "#8B949E"),
                s=28,
                edgecolors=gui_main.PALETTE["text"],
                linewidths=0.35,
                zorder=6,
            )
        self.canvas3.draw()


gui_main.SidePanel = ExperimentalSidePanel
gui_main.ClassificationTab = ExperimentalClassificationTab


class ExperimentalMainWindow(gui_main.MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "✈  Havalimanı Gürültü Tespit Sistemi  v3.4 — Category Shazam v1"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(gui_main.DARK_STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))
    window = ExperimentalMainWindow()
    window.show()
    sys.exit(app.exec())
