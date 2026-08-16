"""Two-stage human A/B review GUI for agent-produced audio catalog records."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSlider, QSplitter, QVBoxLayout, QWidget,
)

from catalog_review import append_decision, create_human_decision, pending_records
from dataset_catalog import CatalogValidationError, ReviewStatus, Taxonomy
from reference_audio_library import ReferenceAudioLibrary
from review_comparison import extract_model_predictions, general_result


ROOT = Path(__file__).resolve().parent
DEFAULT_QUEUE = ROOT / "cache" / "catalog_review_queue_v1.jsonl"
DEFAULT_DECISIONS = ROOT / "cache" / "catalog_review_decisions_v1.jsonl"


class WaveformWidget(QWidget):
    """Small clickable waveform used for human A/B listening."""

    positionRequested = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()
        self.samples = np.asarray([], dtype=np.float32)
        self.position = 0.0
        self.setMinimumHeight(95)
        self.setToolTip("Dinlemeye başlamak istediğiniz noktaya tıklayın")

    def set_samples(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=np.float32)
        self.position = 0.0
        self.update()

    def set_position(self, position: float) -> None:
        self.position = max(0.0, min(1.0, float(position)))
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.width() > 0:
            self.positionRequested.emit(event.position().x() / self.width())

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151a20"))
        if self.samples.size:
            width = max(1, self.width())
            step = max(1, self.samples.size // width)
            envelope = np.max(np.abs(self.samples[: (self.samples.size // step) * step].reshape(-1, step)), axis=1)
            peak = float(np.max(envelope)) or 1.0
            middle = self.height() / 2
            painter.setPen(QPen(QColor("#26c6da"), 1))
            for x, value in enumerate(envelope[:width]):
                height = (float(value) / peak) * (self.height() * 0.43)
                painter.drawLine(x, int(middle - height), x, int(middle + height))
        cursor_x = int(self.position * max(0, self.width() - 1))
        painter.setPen(QPen(QColor("#ffcc4d"), 2))
        painter.drawLine(cursor_x, 0, cursor_x, self.height())


class CatalogReviewWindow(QMainWindow):
    def __init__(self, queue_path: Path | None = None) -> None:
        super().__init__()
        self.taxonomy = Taxonomy.load()
        self.reference_library = ReferenceAudioLibrary()
        self.queue_path = (queue_path or DEFAULT_QUEUE).resolve()
        self.decisions_path = self._decision_path(self.queue_path)
        self.records: list[dict] = []
        self.references = []
        self.audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
        self.active_audio: tuple[str, float, float] | None = None
        self.index = 0
        self.setWindowTitle("Ses Veri Kataloğu — Model Karşılaştırma ve İnsan Onayı")
        self.resize(1220, 780)
        self._build_ui()
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(100)
        self.play_timer.timeout.connect(self._update_play_position)
        self._reload()

    @staticmethod
    def _decision_path(queue_path: Path) -> Path:
        return queue_path.with_name(queue_path.stem.replace("queue", "decisions") + ".jsonl")

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        open_button = QPushButton("İnceleme Kuyruğunu Aç")
        open_button.clicked.connect(self._choose_queue)
        self.queue_label = QLabel()
        self.queue_label.setWordWrap(True)
        top.addWidget(open_button)
        top.addWidget(self.queue_label, 1)
        top.addWidget(QLabel("Ses çıkışı:"))
        self.output_combo = QComboBox()
        self._load_output_devices()
        top.addWidget(self.output_combo)
        layout.addLayout(top)

        self.progress_label = QLabel("Bekleyen kayıt yok")
        self.progress_label.setStyleSheet("font-size:17px; font-weight:700;")
        layout.addWidget(self.progress_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_query_panel())
        splitter.addWidget(self._build_evidence_panel())
        splitter.addWidget(self._build_reference_panel())
        splitter.setSizes([340, 470, 340])
        layout.addWidget(splitter, 1)

        correction = QGroupBox("İnsan kararı / gerekiyorsa etiketi düzelt")
        correction_layout = QFormLayout(correction)
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText("Ad Soyad")
        self.category_combo = QComboBox()
        self.category_combo.addItems(sorted(self.taxonomy.categories))
        self.category_combo.currentTextChanged.connect(self._fill_subtypes)
        self.subtype_combo = QComboBox()
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("Kararın kısa gerekçesi")
        self.note_edit.setMaximumHeight(60)
        correction_layout.addRow("İnceleyen:", self.reviewer_edit)
        correction_layout.addRow("Onaylanan ana tür:", self.category_combo)
        correction_layout.addRow("Onaylanan alt tür:", self.subtype_combo)
        correction_layout.addRow("Not:", self.note_edit)
        layout.addWidget(correction)

        decision_row = QHBoxLayout()
        reject_button = QPushButton("✕ Reddet")
        approve_button = QPushButton("✓ Dinledim, Onayla")
        reject_button.clicked.connect(lambda: self._decide(ReviewStatus.REJECTED))
        approve_button.clicked.connect(lambda: self._decide(ReviewStatus.APPROVED))
        reject_button.setStyleSheet("background:#7c2630;color:white;font-weight:700;padding:11px;")
        approve_button.setStyleSheet("background:#176b3a;color:white;font-weight:700;padding:11px;")
        decision_row.addWidget(reject_button)
        decision_row.addWidget(approve_button)
        layout.addLayout(decision_row)

        warning = QLabel(
            "Model uzlaşması otomatik bir öneridir; insan onayı değildir. Gelen kayıt ile "
            "kayıtlı referansı dinleyip karşılaştırmadan katalog onayı vermeyin."
        )
        warning.setWordWrap(True)
        warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        warning.setStyleSheet("color:#d9b44a;padding:6px;")
        layout.addWidget(warning)
        self.setCentralWidget(root)

    def _build_query_panel(self) -> QGroupBox:
        box = QGroupBox("A — İncelenen yeni ses")
        layout = QVBoxLayout(box)
        self.file_value = QLabel("-")
        self.file_value.setWordWrap(True)
        self.quality_value = QLabel("-")
        self.quality_value.setWordWrap(True)
        layout.addWidget(QLabel("Dosya:"))
        layout.addWidget(self.file_value)
        layout.addWidget(QLabel("Teknik kalite:"))
        layout.addWidget(self.quality_value)
        self.query_waveform = WaveformWidget()
        self.query_waveform.positionRequested.connect(
            lambda value: self._seek("query", value)
        )
        self.query_slider = QSlider(Qt.Orientation.Horizontal)
        self.query_slider.setRange(0, 1000)
        self.query_slider.valueChanged.connect(
            lambda value: self._slider_changed("query", value)
        )
        self.query_time = QLabel("00:00 / 00:00")
        layout.addWidget(QLabel("Dalga formu — tıklayarak konum seçebilirsiniz:"))
        layout.addWidget(self.query_waveform)
        layout.addWidget(self.query_slider)
        layout.addWidget(self.query_time)
        layout.addStretch(1)
        self.query_play_button = QPushButton("▶ Gelen Sesi Dinle")
        self.query_play_button.clicked.connect(self._play_query)
        query_pause_button = QPushButton("Ⅱ Duraklat")
        query_pause_button.clicked.connect(self._pause_audio)
        layout.addWidget(self.query_play_button)
        layout.addWidget(query_pause_button)
        return box

    def _build_evidence_panel(self) -> QGroupBox:
        box = QGroupBox("Modellerin ayrı ayrı tahminleri")
        layout = QVBoxLayout(box)
        self.models_text = QPlainTextEdit()
        self.models_text.setReadOnly(True)
        self.models_text.setStyleSheet("font-family:Consolas;font-size:13px;")
        layout.addWidget(self.models_text, 1)
        self.general_result_value = QLabel("GENEL SONUÇ: -")
        self.general_result_value.setWordWrap(True)
        self.general_result_value.setStyleSheet(
            "background:#142536;border:1px solid #2f85b8;padding:12px;font-size:16px;font-weight:700;"
        )
        self.consensus_value = QLabel("MODEL UZLAŞMASI: -")
        self.consensus_value.setWordWrap(True)
        self.consensus_value.setStyleSheet("padding:8px;font-weight:700;")
        human_state = QLabel("İNSAN KARARI: BEKLİYOR")
        human_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        human_state.setStyleSheet("color:#d9b44a;font-weight:700;")
        layout.addWidget(self.general_result_value)
        layout.addWidget(self.consensus_value)
        layout.addWidget(human_state)
        return box

    def _build_reference_panel(self) -> QGroupBox:
        box = QGroupBox("B — Kayıtlı Shazam referansı")
        layout = QVBoxLayout(box)
        self.reference_label = QLabel("Genel sonuç için farklı bir katalog kaydı aranıyor.")
        self.reference_label.setWordWrap(True)
        self.reference_combo = QComboBox()
        self.reference_combo.currentIndexChanged.connect(self._reference_changed)
        self.reference_detail = QLabel("-")
        self.reference_detail.setWordWrap(True)
        layout.addWidget(self.reference_label)
        layout.addWidget(self.reference_combo)
        layout.addWidget(self.reference_detail)
        self.reference_waveform = WaveformWidget()
        self.reference_waveform.positionRequested.connect(
            lambda value: self._seek("reference", value)
        )
        self.reference_slider = QSlider(Qt.Orientation.Horizontal)
        self.reference_slider.setRange(0, 1000)
        self.reference_slider.valueChanged.connect(
            lambda value: self._slider_changed("reference", value)
        )
        self.reference_time = QLabel("00:00 / 00:00")
        layout.addWidget(QLabel("Referans dalga formu:"))
        layout.addWidget(self.reference_waveform)
        layout.addWidget(self.reference_slider)
        layout.addWidget(self.reference_time)
        layout.addStretch(1)
        self.reference_play_button = QPushButton("▶ Referans Sesi Dinle")
        self.reference_play_button.clicked.connect(self._play_reference)
        self.reference_play_button.setEnabled(False)
        reference_pause_button = QPushButton("Ⅱ Duraklat")
        reference_pause_button.clicked.connect(self._pause_audio)
        stop_button = QPushButton("■ Durdur")
        stop_button.clicked.connect(self._stop_audio)
        layout.addWidget(self.reference_play_button)
        layout.addWidget(reference_pause_button)
        layout.addWidget(stop_button)
        return box

    def _choose_queue(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "İnceleme kuyruğunu aç", str(self.queue_path.parent), "JSON Lines (*.jsonl)"
        )
        if filename:
            self.queue_path = Path(filename)
            self.decisions_path = self._decision_path(self.queue_path)
            self._reload()

    def _reload(self) -> None:
        try:
            self.records = pending_records(self.queue_path, self.decisions_path)
        except CatalogValidationError as exc:
            QMessageBox.critical(self, "Kuyruk hatası", str(exc))
            self.records = []
        self.index = 0
        self.queue_label.setText(str(self.queue_path))
        self._show_current()

    def _show_current(self) -> None:
        sd.stop()
        if not self.records:
            self.progress_label.setText("Bekleyen kayıt yok")
            self.file_value.setText("-")
            self.models_text.clear()
            self._load_references(None)
            return

        record = self.records[self.index]
        self.progress_label.setText(f"Kayıt {self.index + 1} / {len(self.records)}")
        self.file_value.setText(str(record.get("audio_path", "-")))
        self._prepare_audio("query", Path(str(record.get("audio_path") or "")))
        issues = record.get("quality_issues") or []
        self.quality_value.setText("Uygun" if not issues else "Karantina: " + ", ".join(issues))

        rows = extract_model_predictions(record)
        if rows:
            lines = []
            for row in rows:
                confidence = f"%{100 * row['confidence']:.1f}" if row["available"] else "-"
                lines.append(
                    f"{row['name']:<13} → {row['predicted']:<12} {confidence:>7}\n"
                    f"{'':13}   Rol: {row['role']}"
                )
            self.models_text.setPlainText("\n\n".join(lines))
        else:
            self.models_text.setPlainText("Bu eski kuyruk kaydında model bazlı kanıt bulunmuyor.")

        result = general_result(record)
        self.general_result_value.setText(
            f"GENEL SONUÇ\n{result['category']}  /  {result['subtype']}"
        )
        self.consensus_value.setText(result["state"])
        accepted = result["state"] == "MODEL UZLAŞMASI KABUL EDİLDİ"
        self.consensus_value.setStyleSheet(
            f"padding:8px;font-weight:700;color:{'#65d889' if accepted else '#e5b75c'};"
        )

        category = result["category"] if result["category"] in self.taxonomy.categories else "OTHER"
        self.category_combo.setCurrentText(category)
        self._fill_subtypes(category)
        self.subtype_combo.setCurrentText(result["subtype"])
        self.note_edit.clear()
        self._load_references(record)

    def _load_references(self, record: dict | None) -> None:
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        self.references = []
        if record:
            audio_path = Path(str(record.get("audio_path") or ""))
            self.references = self.reference_library.references_for(
                str(record.get("category") or ""),
                str(record.get("subtype") or ""),
                exclude_path=audio_path,
            )
        for reference in self.references:
            self.reference_combo.addItem(reference.name)
        self.reference_combo.blockSignals(False)
        self.reference_play_button.setEnabled(bool(self.references))
        if self.references:
            self.reference_label.setText(
                f"{self.references[0].category} / {self.references[0].subtype} için "
                f"{len(self.references)} farklı katalog referansı bulundu."
            )
            self._reference_changed(0)
        else:
            self.reference_label.setText(
                "Bu genel sonuç için gelen kayıttan farklı, dinlenebilir bir Shazam referansı bulunamadı."
            )
            self.reference_detail.setText("Referans yok — onay vermeden etiketi veya kataloğu kontrol edin.")
            self.reference_waveform.set_samples(np.asarray([], dtype=np.float32))
            self.reference_slider.setValue(0)
            self.reference_time.setText("00:00 / 00:00")

    def _reference_changed(self, index: int) -> None:
        if not (0 <= index < len(self.references)):
            return
        reference = self.references[index]
        self.reference_detail.setText(
            f"Kayıt: {reference.name}\nDosya: {reference.path.name}\n"
            f"Katalog: {reference.catalog}\nParmak izi noktası: {reference.hash_count}"
        )
        self._prepare_audio("reference", reference.path)

    def _fill_subtypes(self, category: str) -> None:
        current = self.subtype_combo.currentText()
        self.subtype_combo.clear()
        self.subtype_combo.addItems(sorted(self.taxonomy.categories.get(category, [])))
        if current in self.taxonomy.categories.get(category, []):
            self.subtype_combo.setCurrentText(current)

    def _load_output_devices(self) -> None:
        self.output_combo.clear()
        try:
            default_output = sd.default.device[1]
            selected = 0
            for device_index, device in enumerate(sd.query_devices()):
                if int(device.get("max_output_channels", 0)) <= 0:
                    continue
                self.output_combo.addItem(str(device["name"]), device_index)
                if device_index == default_output:
                    selected = self.output_combo.count() - 1
            self.output_combo.setCurrentIndex(selected)
        except Exception as exc:
            self.output_combo.addItem(f"Çıkış bulunamadı: {exc}", None)

    def _audio(self, path: Path) -> tuple[np.ndarray, int]:
        path = path.resolve()
        if path not in self.audio_cache:
            samples, sr = librosa.load(str(path), sr=22050, mono=True)
            samples = np.asarray(samples, dtype=np.float32)
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0
            # This gain is only for listening; the original file and model input
            # are never modified. Quiet recordings become audible for review.
            if peak > 0:
                samples = samples * (0.92 / peak)
            self.audio_cache[path] = samples, sr
        return self.audio_cache[path]

    @staticmethod
    def _clock(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _path_for(self, kind: str) -> Path | None:
        if kind == "query" and self.records:
            return Path(str(self.records[self.index]["audio_path"]))
        index = self.reference_combo.currentIndex()
        if kind == "reference" and 0 <= index < len(self.references):
            return self.references[index].path
        return None

    def _controls_for(self, kind: str):
        if kind == "query":
            return self.query_slider, self.query_waveform, self.query_time
        return self.reference_slider, self.reference_waveform, self.reference_time

    def _prepare_audio(self, kind: str, path: Path) -> None:
        try:
            samples, sr = self._audio(path)
            slider, waveform, label = self._controls_for(kind)
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            waveform.set_samples(samples)
            label.setText(f"00:00 / {self._clock(len(samples) / sr)}")
        except Exception as exc:
            _, waveform, label = self._controls_for(kind)
            waveform.set_samples(np.asarray([], dtype=np.float32))
            label.setText(f"Ses okunamadı: {exc}")

    def _seek(self, kind: str, fraction: float) -> None:
        slider, _, _ = self._controls_for(kind)
        slider.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        if self.active_audio and self.active_audio[0] == kind:
            self._play_kind(kind)

    def _slider_changed(self, kind: str, value: int) -> None:
        path = self._path_for(kind)
        if not path:
            return
        try:
            samples, sr = self._audio(path)
            fraction = value / 1000.0
            _, waveform, label = self._controls_for(kind)
            waveform.set_position(fraction)
            label.setText(
                f"{self._clock(fraction * len(samples) / sr)} / {self._clock(len(samples) / sr)}"
            )
        except Exception:
            return

    def _play_audio(self, path: Path, kind: str) -> None:
        try:
            samples, sr = self._audio(path)
            slider, _, _ = self._controls_for(kind)
            start_seconds = (slider.value() / 1000.0) * (len(samples) / sr)
            start_sample = min(len(samples) - 1, max(0, int(start_seconds * sr)))
            device = self.output_combo.currentData()
            sd.stop()
            sd.play(samples[start_sample:], sr, device=device)
            self.active_audio = (kind, time.monotonic(), start_seconds)
            self.play_timer.start()
            if kind == "query":
                self.query_play_button.setText("▶ Devam Et / Yeniden Oynat")
            else:
                self.reference_play_button.setText("▶ Devam Et / Yeniden Oynat")
        except Exception as exc:
            QMessageBox.critical(self, "Ses oynatma hatası", str(exc))

    def _play_kind(self, kind: str) -> None:
        path = self._path_for(kind)
        if path:
            self._play_audio(path, kind)

    def _update_play_position(self) -> None:
        if not self.active_audio:
            self.play_timer.stop()
            return
        kind, started, offset = self.active_audio
        path = self._path_for(kind)
        if not path:
            self.play_timer.stop()
            return
        samples, sr = self._audio(path)
        duration = len(samples) / sr
        position = offset + (time.monotonic() - started)
        slider, waveform, label = self._controls_for(kind)
        fraction = min(1.0, position / duration) if duration else 0.0
        slider.blockSignals(True)
        slider.setValue(int(fraction * 1000))
        slider.blockSignals(False)
        waveform.set_position(fraction)
        label.setText(f"{self._clock(position)} / {self._clock(duration)}")
        if position >= duration:
            self.play_timer.stop()
            self.active_audio = None

    def _stop_audio(self) -> None:
        sd.stop()
        self.play_timer.stop()
        if self.active_audio:
            slider, waveform, _ = self._controls_for(self.active_audio[0])
            slider.setValue(0)
            waveform.set_position(0.0)
        self.active_audio = None

    def _pause_audio(self) -> None:
        """Pause while preserving the current point for a later resume."""
        if not self.active_audio:
            return
        self._update_play_position()
        sd.stop()
        self.play_timer.stop()
        self.active_audio = None

    def _play_query(self) -> None:
        self._play_kind("query")

    def _play_reference(self) -> None:
        self._play_kind("reference")

    def _decide(self, status: ReviewStatus) -> None:
        if not self.records:
            return
        try:
            decision = create_human_decision(
                self.records[self.index], reviewer=self.reviewer_edit.text(), status=status,
                category=self.category_combo.currentText(), subtype=self.subtype_combo.currentText(),
                note=self.note_edit.toPlainText(), taxonomy=self.taxonomy,
            )
            append_decision(decision, self.decisions_path)
        except (CatalogValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Karar kaydedilemedi", str(exc))
            return
        sd.stop()
        self.records.pop(self.index)
        if self.index >= len(self.records):
            self.index = max(0, len(self.records) - 1)
        self._show_current()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ses kataloğu A/B insan onay ekranı")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CatalogReviewWindow(args.queue)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
