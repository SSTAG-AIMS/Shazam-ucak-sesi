"""Clean A/B human-review GUI for the aircraft lab.

Workflow: labeling agent -> verified golden reference -> human decision -> Shazam.
Shazam is never used for prediction or golden-reference selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSlider, QVBoxLayout, QWidget,
    QSizePolicy, QTextBrowser,
)

from aircraft_reference_intake_v1 import DEFAULT_QUEUE, read_jsonl, stage_reference
from aircraft_reference_prediction_v1 import predict_aircraft
from aircraft_audio_comparison_v1 import compare_audio, log_mel_spectrogram
from aircraft_reference_review_v1 import (
    DEFAULT_DATABASE, DEFAULT_DECISIONS, append_review_decision, append_uncertain_decision,
    build_human_verified_index, latest_decisions, pending_intakes,
)
from dataset_catalog import sha256_file
from gui_catalog_review import WaveformWidget
from inspect_aircraft_test_shazam_v1 import inspect_database


class SpectrogramWidget(QWidget):
    """Compact log-mel image used as human-readable visual evidence."""

    def __init__(self) -> None:
        super().__init__(); self._image = QImage(); self.setMinimumHeight(80)

    def set_samples(self, samples: np.ndarray, sample_rate: int) -> None:
        if not samples.size:
            self._image = QImage(); self.update(); return
        mel = log_mel_spectrogram(samples, sample_rate)
        values = np.clip((mel + 80.0) / 80.0, 0.0, 1.0)[::-1]
        red = np.clip(255 * (1.8 * values - .55), 0, 255)
        green = np.clip(255 * (1.8 * values), 0, 255)
        blue = np.clip(255 * (1.15 - values), 0, 255)
        rgb = np.ascontiguousarray(np.stack([red, green, blue], axis=2).astype(np.uint8))
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        self._image = image.copy(); self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self); painter.fillRect(self.rect(), QColor("#121820"))
        if not self._image.isNull(): painter.drawImage(self.rect(), self._image)


class AudioDeck(QGroupBox):
    """Seekable audio deck whose gain changes immediately during playback."""

    playRequested = pyqtSignal(object)

    def __init__(self, title: str, accent: str, device_getter) -> None:
        super().__init__(title)
        self.accent = accent
        self.device_getter = device_getter
        self.samples = np.asarray([], dtype=np.float32)
        self.sample_rate = 22050
        self.path: Path | None = None
        self.position_samples = 0
        self.play_origin = 0
        self.play_started_at = 0.0
        self.playing = False
        self._slider_internal = False
        self.setStyleSheet(f"QGroupBox{{font-weight:800;border:1px solid {accent};border-radius:8px;margin-top:12px;padding-top:9px;}}")
        self._build()
        self.timer = QTimer(self); self.timer.setInterval(80); self.timer.timeout.connect(self._tick)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        self.name_label = QLabel("Ses seçilmedi"); self.name_label.setWordWrap(True)
        self.name_label.setStyleSheet("font-weight:700;padding:3px;")
        self.wave = WaveformWidget(); self.wave.setMinimumHeight(125); self.wave.positionRequested.connect(self.seek_ratio)
        self.spectrogram = SpectrogramWidget()
        self.seek = QSlider(Qt.Orientation.Horizontal); self.seek.setRange(0, 1000); self.seek.valueChanged.connect(self._seek_changed)
        self.time_label = QLabel("00:00.0 / 00:00.0"); self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav = QHBoxLayout()
        for text, handler in (
            ("−5 sn", lambda: self.skip(-5)), ("▶ Dinle", self.play),
            ("Ⅱ Beklet", self.pause), ("+5 sn", lambda: self.skip(5)),
            ("■ Bitir", self.stop),
        ):
            button = QPushButton(text); button.setStyleSheet("font-size:11px;padding:5px 3px;"); button.clicked.connect(handler); nav.addWidget(button)
        gain = QHBoxLayout(); gain.addWidget(QLabel("Ses:"))
        self.volume = QSlider(Qt.Orientation.Horizontal); self.volume.setRange(0, 200); self.volume.setValue(100)
        self.volume.setTickInterval(25); self.volume.valueChanged.connect(self._gain_changed)
        self.volume_label = QLabel("%100"); self.volume_label.setMinimumWidth(45)
        gain.addWidget(self.volume, 1); gain.addWidget(self.volume_label)
        layout.addWidget(self.name_label); layout.addWidget(self.wave); layout.addWidget(self.spectrogram); layout.addWidget(self.seek)
        layout.addWidget(self.time_label); layout.addLayout(nav); layout.addLayout(gain)

    def set_audio(self, path: Path | None, subtitle: str = "") -> None:
        self.stop(); self.path = path
        if not path or not path.is_file():
            self.samples = np.asarray([], dtype=np.float32); self.name_label.setText("Referans bulunamadı")
            self.wave.set_samples(self.samples); self.spectrogram.set_samples(self.samples, self.sample_rate); self._update_ui(0); return
        samples, self.sample_rate = librosa.load(str(path), sr=22050, mono=True)
        samples = np.asarray(samples, dtype=np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        # Playback-only normalization: quiet recordings become audible at 100%.
        if peak > 0: samples = np.clip(samples * (0.88 / peak), -1.0, 1.0)
        self.samples = samples; self.position_samples = 0
        self.name_label.setText(f"{subtitle}\n{path.name}" if subtitle else path.name)
        self.wave.set_samples(samples); self.spectrogram.set_samples(samples, self.sample_rate); self._update_ui(0)

    def duration(self) -> float: return len(self.samples) / max(1, self.sample_rate)
    def position(self) -> float:
        if self.playing:
            return min(self.duration(), self.play_origin / self.sample_rate + time.monotonic() - self.play_started_at)
        return self.position_samples / max(1, self.sample_rate)

    @staticmethod
    def _fmt(sec: float) -> str: return f"{int(sec)//60:02d}:{int(sec)%60:02d}.{int((sec%1)*10):01d}"

    def _update_ui(self, sec: float) -> None:
        duration = self.duration(); ratio = sec / duration if duration else 0
        self._slider_internal = True; self.seek.setValue(round(ratio * 1000)); self._slider_internal = False
        self.wave.set_position(ratio); self.time_label.setText(f"{self._fmt(sec)} / {self._fmt(duration)}")

    def _seek_changed(self, value: int) -> None:
        if self._slider_internal: return
        resume = self.playing
        if resume: self.pause()
        self.position_samples = round(value / 1000 * len(self.samples)); self._update_ui(self.position())
        if resume: self.play()

    def seek_ratio(self, ratio: float) -> None:
        self.seek.setValue(round(max(0, min(1, ratio)) * 1000))

    def skip(self, seconds: float) -> None:
        target = max(0.0, min(self.duration(), self.position() + seconds))
        self.seek.setValue(round(target / max(.001, self.duration()) * 1000))

    def play(self) -> None:
        if not self.samples.size: return
        self.playRequested.emit(self)
        if self.position_samples >= len(self.samples): self.position_samples = 0
        audio = self.samples[self.position_samples:].copy()
        audio = np.clip(audio * (self.volume.value() / 100.0), -1.0, 1.0)
        sd.play(audio, self.sample_rate, device=self.device_getter())
        self.play_origin = self.position_samples; self.play_started_at = time.monotonic()
        self.playing = True; self.timer.start()

    def pause(self) -> None:
        if not self.playing: return
        sec = self.position(); sd.stop(); self.playing = False; self.timer.stop()
        self.position_samples = min(len(self.samples), round(sec * self.sample_rate)); self._update_ui(sec)

    def stop(self) -> None:
        if self.playing: sd.stop()
        self.playing = False
        if hasattr(self, "timer"): self.timer.stop()
        self.position_samples = 0
        if hasattr(self, "seek"): self._update_ui(0)

    def stop_other(self) -> None:
        if self.playing: self.pause()

    def _gain_changed(self, value: int) -> None:
        self.volume_label.setText(f"%{value}")
        # Make volume changes audible immediately, preserving exact position.
        if self.playing:
            sec = self.position(); self.pause(); self.position_samples = round(sec * self.sample_rate); self.play()

    def _tick(self) -> None:
        sec = self.position()
        if sec >= self.duration(): self.stop()
        else: self._update_ui(sec)


class AircraftReferenceIntakeWindow(QMainWindow):
    def __init__(
        self, *, queue_path: Path = DEFAULT_QUEUE, decisions_path: Path = DEFAULT_DECISIONS,
        database_path: Path = DEFAULT_DATABASE, inbox_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.queue_path = queue_path.resolve(); self.decisions_path = decisions_path.resolve()
        self.database_path = database_path.resolve(); self.inbox_path = (inbox_path or self.queue_path.parent / "BEKLEYEN").resolve()
        self.manifest_path = self.queue_path.parent.parent / "test_manifest.json"
        self.manifest = self._manifest(); self.test_rows = {str(Path(r["audio_path"]).resolve()).lower(): r for r in self.manifest.get("records", [])}
        # Açık adla sunum klasörüne kopyalanan kör test kayıtlarını da yalnızca
        # içerikleri manifestteki test kaydıyla birebir aynıysa kabul et.
        self.test_rows_by_hash = {
            sha256_file(Path(row["audio_path"])): row
            for row in self.manifest.get("records", [])
            if Path(row["audio_path"]).is_file()
        }
        self.gold: dict[str, list[dict]] = {}
        for row in self.manifest.get("gold_references", []):
            if row.get("verified"):
                self.gold.setdefault(str(row["aircraft_type"]).upper(), []).append(row)
        self.current_record: dict | None = None; self.current_prediction: dict | None = None
        self.current_query_path: Path | None = None; self.current_comparisons: list[dict] = []
        self.setWindowTitle("Uçak Etiketleme Laboratuvarı — Agent / Altın Referans / İnsan / Shazam")
        self.resize(1520, 900); self._build(); self._refresh_counts(); self._load_proof()

    def _manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.is_file() else {}

    def _build(self) -> None:
        root = QWidget(); outer = QVBoxLayout(root); outer.setContentsMargins(14, 10, 14, 10); outer.setSpacing(8)
        header = QHBoxLayout(); title = QLabel("UÇAK SESİ İNSAN ONAY LABORATUVARI")
        title.setStyleSheet("font-size:20px;font-weight:900;color:#19d7f2;")
        title.setMaximumWidth(430)
        choose = QPushButton("＋ TEST SESİ SEÇ"); choose.setStyleSheet("background:#176b3a;color:white;font-weight:800;padding:10px 20px;"); choose.clicked.connect(self._choose)
        catalogued = int(self.manifest.get("catalogued_subtype_count", len(self.gold)))
        agent_types = int(self.manifest.get("agent_subtype_count", len({r.get("aircraft_type") for r in self.manifest.get("records", [])})))
        coverage = QLabel(f"KATALOG {catalogued} TİP • AGENT {agent_types} TİP")
        coverage.setMaximumWidth(270); coverage.setStyleSheet("color:#e0b94d;font-size:11px;font-weight:800;")
        header.addWidget(title); header.addWidget(coverage); header.addStretch(1); header.addWidget(QLabel("Ses çıkışı:")); self.output = QComboBox(); self.output.setFixedWidth(280); self._outputs(); header.addWidget(self.output); header.addWidget(choose)
        outer.addLayout(header)

        cards = QHBoxLayout(); cards.setSpacing(10)
        self.query_deck = AudioDeck("1 — GELEN TEST SESİ", "#1ebbd7", self._device)
        self.query_deck.playRequested.connect(self._deck_requested); cards.addWidget(self.query_deck, 4)
        prediction_card = self._prediction_card(); prediction_card.setMaximumWidth(430); cards.addWidget(prediction_card, 3)
        reference_column = QWidget(); reference_layout = QVBoxLayout(reference_column); reference_layout.setContentsMargins(0, 0, 0, 0)
        selector_row = QHBoxLayout(); selector_row.addWidget(QLabel("Doğrulanmış referans:"))
        self.reference_selector = QComboBox(); self.reference_selector.currentIndexChanged.connect(self._reference_changed)
        self.reference_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.reference_selector.setMinimumContentsLength(22)
        selector_row.addWidget(self.reference_selector, 1); reference_layout.addLayout(selector_row)
        self.reference_deck = AudioDeck("3 — İNSAN DOĞRULAMALI ALTIN REFERANS", "#e0b94d", self._device)
        self.reference_deck.playRequested.connect(self._deck_requested); reference_layout.addWidget(self.reference_deck, 1)
        cards.addWidget(reference_column, 4)
        outer.addLayout(cards, 1)
        outer.addWidget(self._decision_bar())
        outer.addWidget(self._proof_bar())
        self.setCentralWidget(root); self.statusBar().showMessage("Bir test sesi seçin — Shazam tahminde kullanılmaz")

    def _prediction_card(self) -> QGroupBox:
        box = QGroupBox("2 — AGENT TAHMİNİ"); box.setStyleSheet("QGroupBox{font-weight:800;border:1px solid #5686b0;border-radius:8px;margin-top:12px;padding-top:9px;}")
        layout = QVBoxLayout(box); self.predicted = QLabel("Henüz tahmin yok"); self.predicted.setWordWrap(True)
        self.predicted.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.predicted.setMinimumHeight(104); self.predicted.setMaximumHeight(112)
        self.predicted.setStyleSheet("background:#142536;padding:7px;font-size:14px;font-weight:900;color:#e8f5ff;")
        self.audio_models = QTextBrowser()
        self.audio_models.setHtml("<b>GELİŞMİŞ SES KANALLARI</b><br>Analiz bekleniyor")
        self.audio_models.setMinimumHeight(154); self.audio_models.setMaximumHeight(168)
        self.audio_models.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.audio_models.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.audio_models.setStyleSheet(
            "background:#102c31;border:1px solid #28a9b7;border-radius:5px;"
            "padding:7px;color:#b9f7ff;font-size:11px;font-weight:700;"
        )
        self.models = QPlainTextEdit(); self.models.setReadOnly(True)
        self.models.setMinimumHeight(58); self.models.setMaximumHeight(66)
        self.models.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.models.setStyleSheet("font-family:Consolas;font-size:10px;color:#d8dde3;")
        self.similarity = QLabel("Karşılaştırma kanıtı henüz yok"); self.similarity.setWordWrap(True)
        self.similarity.setStyleSheet("background:#1b2027;color:#9ed9ec;padding:10px;font-weight:700;")
        flow = QLabel(
            "BEATs + Fusion uzlaşırsa gelişmiş aday kullanılır; ayrışmada güvenli yedek yalnızca öneri verir.\n"
            "Shazam bu aşamada kapalıdır. Kesin etiket insan onayıyla oluşur."
        )
        flow.setWordWrap(True); flow.setStyleSheet("color:#e0b94d;font-size:10px;font-weight:700;padding:4px;")
        layout.addWidget(self.predicted); layout.addWidget(self.audio_models); layout.addWidget(self.models); layout.addWidget(self.similarity); layout.addWidget(flow); layout.addStretch(1); return box

    def _decision_bar(self) -> QGroupBox:
        box = QGroupBox("4 — İNSAN KARARI"); grid = QGridLayout(box)
        self.reviewer = QLineEdit(); self.reviewer.setPlaceholderText("İnceleyen Ad Soyad")
        self.approved_type = QLineEdit(); self.approved_type.setPlaceholderText("Onaylanacak uçak tipi")
        self.note = QLineEdit(); self.note.setPlaceholderText("Kısa karar gerekçesi")
        self.source_truth = QLabel("ADS-B kaynak etiketi: gizli")
        reveal = QPushButton("KAYNAK ETİKETİNİ GÖSTER"); reveal.clicked.connect(self._reveal_source_truth)
        reject = QPushButton("✕ REDDET"); reject.setStyleSheet("background:#8b3340;color:white;font-weight:900;padding:13px;"); reject.clicked.connect(lambda: self._decide(False))
        uncertain = QPushButton("? EMİN DEĞİLİM"); uncertain.setStyleSheet("background:#8a6a1f;color:white;font-weight:900;padding:13px;"); uncertain.clicked.connect(self._uncertain)
        approve = QPushButton("✓ ONAYLA → SHAZAM"); approve.setStyleSheet("background:#1b7542;color:white;font-weight:900;padding:13px;"); approve.clicked.connect(lambda: self._decide(True))
        grid.addWidget(QLabel("İnceleyen:"), 0, 0); grid.addWidget(self.reviewer, 0, 1)
        grid.addWidget(QLabel("Agent etiketi / düzeltme:"), 0, 2); grid.addWidget(self.approved_type, 0, 3)
        grid.addWidget(QLabel("Not:"), 1, 0); grid.addWidget(self.note, 1, 1, 1, 3)
        grid.addWidget(self.source_truth, 2, 0, 1, 2); grid.addWidget(reveal, 2, 2, 1, 2)
        grid.addWidget(reject, 3, 0); grid.addWidget(uncertain, 3, 1, 1, 2); grid.addWidget(approve, 3, 3); return box

    def _proof_bar(self) -> QFrame:
        frame = QFrame(); frame.setStyleSheet("QFrame{background:#111b17;border:1px solid #2e8b57;border-radius:6px;}")
        row = QHBoxLayout(frame); self.counts = QLabel("BEKLEYEN 0 | KABUL 0 | RED 0 | EMİN DEĞİL 0"); self.counts.setMaximumWidth(480)
        self.proof = QLabel("Shazam: henüz indeks yok"); self.proof.setWordWrap(True); self.proof.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        inspect = QPushButton("SQLite KANITINI GÖSTER"); inspect.clicked.connect(self._inspect)
        row.addWidget(self.counts); row.addWidget(self.proof, 1); row.addWidget(inspect); return frame

    def _outputs(self) -> None:
        try:
            for index, item in enumerate(sd.query_devices()):
                if item.get("max_output_channels", 0) > 0: self.output.addItem(f"{index}: {item['name']}", index)
            found = self.output.findData(sd.default.device[1]); self.output.setCurrentIndex(found if found >= 0 else 0)
        except Exception: self.output.addItem("Sistem varsayılanı", None)

    def _device(self): return self.output.currentData()
    def _deck_requested(self, deck: AudioDeck) -> None:
        other = self.reference_deck if deck is self.query_deck else self.query_deck; other.stop_other()

    def _test_metadata(self, path: Path) -> dict | None:
        """Asıl manifest yolunu veya onun birebir sunum kopyasını çöz."""
        resolved = path.resolve()
        direct = self.test_rows.get(str(resolved).lower())
        if direct:
            return direct
        try:
            return self.test_rows_by_hash.get(sha256_file(resolved))
        except (OSError, ValueError):
            return None

    def _choose(self) -> None:
        start = str(self.manifest_path.parent / "TEST_SESLERI")
        value, _ = QFileDialog.getOpenFileName(self, "Agent için test sesi seç", start, "Ses (*.wav *.flac *.mp3 *.ogg *.m4a)")
        if value:
            selected = Path(value).resolve()
            if self._test_metadata(selected) is None:
                QMessageBox.warning(
                    self,
                    "Geçersiz test kaydı",
                    "Bu ses bağımsız test manifestinde bulunmuyor.\n\n"
                    "Yalnız TEST_SESLERI klasöründeki manifest kayıtlarını seçin; "
                    "altın referanslar test olarak kullanılamaz.",
                )
                return
            self._analyze(selected)

    def _analyze(self, path: Path) -> None:
        path = path.resolve()
        if self._test_metadata(path) is None:
            QMessageBox.warning(
                self, "Geçersiz test kaydı",
                "Dosya bağımsız test manifestinde yok; analiz ve karar işlemi başlatılmadı.",
            )
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            prediction = predict_aircraft(path); self.current_prediction = prediction; self.current_query_path = path.resolve()
            self.query_deck.set_audio(path, "AGENTA GİREN SES")
            label = str(prediction["predicted_subtype"]); self.approved_type.setText(label)
            source = prediction.get("decision_source", "")
            advanced_agreement = source.startswith("BEATs")
            source_short = "Gelişmiş ses uzlaşması" if advanced_agreement else "Modeller ayrıştı"
            status = "İNSAN DOĞRULAMASI GEREKLİ" if not advanced_agreement else "ADAY ETİKET"
            self.predicted.setStyleSheet(
                "background:#142536;padding:7px;font-size:14px;font-weight:900;color:#e8f5ff;"
                if advanced_agreement else
                "background:#352b16;border:1px solid #d9a928;padding:7px;font-size:14px;font-weight:900;color:#fff0bd;"
            )
            self.predicted.setText(
                f"{status}\n{label}\nGüven %{prediction['confidence']*100:.1f}  •  {source_short}"
            )
            audio_rows = prediction.get("audio_channels", [])
            if audio_rows:
                short_names = {
                    "BEATs-SVM": "BEATs",
                    "AST-SVM": "AST",
                    "PANNs-CNN14-SVM": "PANNs/CNN14",
                    "CLAP-SVM": "CLAP",
                    "Multi-Embedding Fusion": "FUSION",
                    "AST-FineTune-v4 (bagimsiz testli)": "AST FT-V4*",
                }
                table_rows = "".join(
                    "<tr>"
                    f"<td width='78'><b>{short_names.get(row['model'], row['model'])}</b></td>"
                    f"<td>{row['predicted']}</td>"
                    f"<td width='48' align='right'>%{row['confidence']*100:.1f}</td>"
                    "</tr>"
                    for row in audio_rows
                )
                self.audio_models.setHtml(
                    "<div style='font-size:11px;color:#b9f7ff'>"
                    f"<b>{len(audio_rows)} SES KANALI</b> &nbsp; "
                    "<span style='color:#ffd65c'>* bağımsız testli, denetim kanalı</span>"
                    f"<table width='100%' cellspacing='2'>{table_rows}</table>"
                    "</div>"
                )
            else:
                self.audio_models.setHtml("<b>SES ODAKLI MODELLER</b><br>Kanıt üretilemedi")
            legacy = prediction.get("legacy_fallback", {})
            self.models.setPlainText(
                f"DURUM: {source_short}\n"
                f"YEDEK ÖNERİ: {legacy.get('predicted', '-')}  "
                f"({legacy.get('votes', 0)}/{legacy.get('total_models', 0)})\n"
                "SON ETİKET: Henüz insan onayı yok"
            )
            self._set_references(label)
            self.current_record = self._ensure_pending(path, label)
            if self.current_record:
                self.current_record["agent_prediction"] = prediction
                self.current_record["comparison_evidence"] = self.current_comparisons
            self.source_truth.setText("ADS-B kaynak etiketi: gizli")
            if self.current_record:
                self.statusBar().showMessage("Agent tahmini hazır. İki sesi A/B dinleyip insan kararını verin.")
        except Exception as exc:
            QMessageBox.critical(self, "Analiz yapılamadı", str(exc))
        finally: QApplication.restoreOverrideCursor()

    def _set_references(self, label: str) -> None:
        self.reference_selector.blockSignals(True); self.reference_selector.clear(); self.current_comparisons = []
        references = self.gold.get(label.upper(), [])
        for index, row in enumerate(references, 1):
            evidence = compare_audio(self.current_query_path, Path(row["audio_path"])) if self.current_query_path else {}
            combined = evidence.get("combined_similarity", 0)
            self.current_comparisons.append({**row, **evidence})
            self.reference_selector.addItem(f"Referans {index} — yardımcı benzerlik %{combined:.1f}", index - 1)
        self.reference_selector.blockSignals(False)
        if self.current_comparisons:
            best = max(range(len(self.current_comparisons)), key=lambda i: self.current_comparisons[i]["combined_similarity"])
            self.reference_selector.setCurrentIndex(best); self._reference_changed(best)
        else:
            self.reference_deck.set_audio(None); self.similarity.setText("Bu tahmin için doğrulanmış referans bulunamadı")

    def _reference_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.current_comparisons): return
        row = self.current_comparisons[index]; label = str(row["aircraft_type"])
        self.reference_deck.set_audio(Path(row["audio_path"]), f"DOĞRULANMIŞ ETİKET: {label} • Referans {index + 1}")
        self.similarity.setText(
            f"SEÇİLİ REFERANS KANITI\nSpektral profil: %{row['spectral_similarity']:.1f}\n"
            f"Akustik öznitelik: %{row['feature_similarity']:.1f}\n"
            "Not: Bunlar yardımcı benzerliktir; doğruluk olasılığı değildir."
        )

    def _reveal_source_truth(self) -> None:
        if not self.current_query_path: return
        row = self._test_metadata(self.current_query_path) or {}
        truth = str(row.get("aircraft_type", "BİLİNMİYOR"))
        predicted = str((self.current_prediction or {}).get("predicted_subtype", ""))
        result = "UYUŞUYOR" if truth == predicted else "UYUŞMUYOR"
        self.source_truth.setText(f"ADS-B kaynak etiketi: {truth} — Agent ile {result}")

    def _ensure_pending(self, path: Path, label: str) -> dict | None:
        digest = sha256_file(path); decisions = latest_decisions(self.decisions_path)
        for row in read_jsonl(self.queue_path):
            if row.get("sha256") != digest: continue
            previous = decisions.get(str(row.get("intake_id")))
            if previous:
                self.statusBar().showMessage(
                    f"BU SES DAHA ÖNCE İŞLENDİ: {previous.get('review_status')} — başka bir TEST_SESLERI kaydı seçin"
                )
                return None
            return row
        metadata = self._test_metadata(path)
        if not metadata: raise ValueError("Bu dosya test manifestinde yok; kaynak ve lisans doğrulanamadı")
        return stage_reference(
            path, aircraft_type=label, icao_type=metadata["icao_type"],
            physical_airframe_id=metadata["physical_airframe_id"], source_uri=metadata["source_uri"],
            license_name=metadata["license"], inbox=self.inbox_path, queue=self.queue_path,
        )

    def _decide(self, approved: bool) -> None:
        if not self.current_record:
            QMessageBox.warning(self, "Karar verilemedi", "Önce bir test sesi seçin."); return
        try:
            decision = append_review_decision(
                self.current_record, approved=approved, reviewer=self.reviewer.text(),
                subtype=self.approved_type.text(), note=self.note.text(), decisions_path=self.decisions_path,
            )
            if approved:
                report = build_human_verified_index(self.decisions_path, self.database_path); self._show_proof(report)
                self.statusBar().showMessage(
                    f"ONAYLANDI → Shazam indeksine işlendi: {Path(decision['decision_artifact_path']).name}"
                )
            else:
                self.statusBar().showMessage(
                    f"REDDEDİLDİ → {Path(decision['decision_artifact_path']).name}"
                )
            self.current_record = None; self._refresh_counts()
            QTimer.singleShot(80, self._advance_to_next_test)
        except Exception as exc: QMessageBox.critical(self, "Karar kaydedilemedi", str(exc))

    def _uncertain(self) -> None:
        if not self.current_record:
            QMessageBox.warning(self, "Karar verilemedi", "Önce bir test sesi seçin."); return
        try:
            decision = append_uncertain_decision(
                self.current_record, reviewer=self.reviewer.text(), subtype=self.approved_type.text(),
                note=self.note.text(), decisions_path=self.decisions_path,
            )
            self.statusBar().showMessage(
                f"EMİN DEĞİL → Shazam'a yüklenmedi: {Path(decision['decision_artifact_path']).name}"
            )
            self.current_record = None; self._refresh_counts()
            QTimer.singleShot(80, self._advance_to_next_test)
        except Exception as exc: QMessageBox.critical(self, "Karar kaydedilemedi", str(exc))

    def _advance_to_next_test(self) -> None:
        """Open the next undecided manifest item after every review decision."""
        records = self.manifest.get("records", [])
        if not records:
            self.statusBar().showMessage("Test manifestinde başka ses yok.")
            return
        decided_hashes = {
            str(row.get("sha256")) for row in latest_decisions(self.decisions_path).values()
            if row.get("sha256")
        }
        current = str(self.current_query_path.resolve()).lower() if self.current_query_path else ""
        current_index = next(
            (index for index, row in enumerate(records)
             if str(Path(row["audio_path"]).resolve()).lower() == current),
            -1,
        )
        ordered = records[current_index + 1:] + records[:current_index + 1]
        for row in ordered:
            candidate = Path(row["audio_path"]).resolve()
            if not candidate.is_file():
                continue
            if sha256_file(candidate) in decided_hashes:
                continue
            self.note.clear()
            self.query_deck.stop(); self.reference_deck.stop()
            self._analyze(candidate)
            return
        self.statusBar().showMessage("TÜM TEST SESLERİ İŞLENDİ — bekleyen karar kalmadı.")

    def _refresh_counts(self) -> None:
        pending = len(pending_intakes(self.queue_path, self.decisions_path)); rows = list(latest_decisions(self.decisions_path).values())
        accepted = sum(r.get("review_status") == "APPROVED" for r in rows); rejected = sum(r.get("review_status") == "REJECTED" for r in rows)
        uncertain = sum(r.get("review_status") == "UNCERTAIN" for r in rows)
        self.counts.setText(f"BEKLEYEN {pending}   |   KABUL {accepted}   |   RED {rejected}   |   EMİN DEĞİL {uncertain}")

    def _show_proof(self, report: dict) -> None:
        hashes = sum(int(r["hash_count"]) for r in report.get("indexed", []))
        database = Path(report["database"])
        self.proof.setText(f"SHAZAM İNDEKSİ: {report['indexed_count']} kabul kaydı • {hashes} fingerprint • {database.name}")
        self.proof.setToolTip(str(database))

    def _load_proof(self) -> None:
        path = self.database_path.with_suffix(self.database_path.suffix + ".manifest.json")
        if path.is_file():
            try: self._show_proof(json.loads(path.read_text(encoding="utf-8")))
            except Exception: pass

    def _inspect(self) -> None:
        data = inspect_database(self.database_path)
        if not data["exists"]: QMessageBox.information(self, "SQLite kanıtı", "Henüz onaylanıp indekslenen ses yok."); return
        tracks = "\n".join(f"• {r['aircraft_type']} | {r['hash_count']} hash | {r['source_path']}" for r in data["tracks"])
        QMessageBox.information(self, "Shazam SQLite Kanıtı", f"Track: {data['track_count']}\nFingerprint: {data['fingerprint_count']}\nKabul klasörü doğrulandı: {data['all_sources_are_accepted']}\n\n{tracks}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.query_deck.stop(); self.reference_deck.stop(); event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path); args, qt_args = parser.parse_known_args()
    kwargs = {}
    if args.workspace:
        p = args.workspace.resolve(); p.mkdir(parents=True, exist_ok=True)
        kwargs = {"queue_path": p/"intake_queue.jsonl", "decisions_path": p/"intake_decisions.jsonl", "database_path": p/"aircraft_test_fingerprints.sqlite3", "inbox_path": p/"BEKLEYEN"}
    app = QApplication([sys.argv[0], *qt_args]); app.setStyle("Fusion")
    window = AircraftReferenceIntakeWindow(**kwargs); window.show(); sys.exit(app.exec())


if __name__ == "__main__": main()
