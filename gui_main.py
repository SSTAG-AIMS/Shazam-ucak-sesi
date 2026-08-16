"""
╔══════════════════════════════════════════════════════════════════╗
║    HAVALIMANL ÇEVRESEL GÜRÜLTÜ TESPİT SİSTEMİ — GUI v3.1       ║
║    Faz 1: Dosya Analizi  |  Faz 2: Canlı Mikrofon               ║
║    Yeni : Annotation Modu + Staging (Review) + Canlı Spektrum    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys, os, csv, time, queue, collections, warnings
from typing import Optional
warnings.filterwarnings("ignore")
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QTabWidget,
    QProgressBar, QStatusBar, QFrame, QSplitter, QSizePolicy,
    QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit,
    QTableWidget, QTableWidgetItem,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt6.QtGui import QFont, QColor, QPainter, QLinearGradient, QBrush, QPen

from collections import deque, Counter

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except ImportError:
    SOUNDDEVICE_OK = False

try:
    import soundfile as sf
    SOUNDFILE_OK = True
except ImportError:
    SOUNDFILE_OK = False

try:
    from svantek_hid import (
        SvantekWorker, find_svantek_gui_entry, hid_available,
        THIRD_OCT_FREQS,
    )
    SVANTEK_OK = True
    _SVANTEK_ERR = ""
except ImportError as _e:
    SVANTEK_OK = False
    _SVANTEK_ERR = str(_e)
    THIRD_OCT_FREQS = []

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    # Hibrit alt tür akışı: önce Shazam parmak izi; eşleşme yoksa BEATs.
    from noise_detector_category_fp_v2 import (
        CategoryFingerprintV2System as AirportNoiseSystem,
    )
    DETECTOR_OK = True; _DETECTOR_ERR = ""
except ImportError as e:
    DETECTOR_OK = False; _DETECTOR_ERR = str(e)

try:
    from mic_map import MapTab
    MICMAP_OK = True
except ImportError as e:
    MICMAP_OK = False
    print(f"[mic_map] Harita sekmesi yüklenemedi: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  RENK PALETİ & STİL
# ══════════════════════════════════════════════════════════════════════════

PALETTE = {
    "bg":      "#0D1117", "surface": "#161B22", "border":  "#30363D",
    "text":    "#E6EDF3", "muted":   "#8B949E", "accent":  "#00D4FF",
    "accent2": "#FF6B35", "green":   "#7EE8A2", "yellow":  "#FFE66D",
    "red":     "#FF4444",
}

CLASS_COLORS = {
    "AIRCRAFT": "#FF6B35", "AMBIENT":  "#7EE8A2", "SPEECH":   "#A8DADC",
    "TRAFFIC":  "#FFE66D", "WIND":     "#4ECDC4", "UNKNOWN":  "#6C757D",
    "OTHER":    "#9E9E9E",
}

DARK_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {PALETTE['bg']}; color: {PALETTE['text']};
    font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;
}}
QTabWidget::pane {{
    border: 1px solid {PALETTE['border']}; background: {PALETTE['surface']}; border-radius: 4px;
}}
QTabBar::tab {{
    background: {PALETTE['bg']}; color: {PALETTE['muted']};
    padding: 8px 20px; border: 1px solid {PALETTE['border']};
    border-bottom: none; margin-right: 2px; border-radius: 4px 4px 0 0;
}}
QTabBar::tab:selected {{
    background: {PALETTE['surface']}; color: {PALETTE['accent']};
    border-bottom: 2px solid {PALETTE['accent']};
}}
QPushButton {{
    background-color: {PALETTE['surface']}; color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']}; padding: 7px 18px;
    border-radius: 6px; font-weight: 500;
}}
QPushButton:hover {{ background-color: #21262D; border-color: {PALETTE['accent']}; color: {PALETTE['accent']}; }}
QPushButton:disabled {{ color: {PALETTE['muted']}; border-color: {PALETTE['border']}; background-color: {PALETTE['surface']}; }}
QPushButton#analyzeBtn, QPushButton#startMicBtn {{
    background-color: #1a4a2e; border-color: {PALETTE['green']};
    color: {PALETTE['green']}; font-weight: 700;
}}
QPushButton#analyzeBtn:hover, QPushButton#startMicBtn:hover {{ background-color: #2a6a3e; }}
QPushButton#analyzeBtn:disabled, QPushButton#startMicBtn:disabled {{
    background-color: {PALETTE['surface']}; border-color: {PALETTE['border']}; color: {PALETTE['muted']};
}}
QPushButton#stopMicBtn {{
    background-color: #4a1a1a; border-color: {PALETTE['red']}; color: {PALETTE['red']}; font-weight: 700;
}}
QPushButton#stopMicBtn:hover {{ background-color: #6a2a2a; }}
QPushButton#recBtn {{ border-color: {PALETTE['accent2']}; color: {PALETTE['accent2']}; }}
QComboBox {{
    background-color: {PALETTE['surface']}; color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']}; padding: 6px 12px;
    border-radius: 6px; min-width: 130px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: {PALETTE['surface']}; color: {PALETTE['text']};
    selection-background-color: #21262D; border: 1px solid {PALETTE['border']};
}}
QProgressBar {{
    background-color: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
    border-radius: 4px; height: 6px;
}}
QProgressBar::chunk {{ background-color: {PALETTE['accent']}; border-radius: 4px; }}
QStatusBar {{
    background-color: {PALETTE['surface']}; color: {PALETTE['muted']};
    border-top: 1px solid {PALETTE['border']};
}}
QFrame#sidePanel {{
    background-color: {PALETTE['surface']}; border: 1px solid {PALETTE['border']}; border-radius: 8px;
}}
QLabel#statValue {{ color: {PALETTE['accent']}; font-size: 15px; font-weight: 700; }}
QLabel#statLabel {{ color: {PALETTE['muted']}; font-size: 11px; }}
"""


# ══════════════════════════════════════════════════════════════════════════
#  YARDIMCILAR
# ══════════════════════════════════════════════════════════════════════════

def _make_canvas(fig_h=4.0, fig_w=10.0):
    fig = Figure(figsize=(fig_w, fig_h), facecolor=PALETTE["bg"])
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return fig, canvas

def _style_ax(ax):
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
    ax.xaxis.label.set_color(PALETTE["muted"]); ax.yaxis.label.set_color(PALETTE["muted"])
    ax.title.set_color(PALETTE["text"])
    ax.grid(True, color=PALETTE["border"], linewidth=0.5, alpha=0.6)

def _hline():
    f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {PALETTE['border']};"); return f

def _label(text, muted=False, small=False):
    lbl = QLabel(text)
    color = PALETTE["muted"] if muted else PALETTE["text"]
    size  = "10px" if small else "13px"
    lbl.setStyleSheet(f"color: {color}; font-size: {size};")
    return lbl

def fmt_elapsed(secs):
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════
#  ANNOTATION DİALOGU
# ══════════════════════════════════════════════════════════════════════════

class AnnotationDialog(QDialog):
    CLASSES = ["AIRCRAFT", "AMBIENT", "OTHER", "SPEECH", "TRAFFIC", "WIND"]

    def __init__(self, audio_path, window_start, original_label, parent=None):
        super().__init__(parent)
        self.setWindowTitle("✏  Annotation — Etiket Düzeltme")
        self.setMinimumWidth(400)
        self.setStyleSheet(f"""
            QDialog, QWidget {{
                background: {PALETTE['surface']}; color: {PALETTE['text']}; font-size: 13px;
            }}
            QLineEdit, QComboBox {{
                background: {PALETTE['bg']}; color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']}; padding: 6px; border-radius: 4px;
            }}
            QLabel {{ color: {PALETTE['text']}; }}
            QPushButton {{
                background: {PALETTE['bg']}; color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']}; padding: 6px 16px; border-radius: 4px;
            }}
        """)
        self._audio_path   = audio_path
        self._window_start = window_start
        self._original     = original_label

        lay = QVBoxLayout(self); lay.setSpacing(12)

        orig_color = CLASS_COLORS.get(original_label, "#aaa")
        info = QLabel(
            f"<b>Dosya:</b> {os.path.basename(audio_path)}<br>"
            f"<b>Pencere:</b> {window_start:.1f} s<br>"
            f"<b>Model tahmini:</b> "
            f"<span style='color:{orig_color};font-weight:bold'>{original_label}</span>"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet(f"""
            background: {PALETTE['bg']}; padding: 10px;
            border: 1px solid {PALETTE['border']}; border-radius: 6px;
            color: {PALETTE['muted']}; font-size: 11px;
        """)
        lay.addWidget(info)

        form = QFormLayout(); form.setSpacing(10)

        self.label_combo = QComboBox()
        for cls in self.CLASSES:
            self.label_combo.addItem(cls)
        if original_label in self.CLASSES:
            self.label_combo.setCurrentText(original_label)
        form.addRow("✔  Doğru Etiket:", self.label_combo)

        self.conf_combo = QComboBox()
        for c in ["Kesin", "Muhtemelen", "Emin değilim"]:
            self.conf_combo.addItem(c)
        form.addRow("Güven:", self.conf_combo)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Opsiyonel not…")
        form.addRow("Not:", self.note_edit)

        lay.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def annotation_row(self):
        return {
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_file":      self._audio_path,
            "window_start_s":  f"{self._window_start:.2f}",
            "original_label":  self._original,
            "corrected_label": self.label_combo.currentText(),
            "confidence":      self.conf_combo.currentText(),
            "note":            self.note_edit.text().strip(),
        }


# ══════════════════════════════════════════════════════════════════════════
#  CANLI MİKROFON ETİKETLEME DİALOGU
# ══════════════════════════════════════════════════════════════════════════

class LiveLabelDialog(QDialog):
    """
    'Etiketle & Gönder' butonuna basıldığında açılır.
    Modelin anlık tahminini gösterir; kullanıcı kendi etiketini seçerek
    klibini pending havuzuna ekler.
    """
    CLASSES = ["AIRCRAFT", "AMBIENT", "OTHER", "SPEECH", "TRAFFIC", "WIND"]

    def __init__(self, model_label: str, elapsed: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🏷  Canlı Etiketleme")
        self.setMinimumWidth(420)
        self.setStyleSheet(f"""
            QDialog, QWidget {{
                background: {PALETTE['surface']}; color: {PALETTE['text']}; font-size: 13px;
            }}
            QLineEdit, QComboBox {{
                background: {PALETTE['bg']}; color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']}; padding: 6px; border-radius: 4px;
            }}
            QLabel {{ color: {PALETTE['text']}; }}
            QPushButton {{
                background: {PALETTE['bg']}; color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']}; padding: 6px 16px; border-radius: 4px;
            }}
        """)
        self._model_label = model_label
        self._elapsed     = elapsed

        lay = QVBoxLayout(self); lay.setSpacing(14)

        # ── Model tahmini bilgi kartı ──────────────────────────────────
        model_color = CLASS_COLORS.get(model_label, "#aaa")
        info = QLabel(
            f"<b>⏱ Süre:</b> {elapsed:.1f} s<br>"
            f"<b>🤖 Model Tahmini:</b> "
            f"<span style='color:{model_color}; font-size:16px; font-weight:bold'>"
            f"{model_label}</span>"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet(
            f"background: {PALETTE['bg']}; padding: 12px;"
            f"border: 1px solid {PALETTE['border']}; border-radius: 6px;"
            f"color: {PALETTE['muted']}; font-size: 11px;"
        )
        lay.addWidget(info)

        # ── Ayırıcı ───────────────────────────────────────────────────
        sep = QLabel("─── Senin değerlendirmen ───")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px;")
        lay.addWidget(sep)

        # ── Kullanıcı etiketi ──────────────────────────────────────────
        form = QFormLayout(); form.setSpacing(10)

        self.label_combo = QComboBox()
        for cls in self.CLASSES:
            self.label_combo.addItem(cls)
        # Varsayılan: modelin tahmini — kullanıcı değiştirmezse onaylıyor demek
        if model_label in self.CLASSES:
            self.label_combo.setCurrentText(model_label)
        form.addRow("✔  Doğru Etiket:", self.label_combo)

        self.conf_combo = QComboBox()
        for c in ["Kesin", "Muhtemelen", "Emin değilim"]:
            self.conf_combo.addItem(c)
        form.addRow("Güven:", self.conf_combo)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Opsiyonel not… (ör: arka planda trafik vardı)")
        form.addRow("Not:", self.note_edit)

        lay.addLayout(form)

        # ── Uyarı: model ≠ kullanıcı ──────────────────────────────────
        self._mismatch_lbl = QLabel("")
        self._mismatch_lbl.setStyleSheet(
            f"color: {PALETTE['yellow']}; font-size: 10px;"
        )
        lay.addWidget(self._mismatch_lbl)
        self.label_combo.currentTextChanged.connect(self._check_mismatch)
        self._check_mismatch(self.label_combo.currentText())

        # ── Butonlar ──────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Save).setText("📥  Gönder")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _check_mismatch(self, selected: str):
        if selected != self._model_label:
            self._mismatch_lbl.setText(
                f"⚠  Modelin tahmini ({self._model_label}) ile farklı — "
                "düzeltme kaydedilecek."
            )
        else:
            self._mismatch_lbl.setText("✓  Modelin tahmini onaylanıyor.")

    def result_row(self) -> dict:
        return {
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_file":      "LIVE_MIC_STREAM",
            "window_start_s":  "0.00",
            "original_label":  self._model_label,
            "corrected_label": self.label_combo.currentText(),
            "confidence":      self.conf_combo.currentText(),
            "note":            self.note_edit.text().strip(),
        }


# ══════════════════════════════════════════════════════════════════════════
#  PENDING KLİP YÖNETİCİSİ — Staging sistemi
# ══════════════════════════════════════════════════════════════════════════

class PendingClipManager:
    """
    Annotation edilen pencereleri 5s WAV klip olarak önce 'pending' havuzuna kaydeder.
    Kullanıcı DataReviewTab'dan onayladıkça approved/ klasörüne taşır.

    Klasör yapısı:
        live_clips/
          pending/<CLASS>/
          approved/<CLASS>/   ← dataset_builder burayı okur
          rejected/
          pending_manifest.csv
          approved_manifest.csv
    """
    MANIFEST_FIELDS = [
        "clip_id", "timestamp", "source_file", "window_start_s",
        "original_label", "corrected_label", "confidence", "note",
        "clip_path", "status",
    ]

    def __init__(self, base_dir: str = "live_clips"):
        self.base_dir     = base_dir
        self.pending_dir  = os.path.join(base_dir, "pending")
        self.approved_dir = os.path.join(base_dir, "approved")
        self.rejected_dir = os.path.join(base_dir, "rejected")
        self.pending_manifest  = os.path.join(base_dir, "pending_manifest.csv")
        self.approved_manifest = os.path.join(base_dir, "approved_manifest.csv")
        for d in [self.pending_dir, self.approved_dir, self.rejected_dir]:
            os.makedirs(d, exist_ok=True)

    def save_pending_clip(self, samples: np.ndarray, sr: int,
                          window_start_s: float, ann_row: dict) -> tuple:
        start    = int(window_start_s * sr)
        end      = start + int(5.0 * sr)
        clip     = samples[start: min(end, len(samples))].copy()
        if len(clip) < int(5.0 * sr):
            clip = np.pad(clip, (0, int(5.0 * sr) - len(clip)))

        cls     = ann_row["corrected_label"]
        clip_id = f"{int(time.time() * 1000)}"
        cls_dir = os.path.join(self.pending_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        fname   = f"{clip_id}_{cls}.wav"
        path    = os.path.join(cls_dir, fname)

        if SOUNDFILE_OK:
            sf.write(path, clip.astype(np.float32), sr)
        else:
            import wave as _wv, struct as _st
            pcm = (np.clip(clip, -1, 1) * 32767).astype(np.int16)
            with _wv.open(path, "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(_st.pack(f"<{len(pcm)}h", *pcm))

        row = {
            "clip_id":         clip_id,
            "timestamp":       ann_row["timestamp"],
            "source_file":     ann_row["audio_file"],
            "window_start_s":  ann_row["window_start_s"],
            "original_label":  ann_row["original_label"],
            "corrected_label": cls,
            "confidence":      ann_row["confidence"],
            "note":            ann_row["note"],
            "clip_path":       path,
            "status":          "pending",
        }
        self._append_manifest(self.pending_manifest, row)
        return path, clip_id

    def load_pending(self) -> list:
        return [r for r in self._read_manifest(self.pending_manifest)
                if r.get("status") == "pending"]

    def load_approved(self) -> list:
        return [r for r in self._read_manifest(self.approved_manifest)
                if r.get("status") == "approved"]

    def delete_approved(self, clip_id: str, delete_file: bool = True) -> bool:
        """Onaylı klip kaydını manifest'ten siler; delete_file=True ise WAV'ı da kaldırır."""
        rows  = self._read_manifest(self.approved_manifest)
        found = next((r for r in rows if r["clip_id"] == clip_id), None)
        if not found:
            return False
        if delete_file:
            path = found.get("clip_path", "")
            try:
                if os.path.exists(path):
                    os.remove(path)
                    cls_dir = os.path.dirname(path)
                    if os.path.isdir(cls_dir) and not os.listdir(cls_dir):
                        os.rmdir(cls_dir)
            except Exception as e:
                print(f"[PendingClipManager] Dosya silinemedi: {e}")
        remaining = [r for r in rows if r["clip_id"] != clip_id]
        self._write_manifest(self.approved_manifest, remaining)
        return True

    def approve_clip(self, clip_id: str) -> bool:
        rows  = self._read_manifest(self.pending_manifest)
        found = next((r for r in rows if r["clip_id"] == clip_id), None)
        if not found: return False
        cls     = found["corrected_label"]
        dst_dir = os.path.join(self.approved_dir, cls)
        os.makedirs(dst_dir, exist_ok=True)
        src = found["clip_path"]
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            if os.path.exists(src):
                import shutil; shutil.move(src, dst)
            found["clip_path"] = dst
            found["status"]    = "approved"
        except Exception as e:
            print(f"[PendingClipManager] Taşıma hatası: {e}"); return False
        self._write_manifest(self.pending_manifest, rows)
        self._append_manifest(self.approved_manifest, found)
        return True

    def reject_clip(self, clip_id: str) -> bool:
        rows  = self._read_manifest(self.pending_manifest)
        found = next((r for r in rows if r["clip_id"] == clip_id), None)
        if not found: return False
        src = found["clip_path"]
        try:
            if os.path.exists(src):
                import shutil
                shutil.move(src, os.path.join(self.rejected_dir, os.path.basename(src)))
            found["status"] = "rejected"
        except Exception as e:
            print(f"[PendingClipManager] Silme hatası: {e}")
        self._write_manifest(self.pending_manifest, rows)
        return True

    def _read_manifest(self, path: str) -> list:
        if not os.path.exists(path): return []
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _write_manifest(self, path: str, rows: list):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.MANIFEST_FIELDS)
            w.writeheader(); w.writerows(rows)

    def _append_manifest(self, path: str, row: dict):
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.MANIFEST_FIELDS)
            if not exists: w.writeheader()
            w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 1 — YAN PANEL
# ══════════════════════════════════════════════════════════════════════════

class SidePanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("sidePanel"); self.setFixedWidth(250)
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        title = QLabel("✈  GÜRÜLTÜ ANALİZİ")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {PALETTE['accent']}; letter-spacing: 1px;")
        lay.addWidget(title); lay.addWidget(_hline())

        self.stat_widgets = {}
        for key, lbl_text, default in [
            ("duration","SÜRE","--"), ("model","MODEL","--"),
            ("n_windows","PENCERE","--"), ("dominant","DOMAİNANT","--"),
            ("peak_db","PEAK dBFS","--"),
        ]:
            card = QWidget()
            card.setObjectName("statCard")
            cl = QVBoxLayout(card); cl.setContentsMargins(8,8,8,8); cl.setSpacing(2)
            card.setStyleSheet(
                f"QWidget#statCard {{ background: {PALETTE['bg']}; "
                f"border: 1px solid {PALETTE['border']}; border-radius: 6px; }}"
            )
            lbl = QLabel(lbl_text); lbl.setObjectName("statLabel")
            val = QLabel(default); val.setObjectName("statValue")
            val.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            val.setStyleSheet(f"color: {PALETTE['accent']};")
            cl.addWidget(lbl); cl.addWidget(val)
            lay.addWidget(card)
            self.stat_widgets[key] = val

        aircraft_card = QWidget()
        aircraft_card.setObjectName("aircraftCard")
        aircraft_card.setFixedHeight(118)
        aircraft_layout = QVBoxLayout(aircraft_card)
        aircraft_layout.setContentsMargins(8,8,8,8)
        aircraft_layout.setSpacing(2)
        aircraft_card.setStyleSheet(
            f"QWidget#aircraftCard {{ background: {PALETTE['bg']}; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 6px; }}"
        )
        aircraft_label = QLabel("ALT TÜR")
        aircraft_label.setObjectName("statLabel")
        aircraft_value = QLabel("--")
        aircraft_value.setObjectName("subtypeValue")
        aircraft_value.setStyleSheet(
            f"color: {PALETTE['accent']}; font-size: 12px; font-weight: 700;"
        )
        aircraft_value.setTextFormat(Qt.TextFormat.PlainText)
        aircraft_value.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        aircraft_value.setWordWrap(False)
        aircraft_value.setFixedHeight(24)
        aircraft_detail = QLabel("")
        aircraft_detail.setObjectName("subtypeDetail")
        aircraft_detail.setStyleSheet(
            f"color: {PALETTE['muted']}; font-size: 9px; font-weight: 600;"
        )
        aircraft_detail.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        aircraft_detail.setWordWrap(True)
        aircraft_detail.setFixedHeight(42)
        aircraft_layout.addWidget(aircraft_label)
        aircraft_layout.addWidget(aircraft_value)
        aircraft_layout.addWidget(aircraft_detail)
        aircraft_detail.hide()
        lay.addWidget(aircraft_card)
        self.stat_widgets["aircraft_type"] = aircraft_value
        self.aircraft_detail = aircraft_detail

        lay.addWidget(_hline())
        sec = QLabel("SINIF DAĞILIMI")
        sec.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
        lay.addWidget(sec)

        self.class_bars = {}
        for cls, color in CLASS_COLORS.items():
            if cls == "UNKNOWN": continue
            row = QWidget(); rl = QHBoxLayout(row)
            rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
            name = QLabel(cls); name.setFixedWidth(64)
            name.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")
            bar = QProgressBar(); bar.setRange(0,100); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(6)
            bar.setStyleSheet(f"QProgressBar {{ background: {PALETTE['border']}; border-radius: 3px; border: none; }} QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}")
            pct = QLabel("0%"); pct.setFixedWidth(34)
            pct.setAlignment(Qt.AlignmentFlag.AlignRight)
            pct.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
            rl.addWidget(name); rl.addWidget(bar); rl.addWidget(pct)
            lay.addWidget(row)
            self.class_bars[cls] = (bar, pct)

        lay.addStretch()
        self.export_btn = QPushButton("💾  Export (CSV + PNG)")
        self.export_btn.setEnabled(False)
        lay.addWidget(self.export_btn)

    def _set_subtype_display(self, title, detail=""):
        """Alt tür adını ve sonucu üreten yöntemi ayrı satırlarda göster."""
        title_text = str(title or "--")
        detail_text = str(detail or "")
        value = self.stat_widgets["aircraft_type"]
        value.setText(title_text)
        value.setToolTip(
            title_text + (f" | {detail_text}" if detail_text else "")
        )
        self.aircraft_detail.setText(detail_text)
        self.aircraft_detail.setToolTip(detail_text)
        self.aircraft_detail.setVisible(bool(detail_text))

    def update_stats(self, r):
        dur = r["duration"]; m, s = divmod(int(dur), 60)
        self.stat_widgets["duration"].setText(f"{m}:{s:02d}")
        self.stat_widgets["model"].setText(r["model_used"])
        self.stat_widgets["model"].setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.stat_widgets["n_windows"].setText(str(len(r["frame_labels"])))
        summary = r["summary"]
        dominant = r.get("dominant") or (max(summary, key=summary.get) if summary else "--")
        self.stat_widgets["dominant"].setText(dominant)
        self.stat_widgets["dominant"].setStyleSheet(f"color: {CLASS_COLORS.get(dominant, PALETTE['accent'])};")
        aircraft_match = r.get("aircraft_match")
        subtype_match = r.get("subtype_match")
        if aircraft_match:
            aircraft_type = aircraft_match.get("aircraft_type", "UNKNOWN_AIRCRAFT")
            confidence = float(aircraft_match.get("confidence", 0.0)) * 100.0
            predicted = aircraft_match.get("predicted_type", aircraft_type)
            accepted = bool(aircraft_match.get("accepted", True))
            n_votes = int(aircraft_match.get("n_windows", 0))
            won_votes = int(
                aircraft_match.get("vote_counts", {}).get(predicted, 0)
            )
            vote_text = f" · {won_votes}/{n_votes} pencere" if n_votes else ""
            if aircraft_match.get("method") == "shazam":
                detail_text = (
                    f"%{confidence:.1f} · Kaynak: Shazam parmak izi"
                )
            elif aircraft_match.get("method") == "beats_multi_window_vote":
                prefix = "" if accepted else f"Aday: {predicted} · "
                detail_text = (
                    f"{prefix}%{confidence:.1f}{vote_text} · "
                    "Kaynak: BEATs uçak alt tür modeli"
                )
            else:
                detail_text = "Kaynak: Alt tür bulunamadı"
            self._set_subtype_display(aircraft_type, detail_text)
            self.aircraft_detail.setText(detail_text)
        elif subtype_match:
            subtype = subtype_match.get("subtype", "UNKNOWN")
            confidence = float(subtype_match.get("confidence", 0.0)) * 100.0
            method = subtype_match.get("method", "")
            predicted = subtype_match.get("predicted_subtype", subtype)
            accepted = bool(subtype_match.get("accepted", True))
            n_votes = int(subtype_match.get("n_windows", 0))
            won_votes = int(
                subtype_match.get("vote_counts", {}).get(predicted, 0)
            )
            if method == "shazam_v1":
                vote_text = ""
                method_text = " · Kaynak: Shazam parmak izi"
            elif method == "beats_multi_window_vote_v2_calibrated":
                vote_text = (
                    f" · {won_votes}/{n_votes} pencere" if n_votes else ""
                )
                method_text = " · Kaynak: BEATs v2 alt tür modeli"
            elif method == "beats_multi_window_vote":
                vote_text = (
                    f" · {won_votes}/{n_votes} pencere" if n_votes else ""
                )
                method_text = " · Kaynak: BEATs alt tür modeli"
            else:
                vote_text = ""
                method_text = " · Kaynak: Bilinmiyor"
            if accepted:
                title_text = subtype
                detail_text = f"%{confidence:.1f}{vote_text}{method_text}"
            else:
                title_text = subtype
                compact_vote = (
                    f" · {won_votes}/{n_votes}" if n_votes else ""
                )
                detail_text = (
                    f"{predicted} · %{confidence:.1f}"
                    f"{compact_vote}{method_text}"
                )
            self._set_subtype_display(title_text, detail_text)
            self.aircraft_detail.setText(detail_text)
            self.aircraft_detail.setToolTip(
                f"{title_text} | {detail_text}"
            )
        else:
            self._set_subtype_display("--")
            self.aircraft_detail.setText("")
            self.aircraft_detail.setToolTip("")
        db = r["db_values"]
        if len(db): self.stat_widgets["peak_db"].setText(f"{np.max(db):.1f}")
        for cls, (bar, pct) in self.class_bars.items():
            val = summary.get(cls, 0.0); bar.setValue(int(val)); pct.setText(f"{val:.0f}%")
        self.export_btn.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 1 — SINIFLANDIRMA SEKMESİ
# ══════════════════════════════════════════════════════════════════════════

class ClassificationTab(QWidget):
    # Yeni klip pending'e eklendiğinde DataReviewTab'ı tetikler
    clip_added = pyqtSignal()

    def __init__(self, clip_manager=None):
        super().__init__()
        self._result      = None
        self._mgr         = clip_manager
        self._annotations: list[dict] = []
        # Marker'ları ayrı saklıyoruz; _draw_strip fig.clear() sonrası yeniden uygular
        self._annotation_markers: list[dict] = []

        lay = QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)
        self.fig1, self.canvas1 = _make_canvas(fig_h=1.6)
        self.fig2, self.canvas2 = _make_canvas(fig_h=2.8)
        self.fig3, self.canvas3 = _make_canvas(fig_h=3.2)
        for c in [self.canvas1, self.canvas2, self.canvas3]:
            lay.addWidget(c)

        # Alt kontrol satırı
        bottom = QWidget(); bl = QHBoxLayout(bottom)
        bl.setContentsMargins(8,0,8,0); bl.setSpacing(12)

        for cls, color in CLASS_COLORS.items():
            if cls == "UNKNOWN": continue
            d = QLabel(f"● {cls}")
            d.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")
            bl.addWidget(d)

        bl.addStretch()

        self.ann_btn = QPushButton("✏  Annotation: Kapalı")
        self.ann_btn.setCheckable(True)
        self.ann_btn.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid {PALETTE['border']}; color: {PALETTE['muted']};
                padding: 4px 12px; border-radius: 5px; font-size: 11px;
            }}
            QPushButton:checked {{
                border-color: {PALETTE['yellow']}; color: {PALETTE['yellow']};
                background: #2a2600;
            }}
        """)
        self.ann_btn.toggled.connect(self._on_ann_toggle)
        bl.addWidget(self.ann_btn)

        self.ann_count_lbl = QLabel("0 kayıt")
        self.ann_count_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        bl.addWidget(self.ann_count_lbl)

        self.ann_save_btn = QPushButton("💾  Kaydet")
        self.ann_save_btn.setEnabled(False)
        self.ann_save_btn.setStyleSheet("font-size: 11px; padding: 4px 12px;")
        self.ann_save_btn.clicked.connect(self._save_annotations)
        bl.addWidget(self.ann_save_btn)

        lay.addWidget(bottom)

        self.canvas1.mpl_connect("button_press_event", self._on_strip_click)
        self._ann_mode = False

    # ─────────────────────────────────────────────────────────────────────

    def render(self, r):
        self._result = r
        # Yeni dosya yüklenince eski marker'ları temizle
        self._annotation_markers.clear()
        self._draw_strip(r); self._draw_db(r); self._draw_confidence(r)

    def _on_ann_toggle(self, checked):
        self._ann_mode = checked
        if checked:
            self.ann_btn.setText("✏  Annotation: Açık  ← şeride tıkla")
            self.canvas1.setStyleSheet(f"border: 2px solid {PALETTE['yellow']};")
        else:
            self.ann_btn.setText("✏  Annotation: Kapalı")
            self.canvas1.setStyleSheet("")

    def _on_strip_click(self, event):
        if not self._ann_mode or not self._result: return
        if event.inaxes is None or event.xdata is None: return

        ft = self._result["frame_times"]; fl = self._result["frame_labels"]
        if ft is None or len(ft) == 0: return

        hop = ft[1] - ft[0] if len(ft) > 1 else 2.5
        clicked_t = event.xdata
        win_idx = None
        for i, t in enumerate(ft):
            if t <= clicked_t < t + hop:
                win_idx = i; break
        if win_idx is None: return

        dlg = AnnotationDialog(
            audio_path     = self._result.get("audio_path", ""),
            window_start   = float(ft[win_idx]),
            original_label = fl[win_idx],
            parent         = self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted: return

        row = dlg.annotation_row()
        self._annotations.append(row)

        # Marker'ı ayrı listede sakla → _draw_strip yeniden çizince de kalır
        self._annotation_markers.append({
            "win_idx":   win_idx,
            "corrected": row["corrected_label"],
            "hop":       hop,
        })

        n = len(self._annotations)
        self.ann_save_btn.setEnabled(True)

        # Anlık görsel işaret
        self._mark_window(win_idx, row["corrected_label"], hop)

        # Staging: 5s klip → live_clips/pending/
        if self._mgr is not None:
            try:
                samples = self._result.get("samples")
                sr      = self._result.get("sr", 22050)
                if samples is not None:
                    self._mgr.save_pending_clip(samples, sr, float(ft[win_idx]), row)
                    self.ann_count_lbl.setText(f"{n} kayıt  |  📥 pending'e eklendi")
                    self.clip_added.emit()
                else:
                    self.ann_count_lbl.setText(f"{n} kayıt")
            except Exception as e:
                print(f"[Annotation] Klip kaydedilemedi: {e}")
                self.ann_count_lbl.setText(f"{n} kayıt")
        else:
            self.ann_count_lbl.setText(f"{n} kayıt")

    def _apply_marker_to_ax(self, ax, m: dict):
        """Saklı marker bilgisini mevcut ax'a uygula (canvas.draw çağırmaz)."""
        if self._result is None: return
        ft = self._result["frame_times"]
        if m["win_idx"] >= len(ft): return
        t   = ft[m["win_idx"]]
        hop = m["hop"]
        ax.add_patch(plt.Rectangle(
            (t, 0), hop * 0.98, 1.0,
            fill=False, edgecolor="white", lw=2, zorder=10
        ))
        ax.text(t + hop / 2, 0.5, f"→{m['corrected']}", color="white",
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                zorder=11, transform=ax.get_xaxis_transform())

    def _mark_window(self, win_idx, corrected, hop):
        """Şerit ax'ına görsel işaret ekle; layout engine tetiklenmez → şerit küçülmez."""
        if not self.fig1.axes: return
        ax = self.fig1.axes[0]
        self._apply_marker_to_ax(ax, {"win_idx": win_idx, "corrected": corrected, "hop": hop})
        self.canvas1.draw_idle()   # tight_layout çalıştırmaz

    def _save_annotations(self):
        if not self._annotations: return
        path, _ = QFileDialog.getSaveFileName(
            self, "Annotation CSV Kaydet", "annotations.csv",
            "CSV Dosyaları (*.csv)")
        if not path: return
        fields = ["timestamp","audio_file","window_start_s",
                  "original_label","corrected_label","confidence","note"]
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists: w.writeheader()
            w.writerows(self._annotations)
        n = len(self._annotations)
        self._annotations.clear()
        self.ann_count_lbl.setText("0 kayıt"); self.ann_save_btn.setEnabled(False)
        QMessageBox.information(self, "Kaydedildi",
            f"{n} annotation kaydedildi:\n{path}\n\n"
            "Klipleri onaylamak için '📋 Veri Review' sekmesini kullanın.")

    # ─────────────────────────────────────────────────────────────────────

    def _draw_strip(self, r):
        self.fig1.clear()
        ax = self.fig1.add_subplot(111)
        ax.set_facecolor(PALETTE["bg"]); self.fig1.patch.set_facecolor(PALETTE["bg"])
        fl, ft, dur = r["frame_labels"], r["frame_times"], r["duration"]
        if fl is None or len(fl) == 0:
            ax.text(0.5,0.5,"Veri yok",color=PALETTE["muted"],ha="center",va="center",transform=ax.transAxes)
            self.canvas1.draw(); return
        hop = ft[1]-ft[0] if len(ft)>1 else 2.5
        for index, (t, lbl) in enumerate(zip(ft, fl)):
            width = (
                float(ft[index + 1] - t)
                if index + 1 < len(ft)
                else max(float(dur - t), float(hop))
            )
            draw_width = width if index + 1 == len(ft) else width * 0.98
            ax.barh(0, draw_width, left=t, height=1.0,
                    color=CLASS_COLORS.get(lbl,"#6C757D"), alpha=0.85, align="edge")
            if width > 1.0:
                ax.text(t+width/2, 0.5, lbl, color="white", ha="center", va="center",
                        fontsize=7, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xlim(0,dur); ax.set_ylim(0,1); ax.set_yticks([])
        ax.set_xlabel("Zaman (s)",color=PALETTE["muted"],fontsize=8)
        ax.set_title("Sınıflandırma Şeridi  —  ✏ butonu ile annotation modu",
                     color=PALETTE["text"],fontsize=9,pad=4)
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
        ax.tick_params(colors=PALETTE["muted"],labelsize=8)

        # Layout engine'i bir kez çalıştır, sonra dondur → marker draw_idle'da şeridi küçültmez
        self.fig1.tight_layout(pad=0.4)
        try:
            self.fig1.set_layout_engine("none")
        except Exception:
            pass

        # Önceki annotation marker'larını yeniden uygula
        for m in self._annotation_markers:
            self._apply_marker_to_ax(ax, m)

        self.canvas1.draw()

    def _draw_db(self, r):
        self.fig2.clear(); ax = self.fig2.add_subplot(111); _style_ax(ax)
        self.fig2.patch.set_facecolor(PALETTE["bg"])
        db_t, db_v = r["db_times"], r["db_values"]
        fl, ft, dur = r["frame_labels"], r["frame_times"], r["duration"]
        hop = ft[1]-ft[0] if len(ft)>1 else 2.5
        if len(db_v):
            ymin, ymax = np.min(db_v)-5, np.max(db_v)+5
            for index, (t, lbl) in enumerate(zip(ft, fl)):
                end = (
                    float(ft[index + 1])
                    if index + 1 < len(ft) else float(dur)
                )
                ax.axvspan(t, end, alpha=0.12,
                           color=CLASS_COLORS.get(lbl,"#6C757D"), lw=0)
            ax.plot(db_t, db_v, color=PALETTE["accent2"], lw=1.2, zorder=5)
            ax.fill_between(db_t, db_v, ymin, color=PALETTE["accent2"], alpha=0.12, zorder=4)
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(0,dur)
        ax.set_xlabel("Zaman (s)",fontsize=8); ax.set_ylabel("dBFS",fontsize=8)
        ax.set_title("Ses Seviyesi (dBFS)",color=PALETTE["text"],fontsize=9,pad=4)
        self.fig2.tight_layout(pad=0.4); self.canvas2.draw()

    def _draw_confidence(self, r):
        self.fig3.clear(); ax = self.fig3.add_subplot(111); _style_ax(ax)
        self.fig3.patch.set_facecolor(PALETTE["bg"])
        fp, ft, cn, dur = r["frame_probs"], r["frame_times"], r["class_names"], r["duration"]
        if not fp:
            ax.text(0.5,0.5,"Güven eğrileri EfficientNet veya CNN seçilince görünür.",
                    color=PALETTE["muted"],ha="center",va="center",transform=ax.transAxes,fontsize=10)
            ax.set_title("Softmax Güven Eğrileri",color=PALETTE["text"],fontsize=9,pad=4)
            self.fig3.tight_layout(pad=0.4); self.canvas3.draw(); return
        prob_arr = np.array(fp)
        for i, cls in enumerate(cn):
            if i >= prob_arr.shape[1]: continue
            ax.plot(ft, prob_arr[:,i], color=CLASS_COLORS.get(cls,"#8B949E"),
                    lw=1.5, label=cls, alpha=0.85)
        ax.set_xlim(0,dur); ax.set_ylim(0,1.05)
        ax.set_xlabel("Zaman (s)",fontsize=8); ax.set_ylabel("Softmax Olasılığı",fontsize=8)
        ax.set_title("Softmax Güven Eğrileri",color=PALETTE["text"],fontsize=9,pad=4)
        ax.legend(fontsize=8,facecolor=PALETTE["surface"],labelcolor=PALETTE["text"],
                  loc="upper right",framealpha=0.8)
        ax.axhline(0.5,color=PALETTE["border"],lw=0.7,linestyle="--")
        self.fig3.tight_layout(pad=0.4); self.canvas3.draw()


# ══════════════════════════════════════════════════════════════════════════
#  VERİ REVIEW SEKMESİ — Pending klipleri dinle / onayla / reddet
# ══════════════════════════════════════════════════════════════════════════

class DataReviewTab(QWidget):
    _PENDING_COLS  = ["Kullanıcı Etiketi", "Model Tahmini", "Eşleşme", "Kaynak Dosya", "Güven", "Not"]
    _APPROVED_COLS = ["Kullanıcı Etiketi", "Model Tahmini", "Eşleşme", "Kaynak Dosya", "Güven", "Not"]

    def __init__(self, manager: PendingClipManager):
        super().__init__()
        self._mgr            = manager
        self._pending_rows: list[dict] = []
        self._approved_rows: list[dict] = []
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────
    #  UI KURULUM
    # ─────────────────────────────────────────────────────────────────────

    def _make_table(self, cols: list) -> QTableWidget:
        t = QTableWidget()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setStretchLastSection(True)
        t.setStyleSheet(f"""
            QTableWidget {{
                background: {PALETTE['surface']}; color: {PALETTE['text']};
                gridline-color: {PALETTE['border']}; border: 1px solid {PALETTE['border']};
            }}
            QTableWidget::item:selected {{
                background: #21262D; color: {PALETTE['accent']};
            }}
            QTableWidget::item:alternate {{ background: {PALETTE['bg']}; }}
            QHeaderView::section {{
                background: {PALETTE['bg']}; color: {PALETTE['muted']};
                border: 1px solid {PALETTE['border']}; padding: 4px 8px;
                font-size: 11px; font-weight: 600;
            }}
        """)
        return t

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10); root.setSpacing(6)

        # ── BÖLÜM 1: PENDING ─────────────────────────────────────────────
        pending_hdr = QWidget(); ph = QHBoxLayout(pending_hdr)
        ph.setContentsMargins(0, 0, 0, 0); ph.setSpacing(8)

        t1 = QLabel("⏳  Bekleyen Klipler (Staging)")
        t1.setStyleSheet(f"color: {PALETTE['yellow']}; font-size: 12px; font-weight: 700;")
        ph.addWidget(t1); ph.addStretch()

        self.refresh_btn = QPushButton("🔄  Yenile")
        self.refresh_btn.clicked.connect(self.refresh)

        self.play_btn = QPushButton("▶  Dinle")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play_selected_pending)

        self.stop_play_btn = QPushButton("⏹  Durdur")
        self.stop_play_btn.setEnabled(False)
        self.stop_play_btn.clicked.connect(self._stop_play)

        self.approve_btn = QPushButton("✅  Onayla")
        self.approve_btn.setEnabled(False)
        self.approve_btn.setStyleSheet(f"border-color: {PALETTE['green']}; color: {PALETTE['green']};")
        self.approve_btn.clicked.connect(self._approve_selected)

        self.reject_btn = QPushButton("❌  Reddet")
        self.reject_btn.setEnabled(False)
        self.reject_btn.setStyleSheet(f"border-color: {PALETTE['red']}; color: {PALETTE['red']};")
        self.reject_btn.clicked.connect(self._reject_selected)

        self.approve_all_btn = QPushButton("✅✅  Tümünü Onayla")
        self.approve_all_btn.setEnabled(False)
        self.approve_all_btn.setStyleSheet(f"border-color: {PALETTE['green']}; color: {PALETTE['green']};")
        self.approve_all_btn.clicked.connect(self._approve_all)

        for btn in [self.refresh_btn, self.play_btn, self.stop_play_btn,
                    self.approve_btn, self.reject_btn, self.approve_all_btn]:
            ph.addWidget(btn)
        root.addWidget(pending_hdr)

        self.pending_stat_lbl = QLabel(
            "Bekleyen klip yok. Mikrofon sekmesinden 'Etiketle & Gönder' ile ekleyebilirsin.")
        self.pending_stat_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        root.addWidget(self.pending_stat_lbl)

        self.pending_table = self._make_table(self._PENDING_COLS)
        self.pending_table.setMaximumHeight(220)
        self.pending_table.selectionModel().selectionChanged.connect(self._on_pending_selection)
        root.addWidget(self.pending_table)

        # ── AYIRICI ───────────────────────────────────────────────────────
        root.addWidget(_hline())

        # ── BÖLÜM 2: APPROVED ────────────────────────────────────────────
        approved_hdr = QWidget(); ah = QHBoxLayout(approved_hdr)
        ah.setContentsMargins(0, 0, 0, 0); ah.setSpacing(8)

        t2 = QLabel("✅  Onaylanmış Veri Seti")
        t2.setStyleSheet(f"color: {PALETTE['green']}; font-size: 12px; font-weight: 700;")
        ah.addWidget(t2); ah.addStretch()

        self.play_approved_btn = QPushButton("▶  Dinle")
        self.play_approved_btn.setEnabled(False)
        self.play_approved_btn.clicked.connect(self._play_selected_approved)

        self.delete_approved_btn = QPushButton("🗑  Seçileni Sil")
        self.delete_approved_btn.setEnabled(False)
        self.delete_approved_btn.setStyleSheet(
            f"border-color: {PALETTE['red']}; color: {PALETTE['red']};"
        )
        self.delete_approved_btn.clicked.connect(self._delete_approved_selected)

        self.delete_all_approved_btn = QPushButton("🗑🗑  Tümünü Sil")
        self.delete_all_approved_btn.setEnabled(False)
        self.delete_all_approved_btn.setStyleSheet(
            f"border-color: {PALETTE['red']}; color: {PALETTE['red']};"
        )
        self.delete_all_approved_btn.clicked.connect(self._delete_all_approved)

        self.export_csv_btn = QPushButton("📤  Eğitim CSV Dışa Aktar")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.setStyleSheet(
            f"border-color: {PALETTE['accent']}; color: {PALETTE['accent']};"
        )
        self.export_csv_btn.clicked.connect(self._export_training_csv)

        for btn in [self.play_approved_btn, self.delete_approved_btn,
                    self.delete_all_approved_btn, self.export_csv_btn]:
            ah.addWidget(btn)
        root.addWidget(approved_hdr)

        self.approved_stat_lbl = QLabel("Henüz onaylanmış klip yok.")
        self.approved_stat_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        root.addWidget(self.approved_stat_lbl)

        self.approved_table = self._make_table(self._APPROVED_COLS)
        self.approved_table.selectionModel().selectionChanged.connect(self._on_approved_selection)
        root.addWidget(self.approved_table)

        self.refresh()

    # ─────────────────────────────────────────────────────────────────────
    #  VERİ YÜKLEME
    # ─────────────────────────────────────────────────────────────────────

    def add_clip(self):
        """ClassificationTab / MicrophoneTab clip_added sinyalinden tetiklenir."""
        self.refresh()

    def refresh(self):
        self._load_pending_table()
        self._load_approved_table()

    def _fill_table(self, table: QTableWidget, rows: list):
        table.setRowCount(len(rows))
        from PyQt6.QtGui import QColor as _QColor, QFont as _QFont
        for i, row in enumerate(rows):
            user_lbl  = row.get("corrected_label", "")
            model_lbl = row.get("original_label", "")
            match     = "✓" if user_lbl == model_lbl else "≠"
            u_color   = CLASS_COLORS.get(user_lbl,  PALETTE["muted"])
            m_color   = CLASS_COLORS.get(model_lbl, PALETTE["muted"])
            mm_color  = PALETTE["green"] if user_lbl == model_lbl else PALETTE["yellow"]
            cells = [
                (user_lbl,  u_color,         True),
                (model_lbl, m_color,         True),
                (match,     mm_color,        False),
                (os.path.basename(row.get("source_file", "")), PALETTE["text"], False),
                (row.get("confidence", ""),  PALETTE["muted"], False),
                (row.get("note", ""),        PALETTE["muted"], False),
            ]
            for j, (val, color, bold) in enumerate(cells):
                item = QTableWidgetItem(str(val))
                item.setForeground(_QColor(color))
                if bold:
                    item.setFont(_QFont("Segoe UI", 10, _QFont.Weight.Bold))
                table.setItem(i, j, item)
        table.resizeColumnsToContents()

    def _load_pending_table(self):
        self._pending_rows = self._mgr.load_pending()
        self._fill_table(self.pending_table, self._pending_rows)
        n = len(self._pending_rows)
        if n > 0:
            self.pending_stat_lbl.setText(
                f"🕐  {n} bekleyen klip  |  Seç → Dinle → Onayla / Reddet")
            self.approve_all_btn.setEnabled(True)
        else:
            self.pending_stat_lbl.setText(
                "✅  Bekleyen klip yok. Mikrofon sekmesinden 'Etiketle & Gönder' ile ekleyebilirsin.")
            self.approve_all_btn.setEnabled(False)
        self._update_pending_btns()

    def _load_approved_table(self):
        self._approved_rows = self._mgr.load_approved()
        self._fill_table(self.approved_table, self._approved_rows)
        n = len(self._approved_rows)
        corrections = sum(
            1 for r in self._approved_rows
            if r.get("corrected_label") != r.get("original_label")
        )
        if n > 0:
            self.approved_stat_lbl.setText(
                f"✅  {n} onaylı klip  "
                f"({corrections} düzeltme, {n - corrections} model onayı)  "
                f"│  Seç → Dinle  │  Yanlış olanları seçip Sil"
            )
            self.export_csv_btn.setEnabled(True)
            self.delete_all_approved_btn.setEnabled(True)
        else:
            self.approved_stat_lbl.setText("Henüz onaylanmış klip yok.")
            self.export_csv_btn.setEnabled(False)
            self.delete_all_approved_btn.setEnabled(False)
        self._update_approved_btns()

    # ─────────────────────────────────────────────────────────────────────
    #  SEÇİM & BUTON GÜNCELLEME
    # ─────────────────────────────────────────────────────────────────────

    def _on_pending_selection(self, *_):
        self._update_pending_btns()

    def _on_approved_selection(self, *_):
        self._update_approved_btns()

    def _update_pending_btns(self):
        has = bool(self.pending_table.selectionModel().selectedRows())
        self.play_btn.setEnabled(has)
        self.approve_btn.setEnabled(has)
        self.reject_btn.setEnabled(has)

    def _update_approved_btns(self):
        has = bool(self.approved_table.selectionModel().selectedRows())
        self.play_approved_btn.setEnabled(has)
        self.delete_approved_btn.setEnabled(has)

    def _selected_pending(self) -> list:
        idxs = [idx.row() for idx in self.pending_table.selectionModel().selectedRows()]
        return [self._pending_rows[i] for i in idxs if i < len(self._pending_rows)]

    def _selected_approved(self) -> list:
        idxs = [idx.row() for idx in self.approved_table.selectionModel().selectedRows()]
        return [self._approved_rows[i] for i in idxs if i < len(self._approved_rows)]

    # ─────────────────────────────────────────────────────────────────────
    #  PENDING İŞLEMLERİ
    # ─────────────────────────────────────────────────────────────────────

    def _play_selected_pending(self):
        rows = self._selected_pending()
        if not rows: return
        self._play_clip(rows[0]["clip_path"])

    def _approve_selected(self):
        rows = self._selected_pending()
        if not rows: return
        self._stop_play()
        ok = sum(1 for r in rows if self._mgr.approve_clip(r["clip_id"]))
        self.refresh()
        QMessageBox.information(self, "Onaylandı",
            f"✅  {ok} klip onaylandı → live_clips/approved/")

    def _reject_selected(self):
        rows = self._selected_pending()
        if not rows: return
        reply = QMessageBox.question(
            self, "Emin misin?",
            f"{len(rows)} klip reddedilecek ve live_clips/rejected/ klasörüne taşınacak.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes: return
        self._stop_play()
        for r in rows: self._mgr.reject_clip(r["clip_id"])
        self.refresh()

    def _approve_all(self):
        if not self._pending_rows: return
        reply = QMessageBox.question(
            self, "Tümünü Onayla",
            f"Tüm {len(self._pending_rows)} bekleyen klip onaylanacak. Devam?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes: return
        self._stop_play()
        ok = sum(1 for r in self._pending_rows if self._mgr.approve_clip(r["clip_id"]))
        self.refresh()
        QMessageBox.information(self, "Tamamlandı",
            f"✅  {ok} klip onaylandı → live_clips/approved/")

    # ─────────────────────────────────────────────────────────────────────
    #  APPROVED İŞLEMLERİ
    # ─────────────────────────────────────────────────────────────────────

    def _play_selected_approved(self):
        rows = self._selected_approved()
        if not rows: return
        self._play_clip(rows[0]["clip_path"])

    def _delete_approved_selected(self):
        rows = self._selected_approved()
        if not rows: return
        self._stop_play()
        labels = ", ".join(r.get("corrected_label", "?") for r in rows)
        reply = QMessageBox.question(
            self, "Seçileni Sil",
            f"{len(rows)} klip kalıcı olarak silinecek:\nEtiketler: {labels}\nWAV dosyaları ve manifest kaydı birlikte silinir.\nBu işlem geri alınamaz. Devam?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes: return
        ok = sum(1 for r in rows if self._mgr.delete_approved(r["clip_id"], delete_file=True))
        self.refresh()
        QMessageBox.information(self, "Silindi", f"🗑  {ok} klip veri setinden kaldırıldı.")

    def _delete_all_approved(self):
        if not self._approved_rows: return
        self._stop_play()
        reply = QMessageBox.question(
            self, "Tüm Onaylı Veriyi Sil",
            ("⚠️  " + str(len(self._approved_rows)) + " onaylı klip TAMAMEN silinecek!\n\n"
             "WAV dosyaları ve approved_manifest.csv temizlenir.\n"
             "Bu işlem geri alınamaz. Emin misin?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes: return
        ok = sum(1 for r in self._approved_rows
                 if self._mgr.delete_approved(r["clip_id"], delete_file=True))
        self.refresh()
        QMessageBox.information(self, "Temizlendi",
            f"🗑  {ok} klip tamamen silindi. Veri seti sıfırlandı.")

    # ─────────────────────────────────────────────────────────────────────
    #  OYNATMA
    # ─────────────────────────────────────────────────────────────────────

    def _play_clip(self, path: str):
        if not SOUNDDEVICE_OK or not SOUNDFILE_OK:
            QMessageBox.warning(self, "Hata",
                "sounddevice veya soundfile yüklü değil.\npip install sounddevice soundfile")
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Klip Bulunamadı", f"Dosya mevcut değil:\n{path}")
            return
        try:
            sd.stop()
            data, sr = sf.read(path)
            sd.play(data, sr)
            self.stop_play_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "Oynatma Hatası", str(e))

    def _stop_play(self):
        if SOUNDDEVICE_OK:
            try: sd.stop()
            except Exception: pass
        self.stop_play_btn.setEnabled(False)

    # ─────────────────────────────────────────────────────────────────────
    #  CSV DIŞA AKTARMA
    # ─────────────────────────────────────────────────────────────────────

    def _export_training_csv(self):
        """
        Onaylanmış klipler için eğitim verisi CSV'si oluşturur.
        Yalnızca o an approved_manifest.csv'de bulunan satırları yazar.
        """
        approved = self._mgr.load_approved()
        if not approved:
            QMessageBox.information(self, "Veri Yok",
                "Onaylanmış klip bulunmuyor. Pending klipler varsa önce 'Onayla' butonuyla onayla.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Eğitim CSV Kaydet", "training_labels.csv",
            "CSV Dosyaları (*.csv)")
        if not path: return

        fields = ["clip_id", "clip_path", "user_label", "model_label",
                  "label_match", "confidence", "note", "timestamp"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in approved:
                user_lbl  = r.get("corrected_label", "")
                model_lbl = r.get("original_label", "")
                w.writerow({
                    "clip_id":     r.get("clip_id", ""),
                    "clip_path":   r.get("clip_path", ""),
                    "user_label":  user_lbl,
                    "model_label": model_lbl,
                    "label_match": "YES" if user_lbl == model_lbl else "NO",
                    "confidence":  r.get("confidence", ""),
                    "note":        r.get("note", ""),
                    "timestamp":   r.get("timestamp", ""),
                })

        corrections = sum(1 for r in approved
                          if r.get("corrected_label") != r.get("original_label"))
        QMessageBox.information(self, "CSV Dışa Aktarıldı",
            f"✅  {len(approved)} klip dışa aktarıldı:\n{path}\n\n"
            f"  • Model onayları        : {len(approved) - corrections}\n"
            f"  • Kullanıcı düzeltmeleri: {corrections}\n\n"
            "Bu dosyayı model eğitiminde kullanabilirsin.")


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 1 — MEL SPEKTROGRAM SEKMESİ
# ══════════════════════════════════════════════════════════════════════════

class SpectrogramTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(8,8,8,8)
        self.fig, self.canvas = _make_canvas(fig_h=5.5)
        lay.addWidget(self.canvas)

    def render(self, r):
        self.fig.clear()
        mel, dur = r["mel_db"], r["duration"]
        fl, ft = r["frame_labels"], r["frame_times"]
        gs = self.fig.add_gridspec(2,1,height_ratios=[0.07,1],hspace=0.04)
        ax_s = self.fig.add_subplot(gs[0]); ax_m = self.fig.add_subplot(gs[1])
        self.fig.patch.set_facecolor(PALETTE["bg"])
        hop = ft[1]-ft[0] if len(ft)>1 else 2.5
        for t, lbl in zip(ft, fl):
            ax_s.barh(0, hop*0.98, left=t, height=1.0,
                      color=CLASS_COLORS.get(lbl,"#6C757D"), alpha=0.9, align="edge")
        ax_s.set_xlim(0,dur); ax_s.set_yticks([]); ax_s.set_xticks([])
        for sp in ax_s.spines.values(): sp.set_edgecolor(PALETTE["border"])
        ax_s.set_facecolor(PALETTE["bg"])
        im = ax_m.imshow(mel,aspect="auto",origin="lower",cmap="magma",
                         extent=[0,dur,0,mel.shape[0]],vmin=np.percentile(mel,5))
        ax_m.set_xlabel("Zaman (s)",color=PALETTE["muted"],fontsize=9)
        ax_m.set_ylabel("Mel Kanalı",color=PALETTE["muted"],fontsize=9)
        ax_m.tick_params(colors=PALETTE["muted"],labelsize=8)
        ax_m.set_facecolor(PALETTE["surface"])
        for sp in ax_m.spines.values(): sp.set_edgecolor(PALETTE["border"])
        cb = self.fig.colorbar(im, ax=ax_m, pad=0.01)
        cb.set_label("dB",color=PALETTE["muted"],fontsize=8)
        cb.ax.yaxis.set_tick_params(color=PALETTE["muted"],labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(),color=PALETTE["muted"])
        for t in ft: ax_m.axvline(t,color=PALETTE["border"],lw=0.4,alpha=0.5)
        self.fig.suptitle("Mel Spektrogram (128 Mel, 2048-FFT)",
                           color=PALETTE["text"],fontsize=10,y=0.99)
        self.fig.tight_layout(pad=0.5); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 1 — ÖZELLİKLER SEKMESİ
# ══════════════════════════════════════════════════════════════════════════

class FeaturesTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)
        self.fig, self.canvas = _make_canvas(fig_h=6.5)
        lay.addWidget(self.canvas)

    def render(self, r):
        self.fig.clear(); self.fig.patch.set_facecolor(PALETTE["bg"])
        fl, ft, dur = r["frame_labels"], r["frame_times"], r["duration"]
        hop = ft[1]-ft[0] if len(ft)>1 else 2.5
        zcr_t,zcr_v = r["zcr"]; rms_t,rms_v = r["rms"]; sc_t,sc_v = r["sc"]
        for i,(t,v,title,color) in enumerate([
            (zcr_t,zcr_v,"Zero Crossing Rate","#FFE66D"),
            (rms_t,rms_v,"RMS Enerji","#FF6B9D"),
            (sc_t, sc_v, "Spectral Centroid (Hz)","#7EE8A2"),
        ]):
            ax = self.fig.add_subplot(3,1,i+1); _style_ax(ax)
            for ft_,lbl in zip(ft,fl):
                ax.axvspan(ft_,min(ft_+hop,dur),alpha=0.10,color=CLASS_COLORS.get(lbl,"#6C757D"),lw=0)
            ax.plot(t,v,color=color,lw=1.0); ax.fill_between(t,v,alpha=0.10,color=color)
            ax.set_title(title,color=PALETTE["text"],fontsize=9,pad=3); ax.set_xlim(0,dur)
            if i<2: ax.set_xticklabels([])
            else: ax.set_xlabel("Zaman (s)",fontsize=8)
        self.fig.tight_layout(pad=0.6); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 2 — CANLI FREKANS SPEKTRUMU
# ══════════════════════════════════════════════════════════════════════════

class LiveSpectrumWidget(QWidget):
    SR     = 22050
    N_FFT  = 2048
    MAX_HZ = 11025

    def __init__(self):
        super().__init__()
        self._buffer = collections.deque(maxlen=self.N_FFT * 4)
        self._label  = "---"
        self.setMinimumHeight(160)

        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)

        header = QWidget(); hl = QHBoxLayout(header)
        hl.setContentsMargins(4,2,4,2)
        hl.addWidget(_label("CANLI FREKANS SPEKTRUMU", muted=True, small=True))
        hl.addStretch()
        self._label_lbl = QLabel("---")
        self._label_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px; font-weight: 600;")
        hl.addWidget(self._label_lbl)
        lay.addWidget(header)

        self.fig, self.canvas = _make_canvas(fig_h=2.0, fig_w=8)
        lay.addWidget(self.canvas)

        self._freqs = np.fft.rfftfreq(self.N_FFT, 1 / self.SR)

        self._timer = QTimer(); self._timer.timeout.connect(self._redraw)
        self._timer.start(100)

        self._draw_empty()

    def push_chunk(self, chunk: np.ndarray):
        self._buffer.extend(chunk.tolist())

    def set_label(self, label: str):
        self._label = label
        color = CLASS_COLORS.get(label, PALETTE["muted"])
        self._label_lbl.setText(label)
        self._label_lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")

    def clear(self):
        self._buffer.clear(); self._label = "---"
        self._label_lbl.setText("---")
        self._label_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px;")
        self._draw_empty()

    def _redraw(self):
        if len(self._buffer) < self.N_FFT:
            return

        frame = np.array(list(self._buffer)[-self.N_FFT:], dtype=np.float32)
        windowed = frame * np.hanning(self.N_FFT)
        spectrum  = np.abs(np.fft.rfft(windowed))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)

        if not hasattr(self, "_prev_spec") or len(self._prev_spec) != len(spectrum_db):
            self._prev_spec = spectrum_db.copy()
        else:
            self._prev_spec = 0.7 * self._prev_spec + 0.3 * spectrum_db

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        self.fig.patch.set_facecolor(PALETTE["bg"])

        color = CLASS_COLORS.get(self._label, PALETTE["accent"])

        ax.axvspan(0,    100,   alpha=0.06, color="#FFE66D", lw=0)
        ax.axvspan(100,  4000,  alpha=0.05, color="#7EE8A2", lw=0)
        ax.axvspan(4000, 11025, alpha=0.04, color="#A8DADC", lw=0)

        ax.plot(self._freqs, self._prev_spec, color=color, lw=1.0, alpha=0.9)
        ax.fill_between(self._freqs, self._prev_spec,
                        np.min(self._prev_spec) - 5,
                        color=color, alpha=0.12)

        for freq, lbl in [(100,"100Hz"), (1000,"1kHz"), (4000,"4kHz"), (8000,"8kHz")]:
            ax.axvline(freq, color=PALETTE["border"], lw=0.6, linestyle=":")
            ax.text(freq, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else -10,
                    lbl, color=PALETTE["muted"], fontsize=6, ha="center", va="top")

        ax.set_xlim(20, self.MAX_HZ)
        ax.set_xscale("log")
        ax.set_xlabel("Frekans (Hz — log)", fontsize=7)
        ax.set_ylabel("dB", fontsize=7)
        ax.set_title(f"Anlık Spektrum  [{self._label}]",
                     color=PALETTE["text"], fontsize=8, pad=2)

        self.fig.tight_layout(pad=0.3)
        self.canvas.draw()

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111); _style_ax(ax)
        self.fig.patch.set_facecolor(PALETTE["bg"])
        ax.text(0.5, 0.5, "Mikrofon başlatılınca spektrum görünür",
                color=PALETTE["muted"], ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_title("Canlı Frekans Spektrumu", color=PALETTE["text"], fontsize=8, pad=2)
        self.fig.tight_layout(pad=0.3); self.canvas.draw()

    def update_svantek_spectrum(self, freqs: list, levels: list):
        """
        Svantek 1/3 oktav spektrumunu bar grafik olarak çizer.
        Sentetik waveform yerine doğrudan kalibreli dB değerleri kullanılır.
        100 ms'de bir QTimer tetiklenince çizim yapmak yerine burada anında çizer.
        """
        import numpy as _np
        if not freqs or not levels:
            return

        # Üstel yumuşatma
        if not hasattr(self, "_sv_prev_levels") or len(self._sv_prev_levels) != len(levels):
            self._sv_prev_levels = list(levels)
        else:
            self._sv_prev_levels = [
                0.75 * p + 0.25 * c
                for p, c in zip(self._sv_prev_levels, levels)
            ]

        color  = CLASS_COLORS.get(self._label, PALETTE["accent"])
        freqs_arr  = _np.array(freqs,  dtype=float)
        levels_arr = _np.array(self._sv_prev_levels, dtype=float)

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        _style_ax(ax)
        self.fig.patch.set_facecolor(PALETTE["bg"])

        # Bar genişlikleri log ekseninde doğal görünmesi için
        width = freqs_arr * (2 ** (1/6) - 2 ** (-1/6))   # 1/3 oktav genişlik
        ax.bar(freqs_arr, levels_arr,
               width=width, color=color, alpha=0.75,
               align="center", zorder=5, label=self._label)

        # Referans çizgileri
        for db_ref in [50, 60, 70, 80, 90, 100]:
            ax.axhline(db_ref, color=PALETTE["border"], lw=0.5, linestyle="--", alpha=0.5)

        # Bölge gölgeleri
        ax.axvspan(20,   100,  alpha=0.05, color="#FFE66D", lw=0)
        ax.axvspan(100,  4000, alpha=0.04, color="#7EE8A2", lw=0)
        ax.axvspan(4000, 20000,alpha=0.03, color="#A8DADC", lw=0)

        ymin = max(30.0, float(_np.min(levels_arr)) - 5)
        ymax = min(140.0, float(_np.max(levels_arr)) + 10)
        ax.set_xlim(18, 22000)
        ax.set_ylim(ymin, ymax)
        ax.set_xscale("log")

        xticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        xtick_labels = ["31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
        ax.set_xticks(xticks); ax.set_xticklabels(xtick_labels, fontsize=6.5)

        ax.set_xlabel("Frekans (Hz — 1/3 Oktav)", fontsize=7)
        ax.set_ylabel("Seviye (dBSPL)", fontsize=7)
        ax.set_title(f"Svantek SV 971 — 1/3 Oktav Spektrum  [{self._label}]",
                     color=PALETTE["text"], fontsize=8, pad=2)

        self.fig.tight_layout(pad=0.3)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 2 — ÖZEL WİDGET'LAR
# ══════════════════════════════════════════════════════════════════════════

class VUMeter(QWidget):
    def __init__(self):
        super().__init__()
        self._db = self._peak_db = -80.0; self._peak_hold = 0
        self.setFixedHeight(24); self.setMinimumWidth(150)
        t = QTimer(self); t.timeout.connect(self._tick); t.start(40)

    def set_db(self, db):
        self._db = max(-80.0, min(0.0, db))
        if self._db > self._peak_db: self._peak_db = self._db; self._peak_hold = 40
        self.update()

    def reset(self): self._db = self._peak_db = -80.0; self._peak_hold = 0; self.update()

    def _tick(self):
        if self._peak_hold > 0: self._peak_hold -= 1
        else: self._peak_db = max(self._peak_db - 0.3, self._db)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, m = self.width(), self.height(), 3
        p.fillRect(0,0,w,h,QColor(PALETTE["surface"]))
        p.setPen(QPen(QColor(PALETTE["border"]))); p.drawRoundedRect(0,0,w-1,h-1,3,3)
        bw = int((self._db+80)/80*(w-2*m))
        grad = QLinearGradient(m,0,w-m,0)
        grad.setColorAt(0.00,QColor("#7EE8A2")); grad.setColorAt(0.65,QColor("#FFE66D"))
        grad.setColorAt(1.00,QColor("#FF4444"))
        p.fillRect(m,m,bw,h-2*m,QBrush(grad))
        if self._peak_db > -79:
            px = m+int((self._peak_db+80)/80*(w-2*m))
            p.setPen(QPen(QColor("white"),2)); p.drawLine(px,m,px,h-m)
        p.setPen(QPen(QColor(PALETTE["border"]),1))
        for tick in [-60,-40,-20,-6]:
            tx = m+int((tick+80)/80*(w-2*m)); p.drawLine(tx,h-5,tx,h-2)
        p.end()


class RollingClassStrip(QWidget):
    HISTORY = 60
    def __init__(self):
        super().__init__()
        self._hist = collections.deque(maxlen=self.HISTORY)
        self.setMinimumHeight(48); self.setMaximumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def add_label(self, label): self._hist.append(label); self.update()
    def clear(self): self._hist.clear(); self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0,0,w,h,QColor(PALETTE["surface"]))
        p.setPen(QPen(QColor(PALETTE["border"]))); p.drawRect(0,0,w-1,h-1)
        n = len(self._hist)
        if n == 0:
            p.setPen(QPen(QColor(PALETTE["muted"]))); p.setFont(QFont("Segoe UI",9))
            p.drawText(QRect(0,0,w,h), Qt.AlignmentFlag.AlignCenter, "Dinleniyor…")
            p.end(); return
        sw = w / self.HISTORY
        p.setFont(QFont("Segoe UI",7,QFont.Weight.Bold))
        for i, lbl in enumerate(self._hist):
            x = int(w-(n-i)*sw); cw = max(1,int(sw)-1)
            p.fillRect(x,2,cw,h-4,QColor(CLASS_COLORS.get(lbl,"#6C757D")))
            if sw > 28:
                p.setPen(QPen(QColor("white")))
                p.drawText(QRect(x,2,cw,h-4), Qt.AlignmentFlag.AlignCenter, lbl[:3])
        p.setPen(QPen(QColor(PALETTE["border"]),1)); p.setFont(QFont("Segoe UI",7))
        for t in range(10, self.HISTORY+1, 10):
            x = int(w-t*sw)
            if 0 < x < w:
                p.drawLine(x,h-8,x,h); p.setPen(QPen(QColor(PALETTE["muted"])))
                p.drawText(QRect(x-15,h-8,30,8), Qt.AlignmentFlag.AlignCenter, f"-{t}s")
                p.setPen(QPen(QColor(PALETTE["border"]),1))
        p.end()


class DbHistoryWidget(QWidget):
    HISTORY = 60
    def __init__(self):
        super().__init__()
        self._db   = collections.deque(maxlen=self.HISTORY)
        self._lbls = collections.deque(maxlen=self.HISTORY)
        self.setMinimumHeight(120)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.fig, self.canvas = _make_canvas(fig_h=1.6, fig_w=8)
        lay.addWidget(self.canvas)

    def add(self, db, label): self._db.append(db); self._lbls.append(label); self._redraw()
    def clear(self): self._db.clear(); self._lbls.clear(); self._redraw()

    def _redraw(self):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        _style_ax(ax); self.fig.patch.set_facecolor(PALETTE["bg"])
        n = len(self._db)
        if n == 0:
            ax.set_xlim(-self.HISTORY,0); ax.set_ylim(-80,0)
            self.fig.tight_layout(pad=0.3); self.canvas.draw(); return
        xs = np.arange(-n+1,1); ys = np.array(self._db); lbls = list(self._lbls)
        for i,lbl in enumerate(lbls):
            ax.axvspan(xs[i]-0.5,xs[i]+0.5,alpha=0.15,color=CLASS_COLORS.get(lbl,"#6C757D"),lw=0)
        ax.plot(xs,ys,color=PALETTE["accent2"],lw=1.2,zorder=5)
        ax.fill_between(xs,ys,-80,color=PALETTE["accent2"],alpha=0.12,zorder=4)
        ax.set_xlim(-self.HISTORY,0)
        ax.set_ylim(min(-60,np.min(ys)-5),max(-10,np.max(ys)+5))
        ax.set_xlabel("Saniye önce",fontsize=7); ax.set_ylabel("dB (SPL/FS)",fontsize=7)
        ax.set_title("Ses Seviyesi Geçmişi (son 60s)",color=PALETTE["text"],fontsize=8,pad=2)
        self.fig.tight_layout(pad=0.3); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 2 — MİKROFON WORKER
# ══════════════════════════════════════════════════════════════════════════

class MicrophoneWorker(QThread):
    result_signal = pyqtSignal(dict)
    vu_signal     = pyqtSignal(float)
    chunk_signal  = pyqtSignal(object)
    status_signal = pyqtSignal(str)
    error_signal  = pyqtSignal(str)

    SR             = 22050
    WINDOW_SAMPLES = int(5.0 * 22050)
    SLIDE_SAMPLES  = int(1.0 * 22050)
    VU_INTERVAL    = int(0.04 * 22050)
    SPEC_INTERVAL  = int(0.10 * 22050)
    BLOCK_SIZE     = 1024

    def __init__(self, system, device_idx, model_pref):
        super().__init__()
        self._system = system; self._device = device_idx
        self._model_pref = model_pref; self._stop_flag = False
        self._audio_q = queue.Queue(); self._rec_active = False
        self._pred_buffer  = deque(maxlen=5)   # son 5 tahmin (≈5 saniye)
        self._rec_chunks = []; self._rec_path = ""

    def start_recording(self, path):
        self._rec_chunks = []; self._rec_path = path; self._rec_active = True

    def stop_recording(self):
        self._rec_active = False
        if not self._rec_chunks or not self._rec_path: return ""
        audio = np.concatenate(self._rec_chunks)
        if SOUNDFILE_OK: sf.write(self._rec_path, audio, self.SR)
        else:
            import wave as wv, struct
            with wv.open(self._rec_path,"wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.SR)
                int16 = (np.clip(audio,-1,1)*32767).astype(np.int16)
                wf.writeframes(struct.pack(f"<{len(int16)}h", *int16))
        self._rec_chunks = []; return self._rec_path

    def stop(self): self._stop_flag = True

    def run(self):
        if not SOUNDDEVICE_OK:
            self.error_signal.emit("sounddevice yüklü değil.\npip install sounddevice"); return

        buffer = np.zeros(self.WINDOW_SAMPLES, dtype=np.float32)
        s_inf = 0; s_vu = 0; s_spec = 0
        start_time = time.time()

        def callback(indata, frames, time_info, status):
            self._audio_q.put(indata[:,0].copy())

        try:
            stream = sd.InputStream(samplerate=self.SR, channels=1, dtype="float32",
                                     blocksize=self.BLOCK_SIZE, device=self._device,
                                     callback=callback)
            self.status_signal.emit("🎙  Mikrofon aktif — dinleniyor…")

            with stream:
                while not self._stop_flag:
                    try: chunk = self._audio_q.get(timeout=0.15)
                    except queue.Empty: continue

                    n = len(chunk)
                    buffer = np.roll(buffer,-n); buffer[-n:] = chunk

                    if self._rec_active: self._rec_chunks.append(chunk.copy())

                    s_vu += n
                    if s_vu >= self.VU_INTERVAL:
                        s_vu = 0
                        rms = float(np.sqrt(np.mean(chunk**2)))
                        self.vu_signal.emit(float(20*np.log10(rms+1e-10)))

                    s_spec += n
                    if s_spec >= self.SPEC_INTERVAL:
                        s_spec = 0
                        self.chunk_signal.emit(chunk.copy())

                    s_inf += n
                    if s_inf >= self.SLIDE_SAMPLES:
                        s_inf = 0
                        elapsed = time.time() - start_time
                        try:
                            res = self._system.classify_chunk_live(
                                buffer.copy(), self._model_pref
                            )
                            res["elapsed"] = elapsed
                            res["samples"]  = buffer.copy()

                            # ── Majority voting ───────────────────────
                            self._pred_buffer.append(res["label"])
                            votes    = Counter(self._pred_buffer)
                            smoothed = votes.most_common(1)[0][0]
                            res["raw_label"] = res["label"]   # ham tahmin
                            res["label"]     = smoothed       # yumuşatılmış
                            # ─────────────────────────────────────────

                            self.result_signal.emit(res)
                        except Exception as e:
                            print(f"[Inference] {e}")

            self.status_signal.emit("⏹  Mikrofon durduruldu.")
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════
#  FAZ 2 — MİKROFON SEKMESİ
# ══════════════════════════════════════════════════════════════════════════

class MicrophoneTab(QWidget):
    
    clip_added = pyqtSignal()

    MODEL_MAP = {
    "Otomatik (EfficientNet → CNN → SVM)": "auto",
    "EfficientNet-B0": "efficientnet",
    "CNN": "cnn",
    "SVM": "svm",
    "BEATs (Modern)": "beats",
    "Ensemble (EfficientNet + BEATs)": "ensemble",
    }

    def __init__(self, system, clip_manager=None):
        super().__init__()
        self._system = system
        self._mgr = clip_manager
        self._worker = None
        self._last_res = None
        self._rec_file = None; self._rec_csv = None; self._clock_start = 0.0
        self._build_ui(); self._populate_devices()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(10,8,10,8); root.setSpacing(8)

        tb = QWidget(); tl = QHBoxLayout(tb); tl.setContentsMargins(0,0,0,0); tl.setSpacing(8)
        tl.addWidget(_label("🎙 Cihaz:", muted=True))
        self.device_combo = QComboBox(); self.device_combo.setMinimumWidth(230)
        tl.addWidget(self.device_combo)
        tl.addWidget(_label("Model:", muted=True))
        self.model_combo = QComboBox()
        for k in self.MODEL_MAP: self.model_combo.addItem(k)
        tl.addWidget(self.model_combo)

        self.start_btn = QPushButton("▶  Başlat"); self.start_btn.setObjectName("startMicBtn")
        self.start_btn.clicked.connect(self._start_stream)
        self.stop_btn = QPushButton("⏹  Durdur"); self.stop_btn.setObjectName("stopMicBtn")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop_stream)
        self.rec_btn = QPushButton("⚫  Kayıt Başlat"); self.rec_btn.setObjectName("recBtn")
        self.rec_btn.setEnabled(False); self.rec_btn.clicked.connect(self._toggle_recording)

        tl.addWidget(self.start_btn); tl.addWidget(self.stop_btn)
        tl.addWidget(self.rec_btn); tl.addStretch()
        root.addWidget(tb)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {PALETTE['border']}; }}")

        left = QFrame(); left.setObjectName("sidePanel"); left.setFixedWidth(240)
        ll = QVBoxLayout(left); ll.setContentsMargins(14,14,14,14); ll.setSpacing(12)

        self.class_badge = QLabel("— — —")
        self.class_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.class_badge.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.class_badge.setStyleSheet(f"color: {PALETTE['muted']};")
        self.class_badge.setMinimumHeight(68)
        ll.addWidget(self.class_badge)

        self.db_label = QLabel("-∞  dBFS")
        self.db_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.db_label.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 15px;")
        ll.addWidget(self.db_label)

        self.vu = VUMeter(); ll.addWidget(self.vu)

        self.elapsed_lbl = QLabel("⏱  00:00")
        self.elapsed_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.elapsed_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        ll.addWidget(self.elapsed_lbl)

        ll.addWidget(_hline())
        ll.addWidget(_label("CANLI GÜVEN", muted=True, small=True))

        self.conf_bars = {}
        for cls, color in CLASS_COLORS.items():
            if cls == "UNKNOWN": continue
            row = QWidget(); rl = QHBoxLayout(row)
            rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
            name = QLabel(cls); name.setFixedWidth(68)
            name.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")
            bar = QProgressBar(); bar.setRange(0,100); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(7)
            bar.setStyleSheet(f"QProgressBar {{ background: {PALETTE['border']}; border-radius: 3px; border: none; }} QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}")
            pct = QLabel("0%"); pct.setFixedWidth(36)
            pct.setAlignment(Qt.AlignmentFlag.AlignRight)
            pct.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
            rl.addWidget(name); rl.addWidget(bar); rl.addWidget(pct)
            ll.addWidget(row)
            self.conf_bars[cls] = (bar, pct)

        ll.addStretch()
        splitter.addWidget(left)

        right = QWidget(); rl_lay = QVBoxLayout(right)
        rl_lay.setContentsMargins(4,0,4,0); rl_lay.setSpacing(8)

        rl_lay.addWidget(_label("SON 60 SANİYE — SINIFLANDIRMA ŞERİDİ", muted=True, small=True))
        self.rolling_strip = RollingClassStrip(); rl_lay.addWidget(self.rolling_strip)

        leg = QWidget(); legl = QHBoxLayout(leg)
        legl.setContentsMargins(0,0,0,0); legl.setSpacing(14)
        for cls, color in CLASS_COLORS.items():
            if cls == "UNKNOWN": continue
            d = QLabel(f"● {cls}")
            d.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            legl.addWidget(d)
        legl.addStretch(); rl_lay.addWidget(leg)

        rl_lay.addWidget(_hline())
        rl_lay.addWidget(_label("SES SEVİYESİ GEÇMİŞİ — dBFS (son 60s)", muted=True, small=True))
        self.db_history = DbHistoryWidget(); rl_lay.addWidget(self.db_history)

        rl_lay.addWidget(_hline())
        self.live_spectrum = LiveSpectrumWidget()
        rl_lay.addWidget(self.live_spectrum)

        rl_lay.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([240, 900]); splitter.setStretchFactor(1,1)
        root.addWidget(splitter)

        self.send_to_review_btn = QPushButton("🏷  Etiketle & Gönder")
        self.send_to_review_btn.setEnabled(False)
        self.send_to_review_btn.setStyleSheet(
            f"border-color: {PALETTE['yellow']}; color: {PALETTE['yellow']};"
        )
        self.send_to_review_btn.clicked.connect(self._send_to_review)
        tl.addWidget(self.send_to_review_btn) # Layout'a ekle

        self.status_lbl = QLabel("Mikrofon başlatılmadı.")
        self.status_lbl.setStyleSheet(f"""
            background: {PALETTE['surface']}; color: {PALETTE['muted']};
            border-top: 1px solid {PALETTE['border']}; padding: 4px 10px; font-size: 11px;
        """)
        root.addWidget(self.status_lbl)
        self._clock = QTimer(); self._clock.timeout.connect(self._tick_clock)

    def _populate_devices(self):
        """
        Ses giriş cihazlarını ve Svantek USB-HID cihazlarını listeler.

        self._device_map: list[dict]
          Her giriş:
            {"type": "sd",      "idx": int,   "label": str}  <- sounddevice
            {"type": "svantek", "path": bytes, "label": str}  <- Svantek HID
        """
        self.device_combo.clear()
        self._device_map = []

        # 1. sounddevice mikrofonları
        if SOUNDDEVICE_OK:
            try:
                devs = sd.query_devices()
                default_in = sd.default.device[0]
                for i, dev in enumerate(devs):
                    if dev["max_input_channels"] > 0:
                        tag   = " ✦ [Varsayılan]" if i == default_in else ""
                        label = f"{dev['name']}{tag}"
                        self.device_combo.addItem(label)
                        self._device_map.append({"type": "sd", "idx": i, "label": label})
                for ci, entry in enumerate(self._device_map):
                    if entry["type"] == "sd" and entry["idx"] == default_in:
                        self.device_combo.setCurrentIndex(ci); break
            except Exception as e:
                self.device_combo.addItem(f"Cihaz listelenemedi: {e}")
                self.start_btn.setEnabled(False)
        else:
            self.device_combo.addItem("sounddevice yüklü değil")

        # 2. Svantek USB-HID cihazları
        if SVANTEK_OK:
            sv_entries = find_svantek_gui_entry()
            for sv in sv_entries:
                self.device_combo.addItem(sv["label"])
                self._device_map.append({
                    "type":  "svantek",
                    "path":  sv["path"],
                    "label": sv["label"],
                })
            if sv_entries:
                print(f"[GUI] {len(sv_entries)} Svantek cihazı bulundu.")
        else:
            self.device_combo.addItem("-- Svantek: pip install hidapi gerekli --")
            self.device_combo.model().item(
                self.device_combo.count() - 1
            ).setEnabled(False)

        if not self._device_map:
            self.start_btn.setEnabled(False)

    def _selected_device(self):
        """Seçili dropdown girişine ait aygıt bilgisini döner."""
        ci = self.device_combo.currentIndex()
        if hasattr(self, "_device_map") and 0 <= ci < len(self._device_map):
            return self._device_map[ci]
        return None

    def _start_stream(self):
        if self._worker and self._worker.isRunning(): return

        dev_entry  = self._selected_device()
        model_pref = self.MODEL_MAP[self.model_combo.currentText()]
        self.rolling_strip.clear(); self.db_history.clear()
        self.vu.reset(); self.live_spectrum.clear()

        if dev_entry is None:
            self._set_status("❌  Geçerli cihaz seçili değil.")
            return

        if dev_entry["type"] == "svantek":
            # ── Svantek USB-HID Worker ──────────────────────────────────────
            if not SVANTEK_OK:
                QMessageBox.critical(
                    self, "Svantek Hatası",
                    f"svantek_hid.py yüklenemedi.\n{_SVANTEK_ERR}\n\n"
                    "pip install hidapi  komutunu çalıştırın."
                )
                return
            self._worker = SvantekWorker(
                self._system, dev_entry["path"], model_pref
            )
            self._worker.svantek_signal.connect(self._on_svantek)
        else:
            # ── Normal sounddevice Worker ───────────────────────────────────
            sd._terminate(); sd._initialize()
            self._worker = MicrophoneWorker(
                self._system, dev_entry["idx"], model_pref
            )

        self._worker.result_signal.connect(self._on_result)
        self._worker.vu_signal.connect(self._on_vu)
        self._worker.chunk_signal.connect(self.live_spectrum.push_chunk)
        self._worker.status_signal.connect(self._set_status)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.rec_btn.setEnabled(True)
        self.model_combo.setEnabled(False); self.device_combo.setEnabled(False)
        self._clock_start = time.time(); self._clock.start(500)

    def _stop_stream(self):
        self._clock.stop()
        if self._rec_file: self._stop_recording_internal()
        if self._worker:
            self._worker.stop(); self._worker.wait(3000); self._worker = None
        self.live_spectrum.clear()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.rec_btn.setEnabled(False); self.rec_btn.setText("⚫  Kayıt Başlat")
        self.rec_btn.setStyleSheet("")
        self.model_combo.setEnabled(True); self.device_combo.setEnabled(True)
        self._set_status("⏹  Mikrofon durduruldu.")

    def _toggle_recording(self):
        if self._rec_file is None: self._start_recording()
        else: self._stop_recording_internal()

    def _start_recording(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = "outputs_mic"; os.makedirs(out, exist_ok=True)
        wav = os.path.join(out, f"session_{ts}.wav")
        csv_p = os.path.join(out, f"session_{ts}.csv")
        self._rec_wav = wav; self._rec_start = time.time()
        self._rec_file = open(csv_p,"w",newline="",encoding="utf-8")
        self._rec_csv = csv.writer(self._rec_file)
        self._rec_csv.writerow(["timestamp","elapsed_s","label"] +
                                [c for c in CLASS_COLORS if c!="UNKNOWN"])
        if self._worker: self._worker.start_recording(wav)
        self.rec_btn.setText("⏹  Kayıt Durdur")
        self.rec_btn.setStyleSheet(f"background-color: #4a1a1a; border: 1px solid {PALETTE['red']}; color: {PALETTE['red']}; padding: 7px 18px; border-radius: 6px; font-weight: 700;")
        self._set_status(f"● Kayıt: {os.path.abspath(wav)}")

    def _stop_recording_internal(self):
        if self._rec_file:
            self._rec_file.close(); self._rec_file = None; self._rec_csv = None
        if self._worker:
            saved = self._worker.stop_recording()
            if saved:
                self._set_status(f"✓ Kayıt kaydedildi: {saved}")
                QMessageBox.information(self,"Kayıt Tamamlandı",
                    f"WAV ve CSV kaydedildi:\n\n{os.path.abspath(saved)}")
        self.rec_btn.setText("⚫  Kayıt Başlat"); self.rec_btn.setStyleSheet("")

    def _on_result(self, res):
        
        self._last_res = res # Son sonucu sakla
        self.send_to_review_btn.setEnabled(True) # Butonu aktif et

        label = res.get("label","UNKNOWN"); probs = res.get("probs",{})
        db_rms = res.get("db_rms",-80.0); elapsed = res.get("elapsed",0.0)
        color = CLASS_COLORS.get(label, PALETTE["muted"])
        self.class_badge.setText(label); self.class_badge.setStyleSheet(f"color: {color};")
        for cls,(bar,pct) in self.conf_bars.items():
            v = int(probs.get(cls,0.0)*100); bar.setValue(v); pct.setText(f"{v}%")
        self.rolling_strip.add_label(label); self.db_history.add(db_rms, label)
        self.live_spectrum.set_label(label)
        if self._rec_csv:
            self._rec_csv.writerow(
                [time.strftime("%Y-%m-%d %H:%M:%S"), f"{elapsed:.1f}", label]
                + [f"{probs.get(c,0.0):.4f}" for c in CLASS_COLORS if c!="UNKNOWN"]
            )
            if self._rec_file: self._rec_file.flush()

    def _send_to_review(self):
        """
        LiveLabelDialog açar: modelin tahmini gösterilir, kullanıcı
        kendi etiketini seçer. Her iki etiket CSV'ye kaydedilir.
        """
        if not self._last_res or self._mgr is None:
            return

        model_label = self._last_res.get("label", "UNKNOWN")
        elapsed     = self._last_res.get("elapsed", 0.0)

        dlg = LiveLabelDialog(model_label=model_label, elapsed=elapsed, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        ann_row = dlg.result_row()
        samples = self._last_res.get("samples")
        if samples is not None:
            try:
                self._mgr.save_pending_clip(samples, 22050, 0.0, ann_row)
                self.clip_added.emit()
                user_lbl  = ann_row["corrected_label"]
                match_txt = "✓ onay" if user_lbl == model_label else f"→ {user_lbl}"
                self._set_status(
                    f"✓ Klip gönderildi  |  Model: {model_label}  |  Etiketin: {match_txt}"
                )
            except Exception as e:
                print(f"[LiveLabel] Klip kaydedilemedi: {e}")
                self._set_status("❌ Klip kaydedilemedi.")
        else:
            self._set_status("⚠ Ses verisi alınamadı, klip gönderilemedi.")

    def _on_vu(self, db):
        """
        VU metre güncellemesi.
        sounddevice → dBFS (negatif değer, -80..0)
        Svantek     → dBSPL (pozitif değer, 30..140)
        """
        dev_entry = self._selected_device()
        is_svantek = (dev_entry is not None and dev_entry.get("type") == "svantek")

        if is_svantek:
            # Kalibreli dBSPL — VU metre 30-130 dB aralığına normalize et
            vu_val = max(-80.0, min(0.0, (db - 30.0) / 100.0 * 80.0 - 80.0))
            self.vu.set_db(vu_val)
            unit = "dBSPL(A)"
            col  = (PALETTE["red"] if db > 100 else
                    PALETTE["yellow"] if db > 75 else PALETTE["green"])
            self.db_label.setText(f"{db:.1f}  {unit}")
        else:
            self.vu.set_db(db)
            col = PALETTE["red"] if db > -10 else (PALETTE["yellow"] if db > -30 else PALETTE["green"])
            self.db_label.setText(f"{db:+.1f}  dBFS")

        self.db_label.setStyleSheet(f"color: {col}; font-size: 15px; font-weight: 600;")

    def _on_svantek(self, sv_data: dict):
        """
        SvantekWorker.svantek_signal → Svantek'e özgü ham ölçüm verisi.
        L, Leq, Lmax, Lmin, Lpeak ve 1/3 oktav spektrum içerebilir.
        """
        L     = sv_data.get("L",    0.0)
        Leq   = sv_data.get("Leq",  0.0)
        Lmax  = sv_data.get("Lmax", 0.0)
        filt  = sv_data.get("filter", "A")

        # Durum çubuğunu güncelle
        self._set_status(
            f"🔬 Svantek  L={L:.1f} dB{filt}  Leq={Leq:.1f}  Lmax={Lmax:.1f}  ⏱ {fmt_elapsed(time.time()-self._clock_start)}"
        )

        # Spektrum varsa canlı spektrum widget'ine gönder
        spec = sv_data.get("spectrum")
        if spec and SVANTEK_OK:
            freqs  = spec.get("freqs",  [])
            levels = spec.get("levels", [])
            if freqs and levels:
                self.live_spectrum.update_svantek_spectrum(freqs, levels)

    def _on_error(self, msg):
        self._stop_stream(); QMessageBox.critical(self,"Mikrofon Hatası",msg[:600])

    def _set_status(self, msg): self.status_lbl.setText(msg)
    def _tick_clock(self): self.elapsed_lbl.setText(f"⏱  {fmt_elapsed(time.time()-self._clock_start)}")


# ══════════════════════════════════════════════════════════════════════════
#  ANA PENCERE
# ══════════════════════════════════════════════════════════════════════════

class _AnalysisWorker(QThread):
    progress = pyqtSignal(int, str); finished = pyqtSignal(dict); error = pyqtSignal(str)
    def __init__(self,system,path,pref,identify_subtype=False):
        super().__init__(); self.system=system; self.path=path; self.pref=pref
        self.identify_subtype=identify_subtype
    def run(self):
        try:
            self.progress.emit(5,"Ses dosyası yükleniyor…")
            result = self.system.analyze_for_gui(
                self.path, self.pref, identify_subtype=self.identify_subtype
            )
            result["_analysis_stage"] = (
                "subtype" if self.identify_subtype else "primary"
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback; self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    _MODEL_MAP = {0:"auto",1:"efficientnet",2:"cnn",3:"svm",4:"beats",5:"ensemble"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "✈  Havalimanı Gürültü Tespit Sistemi  v3.5"
            " — Shazam Öncelikli Hibrit Alt Tür"
        )
        self.setMinimumSize(1200,750); self.resize(1440,870)
        self._result=None; self._worker=None; self._audio_path=None

        if not DETECTOR_OK:
            QMessageBox.critical(self,"Import Hatası",
                f"noise_detector.py yüklenemedi:\n{_DETECTOR_ERR}"); sys.exit(1)

        self._status("Sistem başlatılıyor…")
        self.system    = AirportNoiseSystem(output_dir="outputs_gui")
        LIVE_CLIPS_DIR = r"D:\Airport_Live_Clips"
        self._clip_mgr = PendingClipManager(base_dir=LIVE_CLIPS_DIR)  # Staging yöneticisi
        self._build_ui(); self._status("Hazır.")

    def _build_ui(self): 
        
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(10,8,10,6); root.setSpacing(8)

        tb = QWidget(); tl = QHBoxLayout(tb); tl.setContentsMargins(0,0,0,0); tl.setSpacing(10)
        self.open_btn = QPushButton("📂  Dosya Aç"); self.open_btn.clicked.connect(self._open_file)
        self.file_lbl = QLabel("Dosya seçilmedi")
        self.file_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        self.file_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Preferred)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Otomatik (EfficientNet → CNN → SVM)",
            "EfficientNet-B0",
            "CNN",
            "SVM",
            "BEATs (Modern)",
            "Ensemble (EfficientNet + BEATs)",
        ])
        self.analyze_btn = QPushButton("①  Üst Sınıfı Bul")
        self.analyze_btn.setObjectName("analyzeBtn"); self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.shazam_btn = QPushButton("②  Shazam / Alt Türü Bul")
        self.shazam_btn.setEnabled(False)
        self.shazam_btn.setToolTip(
            "Kayıtlı parmak izini arar; eşleşme yoksa alt tür modelini kullanır."
        )
        self.shazam_btn.clicked.connect(self._run_subtype_analysis)
        tl.addWidget(self.open_btn); tl.addWidget(self.file_lbl)
        tl.addWidget(_label("Model:",muted=True))
        tl.addWidget(self.model_combo); tl.addWidget(self.analyze_btn)
        tl.addWidget(self.shazam_btn)
        root.addWidget(tb)

        self.progress = QProgressBar(); self.progress.setRange(0,100)
        self.progress.setValue(0); self.progress.setFixedHeight(5)
        self.progress.setTextVisible(False); root.addWidget(self.progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {PALETTE['border']}; }}")

        self.side = SidePanel(); self.side.export_btn.clicked.connect(self._export)
        splitter.addWidget(self.side)

        self.tabs = QTabWidget()

        # Sınıflandırma sekmesi — clip_manager ile oluştur
        self.tab_clf    = ClassificationTab(clip_manager=self._clip_mgr)
        self.tab_spec   = SpectrogramTab()
        self.tab_feat   = FeaturesTab()
        self.tab_review = DataReviewTab(self._clip_mgr)
        self.tab_mic    = MicrophoneTab(self.system, self._clip_mgr)

        self.tab_clf.clip_added.connect(self.tab_review.add_clip)
        self.tab_mic.clip_added.connect(self.tab_review.add_clip)

        self.tabs.addTab(self.tab_clf,    "📊  Sınıflandırma")
        self.tabs.addTab(self.tab_spec,   "🎨  Mel Spektrogram")
        self.tabs.addTab(self.tab_feat,   "📈  Özellikler")
        self.tabs.addTab(self.tab_mic,    "🎙  Canlı Mikrofon")
        self.tabs.addTab(self.tab_review, "📋  Veri Review")

        # Annotation → pending klip eklendi → Review tablosu güncelle
        self.tab_clf.clip_added.connect(self.tab_review.add_clip)

        if MICMAP_OK:
            self.tab_map = MapTab(self.system)
            self.tabs.addTab(self.tab_map,"🗺  Harita")

        splitter.addWidget(self.tabs)
        splitter.setSizes([220,1100]); splitter.setStretchFactor(1,1)
        root.addWidget(splitter); self.setStatusBar(QStatusBar())

    def _open_file(self):
        path,_ = QFileDialog.getOpenFileName(self,"Ses Dosyası Seç","",
            "Ses Dosyaları (*.wav *.mp3 *.flac *.ogg);;Tüm Dosyalar (*)")
        if not path: return
        self._audio_path = path
        self.file_lbl.setText(os.path.basename(path))
        self.file_lbl.setStyleSheet(f"color: {PALETTE['text']}; font-size: 11px;")
        self.analyze_btn.setEnabled(True)
        self.shazam_btn.setEnabled(False)
        self._result = None
        self.side._set_subtype_display("--", "Önce üst sınıf analizini çalıştırın")
        self._status(f"Hazır: {os.path.basename(path)}")

    def _run_analysis(self):
        if not self._audio_path: return
        self._start_analysis(identify_subtype=False)

    def _run_subtype_analysis(self):
        if not self._audio_path or not self._result: return
        self._start_analysis(identify_subtype=True)

    def _start_analysis(self, identify_subtype):
        self.analyze_btn.setEnabled(False); self.shazam_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.progress.setValue(5)
        self._status(
            "Shazam parmak izi ve alt tür modeli çalışıyor…"
            if identify_subtype else
            "Üst sınıf modeli çalışıyor…"
        )
        pref = self._MODEL_MAP[self.model_combo.currentIndex()]
        self._worker = _AnalysisWorker(
            self.system, self._audio_path, pref, identify_subtype
        )
        self._worker.progress.connect(lambda p,m: (self.progress.setValue(p), self._status(m)))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._anim_val=5; self._anim=QTimer()
        self._anim.timeout.connect(lambda:(
            setattr(self,"_anim_val",min(self._anim_val+1,90)),
            self.progress.setValue(self._anim_val)
        )); self._anim.start(80)

    def _on_finished(self, r):
        self._anim.stop(); self.progress.setValue(100); self._result=r
        stage = r.get("_analysis_stage", "primary")
        dominant = r.get("dominant") or (
            max(r["summary"], key=r["summary"].get)
            if r["summary"] else "--"
        )
        self.tab_clf.render(r); self.tab_spec.render(r); self.tab_feat.render(r)
        self.side.update_stats(r)
        if stage == "primary":
            self._status(
                f"✓ Üst sınıf: {dominant} | Model:{r['model_used']} | "
                "Alt tür için Shazam düğmesine basın."
            )
            self.shazam_btn.setEnabled(True)
        else:
            match = r.get("aircraft_match") or r.get("subtype_match") or {}
            method = match.get("method", "")
            if method in {"shazam", "shazam_v1"}:
                method_text = "Shazam parmak izi"
            elif method == "beats_multi_window_vote_v2_calibrated":
                method_text = "BEATs v2 alt tür modeli"
            elif method == "beats_multi_window_vote":
                method_text = "BEATs alt tür modeli"
            else:
                method_text = "Alt tür bulunamadı"
            self._status(f"✓ Üst sınıf: {dominant} | Sonuç kaynağı: {method_text}")
            self.shazam_btn.setEnabled(True)
        self.analyze_btn.setEnabled(True); self.open_btn.setEnabled(True)

    def _on_error(self, msg):
        if hasattr(self,"_anim"): self._anim.stop()
        self.progress.setValue(0)
        self.analyze_btn.setEnabled(True); self.open_btn.setEnabled(True)
        self.shazam_btn.setEnabled(self._result is not None)
        self._status("❌  Hata oluştu.")
        QMessageBox.critical(self,"Analiz Hatası",f"Hata:\n\n{msg[:800]}")

    def _export(self):
        if not self._result: return
        d = QFileDialog.getExistingDirectory(self,"Kayıt Klasörü",
                                              os.path.dirname(self._audio_path))
        if not d: return
        base = os.path.splitext(os.path.basename(self._audio_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S"); prefix=f"{base}_{ts}"
        r = self._result
        cp = os.path.join(d,f"{prefix}_classification.csv")
        with open(cp,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["window_start_s","label"]+r["class_names"])
            fp=r["frame_probs"]
            for i,(t,lbl) in enumerate(zip(r["frame_times"],r["frame_labels"])):
                prbs=list(fp[i]) if i<len(fp) else [""]*len(r["class_names"])
                w.writerow([f"{t:.2f}",lbl]+[f"{p:.4f}" for p in prbs])
        for fig,suf in [(self.tab_clf.fig1,"_strip.png"),(self.tab_clf.fig2,"_db.png"),
                         (self.tab_clf.fig3,"_confidence.png"),(self.tab_spec.fig,"_mel.png"),
                         (self.tab_feat.fig,"_features.png")]:
            fig.savefig(os.path.join(d,f"{prefix}{suf}"),dpi=150,
                        facecolor=PALETTE["bg"],bbox_inches="tight")
        self._status(f"✓  Export → {d}")
        QMessageBox.information(self,"Export Tamamlandı",
            f"Kaydedildi:\n{d}\n\nCSV: {prefix}_classification.csv\nPNG: {prefix}_*.png")

    def _status(self, msg): self.statusBar().showMessage(msg)


# ══════════════════════════════════════════════════════════════════════════
#  GİRİŞ NOKTASI
# ══════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    app.setFont(QFont("Segoe UI", 10))
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
