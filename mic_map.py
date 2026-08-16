"""
mic_map.py — Faz 3 + Faz 4: Çok Mikrofon Harita & Analiz Görünümü
════════════════════════════════════════════════════════════════════

Faz 3: Plan görünümü (tıkla-yerleştir) + OSM gerçek harita (contextily)
Faz 4: Her mikrofon için ayrı canlı analiz sekmesi

Ek kurulum:
    pip install contextily    ← OSM tile indirme (zorunlu değil, olmadan plan görünümü çalışır)
    pip install pyproj        ← Mercator→lat/lon dönüşümü (contextily ile gelir)
"""

from __future__ import annotations

import os
import sys
import time
import queue
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDialog, QDialogButtonBox, QLineEdit, QFrame,
    QScrollArea, QSplitter, QTabWidget, QFileDialog, QMessageBox,
    QSizePolicy, QColorDialog, QDoubleSpinBox, QFormLayout,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt6.QtGui import QFont, QColor, QPainter, QLinearGradient, QBrush, QPen

import matplotlib
matplotlib.use("QtAgg")
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


# ══════════════════════════════════════════════════════════════════════════
#  RENK PALETİ
# ══════════════════════════════════════════════════════════════════════════

PALETTE = {
    "bg":      "#0D1117",
    "surface": "#161B22",
    "border":  "#30363D",
    "text":    "#E6EDF3",
    "muted":   "#8B949E",
    "accent":  "#00D4FF",
    "green":   "#7EE8A2",
    "yellow":  "#FFE66D",
    "red":     "#FF4444",
}

CLASS_COLORS = {
    "AIRCRAFT": "#FF6B35",
    "AMBIENT":  "#7EE8A2",
    "SPEECH":   "#A8DADC",
    "TRAFFIC":  "#FFE66D",
    "WIND":     "#4ECDC4",
    "UNKNOWN":  "#6C757D",
    "---":      "#6C757D",
}

MIC_PALETTE = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8DADC", "#C77DFF"]


# ══════════════════════════════════════════════════════════════════════════
#  PAYLAŞILAN CANLI ANALİZ WİDGET'LARI
#  (gui_main.py ile aynı mantık — döngüsel import önlemek için buraya kopyalandı)
# ══════════════════════════════════════════════════════════════════════════

import collections as _col

def _make_canvas(fig_h=3.0, fig_w=8.0):
    fig = Figure(figsize=(fig_w, fig_h), facecolor=PALETTE["bg"])
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return fig, canvas

def _style_ax(ax):
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.title.set_color(PALETTE["text"])
    ax.grid(True, color=PALETTE["border"], linewidth=0.5, alpha=0.6)


class VUMeter(QWidget):
    """Gradient VU meter + peak hold."""
    def __init__(self):
        super().__init__()
        self._db = self._peak = -80.0; self._hold = 0
        self.setFixedHeight(22); self.setMinimumWidth(120)
        t = QTimer(self); t.timeout.connect(self._tick); t.start(40)

    def set_db(self, db: float):
        self._db = max(-80.0, min(0.0, db))
        if self._db > self._peak: self._peak = self._db; self._hold = 40
        self.update()

    def reset(self): self._db = self._peak = -80.0; self._hold = 0; self.update()

    def _tick(self):
        if self._hold > 0: self._hold -= 1
        else: self._peak = max(self._peak - 0.3, self._db)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, m = self.width(), self.height(), 3
        p.fillRect(0, 0, w, h, QColor(PALETTE["surface"]))
        p.setPen(QPen(QColor(PALETTE["border"]))); p.drawRoundedRect(0, 0, w-1, h-1, 3, 3)
        bw = int((self._db + 80) / 80 * (w - 2*m))
        grad = QLinearGradient(m, 0, w-m, 0)
        grad.setColorAt(0.00, QColor("#7EE8A2"))
        grad.setColorAt(0.65, QColor("#FFE66D"))
        grad.setColorAt(1.00, QColor("#FF4444"))
        p.fillRect(m, m, bw, h-2*m, QBrush(grad))
        if self._peak > -79:
            px = m + int((self._peak+80)/80*(w-2*m))
            p.setPen(QPen(QColor("white"), 2)); p.drawLine(px, m, px, h-m)
        p.setPen(QPen(QColor(PALETTE["border"]), 1))
        for tick in [-60, -40, -20, -6]:
            tx = m + int((tick+80)/80*(w-2*m)); p.drawLine(tx, h-4, tx, h-1)
        p.end()


class RollingClassStrip(QWidget):
    """Son HISTORY saniyelik sınıflandırmayı renk bantlarıyla çizer."""
    HISTORY = 60
    def __init__(self):
        super().__init__()
        self._hist: _col.deque[str] = _col.deque(maxlen=self.HISTORY)
        self.setMinimumHeight(44); self.setMaximumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def add_label(self, label: str): self._hist.append(label); self.update()
    def clear(self): self._hist.clear(); self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PALETTE["surface"]))
        p.setPen(QPen(QColor(PALETTE["border"]))); p.drawRect(0, 0, w-1, h-1)
        n = len(self._hist)
        if n == 0:
            p.setPen(QPen(QColor(PALETTE["muted"]))); p.setFont(QFont("Segoe UI", 9))
            p.drawText(QRect(0,0,w,h), Qt.AlignmentFlag.AlignCenter, "Dinleniyor…")
            p.end(); return
        sw = w / self.HISTORY
        p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        for i, lbl in enumerate(self._hist):
            x = int(w - (n-i)*sw); cw = max(1, int(sw)-1)
            p.fillRect(x, 2, cw, h-4, QColor(CLASS_COLORS.get(lbl, "#6C757D")))
            if sw > 28:
                p.setPen(QPen(QColor("white")))
                p.drawText(QRect(x,2,cw,h-4), Qt.AlignmentFlag.AlignCenter, lbl[:3])
        p.setPen(QPen(QColor(PALETTE["border"]), 1)); p.setFont(QFont("Segoe UI", 7))
        for t in range(10, self.HISTORY+1, 10):
            x = int(w - t*sw)
            if 0 < x < w:
                p.drawLine(x, h-8, x, h)
                p.setPen(QPen(QColor(PALETTE["muted"])))
                p.drawText(QRect(x-15,h-8,30,8), Qt.AlignmentFlag.AlignCenter, f"-{t}s")
                p.setPen(QPen(QColor(PALETTE["border"]), 1))
        p.end()


class DbHistoryChart(QWidget):
    """Son 60 saniyelik dBFS + sınıf renk arka planı."""
    HISTORY = 60
    def __init__(self):
        super().__init__()
        self._db:   _col.deque[float] = _col.deque(maxlen=self.HISTORY)
        self._lbls: _col.deque[str]   = _col.deque(maxlen=self.HISTORY)
        self.setMinimumHeight(110)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.fig, self.canvas = _make_canvas(fig_h=1.5, fig_w=7)
        lay.addWidget(self.canvas)

    def add(self, db: float, label: str):
        self._db.append(db); self._lbls.append(label); self._redraw()

    def clear(self):
        self._db.clear(); self._lbls.clear(); self._redraw()

    def _redraw(self):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        _style_ax(ax); self.fig.patch.set_facecolor(PALETTE["bg"])
        n = len(self._db)
        if n == 0:
            ax.set_xlim(-self.HISTORY,0); ax.set_ylim(-80,0)
            self.fig.tight_layout(pad=0.3); self.canvas.draw(); return
        xs = np.arange(-n+1, 1); ys = np.array(self._db); lbls = list(self._lbls)
        for i, lbl in enumerate(lbls):
            ax.axvspan(xs[i]-0.5, xs[i]+0.5, alpha=0.15,
                       color=CLASS_COLORS.get(lbl,"#6C757D"), lw=0)
        ax.plot(xs, ys, color=PALETTE["accent"], lw=1.2, zorder=5)
        ax.fill_between(xs, ys, -80, color=PALETTE["accent"], alpha=0.10, zorder=4)
        ax.set_xlim(-self.HISTORY, 0)
        ax.set_ylim(min(-60,np.min(ys)-5), max(-10,np.max(ys)+5))
        ax.set_xlabel("Saniye önce", fontsize=7); ax.set_ylabel("dBFS", fontsize=7)
        ax.set_title("dBFS Geçmişi", color=PALETTE["text"], fontsize=8, pad=2)
        self.fig.tight_layout(pad=0.3); self.canvas.draw()


class ConfidenceChart(QWidget):
    """Son 60 saniyelik softmax olasılık çizgileri."""
    HISTORY = 60
    def __init__(self, class_names: list[str]):
        super().__init__()
        self._cn    = class_names
        self._probs: dict[str, _col.deque] = {
            c: _col.deque(maxlen=self.HISTORY) for c in class_names
        }
        self._times: _col.deque[int] = _col.deque(maxlen=self.HISTORY)
        self._t = 0
        self.setMinimumHeight(130)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.fig, self.canvas = _make_canvas(fig_h=1.8, fig_w=7)
        lay.addWidget(self.canvas)

    def add(self, probs: dict):
        self._t += 1; self._times.append(self._t)
        for c in self._cn:
            self._probs[c].append(probs.get(c, 0.0))
        self._redraw()

    def clear(self):
        for c in self._cn: self._probs[c].clear()
        self._times.clear(); self._t = 0; self._redraw()

    def _redraw(self):
        self.fig.clear(); ax = self.fig.add_subplot(111)
        _style_ax(ax); self.fig.patch.set_facecolor(PALETTE["bg"])
        n = len(self._times)
        if n == 0:
            ax.set_xlim(-self.HISTORY,0); ax.set_ylim(0,1)
            ax.set_title("Softmax Güven",color=PALETTE["text"],fontsize=8,pad=2)
            self.fig.tight_layout(pad=0.3); self.canvas.draw(); return
        xs = np.arange(-n+1, 1)
        for c in self._cn:
            ys = np.array(self._probs[c])
            if len(ys) == n:
                ax.plot(xs, ys, color=CLASS_COLORS.get(c,"#8B949E"),
                        lw=1.3, label=c, alpha=0.85)
        ax.set_xlim(-self.HISTORY, 0); ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color=PALETTE["border"], lw=0.7, linestyle="--")
        ax.set_xlabel("Saniye önce",fontsize=7); ax.set_ylabel("Olasılık",fontsize=7)
        ax.set_title("Softmax Güven",color=PALETTE["text"],fontsize=8,pad=2)
        ax.legend(fontsize=7, facecolor=PALETTE["surface"],
                  labelcolor=PALETTE["text"], loc="upper left",
                  framealpha=0.8, ncol=3)
        self.fig.tight_layout(pad=0.3); self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════
#  MİKROFON ANALİZ SEKMESİ (Faz 4)
#  Her mikrofon için ayrı, canlı analiz görünümü.
#  MapTab.view_tabs'a dinamik olarak eklenir/kaldırılır.
# ══════════════════════════════════════════════════════════════════════════

class MicAnalysisTab(QWidget):
    """
    Tek bir mikrofonun canlı analiz görünümü:
      Sol  → Büyük sınıf badge + VU meter + güven barları
      Sağ  → Renk şeridi (60s) + dBFS grafiği + Softmax grafiği
    """

    def __init__(self, cfg: MicConfig):
        super().__init__()
        self._cfg        = cfg
        self._class_names: list[str] = []  # ilk result gelince doldurulur
        self._conf_chart: ConfidenceChart | None = None
        self._build_ui(cfg)

    def _build_ui(self, cfg: MicConfig):
        root = QHBoxLayout(self); root.setContentsMargins(8,8,8,8); root.setSpacing(8)

        # ── Sol panel ────────────────────────────────────────────
        left = QFrame()
        left.setStyleSheet(f"""
            QFrame {{
                background: {PALETTE['surface']};
                border: 2px solid {cfg.color};
                border-radius: 8px;
            }}
        """)
        left.setFixedWidth(210)
        ll = QVBoxLayout(left); ll.setContentsMargins(12,12,12,12); ll.setSpacing(10)

        # Başlık
        title = QLabel(f"● {cfg.name}")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {cfg.color}; border: none;")
        ll.addWidget(title)

        # Büyük badge
        self.badge = QLabel("---")
        self.badge.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setStyleSheet(f"color: {PALETTE['muted']}; border: none;")
        self.badge.setMinimumHeight(60)
        ll.addWidget(self.badge)

        # dBFS metin
        self.db_lbl = QLabel("-∞  dBFS")
        self.db_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.db_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 13px; border: none;")
        ll.addWidget(self.db_lbl)

        # VU meter
        self.vu = VUMeter(); ll.addWidget(self.vu)

        # Ayırıcı
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {PALETTE['border']}; border: none;"); ll.addWidget(sep)

        # Güven barları
        lbl_conf = QLabel("CANLI GÜVEN")
        lbl_conf.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px; "
                                f"font-weight: 600; letter-spacing: 1px; border: none;")
        ll.addWidget(lbl_conf)

        self.conf_bars: dict[str, tuple[QProgressBar, QLabel]] = {}
        for cls, color in CLASS_COLORS.items():
            if cls in ("UNKNOWN", "---"): continue
            row = QWidget(); rl = QHBoxLayout(row)
            rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)
            name = QLabel(cls); name.setFixedWidth(62)
            name.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            bar = QProgressBar(); bar.setRange(0,100); bar.setValue(0)
            bar.setTextVisible(False); bar.setFixedHeight(6)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {PALETTE['border']}; border-radius: 3px; border: none; }}
                QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}
            """)
            pct = QLabel("0%"); pct.setFixedWidth(32)
            pct.setAlignment(Qt.AlignmentFlag.AlignRight)
            pct.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px;")
            rl.addWidget(name); rl.addWidget(bar); rl.addWidget(pct)
            ll.addWidget(row)
            self.conf_bars[cls] = (bar, pct)

        ll.addStretch()
        root.addWidget(left)

        # ── Sağ panel ────────────────────────────────────────────
        right = QWidget(); rl = QVBoxLayout(right)
        rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)

        # Renk şeridi + legend
        rl.addWidget(_lbl("SON 60 SANİYE — SINIFLANDIRMA ŞERİDİ", muted=True, small=True))
        self.strip = RollingClassStrip(); rl.addWidget(self.strip)

        leg = QWidget(); legl = QHBoxLayout(leg)
        legl.setContentsMargins(0,0,0,0); legl.setSpacing(12)
        for cls, color in CLASS_COLORS.items():
            if cls in ("UNKNOWN","---"): continue
            d = QLabel(f"● {cls}")
            d.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 600;")
            legl.addWidget(d)
        legl.addStretch(); rl.addWidget(leg)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {PALETTE['border']};"); rl.addWidget(sep2)

        # dBFS grafiği
        rl.addWidget(_lbl("SES SEVİYESİ — dBFS", muted=True, small=True))
        self.db_chart = DbHistoryChart(); rl.addWidget(self.db_chart)

        # Softmax grafiği — class_names ilk result gelince oluşturulur
        self._conf_placeholder = QLabel("Softmax güven grafiği inference başlayınca görünür.")
        self._conf_placeholder.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        self._conf_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(self._conf_placeholder)

        self._conf_slot = rl      # chart buraya eklenecek
        self._conf_added = False

        rl.addStretch()
        root.addWidget(right)

    # ─────────────────────────────────────────────────────────────────────

    def update_result(self, res: dict):
        label  = res.get("label", "---")
        probs  = res.get("probs", {})
        db_rms = res.get("db_rms", -80.0)

        # Badge
        color = CLASS_COLORS.get(label, PALETTE["muted"])
        self.badge.setText(label)
        self.badge.setStyleSheet(f"color: {color}; font-size: 22px; "
                                  f"font-weight: 700; border: none;")
        # dBFS label
        col_db = (PALETTE["red"] if db_rms > -10 else
                  PALETTE["yellow"] if db_rms > -30 else PALETTE["green"])
        self.db_lbl.setText(f"{db_rms:+.1f}  dBFS")
        self.db_lbl.setStyleSheet(f"color: {col_db}; font-size: 13px; "
                                   f"font-weight: 600; border: none;")
        # Güven barları
        for cls, (bar, pct) in self.conf_bars.items():
            v = int(probs.get(cls, 0.0) * 100)
            bar.setValue(v); pct.setText(f"{v}%")

        # Şerit + dBFS chart
        self.strip.add_label(label)
        self.db_chart.add(db_rms, label)

        # Softmax chart (ilk result'ta sınıf isimleri belli olur)
        if not self._conf_added and probs:
            self._class_names = sorted(probs.keys())
            self.conf_chart   = ConfidenceChart(self._class_names)
            self._conf_slot.removeWidget(self._conf_placeholder)
            self._conf_placeholder.deleteLater()
            # stretch'ten önce ekle
            self._conf_slot.insertWidget(self._conf_slot.count() - 1,
                                          self.conf_chart)
            self._conf_added = True

        if self._conf_added:
            self.conf_chart.add(probs)

    def update_vu(self, db: float):
        self.vu.set_db(db)

    def reset(self):
        self.badge.setText("---")
        self.badge.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 22px; border: none;")
        self.db_lbl.setText("-∞  dBFS")
        self.vu.reset()
        self.strip.clear()
        self.db_chart.clear()
        if self._conf_added:
            self.conf_chart.clear()
        for bar, pct in self.conf_bars.values():
            bar.setValue(0); pct.setText("0%")


# ══════════════════════════════════════════════════════════════════════════
#  VERİ SINIFI
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MicConfig:
    mic_id:     int
    name:       str
    device_idx: int
    color:      str

    pixel_x: float = 0.0
    pixel_y: float = 0.0
    placed:  bool  = False

    lat:     float = 0.0
    lon:     float = 0.0
    has_geo: bool  = False

    current_label: str   = "---"
    current_probs: dict  = field(default_factory=dict)
    current_db:    float = -80.0


# ══════════════════════════════════════════════════════════════════════════
#  DEVICE MULTIPLEXER
#  Aynı USB cihazı birden fazla mikrofona dağıtır.
#  sounddevice bir cihazı yalnızca bir kez açabilir — bu sınıf bu kısıtı
#  aşar: tek stream açılır, gelen chunk'lar tüm abone queue'larına kopyalanır.
# ══════════════════════════════════════════════════════════════════════════

class DeviceMultiplexer:
    SR         = 22050
    BLOCK_SIZE = 1024

    def __init__(self, device_idx: int):
        self._device      = device_idx
        self._subscribers: list[queue.Queue] = []
        self._stream      = None
        self._lock        = threading.Lock()

    def subscribe(self) -> queue.Queue:
        """Yeni bir abone queue'su oluştur ve döndür."""
        q = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def start(self):
        if not SOUNDDEVICE_OK:
            return

        def callback(indata, frames, time_info, status):
            if status:
                pass  # over/underflow — sessizce geç
            chunk = indata[:, 0].copy()
            with self._lock:
                for q in self._subscribers:
                    try:
                        q.put_nowait(chunk)
                    except queue.Full:
                        pass  # yavaş consumer — drop

        self._stream = sd.InputStream(
            samplerate=self.SR,
            channels=1,
            dtype="float32",
            blocksize=self.BLOCK_SIZE,
            device=self._device,
            callback=callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


# ══════════════════════════════════════════════════════════════════════════
#  TEK MİKROFON WORKER
#  Artık cihazı kendisi açmıyor; DeviceMultiplexer'dan queue alıyor.
# ══════════════════════════════════════════════════════════════════════════

class SingleMicWorker(QThread):
    result_signal = pyqtSignal(int, dict)   # (mic_id, {label, probs, db_rms})
    vu_signal     = pyqtSignal(int, float)  # (mic_id, db)
    error_signal  = pyqtSignal(int, str)

    SR             = 22050
    WINDOW_SAMPLES = int(5.0 * 22050)
    SLIDE_SAMPLES  = int(1.0 * 22050)
    VU_INTERVAL    = int(0.04 * 22050)

    def __init__(self, system, mic_id: int,
                 audio_queue: queue.Queue, model_pref: str):
        super().__init__()
        self._system     = system
        self._mic_id     = mic_id
        self._queue      = audio_queue
        self._model_pref = model_pref
        self._stop_flag  = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        buffer                  = np.zeros(self.WINDOW_SAMPLES, dtype=np.float32)
        samples_since_inference = 0
        samples_since_vu        = 0

        while not self._stop_flag:
            try:
                chunk = self._queue.get(timeout=0.15)
            except queue.Empty:
                continue

            n = len(chunk)
            buffer = np.roll(buffer, -n)
            buffer[-n:] = chunk

            # VU ~40 ms
            samples_since_vu += n
            if samples_since_vu >= self.VU_INTERVAL:
                samples_since_vu = 0
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                self.vu_signal.emit(self._mic_id,
                                    float(20 * np.log10(rms + 1e-10)))

            # Inference ~1 s
            samples_since_inference += n
            if samples_since_inference >= self.SLIDE_SAMPLES:
                samples_since_inference = 0
                try:
                    res = self._system.classify_chunk_live(
                        buffer.copy(), self._model_pref)
                    self.result_signal.emit(self._mic_id, res)
                except Exception as e:
                    print(f"[MIC-{self._mic_id}] {e}")


# ══════════════════════════════════════════════════════════════════════════
#  MİKROFON EKLEME DİALOGU
# ══════════════════════════════════════════════════════════════════════════

class AddMicDialog(QDialog):
    def __init__(self, mic_id: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mikrofon Ekle")
        self.setMinimumWidth(420)
        self.setStyleSheet(f"""
            QDialog, QWidget {{
                background: {PALETTE['surface']};
                color: {PALETTE['text']};
                font-size: 13px;
            }}
            QLineEdit, QComboBox, QDoubleSpinBox {{
                background: {PALETTE['bg']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                padding: 6px; border-radius: 4px;
            }}
            QLabel {{ color: {PALETTE['text']}; }}
            QPushButton {{
                background: {PALETTE['bg']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                padding: 6px 16px; border-radius: 4px;
            }}
        """)
        self._color = MIC_PALETTE[mic_id % len(MIC_PALETTE)]
        layout = QVBoxLayout(self)
        form   = QFormLayout()

        self.name_edit = QLineEdit(f"MIC-{mic_id + 1}")
        form.addRow("İsim:", self.name_edit)

        self.device_combo = QComboBox()
        self._device_map  = []
        if SOUNDDEVICE_OK:
            try:
                devs       = sd.query_devices()
                default_in = sd.default.device[0]
                for i, dev in enumerate(devs):
                    if dev["max_input_channels"] > 0:
                        tag = " [Varsayılan]" if i == default_in else ""
                        self.device_combo.addItem(f"{dev['name']}{tag}")
                        self._device_map.append(i)
            except Exception:
                pass
        if not self._device_map:
            self.device_combo.addItem("Cihaz bulunamadı")
        form.addRow("Ses Cihazı:", self.device_combo)

        note_same = QLabel(
            "💡 Aynı cihazı birden fazla mikrofona atayabilirsiniz.\n"
            "   Ses akışı otomatik olarak paylaşılır."
        )
        note_same.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px;")
        form.addRow("", note_same)

        self.color_btn = QPushButton()
        self._refresh_color_btn()
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("Renk:", self.color_btn)

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90, 90)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(0.0)
        form.addRow("Enlem (Lat):", self.lat_spin)

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180, 180)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(0.0)
        form.addRow("Boylam (Lon):", self.lon_spin)

        geo_note = QLabel("↑ Doldurursan OSM haritasında da görünür (0,0 = atla)")
        geo_note.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px;")
        form.addRow("", geo_note)

        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._refresh_color_btn()

    def _refresh_color_btn(self):
        self.color_btn.setStyleSheet(
            f"background-color: {self._color}; border: 1px solid #555; "
            f"min-height: 24px; border-radius: 4px;")
        self.color_btn.setText(self._color)

    def result_config(self, mic_id: int) -> MicConfig:
        ci  = self.device_combo.currentIndex()
        dev = self._device_map[ci] if ci < len(self._device_map) else 0
        lat = self.lat_spin.value()
        lon = self.lon_spin.value()
        return MicConfig(
            mic_id=mic_id,
            name=self.name_edit.text() or f"MIC-{mic_id + 1}",
            device_idx=dev,
            color=self._color,
            lat=lat, lon=lon,
            has_geo=(lat != 0.0 or lon != 0.0),
        )


# ══════════════════════════════════════════════════════════════════════════
#  MİKROFON KARTI (sol liste)
# ══════════════════════════════════════════════════════════════════════════

class MicCard(QFrame):
    place_requested  = pyqtSignal(int)
    remove_requested = pyqtSignal(int)

    def __init__(self, cfg: MicConfig):
        super().__init__()
        self.mic_id = cfg.mic_id
        self.setStyleSheet(f"""
            QFrame {{
                background: {PALETTE['bg']};
                border: 2px solid {cfg.color};
                border-radius: 8px; margin: 2px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(4)

        # Başlık
        header = QHBoxLayout()
        dot  = QLabel("●")
        dot.setStyleSheet(f"color: {cfg.color}; font-size: 16px;")
        name = QLabel(cfg.name)
        name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        name.setStyleSheet(f"color: {cfg.color}; border: none;")
        header.addWidget(dot); header.addWidget(name); header.addStretch()

        rm = QPushButton("✕"); rm.setFixedSize(22, 22)
        rm.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {PALETTE['muted']};
                           border: none; font-size: 12px; }}
            QPushButton:hover {{ color: {PALETTE['red']}; }}
        """)
        rm.clicked.connect(lambda: self.remove_requested.emit(self.mic_id))
        header.addWidget(rm)
        lay.addLayout(header)

        # Sınıf etiketi
        self.label_lbl = QLabel("---")
        self.label_lbl.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.label_lbl.setStyleSheet(f"color: {PALETTE['muted']}; border: none;")
        self.label_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.label_lbl)

        # dBFS
        self.db_lbl = QLabel("-∞ dBFS")
        self.db_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px; border: none;")
        self.db_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.db_lbl)

        # Konum satırı
        pos_row = QHBoxLayout()
        self.pos_lbl = QLabel("Konumlandırılmadı")
        self.pos_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 10px; border: none;")
        pos_row.addWidget(self.pos_lbl); pos_row.addStretch()

        place_btn = QPushButton("📍 Yerleştir"); place_btn.setFixedHeight(22)
        place_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: 1px solid {PALETTE['border']};
                color: {PALETTE['muted']}; font-size: 10px;
                border-radius: 3px; padding: 1px 6px;
            }}
            QPushButton:hover {{ border-color: {cfg.color}; color: {cfg.color}; }}
        """)
        place_btn.clicked.connect(lambda: self.place_requested.emit(self.mic_id))
        pos_row.addWidget(place_btn)
        lay.addLayout(pos_row)

    def update_state(self, label: str, db: float):
        color = CLASS_COLORS.get(label, PALETTE["muted"])
        self.label_lbl.setText(label)
        self.label_lbl.setStyleSheet(
            f"color: {color}; font-size: 15px; font-weight: 700; border: none;")
        col_db = (PALETTE["red"] if db > -10 else
                  PALETTE["yellow"] if db > -30 else PALETTE["green"])
        self.db_lbl.setText(f"{db:+.1f} dBFS")
        self.db_lbl.setStyleSheet(f"color: {col_db}; font-size: 11px; border: none;")

    def update_position(self, x: float, y: float):
        self.pos_lbl.setText(f"({x:.0f}, {y:.0f})")


# ══════════════════════════════════════════════════════════════════════════
#  PLAN CANVAS (matplotlib)
# ══════════════════════════════════════════════════════════════════════════

class PlanCanvas(QWidget):
    mic_placed = pyqtSignal(int, float, float)

    def __init__(self):
        super().__init__()
        self._image = None
        self._mics: dict[int, MicConfig] = {}
        self._placing_id: int | None = None

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.fig    = Figure(facecolor=PALETTE["bg"])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self._timer = QTimer()
        self._timer.timeout.connect(self.refresh)
        self._timer.start(500)
        self.refresh()

    def load_image(self, path: str):
        try:
            import matplotlib.image as mpimg
            self._image = mpimg.imread(path)
            self.refresh()
        except Exception as e:
            QMessageBox.warning(None, "Görsel Hatası", str(e))

    def set_mics(self, mics: dict[int, MicConfig]):
        self._mics = mics

    def start_placement(self, mic_id: int):
        self._placing_id = mic_id
        self.canvas.setStyleSheet(f"border: 2px solid {PALETTE['accent']};")
        self.refresh()

    def _on_click(self, event):
        if self._placing_id is None or event.inaxes is None: return
        px, py = event.xdata, event.ydata
        if px is None or py is None: return
        self.mic_placed.emit(self._placing_id, float(px), float(py))
        self._placing_id = None
        self.canvas.setStyleSheet("")

    def refresh(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(PALETTE["surface"])
        self.fig.patch.set_facecolor(PALETTE["bg"])

        if self._image is None:
            ax.text(0.5, 0.5,
                    "📂  Havalimanı plan görselini yükleyin\n(PNG / JPG / BMP)",
                    color=PALETTE["muted"], ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
            self.fig.tight_layout(pad=0.2)
            self.canvas.draw(); return

        H, W = self._image.shape[:2]
        ax.imshow(self._image, extent=[0, W, H, 0], aspect="equal")
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])

        for mic in self._mics.values():
            if not mic.placed: continue
            lc = CLASS_COLORS.get(mic.current_label, mic.color)
            outer = __import__("matplotlib.pyplot", fromlist=["Circle"]).Circle(
                (mic.pixel_x, mic.pixel_y), W * 0.018,
                color=lc, alpha=0.35, zorder=5)
            inner = __import__("matplotlib.pyplot", fromlist=["Circle"]).Circle(
                (mic.pixel_x, mic.pixel_y), W * 0.010,
                color=mic.color, zorder=6)
            ax.add_patch(outer); ax.add_patch(inner)
            ax.annotate(
                f"{mic.name}\n{mic.current_label}\n{mic.current_db:+.0f} dB",
                xy=(mic.pixel_x, mic.pixel_y),
                xytext=(mic.pixel_x + W * 0.025, mic.pixel_y - H * 0.04),
                fontsize=7, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=mic.color,
                          alpha=0.85, edgecolor="white", lw=0.8),
                arrowprops=dict(arrowstyle="->", color=mic.color, lw=1.0),
                zorder=7,
            )

        if self._placing_id is not None:
            n = self._mics[self._placing_id].name if self._placing_id in self._mics else "?"
            ax.text(0.5, 0.02,
                    f"📍  '{n}' için harita üzerine tıklayın",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color=PALETTE["accent"],
                    bbox=dict(boxstyle="round", facecolor=PALETTE["surface"],
                              alpha=0.9, edgecolor=PALETTE["accent"]))

        self.fig.tight_layout(pad=0.1)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════════════
#  OSM GÖRÜNÜMÜ
# ══════════════════════════════════════════════════════════════════════════

class TileMapView(QWidget):
    """
    Gerçek OSM harita görünümü — folium/WebEngine yok, saf matplotlib.

    contextily kütüphanesi belirtilen bbox için OSM tile'larını indirir,
    matplotlib imshow ile çizer. Tıklayarak mikrofon konumu ayarlanabilir
    (Plan Canvas ile aynı mekanik — lat/lon cinsinden).

    Kurulum: pip install contextily
    """

    mic_placed_geo = pyqtSignal(int, float, float)  # (mic_id, lat, lon)

    # Zoom seviyesi 1-19 arası; yüksek = daha detaylı ama daha yavaş indirme
    DEFAULT_ZOOM  = 15
    CACHE_DIR     = os.path.join(tempfile.gettempdir(), "airport_osm_cache")

    def __init__(self):
        super().__init__()
        self._mics:       dict[int, MicConfig] = {}
        self._placing_id: int | None           = None
        self._tile_img    = None   # son indirilen tile array
        self._tile_extent = None   # (west, east, south, north) — Web Mercator
        self._zoom        = self.DEFAULT_ZOOM

        # contextily kontrol
        try:
            import contextily as ctx
            self._ctx = ctx
            self._ctx_ok = True
        except ImportError:
            self._ctx = None
            self._ctx_ok = False

        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        # ── Araç çubuğu ─────────────────────────────────────────
        bar = QWidget()
        bar.setStyleSheet(f"background: {PALETTE['surface']}; border-bottom: 1px solid {PALETTE['border']};")
        bl  = QHBoxLayout(bar); bl.setContentsMargins(8,4,8,4); bl.setSpacing(8)

        bl.addWidget(_lbl("Merkez Lat:", muted=True, small=True))
        self.lat_edit = QDoubleSpinBox()
        self.lat_edit.setRange(-90, 90); self.lat_edit.setDecimals(6)
        self.lat_edit.setValue(41.0082)   # İstanbul Atatürk
        self.lat_edit.setFixedWidth(110)

        bl.addWidget(_lbl("Lon:", muted=True, small=True))
        self.lon_edit = QDoubleSpinBox()
        self.lon_edit.setRange(-180, 180); self.lon_edit.setDecimals(6)
        self.lon_edit.setValue(28.7453)
        self.lon_edit.setFixedWidth(110)

        bl.addWidget(_lbl("Zoom:", muted=True, small=True))
        self.zoom_combo = QComboBox()
        for z, lbl_z in [(13,"Geniş Çevre"),(14,"Mahalle"),(15,"Detay ✓"),(16,"Yüksek"),(17,"Çok Yüksek")]:
            self.zoom_combo.addItem(f"{z} — {lbl_z}", z)
        self.zoom_combo.setCurrentIndex(2)  # 15

        self.fetch_btn = QPushButton("🗺  Haritayı Yükle")
        self.fetch_btn.clicked.connect(self._fetch_tiles)
        if not self._ctx_ok:
            self.fetch_btn.setEnabled(False)
            self.fetch_btn.setToolTip("pip install contextily")

        bl.addWidget(self.lat_edit); bl.addWidget(self.lon_edit)
        bl.addWidget(self.zoom_combo); bl.addWidget(self.fetch_btn)
        bl.addStretch()

        if not self._ctx_ok:
            warn = QLabel("⚠  pip install contextily  gerekli")
            warn.setStyleSheet(f"color: {PALETTE['yellow']}; font-size: 11px;")
            bl.addWidget(warn)

        lay.addWidget(bar)

        # ── Harita canvas ────────────────────────────────────────
        self.fig    = Figure(facecolor=PALETTE["bg"])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        # Güncelleme timer (marker state değişince)
        self._marker_timer = QTimer()
        self._marker_timer.timeout.connect(self._redraw_markers_only)
        self._marker_timer.start(1000)

        self._draw_empty()

        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────

    def set_mics(self, mics: dict[int, MicConfig]):
        self._mics = mics

    def start_placement(self, mic_id: int):
        self._placing_id = mic_id
        self.canvas.setStyleSheet(f"border: 2px solid {PALETTE['accent']};")
        self._redraw_markers_only()

    def _on_click(self, event):
        if self._placing_id is None or event.inaxes is None: return
        if event.xdata is None or event.ydata is None: return

        # Web Mercator koordinatını lat/lon'a çevir
        if self._ctx_ok and self._tile_extent is not None:
            try:
                import pyproj
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:3857", "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(event.xdata, event.ydata)
            except Exception:
                # pyproj yoksa basit dönüşüm
                import math
                lon = event.xdata / 20037508.34 * 180
                lat = math.degrees(
                    math.atan(math.sinh(math.pi * event.ydata / 20037508.34)))
        else:
            # Tile yüklü değilse ham koordinatı kullan
            lon, lat = event.xdata, event.ydata

        self.mic_placed_geo.emit(self._placing_id, float(lat), float(lon))
        self._placing_id = None
        self.canvas.setStyleSheet("")
        self._redraw_markers_only()

    # ─────────────────────────────────────────────────────────────────────

    def _fetch_tiles(self):
        """Merkez lat/lon etrafındaki OSM tile'larını indir ve göster."""
        if not self._ctx_ok:
            QMessageBox.warning(self, "contextily Eksik",
                "pip install contextily\nkomutuyla yükleyin."); return

        lat  = self.lat_edit.value()
        lon  = self.lon_edit.value()
        zoom = self.zoom_combo.currentData()

        # bbox: merkez ± delta derece (yaklaşık)
        delta = {13: 0.05, 14: 0.025, 15: 0.012, 16: 0.006, 17: 0.003}
        d = delta.get(zoom, 0.012)
        w, e, s, n = lon-d, lon+d, lat-d, lat+d

        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("⏳ İndiriliyor…")
        QApplication_processEvents()

        try:
            img, ext = self._ctx.bounds2img(
                w, s, e, n, zoom=zoom,
                source=self._ctx.providers.OpenStreetMap.Mapnik,
                ll=True,   # girdi lat/lon cinsinden
            )
            self._tile_img    = img
            self._tile_extent = ext   # (west, east, south, north) Web Mercator
            self._zoom        = zoom
            self._draw_map()
            self._set_status_external(f"✓ Harita yüklendi — zoom {zoom}")
        except Exception as e:
            QMessageBox.critical(self, "Tile İndirme Hatası",
                f"OSM tile'ları indirilemedi:\n{e}\n\n"
                "İnternet bağlantısını kontrol edin.")
        finally:
            self.fetch_btn.setEnabled(True)
            self.fetch_btn.setText("🗺  Haritayı Yükle")

    # ─────────────────────────────────────────────────────────────────────

    def _draw_map(self):
        """Tile + marker'ları tam olarak çiz."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["surface"])

        if self._tile_img is not None and self._tile_extent is not None:
            ax.imshow(self._tile_img,
                      extent=self._tile_extent,
                      origin="upper",
                      interpolation="bilinear",
                      aspect="equal")
            ax.set_xlim(self._tile_extent[0], self._tile_extent[1])
            ax.set_ylim(self._tile_extent[2], self._tile_extent[3])
        else:
            ax.text(0.5, 0.5,
                    "📍  Lat/Lon girin ve 'Haritayı Yükle'ye tıklayın\n"
                    "(İnternet bağlantısı gerekir — OSM tiles)",
                    color=PALETTE["muted"], ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])

        self._draw_markers(ax)
        self._draw_placement_hint(ax)

        self.fig.tight_layout(pad=0.1)
        self.canvas.draw()

    def _redraw_markers_only(self):
        """Tile tekrar indirilmeden sadece marker'ları güncelle."""
        self._draw_map()

    def _draw_markers(self, ax):
        import math

        def _to_mercator(lat, lon):
            x = lon * 20037508.34 / 180
            y = math.log(math.tan((90 + lat) * math.pi / 360)) / math.pi
            return x, y * 20037508.34

        geo_mics = [m for m in self._mics.values() if m.has_geo]
        if not geo_mics or self._tile_extent is None: return

        ext = self._tile_extent  # (W, E, S, N) Mercator
        span_x = ext[1] - ext[0]
        span_y = ext[3] - ext[2]

        for mic in geo_mics:
            mx, my = _to_mercator(mic.lat, mic.lon)
            lc = CLASS_COLORS.get(mic.current_label, mic.color)
            r  = span_x * 0.012

            outer = __import__("matplotlib.pyplot", fromlist=["Circle"]).Circle(
                (mx, my), r, color=lc, alpha=0.35, zorder=5)
            inner = __import__("matplotlib.pyplot", fromlist=["Circle"]).Circle(
                (mx, my), r * 0.55, color=mic.color, zorder=6)
            ax.add_patch(outer); ax.add_patch(inner)

            ax.annotate(
                f"{mic.name}\n{mic.current_label}\n{mic.current_db:+.0f} dB",
                xy=(mx, my),
                xytext=(mx + span_x * 0.02, my + span_y * 0.03),
                fontsize=7, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=mic.color,
                          alpha=0.88, edgecolor="white", lw=0.8),
                arrowprops=dict(arrowstyle="->", color=mic.color, lw=1.0),
                zorder=7,
            )

    def _draw_placement_hint(self, ax):
        if self._placing_id is None: return
        name = self._mics[self._placing_id].name \
               if self._placing_id in self._mics else "?"
        ax.text(0.5, 0.02,
                f"📍  '{name}' için harita üzerine tıklayın",
                transform=ax.transAxes, ha="center", fontsize=10,
                color=PALETTE["accent"],
                bbox=dict(boxstyle="round", facecolor=PALETTE["surface"],
                          alpha=0.92, edgecolor=PALETTE["accent"]))

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["surface"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
        ax.text(0.5, 0.5,
                "📍  Lat/Lon girin ve 'Haritayı Yükle'ye tıklayın\n"
                "(pip install contextily  gerekli)",
                color=PALETTE["muted"], ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        self.fig.tight_layout(pad=0.1)
        self.canvas.draw()

    def _set_status_external(self, msg: str):
        # MapTab.status_lbl — parent zincirinden ulaşmak yerine sinyal kullanılabilir
        # Şimdilik sessiz geçiyoruz; MapTab kendi status'unu yönetiyor
        pass


def QApplication_processEvents():
    """Buton text güncellenmesi için event loop'u bir kez döndür."""
    from PyQt6.QtWidgets import QApplication
    QApplication.processEvents()





# ══════════════════════════════════════════════════════════════════════════
#  ANA HARITA SEKMESİ
# ══════════════════════════════════════════════════════════════════════════

class MapTab(QWidget):
    MODEL_MAP = {
        "Otomatik (EfficientNet → CNN → SVM)": "auto",
        "EfficientNet-B0":                      "efficientnet",
        "CNN":                                  "cnn",
        "SVM":                                  "svm",
    }

    def __init__(self, system):
        super().__init__()
        self._system      = system
        self._mics:    dict[int, MicConfig]            = {}
        self._workers: dict[int, SingleMicWorker]      = {}
        self._muxes:   dict[int, DeviceMultiplexer]    = {}
        self._queues:  dict[int, queue.Queue]           = {}
        self._cards:   dict[int, MicCard]               = {}
        self._atabs:   dict[int, MicAnalysisTab]        = {}   # Faz 4
        self._next_id  = 0
        self._running  = False
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8); root.setSpacing(8)

        # ── Araç çubuğu ─────────────────────────────────────────
        tb = QWidget(); tl = QHBoxLayout(tb)
        tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(8)

        self.load_btn = QPushButton("🗺  Plan Yükle (PNG/JPG)")
        self.load_btn.clicked.connect(self._load_plan)

        self.add_btn = QPushButton("➕  Mikrofon Ekle")
        self.add_btn.clicked.connect(self._add_mic)

        self.model_combo = QComboBox()
        for k in self.MODEL_MAP: self.model_combo.addItem(k)

        self.start_btn = QPushButton("▶  Tümünü Başlat")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(self._btn_style(PALETTE["green"], "#1a4a2e"))
        self.start_btn.clicked.connect(self._start_all)

        self.stop_btn = QPushButton("⏹  Tümünü Durdur")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self._btn_style(PALETTE["red"], "#4a1a1a"))
        self.stop_btn.clicked.connect(self._stop_all)

        for w in [self.load_btn, self.add_btn,
                  _lbl("Model:", muted=True), self.model_combo,
                  self.start_btn, self.stop_btn]:
            tl.addWidget(w)
        tl.addStretch()
        root.addWidget(tb)

        # ── Splitter: sol liste | sağ görünümler ─────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {PALETTE['border']}; }}")

        # Sol — mikrofon listesi
        left = QWidget(); left.setFixedWidth(248)
        ll   = QVBoxLayout(left); ll.setContentsMargins(0, 0, 4, 0); ll.setSpacing(4)
        ll.addWidget(_lbl("MİKROFONLAR", muted=True, small=True))

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        self._list_w   = QWidget()
        self._list_lay = QVBoxLayout(self._list_w)
        self._list_lay.setContentsMargins(0, 0, 0, 0); self._list_lay.setSpacing(6)
        self._empty_lbl = QLabel(
            "Henüz mikrofon eklenmedi.\n\n"
            "➕ butonuna tıklayarak\nekleyin."
        )
        self._empty_lbl.setStyleSheet(f"color: {PALETTE['muted']}; font-size: 11px;")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._list_lay.addWidget(self._empty_lbl)
        self._list_lay.addStretch()
        scroll.setWidget(self._list_w)
        ll.addWidget(scroll)
        splitter.addWidget(left)

        # Sağ — Plan + OSM sekmeleri
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        self.view_tabs    = QTabWidget()
        self.plan_canvas  = PlanCanvas()
        self.plan_canvas.mic_placed.connect(self._on_mic_placed)
        self.tile_map     = TileMapView()
        self.tile_map.mic_placed_geo.connect(self._on_mic_placed_geo)
        self.view_tabs.addTab(self.plan_canvas, "🏢  Plan Görünümü")
        self.view_tabs.addTab(self.tile_map,    "🌍  OSM Harita")
        rl.addWidget(self.view_tabs)
        splitter.addWidget(right)

        splitter.setSizes([248, 1000]); splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # Durum çubuğu
        self.status_lbl = QLabel("Plan görselini yükleyin, ardından mikrofon ekleyin.")
        self.status_lbl.setStyleSheet(f"""
            background: {PALETTE['surface']}; color: {PALETTE['muted']};
            border-top: 1px solid {PALETTE['border']};
            padding: 4px 10px; font-size: 11px;
        """)
        root.addWidget(self.status_lbl)

    @staticmethod
    def _btn_style(color: str, bg: str) -> str:
        return (f"QPushButton {{ background: {bg}; border: 1px solid {color}; "
                f"color: {color}; font-weight: 700; "
                f"padding: 7px 18px; border-radius: 6px; }}"
                f"QPushButton:hover {{ background: {bg}AA; }}"
                f"QPushButton:disabled {{ background: {PALETTE['surface']}; "
                f"border-color: {PALETTE['border']}; color: {PALETTE['muted']}; }}")

    # ─────────────────────────────────────────────────────────────────────

    def _load_plan(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Plan Görselini Seç", "",
            "Görsel (*.png *.jpg *.jpeg *.bmp *.tif);;Tüm (*)")
        if not path: return
        self.plan_canvas.load_image(path)
        self._set_status(f"Plan yüklendi: {os.path.basename(path)}")

    # ─────────────────────────────────────────────────────────────────────

    def _add_mic(self):
        if len(self._mics) >= 5:
            QMessageBox.information(self, "Limit", "Maksimum 5 mikrofon.")
            return
        dlg = AddMicDialog(self._next_id, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted: return

        cfg = dlg.result_config(self._next_id)
        self._next_id += 1
        self._mics[cfg.mic_id] = cfg

        card = MicCard(cfg)
        card.place_requested.connect(self._activate_placement)
        card.remove_requested.connect(self._remove_mic)
        self._cards[cfg.mic_id] = card

        idx = self._list_lay.count() - 1  # stretch'ten önce
        self._list_lay.insertWidget(idx, card)
        self._empty_lbl.hide()

        # Faz 4 — analiz sekmesi
        atab = MicAnalysisTab(cfg)
        self._atabs[cfg.mic_id] = atab
        self.view_tabs.addTab(atab, f"📡  {cfg.name}")

        self.plan_canvas.set_mics(self._mics)
        self.tile_map.set_mics(self._mics)
        self._update_start_btn()
        self._set_status(
            f"'{cfg.name}' eklendi — aynı cihaz birden fazla mic'e atanabilir. "
            f"📍 Yerleştir butonuyla plan veya OSM harita üzerinde konumlandırın.")

    def _remove_mic(self, mic_id: int):
        # Worker durdur
        if mic_id in self._workers:
            w = self._workers.pop(mic_id)
            w.stop(); w.wait(2000)
        # Queue'yu mux'tan çıkar
        if mic_id in self._queues:
            q = self._queues.pop(mic_id)
            dev = self._mics[mic_id].device_idx if mic_id in self._mics else -1
            if dev in self._muxes:
                self._muxes[dev].unsubscribe(q)
        # Kart
        if mic_id in self._cards:
            card = self._cards.pop(mic_id)
            self._list_lay.removeWidget(card); card.deleteLater()
        # Faz 4 — analiz sekmesi kaldır
        if mic_id in self._atabs:
            atab = self._atabs.pop(mic_id)
            idx  = self.view_tabs.indexOf(atab)
            if idx >= 0:
                self.view_tabs.removeTab(idx)
            atab.deleteLater()
        self._mics.pop(mic_id, None)
        if not self._mics: self._empty_lbl.show()
        self.plan_canvas.set_mics(self._mics)
        self.tile_map.set_mics(self._mics)
        self._update_start_btn()
        self._set_status("Mikrofon kaldırıldı.")

    def _activate_placement(self, mic_id: int):
        """Plan veya OSM sekmesinde konumlandırma modunu başlat."""
        n = self._mics[mic_id].name if mic_id in self._mics else "?"
        current = self.view_tabs.currentIndex()
        if current == 1:
            # OSM sekmesi aktifse orada başlat
            self.tile_map.start_placement(mic_id)
            self._set_status(f"📍 OSM haritasında '{n}' için konum seçin…")
        else:
            # Varsayılan: plan görünümü
            self.view_tabs.setCurrentIndex(0)
            self.plan_canvas.start_placement(mic_id)
            self._set_status(f"📍 Plan görünümünde '{n}' için konum seçin…")

    def _on_mic_placed_geo(self, mic_id: int, lat: float, lon: float):
        """OSM haritasından gelen lat/lon konum bilgisi."""
        if mic_id not in self._mics: return
        self._mics[mic_id].lat     = lat
        self._mics[mic_id].lon     = lon
        self._mics[mic_id].has_geo = True
        self.tile_map.set_mics(self._mics)
        name = self._mics[mic_id].name
        self._set_status(
            f"'{name}' OSM haritasında konumlandırıldı "
            f"(lat={lat:.5f}, lon={lon:.5f})")

    def _on_mic_placed(self, mic_id: int, px: float, py: float):
        if mic_id not in self._mics: return
        self._mics[mic_id].pixel_x = px
        self._mics[mic_id].pixel_y = py
        self._mics[mic_id].placed  = True
        if mic_id in self._cards:
            self._cards[mic_id].update_position(px, py)
        self.plan_canvas.set_mics(self._mics)
        self._set_status(
            f"'{self._mics[mic_id].name}' konumlandırıldı "
            f"({px:.0f}, {py:.0f}) — ▶ Tümünü Başlat ile inference'ı başlatın.")
        self._update_start_btn()

    # ─────────────────────────────────────────────────────────────────────

    def _update_start_btn(self):
        self.start_btn.setEnabled(bool(self._mics) and not self._running)

    def _start_all(self):
        if not SOUNDDEVICE_OK:
            QMessageBox.warning(self, "sounddevice Eksik",
                "pip install sounddevice"); return

        model_pref = self.MODEL_MAP[self.model_combo.currentText()]

        # Cihaz başına tek Multiplexer
        for mic_id, cfg in self._mics.items():
            if mic_id in self._workers: continue

            dev = cfg.device_idx

            # Mux yoksa oluştur ve başlat
            if dev not in self._muxes:
                mux = DeviceMultiplexer(dev)
                try:
                    mux.start()
                except Exception as e:
                    QMessageBox.critical(self, "Cihaz Hatası",
                        f"Cihaz {dev} açılamadı:\n{e}"); continue
                self._muxes[dev] = mux

            # Worker için ayrı queue
            q = self._muxes[dev].subscribe()
            self._queues[mic_id] = q

            w = SingleMicWorker(self._system, mic_id, q, model_pref)
            w.result_signal.connect(self._on_result)
            w.vu_signal.connect(self._on_vu)
            w.error_signal.connect(self._on_error)
            w.start()
            self._workers[mic_id] = w

        self._running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        n_mics = len(self._mics)
        n_devs = len(self._muxes)
        self._set_status(
            f"🎙  {n_mics} mikrofon aktif — {n_devs} fiziksel cihaz — dinleniyor…")

    def _stop_all(self):
        for w in self._workers.values(): w.stop()
        for w in self._workers.values(): w.wait(3000)
        self._workers.clear()
        self._queues.clear()
        for mux in self._muxes.values(): mux.stop()
        self._muxes.clear()
        self._running = False
        # Analiz sekmelerini sıfırla
        for atab in self._atabs.values(): atab.reset()
        self.stop_btn.setEnabled(False)
        self.add_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self._update_start_btn()
        self._set_status("⏹  Tüm mikrofonlar durduruldu.")

    # ─────────────────────────────────────────────────────────────────────

    def _on_result(self, mic_id: int, res: dict):
        if mic_id not in self._mics: return
        self._mics[mic_id].current_label = res.get("label", "---")
        self._mics[mic_id].current_db    = res.get("db_rms", -80.0)
        self._mics[mic_id].current_probs = res.get("probs", {})
        if mic_id in self._cards:
            self._cards[mic_id].update_state(
                self._mics[mic_id].current_label,
                self._mics[mic_id].current_db)
        # Faz 4 — analiz sekmesini güncelle
        if mic_id in self._atabs:
            self._atabs[mic_id].update_result(res)

    def _on_vu(self, mic_id: int, db: float):
        # Faz 4 — VU meter
        if mic_id in self._atabs:
            self._atabs[mic_id].update_vu(db)

    def _on_error(self, mic_id: int, msg: str):
        n = self._mics[mic_id].name if mic_id in self._mics else str(mic_id)
        QMessageBox.critical(self, f"Hata — {n}", msg[:500])
        self._remove_mic(mic_id)

    def _set_status(self, msg: str):
        self.status_lbl.setText(msg)


# ─────────────────────────────────────────────────────────────────────────

def _lbl(text, muted=False, small=False):
    l = QLabel(text)
    l.setStyleSheet(f"color: {PALETTE['muted'] if muted else PALETTE['text']}; "
                    f"font-size: {'10px' if small else '13px'};")
    return l
