"""
Airport Noise — Edge Device Firmware
=====================================
Cihaz açıldığı anda çalışır. Mikrofonu sürekli dinler;
anlamlı bir ses yakalandığında:
  1. .wav olarak diske kaydeder
  2. SQLite veritabanına loglar
  3. Merkez sunucuya HTTP POST ile iletir

Bağımlılıklar: torch, torchaudio, sounddevice, scipy, numpy, requests
Kullanım    : python firmware.py
              python firmware.py --model-dir D:/models --no-telemetry
"""

import argparse
import collections
import datetime
import logging
import os
import queue
import sqlite3
import sys
import threading
import time

import numpy as np
import requests
import sounddevice as sd
import torch
import torch.nn as nn
import torchaudio

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR   = r"D:\models"
DEFAULT_BEATS_CKPT  = "BEATs_iter3_plus_AS2M.pt"
DEFAULT_MLP_CKPT    = "beats_mlp.pt"
DEFAULT_AUDIO_DIR   = "recordings"          # .wav dosyaları buraya
DEFAULT_DB_PATH     = "device_data.db"
DEFAULT_CENTER_URL  = "http://localhost:8080/api/telemetry"

SAMPLE_RATE         = 22050                 # eğitimle aynı
WINDOW_SEC          = 5                     # inference penceresi (saniye)
HOP_SEC             = 1                     # her N saniyede yeni tahmin
CONFIDENCE_THR      = 0.75                  # altında UNKNOWN (yüksek eşik = az yanlış pozitif)
MAJORITY_LEN        = 7                     # son N tahminin çoğunluğu
MAJORITY_MIN_VOTES  = 5                     # 7 tahmin içinde 5'i aynı olmalı
ENSEMBLE_ALPHA      = 0.5                   # (ilerisi için; şimdi yalnız BEATs)

# RMS enerji eşiği — bu değerin altındaki pencereler sessiz kabul edilir,
# inference'a bile sokulmaz. Fan/elektronik gürültüsünü filtreler.
# 0.002 tipik bir ofis ortamı için iyi başlangıç noktası.
# Çok hassas → artır (0.005), ses kaçırıyor → azalt (0.001).
RMS_THRESHOLD       = 0.002

# Bir olay kaydedildikten sonra kaç saniye beklenir (aynı sesi tekrar tetiklememek için)
COOLDOWN_SEC        = 12

AMBIENT_LABELS      = {"AMBIENT", "UNKNOWN", "OTHER"}  # bu çıkarsa kaydetme
CLASSES             = ["AIRCRAFT", "AMBIENT", "TRAFFIC", "SPEECH", "WIND", "OTHER"]

TELEMETRY_TIMEOUT   = 5                     # saniye; sunucu yoksa beklemez

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("firmware.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("firmware")


# ---------------------------------------------------------------------------
# BEATs yardımcıları  (noise_detector.py'deki yapıyla birebir uyumlu)
# ---------------------------------------------------------------------------

def _build_beats_encoder(ckpt_path: str, device: torch.device):
    """
    BEATs encoder'ı yükler, frozen olarak döndürür.
    noise_detector.py'deki yükleme mantığıyla aynı.
    """
    # Proje kökündeki BEATs.py dosyası import edilmeli.
    # edge_device/ alt klasöründen çalıştırılıyorsa parent dizini path'e eklenir.
    try:
        from BEATs import BEATs, BEATsConfig as _BEATsConfig
    except ImportError:
        import pathlib
        project_root = str(pathlib.Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        try:
            from BEATs import BEATs, BEATsConfig as _BEATsConfig
        except ImportError:
            log.error(
                "BEATs.py bulunamadı.\n"
                "  Çözüm 1 — Proje kökünden çalıştır : python edge_device/firmware.py\n"
                "  Çözüm 2 — edge_device/ içinden    : python firmware.py\n"
                "  Çözüm 3 — BEATs.py'yi edge_device/ klasörüne kopyala"
            )
            sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = _BEATsConfig(ckpt["cfg"])   # projenin kendi BEATsConfig'i — default'ları biliyor
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    log.info(f"BEATs encoder yüklendi: {ckpt_path}")
    return model


def _build_mlp(mlp_path: str, device: torch.device) -> nn.Sequential:
    """768 → 256 → 6 MLP'yi yükler."""
    mlp = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASSES)),
    )
    ckpt = torch.load(mlp_path, map_location=device)
    state = ckpt.get("model_state", ckpt)   # noise_detector.py ile aynı mantık
    mlp.load_state_dict(state)
    mlp.eval()
    mlp.to(device)
    log.info(f"BEATs MLP yüklendi: {mlp_path}")
    return mlp


# ---------------------------------------------------------------------------
# Veritabanı
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            label     TEXT    NOT NULL,
            confidence REAL   NOT NULL,
            wav_path  TEXT    NOT NULL,
            sent      INTEGER DEFAULT 0   -- merkeze iletildi mi?
        )
    """)
    conn.commit()
    log.info(f"Veritabanı hazır: {db_path}")
    return conn


def db_insert(conn: sqlite3.Connection,
              timestamp: str, label: str,
              confidence: float, wav_path: str) -> int:
    cur = conn.execute(
        "INSERT INTO events (timestamp, label, confidence, wav_path) VALUES (?,?,?,?)",
        (timestamp, label, round(confidence, 4), wav_path),
    )
    conn.commit()
    return cur.lastrowid


def db_mark_sent(conn: sqlite3.Connection, row_id: int):
    conn.execute("UPDATE events SET sent=1 WHERE id=?", (row_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# Ses işleme
# ---------------------------------------------------------------------------

def preprocess_waveform(waveform: np.ndarray,
                        sr: int,
                        target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    numpy array → mono float32 tensor, yeniden örnekleme.
    sounddevice çıktısı zaten float32 [samples, channels].
    """
    wav = torch.from_numpy(waveform.copy()).float()
    if wav.ndim == 2:
        wav = wav.mean(dim=1)           # stereo → mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # BEATs beklentisi: [1, samples]
    return wav.unsqueeze(0)


@torch.no_grad()
def classify(wav_tensor: torch.Tensor,
             encoder, mlp,
             device: torch.device) -> tuple[str, float]:
    """
    Waveform tensoru → (etiket, güven).
    """
    wav = wav_tensor.to(device)
    # BEATs encoder: padding_mask olmadan da çalışır
    padding_mask = torch.zeros(wav.shape[0], wav.shape[1],
                               dtype=torch.bool, device=device)
    embeddings, _ = encoder.extract_features(wav, padding_mask=padding_mask)
    # embeddings: [batch, T, 768] → zaman ortalaması → [batch, 768]
    emb = embeddings.mean(dim=1)
    logits  = mlp(emb)                          # [1, 6]
    probs   = torch.softmax(logits, dim=-1)[0]  # [6]
    top_idx = probs.argmax().item()
    confidence = probs[top_idx].item()
    label = CLASSES[top_idx] if confidence >= CONFIDENCE_THR else "UNKNOWN"
    return label, confidence


# ---------------------------------------------------------------------------
# Telemetry (merkeze gönderim)
# ---------------------------------------------------------------------------

def send_telemetry(center_url: str, row_id: int,
                   timestamp: str, label: str,
                   confidence: float, wav_path: str) -> bool:
    """
    Ses dosyasını ve metaveriyi merkez sunucuya POST eder.
    Başarısızlık fatal değil; sadece log'a düşer.
    """
    try:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                center_url,
                data={
                    "device_id":  "EDGE-01",
                    "event_id":   row_id,
                    "timestamp":  timestamp,
                    "label":      label,
                    "confidence": confidence,
                },
                files={"audio": (os.path.basename(wav_path), f, "audio/wav")},
                timeout=TELEMETRY_TIMEOUT,
            )
        if resp.status_code == 200:
            log.info(f"[TELEMETRY] Merkeze iletildi → event_id={row_id}")
            return True
        else:
            log.warning(f"[TELEMETRY] Sunucu yanıtı: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        log.warning("[TELEMETRY] Merkez sunucuya ulaşılamadı (bağlantı hatası).")
    except Exception as exc:
        log.warning(f"[TELEMETRY] Hata: {exc}")
    return False


# ---------------------------------------------------------------------------
# RMS Otomatik Kalibrasyon
# ---------------------------------------------------------------------------

def calibrate_rms(sample_rate: int = SAMPLE_RATE,
                  duration_sec: int = 4,
                  device_index=None,
                  multiplier: float = 3.0) -> float:
    """
    Mikrofonu `duration_sec` saniye sessizce dinler, ortam gürültü tabanını
    ölçer ve RMS eşiğini otomatik ayarlar.
    multiplier=3.0 → gürültü tabanının 3 katı üstü "anlamlı ses"
    """
    log.info(f"[KALİBRASYON] {duration_sec} saniye sessiz ortam ölçümü yapılıyor…")
    log.info("[KALİBRASYON] Lütfen bu süre boyunca konuşmayın.")

    frames = []

    # Cihazın native rate'ini kullan (WASAPI uyumluluğu)
    try:
        dev_info   = sd.query_devices(device_index, kind="input")
        capture_sr = int(dev_info["default_samplerate"])
    except Exception:
        capture_sr = 48000
    log.info(f"[KALİBRASYON] Capture rate: {capture_sr} Hz")

    def _cb(indata, n, t, status):
        frames.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=capture_sr, channels=1, dtype="float32",
                        blocksize=capture_sr // 4, device=device_index,
                        callback=_cb):
        import time as _time
        _time.sleep(duration_sec)

    if not frames:
        log.warning("[KALİBRASYON] Ses verisi alınamadı, varsayılan eşik kullanılıyor.")
        return RMS_THRESHOLD

    all_audio = np.concatenate(frames)
    noise_rms  = float(np.sqrt(np.mean(all_audio ** 2)))
    threshold  = round(noise_rms * multiplier, 5)

    log.info(f"[KALİBRASYON] Ortam RMS={noise_rms:.5f} → Eşik={threshold:.5f} "
             f"(çarpan={multiplier}x)")

    # Çok düşük çıkarsa minimum güvenli değeri kullan
    threshold = max(threshold, 0.0005)
    return threshold


# ---------------------------------------------------------------------------
# Mikrofon işçisi
# ---------------------------------------------------------------------------

class MicrophoneWorker(threading.Thread):
    """
    sounddevice ile mikrofonu sürekli dinler.
    Her HOP_SEC saniyede yeni bir WINDOW_SEC'lik pencere hazırlar
    ve inference kuyruğuna koyar.

    WASAPI gibi native-rate API'lerinde cihazın kendi sample rate'i
    kullanılır, veri inference'a girmeden önce 22050 Hz'e resample edilir.
    """

    def __init__(self, audio_queue: queue.Queue,
                 target_sample_rate: int = SAMPLE_RATE,
                 window_sec: int  = WINDOW_SEC,
                 hop_sec: int     = HOP_SEC,
                 device_index: int | None = None):
        super().__init__(daemon=True, name="MicrophoneWorker")
        self.audio_queue       = audio_queue
        self.target_sr         = target_sample_rate
        self.device_index      = device_index
        self.window_sec        = window_sec
        self.hop_sec           = hop_sec
        self._stop_evt         = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def _detect_device_rate(self) -> int:
        """Cihazın desteklediği native sample rate'i sorgular."""
        try:
            info = sd.query_devices(self.device_index, kind="input")
            native = int(info["default_samplerate"])
            log.info(f"Cihaz native sample rate: {native} Hz")
            return native
        except Exception:
            log.warning("Native sample rate alınamadı, 48000 varsayılıyor.")
            return 48000

    def run(self):
        capture_sr   = self._detect_device_rate()
        needs_resamp = (capture_sr != self.target_sr)

        # Buffer ve pencere boyutları capture_sr cinsinden
        window_size_cap = self.window_sec * capture_sr
        hop_size_cap    = self.hop_sec    * capture_sr

        # Hedef SR cinsinden pencere boyutu (resample sonrası)
        window_size_tgt = self.window_sec * self.target_sr

        _buffer = collections.deque(maxlen=window_size_cap)

        log.info(f"Mikrofon dinlemeye başladı "
                 f"(capture={capture_sr}Hz → target={self.target_sr}Hz, "
                 f"window={self.window_sec}s, hop={self.hop_sec}s, "
                 f"resample={'EVET' if needs_resamp else 'HAYIR'})")

        def callback(indata, frames, time_info, status):
            if status:
                log.debug(f"sounddevice: {status}")
            _buffer.extend(indata[:, 0].tolist())
            if len(_buffer) >= window_size_cap:
                raw = np.array(list(_buffer), dtype=np.float32)

                # Gerekirse resample et
                if needs_resamp:
                    wav_t  = torch.from_numpy(raw).unsqueeze(0)
                    wav_t  = torchaudio.functional.resample(
                                wav_t, capture_sr, self.target_sr)
                    window = wav_t.squeeze(0).numpy()
                else:
                    window = raw

                if self.audio_queue.full():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.audio_queue.put(window)

        # WASAPI veya WDM-KS başaramazsa DirectSound'a düş
        for attempt_device in [self.device_index, None]:
            try:
                with sd.InputStream(
                    samplerate=capture_sr,
                    channels=1,
                    dtype="float32",
                    blocksize=hop_size_cap,
                    device=attempt_device,
                    callback=callback,
                ):
                    if attempt_device != self.device_index:
                        log.warning(f"Sistem varsayılan cihaza düşüldü.")
                    while not self._stop_evt.is_set():
                        time.sleep(0.1)
                break   # başarıyla çalıştı, döngüden çık
            except Exception as exc:
                if attempt_device == self.device_index:
                    log.warning(f"Cihaz {attempt_device} açılamadı: {exc}")
                    log.warning("Sistem varsayılan cihazla yeniden deneniyor…")
                    # Varsayılan cihazın rate'ini al
                    try:
                        dev_info   = sd.query_devices(None, kind="input")
                        capture_sr = int(dev_info["default_samplerate"])
                        hop_size_cap = self.hop_sec * capture_sr
                        needs_resamp = (capture_sr != self.target_sr)
                    except Exception:
                        pass
                else:
                    log.error(f"Varsayılan cihaz da açılamadı: {exc}")

        log.info("Mikrofon durduruldu.")


# ---------------------------------------------------------------------------
# Inference + kayıt döngüsü
# ---------------------------------------------------------------------------

class InferenceWorker(threading.Thread):
    """
    Ses kuyruğundan pencereleri alır, sınıflandırır,
    anlamlı ses varsa kaydeder ve merkeze iletir.
    """

    def __init__(self, audio_queue: queue.Queue,
                 encoder, mlp,
                 device: torch.device,
                 db_conn: sqlite3.Connection,
                 audio_dir: str,
                 center_url: str | None,
                 no_telemetry: bool = False,
                 rms_threshold: float = RMS_THRESHOLD):
        super().__init__(daemon=True, name="InferenceWorker")
        self.audio_queue    = audio_queue
        self.encoder        = encoder
        self.mlp            = mlp
        self.device         = device
        self.db_conn        = db_conn
        self.audio_dir      = audio_dir
        self.center_url     = center_url
        self.no_telemetry   = no_telemetry
        self.rms_threshold  = rms_threshold
        self._majority_buf  = collections.deque(maxlen=MAJORITY_LEN)
        self._stop_evt      = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def _majority_label(self) -> str:
        if not self._majority_buf:
            return "UNKNOWN"
        counts = collections.Counter(self._majority_buf)
        return counts.most_common(1)[0][0]

    def _save_wav(self, waveform: np.ndarray,
                  timestamp_str: str, label: str) -> str:
        safe_ts  = timestamp_str.replace(":", "-").replace(" ", "_")
        filename = f"rec_{safe_ts}_{label}.wav"
        path     = os.path.join(self.audio_dir, filename)
        wav_t    = torch.from_numpy(waveform).float().unsqueeze(0)
        torchaudio.save(path, wav_t, SAMPLE_RATE)
        return path

    def run(self):
        log.info("Inference işçisi başladı.")
        last_event_time = 0.0   # cooldown için

        while not self._stop_evt.is_set():
            try:
                window = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # ---- 1. RMS enerji filtresi ----
            rms = float(np.sqrt(np.mean(window ** 2)))
            if rms < self.rms_threshold:
                log.debug(f"[RMS] Sessiz pencere atlandı (rms={rms:.5f})")
                # Sessiz pencere → buffer'a AMBIENT bas, kararlılığı bozma
                self._majority_buf.append("AMBIENT")
                continue

            # ---- 2. Model çıkarımı ----
            wav_tensor = preprocess_waveform(window, SAMPLE_RATE)
            label, confidence = classify(wav_tensor, self.encoder,
                                         self.mlp, self.device)
            self._majority_buf.append(label)

            # ---- 3. Majority voting — minimum oy şartıyla ----
            counts = collections.Counter(self._majority_buf)
            top_label, top_votes = counts.most_common(1)[0]

            log.info(f"[INFERENCE] Ham={label} ({confidence:.2f}) "
                     f"| rms={rms:.4f} "
                     f"| Majority={top_label} ({top_votes}/{len(self._majority_buf)})")

            # Buffer henüz yeterince dolmadıysa veya üstün etiket AMBIENT/UNKNOWN ise geç
            if top_votes < MAJORITY_MIN_VOTES:
                continue
            if top_label in AMBIENT_LABELS:
                continue

            # ---- 4. Cooldown kontrolü ----
            now = time.time()
            if now - last_event_time < COOLDOWN_SEC:
                remaining = COOLDOWN_SEC - (now - last_event_time)
                log.debug(f"[COOLDOWN] {remaining:.1f}s kaldı, atlanıyor.")
                continue

            # ---- 5. Olay kaydet ----
            last_event_time = now
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.info(f"[OLAY] {top_label} yakalandı — kaydediliyor… "
                     f"(oy={top_votes}/{len(self._majority_buf)}, güven={confidence:.2f})")

            wav_path = self._save_wav(window, timestamp, top_label)
            log.info(f"[KAYIT] {wav_path}")

            row_id = db_insert(self.db_conn, timestamp,
                               top_label, confidence, wav_path)

            if not self.no_telemetry and self.center_url:
                ok = send_telemetry(self.center_url, row_id,
                                    timestamp, top_label,
                                    confidence, wav_path)
                if ok:
                    db_mark_sent(self.db_conn, row_id)

            # Buffer'ı temizle — yeni olay döngüsü başsın
            self._majority_buf.clear()

        log.info("Inference işçisi durduruldu.")


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Airport Noise — Edge Device Firmware")
    p.add_argument("--model-dir",    default=DEFAULT_MODEL_DIR,
                   help="BEATs encoder ve MLP ağırlıklarının bulunduğu klasör")
    p.add_argument("--beats-ckpt",   default=DEFAULT_BEATS_CKPT,
                   help="BEATs encoder checkpoint dosya adı")
    p.add_argument("--mlp-ckpt",     default=DEFAULT_MLP_CKPT,
                   help="BEATs MLP checkpoint dosya adı")
    p.add_argument("--audio-dir",    default=DEFAULT_AUDIO_DIR,
                   help="Yakalanan .wav dosyalarının kaydedileceği klasör")
    p.add_argument("--db-path",      default=DEFAULT_DB_PATH,
                   help="SQLite veritabanı dosyası")
    p.add_argument("--center-url",   default=DEFAULT_CENTER_URL,
                   help="Merkez sunucu telemetry endpoint'i")
    p.add_argument("--no-telemetry", action="store_true",
                   help="Merkeze gönderimi devre dışı bırak")
    p.add_argument("--device",       type=int, default=None,
                   help="sounddevice giriş cihazı indeksi (varsayılan: sistem varsayılanı)")
    p.add_argument("--cpu",          action="store_true",
                   help="GPU olsa bile CPU kullan")
    p.add_argument("--list-devices", action="store_true",
                   help="Mevcut ses cihazlarını listele ve çık")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Otomatik RMS kalibrasyonunu atla, sabit eşik kullan")
    p.add_argument("--rms-multiplier", type=float, default=3.0,
                   help="Kalibrasyon çarpanı: gürültü_tabanı × N = eşik (varsayılan: 3.0)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        print("\n=== Mevcut Ses Cihazları ===")
        devices = sd.query_devices()
        print(devices)
        default_in = sd.query_devices(kind="input")
        print(f"\n→ Varsayılan GİRİŞ cihazı: [{default_in['index'] if hasattr(default_in,'__getitem__') and 'index' in default_in else '?'}] {default_in['name']}")
        print("\nFirmware'i belirli bir cihazla başlatmak için: python firmware.py --device <indeks>")
        sys.exit(0)

    # ---- Klasörler ----
    os.makedirs(args.audio_dir, exist_ok=True)
    log.info(f"Kayıt klasörü: {os.path.abspath(args.audio_dir)}")

    # ---- Cihaz ----
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    log.info(f"Torch cihazı: {device}")

    # ---- Modeller ----
    beats_path = os.path.join(args.model_dir, args.beats_ckpt)
    mlp_path   = os.path.join(args.model_dir, args.mlp_ckpt)

    if not os.path.exists(beats_path):
        log.error(f"BEATs checkpoint bulunamadı: {beats_path}")
        sys.exit(1)
    if not os.path.exists(mlp_path):
        log.error(f"MLP checkpoint bulunamadı: {mlp_path}")
        sys.exit(1)

    encoder = _build_beats_encoder(beats_path, device)
    mlp     = _build_mlp(mlp_path, device)

    # ---- Otomatik RMS kalibrasyonu ----
    if args.no_calibrate:
        active_rms_threshold = RMS_THRESHOLD
        log.info(f"[KALİBRASYON] Atlandı — sabit eşik: {active_rms_threshold}")
    else:
        active_rms_threshold = calibrate_rms(
            sample_rate=SAMPLE_RATE,
            duration_sec=4,
            device_index=args.device,
            multiplier=args.rms_multiplier,
        )

    # ---- Veritabanı ----
    db_conn = init_db(args.db_path)

    # ---- Kuyruk ve işçiler ----
    audio_queue = queue.Queue(maxsize=10)

    mic_worker = MicrophoneWorker(
        audio_queue,
        target_sample_rate=SAMPLE_RATE,
        device_index=args.device,
    )
    inf_worker = InferenceWorker(
        audio_queue,
        encoder, mlp, device,
        db_conn,
        args.audio_dir,
        args.center_url,
        args.no_telemetry,
        rms_threshold=active_rms_threshold,
    )

    log.info("=" * 60)
    log.info("  AIRPORT NOISE — EDGE DEVICE FIRMWARE BAŞLADI")
    log.info(f"  Model dir : {args.model_dir}")
    log.info(f"  DB        : {args.db_path}")
    log.info(f"  Kayıtlar  : {args.audio_dir}")
    log.info(f"  Telemetry : {'KAPALI' if args.no_telemetry else args.center_url}")
    log.info("  Durdurmak için Ctrl+C")
    log.info("=" * 60)

    mic_worker.start()
    inf_worker.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Kapatma sinyali alındı…")
    finally:
        mic_worker.stop()
        inf_worker.stop()
        mic_worker.join(timeout=3)
        inf_worker.join(timeout=3)
        db_conn.close()
        log.info("Firmware kapatıldı.")


if __name__ == "__main__":
    main()