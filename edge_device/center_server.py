"""
Airport Noise — Merkez Sunucu Simülasyonu
==========================================
Farklı bir portta çalışır (varsayılan: 8080).
Edge cihazlarından gelen telemetry POST isteklerini karşılar,
ses dosyasını diske kaydeder, veriyi SQLite'a yazar.

Gerçek bir dağıtımda bu script uzak bir sunucuda çalışır;
burada aynı makinede farklı port simülasyonu yapıyoruz.

Kullanım: python center_server.py
          veya: uvicorn center_server:app --port 8080
Dashboard: http://localhost:8080
"""

import os
import sqlite3
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

CENTER_DB        = os.environ.get("CENTER_DB",        "center_data.db")
CENTER_AUDIO_DIR = os.environ.get("CENTER_AUDIO_DIR", "center_recordings")
CENTER_PORT      = int(os.environ.get("CENTER_PORT",  "8080"))

# ---------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(CENTER_AUDIO_DIR, exist_ok=True)
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT    NOT NULL,
            device_id   TEXT    NOT NULL,
            event_id    INTEGER NOT NULL,
            timestamp   TEXT    NOT NULL,
            label       TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            wav_path    TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"[MERKEZ] Sunucu hazır — port {CENTER_PORT}")
    print(f"[MERKEZ] Ses klasörü: {os.path.abspath(CENTER_AUDIO_DIR)}")
    yield


app = FastAPI(title="Airport Noise — Merkez Sunucu", lifespan=lifespan)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CENTER_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Telemetry endpoint
# ---------------------------------------------------------------------------

@app.post("/api/telemetry")
async def receive_telemetry(
    device_id:  str  = Form(...),
    event_id:   int  = Form(...),
    timestamp:  str  = Form(...),
    label:      str  = Form(...),
    confidence: float= Form(...),
    audio:      UploadFile = File(...),
):
    """
    Edge cihazından gelen ses + metadata kaydını alır.
    """
    received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ses dosyasını kaydet
    safe_name  = Path(audio.filename).name
    dest_path  = os.path.join(CENTER_AUDIO_DIR, f"{device_id}_{event_id}_{safe_name}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    # Veritabanına yaz
    conn = _get_conn()
    conn.execute("""
        INSERT INTO telemetry
            (received_at, device_id, event_id, timestamp, label, confidence, wav_path)
        VALUES (?,?,?,?,?,?,?)
    """, (received_at, device_id, event_id, timestamp, label,
          round(confidence, 4), dest_path))
    conn.commit()
    conn.close()

    print(f"[MERKEZ] YENİ VERİ ← {device_id} | {label} ({confidence:.2f}) | {timestamp}")

    return JSONResponse({"status": "ok", "received_at": received_at})


# ---------------------------------------------------------------------------
# API — merkez dashboard için
# ---------------------------------------------------------------------------

@app.get("/api/events")
def center_events(limit: int = 200):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/stats")
def center_stats():
    conn = _get_conn()
    by_device = conn.execute("""
        SELECT device_id, COUNT(*) as count, MAX(received_at) as last_seen
        FROM telemetry GROUP BY device_id ORDER BY count DESC
    """).fetchall()
    by_label = conn.execute("""
        SELECT label, COUNT(*) as count, AVG(confidence) as avg_conf
        FROM telemetry GROUP BY label ORDER BY count DESC
    """).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
    conn.close()
    return JSONResponse({
        "total": total,
        "by_device": [dict(r) for r in by_device],
        "by_label":  [dict(r) for r in by_label],
    })


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    safe = Path(filename).name
    path = os.path.join(CENTER_AUDIO_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(404, "Ses dosyası bulunamadı.")
    return FileResponse(path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Merkez dashboard sayfası
# ---------------------------------------------------------------------------

CENTER_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>Airport Noise — Merkez Sunucu</title>
<style>
  :root {
    --bg: #0a0f1a; --surface: #111827; --border: #1f2937;
    --accent: #f59e0b; --accent2: #fcd34d;   /* amber — "merkez" tonu */
    --text: #f1f5f9; --muted: #64748b;
    --success: #10b981; --danger: #ef4444;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text);
         font-family:'Segoe UI','Inter',sans-serif; min-height:100vh; }

  header {
    background:var(--surface); border-bottom:1px solid var(--border);
    padding:1.2rem 2rem; display:flex; align-items:center; gap:1rem;
  }
  .logo { font-size:1.3rem; font-weight:700; color:var(--accent2); }
  .logo span { color:var(--muted); font-weight:400; font-size:0.9rem; }
  .badge {
    margin-left:auto; background:var(--accent); color:#000;
    padding:0.25rem 0.75rem; border-radius:20px;
    font-size:0.7rem; font-weight:700; letter-spacing:0.1em;
  }

  .container { max-width:1300px; margin:0 auto; padding:2rem; }

  .kpi-row { display:flex; gap:1rem; margin-bottom:2rem; flex-wrap:wrap; }
  .kpi {
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:1rem 1.5rem; flex:1; min-width:160px;
  }
  .kpi-label { font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); }
  .kpi-value { font-size:2.2rem; font-weight:700; color:var(--accent2); margin-top:0.2rem; }
  .kpi-sub   { font-size:0.7rem; color:var(--muted); margin-top:0.15rem; }

  .two-col { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:2rem; }
  @media(max-width:800px) { .two-col { grid-template-columns:1fr; } }

  .panel {
    background:var(--surface); border:1px solid var(--border);
    border-radius:12px; overflow:hidden;
  }
  .panel-header {
    background:#0d1520; padding:0.65rem 1rem;
    font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted);
    border-bottom:1px solid var(--border);
  }
  table { width:100%; border-collapse:collapse; }
  th { padding:0.6rem 1rem; text-align:left; font-size:0.62rem; letter-spacing:0.1em;
       text-transform:uppercase; color:var(--muted); background:#0d1520;
       border-bottom:1px solid var(--border); }
  td { padding:0.65rem 1rem; border-bottom:1px solid var(--border); font-size:0.8rem; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:rgba(245,158,11,0.04); }

  .device-badge {
    display:inline-block; padding:0.2rem 0.6rem; border-radius:6px;
    background:#1f2937; font-size:0.72rem; font-weight:600; color:var(--accent2);
    border:1px solid var(--accent);
  }

  .label-pill {
    display:inline-block; padding:0.2rem 0.6rem; border-radius:20px;
    font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
    border:1px solid currentColor;
  }

  audio { height:26px; filter:invert(1) hue-rotate(20deg) brightness(0.9); border-radius:4px; }

  .stream-box {
    background:#0d1520; border:1px solid var(--border); border-radius:10px;
    padding:1rem; height:280px; overflow-y:auto; font-family:monospace; font-size:0.75rem;
    color:var(--muted);
  }
  .stream-entry { padding:0.15rem 0; border-bottom:1px solid #1f2937; }
  .stream-entry .ts   { color:var(--muted); }
  .stream-entry .dev  { color:var(--accent); font-weight:600; }
  .stream-entry .lbl  { color:var(--accent2); font-weight:700; }
  .stream-entry .conf { color:#94a3b8; }

  footer { text-align:center; padding:2rem; color:var(--muted); font-size:0.7rem; }
</style>
</head>
<body>
<header>
  <div class="logo">AIRPORT/NOISE <span>— Merkez Sunucu</span></div>
  <div class="badge">PORT 8080</div>
</header>

<div class="container">

  <div class="kpi-row" id="kpiRow">
    <div class="kpi">
      <div class="kpi-label">Toplam Alınan</div>
      <div class="kpi-value" id="kpiTotal">—</div>
      <div class="kpi-sub">Tüm cihazlar</div>
    </div>
  </div>

  <div class="two-col">
    <!-- Cihaz tablosu -->
    <div class="panel">
      <div class="panel-header">Cihazlar</div>
      <table><thead><tr><th>Cihaz</th><th>Olay</th><th>Son Görülme</th></tr></thead>
      <tbody id="deviceTable"><tr><td colspan="3" style="color:var(--muted);padding:1rem">Yükleniyor…</td></tr></tbody></table>
    </div>

    <!-- Canlı akış -->
    <div>
      <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);margin-bottom:0.6rem">
        Canlı Telemetry Akışı
      </div>
      <div class="stream-box" id="streamBox">
        <div style="text-align:center;padding:2rem;color:var(--muted)">
          Cihazlardan veri bekleniyor…
        </div>
      </div>
    </div>
  </div>

  <!-- Son olaylar tablosu -->
  <div class="panel">
    <div class="panel-header">Son Alınan Olaylar</div>
    <table>
      <thead>
        <tr><th>#</th><th>Alındı</th><th>Cihaz</th><th>Etiket</th><th>Güven</th><th>Ses</th></tr>
      </thead>
      <tbody id="eventsBody">
        <tr><td colspan="6" style="padding:2rem;text-align:center;color:var(--muted)">
          Henüz veri alınmadı.
        </td></tr>
      </tbody>
    </table>
  </div>

</div>

<footer>AIRPORT NOISE — Merkez Sunucu &nbsp;|&nbsp; <span id="nowTs"></span></footer>

<script>
const LABEL_COLORS = {
  AIRCRAFT:'#ef4444', SPEECH:'#3b82f6', TRAFFIC:'#f59e0b',
  WIND:'#10b981', OTHER:'#8b5cf6', AMBIENT:'#6b7280', UNKNOWN:'#374151'
};

let lastSeenId = 0;   // akış için yeni kayıtları takip et

function pill(label) {
  const c = LABEL_COLORS[label] || '#6b7280';
  return `<span class="label-pill" style="color:${c};border-color:${c}40;background:${c}18">${label}</span>`;
}

async function loadData() {
  try {
    const [evRes, stRes] = await Promise.all([
      fetch('/api/events?limit=100'), fetch('/api/stats')
    ]);
    const events = await evRes.json();
    const stats  = await stRes.json();

    // KPI
    document.getElementById('kpiTotal').textContent = stats.total;

    // Cihaz tablosu
    if (stats.by_device.length) {
      document.getElementById('deviceTable').innerHTML = stats.by_device.map(d => `
        <tr>
          <td><span class="device-badge">${d.device_id}</span></td>
          <td>${d.count}</td>
          <td style="color:var(--muted);font-size:0.75rem">${d.last_seen}</td>
        </tr>`).join('');
    }

    // Olaylar tablosu
    if (events.length) {
      document.getElementById('eventsBody').innerHTML = events.map(e => `
        <tr>
          <td style="color:var(--muted)">${e.id}</td>
          <td style="font-size:0.75rem">${e.received_at}</td>
          <td><span class="device-badge">${e.device_id}</span></td>
          <td>${pill(e.label)}</td>
          <td style="color:var(--accent2)">${Math.round(e.confidence*100)}%</td>
          <td><audio controls src="/audio/${encodeURIComponent(e.wav_path.split(/[\\/]/).pop())}"></audio></td>
        </tr>`).join('');

      // Akış: sadece yeni kayıtlar
      const newEvents = events.filter(e => e.id > lastSeenId);
      if (newEvents.length) {
        const box = document.getElementById('streamBox');
        if (lastSeenId === 0) box.innerHTML = '';   // ilk yüklemede temizle
        newEvents.reverse().forEach(e => {
          const line = document.createElement('div');
          line.className = 'stream-entry';
          line.innerHTML = `
            <span class="ts">${e.received_at}</span>
            <span class="dev"> ← ${e.device_id}</span>
            <span class="lbl"> [${e.label}]</span>
            <span class="conf"> güven=${Math.round(e.confidence*100)}%</span>`;
          box.prepend(line);
        });
        lastSeenId = Math.max(...events.map(e => e.id));
      }
    }

  } catch(err) { console.error(err); }
}

function updateNow() {
  document.getElementById('nowTs').textContent = new Date().toLocaleString('tr-TR');
}

loadData();
updateNow();
setInterval(loadData, 5_000);    // merkez 5s'de bir yenilenir
setInterval(updateNow, 1_000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def center_dashboard():
    return HTMLResponse(CENTER_HTML)


# ---------------------------------------------------------------------------
# Doğrudan çalıştırma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "center_server:app",
        host="0.0.0.0",
        port=CENTER_PORT,
        reload=False,
        log_level="info",
    )
