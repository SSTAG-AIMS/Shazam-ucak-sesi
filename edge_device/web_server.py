"""
Airport Noise — Local Web Dashboard
=====================================
SQLite veritabanındaki ses olaylarını web üzerinden listeler.
Her satırda tarih-saat, etiket, güven skoru ve tarayıcı içi ses oynatıcı vardır.

Kullanım: uvicorn web_server:app --host 0.0.0.0 --port 5000 --reload
          veya: python web_server.py
Tarayıcı: http://localhost:5000
"""

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Yapılandırma  (firmware.py ile aynı varsayılanlar)
# ---------------------------------------------------------------------------

DB_PATH    = os.environ.get("NOISE_DB",        "device_data.db")
AUDIO_DIR  = os.environ.get("NOISE_AUDIO_DIR", "recordings")

LABEL_COLORS = {
    "AIRCRAFT": "#ef4444",   # kırmızı
    "SPEECH":   "#3b82f6",   # mavi
    "TRAFFIC":  "#f59e0b",   # sarı
    "WIND":     "#10b981",   # yeşil
    "OTHER":    "#8b5cf6",   # mor
    "AMBIENT":  "#6b7280",   # gri
    "UNKNOWN":  "#374151",   # koyu gri
}

# ---------------------------------------------------------------------------
# Uygulama
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Başlangıçta veritabanı tablo varlığını garantile."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT    NOT NULL,
            label      TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            wav_path   TEXT    NOT NULL,
            sent       INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    yield


app = FastAPI(title="Airport Noise Dashboard", lifespan=lifespan)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# API endpoint'leri
# ---------------------------------------------------------------------------

@app.get("/api/events")
def api_events(limit: int = 200, offset: int = 0):
    """Son N olayı JSON olarak döndürür (dashboard için auto-refresh)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/stats")
def api_stats():
    """Etiket bazlı özet istatistikler."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT label,
               COUNT(*)          AS count,
               AVG(confidence)   AS avg_conf,
               MAX(timestamp)    AS last_seen
        FROM events
        GROUP BY label
        ORDER BY count DESC
    """).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close()
    return JSONResponse({"total": total, "by_label": [dict(r) for r in rows]})


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    """Ses dosyasını tarayıcıya gönderir."""
    # Güvenlik: path traversal engeli
    safe_name = Path(filename).name
    path = os.path.join(AUDIO_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Ses dosyası bulunamadı.")
    return FileResponse(path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Ana dashboard sayfası
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Airport Noise — Cihaz Dashboard</title>
<style>
  /* ── Temel Renk Paleti ─────────────────────────────────────────── */
  :root {
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --border:    #2a2d3a;
    --accent:    #6366f1;   /* indigo — havacılık/teknoloji tonu */
    --accent2:   #818cf8;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;

    --aircraft:  #ef4444;
    --speech:    #3b82f6;
    --traffic:   #f59e0b;
    --wind:      #10b981;
    --other:     #8b5cf6;
    --ambient:   #6b7280;
    --unknown:   #374151;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    min-height: 100vh;
  }

  /* ── Header ─────────────────────────────────────────────────────── */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 1.25rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .logo {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--accent2);
  }
  .logo span { color: var(--muted); font-weight: 400; }
  .live-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 1.6s ease-in-out infinite;
    margin-left: auto;
  }
  .live-label { font-size: 0.75rem; color: var(--success); letter-spacing: 0.1em; }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(1.3); }
  }

  /* ── Layout ─────────────────────────────────────────────────────── */
  .container { max-width: 1300px; margin: 0 auto; padding: 2rem; }

  /* ── İstatistik Kartları ─────────────────────────────────────────── */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
  }
  .stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--card-color, var(--accent));
  }
  .stat-label { font-size: 0.65rem; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; }
  .stat-value { font-size: 2rem; font-weight: 700; margin-top: 0.3rem; color: var(--card-color, var(--accent2)); }
  .stat-sub   { font-size: 0.7rem; color: var(--muted); margin-top: 0.2rem; }

  /* ── Tablo Başlığı ─────────────────────────────────────────────── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  .section-title { font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
  .refresh-btn {
    margin-left: auto;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 0.35rem 0.9rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.75rem;
    font-family: inherit;
    transition: border-color 0.2s, color 0.2s;
  }
  .refresh-btn:hover { border-color: var(--accent); color: var(--accent2); }
  .auto-badge {
    font-size: 0.65rem;
    color: var(--success);
    border: 1px solid var(--success);
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
  }

  /* ── Tablo ─────────────────────────────────────────────────────── */
  .table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  table { width: 100%; border-collapse: collapse; }
  th {
    background: #13151f;
    padding: 0.75rem 1rem;
    text-align: left;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(99,102,241,0.05); }

  /* ── Etiket Pill'leri ─────────────────────────────────────────── */
  .label-pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid currentColor;
  }

  /* ── Güven Barı ─────────────────────────────────────────────────── */
  .conf-wrap { display: flex; align-items: center; gap: 0.6rem; }
  .conf-bar-bg {
    flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
  }
  .conf-bar-fill { height: 100%; border-radius: 3px; background: var(--accent); }
  .conf-pct { font-size: 0.75rem; color: var(--muted); width: 38px; text-align: right; }

  /* ── Ses oynatıcı ─────────────────────────────────────────────── */
  audio {
    height: 28px;
    filter: invert(1) hue-rotate(180deg) brightness(0.85) saturate(0.6);
    border-radius: 4px;
  }

  /* ── İletildi rozeti ─────────────────────────────────────────── */
  .sent-yes { color: var(--success); font-size: 0.75rem; }
  .sent-no  { color: var(--muted);   font-size: 0.75rem; }

  /* ── Boş durum ──────────────────────────────────────────────── */
  .empty {
    text-align: center;
    padding: 4rem 1rem;
    color: var(--muted);
  }
  .empty .icon { font-size: 2.5rem; margin-bottom: 1rem; }
  .empty p { font-size: 0.85rem; }

  /* ── Footer ─────────────────────────────────────────────────── */
  footer {
    text-align: center;
    padding: 2rem;
    color: var(--muted);
    font-size: 0.7rem;
    letter-spacing: 0.05em;
  }
</style>
</head>
<body>

<header>
  <div class="logo">AIRPORT<span>/</span>NOISE <span>— Edge Dashboard</span></div>
  <div class="live-label">CANLI</div>
  <div class="live-dot"></div>
</header>

<div class="container">

  <!-- İstatistik kartları -->
  <div class="stats-grid" id="statsGrid">
    <div class="stat-card" style="--card-color:#6366f1">
      <div class="stat-label">Toplam Olay</div>
      <div class="stat-value" id="statTotal">—</div>
      <div class="stat-sub">Tüm zamanlar</div>
    </div>
  </div>

  <!-- Olay tablosu -->
  <div class="section-header">
    <div class="section-title">Ses Olayları</div>
    <div class="auto-badge">↻ 10s oto-yenileme</div>
    <button class="refresh-btn" onclick="loadData()">Şimdi Yenile</button>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Tarih / Saat</th>
          <th>Etiket</th>
          <th>Güven</th>
          <th>Ses</th>
          <th>Merkeze İletildi</th>
        </tr>
      </thead>
      <tbody id="eventsBody">
        <tr><td colspan="6">
          <div class="empty"><div class="icon">🎙️</div><p>Veri yükleniyor…</p></div>
        </td></tr>
      </tbody>
    </table>
  </div>

</div>

<footer>AIRPORT NOISE — TÜBİTAK &nbsp;|&nbsp; Edge Device EDGE-01 &nbsp;|&nbsp; <span id="nowTs"></span></footer>

<script>
const LABEL_COLORS = {
  AIRCRAFT: '#ef4444', SPEECH: '#3b82f6', TRAFFIC: '#f59e0b',
  WIND: '#10b981', OTHER: '#8b5cf6', AMBIENT: '#6b7280', UNKNOWN: '#374151'
};

function labelPill(label) {
  const c = LABEL_COLORS[label] || '#6b7280';
  return `<span class="label-pill" style="color:${c};border-color:${c}40;background:${c}18">${label}</span>`;
}

function confBar(conf) {
  const pct = Math.round(conf * 100);
  const color = conf >= 0.8 ? '#10b981' : conf >= 0.6 ? '#f59e0b' : '#ef4444';
  return `<div class="conf-wrap">
    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:${pct}%;background:${color}"></div></div>
    <div class="conf-pct">${pct}%</div>
  </div>`;
}

function audioPlayer(wavPath) {
  const filename = wavPath.split(/[\\/]/).pop();
  return `<audio controls src="/audio/${encodeURIComponent(filename)}"></audio>`;
}

async function loadData() {
  try {
    // Olaylar
    const evRes  = await fetch('/api/events?limit=200');
    const events = await evRes.json();

    // İstatistikler
    const stRes  = await fetch('/api/stats');
    const stats  = await stRes.json();

    // --- Kartlar ---
    let cardsHtml = `
      <div class="stat-card" style="--card-color:#6366f1">
        <div class="stat-label">Toplam Olay</div>
        <div class="stat-value">${stats.total}</div>
        <div class="stat-sub">Tüm zamanlar</div>
      </div>`;
    for (const s of stats.by_label) {
      const c = LABEL_COLORS[s.label] || '#6b7280';
      cardsHtml += `
        <div class="stat-card" style="--card-color:${c}">
          <div class="stat-label">${s.label}</div>
          <div class="stat-value">${s.count}</div>
          <div class="stat-sub">Ort. güven: ${Math.round(s.avg_conf * 100)}%</div>
        </div>`;
    }
    document.getElementById('statsGrid').innerHTML = cardsHtml;

    // --- Tablo ---
    if (!events.length) {
      document.getElementById('eventsBody').innerHTML = `
        <tr><td colspan="6">
          <div class="empty">
            <div class="icon">🎙️</div>
            <p>Henüz kayıtlı olay yok. Firmware çalışıyor mu?</p>
          </div>
        </td></tr>`;
      return;
    }

    let rows = '';
    for (const e of events) {
      const sent = e.sent
        ? '<span class="sent-yes">✓ İletildi</span>'
        : '<span class="sent-no">Bekliyor</span>';
      rows += `<tr>
        <td style="color:var(--muted)">${e.id}</td>
        <td>${e.timestamp}</td>
        <td>${labelPill(e.label)}</td>
        <td>${confBar(e.confidence)}</td>
        <td>${audioPlayer(e.wav_path)}</td>
        <td>${sent}</td>
      </tr>`;
    }
    document.getElementById('eventsBody').innerHTML = rows;

  } catch (err) {
    console.error('Veri yüklenemedi:', err);
  }
}

// Footer timestamp
function updateNow() {
  document.getElementById('nowTs').textContent = new Date().toLocaleString('tr-TR');
}

// İlk yükleme ve oto-yenileme
loadData();
updateNow();
setInterval(loadData, 10_000);
setInterval(updateNow, 1_000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Doğrudan çalıştırma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info",
    )
