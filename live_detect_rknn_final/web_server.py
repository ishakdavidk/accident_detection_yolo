import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, quote

import config
from config import save_threshold, check_password

# Global buffer holding most recent JPEG frame
latest_jpeg = None

# Root directory where event folders (each with event.mp4) live
EVENTS_ROOT = None


def set_events_root(path: str):
    """Called from main.py after SD card is resolved."""
    global EVENTS_ROOT
    EVENTS_ROOT = path
    print(f"[Web] Events root set to: {EVENTS_ROOT}")


class StreamingHandler(BaseHTTPRequestHandler):
    """
    Handles:
      GET /             -> Login gate then home page
      GET /login        -> Login form
      POST /do_login    -> Check password and set session cookie
      GET /setup        -> Threshold + password setup page
      GET /live         -> Pretty live stream page (HTML)
      GET /stream       -> MJPEG video stream (raw)
      GET /config       -> JSON with current threshold
      POST /config      -> update threshold (and save)
      POST /set_password -> update device password
      GET /events       -> List recorded events
      GET /play?ev=...  -> Show video player for one event
      GET /event_video?ev=... -> Serve raw event.mp4
    """

    def _send_html(self, html: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _redirect(self, location: str):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def _is_authenticated(self) -> bool:
        cookie = self.headers.get("Cookie", "")
        return "ondai_session=1" in cookie

    def do_GET(self):
        global latest_jpeg, EVENTS_ROOT

        open_paths = {"/login"}
        if self.path.startswith("/favicon"):
            open_paths.add(self.path)

        if (self.path not in open_paths) and (not self._is_authenticated()):
            self._redirect("/login")
            return

        # ----- JSON config for threshold -----
        if self.path == "/config":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(
                f'{{"threshold": {config.CONF_THRESHOLD_GLOBAL:.3f}}}'.encode("utf-8")
            )
            return

        # ----- Login page -----
        if self.path == "/login":
            html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OnDAI Login</title>
  <style>
    body { font-family: sans-serif; margin: 20px; background:#0f172a; color:#e5e7eb; }
    .box { border: 1px solid #1f2937; padding: 16px; border-radius: 8px; max-width: 360px; background:#020617; }
    input[type=password] { width: 100%; padding: 6px; margin-top: 4px; border-radius:4px; border:1px solid #374151; background:#020617; color:#e5e7eb; }
    button { margin-top: 10px; padding: 8px 14px; border-radius: 999px;
             border: 1px solid #38bdf8; background-color: #0ea5e9; color: #0f172a; cursor: pointer; font-weight:500; }
    button:hover { background:#38bdf8; }
    .err { color: #fca5a5; margin-top: 8px; font-size:0.85rem; }
  </style>
</head>
<body>
  <h1>OnDAI Login</h1>
  <div class="box">
    <form method="POST" action="/do_login">
      <label>Device Password</label><br>
      <input type="password" name="password" autocomplete="current-password">
      <button type="submit">Login</button>
    </form>
    <div class="err">%ERROR%</div>
  </div>
</body>
</html>
"""
            error_msg = ""
            if "err=1" in self.path:
                error_msg = "Invalid password."
            html = html.replace("%ERROR%", error_msg)
            self._send_html(html)
            return

        # ----- Root page: dashboard with Stream → Events → Setup -----
        if self.path == "/" or self.path.startswith("/index.html"):
            html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OnDAI Device</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a; /* dark navy */
      color: #e5e7eb;
    }}
    header {{
      padding: 16px 24px;
      background: #020617;
      border-bottom: 1px solid #1f2937;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    header .title {{
      font-size: 1.1rem;
      font-weight: 600;
    }}
    header .badge {{
      font-size: 0.75rem;
      padding: 4px 8px;
      border-radius: 999px;
      background: #22c55e22;
      border: 1px solid #22c55e55;
      color: #bbf7d0;
    }}
    main {{
      padding: 24px 16px 32px;
      max-width: 900px;
      margin: 0 auto;
    }}
    main h1 {{
      margin-top: 4px;
      margin-bottom: 4px;
    }}
    .subtitle {{
      margin-top: 4px;
      font-size: 0.9rem;
      color: #9ca3af;
    }}
    .cards {{
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #020617;
      border-radius: 12px;
      padding: 16px 16px 18px;
      border: 1px solid #1f2937;
      box-shadow: 0 18px 40px rgba(15,23,42,0.65);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 150px;
    }}
    .card h2 {{
      font-size: 1rem;
      margin: 0 0 4px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .card p {{
      margin: 4px 0 10px;
      font-size: 0.9rem;
      color: #9ca3af;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      padding: 4px 8px;
      border-radius: 999px;
      background: #0b1120;
      border: 1px solid #1f2937;
      color: #e5e7eb;
    }}
    .primary-btn {{
      margin-top: 8px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #38bdf8;
      background: #0ea5e9;
      color: #0f172a;
      font-size: 0.9rem;
      font-weight: 500;
      text-decoration: none;
      cursor: pointer;
    }}
    .primary-btn:hover {{
      background: #38bdf8;
      border-color: #7dd3fc;
    }}
    .ghost-btn {{
      margin-top: 8px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid #374151;
      background: transparent;
      color: #e5e7eb;
      font-size: 0.9rem;
      text-decoration: none;
      cursor: pointer;
    }}
    .ghost-btn:hover {{
      background: #111827;
    }}
    footer {{
      margin-top: 24px;
      font-size: 0.75rem;
      color: #6b7280;
      text-align: center;
    }}
  </style>
</head>
<body>
  <header>
    <div class="title">OnDAI Edge Device</div>
    <div class="badge">Accident Detection · Online</div>
  </header>

  <main>
    <h1>Dashboard</h1>
    <p class="subtitle">
      Access the live camera stream, review saved accident events, or adjust detection settings.
    </p>

    <div class="cards">
      <!-- 1. Live Stream -->
      <div class="card">
        <div>
          <h2>📺 Live Stream</h2>
          <p>View the real-time camera feed with accident detection overlay.</p>
          <span class="pill">Raw MJPEG: /stream</span>
        </div>
        <a class="primary-btn" href="/live">
          Open Live Stream
        </a>
      </div>

      <!-- 2. Events -->
      <div class="card">
        <div>
          <h2>🎬 Events</h2>
          <p>Browse recorded accident clips stored on the SD card.</p>
          <span class="pill">Folder: /media/sdcard/{config.OUTPUT_SUBDIR}</span>
        </div>
        <a class="ghost-btn" href="/events">
          View Events
        </a>
      </div>

      <!-- 3. Setup -->
      <div class="card">
        <div>
          <h2>⚙️ Setup</h2>
          <p>Adjust accident threshold sensitivity and change the device password.</p>
          <span class="pill">Current threshold: {config.CONF_THRESHOLD_GLOBAL:.2f}</span>
        </div>
        <a class="ghost-btn" href="/setup">
          Open Setup
        </a>
      </div>
    </div>

    <footer>
      OnDAI · Local edge AI accident detection. Access from the same network using the device IP.
    </footer>
  </main>
</body>
</html>
"""
            self._send_html(html)
            return

        # ----- Pretty Live Stream page (/live) -----
        if self.path == "/live":
            html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OnDAI Live Stream</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin:0; padding:0;
      background:#0f172a; color:#e5e7eb;
    }}
    header {{
      padding:16px 24px; background:#020617; border-bottom:1px solid #1f2937;
      display:flex; align-items:center; justify-content:space-between;
    }}
    header a {{
      color:#9ca3af; text-decoration:none; font-size:0.85rem;
    }}
    header a:hover {{
      color:#e5e7eb;
    }}
    main {{
      padding:24px 16px 32px;
      max-width:960px;
      margin:0 auto;
    }}
    h1 {{
      margin-top:4px; margin-bottom:6px; font-size:1.4rem;
    }}
    .subtitle {{
      margin-top:4px; font-size:0.9rem; color:#9ca3af;
    }}
    .stream-wrap {{
      margin-top:20px;
      background:#020617;
      border-radius:12px;
      padding:12px;
      border:1px solid #1f2937;
      box-shadow:0 18px 40px rgba(15,23,42,0.7);
    }}
    .stream-inner {{
      background:#000;
      border-radius:8px;
      overflow:hidden;
    }}
    img.stream {{
      display:block;
      width:100%;
      height:auto;
      background:#000;
    }}
    .hint {{
      margin-top:10px;
      font-size:0.8rem;
      color:#6b7280;
    }}
    code {{
      background:#111827;
      padding:2px 4px;
      border-radius:4px;
      font-size:0.8rem;
    }}
  </style>
</head>
<body>
  <header>
    <div>OnDAI · Live Stream</div>
    <div><a href="/">← Back to Dashboard</a></div>
  </header>
  <main>
    <h1>Live Stream</h1>
    <p class="subtitle">
      Real-time camera feed with accident detection overlay from the OnDAI device.
    </p>
    <div class="stream-wrap">
      <div class="stream-inner">
        <!-- MJPEG stream as an <img> -->
        <img class="stream" src="/stream" alt="OnDAI Live Stream">
      </div>
      <div class="hint">
        If the stream does not appear, ensure you are on the same network as the device and that
        MJPEG streaming is enabled. Raw endpoint: <code>http://&lt;IP&gt;:{config.STREAM_PORT}/stream</code>
      </div>
    </div>
  </main>
</body>
</html>
"""
            self._send_html(html)
            return

        # ----- Setup page (dark, pretty) -----
        if self.path == "/setup":
            html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OnDAI Setup</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: #e5e7eb;
    }}
    header {{
      padding: 16px 24px;
      background: #020617;
      border-bottom: 1px solid #1f2937;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    header a {{
      color: #9ca3af;
      text-decoration: none;
      font-size: 0.85rem;
    }}
    header a:hover {{
      color: #e5e7eb;
    }}
    main {{
      padding: 24px 16px 32px;
      max-width: 720px;
      margin: 0 auto;
    }}
    h1 {{
      margin-top: 4px;
      margin-bottom: 6px;
      font-size: 1.4rem;
    }}
    .subtitle {{
      margin-top: 4px;
      font-size: 0.9rem;
      color: #9ca3af;
    }}
    .panel {{
      border: 1px solid #1f2937;
      padding: 16px 18px 18px;
      border-radius: 12px;
      margin-top: 20px;
      background: #020617;
      box-shadow: 0 18px 40px rgba(15,23,42,0.7);
    }}
    label {{
      font-weight: 500;
      font-size: 0.95rem;
    }}
    input[type=range] {{
      width: 100%;
      margin-top: 10px;
    }}
    input[type=password], input[type=text] {{
      width: 100%;
      padding: 8px;
      margin-top: 8px;
      border-radius: 8px;
      border: 1px solid #374151;
      background:#020617;
      color:#e5e7eb;
      font-size:0.9rem;
    }}
    button {{
      margin-top: 10px;
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid #38bdf8;
      background-color: #0ea5e9;
      color: #0f172a;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 500;
    }}
    button:disabled {{
      background-color: #4b5563;
      border-color: #4b5563;
      color: #9ca3af;
      cursor: default;
    }}
    #status, #pw_status {{
      margin-top: 8px;
      font-size: 0.85rem;
      color: #9ca3af;
    }}
    code {{
      background:#111827;
      padding: 2px 4px;
      border-radius:4px;
      font-size:0.8rem;
    }}
  </style>
</head>
<body>
  <header>
    <div>OnDAI · Setup</div>
    <div><a href="/">← Back to Dashboard</a></div>
  </header>

  <main>
    <h1>Device Setup</h1>
    <p class="subtitle">
      Tune the accident detection threshold and manage the device password.
    </p>

    <div class="panel">
      <label>Accident Threshold:
        <span id="val">...</span>
      </label>
      <input id="th" type="range" min="0" max="1" step="0.01" value="0">
      <p style="font-size: 0.85rem; color:#9ca3af; margin-top:10px;">
        • Lower threshold = more sensitive (more detections, more false alarms).<br>
        • Higher threshold = stricter (fewer detections, but may miss some events).
      </p>
      <button id="saveBtn" disabled>Save Threshold</button>
      <div id="status"></div>
    </div>

    <div class="panel">
      <label>Change Device Password</label>
      <p style="font-size: 0.85rem; color:#9ca3af; margin-top:6px;">
        Default password is <code>123</code>. Please change it after first setup.
      </p>
      <input id="pw1" type="password" placeholder="New password">
      <input id="pw2" type="password" placeholder="Confirm new password">
      <button id="pwSaveBtn">Save Password</button>
      <div id="pw_status"></div>
    </div>
  </main>

  <script>
    let currentValue = null;
    let slider = document.getElementById('th');
    let valSpan = document.getElementById('val');
    let saveBtn = document.getElementById('saveBtn');
    let statusDiv = document.getElementById('status');

    let pw1 = document.getElementById('pw1');
    let pw2 = document.getElementById('pw2');
    let pwSaveBtn = document.getElementById('pwSaveBtn');
    let pwStatus = document.getElementById('pw_status');

    async function loadThreshold() {{
      try {{
        let r = await fetch('/config');
        if (!r.ok) return;
        let j = await r.json();
        let t = j.threshold;
        currentValue = t;
        valSpan.innerText = t.toFixed(2);
        slider.value = t.toFixed(2);
        saveBtn.disabled = true;
        statusDiv.innerText = '';
      }} catch (e) {{
        console.log('Failed to load threshold', e);
        statusDiv.innerText = 'Failed to load current threshold.';
      }}
    }}

    async function saveThreshold() {{
      let v = parseFloat(slider.value);
      try {{
        let resp = await fetch('/config', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
          body: 'threshold=' + encodeURIComponent(v)
        }});
        if (!resp.ok) {{
          statusDiv.innerText = 'Failed to save threshold.';
          return;
        }}
        currentValue = v;
        saveBtn.disabled = true;
        statusDiv.innerText = 'Saved. (Current: ' + v.toFixed(2) + ')';
      }} catch (e) {{
        console.log('Failed to update threshold', e);
        statusDiv.innerText = 'Error while saving threshold.';
      }}
    }}

    async function savePassword() {{
      let a = pw1.value;
      let b = pw2.value;
      if (!a || !b) {{
        pwStatus.innerText = 'Password fields must not be empty.';
        return;
      }}
      if (a !== b) {{
        pwStatus.innerText = 'Passwords do not match.';
        return;
      }}
      try {{
        let resp = await fetch('/set_password', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
          body: 'new_password=' + encodeURIComponent(a)
        }});
        let text = await resp.text();
        if (!resp.ok) {{
          pwStatus.innerText = 'Failed to save password: ' + text;
          return;
        }}
        pwStatus.innerText = 'Password updated successfully.';
        pw1.value = '';
        pw2.value = '';
      }} catch (e) {{
        console.log('Failed to set password', e);
        pwStatus.innerText = 'Error while saving password.';
      }}
    }}

    slider.addEventListener('input', (e) => {{
      let v = parseFloat(e.target.value);
      valSpan.innerText = v.toFixed(2);
      if (currentValue === null) {{
        saveBtn.disabled = false;
      }} else {{
        saveBtn.disabled = Math.abs(v - currentValue) < 0.0001;
      }}
    }});

    saveBtn.addEventListener('click', (e) => {{
      e.preventDefault();
      saveThreshold();
    }});

    pwSaveBtn.addEventListener('click', (e) => {{
      e.preventDefault();
      savePassword();
    }});

    loadThreshold();
  </script>
</body>
</html>
"""
            self._send_html(html)
            return

        # ----- Events page: list recorded events (prettier) -----
        if self.path == "/events":
            if not EVENTS_ROOT or not os.path.isdir(EVENTS_ROOT):
                html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OnDAI Events</title>
  <style>
    body { font-family: sans-serif; margin:0; padding:0; background:#0f172a; color:#e5e7eb; }
    header { padding:16px 24px; background:#020617; border-bottom:1px solid #1f2937;
             display:flex; align-items:center; justify-content:space-between; }
    header a { color:#9ca3af; text-decoration:none; font-size:0.85rem; }
    header a:hover { color:#e5e7eb; }
    main { padding:24px 16px 32px; max-width:900px; margin:0 auto; }
    h1 { margin-top:4px; margin-bottom:6px; font-size:1.4rem; }
    .subtitle { margin-top:4px; font-size:0.9rem; color:#9ca3af; }
  </style>
</head>
<body>
  <header>
    <div>OnDAI · Events</div>
    <div><a href="/">← Back to Dashboard</a></div>
  </header>
  <main>
    <h1>Recorded Events</h1>
    <p class="subtitle">No events directory found.</p>
  </main>
</body>
</html>
"""
                self._send_html(html)
                return

            entries = []
            for name in sorted(os.listdir(EVENTS_ROOT), reverse=True):
                full = os.path.join(EVENTS_ROOT, name)
                if not os.path.isdir(full):
                    continue
                video_path = os.path.join(full, "event.mp4")
                has_video = os.path.isfile(video_path)
                entries.append((name, has_video))

            count = len(entries)

            cards = []
            if not entries:
                cards.append("""
        <div class="empty">
          No events recorded yet. Once an accident is detected, clips will appear here.
        </div>
""")
            else:
                for name, has_video in entries:
                    safe = quote(name)
                    if has_video:
                        status = '<span class="status ready">Ready</span>'
                        button = f'<a class="card-btn" href="/play?ev={safe}">▶ Play</a>'
                    else:
                        status = '<span class="status processing">Processing</span>'
                        button = '<span class="card-btn disabled">Encoding…</span>'
                    cards.append(f"""
        <div class="card">
          <div class="card-main">
            <div class="card-title">{name}</div>
            <div class="card-meta">
              {status}
            </div>
          </div>
          <div class="card-actions">
            {button}
          </div>
        </div>
""")

            cards_html = "\n".join(cards)

            html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OnDAI Events</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin:0; padding:0;
      background:#0f172a; color:#e5e7eb;
    }}
    header {{
      padding:16px 24px; background:#020617; border-bottom:1px solid #1f2937;
      display:flex; align-items:center; justify-content:space-between;
    }}
    header a {{
      color:#9ca3af; text-decoration:none; font-size:0.85rem;
    }}
    header a:hover {{
      color:#e5e7eb;
    }}
    main {{
      padding:24px 16px 32px;
      max-width:900px;
      margin:0 auto;
    }}
    h1 {{
      margin-top:4px; margin-bottom:6px; font-size:1.4rem;
    }}
    .subtitle {{
      margin-top:4px; font-size:0.9rem; color:#9ca3af;
    }}
    .count {{
      margin-top:4px; font-size:0.8rem; color:#6b7280;
    }}
    .list {{
      margin-top:20px;
      display:flex;
      flex-direction:column;
      gap:10px;
    }}
    .card {{
      background:#020617;
      border-radius:10px;
      padding:10px 14px;
      border:1px solid #1f2937;
      display:flex;
      align-items:center;
      justify-content:space-between;
      box-shadow:0 12px 30px rgba(15,23,42,0.6);
    }}
    .card-main {{
      display:flex;
      flex-direction:column;
      gap:4px;
    }}
    .card-title {{
      font-size:0.95rem;
      font-weight:500;
    }}
    .card-meta {{
      font-size:0.8rem;
      color:#9ca3af;
    }}
    .status {{
      display:inline-flex;
      align-items:center;
      padding:2px 8px;
      border-radius:999px;
      font-size:0.75rem;
    }}
    .status.ready {{
      background:#22c55e22;
      border:1px solid #22c55e55;
      color:#bbf7d0;
    }}
    .status.processing {{
      background:#eab30822;
      border:1px solid #eab30855;
      color:#facc15;
    }}
    .card-actions {{
      display:flex;
      align-items:center;
      justify-content:flex-end;
    }}
    .card-btn {{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      padding:6px 12px;
      border-radius:999px;
      font-size:0.85rem;
      text-decoration:none;
      cursor:pointer;
      border:1px solid #38bdf8;
      background:#0ea5e9;
      color:#0f172a;
      font-weight:500;
    }}
    .card-btn:hover {{
      background:#38bdf8;
      border-color:#7dd3fc;
    }}
    .card-btn.disabled {{
      border-color:#4b5563;
      background:#1f2937;
      color:#9ca3af;
      cursor:default;
    }}
    .empty {{
      margin-top:20px;
      font-size:0.9rem;
      color:#9ca3af;
    }}
  </style>
</head>
<body>
  <header>
    <div>OnDAI · Events</div>
    <div><a href="/">← Back to Dashboard</a></div>
  </header>
  <main>
    <h1>Recorded Events</h1>
    <p class="subtitle">
      Each card corresponds to an accident event saved on the SD card.
    </p>
    <div class="count">
      Total events: {count}
    </div>
    <div class="list">
{cards_html}
    </div>
  </main>
</body>
</html>
"""
            self._send_html(html)
            return

        # ----- Play a single event video (prettier) -----
        if self.path.startswith("/play"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            ev = params.get("ev", [""])[0]

            # basic sanitization
            if not ev or ".." in ev or "/" in ev or "\\" in ev:
                self._send_html("Invalid event name", code=400)
                return

            if not EVENTS_ROOT or not os.path.isdir(EVENTS_ROOT):
                self._send_html("Events directory not available", code=500)
                return

            event_dir = os.path.join(EVENTS_ROOT, ev)
            video_path = os.path.join(event_dir, "event.mp4")

            if not os.path.isfile(video_path):
                self._send_html("Video not found (event.mp4 missing)", code=404)
                return

            ev_safe = quote(ev)
            html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OnDAI Event - {ev}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin:0; padding:0;
      background:#0f172a; color:#e5e7eb;
    }}
    header {{
      padding:16px 24px; background:#020617; border-bottom:1px solid #1f2937;
      display:flex; align-items:center; justify-content:space-between;
    }}
    header a {{
      color:#9ca3af; text-decoration:none; font-size:0.85rem;
    }}
    header a:hover {{
      color:#e5e7eb;
    }}
    main {{
      padding:24px 16px 32px;
      max-width:960px;
      margin:0 auto;
    }}
    h1 {{
      margin-top:4px; margin-bottom:8px; font-size:1.3rem;
      word-break:break-all;
    }}
    .subtitle {{
      margin-top:4px; font-size:0.9rem; color:#9ca3af;
    }}
    .video-wrap {{
      margin-top:20px;
      background:#020617;
      border-radius:12px;
      padding:12px;
      border:1px solid #1f2937;
      box-shadow:0 18px 40px rgba(15,23,42,0.7);
    }}
    video {{
      width:100%;
      height:auto;
      border-radius:8px;
      background:#000;
    }}
  </style>
</head>
<body>
  <header>
    <div>OnDAI · Event Playback</div>
    <div><a href="/events">← Back to Events</a></div>
  </header>
  <main>
    <h1>{ev}</h1>
    <p class="subtitle">
      Playing stitched accident clip from SD card.
    </p>
    <div class="video-wrap">
      <video controls>
        <source src="/event_video?ev={ev_safe}" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>
  </main>
</body>
</html>
"""
            self._send_html(html)
            return

        # ----- Raw event video file (MP4) -----
        if self.path.startswith("/event_video"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            ev = params.get("ev", [""])[0]

            print(f"[EventVideo] Request for ev={ev!r}")

            # basic sanitization
            if not ev or ".." in ev or "/" in ev or "\\" in ev:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid event name")
                print("[EventVideo] Invalid event name")
                return

            if not EVENTS_ROOT or not os.path.isdir(EVENTS_ROOT):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Events directory not available")
                print("[EventVideo] EVENTS_ROOT missing or not a directory")
                return

            event_dir = os.path.join(EVENTS_ROOT, ev)
            video_path = os.path.join(event_dir, "event.mp4")
            print(f"[EventVideo] Resolved path: {video_path}")

            if not os.path.isfile(video_path):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Video not found")
                print("[EventVideo] File not found")
                return

            try:
                # Stream file in chunks (better than reading all into RAM)
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()

                with open(video_path, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)

                print("[EventVideo] Served OK")
                return
            except Exception as e:
                print(f"[EventVideo] Failed to read/send {video_path}: {e}")
                # We already sent headers; just stop sending.
                try:
                    self.wfile.write(b"")
                except Exception:
                    pass
                return

        # ----- MJPEG stream (raw) -----
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            try:
                while True:
                    if latest_jpeg is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(latest_jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                return

        # ----- fallback -----
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def do_POST(self):
        # ----- login -----
        if self.path == "/do_login":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                data = parse_qs(body)
                pw = data.get("password", [""])[0]
                if check_password(pw):
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.send_header("Set-Cookie", "ondai_session=1; Path=/; HttpOnly")
                    self.end_headers()
                    return
                else:
                    self._redirect("/login?err=1")
                    return
            except Exception as e:
                print(f"[Login] Error processing login: {e}")
                self._redirect("/login?err=1")
                return

        # ----- update threshold -----
        if self.path == "/config":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                data = parse_qs(body)
                if "threshold" in data:
                    raw_val = data["threshold"][0]
                    new_val = float(raw_val)
                    new_val = max(0.0, min(1.0, new_val))
                    save_threshold(new_val)
                    print(
                        f"[Config] CONF_THRESHOLD_GLOBAL updated -> "
                        f"{config.CONF_THRESHOLD_GLOBAL:.3f}"
                    )
                    self._send_html("OK", code=200)
                    return
            except Exception as e:
                print(f"[Config] Failed to parse threshold: {e}")
            self._send_html("Invalid threshold", code=400)
            return

        # ----- update password -----
        if self.path == "/set_password":
            try:
                from config import save_password  # avoid circular at import time
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                data = parse_qs(body)
                new_pw = data.get("new_password", [""])[0]
                if not new_pw.strip():
                    self._send_html("Password must not be empty", code=400)
                    return
                if len(new_pw) < 3:
                    self._send_html("Password too short (min 3 chars).", code=400)
                    return
                save_password(new_pw)
                self._send_html("OK", code=200)
                return
            except Exception as e:
                print(f"[Password] Failed to update password: {e}")
                self._send_html("Failed to update password", code=500)
                return

        # ----- fallback -----
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"Not Found")


def run_stream_server():
    server = HTTPServer(("0.0.0.0", config.STREAM_PORT), StreamingHandler)
    print(f"[Stream] HTTP server at http://0.0.0.0:{config.STREAM_PORT}/ (root)")
    print(f"[Stream] Setup page:  http://0.0.0.0:{config.STREAM_PORT}/setup")
    print(f"[Stream] Live page:   http://0.0.0.0:{config.STREAM_PORT}/live")
    print(f"[Stream] MJPEG:       http://0.0.0.0:{config.STREAM_PORT}/stream")
    print(f"[Stream] Config:      http://0.0.0.0:{config.STREAM_PORT}/config")
    print(f"[Stream] Events page: http://0.0.0.0:{config.STREAM_PORT}/events")
    server.serve_forever()
