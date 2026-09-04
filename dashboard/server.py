"""Dashboard sidecar — the always-on half of the remote dashboard (Sep 2026).

Runs OUTSIDE machine.py so start/stop and the room view work when the machine
is down. Touches only files, shell scripts, and its own webcam; live machine
state comes through the /machine/* proxy to the in-process API (see
dashboard/machine_api.py, started inside machine.py on 127.0.0.1:8801).

    dashboard/start_dashboard.sh        # tmux session 'impostor-dashboard'
    then open http://<host>:8800        # (Tailscale IP from off-LAN)

Endpoints:
    GET  /                       the UI (index.html, single file, offline)
    GET  /api/status             machine pid/tmux/STOP-file/api-alive facts
    GET  /api/health             llama + ComfyUI + GPU + disk (cached 5s)
    POST /api/machine/start      wraps start_impostor.sh (409 if running)
    POST /api/machine/stop       wraps stop_machine.sh (idempotent)
    GET/POST /api/mode           runtime_mode.json (low_energy toggle)
    GET  /api/captions/stream    SSE tail of event_log/live_captions.txt
    GET  /api/history            merged recent event-log entries (newest runs)
    GET  /api/runs               newest run logs (id, start, size)
    GET  /api/comfy/list         ComfyUI output PNGs, newest first
    GET  /api/comfy/img          one PNG (?name=..., &thumb=1 for 480px JPEG)
    GET  /api/drawings           drawing_memory.json ledger passthrough
    GET  /api/roomcam.mjpg       room-POV MJPEG (placeholder when no cam)
    /machine/*                   reverse proxy -> 127.0.0.1:8801 (503 when down)
"""

import http.client
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from utils import runtime_mode

try:
    from config.config import MOOD_SNAPSHOT_FOLDER
except Exception:
    MOOD_SNAPSHOT_FOLDER = os.path.join(REPO, "event_log")
try:
    from config.config import COMFY_OUTPUT_FOLDER
except Exception:
    COMFY_OUTPUT_FOLDER = os.path.expanduser("~/ComfyUI/output")
try:
    from config.config import CAMERA_2_DEVICE, CAMERA_2_FPS, CAMERA_2_HEIGHT, CAMERA_2_WIDTH
except Exception:
    CAMERA_2_DEVICE, CAMERA_2_WIDTH, CAMERA_2_HEIGHT, CAMERA_2_FPS = "", 640, 480, 15

PORT = int(os.getenv("DASHBOARD_PORT", 8800))
MACHINE_API = ("127.0.0.1", int(os.getenv("MACHINE_API_PORT", 8801)))
PIDFILE = "/tmp/love_yourself_machine.pid"
LIVE_CAPTIONS = os.path.join(MOOD_SNAPSHOT_FOLDER, "live_captions.txt")
STOP_FILE = os.path.join(REPO, "STOP")
LLAMA_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
COMFY_URL = "http://localhost:8188"

_STREAM_SEM = threading.Semaphore(6)  # SSE + MJPEG + proxied streams each hold a thread
_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Machine status (facts only — the frontend derives the state word)
# ---------------------------------------------------------------------------


def machine_pid() -> int:
    """Live machine.py pid per the single_instance pidfile, else 0."""
    try:
        pid = int(open(PIDFILE).read().strip())
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode(errors="replace")
        return pid if "machine.py" in cmd else 0
    except Exception:
        return 0


def machine_api_ok() -> bool:
    try:
        conn = http.client.HTTPConnection(*MACHINE_API, timeout=0.5)
        conn.request("GET", "/mode")
        ok = conn.getresponse().status == 200
        conn.close()
        return ok
    except Exception:
        return False


def tmux_has(session: str) -> bool:
    try:
        return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


def build_status() -> dict:
    pid = machine_pid()
    return {
        "pid_alive": bool(pid),
        "pid": pid or None,
        "api_ok": machine_api_ok() if pid else False,
        "tmux_session": tmux_has("impostor-system"),
        "stop_file": os.path.exists(STOP_FILE),
        "low_energy": runtime_mode.low_energy(),
        "ts": time.time(),
    }


# ---------------------------------------------------------------------------
# Health (cached — probes are cheap but shouldn't run per poll per client)
# ---------------------------------------------------------------------------

_health_cache = {"ts": 0.0, "data": {}}
_health_lock = threading.Lock()


def _probe_http(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _gpu_stats() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0]
        used, total, util = [int(x.strip()) for x in out.split(",")]
        return {"vram_used_mb": used, "vram_total_mb": total, "gpu_util": util}
    except Exception:
        return {"vram_used_mb": None, "vram_total_mb": None, "gpu_util": None}


def build_health() -> dict:
    with _health_lock:
        if time.time() - _health_cache["ts"] < 5:
            return _health_cache["data"]
        disk = None
        try:
            du = shutil.disk_usage(MOOD_SNAPSHOT_FOLDER)
            disk = {"free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1)}
        except Exception:
            pass
        data = {
            "llama": _probe_http(f"{LLAMA_URL}/health"),
            "comfy": _probe_http(f"{COMFY_URL}/system_stats"),
            **_gpu_stats(),
            "disk": disk,
            "sidecar_uptime_s": round(time.time() - _START_TIME),
            "ts": time.time(),
        }
        _health_cache.update(ts=time.time(), data=data)
        return data


# ---------------------------------------------------------------------------
# Event-log history — newest runs only, tail-reads, per-(path,mtime) cache
# ---------------------------------------------------------------------------

HISTORY_TYPES = ("caption", "reflection", "comfy_prompt", "decision", "new_drawing")
_TAIL_BYTES = 4 * 1024 * 1024
_hist_cache = {}  # path -> (mtime_ns, entries)
_hist_lock = threading.Lock()


def list_run_logs():
    out = []
    try:
        for e in os.scandir(MOOD_SNAPSHOT_FOLDER):
            if e.name.endswith("-event-log.json") and e.is_file():
                st = e.stat()
                out.append({"path": e.path, "name": e.name, "mtime": st.st_mtime, "mtime_ns": st.st_mtime_ns, "size": st.st_size})
    except FileNotFoundError:
        pass
    out.sort(key=lambda r: -r["mtime"])
    return out


def _entry_view(e: dict) -> dict:
    keep = ("timestamp", "iso_timestamp", "type", "run_id", "caption", "mood", "boredom", "mode",
            "subject", "reflection", "prompt", "decision", "reason", "will_draw", "drive_level",
            "desire", "image_path", "action", "progress_percent", "message")
    return {k: e[k] for k in keep if k in e}


def tail_entries(path: str, mtime_ns: int) -> list:
    """Parse the last _TAIL_BYTES of a JSONL run log; cached until the file changes."""
    with _hist_lock:
        cached = _hist_cache.get(path)
        if cached and cached[0] == mtime_ns:
            return cached[1]
    entries = []
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > _TAIL_BYTES:
                f.seek(size - _TAIL_BYTES)
                f.readline()  # drop the partial line
            for raw in f:
                raw = raw.strip().rstrip(b",")
                if not raw.startswith(b"{"):
                    continue
                try:
                    o = json.loads(raw)
                    if isinstance(o, dict):
                        entries.append(o)
                except json.JSONDecodeError:
                    continue  # truncated trailing line mid-write — skip
    except OSError:
        pass
    with _hist_lock:
        _hist_cache[path] = (mtime_ns, entries)
        if len(_hist_cache) > 12:
            _hist_cache.pop(next(iter(_hist_cache)))
    return entries


def build_history(q: dict) -> dict:
    runs = min(int(q.get("runs", ["3"])[0]), 10)
    limit = min(int(q.get("limit", ["100"])[0]), 500)
    before = float(q.get("before", ["0"])[0]) or None
    types = set((q.get("types", [",".join(HISTORY_TYPES)])[0]).split(","))
    merged = []
    for run in list_run_logs()[:runs]:
        for e in tail_entries(run["path"], run["mtime_ns"]):
            t = e.get("type")
            if t not in types:
                continue
            if t == "decision" and e.get("decision") != "trigger_decision":
                continue
            ts = e.get("timestamp", 0)
            if before and ts >= before:
                continue
            merged.append(_entry_view(e))
    merged.sort(key=lambda e: -(e.get("timestamp") or 0))
    return {"entries": merged[:limit], "truncated": len(merged) > limit}


def build_runs() -> dict:
    out = []
    for run in list_run_logs()[:10]:
        meta = {}
        try:
            with open(run["path"], encoding="utf-8", errors="replace") as f:
                first = json.loads(f.readline().strip().rstrip(","))
            if first.get("type") == "run_metadata":
                meta = {"start": first.get("timestamp"), "iso": first.get("iso_timestamp")}
        except Exception:
            pass
        out.append({"run": run["name"].replace("-event-log.json", ""), "mtime": run["mtime"], "size": run["size"], **meta})
    return {"runs": out}


# ---------------------------------------------------------------------------
# ComfyUI gallery
# ---------------------------------------------------------------------------

_thumb_cache = {}  # (name, mtime_ns) -> jpeg bytes
_thumb_lock = threading.Lock()


def comfy_list(q: dict) -> dict:
    limit = min(int(q.get("limit", ["24"])[0]), 100)
    before = float(q.get("before", ["0"])[0]) or None
    out = []
    try:
        for e in os.scandir(COMFY_OUTPUT_FOLDER):
            if e.name.lower().endswith(".png") and e.is_file():
                st = e.stat()
                if before and st.st_mtime >= before:
                    continue
                out.append({"name": e.name, "mtime": st.st_mtime, "size": st.st_size})
    except FileNotFoundError:
        return {"images": [], "error": "output folder missing"}
    out.sort(key=lambda r: -r["mtime"])
    return {"images": out[:limit], "truncated": len(out) > limit}


def comfy_thumb(path: str, name: str, mtime_ns: int) -> bytes:
    key = (name, mtime_ns)
    with _thumb_lock:
        if key in _thumb_cache:
            return _thumb_cache[key]
    import cv2  # deferred — only thumbnails and the room cam need it

    img = cv2.imread(path)
    if img is None:
        raise ValueError("unreadable image")
    h, w = img.shape[:2]
    if w > 480:
        img = cv2.resize(img, (480, int(h * 480 / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise ValueError("encode failed")
    data = buf.tobytes()
    with _thumb_lock:
        _thumb_cache[key] = data
        while len(_thumb_cache) > 200:
            _thumb_cache.pop(next(iter(_thumb_cache)))
    return data


def drawings_ledger() -> dict:
    try:
        with open(os.path.join(MOOD_SNAPSHOT_FOLDER, "drawing_memory.json"), encoding="utf-8") as f:
            return {"ledger": json.load(f)}
    except Exception:
        return {"ledger": None}


# ---------------------------------------------------------------------------
# Room cam — sidecar-owned second webcam; placeholder when absent
# ---------------------------------------------------------------------------


class RoomCam:
    """One grab thread, one latest-JPEG slot. Never raises, never exits.

    Opens CAMERA_2_DEVICE (a /dev/v4l/by-id path) with MJPG forced so two
    cams fit one USB controller. Encode is throttled to 2fps when no MJPEG
    client has asked recently; the device stays open (frames drained) so
    reattachment is instant.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._jpg = self._placeholder("ROOM CAM OFFLINE" if CAMERA_2_DEVICE else "NO ROOM CAM CONFIGURED")
        self.last_client = 0.0
        threading.Thread(target=self._run, daemon=True, name="roomcam").start()

    def latest(self):
        with self._lock:
            return self._seq, self._jpg

    def _set(self, jpg: bytes):
        with self._lock:
            self._seq += 1
            self._jpg = jpg

    def _placeholder(self, text: str) -> bytes:
        try:
            import cv2
            import numpy as np

            img = np.zeros((CAMERA_2_HEIGHT, CAMERA_2_WIDTH, 3), dtype=np.uint8)
            img[:] = (12, 20, 20)
            cv2.putText(img, text, (30, CAMERA_2_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (32, 160, 232), 2)
            return cv2.imencode(".jpg", img)[1].tobytes()
        except Exception:
            # 1x1 black JPEG, pre-encoded — cv2/numpy truly unavailable
            return bytes.fromhex(
                "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fbfe8a28a2800a28a2803ffd9"
            )

    def _run(self):
        import cv2

        cap = None
        last_encode = 0.0
        while True:
            if not CAMERA_2_DEVICE:
                time.sleep(30)
                continue
            if cap is None:
                try:
                    cap = cv2.VideoCapture(CAMERA_2_DEVICE, cv2.CAP_V4L2)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_2_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_2_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_2_FPS)
                    if not cap.isOpened():
                        raise RuntimeError("open failed")
                except Exception:
                    try:
                        if cap:
                            cap.release()
                    except Exception:
                        pass
                    cap = None
                    self._set(self._placeholder("ROOM CAM OFFLINE"))
                    time.sleep(10)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                self._set(self._placeholder("ROOM CAM OFFLINE"))
                time.sleep(5)
                continue
            now = time.time()
            active = (now - self.last_client) < 60
            if active or (now - last_encode) > 0.5:
                try:
                    okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if okj:
                        self._set(buf.tobytes())
                        last_encode = now
                except Exception:
                    pass
            if not active:
                time.sleep(0.3)  # idle: drain slower, encode at ~2fps


roomcam = RoomCam()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    # -- routing ------------------------------------------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        route = parsed.path
        try:
            if route.startswith("/machine/"):
                return self._proxy()
            if route == "/api/status":
                return self._json(build_status())
            if route == "/api/health":
                return self._json(build_health())
            if route == "/api/mode":
                return self._json(runtime_mode.mode() or {"low_energy": False})
            if route == "/api/history":
                return self._json(build_history(q))
            if route == "/api/runs":
                return self._json(build_runs())
            if route == "/api/comfy/list":
                return self._json(comfy_list(q))
            if route == "/api/comfy/img":
                return self._comfy_img(q)
            if route == "/api/drawings":
                return self._json(drawings_ledger())
            if route == "/api/captions/stream":
                return self._sse_captions()
            if route == "/api/roomcam.mjpg":
                return self._mjpeg_roomcam()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            try:
                return self._json({"error": str(e)}, 500)
            except (BrokenPipeError, ConnectionResetError):
                return
        if route == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except Exception:
            return self._json({"error": "bad request body"}, 400)
        try:
            if route.startswith("/machine/"):
                return self._proxy(body_bytes=json.dumps(body).encode())
            if route == "/api/machine/start":
                if machine_pid():
                    return self._json({"error": "already running", "pid": machine_pid()}, 409)
                subprocess.Popen(["bash", os.path.join(REPO, "start_impostor.sh")], cwd=REPO,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self._json({"ok": True, "message": "supervisor starting"})
            if route == "/api/machine/stop":
                r = subprocess.run(["bash", os.path.join(REPO, "stop_machine.sh")], cwd=REPO,
                                   capture_output=True, text=True, timeout=15)
                return self._json({"ok": True, "output": (r.stdout or "").strip()})
            if route == "/api/mode":
                if "low_energy" not in body:
                    return self._json({"error": "expected {low_energy: bool}"}, 400)
                return self._json(runtime_mode.set_low_energy(bool(body["low_energy"])))
        except Exception as e:
            return self._json({"error": str(e)}, 500)
        return self._json({"error": "unknown endpoint"}, 404)

    # -- comfy image --------------------------------------------------------

    def _comfy_img(self, q):
        name = os.path.basename(q.get("name", [""])[0])
        if not name or not name.lower().endswith(".png"):
            return self._json({"error": "bad name"}, 400)
        path = os.path.join(COMFY_OUTPUT_FOLDER, name)
        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError:
            return self._json({"error": "not found"}, 404)
        if q.get("thumb", ["0"])[0] == "1":
            return self._bytes(comfy_thumb(path, name, mtime_ns), "image/jpeg")
        with open(path, "rb") as f:
            return self._bytes(f.read(), "image/png")

    # -- streams ------------------------------------------------------------

    def _stream_headers(self, ctype):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    @staticmethod
    def _classify(line: str) -> str:
        if line.startswith("Drawing: "):
            return "drawing"
        if line.startswith("Finished drawing: "):
            return "drawing_done"
        if line.startswith(("Wanted to draw", "Tried to draw")):
            return "drawing_failed"
        if line.startswith("Reflected on "):
            return "reflection"
        return "caption"

    def _sse_event(self, line: str):
        payload = json.dumps({"kind": self._classify(line), "text": line})
        self.wfile.write(f"data: {payload}\n\n".encode())

    def _sse_captions(self):
        if not _STREAM_SEM.acquire(blocking=False):
            return self._json({"error": "too many streams"}, 503)
        try:
            self._stream_headers("text/event-stream")
            self.wfile.write(b"retry: 3000\n\n")
            offset = 0
            try:
                with open(LIVE_CAPTIONS, "rb") as f:
                    size = os.path.getsize(LIVE_CAPTIONS)
                    f.seek(max(0, size - 65536))
                    tail = f.read().decode("utf-8", errors="replace").splitlines()
                    if size > 65536 and tail:
                        tail = tail[1:]  # drop the partial first line
                    for line in tail[-30:]:
                        if line.strip():
                            self._sse_event(line.strip())
                    offset = size
            except OSError:
                pass
            last_write = time.time()
            while True:
                try:
                    size = os.path.getsize(LIVE_CAPTIONS)
                except OSError:
                    size = 0
                if size < offset:
                    offset = 0  # file reset/rotated — start over
                if size > offset:
                    with open(LIVE_CAPTIONS, "rb") as f:
                        f.seek(offset)
                        chunk = f.read(size - offset)
                    offset = size
                    for line in chunk.decode("utf-8", errors="replace").splitlines():
                        if line.strip():
                            self._sse_event(line.strip())
                    last_write = time.time()
                if time.time() - last_write > 15:
                    self.wfile.write(b": ping\n\n")
                    last_write = time.time()
                self.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _STREAM_SEM.release()

    def _mjpeg_roomcam(self):
        if not _STREAM_SEM.acquire(blocking=False):
            return self._json({"error": "too many streams"}, 503)
        try:
            self._stream_headers("multipart/x-mixed-replace; boundary=frame")
            last_seq = -1
            while True:
                roomcam.last_client = time.time()
                seq, jpg = roomcam.latest()
                if seq != last_seq:
                    last_seq = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(1.0 / max(CAMERA_2_FPS, 1))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _STREAM_SEM.release()

    # -- proxy --------------------------------------------------------------

    def _proxy(self, body_bytes: bytes = None):
        target = self.path[len("/machine"):] or "/"
        try:
            conn = http.client.HTTPConnection(*MACHINE_API, timeout=1.5)
            conn.request(self.command, target, body=body_bytes,
                         headers={"Content-Type": "application/json"} if body_bytes else {})
            # Read timeout must be set BEFORE getresponse(): will_close
            # responses (our streams send Connection: close) null out
            # conn.sock inside getresponse. 10s covers MJPEG inter-frame
            # gaps and bounds the hang if the machine API freezes.
            conn.sock.settimeout(10)
            resp = conn.getresponse()
        except Exception:
            return self._json({"error": "MACHINE OFFLINE"}, 503)
        ctype = resp.getheader("Content-Type", "application/octet-stream")
        streaming = ctype.startswith(("multipart/x-mixed-replace", "text/event-stream"))
        if not streaming:
            data = resp.read()
            conn.close()
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if not _STREAM_SEM.acquire(blocking=False):
            conn.close()
            return self._json({"error": "too many streams"}, 503)
        try:
            self._stream_headers(ctype)
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            _STREAM_SEM.release()
            try:
                conn.close()
            except Exception:
                pass

    def log_message(self, *a):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        print(f"[dashboard] events: {MOOD_SNAPSHOT_FOLDER}")
        print(f"[dashboard] comfy output: {COMFY_OUTPUT_FOLDER}")
        print(f"[dashboard] room cam: {CAMERA_2_DEVICE or '(none configured)'}")
        print(f"[dashboard] http://0.0.0.0:{PORT}  (machine api proxy -> {MACHINE_API[0]}:{MACHINE_API[1]})")
        httpd.serve_forever()
