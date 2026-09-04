"""In-process dashboard API — the live half of the remote dashboard (Sep 2026).

Started as a daemon thread from machine.py (right before the main loop) so it
holds real object refs — this is the only way to read live state; a separate
process importing the singletons gets fresh empty ones. Bound to 127.0.0.1
only; the browser reaches it through the sidecar's /machine/* proxy
(dashboard/server.py on :8800).

Failure isolation: start_machine_api never raises; every /state field is
fetched under its own guard and degrades to null; a handler can never take
the machine down.

Endpoints:
    GET  /state              full live snapshot (all fields nullable)
    GET  /cam/pov            MJPEG, clean machine POV (frame_buffer JPEGs, ~2fps)
    GET  /cam/pov/annotated  MJPEG, overlay frame (shared frame + own encode, 4fps)
    GET  /mode               runtime_mode + which gates are engaged
    POST /shutdown           SIGINT to self = the RESTART button (supervisor
                             restarts the machine; STOP stays untouched)
"""

import http.server
import json
import os
import signal
import threading
import time

from utils import runtime_mode

_STREAM_SEM = threading.Semaphore(4)
_START_TIME = time.time()


def _snapshot(refs: dict) -> dict:
    """Live state, every field independently guarded."""

    def g(fn):
        try:
            return fn()
        except Exception:
            return None

    captioner = refs.get("captioner")
    mood_engine = refs.get("mood_engine")
    kinetic_bus = refs.get("kinetic_bus")

    from captioner.context_compression import context_compressor
    from captioner.frame_buffer import frame_buffer
    from utils.drawing_state import DrawingState
    from utils.state_manager import state_manager

    def _failures():
        from utils.error_tracking import get_failure_tracker

        t = get_failure_tracker()
        now = time.time()
        return {
            name: {"ago_s": round(now - ts), "errors": t.component_errors.get(name, 0)}
            for name, ts in t.component_heartbeats.items()
        }

    def _gaze():
        from vision.gaze import get_gaze_description, get_gaze_state

        s = get_gaze_state()
        s["description"] = get_gaze_description()
        return s

    def _aruco():
        from safety.aruco_detector import get_aruco_detector

        return get_aruco_detector().get_status()

    def _person():
        from perception.person_detection_state import get_person_detection_state

        return get_person_detection_state().get_person_state()

    def _run_id():
        from event_logging.event_logger import get_current_run_id

        return get_current_run_id()

    def _fb(attr):
        v = getattr(frame_buffer, attr)
        return v() if callable(v) else v

    return {
        "ts": time.time(),
        "uptime_s": round(time.time() - (refs.get("start_time") or _START_TIME)),
        "run_id": g(_run_id),
        "caption": g(lambda: captioner.last_caption),
        "caption_ts": g(lambda: captioner.last_caption_time),
        "mood": g(lambda: mood_engine.get_current_mood()),
        "mood_vector": g(lambda: list(mood_engine.mood_vector)),
        "emotion": g(lambda: mood_engine.get_emotion_for_hand_controller()),
        "boredom": g(lambda: captioner.boredom),
        "baseline": g(lambda: (context_compressor.get_baseline_context() or "")[:600]),
        "desire": g(lambda: context_compressor.get_current_desire()),
        "belief": g(lambda: context_compressor.get_current_belief()),
        "journal_tail": g(lambda: list(context_compressor.journal)[-3:]),
        "reflection": g(lambda: (captioner.get_last_reflection() or "")[:800]),
        "drive_level": g(lambda: round(captioner.drawing.drive.level, 3)),
        "draw_block_reason": g(lambda: captioner.drawing.last_block_reason),
        "desire_shadow": g(lambda: captioner.drawing.desire_shadow_verdict()),
        "drawing": g(
            lambda: {
                **DrawingState.get_drawing_info(),
                "is_generating": state_manager.is_generating_drawing,
                "is_executing_cnc": state_manager.is_executing_cnc,
                "phase": state_manager.current_drawing_phase,
                "paper_present": state_manager.paper_present,
                "paper_state": state_manager.paper_state,
                "last_paper_check_ts": state_manager.last_paper_check_ts,
            }
        ),
        "aruco": g(_aruco),
        "gaze": g(_gaze),
        "kinetic": g(lambda: kinetic_bus.status() if kinetic_bus else None),
        "person": g(_person),
        "failures": g(_failures),
        "low_energy": g(runtime_mode.low_energy),
        "frame_buffer": g(lambda: {"frames": _fb("frame_count"), "seconds": _fb("seconds_buffered")}),
    }


def start_machine_api(refs: dict, port: int = None):
    """Spawn the API thread. Never raises — a dashboard failure must never
    stop the machine."""
    try:
        port = port or int(os.getenv("MACHINE_API_PORT", 8801))

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _json(self, obj, status=200):
                body = json.dumps(obj, default=str).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                # route on the bare path — the frontend appends ?t= cache-busters
                route = self.path.split("?", 1)[0]
                try:
                    if route == "/state":
                        return self._json(_snapshot(refs))
                    if route == "/mode":
                        kin = None
                        try:
                            kin = refs["kinetic_bus"].status().get("state") if refs.get("kinetic_bus") else None
                        except Exception:
                            pass
                        return self._json({**runtime_mode.mode(), "kinetic_state": kin})
                    if route == "/cam/pov":
                        return self._mjpeg_clean()
                    if route == "/cam/pov/annotated":
                        return self._mjpeg_annotated()
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as e:
                    try:
                        return self._json({"error": str(e)}, 500)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return self._json({"error": "unknown endpoint"}, 404)

            def do_POST(self):
                if self.path.split("?", 1)[0] == "/shutdown":
                    self._json({"ok": True, "message": "SIGINT sent — graceful shutdown"})
                    try:
                        self.wfile.flush()
                    except Exception:
                        pass
                    os.kill(os.getpid(), signal.SIGINT)
                    return
                return self._json({"error": "unknown endpoint"}, 404)

            def _stream_headers(self):
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()

            def _emit(self, jpg: bytes):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            def _mjpeg_clean(self):
                """Clean POV — the frame buffer's own JPEGs, zero new encode work."""
                if not _STREAM_SEM.acquire(blocking=False):
                    return self._json({"error": "too many streams"}, 503)
                try:
                    from captioner.frame_buffer import frame_buffer

                    self._stream_headers()
                    last = None
                    while True:
                        frames = frame_buffer.get_recent(seconds=3.0, max_frames=3)
                        jpg = frames[-1] if frames else None
                        if jpg is not None and jpg is not last:
                            last = jpg
                            self._emit(jpg)
                        time.sleep(0.25)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    _STREAM_SEM.release()

            def _mjpeg_annotated(self):
                """Overlay POV — copy of the loop's annotated frame, encoded here
                (off the main thread), 4fps, 960px wide."""
                if not _STREAM_SEM.acquire(blocking=False):
                    return self._json({"error": "too many streams"}, 503)
                try:
                    import cv2

                    from utils.state_manager import state_manager

                    self._stream_headers()
                    while True:
                        frame = state_manager.get_shared_frame(max_age=1.0)
                        if frame is not None:
                            h, w = frame.shape[:2]
                            if w > 960:
                                frame = cv2.resize(frame, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)
                            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ok:
                                self._emit(buf.tobytes())
                        time.sleep(0.25)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    _STREAM_SEM.release()

            def log_message(self, *a):
                pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        srv = Server(("127.0.0.1", port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True, name="dashboard-api").start()
        print(f"[INIT] Dashboard machine API on 127.0.0.1:{port} (proxied via dashboard sidecar :8800)")
        return srv
    except Exception as e:
        print(f"[WARNING] Dashboard machine API failed to start: {e}")
        return None
