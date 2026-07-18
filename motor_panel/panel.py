"""Unified motor control panel — every servo and stepper in one window.

Standalone tool: run it while machine.py is STOPPED (serial ports are
exclusive). Devices connect individually on demand; unconnected devices
simulate, logging their commands to the console pane, so the whole panel
works with nothing plugged in.

    python motor_panel/panel.py

Covers: gaze pan/tilt + lung (lunggaze Arduino), 4 fingers + elbow/shoulder
(lefthand Arduino), lightbulb PWM, and the GRBL XY steppers + pen servo.
The uArm is deliberately excluded.

The "body session" frame is a looper for whole-body choreography: perform
each subsystem on its own workspace (bed view for the gantry, draggable
linkage for the left arm), layer takes against a fixed loop, then train one
joint markov chain over all layers and let the machine improvise inside it.
"""

import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import scrolledtext, ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ARMS_DUET_MAX_FEED, ARMS_DUET_ZONE, GRBL_PEN_DOWN_S, GRBL_PEN_UP_S
from motor_panel.devices import SerialDevice, build_devices
from motor_panel.session import GROUPS, Session, Transport, import_legacy_hand_take, list_legacy_hand_datasets

JOG_STEPS = [0.5, 1, 2, 5, 10]
LOOP_LENGTHS = [15, 30, 45, 60]


def parse_position(status_line: str):
    """X,Y,Z out of a GRBL status line. This fork emits the old comma style
    with '>' glued to the last coord — <Idle,WPos:12.000,34.000,0.000> —
    so strip terminators per token or float() throws and position silently
    reads (0,0) forever (it did, until July 12)."""
    try:
        part = status_line.split("WPos:" if "WPos:" in status_line else "MPos:")[1]
        coords = [c.strip().rstrip(">|") for c in part.split("|")[0].split(",")]
        return float(coords[0]), float(coords[1]), float(coords[2]) if len(coords) > 2 else 0.0
    except (IndexError, ValueError):
        return 0.0, 0.0, 0.0


class DeviceFrame(ttk.LabelFrame):
    """Connect button + one slider row per channel + extras for one Arduino."""

    def __init__(self, parent, device: SerialDevice, log):
        super().__init__(parent, text=f"{device.name}  ({device.port})")
        self.device = device
        self.log = log
        device.on_line = log

        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=2)
        self.status = tk.Label(top, text="●", fg="gray")
        self.status.pack(side="left")
        self.btn = ttk.Button(top, text="Connect", command=self.toggle, width=11)
        self.btn.pack(side="left", padx=4)
        ttk.Button(top, text="Neutral", command=self.device.all_neutral, width=8).pack(side="left")

        self.sliders = {}
        for name in device.channel_order:
            ch = device.channels[name]
            row = ttk.Frame(self)
            row.pack(fill="x", padx=4)
            ttk.Label(row, text=name, width=9).pack(side="left")
            var = tk.IntVar(value=ch.neutral)
            val_lbl = ttk.Label(row, text=str(ch.neutral), width=4)

            def on_move(v, n=name, vl=val_lbl):
                vl.config(text=str(int(float(v))))
                self.device.set_channel(n, int(float(v)))  # non-blocking: writer queue

            s = ttk.Scale(row, from_=ch.lo, to=ch.hi, variable=var, command=on_move)
            s.pack(side="left", fill="x", expand=True, padx=4)
            val_lbl.pack(side="left")
            rev_var = tk.BooleanVar(value=ch.invert)
            ttk.Checkbutton(row, text="rev", variable=rev_var,
                            command=lambda n=name, v=rev_var: self.device.set_invert(n, v.get())).pack(side="left")
            self.sliders[name] = (s, var, val_lbl)

        if device.extras:
            ex = ttk.Frame(self)
            ex.pack(fill="x", padx=4, pady=2)
            for label in device.extras:
                ttk.Button(ex, text=label, command=lambda l=label: self.device.send_extra(l)).pack(side="left", padx=2)

    def toggle(self):
        if self.device.connected:
            self.device.disconnect()
            self.log(self.device.name, "disconnected", False)
        else:
            msg = self.device.connect()
            self.log(self.device.name, msg, not self.device.connected)
        self.refresh()

    def refresh(self):
        ok = self.device.connected
        self.status.config(fg="green" if ok else "gray")
        self.btn.config(text="Disconnect" if ok else "Connect")


class GrblFrame(ttk.LabelFrame):
    """GRBL is G-code-shaped, not slider-shaped: jog pad, pen, home/unlock."""

    def __init__(self, parent, log):
        super().__init__(parent, text="grbl CNC  (/dev/arduino_cnc @ 115200)")
        self.log = log
        self.ser = None
        self.step = tk.DoubleVar(value=5)
        self.position = (0.0, 0.0)  # last known WPos (or commanded, when simulating)
        # Single ordered writer queue. Thread-per-send + a lock serializes
        # but does NOT order — a G90 could run before its G91 partner, or a
        # streamed G1 could land between them, stranding the machine in
        # relative mode where every absolute target becomes a huge relative
        # lunge (the July 11 "flailing beyond limits"). One writer thread,
        # strictly FIFO, and no modal changes anywhere: absolute G90 only.
        self._cmd_q: "queue.Queue" = queue.Queue()
        # Path queue, NOT latest-wins: fast complex gestures must reach the
        # planner as the shape you drew. Under backlog the queue decimates
        # (every 2nd point) — shape preserved, lag bounded — instead of
        # collapsing to the newest point (which turned circles into jagged
        # polygons with a brake at every phantom corner).
        self._path: deque = deque()
        self.max_feed = ARMS_DUET_MAX_FEED  # UI slider can raise/lower this
        self.poll_rate_hint = lambda: 0.1  # 10Hz status, standard sender cadence
        self._write_lock = threading.Lock()  # '?' injects between queued commands
        self._resp_q: "queue.Queue" = queue.Queue()  # reader -> writer responses
        self.alarm = False  # GRBL boots alarm-locked until homed/unlocked

        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=2)
        self.status = tk.Label(top, text="●", fg="gray")
        self.status.pack(side="left")
        self.btn = ttk.Button(top, text="Connect", command=self.toggle, width=11)
        self.btn.pack(side="left", padx=4)
        ttk.Button(top, text="Home $H", command=lambda: self.send("$H")).pack(side="left", padx=2)
        ttk.Button(top, text="Unlock $X", command=lambda: self.send("$X")).pack(side="left", padx=2)
        ttk.Button(top, text="Status ?", command=lambda: self.send("?")).pack(side="left", padx=2)
        self.state_lbl = ttk.Label(top, text="")
        self.state_lbl.pack(side="left", padx=8)
        self.state_text = ""  # writer thread writes this; the label polls it (no cross-thread Tk)
        self._label_tick()

        jog = ttk.Frame(self)
        jog.pack(padx=4, pady=2)
        ttk.Button(jog, text="Y+", width=4, command=lambda: self.jog(0, 1)).grid(row=0, column=1)
        ttk.Button(jog, text="X-", width=4, command=lambda: self.jog(-1, 0)).grid(row=1, column=0)
        ttk.Button(jog, text="X+", width=4, command=lambda: self.jog(1, 0)).grid(row=1, column=2)
        ttk.Button(jog, text="Y-", width=4, command=lambda: self.jog(0, -1)).grid(row=2, column=1)
        stepbox = ttk.Frame(jog)
        stepbox.grid(row=1, column=1)
        ttk.Label(stepbox, text="mm").pack()
        ttk.OptionMenu(stepbox, self.step, 5, *JOG_STEPS).pack()

        pen = ttk.Frame(self)
        pen.pack(fill="x", padx=4, pady=2)
        ttk.Label(pen, text="pen S").pack(side="left")
        self.pen_lbl = ttk.Label(pen, text=str(GRBL_PEN_UP_S), width=4)

        def on_pen(v):
            self.pen_lbl.config(text=str(int(float(v))))
            self.send(f"M3 S{int(float(v))}")

        self.pen_var = tk.IntVar(value=GRBL_PEN_UP_S)
        ttk.Scale(pen, from_=0, to=255, variable=self.pen_var, command=on_pen).pack(side="left", fill="x", expand=True, padx=4)
        self.pen_lbl.pack(side="left")
        ttk.Button(pen, text=f"Up ({GRBL_PEN_UP_S})", command=lambda: self.set_pen(GRBL_PEN_UP_S)).pack(side="left", padx=2)
        ttk.Button(pen, text=f"Down ({GRBL_PEN_DOWN_S})", command=lambda: self.set_pen(GRBL_PEN_DOWN_S)).pack(side="left", padx=2)

        raw = ttk.Frame(self)
        raw.pack(fill="x", padx=4, pady=2)
        self.raw_entry = ttk.Entry(raw)
        self.raw_entry.pack(side="left", fill="x", expand=True, padx=2)
        self.raw_entry.bind("<Return>", lambda e: self.send_raw())
        ttk.Button(raw, text="Send", command=self.send_raw).pack(side="left")

    def toggle(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            self.log("grbl", "disconnected", False)
        else:
            def worker():
                try:
                    from grbl.grbl_utils import find_grbl_port
                    self.ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
                    self.log("grbl", f"connected {self.ser.port}" if self.ser else "no GRBL found", self.ser is None)
                    if self.ser is not None:
                        self._cmd_q = queue.Queue()  # drop anything stale
                        self._resp_q = queue.Queue()
                        threading.Thread(target=self._reader_loop, daemon=True).start()
                        threading.Thread(target=self._writer_loop, daemon=True).start()
                        self.send("G21", quiet=True)  # mm
                        self.send("G90", quiet=True)  # absolute — the only modal state, ever
                        threading.Thread(target=self._poll_loop, daemon=True).start()
                except Exception as e:
                    self.log("grbl", f"connect failed: {e}", True)
                self.refresh()
            threading.Thread(target=worker, daemon=True).start()
        self.refresh()

    def _poll_loop(self):
        """'?' is a REALTIME char: GRBL answers immediately even mid-move and
        it consumes no line-buffer slot — so it's written directly under the
        write lock, bypassing the command queue. Position keeps flowing even
        while motion commands sit in planner flow-control (the reason the
        trail lagged while motors got faster)."""
        import time as _t
        while self.ser is not None:
            try:
                with self._write_lock:
                    self.ser.write(b"?")
                    self.ser.flush()
            except Exception:
                pass
            _t.sleep(self.poll_rate_hint())

    def _reader_loop(self):
        """Owns ALL reading. '<...>' reports update position the instant they
        arrive; every other line is a command response, handed to the writer
        via _resp_q. Trail/recorder are fully decoupled from motion timing."""
        while self.ser is not None:
            try:
                line = self.ser.readline().decode(errors="replace").strip()
            except Exception:
                time.sleep(0.2)
                continue
            if not line:
                continue
            if line.startswith("<"):
                pos = parse_position(line)
                self.position = (pos[0], pos[1])
                self._set_state_label(line)
            else:
                self._resp_q.put(line)

    def _drain_responses(self):
        try:
            while True:
                self._resp_q.get_nowait()
        except queue.Empty:
            pass

    def _await_ok(self, timeout: float) -> str:
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                s = self._resp_q.get(timeout=min(0.5, max(0.05, deadline - time.time())))
            except queue.Empty:
                continue
            lines.append(s)
            if s == "ok" or s.lower().startswith("error"):
                break
        return " | ".join(lines)

    def refresh(self):
        ok = self.ser is not None
        self.status.config(fg="green" if ok else "gray")
        self.btn.config(text="Disconnect" if ok else "Connect")

    def send(self, cmd: str, quiet: bool = False):
        """Enqueue for the single writer thread — strict FIFO, no reordering."""
        if self.ser is None:
            if not quiet:
                self.log("grbl", cmd, True)
            return
        self._cmd_q.put((cmd, quiet))

    GARBAGE_ERRORS = ("bad number", "line overflow", "expected command", "invalid gcode")

    def _writer_loop(self):
        """Writes commands, never reads — the reader thread owns the port's
        RX side and feeds responses through _resp_q. This GRBL fork predates
        1.1 (no $J). '__MOTION__' resolves to the LATEST goto target (drag
        storms collapse instead of backlogging).

        Hard-won rules (July 12 logs): homing blocks GRBL for many seconds
        during which it does NOT drain serial — anything sent meanwhile
        overflows its RX buffer into parse garbage — so $H waits up to 60s
        with zero traffic; and repeated parse-garbage errors trigger a
        drain-and-pause resync instead of carrying on blind."""
        last_motion = None
        garbage_streak = 0
        # Character-counting protocol (what real senders do): track BYTES in
        # flight against the fork's ~127-byte RX buffer, not command counts.
        # ~25-byte lines -> 4-5 segments pipelined = deep planner lookahead,
        # speed carried through junctions like g-code file execution.
        RX_BUDGET = 120
        inflight_bytes = 0
        sent_lens: deque = deque()

        def reap(timeout: float) -> bool:
            """Consume one outstanding motion response; returns False on timeout."""
            nonlocal inflight_bytes, garbage_streak
            try:
                s = self._resp_q.get(timeout=timeout)
            except queue.Empty:
                return False
            if sent_lens:
                inflight_bytes = max(0, inflight_bytes - sent_lens.popleft())
            slow = s.lower()
            if "alarm" in slow:
                self.alarm = True
                self._set_state_label("ALARM — press Home $H")
            if "error" in slow:
                self.log("grbl", f"(motion)  ->  {s}", False)
                if any(g in slow for g in self.GARBAGE_ERRORS):
                    garbage_streak += 1
            else:
                garbage_streak = 0
            return True

        def reset_accounting():
            nonlocal inflight_bytes
            self._drain_responses()
            inflight_bytes = 0
            sent_lens.clear()

        while self.ser is not None:
            try:
                cmd, quiet = self._cmd_q.get(timeout=0.5)
            except queue.Empty:
                if inflight_bytes > 0:
                    reap(0.1)  # keep reaping while idle so accounting settles
                continue
            if cmd == "__MOTION__":
                if self.alarm:
                    self._path.clear()
                    continue  # never stream motion into a locked controller
                if not self._path:
                    continue
                x, y, feed = self._path.popleft()
                cmd = f"G0 X{x:.2f} Y{y:.2f}" if feed is None else f"G1 X{x:.2f} Y{y:.2f} F{feed}"
                if cmd == last_motion:
                    continue
                last_motion = cmd
                need = len(cmd) + 1
                misses = 0
                while inflight_bytes + need > RX_BUDGET and self.ser is not None:
                    if not reap(1.0):
                        misses += 1
                        if misses >= 5:  # oks lost (desync) — reset accounting
                            reset_accounting()
                            self.log("grbl", "motion acks lost — reset in-flight accounting", False)
                            break
                if garbage_streak >= 3:
                    time.sleep(0.3)
                    reset_accounting()
                    garbage_streak = 0
                    self.log("grbl", "serial desync detected — drained and resynced", False)
                try:
                    with self._write_lock:
                        self.ser.write((cmd + "\n").encode())
                        self.ser.flush()
                    sent_lens.append(need)
                    inflight_bytes += need
                except Exception as e:
                    self.log("grbl", f"{cmd}  [FAILED: {e}]", True)
                continue  # pipelined: no synchronous wait
            # non-motion command: settle all outstanding motion acks first
            misses = 0
            while inflight_bytes > 0 and self.ser is not None:
                if not reap(1.0):
                    misses += 1
                    if misses >= 5:
                        reset_accounting()
                        break
            timeout = 60.0 if cmd == "$H" else 10.0 if cmd == "$X" else 5.0
            try:
                if cmd in ("$H", "$X"):
                    self._drain_responses()  # clean slate at the lock boundary
                with self._write_lock:
                    self.ser.write((cmd + "\n").encode())
                    self.ser.flush()
                resp = self._await_ok(timeout)
            except Exception as e:
                self.log("grbl", f"{cmd}  [FAILED: {e}]", True)
                continue
            low = (resp or "").lower()
            if "alarm" in low:
                if not self.alarm:
                    self.log("grbl", "ALARM state — motion blocked until Home $H (or Unlock $X)", False)
                self.alarm = True
                self._set_state_label("ALARM — press Home $H")
            if cmd in ("$H", "$X") and "ok" in low.split("|")[-1]:
                self.alarm = False
                self._drain_responses()
                for setup in ("G21", "G90"):  # rejected while alarmed at connect
                    with self._write_lock:
                        self.ser.write((setup + "\n").encode())
                        self.ser.flush()
                    self._await_ok(5.0)
                self.log("grbl", f"{cmd} complete — unlocked, G21/G90 re-asserted", False)
            if any(g in low for g in self.GARBAGE_ERRORS):
                garbage_streak += 1
                if garbage_streak >= 3:
                    time.sleep(0.3)
                    self._drain_responses()
                    garbage_streak = 0
                    last_motion = None
                    self.log("grbl", "serial desync detected — drained and resynced", False)
            elif "error" not in low:
                garbage_streak = 0
            if resp and (not quiet or "error" in low):
                self.log("grbl", f"{cmd}  ->  {resp}", False)

    def _set_state_label(self, text: str):
        self.state_text = text  # picked up by _label_tick on the main thread

    def _label_tick(self):
        if self.state_lbl.cget("text") != self.state_text:
            self.state_lbl.config(text=self.state_text)
        self.after(150, self._label_tick)

    def jog(self, dx: int, dy: int):
        """Computed absolute target — never G91: an out-of-order or
        interleaved modal switch strands the machine in relative mode and
        every later absolute move becomes a wild relative lunge. Sent FIFO
        (not coalesced) so rapid clicks compound; position is updated
        optimistically and corrected by the status poll."""
        d = self.step.get()
        x = self.position[0] + dx * d
        y = self.position[1] + dy * d
        self.position = (x, y)
        self.send(f"G0 X{x:.2f} Y{y:.2f}")  # rapid — jogs are travel moves

    def goto(self, x: float, y: float, dt: float = None):
        """Absolute move into the session envelope.

        dt=None (live performance — pad drags, jogs): G0 RAPID. Dragging is
        pen-up travel, and rapids are why deployment/homing/node-jumps feel
        fast — G1 at any F is the deliberate drawing gait, and no streaming
        optimization changes which gait it is (July 12 lesson). dt given
        (playback / generation): G1 with feed derived from recorded timing
        so takes keep their performed tempo."""
        if self.alarm:
            return  # locked controller — motion is meaningless until $H
        x = max(ARMS_DUET_ZONE[0], min(ARMS_DUET_ZONE[1], x))
        y = max(ARMS_DUET_ZONE[2], min(ARMS_DUET_ZONE[3], y))
        if dt is None:
            feed = None  # rapid
        else:
            px, py = self._path[-1][:2] if self._path else self.position
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            feed = max(100, min(int(self.max_feed), int(dist / max(0.05, dt) * 60)))
        if self.ser is None:
            self.position = (x, y)  # simulate: commanded == actual
            return
        if self._path:
            lx, ly, _ = self._path[-1]
            if abs(x - lx) < 0.1 and abs(y - ly) < 0.1:
                return  # sub-resolution jitter — not a new waypoint
        self._path.append((x, y, feed))
        if len(self._path) > 24:  # bounded lag: thin the path, keep the shape
            self._path = deque(list(self._path)[::2])
        self._cmd_q.put(("__MOTION__", True))

    def set_pen(self, s: int):
        self.pen_var.set(s)
        self.pen_lbl.config(text=str(s))
        self.send(f"M3 S{s}")

    def send_raw(self):
        cmd = self.raw_entry.get().strip()
        if cmd:
            self.send(cmd)
            self.raw_entry.delete(0, "end")


class BedView(tk.Canvas):
    """Right-arm workspace: the plotting bed to scale — envelope, live
    position, a fading trail of the last seconds. Drag to perform."""

    W, H, M = 340, 250, 22
    TRAIL_SECONDS = 10.0

    def __init__(self, parent, grbl: GrblFrame):
        super().__init__(parent, width=self.W, height=self.H, bg="#101422",
                         highlightthickness=0, cursor="crosshair")
        self.grbl = grbl
        self.trail = []  # (x, y, t)
        x0, x1, y0, y1 = ARMS_DUET_ZONE
        self.create_rectangle(*self._px(x0, y1), *self._px(x1, y0), outline="#3a4a6b")
        for f in (1 / 3, 2 / 3):  # faint grid — spatial reference while performing
            gx0, gy0 = self._px(x0 + (x1 - x0) * f, y0)
            gx1, gy1 = self._px(x0 + (x1 - x0) * f, y1)
            self.create_line(gx0, gy0, gx1, gy1, fill="#1c2438")
            gx0, gy0 = self._px(x0, y0 + (y1 - y0) * f)
            gx1, gy1 = self._px(x1, y0 + (y1 - y0) * f)
            self.create_line(gx0, gy0, gx1, gy1, fill="#1c2438")
        self.create_text(self.W // 2, 12, text=f"bed  {x0}-{x1} × {y0}-{y1} mm   ✛ = target   ● = machine",
                         fill="#667", font=("monospace", 8))
        self.trail_line = self.create_line(0, 0, 0, 0, fill="#7a3448", smooth=True, width=2, state="hidden")
        # commanded target (where you asked it to go) vs reported position
        self.target_h = self.create_line(0, 0, 0, 0, fill="#f5c04a", width=1)
        self.target_v = self.create_line(0, 0, 0, 0, fill="#f5c04a", width=1)
        self.dot = self.create_oval(0, 0, 0, 0, fill="#e94560", outline="")
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self._tick()

    def _px(self, x: float, y: float):
        x0, x1, y0, y1 = ARMS_DUET_ZONE
        px = self.M + (x - x0) / (x1 - x0) * (self.W - 2 * self.M)
        py = self.H - self.M - (y - y0) / (y1 - y0) * (self.H - 2 * self.M)
        return px, py

    def _from_px(self, px: float, py: float):
        x0, x1, y0, y1 = ARMS_DUET_ZONE
        x = x0 + (px - self.M) / (self.W - 2 * self.M) * (x1 - x0)
        y = y0 + (self.H - self.M - py) / (self.H - 2 * self.M) * (y1 - y0)
        return max(x0, min(x1, x)), max(y0, min(y1, y))

    def _drag(self, ev):
        x, y = self._from_px(ev.x, ev.y)
        px, py = self._px(x, y)
        self.coords(self.target_h, px - 7, py, px + 7, py)
        self.coords(self.target_v, px, py - 7, px, py + 7)
        self.grbl.goto(x, y)  # dt omitted: live performance runs at full slider feed

    def _tick(self):
        now = time.time()
        x, y = self.grbl.position
        if not self.trail or (self.trail[-1][0], self.trail[-1][1]) != (x, y):
            self.trail.append((x, y, now))
        self.trail = [p for p in self.trail if now - p[2] < self.TRAIL_SECONDS]
        px, py = self._px(x, y)
        # clamp into view: an unhomed machine reports positions outside the
        # envelope, which used to park the dot invisibly off-canvas
        px = min(self.W - 6, max(6, px))
        py = min(self.H - 6, max(6, py))
        self.coords(self.dot, px - 5, py - 5, px + 5, py + 5)
        if len(self.trail) >= 2:
            pts = []
            for tx, ty, _ in self.trail:
                pts.extend(self._px(tx, ty))
            self.coords(self.trail_line, *pts)
            self.itemconfig(self.trail_line, state="normal")
        self.after(50, self._tick)


class LinkageView(tk.Canvas):
    """Left-arm workspace: a SQUARE pad drives the joints; the skeleton
    animates beside it for reference. Two pad mappings:

      joint-space (default) — pad x -> shoulder, pad y -> elbow, linear
        over each joint's range. Every pad point is a valid pose; corners
        are the extremes. Square because it IS square, in joint coords.

      calibrated — you drive the physical wrist to 9 points of a real
        square (taped next to the arm), pressing Set at each; bilinear
        interpolation between the captured poses makes pad-square =
        PHYSICAL square, the linkage's mechanical nonlinearity baked into
        the samples. No geometric model needed. Persists in
        motor_panel/arm_calibration.json.
    """

    W, H = 520, 250
    PAD = (14, 40, 194, 220)  # square drag surface (x0, y0, x1, y1)
    P0 = (330, 200)    # skeleton pivot
    L1, L2 = 105, 85   # stylized segment lengths (visual only)
    VISUAL_SWEEP = 70.0
    # Sign of each joint's visual mapping vs the physical arm. July 11:
    # dragging read inverted on hardware, so both flipped. If ONE joint
    # still mirrors, flip only its sign.
    S_SIGN = -1.0
    E_SIGN = -1.0
    CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_calibration.json")
    CAL_NAMES = ["top-left", "top-center", "top-right",
                 "mid-left", "center", "mid-right",
                 "bottom-left", "bottom-center", "bottom-right"]

    def __init__(self, parent, lefthand: SerialDevice):
        super().__init__(parent, width=self.W, height=self.H, bg="#0f1a17", highlightthickness=0)
        self.lefthand = lefthand
        self.s_ch = lefthand.channels["shoulder"]
        self.e_ch = lefthand.channels["elbow"]
        span = max(self.s_ch.hi - self.s_ch.lo, self.e_ch.hi - self.e_ch.lo, 1)
        self.vg = self.VISUAL_SWEEP / span
        self.s_scale = 1.0  # joint-space range fractions (sensitivity knobs)
        self.e_scale = 1.0
        self.mode = "joint"
        self.calib = self._load_calib()
        self.calibrating = None  # index into CAL_NAMES while capturing

        x0, y0, x1, y1 = self.PAD
        self.create_rectangle(x0, y0, x1, y1, outline="#3d6b5c")
        for f in (1 / 3, 2 / 3):
            self.create_line(x0 + (x1 - x0) * f, y0, x0 + (x1 - x0) * f, y1, fill="#1e3a30")
            self.create_line(x0, y0 + (y1 - y0) * f, x1, y0 + (y1 - y0) * f, fill="#1e3a30")
        self.pad_label = self.create_text((x0 + x1) // 2, y0 - 14, text="", fill="#667", font=("monospace", 8))
        self.pad_dot = self.create_oval(0, 0, 0, 0, fill="#ffeaa7", outline="")

        self.zone = self.create_polygon(0, 0, 0, 0, fill="#1c3a30", outline="#2f5c4c")
        self._draw_reach_shade()
        self.create_text(self.P0[0] + 30, 12, text="skeleton (reference)", fill="#667", font=("monospace", 8))
        self.bone1 = self.create_line(0, 0, 0, 0, fill="#0f9b8e", width=5, capstyle="round")
        self.bone2 = self.create_line(0, 0, 0, 0, fill="#12b3a4", width=4, capstyle="round")
        self.joint = self.create_oval(0, 0, 0, 0, fill="#dfe6e9", outline="")
        self.wrist = self.create_oval(0, 0, 0, 0, fill="#ffeaa7", outline="")
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self._update_pad_label()
        self._tick()

    # --- mapping --------------------------------------------------------------
    def _eff_range(self, ch, scale: float):
        lo = ch.neutral - (ch.neutral - ch.lo) * scale
        hi = ch.neutral + (ch.hi - ch.neutral) * scale
        return lo, max(hi, lo + 1)

    def map_uv(self, u: float, v: float):
        """Pad (u right, v up, both 0..1) -> (shoulder, elbow)."""
        if self.mode == "calibrated" and self.calib:
            gx, gy = min(u * 2, 1.999), min(v * 2, 1.999)
            ix, iy = int(gx), int(gy)
            fx, fy = gx - ix, gy - iy
            g = self.calib  # g[iy][ix] = (s, e), iy 0 = bottom row
            s = ((g[iy][ix][0] * (1 - fx) + g[iy][ix + 1][0] * fx) * (1 - fy)
                 + (g[iy + 1][ix][0] * (1 - fx) + g[iy + 1][ix + 1][0] * fx) * fy)
            e = ((g[iy][ix][1] * (1 - fx) + g[iy][ix + 1][1] * fx) * (1 - fy)
                 + (g[iy + 1][ix][1] * (1 - fx) + g[iy + 1][ix + 1][1] * fx) * fy)
            return s, e
        s_lo, s_hi = self._eff_range(self.s_ch, self.s_scale)
        e_lo, e_hi = self._eff_range(self.e_ch, self.e_scale)
        return s_lo + (s_hi - s_lo) * u, e_lo + (e_hi - e_lo) * v

    def inv_uv(self, s: float, e: float):
        """(shoulder, elbow) -> pad (u, v) for the position dot."""
        if self.mode == "calibrated" and self.calib:
            best, best_d = (0.5, 0.5), float("inf")
            for i in range(21):
                for j in range(21):
                    u, v = i / 20, j / 20
                    cs, ce = self.map_uv(u, v)
                    d = (cs - s) ** 2 + (ce - e) ** 2
                    if d < best_d:
                        best, best_d = (u, v), d
            return best
        s_lo, s_hi = self._eff_range(self.s_ch, self.s_scale)
        e_lo, e_hi = self._eff_range(self.e_ch, self.e_scale)
        u = (s - s_lo) / (s_hi - s_lo)
        v = (e - e_lo) / (e_hi - e_lo)
        return min(1, max(0, u)), min(1, max(0, v))

    def _drag(self, ev):
        x0, y0, x1, y1 = self.PAD
        if not (x0 - 10 <= ev.x <= x1 + 10 and y0 - 10 <= ev.y <= y1 + 10):
            return
        u = min(1, max(0, (ev.x - x0) / (x1 - x0)))
        v = min(1, max(0, 1 - (ev.y - y0) / (y1 - y0)))
        s, e = self.map_uv(u, v)
        self.lefthand.set_channel("shoulder", s)  # smoothed + queued downstream
        self.lefthand.set_channel("elbow", e)

    # --- calibration ------------------------------------------------------------
    def start_calibration(self) -> str:
        self.calibrating = 0
        self._grid_wip = [[None] * 3 for _ in range(3)]
        return f"drive the wrist to the {self.CAL_NAMES[0]} of your physical square, then Set"

    def capture_point(self) -> str:
        if self.calibrating is None:
            return "not calibrating — press Calibrate first"
        i = self.calibrating
        ix, iy = i % 3, 2 - i // 3  # CAL_NAMES go top-first; grid row 0 = bottom
        self._grid_wip[iy][ix] = (float(self.s_ch.value), float(self.e_ch.value))
        self.calibrating += 1
        if self.calibrating >= 9:
            self.calib = self._grid_wip
            self.calibrating = None
            with open(self.CALIB_PATH, "w") as f:
                json.dump({"grid": self.calib}, f)
            self.set_mode("calibrated")
            return "calibrated — pad square now maps to your physical square"
        return f"point {self.calibrating + 1}/9: {self.CAL_NAMES[self.calibrating]}, then Set"

    def _load_calib(self):
        try:
            with open(self.CALIB_PATH) as f:
                return json.load(f)["grid"]
        except Exception:
            return None

    def set_mode(self, mode: str) -> str:
        if mode == "calibrated" and not self.calib:
            return "no calibration yet — run Calibrate first"
        self.mode = mode
        self._update_pad_label()
        return f"pad mapping: {mode}"

    def _update_pad_label(self):
        text = "pad — joint-space (x=shoulder y=elbow)" if self.mode == "joint" else "pad — calibrated physical square"
        self.itemconfig(self.pad_label, text=text)

    # --- drawing ----------------------------------------------------------------
    def set_scale(self, joint: str, frac: float):
        if joint == "shoulder":
            self.s_scale = frac
        else:
            self.e_scale = frac

    def _angles(self, s: float, e: float):
        a1 = math.radians(-70 + (s - self.s_ch.neutral) * self.vg * self.S_SIGN)
        a2 = a1 + math.radians(45 + (e - self.e_ch.neutral) * self.vg * self.E_SIGN)
        return a1, a2

    def _wrist_px(self, s: float, e: float):
        a1, a2 = self._angles(s, e)
        ex = self.P0[0] + self.L1 * math.cos(a1)
        ey = self.P0[1] + self.L1 * math.sin(a1)
        return ex + self.L2 * math.cos(a2), ey + self.L2 * math.sin(a2)

    def _draw_reach_shade(self):
        s_lo, s_hi, e_lo, e_hi = self.s_ch.lo, self.s_ch.hi, self.e_ch.lo, self.e_ch.hi
        pts = ([self._wrist_px(s, e_lo) for s in range(s_lo, s_hi + 1)]
               + [self._wrist_px(s_hi, e) for e in range(e_lo, e_hi + 1)]
               + [self._wrist_px(s, e_hi) for s in range(s_hi, s_lo - 1, -1)]
               + [self._wrist_px(s_lo, e) for e in range(e_hi, e_lo - 1, -1)])
        self.coords(self.zone, *[c for p in pts for c in p])

    def _tick(self):
        s, e = self.s_ch.value, self.e_ch.value
        a1, a2 = self._angles(s, e)
        ex = self.P0[0] + self.L1 * math.cos(a1)
        ey = self.P0[1] + self.L1 * math.sin(a1)
        wx, wy = ex + self.L2 * math.cos(a2), ey + self.L2 * math.sin(a2)
        self.coords(self.bone1, *self.P0, ex, ey)
        self.coords(self.bone2, ex, ey, wx, wy)
        self.coords(self.joint, ex - 4, ey - 4, ex + 4, ey + 4)
        self.coords(self.wrist, wx - 5, wy - 5, wx + 5, wy + 5)
        u, v = self.inv_uv(s, e)
        x0, y0, x1, y1 = self.PAD
        px = x0 + (x1 - x0) * u
        py = y0 + (y1 - y0) * (1 - v)
        self.coords(self.pad_dot, px - 5, py - 5, px + 5, py + 5)
        self.after(100, self._tick)


class HandPad(tk.Canvas):
    """Hand workspace — the cursor paradigm from the hand controller: four
    finger columns, vertical drag = curl. Dragging across columns sweeps the
    hand; neighbors follow at 30% for organic coupling."""

    W, H = 340, 220

    def __init__(self, parent, lefthand: SerialDevice):
        super().__init__(parent, width=self.W, height=self.H, bg="#1a1426", highlightthickness=0)
        self.lefthand = lefthand
        col_w = self.W / 4
        self.bars = []
        for i in range(4):
            x = i * col_w
            self.create_rectangle(x + 4, 20, x + col_w - 4, self.H - 8, outline="#3a2f52")
            self.create_text(x + col_w / 2, 10, text=f"f{i}", fill="#667", font=("monospace", 8))
            self.bars.append(self.create_rectangle(x + 8, 0, x + col_w - 8, 0, fill="#8a63d2", outline=""))
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self._tick()

    def _drag(self, ev):
        col = min(3, max(0, int(ev.x / (self.W / 4))))
        curl = 180 * min(1, max(0, 1 - (ev.y - 20) / (self.H - 28)))
        for i in range(4):
            ch = self.lefthand.channels[f"finger{i}"]
            blend = 1.0 if i == col else 0.3 if abs(i - col) == 1 else 0.0
            if blend:
                self.lefthand.set_channel(f"finger{i}", ch.value + (curl - ch.value) * blend)

    def _tick(self):
        col_w = self.W / 4
        for i in range(4):
            v = self.lefthand.channels[f"finger{i}"].value / 180.0
            x = i * col_w
            top = 20 + (self.H - 28) * (1 - v)
            self.coords(self.bars[i], x + 8, top, x + col_w - 8, self.H - 8)
        self.after(100, self._tick)


class LungStrip(tk.Canvas):
    """Lung workspace — breath as a scrolling waveform: vertical drag sets
    the lung position, and you SEE the rhythm you're performing."""

    W, H = 340, 220
    WINDOW = 12.0  # seconds of breath history shown

    def __init__(self, parent, lunggaze: SerialDevice):
        super().__init__(parent, width=self.W, height=self.H, bg="#0d1f26", highlightthickness=0)
        self.lunggaze = lunggaze
        self.ch = lunggaze.channels["lung"]
        self.history = []  # (t, value)
        self.create_text(self.W // 2, 10, text="lung — drag vertically, breathe with the wave",
                         fill="#667", font=("monospace", 8))
        self.wave = self.create_line(0, 0, 0, 0, fill="#3ba7a0", width=2, smooth=True)
        self.now_dot = self.create_oval(0, 0, 0, 0, fill="#ffeaa7", outline="")
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self._tick()

    def _y(self, value):
        f = (value - self.ch.lo) / max(1, self.ch.hi - self.ch.lo)
        return self.H - 14 - f * (self.H - 34)

    def _drag(self, ev):
        f = min(1, max(0, (self.H - 14 - ev.y) / (self.H - 34)))
        self.lunggaze.set_channel("lung", self.ch.lo + f * (self.ch.hi - self.ch.lo))

    def _tick(self):
        now = time.time()
        self.history.append((now, self.ch.value))
        self.history = [(t, v) for t, v in self.history if now - t < self.WINDOW]
        if len(self.history) >= 2:
            pts = []
            for t, v in self.history:
                x = self.W - 12 - (now - t) / self.WINDOW * (self.W - 24)
                pts.extend((x, self._y(v)))
            self.coords(self.wave, *pts)
        y = self._y(self.ch.value)
        self.coords(self.now_dot, self.W - 17, y - 4, self.W - 9, y + 4)
        self.after(80, self._tick)


class SessionFrame(ttk.LabelFrame):
    """The looper: tracks per subsystem, one workspace per track, layered
    recording against a fixed loop. Tracks share a group (trained as ONE
    joint chain — they move in relation) or go solo (own chain); Generate
    runs every chain simultaneously."""

    def __init__(self, parent, lunggaze: SerialDevice, lefthand: SerialDevice, grbl: GrblFrame, log):
        super().__init__(parent, text="body session  (layered choreography → joint markov)")
        self.lunggaze = lunggaze
        self.lefthand = lefthand
        self.grbl = grbl
        self.log = log
        self.session = Session()
        self._route = {c: lefthand for c in ("elbow", "shoulder", "finger0", "finger1", "finger2", "finger3")}
        self._route["lung"] = lunggaze
        self.transport = self._make_transport()

        # transport row
        tr = ttk.Frame(self)
        tr.pack(fill="x", padx=4, pady=2)
        ttk.Button(tr, text="● Record pass", command=self.record).pack(side="left", padx=2)
        ttk.Button(tr, text="▶ Play loop", command=self.play).pack(side="left", padx=2)
        ttk.Button(tr, text="■ Stop", command=self.transport_stop).pack(side="left", padx=2)
        ttk.Button(tr, text="∿ Generate", command=self.generate).pack(side="left", padx=2)
        ttk.Label(tr, text="loop").pack(side="left", padx=(10, 2))
        self.loop_var = tk.IntVar(value=int(self.session.loop_len))
        ttk.OptionMenu(tr, self.loop_var, int(self.session.loop_len), *LOOP_LENGTHS,
                       command=lambda v: setattr(self.session, "loop_len", float(v))).pack(side="left")
        ttk.Label(tr, text="speed").pack(side="left", padx=(10, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_lbl = ttk.Label(tr, text="1.0x", width=5)
        ttk.Scale(tr, from_=0.25, to=2.0, variable=self.speed_var, length=90,
                  command=lambda v: self.speed_lbl.config(text=f"{float(v):.2f}x")).pack(side="left")
        self.speed_lbl.pack(side="left")

        # session persistence row
        sr = ttk.Frame(self)
        sr.pack(fill="x", padx=4, pady=2)
        ttk.Label(sr, text="session").pack(side="left")
        self.name_entry = ttk.Entry(sr, width=16)
        self.name_entry.insert(0, self.session.name)
        self.name_entry.pack(side="left", padx=2)
        ttk.Button(sr, text="Save", command=self.save).pack(side="left", padx=2)
        self.saved_var = tk.StringVar()
        self.saved_menu = ttk.Combobox(sr, textvariable=self.saved_var, width=24, state="readonly")
        self.saved_menu.pack(side="left", padx=2)
        ttk.Button(sr, text="Load", command=self.load).pack(side="left", padx=2)
        self._refresh_saved()

        # track rows
        self.tracks_box = ttk.Frame(self)
        self.tracks_box.pack(fill="x", padx=4, pady=2)
        self._build_tracks()

        # workspaces
        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=4, pady=3)
        bed_tab = ttk.Frame(nb)
        self.bed = BedView(bed_tab, grbl)
        self.bed.pack()
        feed_row = ttk.Frame(bed_tab)
        feed_row.pack(fill="x", pady=2)
        ttk.Label(feed_row, text="max feed", font=("monospace", 8)).pack(side="left", padx=4)
        feed_lbl = ttk.Label(feed_row, text=str(int(grbl.max_feed)), width=5)

        def on_feed(v):
            grbl.max_feed = float(v)
            feed_lbl.config(text=str(int(float(v))))

        fs = ttk.Scale(feed_row, from_=200, to=3000, command=on_feed)
        fs.set(grbl.max_feed)
        fs.pack(side="left", fill="x", expand=True, padx=4)
        feed_lbl.pack(side="left")
        nb.add(bed_tab, text="right arm — bed")
        link_tab = ttk.Frame(nb)
        self.linkage = LinkageView(link_tab, lefthand)
        self.linkage.pack()
        knobs = ttk.Frame(link_tab)
        knobs.pack(fill="x", pady=2)
        for label, cb, lo, hi, init in (
            ("elbow range %", lambda v: self.linkage.set_scale("elbow", float(v) / 100), 10, 100, 100),
            ("shoulder range %", lambda v: self.linkage.set_scale("shoulder", float(v) / 100), 10, 100, 100),
            ("smoothing s", lambda v: setattr(lefthand, "smooth_time", float(v)), 0.05, 0.8, 0.25),
        ):
            box = ttk.Frame(knobs)
            box.pack(side="left", fill="x", expand=True, padx=3)
            ttk.Label(box, text=label, font=("monospace", 8)).pack()
            sc = ttk.Scale(box, from_=lo, to=hi, command=cb)
            sc.set(init)
            sc.pack(fill="x")

        cal = ttk.Frame(link_tab)
        cal.pack(fill="x", pady=2)
        self.cal_lbl = ttk.Label(cal, text="", font=("monospace", 8))

        def set_cal(msg):
            self.cal_lbl.config(text=msg)

        def toggle_mode():
            new = "calibrated" if self.linkage.mode == "joint" else "joint"
            set_cal(self.linkage.set_mode(new))
            mode_btn.config(text=f"mapping: {self.linkage.mode}")

        mode_btn = ttk.Button(cal, text=f"mapping: {self.linkage.mode}", command=toggle_mode)
        mode_btn.pack(side="left", padx=2)
        ttk.Button(cal, text="Calibrate 9-pt", command=lambda: set_cal(self.linkage.start_calibration())).pack(side="left", padx=2)

        def set_point():
            set_cal(self.linkage.capture_point())
            mode_btn.config(text=f"mapping: {self.linkage.mode}")

        ttk.Button(cal, text="Set point", command=set_point).pack(side="left", padx=2)
        self.cal_lbl.pack(side="left", padx=6)
        nb.add(link_tab, text="left arm — linkage")

        hand_tab = ttk.Frame(nb)
        self.hand_pad = HandPad(hand_tab, lefthand)
        self.hand_pad.pack()
        imp = ttk.Frame(hand_tab)
        imp.pack(fill="x", pady=2)
        ttk.Label(imp, text="legacy dataset", font=("monospace", 8)).pack(side="left", padx=2)
        self.legacy_var = tk.StringVar()
        legacy_menu = ttk.Combobox(imp, textvariable=self.legacy_var, width=34, state="readonly",
                                   values=list_legacy_hand_datasets())
        legacy_menu.pack(side="left", padx=2)

        def do_import():
            if not self.legacy_var.get():
                return
            try:
                samples = import_legacy_hand_take(self.legacy_var.get(), self.session.loop_len)
            except Exception as e:
                self.status.config(text=f"import failed: {e}")
                return
            for t in self.session.tracks:
                if t.name == "hand (fingers)":
                    t.samples = samples
            self._refresh_tracks()
            self.status.config(text=f"imported {self.legacy_var.get()} as hand take ({len(samples)} samples)")

        ttk.Button(imp, text="Import as hand take", command=do_import).pack(side="left", padx=2)
        nb.add(hand_tab, text="hand")

        lung_tab = ttk.Frame(nb)
        self.lung_strip = LungStrip(lung_tab, lunggaze)
        self.lung_strip.pack()
        nb.add(lung_tab, text="lung")

        self.status = ttk.Label(self, text="idle")
        self.status.pack(fill="x", padx=6, pady=2)

    # --- transport plumbing ---------------------------------------------------
    def _make_transport(self) -> Transport:
        def get_state():
            s = {c: dev.channels[c].value for c, dev in self._route.items()}
            s["x"], s["y"] = self.grbl.position
            return s

        return Transport(
            self.session,
            get_state=get_state,
            send_ease=lambda d: [self._route[c].set_channel(c, int(round(v))) for c, v in d.items()],
            send_plan=lambda d, dt: self.grbl.goto(d["x"], d["y"], dt),
            on_status=self._set_status,
        )

    def _set_status(self, msg: str):
        try:
            self.after(0, lambda: (self.status.config(text=msg), self._refresh_tracks()))
        except RuntimeError:
            pass

    def record(self):
        self.grbl.set_pen(GRBL_PEN_UP_S)  # never draw during choreography
        self.transport.record()

    def play(self):
        self.grbl.set_pen(GRBL_PEN_UP_S)
        self.transport.play(speed=self.speed_var.get())

    def generate(self):
        self.grbl.set_pen(GRBL_PEN_UP_S)
        self.transport.generate(speed=self.speed_var.get())

    def transport_stop(self):
        self.transport.stop()

    # --- tracks ----------------------------------------------------------------
    def _build_tracks(self):
        for w in self.tracks_box.winfo_children():
            w.destroy()
        self._track_widgets = []
        for t in self.session.tracks:
            row = ttk.Frame(self.tracks_box)
            row.pack(fill="x")
            arm_var = tk.BooleanVar(value=t.armed)
            mute_var = tk.BooleanVar(value=t.mute)
            ttk.Checkbutton(row, text="arm", variable=arm_var,
                            command=lambda t=t, v=arm_var: setattr(t, "armed", v.get())).pack(side="left")
            ttk.Checkbutton(row, text="mute", variable=mute_var,
                            command=lambda t=t, v=mute_var: setattr(t, "mute", v.get())).pack(side="left")
            group_var = tk.StringVar(value=t.group)
            ttk.OptionMenu(row, group_var, t.group, *GROUPS,
                           command=lambda v, t=t: setattr(t, "group", v)).pack(side="left", padx=2)
            take_lbl = tk.Label(row, text="●", fg="green" if t.has_take else "gray")
            take_lbl.pack(side="left", padx=4)
            ttk.Label(row, text=f"{t.name}  [{', '.join(t.channels)}]").pack(side="left")
            ttk.Button(row, text="clear", width=5,
                       command=lambda t=t: (setattr(t, "samples", None), self._refresh_tracks())).pack(side="right")
            self._track_widgets.append((t, arm_var, mute_var, take_lbl))

    def _refresh_tracks(self):
        for t, arm_var, mute_var, take_lbl in self._track_widgets:
            arm_var.set(t.armed)
            mute_var.set(t.mute)
            take_lbl.config(fg="green" if t.has_take else "gray")

    # --- persistence -------------------------------------------------------------
    def _refresh_saved(self):
        self.saved_menu["values"] = Session.list_saved()

    def save(self):
        self.session.name = self.name_entry.get().strip() or "session"
        path = self.session.save()
        self.log("session", f"saved {os.path.basename(path)}", False)
        self._refresh_saved()

    def load(self):
        if not self.saved_var.get():
            return
        self.transport.stop()
        self.session = Session.load(self.saved_var.get())
        self.loop_var.set(int(self.session.loop_len))
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, self.session.name)
        self.transport = self._make_transport()
        self._build_tracks()
        self.status.config(text=f"loaded {self.session.name}")

    def shutdown(self):
        self.transport.stop()


def main():
    root = tk.Tk()
    root.title("mslint — unified motor panel")

    warn = tk.Label(root, text="⚠ stop machine.py before connecting — serial ports are exclusive",
                    fg="darkorange")
    warn.pack(fill="x", pady=2)

    console = scrolledtext.ScrolledText(root, height=9, state="disabled", font=("monospace", 9))

    def log(device: str, line: str, simulated: bool):
        def append():
            console.config(state="normal")
            tag = "[sim] " if simulated else ""
            console.insert("end", f"{device:>10} | {tag}{line}\n")
            console.see("end")
            console.config(state="disabled")
        try:
            root.after(0, append)
        except RuntimeError:
            pass

    frames = []
    devices = build_devices()
    cols = ttk.Frame(root)
    cols.pack(fill="both", expand=True)
    left = ttk.Frame(cols)
    left.pack(side="left", fill="both", expand=True)
    right = ttk.Frame(cols)
    right.pack(side="left", fill="both", expand=True)

    for device, parent in zip(devices, [left, left, right]):
        f = DeviceFrame(parent, device, log)
        f.pack(fill="x", padx=4, pady=3)
        frames.append(f)
    grbl = GrblFrame(right, log)
    grbl.pack(fill="x", padx=4, pady=3)
    body = SessionFrame(left, devices[0], devices[1], grbl, log)  # lunggaze, lefthand
    body.pack(fill="x", padx=4, pady=3)

    bottom = ttk.Frame(root)
    bottom.pack(fill="x")
    def everything_neutral():
        for d in devices:
            d.all_neutral()
    ttk.Button(bottom, text="ALL NEUTRAL", command=everything_neutral).pack(side="left", padx=4, pady=2)

    console.pack(fill="both", expand=False, padx=4, pady=3)

    def on_close():
        body.shutdown()
        for d in devices:
            d.disconnect()
        if grbl.ser is not None:
            try:
                grbl.ser.close()
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
