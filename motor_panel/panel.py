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

from config.config import ARMS_DUET_MAX_FEED, GRBL_PEN_DOWN_S, GRBL_PEN_UP_S, KINETIC_BUS_ENABLED
from grbl.warp_calibration import clamp_to_reach, reach_polygon
from motor_panel.devices import EMOTIONS, SerialDevice, build_devices
from motor_panel.kinetic_bus import STATES, KineticBus, TemperamentLibrary
from motor_panel.session import Session, Transport, import_legacy_hand_take, list_legacy_hand_datasets

JOG_STEPS = [0.5, 1, 2, 5, 10]
LOOP_LENGTHS = [15, 30, 45, 60]
# the lab hosts the runtime bus with the FULL body (runtime v1 owns lefthand only)
LAB_CHANNELS = {"x", "y", "pen", "elbow", "shoulder", "wrist", "finger0", "finger1", "finger2", "finger3", "lung"}


def labeled_slider(parent, label, lo, hi, init, cb, fmt=lambda v: f"{v:.2f}"):
    """Side-panel slider: caption above, live value readout beside."""
    box = ttk.Frame(parent)
    box.pack(fill="x", pady=3)
    ttk.Label(box, text=label, font=("monospace", 8)).pack(anchor="w")
    row = ttk.Frame(box)
    row.pack(fill="x")
    val = ttk.Label(row, text=fmt(float(init)), width=6)

    def on_move(v):
        cb(float(v))
        val.config(text=fmt(float(v)))

    sc = ttk.Scale(row, from_=lo, to=hi, command=on_move)
    sc.set(init)
    sc.pack(side="left", fill="x", expand=True)
    val.pack(side="left")
    return sc


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
        self._syncing = False  # True while _sync_tick writes vars — their command must not echo back to the device
        for name in device.channel_order:
            ch = device.channels[name]
            row = ttk.Frame(self)
            row.pack(fill="x", padx=4)
            ttk.Label(row, text=name, width=9).pack(side="left")
            var = tk.IntVar(value=ch.neutral)
            val_lbl = ttk.Label(row, text=str(ch.neutral), width=4)

            def on_move(v, n=name, vl=val_lbl):
                vl.config(text=str(int(float(v))))
                if not self._syncing:
                    self.device.set_channel(n, int(float(v)))  # non-blocking: writer queue

            s = ttk.Scale(row, from_=ch.lo, to=ch.hi, variable=var, command=on_move)
            s.pack(side="left", fill="x", expand=True, padx=4)
            val_lbl.pack(side="left")
            rev_var = tk.BooleanVar(value=ch.invert)
            ttk.Checkbutton(row, text="rev", variable=rev_var, command=lambda n=name, v=rev_var: self.device.set_invert(n, v.get())).pack(side="left")
            self.sliders[name] = (s, var, val_lbl)

        if device.extras:
            ex = ttk.Frame(self)
            ex.pack(fill="x", padx=4, pady=2)
            row = None
            for idx, label in enumerate(device.extras):
                if idx % 3 == 0:
                    row = ttk.Frame(ex)
                    row.pack(fill="x")
                ttk.Button(row, text=label, command=lambda l=label: self.device.send_extra(l)).pack(side="left", padx=2, pady=1)
        self._sync_tick()

    def _sync_tick(self):
        """Sliders follow the CHANNELS, not just the hand that drags them:
        playback/generation/lab moves used to leave every slider stale,
        which read as broken. Writing the same value back is free (the
        writer queue dedupes identical lines)."""
        self._syncing = True
        try:
            for name, (s, var, val_lbl) in self.sliders.items():
                v = self.device.channels[name].value
                if var.get() != v:
                    var.set(v)
                    val_lbl.config(text=str(int(v)))
        finally:
            self._syncing = False
        self.after(300, self._sync_tick)

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
        # homing hooks: main() wires these to the kinetic bus so the left
        # arm tucks clear before the gantry homes and returns after
        self.on_home = None
        self.on_home_done = None
        ttk.Button(top, text="Home $H", command=self._home_clicked).pack(side="left", padx=2)
        ttk.Button(top, text="Unlock $X", command=lambda: self.send("$X")).pack(side="left", padx=2)
        ttk.Button(top, text="Status ?", command=lambda: self.send("?")).pack(side="left", padx=2)
        self.state_lbl = ttk.Label(top, text="", width=1, anchor="w")  # GRBL status lines vary wildly — never let them push the row
        self.state_lbl.pack(side="left", fill="x", expand=True, padx=8)
        self.state_text = ""  # writer thread writes this; the label polls it (no cross-thread Tk)

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
        self.pen_s = GRBL_PEN_UP_S  # commanded pen state — plain attr so worker threads can read/write it
        self.pen_lbl = ttk.Label(pen, text=str(GRBL_PEN_UP_S), width=4)

        def on_pen(v):
            v = int(float(v))
            self.pen_lbl.config(text=str(v))
            self.pen_command(v)

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
        self._label_tick()  # after all polled widgets exist (state label + pen slider)

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
                if cmd == "$H" and self.on_home_done:
                    try:
                        self.on_home_done()  # release the tucked left arm (writer thread — no Tk in the callback)
                    except Exception:
                        pass
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
        if self.pen_var.get() != self.pen_s:  # playback/generation moved the pen — sync the slider
            self.pen_var.set(self.pen_s)
        self.after(150, self._label_tick)

    def _home_clicked(self):
        delay = 0.0
        if self.on_home:
            try:
                delay = float(self.on_home() or 0.0)  # seconds the tuck ramp needs
            except Exception:
                delay = 0.0
        if delay > 0:
            # the arm must be CLEAR before the gantry sweeps — $H waits out
            # the gentle tuck instead of racing it
            self.log("grbl", f"homing in {delay:.1f}s — left arm tucking clear", False)
            self.after(int(delay * 1000), lambda: self.send("$H"))
        else:
            self.send("$H")

    def jog(self, dx: int, dy: int):
        """Computed absolute target — never G91: an out-of-order or
        interleaved modal switch strands the machine in relative mode and
        every later absolute move becomes a wild relative lunge. Sent FIFO
        (not coalesced) so rapid clicks compound; position is updated
        optimistically and corrected by the status poll. Targets clamp into
        the measured reach polygon — a jog can only creep back toward the
        envelope, never past a joint stop."""
        d = self.step.get()
        x, y = clamp_to_reach(self.position[0] + dx * d, self.position[1] + dy * d)
        self.position = (x, y)
        self.send(f"G0 X{x:.2f} Y{y:.2f}")  # rapid — jogs are travel moves

    def goto(self, x: float, y: float, dt: float = None):
        """Absolute move into the MEASURED reach envelope: every target —
        drag, playback, generation — projects into the walked polygon
        (grbl.warp_calibration.MEASURED_BOUNDARY, hardware truth July 20).
        The polygon is convex, so straight moves between clamped points
        stay inside; nothing the panel emits can grind a joint stop.

        dt=None (live performance — pad drags, jogs): G0 RAPID. Dragging is
        pen-up travel, and rapids are why deployment/homing/node-jumps feel
        fast — G1 at any F is the deliberate drawing gait, and no streaming
        optimization changes which gait it is (July 12 lesson). dt given
        (playback / generation): G1 with feed derived from recorded timing
        so takes keep their performed tempo."""
        if self.alarm:
            return  # locked controller — motion is meaningless until $H
        x, y = clamp_to_reach(x, y)
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

    def pen_command(self, s: int):
        """Thread-safe pen move (playback/generation call this off-thread):
        no Tk access, deduped so replayed step samples cost nothing."""
        s = int(s)
        if s == self.pen_s:
            return
        self.pen_s = s
        self.send(f"M3 S{s}")

    def pen_is_down(self) -> bool:
        mid = (GRBL_PEN_UP_S + GRBL_PEN_DOWN_S) / 2
        return (self.pen_s > mid) == (GRBL_PEN_DOWN_S > GRBL_PEN_UP_S)

    def set_pen(self, s: int):
        self.pen_var.set(s)
        self.pen_lbl.config(text=str(s))
        self.pen_command(s)

    def send_raw(self):
        cmd = self.raw_entry.get().strip()
        if cmd:
            self.send(cmd)
            self.raw_entry.delete(0, "end")


class BedView(tk.Canvas):
    """Right-arm workspace: the MEASURED reach envelope drawn to scale
    (uniform mm-per-px, aspect-true), live position, a fading trail.
    Left-drag to perform — every target projects into the walked polygon
    (grbl.warp_calibration.MEASURED_BOUNDARY), so the cursor physically
    cannot ask for a pose past a joint stop. HOLD the right button to put
    the pen down (it draws — the pen is a recordable layer). Pen-down
    drags move at the performed tempo (G1), pen-up drags stay rapids."""

    TRAIL_SECONDS = 10.0
    PAD_UNITS = 4.0  # view padding around the envelope, command units
    GRID = 20.0  # reference grid spacing, command units

    def __init__(self, parent, grbl: GrblFrame, w=340, h=250):
        self.W, self.H = w, h
        self.M = 24
        boundary = reach_polygon()
        xs, ys = [p[0] for p in boundary], [p[1] for p in boundary]
        self.x0, self.x1 = min(xs) - self.PAD_UNITS, max(xs) + self.PAD_UNITS
        self.y0, self.y1 = min(ys) - self.PAD_UNITS, max(ys) + self.PAD_UNITS
        # one scale for both axes: the polygon keeps its true shape
        self.scale = min((self.W - 2 * self.M) / (self.x1 - self.x0), (self.H - 2 * self.M) / (self.y1 - self.y0))
        self.ox = (self.W - (self.x1 - self.x0) * self.scale) / 2
        self.oy = (self.H - (self.y1 - self.y0) * self.scale) / 2
        super().__init__(parent, width=self.W, height=self.H, bg="#101422", highlightthickness=0, cursor="crosshair")
        self.grbl = grbl
        self.trail = []  # (x, y, t)
        g = self.GRID
        gx = math.ceil(self.x0 / g) * g
        while gx <= self.x1:  # reference grid every GRID command units
            self.create_line(*self._px(gx, self.y0), *self._px(gx, self.y1), fill="#1c2438")
            gx += g
        gy = math.ceil(self.y0 / g) * g
        while gy <= self.y1:
            self.create_line(*self._px(self.x0, gy), *self._px(self.x1, gy), fill="#1c2438")
            gy += g
        # walked envelope (fill) and the actual clamp polygon (dashed, inset)
        self.create_polygon(*[c for p in boundary for c in self._px(*p)], fill="#151d33", outline="#3a4a6b")
        self.create_polygon(*[c for p in reach_polygon(0.5) for c in self._px(*p)], fill="", outline="#2f5c4c", dash=(3, 4))
        hx, hy = self._px(0.0, 0.0)
        self.create_text(hx + 4, hy + 10, anchor="w", text="0,0", fill="#556", font=("monospace", 8))
        self.create_text(self.W // 2, 12, text="measured reach — targets clamp inside   ✛ = target   ● = machine", fill="#667", font=("monospace", 8))
        self.trail_line = self.create_line(0, 0, 0, 0, fill="#7a3448", smooth=True, width=2, state="hidden")
        # commanded target (where you asked it to go) vs reported position
        self.target_h = self.create_line(0, 0, 0, 0, fill="#f5c04a", width=1)
        self.target_v = self.create_line(0, 0, 0, 0, fill="#f5c04a", width=1)
        self.dot = self.create_oval(0, 0, 0, 0, fill="#e94560", outline="")
        self.pen_ind = self.create_text(
            self.M, self.H - 10, anchor="w", text="pen ▲  (hold right button to draw)", fill="#667", font=("monospace", 8)
        )
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self.bind("<ButtonPress-3>", lambda e: self.grbl.set_pen(GRBL_PEN_DOWN_S))
        self.bind("<ButtonRelease-3>", lambda e: self.grbl.set_pen(GRBL_PEN_UP_S))
        self._last_drag_t = None
        self._tick()

    def _px(self, x: float, y: float):
        px = self.ox + (x - self.x0) * self.scale
        py = self.H - self.oy - (y - self.y0) * self.scale
        return px, py

    def _from_px(self, px: float, py: float):
        x = self.x0 + (px - self.ox) / self.scale
        y = self.y0 + (self.H - self.oy - py) / self.scale
        return x, y

    def _drag(self, ev):
        # project into the clamp polygon HERE too, so the crosshair shows
        # the target the machine will actually get, not the raw cursor
        x, y = clamp_to_reach(*self._from_px(ev.x, ev.y))
        px, py = self._px(x, y)
        self.coords(self.target_h, px - 7, py, px + 7, py)
        self.coords(self.target_v, px, py - 7, px, py + 7)
        now = time.time()
        if self.grbl.pen_is_down() and self._last_drag_t is not None:
            # ink on paper: G1 at the tempo you're actually drawing, not a rapid lunge
            self.grbl.goto(x, y, min(0.5, max(0.05, now - self._last_drag_t)))
        else:
            self.grbl.goto(x, y)  # pen-up travel: rapids
        self._last_drag_t = now

    def _tick(self):
        now = time.time()
        down = self.grbl.pen_is_down()
        self.itemconfig(self.trail_line, fill="#dfe6e9" if down else "#7a3448")
        self.itemconfig(self.pen_ind, text="pen ▼ DRAWING" if down else "pen ▲  (hold right button to draw)", fill="#dfe6e9" if down else "#667")
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

    BASE_W, BASE_H = 520, 250  # the proven layout; everything scales uniformly from it
    VISUAL_SWEEP = 70.0
    # Sign of each joint's visual mapping vs the physical arm. July 11:
    # dragging read inverted on hardware, so both flipped. If ONE joint
    # still mirrors, flip only its sign.
    S_SIGN = -1.0
    E_SIGN = -1.0
    CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_calibration.json")
    CAL_NAMES = ["top-left", "top-center", "top-right", "mid-left", "center", "mid-right", "bottom-left", "bottom-center", "bottom-right"]

    def __init__(self, parent, lefthand: SerialDevice, w=520, h=250):
        k = min(w / self.BASE_W, h / self.BASE_H)
        self.W, self.H = int(self.BASE_W * k), int(self.BASE_H * k)
        self.PAD = tuple(int(c * k) for c in (14, 40, 194, 220))  # square drag surface (x0, y0, x1, y1)
        self.P0 = (int(330 * k), int(200 * k))  # skeleton pivot
        self.L1, self.L2 = 105 * k, 85 * k  # stylized segment lengths (visual only)
        super().__init__(parent, width=self.W, height=self.H, bg="#0f1a17", highlightthickness=0)
        self.lefthand = lefthand
        self.s_ch = lefthand.channels["shoulder"]
        self.e_ch = lefthand.channels["elbow"]
        self.w_ch = lefthand.channels["wrist"]
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
        # wrist (pin 6): scroll wheel over the canvas rotates it — drag drives
        # shoulder/elbow, the wheel is the third joint in the same hand
        self.hand_seg = self.create_line(0, 0, 0, 0, fill="#f5c04a", width=3, capstyle="round")
        x0, y0, x1, y1 = self.PAD
        self.wrist_lbl = self.create_text((x0 + x1) // 2, y1 + 12, text="", fill="#667", font=("monospace", 8))
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self.bind("<Button-4>", lambda e: self._wrist_nudge(+1))  # Linux wheel up
        self.bind("<Button-5>", lambda e: self._wrist_nudge(-1))
        self.bind("<MouseWheel>", lambda e: self._wrist_nudge(1 if e.delta > 0 else -1))
        self.wheel_step = 3.0  # degrees per wheel notch — the wrist's sensitivity knob
        self.w_scale = 1.0  # wrist range fraction around neutral
        self._update_pad_label()
        self._tick()

    def _wrist_nudge(self, direction: float):
        lo, hi = self._eff_range(self.w_ch, self.w_scale)
        target = self.w_ch.target + direction * self.wheel_step  # target, not value: wheel outruns the smoother
        self.lefthand.set_channel("wrist", max(lo, min(hi, target)))

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
            s = (g[iy][ix][0] * (1 - fx) + g[iy][ix + 1][0] * fx) * (1 - fy) + (g[iy + 1][ix][0] * (1 - fx) + g[iy + 1][ix + 1][0] * fx) * fy
            e = (g[iy][ix][1] * (1 - fx) + g[iy][ix + 1][1] * fx) * (1 - fy) + (g[iy + 1][ix][1] * (1 - fx) + g[iy + 1][ix + 1][1] * fx) * fy
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
        elif joint == "wrist":
            self.w_scale = frac
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
        pts = (
            [self._wrist_px(s, e_lo) for s in range(s_lo, s_hi + 1)]
            + [self._wrist_px(s_hi, e) for e in range(e_lo, e_hi + 1)]
            + [self._wrist_px(s, e_hi) for s in range(s_hi, s_lo - 1, -1)]
            + [self._wrist_px(s_lo, e) for e in range(e_hi, e_lo - 1, -1)]
        )
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
        w = self.w_ch.value
        span = max(1, self.w_ch.hi - self.w_ch.lo)
        a3 = a2 + math.radians((w - self.w_ch.neutral) / span * 90)  # hand segment rotates with the wrist
        hx, hy = wx + self.L2 * 0.45 * math.cos(a3), wy + self.L2 * 0.45 * math.sin(a3)
        self.coords(self.hand_seg, wx, wy, hx, hy)
        self.itemconfig(self.wrist_lbl, text=f"wrist {int(w)}°  (scroll wheel — own lane, overdub it)")
        u, v = self.inv_uv(s, e)
        x0, y0, x1, y1 = self.PAD
        px = x0 + (x1 - x0) * u
        py = y0 + (y1 - y0) * (1 - v)
        self.coords(self.pad_dot, px - 5, py - 5, px + 5, py + 5)
        self.after(100, self._tick)


class HandSpace(tk.Canvas):
    """Hand workspace — the full cursor paradigm from the original hand
    controller (hand_control_interface.py), distilled. A free cursor over
    four finger home columns: each finger follows the cursor's height in
    proportion to horizontal proximity (the gravity/wave field) and relaxes
    to the default curl outside it, so sweeping across the hand makes a
    wave. Per-finger keyboard control (w/s e/d r/f t/g) locks a finger
    while its keys are held, releasing it back to the field on key-up.

    Control engages only while the pointer is over this canvas — leave and
    the hand holds its pose, and playback/generation own the channels.
    """

    HOME_START, HOME_SPAN = 0.25, 0.5  # finger columns condensed to mid-canvas (matches the original)
    KEYS = {"w": (0, +1), "s": (0, -1), "e": (1, +1), "d": (1, -1), "r": (2, +1), "f": (2, -1), "t": (3, +1), "g": (3, -1)}
    STEP = 2.0  # degrees per tick while a key is held
    TOP, BOT = 30, 34  # bar margins (labels above, degrees below)

    def __init__(self, parent, lefthand: SerialDevice, w=340, h=220):
        super().__init__(parent, width=w, height=h, bg="#1a1426", highlightthickness=0, cursor="crosshair")
        self.W, self.H = w, h
        self.lefthand = lefthand
        # the proven field parameters (defaults = the old interface's working values)
        self.sensitivity = tk.DoubleVar(value=3.0)
        self.wave = tk.DoubleVar(value=2.0)
        self.gravity = tk.DoubleVar(value=0.4)
        self.default_pos = tk.DoubleVar(value=90.0)
        self.servo_range = tk.DoubleVar(value=60.0)
        self.reverse = tk.BooleanVar(value=True)
        self.mouse = None  # (u, v) normalized; None while the pointer is away
        self.pressed = set()
        self.locked = [False] * 4
        self.lock_target = [90.0] * 4
        self.key_step = self.STEP  # degrees per tick while a key is held — keyboard sensitivity

        self.create_text(
            self.W // 2,
            12,
            fill="#667",
            font=("monospace", 9),
            text="hover to perform — cursor is the wave, keys w/s e/d r/f t/g grab single fingers",
        )
        self.col_rects, self.bars, self.deg_lbls, self.key_lbls = [], [], [], []
        half = self.W * self.HOME_SPAN / 4 * 0.42
        for i in range(4):
            cx = self._home_u(i) * self.W
            self.col_rects.append(self.create_rectangle(cx - half, self.TOP, cx + half, self.H - self.BOT, outline="#3a2f52"))
            self.create_text(cx, self.TOP - 8, text=f"f{i}", fill="#889", font=("monospace", 9))
            keys = [k for k, (fi, _) in self.KEYS.items() if fi == i]
            self.key_lbls.append(
                self.create_text(cx, self.H - self.BOT + 12, text="/".join(sorted(keys, reverse=True)), fill="#554a6e", font=("monospace", 8))
            )
            self.bars.append(self.create_rectangle(cx - half + 3, 0, cx + half - 3, 0, fill="#8a63d2", outline=""))
            self.deg_lbls.append(self.create_text(cx, self.H - self.BOT + 24, text="90°", fill="#667", font=("monospace", 8)))
        self.default_line = self.create_line(0, 0, 0, 0, fill="#3a4a3a", dash=(3, 4))
        self.gravity_line = self.create_line(0, 0, 0, 0, fill="#4a5a7a", width=2, state="hidden")
        self.cursor_h = self.create_line(0, 0, 0, 0, fill="#ffeaa7", state="hidden")
        self.cursor_v = self.create_line(0, 0, 0, 0, fill="#ffeaa7", state="hidden")
        self.bind("<Motion>", self._motion)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<KeyPress>", self._key_down)
        self.bind("<KeyRelease>", self._key_up)
        self._tick()

    def _home_u(self, i: int) -> float:
        return self.HOME_START + (i + 0.5) / 4 * self.HOME_SPAN

    def _bar_y(self, deg: float) -> float:
        return self.TOP + (self.H - self.TOP - self.BOT) * (1 - deg / 180.0)

    def _enter(self, ev):
        self.focus_set()  # keyboard finger control while hovering
        self._motion(ev)

    def _leave(self, _ev):
        self.mouse = None  # hand holds its pose; playback/generation regain the channels
        self.pressed.clear()
        self.locked = [False] * 4

    def _motion(self, ev):
        self.mouse = (min(1, max(0, ev.x / self.W)), min(1, max(0, ev.y / self.H)))

    def _key_down(self, ev):
        key = ev.keysym.lower()
        if key not in self.KEYS or self.mouse is None:
            return
        i, _ = self.KEYS[key]
        if not self.locked[i]:
            self.locked[i] = True
            self.lock_target[i] = float(self.lefthand.channels[f"finger{i}"].value)
        self.pressed.add(key)

    def _key_up(self, ev):
        key = ev.keysym.lower()
        self.pressed.discard(key)
        if key in self.KEYS:
            i, _ = self.KEYS[key]
            if not any(self.KEYS[k][0] == i for k in self.pressed):
                self.locked[i] = False  # back to the field

    def field_target(self, i: int, u: float, v: float) -> float:
        """The original calculate_cursor_targets math, verbatim: influence
        falls off linearly inside the gravity width; outside it the finger
        relaxes to the default curl."""
        d = abs(u - self._home_u(i))
        g = self.gravity.get()
        if d >= g:
            return self.default_pos.get()
        influence = 1.0 - d / g
        y_off = (v - 0.5) * self.sensitivity.get() * self.wave.get() * influence
        if self.reverse.get():
            y_off = -y_off
        return max(0.0, min(180.0, self.default_pos.get() + y_off * self.servo_range.get()))

    def _tick(self):
        for key in self.pressed:
            i, d = self.KEYS[key]
            if self.locked[i]:
                self.lock_target[i] = max(0.0, min(180.0, self.lock_target[i] + d * self.key_step))
                self.lefthand.set_channel(f"finger{i}", self.lock_target[i])
        if self.mouse is not None:
            u, v = self.mouse
            for i in range(4):
                if not self.locked[i]:
                    self.lefthand.set_channel(f"finger{i}", self.field_target(i, u, v))
        self._redraw()
        self.after(33, self._tick)

    def _redraw(self):
        half = self.W * self.HOME_SPAN / 4 * 0.42
        for i in range(4):
            deg = self.lefthand.channels[f"finger{i}"].value
            cx = self._home_u(i) * self.W
            self.coords(self.bars[i], cx - half + 3, self._bar_y(deg), cx + half - 3, self.H - self.BOT)
            self.itemconfig(self.bars[i], fill="#c9a63d" if self.locked[i] else "#8a63d2")
            self.itemconfig(self.deg_lbls[i], text=f"{int(deg)}°")
        dy = self._bar_y(self.default_pos.get())
        self.coords(self.default_line, self.W * 0.08, dy, self.W * 0.92, dy)
        if self.mouse is not None:
            px, py = self.mouse[0] * self.W, self.mouse[1] * self.H
            r = self.gravity.get() * self.W
            self.coords(self.gravity_line, px - r, py, px + r, py)
            self.coords(self.cursor_h, px - 8, py, px + 8, py)
            self.coords(self.cursor_v, px, py - 8, px, py + 8)
            state = "normal"
        else:
            state = "hidden"
        for item in (self.gravity_line, self.cursor_h, self.cursor_v):
            self.itemconfig(item, state=state)


class LungStrip(tk.Canvas):
    """Lung workspace — breath as a scrolling waveform: vertical drag sets
    the lung position, and you SEE the rhythm you're performing."""

    WINDOW = 12.0  # seconds of breath history shown

    def __init__(self, parent, lunggaze: SerialDevice, w=340, h=220):
        self.W, self.H = w, h
        super().__init__(parent, width=self.W, height=self.H, bg="#0d1f26", highlightthickness=0)
        self.lunggaze = lunggaze
        self.ch = lunggaze.channels["lung"]
        self.history = []  # (t, value)
        self.create_text(self.W // 2, 10, text="lung — drag vertically, breathe with the wave", fill="#667", font=("monospace", 8))
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


class GazePad(tk.Canvas):
    """Gaze simulation for the runtime tab: drag to point the machine's
    eyes. The arrow from center IS the modifier — in vector mode it biases
    which transitions the markov chains prefer (movement direction), in
    offset mode it leans the servo targets. Double-click to re-center."""

    def __init__(self, parent, size, on_gaze):
        super().__init__(parent, width=size, height=size, bg="#101820", highlightthickness=0, cursor="crosshair")
        self.S = size
        self.on_gaze = on_gaze
        c, r = size / 2, size / 2 - 16
        self.c, self.r = c, r
        self.create_oval(c - r, c - r, c + r, c + r, outline="#33445a")
        self.create_oval(c - r / 2, c - r / 2, c + r / 2, c + r / 2, outline="#1e2a38")
        self.create_line(c - r, c, c + r, c, fill="#1e2a38")
        self.create_line(c, c - r, c, c + r, fill="#1e2a38")
        self.create_text(c, 10, text="gaze — drag to look (drives pan/tilt live); the arrow nudges the flow", fill="#667", font=("monospace", 8))
        self.arrow = self.create_line(c, c, c, c, fill="#f5c04a", width=2, arrow="last")
        self.dot = self.create_oval(c - 5, c - 5, c + 5, c + 5, fill="#ffeaa7", outline="")
        self.readout = self.create_text(c, size - 10, text="gaze +0.00, +0.00", fill="#667", font=("monospace", 8))
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Button-1>", self._drag)
        self.bind("<Double-Button-1>", lambda e: self.set_gaze(0.0, 0.0))

    def _drag(self, ev):
        gx = max(-1.0, min(1.0, (ev.x - self.c) / self.r))
        gy = max(-1.0, min(1.0, (self.c - ev.y) / self.r))
        self.set_gaze(gx, gy)

    def set_gaze(self, gx: float, gy: float):
        px, py = self.c + gx * self.r, self.c - gy * self.r
        self.coords(self.dot, px - 5, py - 5, px + 5, py + 5)
        self.coords(self.arrow, self.c, self.c, px, py)
        self.itemconfig(self.readout, text=f"gaze {gx:+.2f}, {gy:+.2f}")
        self.on_gaze(gx, gy)


class SessionFrame(ttk.LabelFrame):
    """The looper, timeline-first (July 26 — modeled on the servocontroller
    ui-modernize interface): every track is a LANE with a waveform of its
    take and a shared playhead. ● on a lane = record-enable (red, DAW
    style — it was labeled 'arm', which read as robot-arm nonsense);
    Record with nothing enabled records the workspace you're standing on.
    Stop mid-pass KEEPS the partial take. 'link'ed lanes train as ONE
    joint chain (they move in relation); unlinked lanes get their own.
    Generate runs every chain simultaneously."""

    # Record with no lane rec-enabled: record what you're performing on.
    TAB_TRACKS = {
        "right arm — bed": ["right arm (grbl)", "pen (right hand)"],
        "left arm — linkage": ["left arm", "wrist"],
        "hand": ["hand (fingers)"],
        "lung": ["lung"],
    }
    LANE_COLORS = ["#8a63d2", "#3ba7a0", "#f5c04a", "#e94560"]

    def __init__(self, parent, lunggaze: SerialDevice, lefthand: SerialDevice, grbl: GrblFrame, log, ws=(760, 380)):
        super().__init__(parent, text="body session  (layered choreography → joint markov)")
        self.lunggaze = lunggaze
        self.lefthand = lefthand
        self.grbl = grbl
        self.log = log
        self.session = Session()
        self._route = {c: lefthand for c in ("elbow", "shoulder", "wrist", "finger0", "finger1", "finger2", "finger3")}
        self._route["lung"] = lunggaze
        self.transport = self._make_transport()
        self._lane_w = max(320, ws[0] - 450)
        self._lanes = []

        # transport row
        tr = ttk.Frame(self)
        tr.pack(fill="x", padx=4, pady=2)
        ttk.Button(tr, text="● Record", command=self.record).pack(side="left", padx=2)
        ttk.Button(tr, text="▶ Play", command=self.play).pack(side="left", padx=2)
        ttk.Button(tr, text="■ Stop", command=self.transport_stop).pack(side="left", padx=2)
        ttk.Button(tr, text="∿ Generate", command=self.generate).pack(side="left", padx=2)
        ttk.Button(tr, text="✕ Clear all", command=self.clear_all).pack(side="left", padx=(10, 2))
        ttk.Label(tr, text="loop").pack(side="left", padx=(10, 2))
        self.loop_var = tk.IntVar(value=int(self.session.loop_len))
        ttk.OptionMenu(
            tr, self.loop_var, int(self.session.loop_len), *LOOP_LENGTHS, command=lambda v: setattr(self.session, "loop_len", float(v))
        ).pack(side="left")
        ttk.Label(tr, text="speed").pack(side="left", padx=(10, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_lbl = ttk.Label(tr, text="1.0x", width=5)
        ttk.Scale(tr, from_=0.25, to=2.0, variable=self.speed_var, length=90, command=lambda v: self.speed_lbl.config(text=f"{float(v):.2f}x")).pack(
            side="left"
        )
        self.speed_lbl.pack(side="left")

        # session persistence row — project workflow only; publishing to the
        # runtime library lives in the "temperaments" tab, where you SEE
        # what you're assigning to
        sr = ttk.Frame(self)
        sr.pack(fill="x", padx=4, pady=2)
        ttk.Label(sr, text="session").pack(side="left")
        self.name_entry = ttk.Entry(sr, width=16)
        self.name_entry.insert(0, self.session.name)
        self.name_entry.pack(side="left", padx=2)
        ttk.Button(sr, text="Save project", command=self.save).pack(side="left", padx=2)
        self.saved_var = tk.StringVar()
        self.saved_menu = ttk.Combobox(sr, textvariable=self.saved_var, width=28, state="readonly")
        self.saved_menu.pack(side="left", padx=2)
        ttk.Button(sr, text="Load", command=self.load).pack(side="left", padx=2)
        self._refresh_saved()

        # track rows
        self.tracks_box = ttk.Frame(self)
        self.tracks_box.pack(fill="x", padx=4, pady=2)
        self._build_tracks()

        # workspaces — each tab: big canvas left, controls in a side panel
        ws_w, ws_h = ws
        side_w = 250
        cv_w = max(520, ws_w - side_w)
        nb = self.nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=4, pady=3)

        def make_tab(title):
            t = ttk.Frame(nb)
            nb.add(t, text=title)
            side = ttk.Frame(t, width=side_w)
            side.pack(side="right", fill="y", padx=8, pady=4)
            side.pack_propagate(False)
            holder = ttk.Frame(t)
            holder.pack(side="left", fill="both", expand=True, padx=4, pady=4)
            return holder, side

        holder, side = make_tab("right arm — bed")
        self.bed = BedView(holder, grbl, w=cv_w, h=ws_h)
        self.bed.pack(anchor="nw")
        labeled_slider(side, "max feed (playback/gen)", 200, 3000, grbl.max_feed, lambda v: setattr(grbl, "max_feed", v), fmt=lambda v: str(int(v)))
        ttk.Label(
            side,
            text="drag = rapids (travel)\nright-hold = pen down:\ndraws, drags become G1\nat your tempo\n\npen is a track — arm it\nto record up/down as a\nlayer; group with the\narm to learn WHERE it\ndraws",
            font=("monospace", 8),
            foreground="#888",
        ).pack(anchor="w", pady=6)

        holder, side = make_tab("left arm — linkage")
        self.linkage = LinkageView(holder, lefthand, w=cv_w, h=ws_h)
        self.linkage.pack(anchor="nw")
        labeled_slider(side, "elbow range %", 10, 100, 100, lambda v: self.linkage.set_scale("elbow", v / 100), fmt=lambda v: str(int(v)))
        labeled_slider(side, "shoulder range %", 10, 100, 100, lambda v: self.linkage.set_scale("shoulder", v / 100), fmt=lambda v: str(int(v)))
        labeled_slider(side, "wrist range %", 10, 100, 100, lambda v: self.linkage.set_scale("wrist", v / 100), fmt=lambda v: str(int(v)))
        labeled_slider(side, "wrist °/wheel notch", 0.5, 10.0, 3.0, lambda v: setattr(self.linkage, "wheel_step", v), fmt=lambda v: f"{v:.1f}")
        labeled_slider(side, "smoothing s", 0.05, 0.8, 0.25, lambda v: setattr(lefthand, "smooth_time", v))
        self.cal_lbl = ttk.Label(side, text="", font=("monospace", 8), wraplength=side_w - 16)

        def set_cal(msg):
            self.cal_lbl.config(text=msg)

        def toggle_mode():
            new = "calibrated" if self.linkage.mode == "joint" else "joint"
            set_cal(self.linkage.set_mode(new))
            mode_btn.config(text=f"mapping: {self.linkage.mode}")

        mode_btn = ttk.Button(side, text=f"mapping: {self.linkage.mode}", command=toggle_mode)
        mode_btn.pack(fill="x", pady=(10, 2))
        ttk.Button(side, text="Calibrate 9-pt", command=lambda: set_cal(self.linkage.start_calibration())).pack(fill="x", pady=2)

        def set_point():
            set_cal(self.linkage.capture_point())
            mode_btn.config(text=f"mapping: {self.linkage.mode}")

        ttk.Button(side, text="Set point", command=set_point).pack(fill="x", pady=2)
        self.cal_lbl.pack(fill="x", pady=4)

        holder, side = make_tab("hand")
        self.hand_space = HandSpace(holder, lefthand, w=cv_w, h=ws_h)
        self.hand_space.pack(anchor="nw")
        hs = self.hand_space
        labeled_slider(side, "sensitivity", 0.5, 6.0, 3.0, hs.sensitivity.set)
        labeled_slider(side, "wave strength", 0.2, 4.0, 2.0, hs.wave.set)
        labeled_slider(side, "gravity width", 0.05, 1.0, 0.4, hs.gravity.set)
        labeled_slider(side, "default curl °", 0, 180, 90, hs.default_pos.set, fmt=lambda v: str(int(v)))
        labeled_slider(side, "range ±°", 10, 90, 60, hs.servo_range.set, fmt=lambda v: str(int(v)))
        labeled_slider(side, "key step °/tick", 0.5, 6.0, 2.0, lambda v: setattr(hs, "key_step", v), fmt=lambda v: f"{v:.1f}")
        ttk.Checkbutton(side, text="reverse vertical", variable=hs.reverse).pack(anchor="w", pady=4)
        ttk.Label(side, text="legacy dataset", font=("monospace", 8)).pack(anchor="w", pady=(12, 2))
        self.legacy_var = tk.StringVar()
        ttk.Combobox(side, textvariable=self.legacy_var, state="readonly", values=list_legacy_hand_datasets()).pack(fill="x", pady=2)

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

        ttk.Button(side, text="Import as hand take", command=do_import).pack(fill="x", pady=2)

        holder, side = make_tab("lung")
        self.lung_strip = LungStrip(holder, lunggaze, w=cv_w, h=ws_h)
        self.lung_strip.pack(anchor="nw")
        ttk.Label(side, text="drag vertically —\nbreathe with the wave", font=("monospace", 8), foreground="#888").pack(anchor="w", pady=6)

        rt = ttk.Frame(nb)
        nb.add(rt, text="runtime")
        self._build_runtime(rt, ws_h)
        nb.bind("<<NotebookTabChanged>>", lambda e: self._refresh_library() if nb.tab(nb.select(), "text") == "runtime" else None)
        if KINETIC_BUS_ENABLED:
            nb.select(rt)  # runtime build: open where the tuning happens

        self._playhead_tick()  # one shared timer for all lanes, started once
        # width=1 + fill="x": the label stretches to the frame but its text
        # can never REQUEST width — long status lines used to widen the
        # whole session frame and shove the canvas mid-recording
        self.status = ttk.Label(self, text="idle", width=1, anchor="w")
        self.status.pack(fill="x", padx=6, pady=2)

    # --- transport plumbing ---------------------------------------------------
    # --- routing (shared by the looper transport and the temperament lab) -----
    def _route_state(self) -> dict:
        s = {c: dev.channels[c].value for c, dev in self._route.items()}
        s["x"], s["y"] = self.grbl.position
        s["pen"] = float(self.grbl.pen_s)
        return s

    def _route_ease(self, d):
        for c, v in d.items():
            if c in self._route:
                self._route[c].set_channel(c, int(round(v)))

    def _route_plan(self, d, dt):
        self.grbl.goto(d["x"], d["y"], dt)

    def _route_step(self, d):
        if "pen" in d:
            self.grbl.pen_command(int(round(d["pen"])))

    def _make_transport(self) -> Transport:
        return Transport(
            self.session,
            get_state=self._route_state,
            send_ease=self._route_ease,
            send_plan=self._route_plan,
            on_status=self._set_status,
            send_step=self._route_step,
        )

    def _set_status(self, msg: str):
        try:
            self.after(0, lambda: (self.status.config(text=msg), self._refresh_tracks()))
        except RuntimeError:
            pass

    def _pen_layer_active(self) -> bool:
        """The pen draws only when its track is part of the session — armed
        for this pass, or holding an unmuted take. Otherwise choreography
        stays pen-up (the old always-up rule, now scoped)."""
        for t in self.session.tracks:
            if t.channels == ["pen"]:
                return t.armed or (t.has_take and not t.mute)
        return False

    def record(self):
        self._lab_stop()  # the lab and the looper share the body — one owner
        if not any(t.armed for t in self.session.tracks):
            # nothing rec-enabled: record the workspace you're standing on
            names = self.TAB_TRACKS.get(self.nb.tab(self.nb.select(), "text"), [])
            for t in self.session.tracks:
                t.armed = t.name in names
            self._refresh_tracks()
        if not self._pen_layer_active():
            self.grbl.set_pen(GRBL_PEN_UP_S)
        self.transport.record()
        self._refresh_tracks()

    def play(self):
        self._lab_stop()
        if not self._pen_layer_active():
            self.grbl.set_pen(GRBL_PEN_UP_S)
        self.transport.play(speed=self.speed_var.get())

    def generate(self):
        self._lab_stop()
        if not self._pen_layer_active():
            self.grbl.set_pen(GRBL_PEN_UP_S)
        self.transport.generate(speed=self.speed_var.get())

    def transport_stop(self):
        self.transport.stop()

    def clear_all(self):
        """Wipe every lane's take (saved session files are untouched —
        reload to get them back)."""
        self.transport.stop()
        n = sum(1 for t in self.session.tracks if t.has_take)
        for t in self.session.tracks:
            t.samples = None
            t.armed = False
        self._refresh_tracks()
        self.status.config(text=f"cleared {n} take(s) — saved sessions untouched")

    # --- runtime tab: library | gaze simulation | lab controls -----------------
    def _build_runtime(self, parent, ws_h):
        """The runtime room, laid out like the legacy hand controller: a
        column per concern instead of everything crammed against a canvas
        template. LEFT — the temperament library as a tree (each state and
        exactly which bundles it owns; assignment used to be an invisible
        filename side-effect). CENTER — the gaze simulation pad: drag to
        look, the arrow is the modifier; in vector mode it biases WHICH
        transitions the chains prefer (movement direction — dynamic range
        nudging that never distorts poses), in offset mode it leans servo
        targets. RIGHT — the lab: the actual KineticBus on this panel's
        routing. machine.py-flagged builds (KINETIC_BUS_ENABLED) open the
        panel on this tab."""
        self._lab_ctx = {"drawing": False, "gx": 0.0, "gy": 0.0, "person": "absent"}
        self._lab_on = False
        self.lab = KineticBus(
            library=TemperamentLibrary(owned=LAB_CHANNELS),
            is_drawing=lambda: self._lab_ctx["drawing"],
            get_gaze=lambda: (self._lab_ctx["gx"], self._lab_ctx["gy"]),
            get_person=lambda: self._lab_ctx["person"],  # toggle = arrival startle + reach ramp, like the runtime
            on_log=lambda m: self.log("lab", m, False),
            send_ease=self._route_ease,
            send_plan=self._route_plan,
            send_step=self._route_step,
            get_state=self._route_state,
            owned=LAB_CHANNELS,
        )

        left = ttk.Frame(parent)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=4)
        ttk.Label(left, text="temperament library — one DATASET plays per state, several rotate on a timer", font=("monospace", 9, "bold")).pack(
            anchor="w"
        )
        # the banner answers the confusing questions at a glance: what is
        # playing NOW, and when does it rotate to the next dataset
        self.now_lbl = tk.Label(left, text="", font=("monospace", 11, "bold"), fg="#8a63d2", anchor="w", width=1)
        self.now_lbl.pack(fill="x", pady=(2, 4))
        self.temp_tree = ttk.Treeview(left, show="tree", selectmode="browse")
        self.temp_tree.tag_configure("active", foreground="#8a63d2")
        self.temp_tree.tag_configure("empty", foreground="#777")
        self.temp_tree.pack(fill="both", expand=True, pady=2)
        lrow = ttk.Frame(left)
        lrow.pack(fill="x")
        ttk.Button(lrow, text="＋ Assign looper session as dataset here", command=self._assign_selected).pack(side="left", padx=2)
        ttk.Button(lrow, text="Edit in looper", command=self._load_selected).pack(side="left", padx=2)
        ttk.Button(lrow, text="Retire ▸ projects", command=self._retire_selected).pack(side="left", padx=2)

        center = ttk.Frame(parent)
        center.pack(side="left", fill="y", padx=6, pady=4)
        pad_size = max(240, min(340, ws_h - 90))
        self.gaze_pad = GazePad(center, pad_size, on_gaze=self._on_gaze)
        self.gaze_pad.pack()
        # one knob for the whole gaze current (lean + tempo + choice together)
        labeled_slider(center, "gaze influence", 0.0, 2.0, self.lab.gaze_strength, lambda v: setattr(self.lab, "gaze_strength", v))
        self.lab_status = ttk.Label(center, text="", font=("monospace", 9), width=1, anchor="w")  # width=1: text never resizes the column
        self.lab_status.pack(fill="x", pady=4)

        right = ttk.Frame(parent, width=210)
        right.pack(side="left", fill="y", padx=6, pady=4)
        right.pack_propagate(False)
        ttk.Label(right, text="lab (runtime bus)", font=("monospace", 9, "bold")).pack(anchor="w")
        self.lab_btn = ttk.Button(right, text="▶ Start lab", command=self._lab_toggle)
        self.lab_btn.pack(fill="x", pady=3)
        ttk.Label(right, text="mood — ▶ marks current", font=("monospace", 8), foreground="#888").pack(anchor="w")
        self._emotion_btns = {}
        for e in EMOTIONS:
            b = ttk.Button(right, text=e, command=lambda e=e: self.lab.set_emotion(e))
            b.pack(fill="x", pady=1)
            self._emotion_btns[e] = b
        draw_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            right,
            text="drawing state\n(overrides mood;\nright hand yields)",
            variable=draw_var,
            command=lambda: self._lab_ctx.__setitem__("drawing", draw_var.get()),
        ).pack(anchor="w", pady=3)
        person_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            right,
            text="person present\n(startle on arrival,\narm reaches at gaze)",
            variable=person_var,
            command=lambda: self._lab_ctx.__setitem__("person", "visible" if person_var.get() else "absent"),
        ).pack(anchor="w", pady=3)
        ttk.Button(right, text="⚡ startle", command=lambda: self.lab.startle() if self._lab_on else None).pack(fill="x", pady=3)
        self._refresh_library()
        self._lab_tick()

    def _on_gaze(self, gx: float, gy: float):
        self._lab_ctx["gx"] = gx
        self._lab_ctx["gy"] = gy
        # the pad IS the head: drive pan/tilt whenever the lunggaze arduino
        # is connected (sim-logged otherwise) — not only while the lab runs
        pan, tilt = self.lunggaze.channels["pan"], self.lunggaze.channels["tilt"]
        self.lunggaze.set_channel("pan", pan.neutral + gx * (pan.hi - pan.lo) / 2)
        self.lunggaze.set_channel("tilt", tilt.neutral + gy * (tilt.hi - tilt.lo) / 2)

    def _refresh_library(self):
        tree = self.temp_tree
        tree.delete(*tree.get_children())
        buckets = self.lab.library.scan()
        for state in STATES:
            fns = buckets.get(state, [])
            label = f"{state} — no datasets yet" if not fns else f"{state} — {len(fns)} dataset(s)"
            parent = tree.insert("", "end", iid=f"state:{state}", text=label, open=True, tags=() if fns else ("empty",))
            for fn in fns:
                tree.insert(parent, "end", iid=f"bundle:{fn}", text=f"   {fn[len('session_'):-len('.json')]}")
        projects = [s for s in Session.list_saved() if s.startswith("projects/")]
        parent = tree.insert(
            "", "end", iid="state:projects", text=f"projects (working area — the runtime never plays these) — {len(projects)}", open=False
        )
        for p in projects:
            tree.insert(parent, "end", iid=f"project:{p}", text=f"   {os.path.basename(p)[len('session_'):-len('.json')]}")
        self._active_marked = None  # force re-mark on next tick
        self._mark_active()

    def _mark_active(self):
        """▶ on the dataset the bus is playing RIGHT NOW (and its state row)."""
        bundle = self.lab.status()["bundle"] if self._lab_on else None
        if bundle == getattr(self, "_active_marked", None):
            return
        tree = self.temp_tree
        for iid in tree.get_children(""):
            for child in tree.get_children(iid):
                if child.startswith("bundle:"):
                    fn = child[len("bundle:") :]
                    active = fn == bundle
                    tree.item(child, text=("▶  " if active else "   ") + fn[len("session_") : -len(".json")], tags=("active",) if active else ())
        self._active_marked = bundle

    def _selected_state(self):
        sel = self.temp_tree.selection()
        if not sel:
            return None
        iid = sel[0]
        if iid.startswith("bundle:"):
            iid = self.temp_tree.parent(iid)
        if iid.startswith("state:"):
            return iid[len("state:") :]
        return None

    def _assign_selected(self):
        state = self._selected_state()
        if state is None or state == "projects":
            self.status.config(text="pick a state (or one of its bundles) in the tree, then Assign")
            return
        if not any(t.has_take for t in self.session.tracks):
            self.status.config(text="nothing to assign — record a take first")
            return
        existing = set(self.lab.library.scan().get(state, []))
        name = f"{state}_{len(existing)}"
        for i in range(26):  # first FREE letter — count-based naming overwrites after a retire leaves a gap
            cand = f"{state}_{chr(ord('a') + i)}"
            if f"session_{cand}.json" not in existing:
                name = cand
                break
        self.session.name = name
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, name)
        self.session.save(export=True)
        self.status.config(text=f"assigned as dataset {name} under {state}")
        self._refresh_library()
        self._refresh_saved()
        self.temp_tree.selection_set(f"bundle:session_{name}.json")  # show exactly where it landed

    def _load_selected(self):
        sel = self.temp_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("bundle:"):
            self._load_file(iid[len("bundle:") :])
        elif iid.startswith("project:"):
            self._load_file(iid[len("project:") :])

    def _retire_selected(self):
        sel = self.temp_tree.selection()
        if not sel or not sel[0].startswith("bundle:"):
            self.status.config(text="select a bundle to retire (states and projects can't be retired)")
            return
        fn = sel[0][len("bundle:") :]
        dst = self.lab.library.retire(fn)
        self.status.config(text=f"retired {fn} ▸ projects/{os.path.basename(dst)}")
        self._refresh_library()
        self._refresh_saved()

    def _lab_toggle(self):
        if self._lab_on:
            self._lab_stop()
        else:
            self.transport.stop()  # the lab and the looper share the body
            self.grbl.set_pen(GRBL_PEN_UP_S)  # bundles with pen takes will lower it themselves
            self.lab.enable()
            self._lab_on = True
            self.lab_btn.config(text="■ Stop lab")

    def _lab_stop(self):
        if getattr(self, "_lab_on", False):
            self.lab.shutdown()
            self._lab_on = False
            self.lab_btn.config(text="▶ Start lab")

    def _lab_tick(self):
        s = self.lab.status()
        if self._lab_on:
            if s["bundle"]:
                name = s["bundle"][len("session_") : -len(".json")]
                mins, secs = divmod(int(s["rotate_in"] or 0), 60)
                self.now_lbl.config(text=f"▶ PLAYING  {name}  ({s['state']})   —   next dataset in {mins}:{secs:02d}")
            else:
                self.now_lbl.config(text=f"▶ {s['state'] or '…'} — no dataset assigned yet, body idle")
            self.lab_status.config(
                text=f"{s['chains']} chain(s) live · gaze {self._lab_ctx['gx']:+.2f}, {self._lab_ctx['gy']:+.2f} · reach {s['reach']:.2f}"
            )
        else:
            self.now_lbl.config(text="nothing playing — ▶ Start lab runs the runtime bus")
            self.lab_status.config(text="")
        for e, b in self._emotion_btns.items():
            b.config(text=("▶ " + e) if (self._lab_on and s["emotion"] == e) else e)
        self._mark_active()
        self.after(500, self._lab_tick)

    # --- track lanes ----------------------------------------------------------
    def _build_tracks(self):
        for w in self.tracks_box.winfo_children():
            w.destroy()
        self._lanes = []
        for t in self.session.tracks:
            row = ttk.Frame(self.tracks_box)
            row.pack(fill="x", pady=1)
            rec_btn = tk.Button(row, text="●", width=2, relief="flat", bd=0, activeforeground="#e94560", command=lambda t=t: self._toggle_rec(t))
            rec_btn.pack(side="left")
            ttk.Label(row, text=t.name, width=16).pack(side="left")
            cv = tk.Canvas(row, width=self._lane_w, height=24, bg="#141824", highlightthickness=0)
            cv.pack(side="left", fill="x", expand=True, padx=3)
            ph = cv.create_line(0, 0, 0, 24, fill="#ffeaa7", state="hidden")
            mute_var = tk.BooleanVar(value=t.mute)
            ttk.Checkbutton(
                row, text="mute", variable=mute_var, command=lambda t=t, v=mute_var: (setattr(t, "mute", v.get()), self._refresh_tracks())
            ).pack(side="left")
            link_var = tk.BooleanVar(value=t.group != "solo")
            ttk.Checkbutton(
                row, text="link", variable=link_var, command=lambda t=t, v=link_var: setattr(t, "group", "A" if v.get() else "solo")
            ).pack(side="left")
            ttk.Button(row, text="✕", width=2, command=lambda t=t: (setattr(t, "samples", None), self._refresh_tracks())).pack(side="left", padx=2)
            self._lanes.append({"t": t, "rec": rec_btn, "cv": cv, "ph": ph, "mute": mute_var, "link": link_var})
        self._refresh_tracks()

    def _toggle_rec(self, t):
        t.armed = not t.armed
        self._refresh_tracks()

    def _refresh_tracks(self):
        for lane in self._lanes:
            t = lane["t"]
            lane["rec"].config(fg="#e94560" if t.armed else "#555")
            lane["mute"].set(t.mute)
            lane["link"].set(t.group != "solo")
            self._draw_take(lane)

    def _draw_take(self, lane):
        """The take, visible: one polyline per channel across the loop, each
        autoscaled to its own range. A partial take stops mid-lane; a flat
        take is a flat line — the lane never lies about what was captured."""
        t, cv = lane["t"], lane["cv"]
        cv.delete("wave")
        w = cv.winfo_width() if cv.winfo_width() > 1 else int(cv["width"])
        h = int(cv["height"])
        if not t.has_take:
            cv.create_text(w // 2, h // 2, text="empty — ● then Record", fill="#333", font=("monospace", 8), tags="wave")
            cv.tag_raise(lane["ph"])
            return
        loop = max(0.1, self.session.loop_len)
        for ci, c in enumerate(t.channels):
            vals = [(s["t"], s[c]) for s in t.samples if c in s]
            if not vals:
                continue
            vmin = min(v for _, v in vals)
            vmax = max(v for _, v in vals)
            span = vmax - vmin
            step = max(1, len(vals) // max(1, w))
            pts = []
            for st, v in vals[::step]:
                x = min(w - 1, st / loop * w)
                y = h / 2 if span < 1e-6 else h - 3 - (v - vmin) / span * (h - 6)
                pts.extend((x, y))
            if len(pts) >= 4:
                color = "#3a4152" if t.mute else self.LANE_COLORS[ci % len(self.LANE_COLORS)]
                cv.create_line(*pts, fill=color, tags="wave")
        cv.tag_raise(lane["ph"])

    def _playhead_tick(self):
        pos = self.transport.loop_pos()
        recording = self.transport.state == "recording"
        for lane in self._lanes:
            cv, ph, t = lane["cv"], lane["ph"], lane["t"]
            if pos is None:
                cv.itemconfig(ph, state="hidden")
            else:
                w = cv.winfo_width() if cv.winfo_width() > 1 else int(cv["width"])
                x = pos / max(0.1, self.session.loop_len) * w
                cv.coords(ph, x, 0, x, int(cv["height"]))
                cv.itemconfig(ph, state="normal", fill="#e94560" if recording and t.armed else "#ffeaa7")
            want = "#241019" if recording and t.armed else "#141824"
            if cv["bg"] != want:
                cv.config(bg=want)
        self.after(100, self._playhead_tick)

    # --- persistence -------------------------------------------------------------
    def _refresh_saved(self):
        self.saved_menu["values"] = Session.list_saved()

    def save(self):
        """Working save — projects/ is the editing area, invisible to the
        runtime. Iterate freely, then Export when a take deserves to be a
        temperament."""
        self.session.name = self.name_entry.get().strip() or "session"
        path = self.session.save()
        self.log("session", f"project saved: {os.path.basename(path)}", False)
        self.status.config(text=f"project saved: {self.session.name} (edit freely — runtime can't see it)")
        self._refresh_saved()

    def load(self):
        if self.saved_var.get():
            self._load_file(self.saved_var.get())

    def _load_file(self, filename: str):
        self.transport.stop()
        self.session = Session.load(filename)
        self.loop_var.set(int(self.session.loop_len))
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, self.session.name)
        self.transport = self._make_transport()
        self._build_tracks()
        self.status.config(text=f"loaded {self.session.name}")

    def shutdown(self):
        self.transport.stop()
        self._lab_stop()


def build_ui(root):
    """Assemble the whole panel into `root`. Split from main() so debug
    scripts can measure/verify the layout without entering the mainloop."""
    root.title("mslint — unified motor panel")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{sw}x{sh - 70}+0+0")
    try:
        root.attributes("-zoomed", True)  # fill the screen where the WM supports it
    except tk.TclError:
        pass

    warn = tk.Label(root, text="⚠ stop machine.py before connecting — serial ports are exclusive", fg="darkorange")
    warn.pack(fill="x", pady=2)

    console = scrolledtext.ScrolledText(root, height=7, state="disabled", font=("monospace", 9))

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

    devices = build_devices()

    # bottom-up packing: the action bar and console claim their space FIRST,
    # so a crowded layout squeezes the workspaces — never the buttons
    bottom = ttk.Frame(root)
    bottom.pack(side="bottom", fill="x")

    def everything_neutral():
        for d in devices:
            d.all_neutral()

    ttk.Button(bottom, text="ALL NEUTRAL", command=everything_neutral).pack(side="left", padx=4, pady=2)
    console.pack(side="bottom", fill="x", padx=4, pady=3)

    cols = ttk.Frame(root)
    cols.pack(fill="both", expand=True)
    dev_col = ttk.Frame(cols)
    dev_col.pack(side="left", fill="y", padx=4)
    frames = []
    for device in devices:
        f = DeviceFrame(dev_col, device, log)
        f.pack(fill="x", pady=3)
        frames.append(f)
    grbl = GrblFrame(dev_col, log)
    grbl.pack(fill="x", pady=3)

    ws_h = max(380, min(640, sh - 620))
    ws_w = max(760, min(1250, sw - 700))
    body = SessionFrame(cols, devices[0], devices[1], grbl, log, ws=(ws_w, ws_h))  # lunggaze, lefthand
    body.pack(side="left", fill="both", expand=True, padx=4, pady=3)
    # homing safety: the left arm tucks to its "homing" dataset pose before
    # $H and blends back when homing completes (bus max-hold as failsafe)
    grbl.on_home = body.lab.home_clear
    grbl.on_home_done = body.lab.home_release

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
    return {"body": body, "grbl": grbl, "devices": devices, "frames": frames}


def main():
    root = tk.Tk()
    build_ui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
