"""Headless gantry link — the right arm joins the runtime temperament.

The panel's GrblFrame proved the serial discipline (single writer, strict
FIFO, reader owns RX, absolute G90 only, bounded pipelining); this is that
discipline without Tkinter, sized for what generation actually sends:
G1 targets at a few Hz plus the occasional pen barrier. The kinetic bus
owns one of these at runtime — connect+home at the awakening, markov
gantry motion between drawings, released whenever the drawing pipeline
needs the port (a serial open RESETS GRBL, so every re-acquire re-homes,
which fires the tuck choreography — by design).

Every target projects into the measured reach envelope (clamp_to_reach),
same as the panel and the drawing g-code. Pen stays UP unless the bus is
explicitly configured to let recorded pen takes ink (KINETIC_GANTRY_PEN).
"""

import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ARMS_DUET_MAX_FEED, GRBL_PEN_UP_S
from grbl.warp_calibration import clamp_to_reach

MAX_UNACKED = 3  # segments in flight — enough planner lookahead for blending


class GantryLink:
    """One serial GRBL connection, bus-owned. attach(ser) takes an already
    homed/configured port (ensure_homed ran); release() gives it back."""

    def __init__(self, preferred_port: Optional[str] = None):
        self.preferred_port = preferred_port or os.getenv("GRBL_PORT", "/dev/arduino_cnc")
        self.ser = None
        self.max_feed = ARMS_DUET_MAX_FEED
        self.position = (0.0, 0.0)  # last commanded (feed math); homing leaves us at machine home
        self._path: deque = deque()
        self._cmd_q: "queue.Queue" = queue.Queue()
        self._resp_q: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()  # attach/release vs writer
        self._running = False
        self._sent = 0
        self._last_err = ""
        self.on_log = lambda m: None

    @property
    def alive(self) -> bool:
        return self._running and self.ser is not None

    # --- lifecycle ------------------------------------------------------------
    def connect_and_home(self, max_retries: int = 3) -> bool:
        """Open the port (this RESETS GRBL) and home it — ensure_homed fires
        the in-process homing hooks, so the tuck choreography runs with the
        sweep. Then start the reader/writer pair."""
        from grbl.grbl_utils import ensure_homed, find_grbl_port

        with self._lock:
            if self.alive:
                return True
            ser = find_grbl_port(preferred_port=self.preferred_port)
            if not ser:
                self.on_log("gantry: no GRBL found")
                return False
            try:
                ensure_homed(ser, max_retries=max_retries)
            except Exception as e:
                self.on_log(f"gantry: homing failed: {e}")
                try:
                    ser.close()
                except Exception:
                    pass
                return False
            self.attach(ser)
            self.on_log("gantry: homed and attached — the right arm is in the temperament")
            return True

    def attach(self, ser):
        """Adopt an already configured (G21/G90, homed) serial port."""
        self.ser = ser
        self._path.clear()
        self._drain(self._cmd_q)
        self._drain(self._resp_q)
        self._probe_status()
        self._sent = 0
        self._last_err = ""
        self._running = True
        threading.Thread(target=self._reader_loop, daemon=True, name="gantry-reader").start()
        threading.Thread(target=self._writer_loop, daemon=True, name="gantry-writer").start()

    def _probe_status(self):
        """One '?' before the threads own the port: if GRBL sits in Alarm,
        every G1 we ever send dies silently — say so out loud instead."""
        try:
            self.ser.reset_input_buffer()
            self.ser.write(b"?")
            self.ser.flush()
            for _ in range(4):
                line = self.ser.readline().decode(errors="replace").strip()
                if line.startswith("<"):
                    self.on_log(f"gantry: status {line}")
                    if "Alarm" in line:
                        self.on_log("gantry: ⚠ GRBL IS ALARM-LOCKED — motion will be rejected until homed/unlocked")
                    return
        except Exception:
            pass

    def release(self):
        """Give the port back (drawing pipeline needs it). Pen up first —
        never hand over with the pen down."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            try:
                if self.ser is not None:
                    self.ser.write(f"M3 S{GRBL_PEN_UP_S}\n".encode())
                    self.ser.flush()
                    time.sleep(0.2)
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            self._path.clear()
            self._drain(self._cmd_q)
            self.on_log("gantry: released")

    @staticmethod
    def _drain(q):
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    # --- sends (the bus calls these) -------------------------------------------
    def goto(self, x: float, y: float, dt: Optional[float] = None):
        """Absolute G1 into the reach envelope at the tempo generation asked
        for (dt from the chain's recorded timing)."""
        if not self.alive:
            return
        x, y = clamp_to_reach(x, y)
        px, py = self._path[-1][:2] if self._path else self.position
        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        if dist < 0.1:
            return  # sub-resolution jitter
        feed = max(100, min(int(self.max_feed), int(dist / max(0.05, dt or 0.2) * 60)))
        self._path.append((x, y, feed))
        if len(self._path) > 24:  # bounded lag: thin, keep the shape
            self._path = deque(list(self._path)[::2])
        self._cmd_q.put("__MOTION__")

    def pen(self, s: int):
        """M3 barrier — settles in stream order between its segments."""
        if self.alive:
            self._cmd_q.put(f"M3 S{int(s)}")

    # --- serial threads (GrblFrame's discipline, reduced) -----------------------
    def _reader_loop(self):
        while self._running and self.ser is not None:
            try:
                line = self.ser.readline().decode(errors="replace").strip()
            except Exception:
                time.sleep(0.2)
                continue
            if not line or line.startswith("<"):
                continue  # no status polling — position is commanded, not reported
            # GRBL talking back is DATA — a deaf link parked the right arm
            # for a whole evening. Repeats collapse to one line.
            if line != "ok" and line != self._last_err:
                self._last_err = line
                self.on_log(f"gantry: GRBL said {line!r}")
            self._resp_q.put(line)

    def _writer_loop(self):
        unacked = 0

        def reap(timeout: float) -> bool:
            nonlocal unacked
            try:
                self._resp_q.get(timeout=timeout)
            except queue.Empty:
                return False
            unacked = max(0, unacked - 1)
            return True

        while self._running and self.ser is not None:
            try:
                cmd = self._cmd_q.get(timeout=0.5)
            except queue.Empty:
                if unacked > 0:
                    reap(0.1)
                continue
            if cmd == "__MOTION__":
                if not self._path:
                    continue
                x, y, feed = self._path.popleft()
                line = f"G1 X{x:.2f} Y{y:.2f} F{feed}"
                misses = 0
                while unacked >= MAX_UNACKED and self._running:
                    if not reap(1.0):
                        misses += 1
                        if misses >= 5:
                            unacked = 0  # acks lost — reset accounting
                            break
                try:
                    self.ser.write((line + "\n").encode())
                    self.ser.flush()
                    unacked += 1
                    self.position = (x, y)
                    self._sent += 1
                    if self._sent <= 3:
                        self.on_log(f"gantry: streaming {line}" + (" …" if self._sent == 3 else ""))
                except Exception as e:
                    self.on_log(f"gantry write failed: {e}")
                continue
            # non-motion (pen): barrier — settle everything first
            misses = 0
            while unacked > 0 and self._running:
                if not reap(1.0):
                    misses += 1
                    if misses >= 5:
                        unacked = 0
                        break
            try:
                self.ser.write((cmd + "\n").encode())
                self.ser.flush()
                unacked += 1
            except Exception as e:
                self.on_log(f"gantry write failed: {e}")
