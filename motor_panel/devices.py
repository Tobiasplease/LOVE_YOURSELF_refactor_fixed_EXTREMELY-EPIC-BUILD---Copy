"""UI-free actuator layer for the unified motor panel.

Every servo/stepper in the machine (except the uArm) behind one small
abstraction: a Device owns a serial port and a set of Channels; setting a
channel value formats and sends the device's wire command, rate-limited and
deduplicated. Unconnected devices run in simulate mode — commands are logged,
not sent — so the panel works with nothing plugged in.

This layer is deliberately Tkinter-free: the runtime kinetic bus can drive
the same objects later.

Wire protocols (verified against the live controllers July 10 2026):
  lunggaze   PAN:{deg}  TILT:{deg}  LUNG:{deg}                 9600 baud
  lefthand   HAND,f0,f1,f2,f3   SERVO,{pin},{deg}   MOOD,{e}   9600 baud
             LEFT_ARM_ENABLE / LEFT_ARM_DISABLE
  lightbulb  B:{0-255}   F                                     9600 baud
  (GRBL is G-code-shaped, handled by grbl/grbl_utils.py, not this class)
"""

import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

DIRECTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motor_directions.json")


def load_directions() -> dict:
    try:
        with open(DIRECTIONS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_directions(device: str, channel: str, flag: bool):
    d = load_directions()
    d[f"{device}.{channel}"] = flag
    with open(DIRECTIONS_PATH, "w") as f:
        json.dump(d, f, indent=1)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    LEFT_ARM_ELBOW_LIMITS,
    LEFT_ARM_SHOULDER_LIMITS,
    LEFT_ARM_WRIST_LIMITS,
    LUNG_MAX,
    LUNG_MIN,
    PAN_MAX,
    PAN_MIN,
    TILT_MAX,
    TILT_MIN,
)

try:
    import serial
except ImportError:
    serial = None


@dataclass
class Channel:
    name: str
    lo: int
    hi: int
    neutral: int
    fmt: Optional[Callable[[int], str]] = None  # per-channel command; None -> group command
    group: Optional[str] = None  # channels sharing one composite command line
    smooth: bool = False  # ease toward targets instead of jumping (arm servos)
    invert: bool = False  # mirror the WIRE value (mounting direction) — logical values unchanged
    value: int = field(init=False)
    target: float = field(init=False)
    pos: float = field(init=False)

    def __post_init__(self):
        self.value = self.neutral
        self.target = float(self.neutral)
        self.pos = float(self.neutral)


class SerialDevice:
    """One Arduino: a port, its channels, and named one-shot commands."""

    def __init__(
        self,
        name: str,
        port: str,
        baud: int,
        channels: List[Channel],
        extras: Optional[Dict[str, str]] = None,
        group_fmt: Optional[Dict[str, Callable[[List[int]], str]]] = None,
        boot_delay: float = 2.0,
        min_interval: float = 0.02,
    ):
        self.name = name
        self.port = port
        self.baud = baud
        self.channels = {c.name: c for c in channels}
        self.channel_order = [c.name for c in channels]
        self.extras = extras or {}
        self.group_fmt = group_fmt or {}
        self.boot_delay = boot_delay
        self.min_interval = min_interval
        self.ser = None
        self.on_line: Optional[Callable[[str, str, bool], None]] = None  # (device, line, simulated)
        self._last_send = 0.0
        self._last_line: Dict[str, str] = {}
        # Single ordered writer: UI events (drag storms at 60Hz) must never
        # block, back up, or reorder. Keyed lines coalesce latest-wins in
        # _pending, so a thousand stale drag targets collapse to the newest.
        self._q: "queue.Queue" = queue.Queue()
        self._pending: Dict[str, str] = {}
        self._plock = threading.Lock()
        threading.Thread(target=self._writer_loop, daemon=True).start()
        # Exponential easing for smooth channels: raw 20Hz drag targets and
        # replayed samples become continuous motion instead of step jumps.
        self.smooth_time = 0.25  # seconds to close ~63% of the gap
        if any(c.smooth for c in channels):
            threading.Thread(target=self._smoother_loop, daemon=True).start()

    @property
    def connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def connect(self) -> str:
        if serial is None:
            return "pyserial not installed — simulate only"
        if self.connected:
            return "already connected"
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(self.boot_delay)
            return f"connected {self.port} @ {self.baud}"
        except Exception as e:
            self.ser = None
            return f"connect failed: {e}"

    def disconnect(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def send_line(self, line: str, key: Optional[str] = None):
        """Non-blocking enqueue. Keyed lines coalesce (latest wins); unkeyed
        lines are sent in order. The writer thread does the actual I/O."""
        if key is not None:
            with self._plock:
                fresh = key not in self._pending
                self._pending[key] = line
            if fresh:
                self._q.put(key)
        else:
            self._q.put((None, line))

    def _writer_loop(self):
        while True:
            item = self._q.get()
            if isinstance(item, tuple):
                key, line = item
            else:
                key = item
                with self._plock:
                    line = self._pending.pop(key, None)
                if line is None:
                    continue
            self._write_now(line, key)

    def _write_now(self, line: str, key: Optional[str]):
        if key is not None and self._last_line.get(key) == line:
            return
        wait = self.min_interval - (time.time() - self._last_send)
        if wait > 0:
            time.sleep(wait)
        simulated = not self.connected
        if not simulated:
            try:
                self.ser.write((line + "\n").encode())
                self.ser.flush()
            except Exception as e:
                line = f"{line}  [WRITE FAILED: {e}]"
                simulated = True
        self._last_send = time.time()
        if key is not None:
            self._last_line[key] = line
        if self.on_line:
            self.on_line(self.name, line, simulated)

    def _wire(self, ch: Channel, value: int) -> int:
        """Logical -> wire value: mirrored within the range when inverted."""
        return ch.lo + ch.hi - value if ch.invert else value

    def set_invert(self, name: str, flag: bool):
        ch = self.channels[name]
        ch.invert = flag
        save_directions(self.name, name, flag)
        self._last_line.pop(f"ch:{name}", None)  # force re-emit so the flip shows now
        if ch.smooth:
            ch.pos += 0.5  # nudge the smoother to resend
        else:
            self.set_channel(name, ch.value)

    def _smoother_loop(self):
        TICK = 0.02
        smooth_chs = [c for c in self.channels.values() if c.smooth]
        while True:
            for ch in smooth_chs:
                delta = ch.target - ch.pos
                if abs(delta) < 0.05:
                    ch.pos = ch.target
                else:
                    ch.pos += delta * min(1.0, TICK / max(0.02, self.smooth_time))
                iv = int(round(ch.pos))
                if iv != ch.value:
                    ch.value = iv
                    self.send_line(ch.fmt(self._wire(ch, iv)), key=f"ch:{ch.name}")
            time.sleep(TICK)

    def set_channel(self, name: str, value: float):
        ch = self.channels[name]
        if ch.smooth:
            ch.target = float(max(ch.lo, min(ch.hi, value)))
            return  # the smoother emits the actual motion
        ch.value = max(ch.lo, min(ch.hi, int(value)))
        if ch.group:
            vals = [self._wire(self.channels[n], self.channels[n].value) for n in self.channel_order if self.channels[n].group == ch.group]
            self.send_line(self.group_fmt[ch.group](vals), key=f"group:{ch.group}")
        else:
            self.send_line(ch.fmt(self._wire(ch, ch.value)), key=f"ch:{name}")

    def send_extra(self, name: str):
        self.send_line(self.extras[name], key=None)

    def all_neutral(self):
        groups_done = set()
        for cname in self.channel_order:
            ch = self.channels[cname]
            if ch.smooth:
                ch.target = float(ch.neutral)
                continue
            ch.value = ch.neutral
            if ch.group:
                if ch.group not in groups_done:
                    groups_done.add(ch.group)
                    vals = [self._wire(self.channels[n], self.channels[n].neutral) for n in self.channel_order if self.channels[n].group == ch.group]
                    self.send_line(self.group_fmt[ch.group](vals), key=f"group:{ch.group}")
            else:
                self.send_line(ch.fmt(self._wire(ch, ch.neutral)), key=f"ch:{cname}")


EMOTIONS = ["energized_engaged", "alert_curious", "calm_observant", "quiet_detached", "withdrawn_distant"]


def build_devices() -> List[SerialDevice]:
    lunggaze = SerialDevice(
        "lunggaze",
        os.getenv("LUNGGAZE_PORT", "/dev/arduino_lunggaze"),
        9600,
        channels=[
            Channel("pan", PAN_MIN, PAN_MAX, 90, fmt=lambda v: f"PAN:{v}"),
            Channel("tilt", TILT_MIN, TILT_MAX, (TILT_MIN + TILT_MAX) // 2, fmt=lambda v: f"TILT:{v}"),
            Channel("lung", LUNG_MIN, LUNG_MAX, (LUNG_MIN + LUNG_MAX) // 2, fmt=lambda v: f"LUNG:{v}"),
        ],
        boot_delay=0.5,
        min_interval=0.01,
    )
    lefthand = SerialDevice(
        "lefthand",
        os.getenv("HAND_CONTROLLER_PORT", "/dev/arduino_lefthand"),
        9600,
        channels=[
            Channel("finger0", 0, 180, 90, group="hand"),
            Channel("finger1", 0, 180, 90, group="hand"),
            Channel("finger2", 0, 180, 90, group="hand"),
            Channel("finger3", 0, 180, 90, group="hand"),
            Channel("elbow", *LEFT_ARM_ELBOW_LIMITS, fmt=lambda v: f"SERVO,4,{v}", smooth=True),
            Channel("shoulder", *LEFT_ARM_SHOULDER_LIMITS, fmt=lambda v: f"SERVO,5,{v}", smooth=True),
            Channel("wrist", *LEFT_ARM_WRIST_LIMITS, fmt=lambda v: f"SERVO,6,{v}", smooth=True),
        ],
        # extras removed July 27: MOOD,{e} and LEFT_ARM_ENABLE/DISABLE were
        # wanderer-firmware commands — the clean firmware ignores them all
        # (the temperament system replaced what they did)
        group_fmt={"hand": lambda vals: "HAND," + ",".join(str(v) for v in vals)},
        boot_delay=2.0,
        min_interval=0.02,
    )
    lightbulb = SerialDevice(
        "lightbulb",
        os.getenv("LIGHTBULB_PORT", "/dev/arduino_lightbulb"),
        9600,
        channels=[Channel("brightness", 0, 255, 0, fmt=lambda v: f"B:{v}")],
        extras={"caption flash": "F"},
        boot_delay=0.2,
        min_interval=0.02,
    )
    devices = [lunggaze, lefthand, lightbulb]
    saved = load_directions()
    for d in devices:
        for name, ch in d.channels.items():
            ch.invert = saved.get(f"{d.name}.{name}", False)
    return devices
