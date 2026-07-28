"""Prove the right arm joins the runtime temperament.

1. GantryLink serial discipline against a pty GRBL: G1 lines with feeds
   from chain tempo, reach-clamped coords, pen as ordered barrier,
   clean release (pen up on handover)
2. KineticBus with a gantry: owned set widens to x/y, generation flows
   goto calls, the drawing gate freezes them, release stops them,
   re-acquire resumes

  python debug/test_gantry_runtime.py
"""

import math
import os
import pty
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serial as pyserial

import motor_panel.session as session_mod
from motor_panel import kinetic_bus as kb
from motor_panel.gantry import GantryLink
from motor_panel.session import Session

LOOP = 2.0
RATE = 20


def grbl_emulator(master_fd, received, stop):
    """Reads lines, records them, answers ok — a polite fake GRBL."""
    buf = b""
    while not stop.is_set():
        try:
            data = os.read(master_fd, 256)
        except OSError:
            break
        if not data:
            break
        if b"?" in data:  # real GRBL: '?' is an immediate status char, not a line
            os.write(master_fd, b"<Idle|MPos:0.000,0.000,0.000|FS:0,0>\r\n")
            data = data.replace(b"?", b"")
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            received.append(line.decode())
            os.write(master_fd, b"ok\r\n")


def main():
    failures = []

    # --- 1: link discipline against the pty ------------------------------------
    master, slave = pty.openpty()
    received, stop = [], threading.Event()
    threading.Thread(target=grbl_emulator, args=(master, received, stop), daemon=True).start()
    link = GantryLink()
    link.attach(pyserial.Serial(os.ttyname(slave), 115200, timeout=0.3))
    link.goto(30.0, 20.0, 0.5)
    link.goto(35.0, 22.0, 0.5)
    link.pen(52)
    link.goto(32.0, 18.0, 0.4)
    time.sleep(1.0)
    g1s = [l for l in received if l.startswith("G1")]
    pens = [l for l in received if l.startswith("M3")]
    if len(g1s) != 3:
        failures.append(f"expected 3 G1 segments, got {g1s}")
    if not all("F" in l for l in g1s):
        failures.append(f"G1 without feed: {g1s}")
    if pens != ["M3 S52"]:
        failures.append(f"pen barrier wrong: {pens}")
    if received and received.index(pens[0]) < received.index(g1s[1]):
        failures.append("pen arrived before its preceding segment (stream order broken)")
    # reach clamp: a wild target must land inside the envelope
    link.goto(500.0, 500.0, 0.5)
    time.sleep(0.4)
    wild = [l for l in received if l.startswith("G1")][-1]
    x = float(wild.split("X")[1].split(" ")[0])
    if x > 120:
        failures.append(f"wild target not reach-clamped: {wild}")
    link.release()
    time.sleep(0.3)
    if "M3 S" not in received[-1]:
        failures.append(f"release did not raise the pen last (tail: {received[-3:]})")
    if link.alive:
        failures.append("link still alive after release")
    stop.set()
    print(f"link: {len(g1s)}+1 clamped G1s with feeds, pen barrier ordered, released pen-up")

    # --- 2: the bus plays the gantry -------------------------------------------
    tmp = tempfile.mkdtemp(prefix="gantry_test_")
    session_mod.SESSIONS_DIR = tmp
    try:
        s = Session("calm_observant_a", loop_len=LOOP)
        gantry_t = next(t for t in s.tracks if t.channels == ["x", "y"])
        arm = next(t for t in s.tracks if t.name == "left arm")
        n = int(LOOP * RATE)
        gantry_t.samples = [
            {"t": i / RATE, "dt": 1 / RATE, "x": 25 + 8 * math.sin(2 * math.pi * i / n) + (i % 3) - 1, "y": 20 + 6 * math.cos(2 * math.pi * i / n)}
            for i in range(n)
        ]
        arm.samples = [{"t": i / RATE, "dt": 1 / RATE, "elbow": 90 + 10 * math.sin(2 * math.pi * i / n), "shoulder": 90.0} for i in range(n)]
        s.save(export=True)

        class FakeGantry:
            def __init__(self):
                self.alive = False
                self.gotos = []
                self.on_log = lambda m: None

            def connect_and_home(self):
                self.alive = True
                return True

            def goto(self, x, y, dt=None):
                self.gotos.append((x, y))

            def pen(self, s):
                pass

            def release(self):
                self.alive = False

        fake = FakeGantry()
        ctx = {"drawing": False}
        bus = kb.KineticBus(
            library=kb.TemperamentLibrary(sessions_dir=tmp, owned=kb.OWNED_CHANNELS | {"x", "y"}),
            get_emotion=lambda: "calm_observant",
            is_drawing=lambda: ctx["drawing"],
            get_gaze=lambda: (0.0, 0.0),
            get_person=lambda: "absent",
            on_log=lambda m: None,
            send_ease=lambda d: None,
            get_state=lambda: {"x": 25.0, "y": 20.0, "elbow": 90.0, "shoulder": 90.0},
            gantry=fake,
        )
        if "x" not in bus.owned or "y" not in bus.owned:
            failures.append(f"gantry bus did not widen ownership: {sorted(bus.owned)}")
        bus.gantry_acquire()
        bus.enable()
        time.sleep(3.5)
        flowing = len(fake.gotos)
        if flowing == 0:
            failures.append("generation never reached the gantry")
        ctx["drawing"] = True  # the gate must freeze the gantry instantly
        time.sleep(0.5)
        frozen = len(fake.gotos)
        time.sleep(1.5)
        if len(fake.gotos) != frozen:
            failures.append("drawing gate leaked gantry sends")
        ctx["drawing"] = False
        bus.gantry_release()
        if fake.alive:
            failures.append("gantry_release left the link alive")
        time.sleep(1.5)
        released_count = len(fake.gotos)
        time.sleep(1.0)
        if len(fake.gotos) != released_count:
            failures.append("sends continued into a released gantry")
        bus.gantry_acquire()
        time.sleep(2.0)
        if len(fake.gotos) <= released_count:
            failures.append("re-acquire did not resume gantry generation")
        print(f"bus: {flowing} gotos flowing, gate froze at {frozen}, release stopped, re-acquire resumed ({len(fake.gotos)} total)")
        bus.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
