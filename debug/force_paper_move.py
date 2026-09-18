"""Force the uArm paper move on the sheet that is on the table right now.

The discard normally only happens at the end of a drawing (the completion
ritual in grbl/grbl_utils.py, then machine.py's on_grbl_drawing_complete hook).
Sep 15 the machine marked a fresh sheet between drawings, so every paper check
since has refused it and no drawing can run — the sheet has to go without a
drawing having been made. This replays the same ritual order by hand:

    photo -> pen UP -> home the gantry clear -> uArm plays UARM_PLAY_FILE -> photo

machine.py must be stopped: it holds /dev/arduino_cnc, /dev/arduino_uarm and
the camera. Photos land in a temp dir, never in event_log/finished_drawings
(an over-broad cleanup glob ate four real captures on Sep 10).

    python debug/force_paper_move.py [--skip-grbl] [--no-photos] [--yes]

--skip-grbl plays the uArm take alone. Use it ONLY when the gantry is visibly
clear of the sheet: Sep 18 homing failed outright (ALARM: Homing fail after a
full 60s cycle), but that failed cycle had already parked the gantry off the
sheet, confirmed by eye and by photo, so the sweep was safe to run anyway.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config.config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    PAPER_DETECTION_GAZE_PAN,
    PAPER_DETECTION_GAZE_TILT,
    UARM_PLAY_FILE,
)

argv = sys.argv[1:]
PHOTOS = "--no-photos" not in argv
OUT_DIR = tempfile.mkdtemp(prefix="paper_move_")


def _fail(msg):
    print(f"\n[ABORT] {msg}")
    sys.exit(1)


def photo(tag):
    """One frame at the paper-check angle. Never fatal — the sheet still goes."""
    if not PHOTOS:
        return None
    try:
        from servo_control.servo_control import ServoController

        servos = ServoController(port="/dev/arduino_lunggaze", baudrate=9600)
        time.sleep(2.0)
        servos.set_pan(PAPER_DETECTION_GAZE_PAN)
        time.sleep(0.4)
        servos.set_tilt(PAPER_DETECTION_GAZE_TILT)
        time.sleep(1.8)
        cam = cv2.VideoCapture(CAMERA_INDEX)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        frame = None
        for _ in range(12):  # let exposure settle
            ok, fr = cam.read()
            if ok:
                frame = fr
            time.sleep(0.1)
        cam.release()
        if frame is None:
            print(f"[photo] no frame for {tag}")
            return None
        path = os.path.join(OUT_DIR, f"{tag}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[photo] {tag} -> {path}")
        return path
    except Exception as e:
        print(f"[photo] {tag} failed: {e}")
        return None


if os.popen("pgrep -f '[p]ython machine.py'").read().strip():  # bracket: the check must not match its own shell
    _fail("machine.py is running — it owns the CNC, the uArm and the camera. Stop it first (./stop_machine.sh).")

target = UARM_PLAY_FILE
base, ext = os.path.splitext(target)
while base.lower().endswith(".smooth"):
    base = base[:-7]
target = f"{base}{ext or '.txt'}"
if not os.path.exists(target):
    _fail(f"play file missing: {target}")

print(f"Paper move : {os.path.basename(target)}")
print(f"Photos     : {'on -> ' + OUT_DIR if PHOTOS else 'off'}")
print("Sequence   : photo, pen up, home the gantry, uArm play, photo")
if "--yes" not in argv:
    if input("\nNobody is in the room. Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        _fail("cancelled")

photo("before")

if "--skip-grbl" in argv:
    print("\n=== GRBL: SKIPPED — the gantry is clear by eye, not by homing ===")
else:
    print("\n=== GRBL: pen up, then home so the gantry is clear of the sheet ===")
    ser = None
    try:
        from grbl.grbl_utils import ensure_homed, find_grbl_port

        ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
        if not ser:
            _fail("no GRBL port — refusing to sweep the arm over an un-homed gantry")
        # ensure_homed does its own prep: status, $X out of Alarm, then the
        # centralized pen-up. Do NOT send a pen-up burst first — it is
        # wait_ok=False x6, and the stale "ok"s get eaten by whatever streams
        # next (Sep 10). Sep 18 that ate the status read, the prep never saw
        # Alarm, so $X never went and $H was rejected against an alarm lock.
        ensure_homed(ser, max_retries=3)
        print("[grbl] homed, pen up")
    except SystemExit:
        raise
    except Exception as e:
        _fail(f"GRBL prep failed ({e}) — the gantry may still be over the sheet, so the uArm was NOT run")
    finally:
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass

print("\n=== uArm: play the paper move ===")
app = None
try:
    from uarm_control.teach_menu import UArmTeachApp

    app = UArmTeachApp()
    app.connect()
    app.smoothing_enabled = False
    app.use_home_before_play = False
    app.use_home_after_play = False
    app.play_file = target
    app.play()
    if getattr(app, "teach", None):
        deadline = time.time() + 90
        while app.teach.is_playing() and time.time() < deadline:
            time.sleep(0.5)
        print("[uarm] " + ("timed out after 90s" if app.teach.is_playing() else "movement completed"))
    else:
        print("[uarm] no teach handle — waiting 20s as a fallback")
        time.sleep(20.0)
except Exception as e:
    print(f"[uarm] play failed: {e}")
finally:
    try:
        if app is not None:
            app.disconnect()
    except Exception:
        pass

time.sleep(2.0)
photo("after")

print(f"\nDone. Photos: {OUT_DIR}")
print("The machine is still stopped — ./start_impostor.sh when you want it back.")
