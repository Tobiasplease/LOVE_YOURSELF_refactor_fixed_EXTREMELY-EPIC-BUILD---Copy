#!/usr/bin/env python3
"""
Minimal uArm Teach/Play CLI (Official SDK)

Purpose
- Record and replay motions using the official uArm Teach class.
- Avoids GUI and extra layers to troubleshoot timing/speed capture.

Usage examples
- Record 8s at 50 Hz into a file:
    python debug/uarm_teach_cli.py record --duration 8 --interval 0.02 --file movement_recordings/uarm/teach_test.txt

- Play the recorded file (attaches servos, disables standby before play):
    python debug/uarm_teach_cli.py play --file movement_recordings/uarm/teach_test.txt

Port selection
- Auto-detect by default via USB VID/PID filter.
- Override with --port or env UARM_PORT.
"""

import argparse
import os
import sys
import time

try:
    from uarm.wrapper.swift_api import SwiftAPI
    from uarm.swift.teach import Teach
except Exception as e:
    print("Error: uArm SDK not available. Ensure pyuf is installed.")
    print(f"Details: {e}")
    sys.exit(1)


def connect_uarm(port: str | None) -> SwiftAPI:
    if port:
        print(f"Connecting to uArm on explicit port: {port}")
        swift = SwiftAPI(port=port)
    else:
        print("Connecting to uArm via USB VID/PID filter (2341:0042)…")
        swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
    swift.waiting_ready(timeout=10)
    print("Connected. Firmware:", swift.get_device_info())
    return swift


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def cmd_record(args):
    port = args.port or os.getenv('UARM_PORT')
    swift = connect_uarm(port)
    teach = Teach(args.file, swift)
    try:
        print("Enabling standby mode (limp) for manual guidance…")
        try:
            teach.start_standby_mode()
        except Exception as e:
            print("Warn: start_standby_mode failed:", e)

        ensure_dir(args.file)
        print(f"Starting record for {args.duration:.2f}s at interval {args.interval:.3f}s (≈{1/args.interval:.0f} Hz)…")
        teach.start_record(interval=float(args.interval))
        time.sleep(float(args.duration))
        teach.stop_record()

        size = os.path.getsize(args.file) if os.path.exists(args.file) else 0
        print(f"Recording saved to {args.file} ({size} bytes)")
    finally:
        try:
            print("Restoring standby mode…")
            teach.start_standby_mode()
        except Exception:
            pass
        try:
            swift.disconnect()
        except Exception:
            pass


def cmd_play(args):
    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    port = args.port or os.getenv('UARM_PORT')
    swift = connect_uarm(port)
    teach = Teach(args.file, swift)

    try:
        print("Disabling standby and attaching servos for playback…")
        try:
            teach.stop_standby_mode()
        except Exception as e:
            print("Warn: stop_standby_mode failed:", e)
        try:
            swift.set_servo_attach()
        except Exception as e:
            print("Warn: set_servo_attach failed:", e)

        print("Starting playback…")
        teach.start_play()

        last_pct = -1
        while teach.is_playing():
            try:
                prog = teach.get_progress(wait=False)  # (index, percent)
                if prog and len(prog) >= 2:
                    pct = prog[1]
                    if pct != last_pct:
                        print(f"Progress: {pct:.1f}%")
                        last_pct = pct
            except Exception:
                pass
            time.sleep(0.1)

        print("Playback complete.")
    finally:
        try:
            print("Re-enabling standby mode…")
            teach.start_standby_mode()
        except Exception:
            pass
        try:
            swift.disconnect()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Minimal uArm Teach/Play CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="Record a motion to file")
    p_rec.add_argument("--file", default="movement_recordings/uarm/teach_motion.txt")
    p_rec.add_argument("--duration", type=float, default=8.0)
    p_rec.add_argument("--interval", type=float, default=0.02, help="Sampling interval seconds (lower = smoother)")
    p_rec.add_argument("--port", default=None, help="Explicit serial port (overrides env UARM_PORT)")
    p_rec.set_defaults(func=cmd_record)

    p_play = sub.add_parser("play", help="Play a recorded motion from file")
    p_play.add_argument("--file", default="movement_recordings/uarm/teach_motion.txt")
    p_play.add_argument("--port", default=None, help="Explicit serial port (overrides env UARM_PORT)")
    p_play.set_defaults(func=cmd_play)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

