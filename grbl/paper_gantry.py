#!/usr/bin/env python3
"""
Paper get-clear, gantry half (Sep 10 2026).

The recorded 'paper' take moves BOTH arms and the gantry, and the pre-draw
paper check plays all of it. During the completion ritual the kinetic bus can
only play the arms: `_send_plan_raw` returns early while `is_executing_cnc` is
set (cleared at Step 5) and the gantry port belongs to the drawing pipeline
until Step 7 — so the take's x/y track is silently dropped, and the post-draw
capture would see a different body position than the pre-draw check does.

This replays that same x/y track onto the serial link grbl_utils already holds
open. Same file the bus picks, same samples, same clamp_to_reach floor, same
sub-resolution jitter skip and feed derivation as GantryLink.goto — so the
gantry lands where the recording put it, without any port handoff (acquiring
it through the bus would reset GRBL and home, which is exactly what the ritual
is deferring until after the photograph).

The pen must already be UP: this drags the carriage across the sheet otherwise.
Step 2a of the ritual guarantees that.
"""

import os
import time
from typing import List, Optional, Tuple

from event_logging.event_logger import LogType, log_json_entry

_MIN_STEP = 0.1  # sub-resolution jitter, same threshold GantryLink.goto uses
_MAX_FEED = 3000  # mm/min ceiling for the replay


def _paper_take_path() -> Optional[str]:
    """The session file the bus's paper_clear would play."""
    from motor_panel.kinetic_bus import PAPER_STATE, TemperamentLibrary

    buckets = TemperamentLibrary(owned=set()).scan()
    takes = sorted(buckets.get(PAPER_STATE) or [])
    if not takes:
        return None
    if len(takes) > 1:
        # paper_clear() uses random.choice, so with several takes the arms and
        # the gantry can end up playing different recordings. Say so out loud.
        print(f"[📷] ⚠️ {len(takes)} 'paper' takes recorded — arms pick at random, gantry replays {takes[0]}; keep one to stay in sync")
    return takes[0]


def paper_gantry_plan() -> Tuple[List[Tuple[float, float, int]], float]:
    """(x, y, feed) moves for the paper take's gantry track, plus its duration.

    Empty when there is no paper take, no x/y track in it, or the motor panel
    cannot be read — every caller treats that as "no gantry move".
    """
    try:
        from grbl.warp_calibration import clamp_to_reach
        from motor_panel.session import Session

        fn = _paper_take_path()
        if not fn:
            return [], 0.0
        session = Session.load(fn)
        track = next((t for t in session.tracks if t.has_take and set(t.channels) & {"x", "y"}), None)
        if track is None:
            return [], 0.0

        moves: List[Tuple[float, float, int]] = []
        px = py = None
        duration = 0.0
        for s in track.samples:
            if "x" not in s or "y" not in s:
                continue
            dt = float(s.get("dt") or 0.0)
            duration += dt
            x, y = clamp_to_reach(float(s["x"]), float(s["y"]))
            if px is None:
                px, py = x, y
                moves.append((x, y, _MAX_FEED // 3))
                continue
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < _MIN_STEP:
                continue
            feed = max(100, min(_MAX_FEED, int(dist / max(0.05, dt) * 60)))
            moves.append((x, y, feed))
            px, py = x, y
        return moves, duration
    except Exception as e:
        print(f"[📷] Gantry get-clear plan unavailable: {e}")
        return [], 0.0


def replay_paper_gantry(ser) -> float:
    """Stream the paper take's gantry track on an already-open GRBL link.

    Returns seconds spent. Never raises — a failed get-clear costs a worse
    photograph, nothing more, and the ritual continues either way.
    """
    from grbl.grbl_utils import send_cmd, wait_until_idle

    started = time.time()
    try:
        moves, duration = paper_gantry_plan()
        if not moves:
            return 0.0

        print(f"[📷] Gantry get-clear: replaying {len(moves)} moves (~{duration:.1f}s of recording)")

        # Drain first. Step 2a asserts pen-up 5x plus a dwell, all deliberately
        # wait_ok=False, so SIX unread "ok"s are sitting in the buffer. Without
        # this, the first six send_cmd calls below consume those instead of
        # their own, every read runs six commands ahead of reality, and the
        # stream loses its backpressure — GRBL's 128-byte RX buffer overflows
        # and rejects a garbled line (observed live: "error: Invalid gcode
        # ID:24" reported against a coordinate that was never the problem).
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        send_cmd(ser, "G90")  # absolute — setup_basic_grbl only sends this when use_absolute_positioning is on
        for x, y, feed in moves:
            send_cmd(ser, f"G1 X{x:.3f} Y{y:.3f} F{feed}")
        wait_until_idle(ser, 60)

        elapsed = time.time() - started
        log_json_entry(
            LogType.GRBL,
            {"action": "paper_gantry_replayed", "moves": len(moves), "recording_s": duration, "duration": elapsed, "take": _paper_take_path()},
            print_message=f"[📷] Gantry clear of the camera ({elapsed:.1f}s)",
        )
        return elapsed
    except Exception as e:
        print(f"[📷] Gantry get-clear replay failed: {e}")
        # Leave the link parked and idle whatever went wrong — $H follows this
        # in the ritual and must not inherit a half-streamed motion or an
        # unread error.
        try:
            ser.reset_input_buffer()
            wait_until_idle(ser, 30)
        except Exception:
            pass
        return time.time() - started


def paper_take_name() -> str:
    """Basename of the take the gantry replay would use ('' when none)."""
    fn = _paper_take_path()
    return os.path.basename(fn) if fn else ""
