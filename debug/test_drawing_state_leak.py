"""The stuck-drawing-flag regression (Aug 10).

A TimeoutError on one g-code line ("M3 S38 ; PEN UP (fast)") raised out of
execute_gcode_file past DrawingState.end_drawing(). The flag latched, and for
the next half hour every drawing check printed "GRBL execution currently in
progress" while the machine sat idle — reported by the artist as "there is no
grbl execution in progress".

Covers: the error path releases the flag, the staleness backstop clears a flag
leaked by any other path, and the block reason is no longer mislabelled.

    python debug/test_drawing_state_leak.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.drawing_state import DrawingState

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


print("\n1. the error path releases the flag")
from grbl.grbl_utils import _release_drawing_state

DrawingState.start_drawing(intent="a chair", drawing_file="/tmp/x.gcode", description="a chair")
check("flag set while drawing", DrawingState.is_drawing() is True)
_release_drawing_state()
check("flag cleared after the error path runs", DrawingState.is_drawing() is False)

print("\n2. staleness backstop (any other leak path)")
DrawingState.start_drawing(intent="a chair", drawing_file="/tmp/x.gcode", description="a chair")
DrawingState._drawing_start_time = time.time() - (DrawingState._MAX_DRAWING_SECONDS + 60)
check("a flag older than the cap reads as not-drawing", DrawingState.is_drawing() is False)
check("stale flag is actually reset, not just reported", DrawingState._is_drawing is False)

print("\n3. a genuinely running drawing is NOT cleared early")
DrawingState.start_drawing(intent="a chair", drawing_file="/tmp/x.gcode", description="a chair")
DrawingState._drawing_start_time = time.time() - 240  # the observed ~4 min run
check("4-minute-old drawing still counts as drawing", DrawingState.is_drawing() is True)
DrawingState.end_drawing()

print("\n4. the block reason is reported, not assumed to be cooldown")
from drawing.drawing import DrawingController

d = DrawingController()
d.last_drawing_time = time.time() - 10_000  # cooldown long expired
DrawingState.start_drawing(intent="a chair", drawing_file="/tmp/x.gcode", description="a chair")
ready = d.ready_to_draw()
check("blocked while the flag is set", ready is False)
check(f"reason names GRBL, not cooldown (got: {d.last_block_reason!r})", "GRBL" in d.last_block_reason)
DrawingState.end_drawing()
check("ready once the flag clears", d.ready_to_draw() is True)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
