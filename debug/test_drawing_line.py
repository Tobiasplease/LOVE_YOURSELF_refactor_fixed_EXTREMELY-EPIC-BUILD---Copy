"""Offline checks for the drawing-history line (Sep 5): condensed, named as
drawings, and dosed instead of riding every caption.

Run:  python debug/test_drawing_line.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.prompts import _drawing_line_due  # noqa: E402
from config import config  # noqa: E402
from drawing.drawing_memory import DrawingMemory  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


dm = object.__new__(DrawingMemory)
now = time.time()
dm._executed_subjects = lambda max_count=5, title_chars=45: (
    [
        "Two gloved hands holding a tiny, indistinct lump",
        "A single red foam finger pointing straight down",
        "The man at the desk, hunching over a small red thing",
    ],
    [now - 30 * 3600, now - 3 * 86400, now - 9 * 86400],
)
line = dm.get_arc_line_named()
check(
    "named as drawings, quoted, newest with age",
    line.startswith("Your last drawing, ") and '"Two gloved hands holding a tiny, indistinct lump"' in line,
    line,
)
check(
    "older two as titles, not a scene sentence",
    'Before it: "A single red foam finger pointing straight down"; "The man at the desk, hunching over a small red thing".' in line,
    line,
)
check("no 'earlier,' scene phrasing", "earlier," not in line and "My last" not in line)
dm._executed_subjects = lambda max_count=5, title_chars=45: (["A spiral", "A spiral", "A cup"], [now - 600, now - 7200, now - 86400])
dm._same_motif = lambda a, b: a == b
line = dm.get_arc_line_named()
check("repeat run folded", "drawn twice in a row" in line and 'Before it: "A cup".' in line, line)


class A:
    pass


class D:
    last_drawing_time = 0.0

    def desire_shadow_verdict(self):
        return {"drawing_directed": False}


a = A()
a._caption_count = 50
a.drawing = D()
rides = [_drawing_line_due(a) for _ in range(config.DRAWING_LINE_EVERY_N * 2)]
check("dosed: every Nth build", sum(rides) == 2, rides)
b = A()
b._caption_count = 1
b.drawing = D()
check("first captions after boot: always", _drawing_line_due(b))
c = A()
c._caption_count = 50
c.drawing = D()
c.drawing.last_drawing_time = time.time() - 60
check("right after a drawing: always", _drawing_line_due(c))
d = A()
d._caption_count = 50
d.drawing = D()
d.drawing.desire_shadow_verdict = lambda: {"drawing_directed": True}
check("drawing-directed want: always", _drawing_line_due(d))

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
