"""Offline check of the felt-time stream changes (Aug 20).

1. _stream_history renders inter-entry and trailing gaps as unstamped
   "(about 20 minutes later)" lines — words, never raw stamp arithmetic.
2. llama_server's hybrid seam skips a trailing gap marker: after a silence
   the log ends with the marker and generation starts fresh (no prefill),
   instead of continuing a pre-gap thought mid-clause.
3. The break threshold: 20-minute lulls survive (marked), only ≥2h wipes.

Run: python debug/test_stream_gaps.py  (no server, no camera needed)
"""

import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAIL += 1


def make_captioner_shell():
    """Captioner without __init__ (threads, camera) — just the stream state."""
    from captioner.captioner import Captioner

    c = Captioner.__new__(Captioner)
    c._stream = deque(maxlen=24)
    c._stream_ts = deque(maxlen=24)
    return c


def test_render():
    print("\n[1] _stream_history gap rendering")
    c = make_captioner_shell()
    now = time.time()
    entries = [
        ("The lamp is still on.", now - 1500),  # 25 min ago
        ("He set a cup down by the keyboard.", now - 1440),  # 24 min ago — then a ~22 min lull
        ("The chair is empty now.", now - 100),
        ("Dust in the window light.", now - 40),  # newest, 40s ago — no trailing gap
    ]
    for text, ts in entries:
        c._stream.append(text)
        c._stream_ts.append(ts)

    lines = c._stream_history()
    print("    " + "\n    ".join(lines))
    markers = [l for l in lines if l.startswith("(") and l.endswith(" later)")]
    check("one gap marker for the 22-min lull", len(markers) == 1, str(markers))
    check("marker uses words not integers", markers and "about 20 minutes later" in markers[0], str(markers))
    check("marker sits between the right entries", lines.index(markers[0]) == 2 if markers else False)
    check("no trailing marker when newest is fresh", not lines[-1].startswith("("))
    check("stamped lines keep the log shape", all(":" in l[:6] for l in lines if not l.startswith("(")))

    # Trailing gap: newest entry 8 minutes old
    c2 = make_captioner_shell()
    c2._stream.append("The room settles.")
    c2._stream_ts.append(now - 480)
    lines2 = c2._stream_history()
    print("    " + "\n    ".join(lines2))
    check("trailing marker when NOW is far from the last entry", lines2[-1] == "(about 8 minutes later)", str(lines2))


def test_hybrid_seam():
    print("\n[2] hybrid seam vs gap markers (llama_server._append_stream_and_user)")
    os.environ["STREAM_MODE"] = "hybrid"
    import config.config as cfg

    cfg.STREAM_MODE = "hybrid"
    from utils.llama_server import _append_stream_and_user

    user = {"role": "user", "content": "the world's turn"}

    # No gap: newest real entry becomes the seam
    msgs = []
    prefill = _append_stream_and_user(msgs, ["14:02 — The lamp is on.", "14:03 — He sits down and opens the notebook"], dict(user))
    check("no-gap: newest entry becomes the seam", prefill.startswith("He sits down"), repr(prefill))
    check("no-gap: log keeps the older line", msgs[0]["content"] == "14:02 — The lamp is on.")

    # Trailing gap marker: no seam, marker stays as the log's last line
    msgs = []
    history = ["14:02 — The lamp is on.", "14:03 — He sits down and opens the notebook.", "(about 20 minutes later)"]
    prefill = _append_stream_and_user(msgs, history, dict(user))
    check("gap-final: no prefill seam", prefill == "", repr(prefill))
    check("gap-final: marker is the log's last line", msgs[0]["content"].endswith("(about 20 minutes later)"), msgs[0]["content"])
    check("gap-final: user turn comes last", msgs[-1]["role"] == "user")

    # Mid-log marker with a fresh entry after it: seam works normally
    msgs = []
    history = ["14:02 — The lamp is on.", "(about 20 minutes later)", "14:23 — The chair is empty now"]
    prefill = _append_stream_and_user(msgs, history, dict(user))
    check("mid-log marker: fresh entry still seams", prefill.startswith("The chair is empty"), repr(prefill))
    check("mid-log marker: marker stays in the log", "(about 20 minutes later)" in msgs[0]["content"])


def test_marker_echo_strip():
    print("\n[4] marker-echo strip at storage (_strip_list_shape)")
    from captioner.captioner import Captioner

    live = "23 - he stays seated laptop open on chair... (about 8 minutes later) the man in white shirt turns."
    out = Captioner._strip_list_shape(live)
    check("echoed marker stripped (live Aug 20 23:51 shape)", "later)" not in out and "(about" not in out, repr(out))
    check("surrounding words survive", "he stays seated" in out and "the man in white shirt turns" in out, repr(out))
    honest = "A moment later he shifts his weight toward the window."
    check("honest prose time talk untouched", Captioner._strip_list_shape(honest) == honest)


def test_break_threshold():
    print("\n[3] break threshold")
    from config.config import STREAM_BREAK_SECONDS, STREAM_GAP_MARK_SECONDS

    check("mark threshold is minutes-scale", STREAM_GAP_MARK_SECONDS == 180)
    check("hard break is hours-scale (reorientation takes over)", STREAM_BREAK_SECONDS == 7200)


if __name__ == "__main__":
    test_render()
    test_hybrid_seam()
    test_marker_echo_strip()
    test_break_threshold()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
