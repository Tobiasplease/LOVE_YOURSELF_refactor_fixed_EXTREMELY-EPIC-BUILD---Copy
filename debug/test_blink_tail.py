"""Offline check of the blink tail-splice (Sep 4).

A blink used to seed ONE entry (the prior session's last thought — lately
always a monoculture sample). Now the persisted stream tail splices in with
real timestamps, every entry passing the same mouth gates.

Run: python debug/test_blink_tail.py  (no server, no camera needed)
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


def shell(gap=300, tail=None, last=""):
    import captioner.captioner as cap_mod
    from captioner.captioner import Captioner

    cap_mod.log_json_entry = lambda *a, **k: None
    c = Captioner.__new__(Captioner)
    c._stream = deque(maxlen=24)
    c._stream_ts = deque(maxlen=24)
    c.memory_loaded_from_previous = True
    c.last_session_gap = gap
    c.prior_session_stream_tail = tail or []
    c.prior_session_last_caption = last
    return c


def main():
    print("\n[1] tail splice")
    now = time.time()
    tail = [
        {"text": "The lamp is still on.", "ts": now - 900},
        {"text": "no sentence structure here at all", "ts": now - 800},
        {"text": "Let me know what you think!", "ts": now - 700},  # meta — inadmissible
        {"text": "The pen sat untouched while he typed.", "ts": now - 600},
    ]
    c = shell(tail=tail)
    check("blink accepted", c._try_blink_resume())
    check(
        "clean entries spliced, gated ones dropped",
        list(c._stream) == ["The lamp is still on.", "The pen sat untouched while he typed."],
        str(list(c._stream)),
    )
    check("real timestamps kept (backdated)", all(ts < now - 250 for ts in c._stream_ts), str([round(now - t) for t in c._stream_ts]))

    print("\n[2] fallback + gates")
    c2 = shell(tail=[], last="One prior thought, structured.")
    c2._try_blink_resume()
    check("no tail -> single-seed fallback", list(c2._stream) == ["One prior thought, structured."])

    c3 = shell(gap=9999)
    check("long gap refuses blink (ceremony runs)", not c3._try_blink_resume())

    c4 = shell(
        tail=[
            {
                "text": "word salad incible indestructible immortal eternal everlasting permanent boundless ceaseless timeless deathless ageless endless",
                "ts": now - 100,
            }
        ]
    )
    c4._try_blink_resume()
    check("gated-out tail -> empty stream, still a blink", len(c4._stream) == 0)

    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
