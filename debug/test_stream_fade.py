#!/usr/bin/env python3
"""Sep 14 — the window fades by position (captioner._consolidate_stream_if_needed).

Artist: "The shape needs to be time-aware so it's never exactly the same
regardless. Just like a real memory it degrades and changes depending on the
current state of the mind."

The fold has existed since July and fired on total length: the window runs
~3,100 characters against a 12,000 threshold, so it fired zero times in the runs
measured and the model was handed a perfect transcript of its last twenty-four
lines on every call. That is what let three different forms lock in over two
days. It now fires on position: the head stays word-for-word, because the prefill
continues its last sentence mid-clause, and the tail becomes one note that is
itself folded again later.

Fakes the model call. Importing captioner.captioner mints a stub run log; sweep
event_log afterwards. Run: .venv/bin/python debug/test_stream_fade.py"""
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.inference as _inf  # noqa: E402
from captioner.captioner import Captioner  # noqa: E402
from config import config as cfg  # noqa: E402

fails = 0
CALLS = []


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


_inf.query_model = lambda **kw: CALLS.append(kw) or "A note in its own words about the older part."
import captioner.captioner as _cap  # noqa: E402

_cap.query_model = _inf.query_model
_cap.log_json_entry = lambda *a, **k: None


class Fake(Captioner):
    def __init__(self):
        self._stream = deque(maxlen=24)
        self._stream_ts = deque(maxlen=24)

    def _stream_admissible(self, line):
        return True


def feed(a, n, start=0):
    for i in range(n):
        a._stream.append(f"caption number {start + i} about the wooden chair and the light")
        a._stream_ts.append(time.time() - (n - i) * 8)
        a._consolidate_stream_if_needed()


cfg.STREAM_FADE_KEEP, cfg.STREAM_FOLD_OLDEST = 16, 5

a = Fake()
feed(a, 16)
check("under the keep, nothing folds", len(CALLS) == 0 and len(a._stream) == 16, (len(CALLS), len(a._stream)))

feed(a, 1, 16)
check("one past it folds once", len(CALLS) == 1, len(CALLS))
check("five oldest became one note", len(a._stream) == 13, len(a._stream))
check("the note is at the head", a._stream[0] == "A note in its own words about the older part.", a._stream[0])
check("the newest is untouched, word for word", a._stream[-1].endswith("about the wooden chair and the light"), a._stream[-1])
check("the note keeps the oldest entry's time", len(a._stream_ts) == len(a._stream))
check("it folded the machine's own words, not new ones", "Compress them into ONE short sentence" in CALLS[0]["prompt"] and "reusing their own words" in CALLS[0]["prompt"])

before = len(CALLS)
feed(a, 12, 20)
check("it keeps folding as the stream moves", len(CALLS) > before, (before, len(CALLS)))
check("a fold every four captions or so, not every one", 2 <= (len(CALLS) - before) <= 4, len(CALLS) - before)
check("the head stays verbatim throughout", a._stream[-1].startswith("caption number"), a._stream[-1])
check("the window stays short", len(a._stream) <= 16, len(a._stream))

# the note is itself folded again: the far past fades further
notes = [e for e in a._stream if e.startswith("A note in its own words")]
check("only one note stands at a time; older notes are folded into it", len(notes) <= 1, notes)
folded_texts = [c["prompt"] for c in CALLS[1:]]
check("a later fold takes the earlier note as input", any("A note in its own words" in p for p in folded_texts))

# switched off, the old length behaviour returns
CALLS.clear()
cfg.STREAM_FADE_KEEP = 0
b = Fake()
feed(b, 20)
check("with fading off, a short window never folds", len(CALLS) == 0 and len(b._stream) == 20, (len(CALLS), len(b._stream)))
cfg.STREAM_FADE_KEEP = 16

# a refusal leaves the stream alone
CALLS.clear()
_inf.query_model = _cap.query_model = lambda **kw: "no"  # too short to be admissible
c = Fake()
feed(c, 17)
check("a bad fold leaves the raw entries in place", len(c._stream) == 17, len(c._stream))

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
