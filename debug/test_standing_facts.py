#!/usr/bin/env python3
"""Sep 11 — standing facts (artist: "the appropriate data should reach every
single call"): the stillness, head and felt lines ride on every call with
their durations in words, not once at an edge. The felt line needs the live
compressor's history and is not exercised here. Importing the prompts module
mints no run log; the MOOD_SNAPSHOT_FOLDER line is belt-and-braces."""
import os
import sys
import tempfile
import time

os.environ["MOOD_SNAPSHOT_FOLDER"] = tempfile.mkdtemp(prefix="standing-")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.episodic_log as _el  # noqa: E402
from captioner.prompts import _head_line_from, build_standing_facts, casual_time_string, get_unchanged_line  # noqa: E402

_el.episodic_log.get_last_event = lambda etype: None  # no real ledger events in the test

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


class A:
    pass


def agent(still_s):
    a = A()
    a.true_session_start = time.time() - still_s
    return a


# --- words, never integers
check("6 min", casual_time_string(6) == "about six minutes", casual_time_string(6))
check("20 min", casual_time_string(20) == "about twenty minutes", casual_time_string(20))
check("45 min", casual_time_string(45) == "about three quarters of an hour", casual_time_string(45))
check("2 h", casual_time_string(120) == "about two hours", casual_time_string(120))
check("7 h", casual_time_string(7 * 60) == "about seven hours", casual_time_string(7 * 60))
check("3 days", casual_time_string(3 * 1440) == "about three days", casual_time_string(3 * 1440))
digits = [m for m in range(0, 3000) if any(ch.isdigit() for ch in casual_time_string(m))]
check("no digit anywhere, 0..3000 min", not digits, digits[:5])

# --- stillness: standing
check("90 s → nothing yet", get_unchanged_line(agent(90)) == "", get_unchanged_line(agent(90)))
check("3 min → standing", get_unchanged_line(agent(180)) == "Nothing has happened for a few minutes.", get_unchanged_line(agent(180)))
l1, l2 = get_unchanged_line(agent(2 * 3600)), get_unchanged_line(agent(2 * 3600))
check("2 h → standing, and again on the next call", l1 == l2 == "Nothing has happened for about two hours.", (l1, l2))

# --- the head: verdict right after a turn, posture once held
now = time.time()
a = A()
check("turn + verdict unchanged", _head_line_from(a, 90, 100, "looking left", "unchanged", now) == "You've just turned left; the view here is as it was when you last looked.")
check("verdict new", _head_line_from(A(), 90, 100, "looking down-left", "baselined", now) == "You've just turned down-left; you hadn't looked this way before.")
check("verdict changed", "has changed since you last looked" in _head_line_from(A(), 90, 100, "looking straight ahead", "changed", now))
b = A()
b._expect_checked_at = now
check("expectation check spoke this call → yields", _head_line_from(b, 90, 100, "looking up", "unchanged", now) == "")
check("no verdict → nothing right after a turn", _head_line_from(A(), 90, 100, "looking left", "off_center", now) == "")
check("past the verdict window, before standing → nothing", _head_line_from(a, 90, 100, "looking left", "unchanged", now + 60) == "")
check("held 3 min → standing, in words", _head_line_from(a, 91, 101, "looking left", "unchanged", now + 180) == "You've been looking left for a few minutes.")
check("held 2 h → standing", _head_line_from(a, 90, 100, "looking left", "unchanged", now + 7200) == "You've been looking left for about two hours.")
check("a big move resets the clock", _head_line_from(a, 140, 100, "looking right", "baselined", now + 7300) == "You've just turned right; you hadn't looked this way before.")

# --- the block
f = build_standing_facts(agent(3600), include_felt=False)
check("facts block carries the standing stillness line", "Nothing has happened for about an hour." in f, f)

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
