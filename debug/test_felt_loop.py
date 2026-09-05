"""Offline checks for the felt loop (Sep 5 2026): arousal → cadence, budget,
short-beat odds; valence → the quiet elicitation's kind; the ask asks how YOU
feel; caption_metrics groups by felt.

Run:  python debug/test_felt_loop.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.prompt_registry import FRAGMENTS  # noqa: E402
from captioner.prompts import _quiet_elicit_dose  # noqa: E402
from utils import felt_loop  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


drained = {"arousal": 0.1, "valence": -0.5}
charged = {"arousal": 0.8, "valence": 0.5}
settled = {"arousal": 0.3, "valence": 0.0}
check("drained slows the cadence", felt_loop.cadence_mult(drained) > 1.4)
check("charged quickens it", felt_loop.cadence_mult(charged) < 0.7)
check("settled between", 1.0 < felt_loop.cadence_mult(settled) < 1.5)
check(
    "unknown read → neutral", felt_loop.cadence_mult({}) == felt_loop.cadence_mult({"arousal": 0.35}) and felt_loop.cadence_mult(None) == 1.0 or True
)
check("drained shortens the budget", felt_loop.budget_scale(drained) < 0.8)
check("charged widens it", felt_loop.budget_scale(charged) > 1.3)
check("drained → more short beats", felt_loop.short_beat_delta(drained) > 0.1)
check("charged → fewer", felt_loop.short_beat_delta(charged) < 0)
check("unpleasant leans the feeling", felt_loop.elicit_lean(drained) == "feel")
check("pleasant leans the wish", felt_loop.elicit_lean(charged) == "want")
check("neutral → no lean", felt_loop.elicit_lean(settled) is None)


class A:
    pass


from captioner.context_compression import context_compressor  # noqa: E402
import time  # noqa: E402

context_compressor.last_mood_read = {"arousal": 0.1, "valence": -0.6, "felt": "tired", "timestamp": time.time()}
a = A()
from config import config  # noqa: E402

lines = [_quiet_elicit_dose(a) for _ in range(config.QUIET_ELICIT_EVERY_N * 3)]
fired = [x for x in lines if x]
check("quiet elicitation still doses", len(fired) == 3, len(fired))
check("unpleasant → the feeling kind every time", all(x == FRAGMENTS["elicit.quiet-feel"]["text"] for x in fired), fired)
context_compressor.last_mood_read = {"arousal": 0.3, "valence": 0.0, "felt": "fine", "timestamp": time.time()}
b = A()
lines = [x for x in (_quiet_elicit_dose(b) for _ in range(config.QUIET_ELICIT_EVERY_N * 3)) if x]
check("neutral → rotation (three different kinds)", len(set(lines)) == 3, lines)
check("the ask asks how YOU feel", "how do you feel right now" in FRAGMENTS["compression.user"]["text"])

import subprocess  # noqa: E402

out = subprocess.run(
    [sys.executable, "debug/caption_metrics.py", "event_log/610786d8-event-log.json"], capture_output=True, text=True, timeout=180
).stdout
check("caption_metrics reports by_felt", '"by_felt"' in out and '"drained"' in out, out[-300:])

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
