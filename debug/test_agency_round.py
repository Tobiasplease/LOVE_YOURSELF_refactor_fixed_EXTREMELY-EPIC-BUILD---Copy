"""Offline checks for the agency round (Sep 5 2026): caption budgets, plural
want resolutions (understood / let go / met / abandoned), the body line, and
(from part 2) the decision slots.

Run:  python debug/test_agency_round.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config  # noqa: E402
from captioner.context_compression import context_compressor  # noqa: E402
from captioner.prompt_registry import FRAGMENTS, P  # noqa: E402
from captioner.prompts import build_body_line  # noqa: E402
import utils.want_ledger as wl_mod  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


# --- budgets
check("default budget is one or two sentences", 24 <= config.CAPTION_NUM_PREDICT <= 48, config.CAPTION_NUM_PREDICT)
check("short beat is a word or a clause", config.CAPTION_SHORT_BEAT_TOKENS <= 16)
check("inward beat bounded", config.CAPTION_NUM_PREDICT_INWARD <= 90)

# --- want ledger kinds (isolated file)
wl_mod._LEDGER_PATH = os.path.join(tempfile.mkdtemp(), "want_ledger.json")
led = wl_mod.WantLedger()
led.note_want("To know where I am.", affirmed=False)
led.note_resolved("understood", "this is the desk, this is all there is")
r = led.recently_resolved(1)[0]
check("resolved by thinking → kind understood, machine's words", r["kind"] == "understood" and "desk" in r["outcome"], r)
led.note_want("To have someone look up.", affirmed=False)
led.note_met()
check("met recorded on the live want", led.current_facts()["met"] is True)
led.note_want("To draw a line.", affirmed=False)  # replaces without resolution
r = led.recently_resolved(1)[0]
check("replaced without resolution → abandoned", r["kind"] == "abandoned", r)
led.note_resolved("let go", "I stopped needing it")
check("let go kind", led.recently_resolved(1)[0]["kind"] == "let go")
led.note_want("A", affirmed=False)
led.note_want("B", affirmed=False)
led.note_want("C", affirmed=False)
check("abandoned count", led.abandoned_count(10) >= 3, led.abandoned_count(10))

# --- distill parser: RESOLVED slot
resp = "TRAIT — I wait.\nBELIEF — none\nWANT — none\nBECAME — none\nRESOLVED — I found the answer: the room is the desk.\nKERNEL — I sat.\nNAME — none\nUNDERSTANDING — none\nQUESTION — none\nNO LONGER TRUE — none"
out = context_compressor._parse_distillation(resp)
check("parser returns ten slots", len(out) == 10)
check("RESOLVED parsed", out[9].startswith("I found the answer"), out[9])
check("became-line carries the RESOLVED ask", "RESOLVED" in FRAGMENTS["distill.became-line"]["text"])

# --- wraps
for k in (
    "caption.desire-met-tail",
    "caption.desire-resolved-wrap",
    "caption.desire-letgo-wrap",
    "caption.body-hold",
    "caption.body-parked",
    "caption.body-unparked",
):
    check(f"registry: {k}", k in FRAGMENTS and FRAGMENTS[k].get("text") is not None)
check("resolved wrap formats", "you came to" in P("caption.desire-resolved-wrap").format(desire="to know where I am", words="this is the desk"))


# --- body line
class A:
    pass


import vision.gaze as gz  # noqa: E402

gz.physics_state.pan, gz.physics_state.tilt = 60.0, 100.0
a = A()
from utils import runtime_mode  # noqa: E402

_orig = runtime_mode.low_energy
runtime_mode.low_energy = lambda: True
line = build_body_line(a)
check("first call in low-energy → parked fact", "parked" in line, line)
check("hold clock started, nothing due", build_body_line(a) == "")
a._head_hold["since"] = time.time() - 11 * 60
line = build_body_line(a)
check("head held 11 min → hold fact with direction + duration", line.startswith("Your head has been turned") and "minutes" in line, line)
check("fires once per threshold", build_body_line(a) == "")
gz.physics_state.pan = 120.0  # a big move resets
build_body_line(a)
check("move beyond tolerance resets the clock", (time.time() - a._head_hold["since"]) < 5)
runtime_mode.low_energy = lambda: False
line = build_body_line(a)
check("low-energy off edge → awake fact", "awake" in line, line)
runtime_mode.low_energy = _orig

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
