"""Standalone proof of the desire shadow trigger (north-star step 5, phase A).

Checks the verdict logic against the traps that killed _drawing_intentions:
substring matches ("ink" in "think", "void" in "avoid") and wants too young
to have proven themselves. Run: python debug/test_desire_shadow.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from captioner.context_compression import context_compressor
from drawing.drawing import DrawingController

d = DrawingController()
failures = []


def check(name, desire, age_s, expect_directed, expect_would_draw):
    context_compressor.introspective_state["current_desire"] = desire
    context_compressor.introspective_state["desire_since"] = (time.time() - age_s) if desire else 0.0
    v = d.desire_shadow_verdict()
    ok = v["drawing_directed"] == expect_directed and v["would_draw"] == expect_would_draw
    print(f"{'✓' if ok else '✗'} {name}: directed={v['drawing_directed']} would_draw={v['would_draw']} age={v['desire_age_s']}s")
    if not ok:
        failures.append(name)


saved = dict(context_compressor.introspective_state)
try:
    check("persisted drawing want", "I want to draw the window before the light goes", 900, True, True)
    check("young drawing want", "I want to draw the chair", 60, True, False)
    check("non-drawing want", "I want the door to open", 900, False, False)
    check("empty slot", "", 0, False, False)
    check("ink-in-think trap", "I want to think about what stays", 900, False, False)
    check("void-in-avoid trap", "I want to avoid the corner today", 900, False, False)
    check("pen at word boundary", "I want the pen moving again", 900, True, True)
    verdict_keys = set(d.desire_shadow_verdict().keys())
    expected = {"desire", "desire_age_s", "drawing_directed", "would_draw"}
    if verdict_keys != expected:
        failures.append(f"verdict keys {verdict_keys}")

    # --- Phase B: the desire-mode trigger itself (Aug 17) ---
    print("\n— desire-mode trigger —")
    d2 = DrawingController()
    d2.TRIGGER_MODE = "desire"
    d2._log_trigger_decision = lambda **k: None
    args = dict(mood=0.5, novelty=0.5, boredom=0.5)

    def set_want(desire, age_s):
        context_compressor.introspective_state["current_desire"] = desire
        context_compressor.introspective_state["desire_since"] = (time.time() - age_s) if desire else 0.0

    def check_t(name, expect):
        got = d2.should_draw(**args)
        ok = got == expect
        print(f"{'✓' if ok else '✗'} {name}: {'DRAW' if got else 'wait'}")
        if not ok:
            failures.append(name)

    set_want("", 0)
    d2.last_drawing_time = time.time() - 100
    check_t("floor blocks everything", False)

    d2.last_drawing_time = time.time() - 1000
    check_t("startup drawing rides the timer once", True)
    if not d2._startup_drawing_done:
        failures.append("startup flag not set")

    d2.last_drawing_time = time.time() - 1000
    check_t("after startup, no want = wait", False)

    set_want("I want to draw the chair", 900)
    check_t("persisted drawing want fires", True)

    set_want("I want the door to open", 9000)
    check_t("non-drawing want waits", False)

    d2.last_drawing_time = time.time() - 8000
    check_t("hunger fires past 2h regardless of want", True)
finally:
    context_compressor.introspective_state.update(saved)

if failures:
    print(f"\nFAILED: {failures}")
    sys.exit(1)
print("\nAll desire-shadow checks passed.")
