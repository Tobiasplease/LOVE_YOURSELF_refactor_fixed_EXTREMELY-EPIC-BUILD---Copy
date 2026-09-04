"""Offline check of the attention round (Sep 4).

1. Investigate glances: familiar strangers (high hits, low conf) get picked;
   settled entries don't; per-term cooldown holds; kind flows through.
2. Open questions: harvested to the ledger (dedupe by overlap, cap+fade),
   re-entry line paced, least-recently-surfaced rotation.
3. Drift presence fact: rides only when belief active + person not in frame.
4. Close look accepts investigate glances (source check).
5. Presence look tolerance tightened inside the frame.

Run: python debug/test_attention_round.py  (no server, no camera needed)
"""

import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAIL += 1


def test_investigate():
    print("\n[1] investigate glances")
    import config.config as cfg
    from perception.spatial_registry import SpatialRegistry

    reg = SpatialRegistry(state_path=os.path.join(tempfile.mkdtemp(), "reg.json"))
    now = time.time()
    reg.entries = {
        "wall lamp": {"pan": 60, "tilt": 90, "conf": 0.20, "hits": 783000, "first_seen": now - 9e5, "last_seen": now},
        "wooden chair": {"pan": 100, "tilt": 95, "conf": 0.65, "hits": 86000, "first_seen": now - 9e5, "last_seen": now},
    }
    saved = cfg.INVESTIGATE_WEIGHT
    cfg.INVESTIGATE_WEIGHT = 1.0
    try:
        picks = [reg.pick_glance_target(explore_weight=0.0) for _ in range(6)]
        kinds = {p["kind"] for p in picks}
        inv = [p for p in picks if p["kind"] == "investigate"]
        check("investigate fires", bool(inv), str(kinds))
        check("it targets the familiar stranger", all(p["term"] == "wall lamp" for p in inv))
        check("cooldown: second pick within 15min is not investigate", sum(1 for p in picks if p["kind"] == "investigate") == 1, str(kinds))

        reg2 = SpatialRegistry(state_path=os.path.join(tempfile.mkdtemp(), "reg.json"))
        reg2.entries = {"wooden chair": {"pan": 100, "tilt": 95, "conf": 0.65, "hits": 86000, "first_seen": now - 9e5, "last_seen": now}}
        p = reg2.pick_glance_target(explore_weight=0.0)
        check("no strangers -> plain revisit", p["kind"] == "revisit")
    finally:
        cfg.INVESTIGATE_WEIGHT = saved


def test_questions():
    print("\n[2] open questions")
    import config.config as cfg
    import utils.lore_ledger as ll_mod
    from captioner.prompts import get_question_line
    from utils.lore_ledger import LoreLedger

    led = LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore.json"))
    check("question stored", led.note_question("What does he build all day?"))
    check("statement rejected (no question mark)", not led.note_question("He builds things all day"))
    check("near-duplicate refreshes, not duplicates", led.note_question("What does he build all day, really?") and len(led.open_questions(20)) == 1)
    for i in range(cfg.QUESTIONS_MAX + 2):
        led.note_question(f"Question number {i} about the {i}th unrelated mystery thing?")
    check("cap fades the oldest", len(led.open_questions(20)) <= cfg.QUESTIONS_MAX)

    saved = ll_mod.lore_ledger
    ll_mod.lore_ledger = led
    try:
        agent = types.SimpleNamespace(_question_line_counter=0)
        lines = [get_question_line(agent) for _ in range(cfg.QUESTION_LINE_EVERY_N)]
        fired = [ln for ln in lines if ln]
        check("question line paced", len(fired) == 1, str(len(fired)))
        check("carries the machine's own question", fired and "?" in fired[0], fired[0] if fired else "")
        a = led.pick_question()["text"]
        b = led.pick_question()["text"]
        check("rotation avoids monopoly", a != b)
    finally:
        ll_mod.lore_ledger = saved


def test_drift_presence():
    print("\n[3] drift presence fact")
    from collections import deque

    import captioner.captioner as cap_mod
    import captioner.frame_buffer as fb_mod
    import config.config as cfg
    import utils.caption_display as disp_mod
    import utils.inference as inf_mod
    import utils.lore_ledger as ll_mod
    from captioner.captioner import Captioner
    from utils.lore_ledger import LoreLedger

    calls = []
    saved = (inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display, ll_mod.lore_ledger, cfg.LORE_SEED_P, cfg.DRIFT_SEND_IMAGE)
    inf_mod.query_model = lambda **kw: calls.append(kw) or "Something about the far side of the curtain keeps not resolving."
    cap_mod.log_json_entry = lambda *a, **k: None
    disp_mod.send_caption_to_display = lambda t: None
    ll_mod.lore_ledger = LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore.json"))
    cfg.LORE_SEED_P = 0.0
    cfg.DRIFT_SEND_IMAGE = False

    class _FB:
        def __init__(self, person):
            self._p = person

        def get_recent_with_metadata(self, **kw):
            return [{"detection": {"person": self._p}}]

    fb_saved = fb_mod.frame_buffer
    try:

        def shell(believed, person_in_frame):
            fb_mod.frame_buffer = _FB(person_in_frame)
            c = Captioner.__new__(Captioner)
            c._stream = deque(maxlen=24)
            c._stream_ts = deque(maxlen=24)
            now = time.time()
            for i in range(3):
                c._stream.append(f"The lamp is still on, entry {i}.")
                c._stream_ts.append(now - (3 - i) * 20)
            c._presence_believed = believed
            return c

        calls.clear()
        shell(True, False)._run_drift_turn(time.time(), None)
        check("belief + empty frame -> fact rides", "out of view right now" in calls[0]["prompt"], calls[0]["prompt"][:80])

        calls.clear()
        shell(True, True)._run_drift_turn(time.time(), None)
        check("belief + person visible -> no fact", "out of view" not in calls[0]["prompt"])

        calls.clear()
        shell(False, False)._run_drift_turn(time.time(), None)
        check("no belief -> no fact", "out of view" not in calls[0]["prompt"])
    finally:
        fb_mod.frame_buffer = fb_saved
        inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display, ll_mod.lore_ledger, cfg.LORE_SEED_P, cfg.DRIFT_SEND_IMAGE = (
            saved
        )


def test_wiring():
    print("\n[4] wiring checks")
    import inspect

    import config.config as cfg
    from captioner.captioner import Captioner

    src = inspect.getsource(Captioner._maybe_close_look)
    check("close look accepts investigate", '"investigate"' in src)
    check("look tolerance inside the frame", cfg.PRESENCE_ABSENCE_LOOK_TOLERANCE <= 20.0, str(cfg.PRESENCE_ABSENCE_LOOK_TOLERANCE))
    import vision.gaze as gaze

    check("glance choices are logged", "glance_start" in inspect.getsource(gaze._update_registry_glance))


if __name__ == "__main__":
    test_investigate()
    test_questions()
    test_drift_presence()
    test_wiring()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
