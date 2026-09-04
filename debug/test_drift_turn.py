"""Offline check of the drift turn (Sep 3 — interiority as population, not residue).

1. _drift_due gates: disabled / pre-first-caption / short stream / drawing all
   veto; the roll's probability tracks DRIFT_BASE_P * (1 + GAIN * boredom)
   (Monte Carlo vs formula at boredom 0 and 1).
2. Call-site guard: a salience-hot cycle never reaches the roll (source check —
   the guard is inline in _process_frame).
3. _run_drift_turn mechanics: image=None, the stream rides as history, DRIFT_TEMP,
   output pushed to the stream; a too-short response pushes nothing.
4. Firewall: the runner's source touches none of observe/add_caption/
   recent_captions/hour_log — invention can never become a fact.
5. The loneliness clocks are gone: no STORY_BEAT_* left in config.

Run: python debug/test_drift_turn.py  (no server, no camera needed)
"""

import inspect
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


def make_shell(boredom=0.0, stream_entries=3):
    """Captioner without __init__ (threads, camera) — just the drift state."""
    from captioner.captioner import Captioner

    c = Captioner.__new__(Captioner)
    c._stream = deque(maxlen=24)
    c._stream_ts = deque(maxlen=24)
    now = time.time()
    for i in range(stream_entries):
        c._stream.append(f"The lamp is still on, entry {i}.")
        c._stream_ts.append(now - (stream_entries - i) * 20)
    c.first_caption_done = True
    c._boredom = boredom
    c._is_currently_drawing = lambda: False
    return c


def test_drift_due_gates():
    print("\n[1] _drift_due gates + probability scaling")
    import config.config as cfg

    saved = {k: getattr(cfg, k) for k in ("DRIFT_ENABLED", "DRIFT_BASE_P", "DRIFT_BOREDOM_GAIN")}
    try:
        # certainty configuration: p > 1 so the roll always lands unless vetoed
        cfg.DRIFT_ENABLED, cfg.DRIFT_BASE_P, cfg.DRIFT_BOREDOM_GAIN = True, 1.1, 0.0

        check("fires when quiet, seeded, and enabled", make_shell()._drift_due())

        c = make_shell()
        c.first_caption_done = False
        check("vetoed before the first caption", not c._drift_due())

        c = make_shell(stream_entries=1)
        check("vetoed with fewer than 2 stream entries", not c._drift_due())

        c = make_shell()
        c._is_currently_drawing = lambda: True
        check("vetoed while the arm draws", not c._drift_due())

        cfg.DRIFT_ENABLED = False
        check("vetoed when disabled", not make_shell()._drift_due())
        cfg.DRIFT_ENABLED = True

        # Monte Carlo at the real defaults: p = base * (1 + gain * boredom)
        cfg.DRIFT_BASE_P, cfg.DRIFT_BOREDOM_GAIN = 0.05, 2.0
        n = 40000
        for boredom, expect in ((0.0, 0.05), (1.0, 0.15)):
            c = make_shell(boredom=boredom)
            rate = sum(c._drift_due() for _ in range(n)) / n
            check(
                f"empirical rate ~{expect:.2f} at boredom {boredom:.0f}",
                abs(rate - expect) < 0.007,
                f"measured {rate:.4f}",
            )
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)


def test_salience_guard():
    print("\n[2] call-site salience guard")
    src = open(os.path.join(os.path.dirname(__file__), "..", "captioner", "captioner.py")).read()
    check(
        "drift roll only reached on quiet cycles",
        "not self._salience_hot and self._drift_due()" in src,
    )


def test_run_drift_turn():
    print("\n[3] _run_drift_turn mechanics")
    import tempfile

    import captioner.captioner as cap_mod
    import config.config as cfg
    import utils.caption_display as disp_mod
    import utils.inference as inf_mod
    import utils.lore_ledger as ll_mod
    from captioner.prompt_registry import P
    from utils.lore_ledger import LoreLedger

    # ISOLATION (Sep 4): the drift turn now writes reveries and rolls the
    # lore seed — a test run must never touch the LIVE ledger (17 fake
    # reveries once leaked in) nor flake on the seed roll.
    saved_ledger, saved_seed_p = ll_mod.lore_ledger, cfg.LORE_SEED_P
    ll_mod.lore_ledger = LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore.json"))
    cfg.LORE_SEED_P = 0.0

    calls, logs, displayed = [], [], []

    def fake_query_model(**kwargs):
        calls.append(kwargs)
        return "The pen could be a mast. A ship of white paper, sailing off the table edge."

    saved = (inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display)
    inf_mod.query_model = fake_query_model
    cap_mod.log_json_entry = lambda *a, **k: logs.append((a, k))
    disp_mod.send_caption_to_display = lambda t: displayed.append(t)
    saved_flag = cfg.DRIFT_SEND_IMAGE
    try:
        cfg.DRIFT_SEND_IMAGE = True
        c = make_shell(stream_entries=3)
        before = len(c._stream)
        c._run_drift_turn(time.time(), "/tmp/fake_frame.jpg")

        check("exactly one model call", len(calls) == 1)
        if calls:
            kw = calls[0]
            check("eyes open — the frame rides along", kw.get("image") == "/tmp/fake_frame.jpg")
            check("stream rides as history", bool(kw.get("history")))
            check("hot slot temperature", kw.get("options", {}).get("temperature") == cfg.DRIFT_TEMP)
            check("asks the registry ask", kw.get("prompt") == P("drift.ask"))
            check("logged as drift_turn", kw.get("prompt_type") == "drift_turn")
        check("output entered the stream", len(c._stream) == before + 1)
        check("action logged", any(a[1].get("action") == "drift_turn" for a, k in logs))
        check("stored flag logged true", any(a[1].get("stored") is True for a, k in logs))
        check("last_caption_time stamped", getattr(c, "last_caption_time", 0) > 0)

        # the blind A/B arm: flag off drops the image even when a frame exists
        calls.clear()
        cfg.DRIFT_SEND_IMAGE = False
        c2 = make_shell(stream_entries=3)
        c2._run_drift_turn(time.time(), "/tmp/fake_frame.jpg")
        check("DRIFT_SEND_IMAGE=false drops the image", calls and calls[0].get("image") is None)
        cfg.DRIFT_SEND_IMAGE = True

        # a too-short response pushes nothing and stamps nothing new
        calls.clear()
        inf_mod.query_model = lambda **kw: "..."
        c3 = make_shell(stream_entries=3)
        c3._run_drift_turn(time.time(), None)
        check("short response pushes nothing", len(c3._stream) == 3)

        # the storage law (Sep 3 evening): a drift that recites a stream
        # refrain is spoken but never stored; assistant-register is skipped
        # the refrain entry must NOT be the newest one: the seam is excluded
        # from comparison in prefill modes (Aug 1 law — continuing the seam
        # is what continuation means; older entries are still fair game)
        refrain = "a faint pulse against his dark hoodie in the corner"
        c4 = make_shell(stream_entries=3)
        c4._stream.append(f"The light is the only thing moving, {refrain}.")
        c4._stream_ts.append(time.time())
        c4._stream.append("The chair has not moved since he sat down.")
        c4._stream_ts.append(time.time())
        inf_mod.query_model = lambda **kw: f"Still here. The screen glows on, {refrain}. Nothing asks me to move."
        before4 = len(c4._stream)
        logs.clear()
        c4._run_drift_turn(time.time(), None)
        check("refrain drift spoken not stored", len(c4._stream) == before4 and any(a[1].get("reason") == "refrain_echo" for a, k in logs))

        inf_mod.query_model = lambda **kw: "Let me know what you think of this scene and feel free to ask!"
        c5 = make_shell(stream_entries=3)
        displayed.clear()
        c5._run_drift_turn(time.time(), None)
        check("shape-class drift skipped entirely", len(c5._stream) == 3 and not displayed)
    finally:
        cfg.DRIFT_SEND_IMAGE = saved_flag
        inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display = saved
        ll_mod.lore_ledger, cfg.LORE_SEED_P = saved_ledger, saved_seed_p


def test_firewall():
    print("\n[4] firewall — drift never reaches a fact ledger")
    from captioner.captioner import Captioner

    src = inspect.getsource(Captioner._run_drift_turn)
    for banned in ("self.observe(", "add_caption", "recent_captions", "hour_log", "observe_caption"):
        check(f"runner never touches {banned.strip('self.(')}", banned not in src)


def test_clocks_gone():
    print("\n[5] the loneliness clocks are deleted")
    import config.config as cfg

    for gone in ("STORY_BEAT_ENABLED", "STORY_BEAT_AFTER_S", "STORY_BEAT_MIN_GAP_S"):
        check(f"{gone} gone from config", not hasattr(cfg, gone))
    for knob in ("DRIFT_ENABLED", "DRIFT_BASE_P", "DRIFT_BOREDOM_GAIN", "DRIFT_TEMP", "DRIFT_NUM_PREDICT"):
        check(f"{knob} present", hasattr(cfg, knob))


def test_registry_panel_path():
    print("\n[6] registry + panel edit path")
    import string as _string

    from captioner.prompt_registry import FRAGMENTS, PASSES, validate_override

    def ok(fid, text):
        try:
            validate_override(fid, text)
            return True
        except (KeyError, ValueError):
            return False

    check("drift.stream-frame accepts a marker edit", ok("drift.stream-frame", "~ {text}"))
    check("caption.unchanged editable keeping {duration}", ok("caption.unchanged", "Still nothing, {duration} now."))
    check("bogus placeholder rejected", not ok("drift.ask", "where does it {bogus}?"))
    check("unknown fragment rejected", not ok("story.ask", "gone"))

    p = PASSES.get("drift_turn", {})
    frags = [b["frag"] for b in p.get("system", []) + p.get("user", []) if "frag" in b]
    check("drift_turn pass declared + migrated", p.get("migrated") is True)
    check("pass references only real fragments", frags and all(f in FRAGMENTS for f in frags), str(frags))

    # consistency sweep: any fragment whose text has {fields} must declare them
    undeclared = []
    for fid, frag in FRAGMENTS.items():
        fields = {f for _l, f, _s, _c in _string.Formatter().parse(frag["text"]) if f}
        if fields - set(frag.get("placeholders", [])):
            undeclared.append(fid)
    check("no fragment with undeclared placeholders", not undeclared, str(undeclared))


if __name__ == "__main__":
    test_drift_due_gates()
    test_salience_guard()
    test_run_drift_turn()
    test_firewall()
    test_clocks_gone()
    test_registry_panel_path()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
