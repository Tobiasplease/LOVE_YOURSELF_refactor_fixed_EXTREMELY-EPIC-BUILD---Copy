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
    import captioner.captioner as cap_mod
    import config.config as cfg
    import utils.caption_display as disp_mod
    import utils.inference as inf_mod
    from captioner.prompt_registry import P

    calls, logs, displayed = [], [], []

    def fake_query_model(**kwargs):
        calls.append(kwargs)
        return "The pen could be a mast. A ship of white paper, sailing off the table edge."

    saved = (inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display)
    inf_mod.query_model = fake_query_model
    cap_mod.log_json_entry = lambda *a, **k: logs.append((a, k))
    disp_mod.send_caption_to_display = lambda t: displayed.append(t)
    try:
        c = make_shell(stream_entries=3)
        before = len(c._stream)
        c._run_drift_turn(time.time())

        check("exactly one model call", len(calls) == 1)
        if calls:
            kw = calls[0]
            check("no image — the drift doesn't look", kw.get("image") is None)
            check("stream rides as history", bool(kw.get("history")))
            check("hot slot temperature", kw.get("options", {}).get("temperature") == cfg.DRIFT_TEMP)
            check("asks the registry ask", kw.get("prompt") == P("drift.ask"))
            check("logged as drift_turn", kw.get("prompt_type") == "drift_turn")
        check("output entered the stream", len(c._stream) == before + 1)
        check("action logged", any(a[1].get("action") == "drift_turn" for a, k in logs))
        check("stored flag logged true", any(a[1].get("stored") is True for a, k in logs))
        check("last_caption_time stamped", getattr(c, "last_caption_time", 0) > 0)

        # a too-short response pushes nothing and stamps nothing new
        calls.clear()
        inf_mod.query_model = lambda **kw: "..."
        c2 = make_shell(stream_entries=3)
        c2._run_drift_turn(time.time())
        check("short response pushes nothing", len(c2._stream) == 3)
    finally:
        inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display = saved


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


if __name__ == "__main__":
    test_drift_due_gates()
    test_salience_guard()
    test_run_drift_turn()
    test_firewall()
    test_clocks_gone()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
