"""Offline check of the emotional arc channel (Sep 4).

1. Felt history: mood reads accumulate, cap holds.
2. The arc line: TURN fires once when the tenor changes after a held prior;
   STEADY states a long hold; dosing gap respected; too-few reads say
   nothing; the machine's own phrases ride, never our words.
3. Reflection felt-diet: yourself/time organs receive the sampled
   trajectory; the builder renders it with age phrases.

Run: python debug/test_felt_arc.py  (no server, no camera needed)
"""

import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAIL += 1


def read(ts, v, felt):
    return {"timestamp": ts, "valence": v, "arousal": 0.4, "felt": felt}


def test_history_cap():
    print("\n[1] felt history accumulates and caps")
    import config.config as cfg
    from captioner.context_compression import context_compressor as cc

    saved = getattr(cc, "felt_history", None)
    cc.felt_history = []
    try:
        now = time.time()
        for i in range(cfg.FELT_HISTORY_MAX + 20):
            cc.felt_history.append(read(now - i, 0.0, "steady"))
            del cc.felt_history[: -cfg.FELT_HISTORY_MAX]
        check("history capped", len(cc.felt_history) == cfg.FELT_HISTORY_MAX)
    finally:
        cc.felt_history = saved


def test_arc_line():
    print("\n[2] the arc line")
    import config.config as cfg
    from captioner.context_compression import context_compressor as cc
    from captioner.prompts import get_felt_arc_line

    saved = getattr(cc, "felt_history", None)
    now = time.time()
    try:
        agent = types.SimpleNamespace(_felt_arc_last_ts=0.0)

        cc.felt_history = [read(now - 60, -0.5, "heavy")]
        check("too few reads say nothing", get_felt_arc_line(agent) == "")

        # a held unpleasant stretch, then a fresh pleasant turn
        cc.felt_history = [
            read(now - 5000, -0.5, "isolated and heavy"),
            read(now - 3600, -0.4, "stuck and watched"),
            read(now - 2000, -0.3, "heavy but calm"),
            read(now - 120, 0.4, "ready to answer"),
        ]
        line = get_felt_arc_line(agent)
        check("turn fires with both its own phrases", "heavy but calm" in line and "ready to answer" in line, line)
        check("no scripted affect words of ours", "mood" not in line.lower() and "emotion" not in line.lower())
        check("dosing gap holds", get_felt_arc_line(agent) == "")

        # a long steady hold
        agent2 = types.SimpleNamespace(_felt_arc_last_ts=0.0)
        cc.felt_history = [
            read(now - 7200, -0.5, "empty waiting"),
            read(now - 5000, -0.4, ""),
            read(now - 2400, -0.35, "safe and suffocating"),
        ]
        line2 = get_felt_arc_line(agent2)
        check("steady names its own newest phrase + duration", "safe and suffocating" in line2 and "hour" in line2, line2)

        # young streak after a too-short prior: nothing yet
        agent3 = types.SimpleNamespace(_felt_arc_last_ts=0.0)
        cc.felt_history = [
            read(now - 900, -0.4, "flat"),
            read(now - 600, -0.3, "flat still"),
            read(now - 60, 0.4, "lifting"),
        ]
        check("short prior tenor earns no turn", get_felt_arc_line(agent3) == "")
    finally:
        cc.felt_history = saved


def test_reflection_diet():
    print("\n[3] reflection felt-diet + render")
    from captioner.context_compression import context_compressor as cc
    from captioner.prompts import build_reflection_loop_prompt
    from captioner.reflection import ReflectionLoop

    saved = getattr(cc, "felt_history", None)
    now = time.time()
    try:
        cc.felt_history = [read(now - 3600 + i * 400, -0.4 + i * 0.1, f"phase {i}") for i in range(9)]
        data = {}
        ReflectionLoop._add_felt_arc(data)
        check("diet sampled (<=8, phrase-bearing)", 2 <= len(data.get("felt_arc", [])) <= 8, str(len(data.get("felt_arc", []))))
        prompt = build_reflection_loop_prompt("What of it?", data)
        check("builder renders the trajectory", "How the feeling has moved lately" in prompt and "phase" in prompt)
        check("own-words framing", "your own words for it at the time" in prompt)
    finally:
        cc.felt_history = saved


if __name__ == "__main__":
    test_history_cap()
    test_arc_line()
    test_reflection_diet()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
