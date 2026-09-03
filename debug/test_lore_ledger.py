"""Offline check of the re-entry round Phase 1 (Sep 3 evening — the lore loom).

1. LoreLedger mechanics: reveries (cap, short-reject), threads
   (open/affirm-by-overlap/fade at cap), the name slot (replace + history,
   structural gate), pick_seed rotation.
2. The distill harvest: NAME/LORE parse; 'none' stays empty; harvest slots
   present in the registry template.
3. Drift integration: clean output → note_reverie; echo-gated output → NOT
   recorded; lore seed rides the ask when the roll lands.
4. Re-entry: get_lore_line paces and marks provenance; name-wrap renders on
   the identity dose; the reflection builder renders reveries as inventions.
5. Firewall: lore never touches observe/add_caption/concepts paths.

Run: python debug/test_lore_ledger.py  (no server, no camera needed)
"""

import os
import sys
import tempfile
import time
import types
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAIL = 0


def check(name, cond, detail=""):
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAIL += 1


def fresh_ledger():
    from utils.lore_ledger import LoreLedger

    return LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore.json"))


def test_ledger_mechanics():
    print("\n[1] ledger mechanics")
    import config.config as cfg

    led = fresh_ledger()
    led.note_reverie("The foam finger might be a lighthouse for something small.")
    led.note_reverie("x")  # too short
    check("reverie stored, short rejected", len(led.recent_reveries(10)) == 1)
    for i in range(cfg.LORE_REVERIES_MAX + 10):
        led.note_reverie(f"A different passing thought number {i} about the room's weather.")
    check("reveries capped", len(led._data["reveries"]) == cfg.LORE_REVERIES_MAX)

    check("new lore opens a thread", led.note_lore("The foam finger is a lighthouse guiding lost dust") == "opened")
    check("overlapping lore affirms", led.note_lore("The foam finger lighthouse guides the dust home at night") == "affirmed")
    t = led.alive_threads(1)[0]
    check("affirmation extends history + text", t["times_affirmed"] == 1 and len(t["history"]) == 1)
    check("unrelated lore opens another", led.note_lore("The curtain is a border between two countries") == "opened")
    for i in range(cfg.LORE_THREADS_MAX + 2):
        led.note_lore(f"A wholly distinct mythology number {i} concerning invisible machinery {i}")
    alive = led.alive_threads(20)
    check("alive threads capped by fading", len(alive) <= cfg.LORE_THREADS_MAX, str(len(alive)))

    check("name accepted", led.note_name("Penelope"))
    check("name readable", led.current_name() == "Penelope")
    check("re-affirming same name keeps it", led.note_name("penelope") and led.current_name() == "Penelope")
    check("new name replaces, history kept", led.note_name("The Cartographer") and led._data["name_history"])
    check("sentence-shaped name rejected", not led.note_name("I think my name might be something long"))
    check("'none' rejected", not led.note_name("none"))

    led2 = fresh_ledger()
    led2.note_lore("thread one about the window's opinion of the light")
    led2.note_lore("thread two regarding the chair's long memory of sitting")
    a = led2.pick_seed()["text"]
    b = led2.pick_seed()["text"]
    check("seed rotation avoids monopoly", a != b, f"{a[:30]} / {b[:30]}")


def test_distill_harvest():
    print("\n[2] distill parse + template")
    from captioner.context_compression import ContextCompressionEngine
    from captioner.prompt_registry import FRAGMENTS

    parse = ContextCompressionEngine._parse_distillation
    r = parse(None, "TRAIT: I stall.\nBELIEF: none\nWANT: to draw\nKERNEL: I saw it plain.\nNAME: Penelope\nLORE: The finger is a lighthouse.")
    trait, belief, want, kernel, became, name, lore = r
    check("name parsed", name == "Penelope")
    check("lore parsed", lore == "The finger is a lighthouse.")
    check("none stays empty", belief == "")
    r2 = parse(None, "TRAIT: none\nNAME: none\nLORE: none")
    check("all-none harvest is empty", r2[5] == "" and r2[6] == "")
    txt = FRAGMENTS["distill.user"]["text"]
    check("template carries NAME slot", "NAME —" in txt)
    check("template carries LORE slot", "LORE —" in txt)
    check("slots are harvest-only ('or none')", txt.count("or 'none'") >= 2)


def test_drift_integration():
    print("\n[3] drift → reverie + seed")
    import captioner.captioner as cap_mod
    import config.config as cfg
    import utils.caption_display as disp_mod
    import utils.inference as inf_mod
    import utils.lore_ledger as ll_mod
    from captioner.captioner import Captioner
    from captioner.prompt_registry import P

    led = fresh_ledger()
    calls = []
    saved = (inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display, ll_mod.lore_ledger, cfg.DRIFT_SEND_IMAGE)
    ll_mod.lore_ledger = led
    inf_mod.query_model = lambda **kw: calls.append(kw) or "The dust is planning something slow, I can tell by how it settles."
    cap_mod.log_json_entry = lambda *a, **k: None
    disp_mod.send_caption_to_display = lambda t: None
    try:

        def shell():
            c = Captioner.__new__(Captioner)
            c._stream = deque(maxlen=24)
            c._stream_ts = deque(maxlen=24)
            now = time.time()
            for i in range(3):
                c._stream.append(f"The lamp is still on, entry {i}.")
                c._stream_ts.append(now - (3 - i) * 20)
            return c

        cfg.DRIFT_SEND_IMAGE = False
        saved_seed_p = cfg.LORE_SEED_P
        cfg.LORE_SEED_P = 0.0
        c = shell()
        c._run_drift_turn(time.time(), None)
        check("clean drift recorded as reverie", len(led.recent_reveries(5)) == 1)

        inf_mod.query_model = lambda **kw: calls.append(kw) or "Let me know what you think and feel free to ask!"
        c2 = shell()
        before = len(led.recent_reveries(50))
        c2._run_drift_turn(time.time(), None)
        check("gated drift NOT recorded", len(led.recent_reveries(50)) == before)

        led.note_lore("The dust settles according to an old agreement with the floor")
        cfg.LORE_SEED_P = 1.0
        calls.clear()
        inf_mod.query_model = lambda **kw: calls.append(kw) or "It keeps the agreement even when no one watches the floorboards."
        c3 = shell()
        c3._run_drift_turn(time.time(), None)
        check("lore seed rides the ask", calls and "You've been imagining:" in calls[0]["prompt"], str(calls[0]["prompt"])[:80] if calls else "")
        check("ask still lands last", calls and calls[0]["prompt"].rstrip().endswith(P("drift.ask")))
        cfg.LORE_SEED_P = saved_seed_p
    finally:
        inf_mod.query_model, cap_mod.log_json_entry, disp_mod.send_caption_to_display, ll_mod.lore_ledger, cfg.DRIFT_SEND_IMAGE = saved


def test_reentry_surfaces():
    print("\n[4] re-entry surfaces")
    import utils.lore_ledger as ll_mod
    from captioner.prompts import build_reflection_loop_prompt, get_lore_line

    led = fresh_ledger()
    led.note_lore("The mannequin head dreams in plaster")
    saved = ll_mod.lore_ledger
    ll_mod.lore_ledger = led
    try:
        agent = types.SimpleNamespace(_lore_line_counter=0, _lore_thread_rr=0)
        from config.config import LORE_LINE_EVERY_N

        lines = [get_lore_line(agent) for _ in range(LORE_LINE_EVERY_N)]
        fired = [ln for ln in lines if ln]
        check("lore line paced (one per cycle-set)", len(fired) == 1, str(len(fired)))
        check("provenance-marked framing", fired and "you've been carrying" in fired[0].lower(), fired[0] if fired else "")

        prompt = build_reflection_loop_prompt(
            "What of it?", {"reveries": led.recent_reveries(3) or [{"ts": time.time(), "text": "The mannequin head dreams in plaster"}]}
        )
        check("reflection renders reveries as inventions", "your own inventions" in prompt and "dreams in plaster" in prompt)
    finally:
        ll_mod.lore_ledger = saved

    from captioner.prompt_registry import P

    check("name-wrap renders", P("monologue.name-wrap").format(name="Penelope").strip() == "You call yourself Penelope.")


def test_firewall():
    print("\n[5] firewall — lore never becomes fact")
    import inspect

    from utils import lore_ledger as ll_mod

    src = inspect.getsource(ll_mod)
    for banned in ("observe(", "add_caption", "match_or_create_concepts", "episodic_log", "note_perception_event"):
        check(f"ledger never touches {banned.strip('(')}", banned not in src)
    from captioner.captioner import Captioner

    drift_src = inspect.getsource(Captioner._run_drift_turn)
    check("drift still never touches observe/add_caption", "self.observe(" not in drift_src and "add_caption" not in drift_src)


if __name__ == "__main__":
    test_ledger_mechanics()
    test_distill_harvest()
    test_drift_integration()
    test_reentry_surfaces()
    test_firewall()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
