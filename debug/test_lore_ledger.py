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


def test_thread_returns():
    """Sep 13 — the write-back. The Sep 12 night: lore threads returned to, 0 of
    247. Nothing recorded that a thread had been taken up, so the ledger could
    neither compound a story nor drop a dead one."""
    print("\n[1b] thread returns, pruning, revival")
    import tempfile

    from utils.lore_ledger import LoreLedger

    led = LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore.json"))
    led.note_lore("The red foam finger is a lighthouse for something that never arrives.")
    led.note_lore("The wooden chair keeps the shape of whoever sat in it.")
    t_finger = next(t for t in led.alive_threads(6) if "lighthouse" in t["text"])

    check("a fresh thread has no returns", not t_finger.get("returns"))
    check("stats see them offered nowhere yet", led.thread_stats()["returned"] == 0, led.thread_stats())

    check("a return is recorded", led.note_return(t_finger, "drift", "Maybe it signals the room, not me."))
    t2 = next(t for t in led.alive_threads(6) if "lighthouse" in t["text"])
    check("with its source and where it got to", t2["returns"][-1]["source"] == "drift" and "signals the room" in t2["returns"][-1]["advance"], t2["returns"])
    check("stats count it", led.thread_stats()["returned"] == 1, led.thread_stats())

    # wandering back in on its own, by content words alone
    hit = led.note_return_by_overlap("The chair still keeps that shape, whoever sat there.", source="caption")
    check("a caption that wanders into a thread returns to it", hit and "wooden chair" in hit["text"], hit)
    check("an unrelated caption returns to nothing", led.note_return_by_overlap("A pink rack of cables.", source="caption") is None)
    check("one content word is not a return", led.note_return_by_overlap("chair.", source="caption") is None)

    # pruning: offered and never returned to
    led2 = LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore2.json"))
    led2.note_lore("A thread nobody ever comes back to, about the ceiling tiles.")
    for _ in range(3):
        led2.pick_seed()
    led2.pick_seed()
    check("offered three times with no return → dormant", led2.thread_stats()["dormant"] == 1, led2.thread_stats())
    check("and no longer offered", led2.pick_seed() is None, led2.pick_seed())
    woke = led2.note_return_by_overlap("Those ceiling tiles again, the thread of them.", source="caption")
    check("the machine wandering back into it wakes it", woke is not None and led2.thread_stats()["dormant"] == 0, led2.thread_stats())
    check("and it is offered again", led2.pick_seed() is not None)

    # a returned-to thread is preferred as a seed
    led3 = LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore3.json"))
    led3.note_lore("Thread A, about the black curtain and what is behind it.")
    led3.note_lore("Thread B, about the pen and the paper it never reaches.")
    a = next(t for t in led3.alive_threads(6) if "curtain" in t["text"])
    led3.note_return(a, "reflection", "Behind it is just the wall, and that is worse.")
    picks = {led3.pick_seed()["text"][:8] for _ in range(3)}
    check("the living thread is the one offered", picks == {"Thread A"}, picks)


def test_drift_ask():
    """Sep 13 — the drift opens a thread by asking. Reviewed plan: "a live thread
    must be present by QUESTION, not by statement, or it is the next refrain."
    The old line claimed "You've been coming back to this" about threads nothing
    had ever come back to."""
    print("\n[1c] the thread-anchored drift ask")
    import tempfile

    from captioner.prompts import drift_seed_ask
    from utils import lore_ledger as _mod
    from utils.lore_ledger import LoreLedger

    led = LoreLedger(state_path=os.path.join(tempfile.mkdtemp(), "lore4.json"))
    real = _mod.lore_ledger
    _mod.lore_ledger = led
    try:
        led.note_lore("The red foam finger points at a hole in the ceiling tile.")
        seed = led.pick_seed()

        ask = drift_seed_ask(seed)
        check("a bare thread is opened with a question", ask.endswith("Where does it go from here?") and "red foam finger" in ask, ask)
        check("it no longer claims the machine kept coming back", "coming back to this" not in ask, ask)

        led.note_return(seed, "drift", "The hole was there before the finger was.")
        seed2 = next(t for t in led.alive_threads(6) if "foam" in t["text"])
        ask2 = drift_seed_ask(seed2)
        check("a carried-on thread opens from where it got to", "The hole was there before the finger was." in ask2 and ask2.endswith("Where does it go from here?"), ask2)

        led.note_question("What is the hole in that ceiling tile for?")
        ask3 = drift_seed_ask(seed2)
        check("its own question about the thread is the door", ask3.endswith('You asked: "What is the hole in that ceiling tile for?"'), ask3)
        check("an unrelated question is not used", led.question_for({"text": "A thread about the pink rack of cables."}) is None)
        check("no seed, no ask", drift_seed_ask({}) == "" and drift_seed_ask(None) == "")
    finally:
        _mod.lore_ledger = real


def test_distill_harvest():
    print("\n[2] distill parse + template")
    from captioner.context_compression import ContextCompressionEngine
    from captioner.prompt_registry import FRAGMENTS

    parse = ContextCompressionEngine._parse_distillation
    r = parse(
        None,
        "TRAIT: I stall.\nBELIEF: none\nWANT: to draw\nKERNEL: I saw it plain.\nNAME: Penelope\n"
        "UNDERSTANDING: The finger is a lighthouse.\nQUESTION: What does he build all day?",
    )
    trait, belief, want, kernel, became, name, lore, question = r[:8]  # the tuple has grown twice since (no_longer, resolved, thread)
    check("name parsed", name == "Penelope")
    check("understanding parsed", lore == "The finger is a lighthouse.")
    check("question parsed", question == "What does he build all day?")
    check("legacy LORE label still parses", parse(None, "LORE: old label")[6] == "old label")
    check("none stays empty", belief == "")
    # Sep 13: the THREAD slot — which earlier thought this reflection carried on.
    r2 = parse(None, "KERNEL: I got further.\nTHREAD: the finger as a lighthouse\nQUESTION: none")
    check("thread slot parsed", r2[10] == "the finger as a lighthouse", r2[10])
    check("'none' thread stays empty", parse(None, "THREAD: none")[10] == "")
    check("the distill prompt asks for it", "THREAD —" in FRAGMENTS["distill.user"]["text"])
    r2 = parse(None, "TRAIT: none\nNAME: none\nLORE: none\nQUESTION: none")
    check("all-none harvest is empty", r2[5] == "" and r2[6] == "" and r2[7] == "")
    txt = FRAGMENTS["distill.user"]["text"]
    check("template carries NAME slot", "NAME —" in txt)
    check("template carries UNDERSTANDING slot", "UNDERSTANDING —" in txt)
    check("template carries QUESTION slot", "QUESTION —" in txt)
    check("slots are harvest-only ('or none')", txt.count("or 'none'") >= 3)


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
        check(
            "lore seed rides the ask, as a question (Sep 13)",
            calls and "A thought you were having:" in calls[0]["prompt"] and "Where does it go from here?" in calls[0]["prompt"],
            str(calls[0]["prompt"])[:110] if calls else "",
        )
        check(
            "and the drift's answer is written back to that thread",
            (ll_mod.lore_ledger.thread_stats()["returned"] == 1) and "agreement even when no one watches" in (ll_mod.lore_ledger.alive_threads(3)[0]["returns"][-1]["advance"]),
            ll_mod.lore_ledger.thread_stats(),
        )
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
        check("provenance-marked framing", fired and "you've been developing" in fired[0].lower(), fired[0] if fired else "")

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
    test_thread_returns()
    test_drift_ask()
    test_distill_harvest()
    test_drift_integration()
    test_reentry_surfaces()
    test_firewall()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
