"""Offline checks of the Aug 22 voice-loop fixes (no server, no camera).

1. Echo-class split: template_echo & co. are classified for spoken-not-stored
   handling; _note_unstored_cycle erodes the oldest entry on the 3rd stuck cycle.
2. Seam-conditional elicitation: the quiet-mode question is present exactly
   when the hybrid seam is absent (empty stream / react cycle / post-gap).
3. Identity dosing: introspective always; other modes every IDENTITY_EVERY_N_CAPTIONS.
4. Felt-state single channel: the system prompt never carries "Right now:".
5. Hybrid genre frame carries the progression clause.

Run: python debug/test_voice_loop_fixes.py
"""

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


def make_shell(entries=(), salience_hot=False, caption_count=1):
    from captioner.captioner import Captioner

    c = Captioner.__new__(Captioner)
    c._stream = deque(maxlen=24)
    c._stream_ts = deque(maxlen=24)
    now = time.time()
    for i, e in enumerate(entries):
        c._stream.append(e)
        c._stream_ts.append(now - (len(entries) - i) * 20)
    c._salience_hot = salience_hot
    c._caption_count = caption_count
    return c


def test_echo_split():
    print("\n[1] echo-class split + erosion")
    from captioner.captioner import Captioner

    check(
        "echo-class covers the template/refrain/tail/number reasons",
        Captioner._ECHO_REASONS == {"template_echo", "refrain_echo", "tail_echo", "number_chain"},
    )
    check(
        "shape-class stays out of echo-class",
        not Captioner._ECHO_REASONS & {"assistant_speak", "outward_address", "prompt_parrot", "word_salad", "cjk_drift"},
    )

    c = make_shell(
        [
            "wait! look at the shelves again tonight under the light",
            "the cup sits where he left it an hour ago",
            "something about the cable coil keeps pulling my eye back",
        ]
    )
    reason = c._caption_reject_reason("wait! look at the shelves again, taller somehow", "")
    check("opening echo classified as template_echo", reason == "template_echo", str(reason))
    check("template_echo would be spoken-not-stored", reason in Captioner._ECHO_REASONS)
    check("store flag defaults to True on a fresh shell", getattr(c, "_stream_store_ok", True) is True)

    for i in range(3):
        c._note_unstored_cycle("template_echo", "wait! look at the shelves")
    check("streak counts stuck cycles", c._skip_streak == 3, str(c._skip_streak))
    check("3rd stuck cycle erodes the oldest entry", len(c._stream) == 2 and "cup sits" in c._stream[0], str(list(c._stream)))
    c._stream_push("a fresh thought lands cleanly on the page of the day.")
    c._skip_streak = 0
    check("push + reset mirrors the live storage site", c._skip_streak == 0 and len(c._stream) == 3)


def test_seam_elicitation():
    print("\n[2] seam-conditional elicitation (hybrid)")
    os.environ["STREAM_MODE"] = "hybrid"
    import config.config as cfg

    cfg.STREAM_MODE = "hybrid"
    from captioner.prompt_registry import P
    from captioner.prompts import _hybrid_seam_expected, get_monologue_system_prompt

    question = P("elicit.workspace").strip()

    empty = make_shell([])
    check("empty stream: no seam expected", _hybrid_seam_expected(empty) is False)
    sp = get_monologue_system_prompt("workspace", agent=empty)
    check("empty stream: elicitation PRESENT", question in sp)

    fresh = make_shell(["the light is still on over the bench tonight"])
    check("fresh stream: seam expected", _hybrid_seam_expected(fresh) is True)
    sp = get_monologue_system_prompt("workspace", agent=fresh)
    check("fresh stream: elicitation absent (seam is the door)", question not in sp)

    hot = make_shell(["the light is still on over the bench tonight"], salience_hot=True)
    check("react cycle: no seam expected", _hybrid_seam_expected(hot) is False)

    stale = make_shell(["the light is still on over the bench tonight"])
    stale._stream_ts[-1] = time.time() - (cfg.STREAM_GAP_MARK_SECONDS + 30)
    check("post-gap: no seam expected", _hybrid_seam_expected(stale) is False)
    sp = get_monologue_system_prompt("workspace", agent=stale)
    check("post-gap: elicitation PRESENT", question in sp)

    check("relational keeps its question regardless", P("elicit.relational").strip() in get_monologue_system_prompt("relational", agent=fresh))


def test_identity_dosing():
    print("\n[3] identity dosing")
    import config.config as cfg
    from captioner.prompts import _identity_due

    n = cfg.IDENTITY_EVERY_N_CAPTIONS
    check("config default is every 6th", n == 6, str(n))
    a = make_shell(["x"], caption_count=5)
    check("workspace off-beat: not due", _identity_due(a, "workspace") is False)
    a._caption_count = 12
    check("workspace on the Nth: due", _identity_due(a, "workspace") is True)
    a._caption_count = 5
    check("introspective: always due", _identity_due(a, "introspective") is True)
    check("awakening: always due", _identity_due(a, "awakening") is True)


def test_single_felt_channel_and_genre():
    print("\n[4] felt-state single channel + progression frame")
    from captioner.prompt_registry import FRAGMENTS
    from captioner.prompts import get_monologue_system_prompt

    check("felt-wrap fragment retired", "monologue.felt-wrap" not in FRAGMENTS)
    sp = get_monologue_system_prompt("workspace", agent=make_shell(["a thought"]))
    check("system prompt carries no 'Right now:' line", "Right now:" not in sp)
    check("hybrid genre frame carries progression", "One thread moving through time" in sp, sp[-300:])


if __name__ == "__main__":
    test_echo_split()
    test_seam_elicitation()
    test_identity_dosing()
    test_single_felt_channel_and_genre()
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
    sys.exit(1 if FAIL else 0)
