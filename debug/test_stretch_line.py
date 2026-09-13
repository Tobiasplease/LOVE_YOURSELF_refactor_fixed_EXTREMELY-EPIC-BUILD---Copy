#!/usr/bin/env python3
"""Sep 13 — the stretch line (prompts.get_unchanged_line): what the machine did
with a still stretch, from its own record, counts in words; the repeated
phrase counted as an event. Fakes the ledgers and the compressor."""
import os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.episodic_log as _el  # noqa: E402
_el.episodic_log.get_last_event = lambda etype: None
sys.modules["captioner.context_compression"] = types.SimpleNamespace(context_compressor=types.SimpleNamespace(introspective_state={}))
CC = sys.modules["captioner.context_compression"].context_compressor
from captioner.prompts import _DRIFTED_LADDER, _LOOKED_LADDER, _NAMED_LADDER, _ladder, _rung, _top_phrase, count_words, get_unchanged_line  # noqa: E402

fails = 0
def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else "")); fails += 0 if ok else 1

T = time.time()
class A: pass
def agent(still_s, acts):
    a = A(); a.true_session_start = T - still_s; a._acts = acts; a._current_caption_interval = lambda now: 8.0; return a

check("count words", [count_words(n) for n in (0,1,2,3,9,12,17,25,40,70,150)] == ["not at all","once","twice","three times","nine times","a dozen times","fifteen-odd times","twenty-odd times","thirty or forty times","dozens of times","a hundred times or more"])
check("no digits in any count", not any(ch.isdigit() for n in range(0,400) for ch in count_words(n)))

# a fifty-minute stretch: nine mentions of the finger, twelve looks, one drift, forty quiet cycles
acts = [(T - 3000 + i * 30, "said", "The red foam finger is still up there, pointing at nothing.") for i in range(9)]
acts += [(T - 2900 + i * 200, "look", "") for i in range(12)]
acts += [(T - 1500, "drift", "the factory it came from")]
acts += [(T - 2000 + i * 8, "silence", "") for i in range(40)]
acts += [(T - 2500 + i * 100, "said", "The black curtain hangs there.") for i in range(4)]
a = agent(3000, acts)
CC.introspective_state = {"loop_notice": {"phrase": "the red foam finger", "ts": T - 600}}
line = get_unchanged_line(a)
check("the stretch line", line == "It's been about three quarters of an hour since anything happened. In that time you've looked at every corner of it, said the red foam finger so often it has stopped describing anything, drifted off once, and been quiet for a few minutes of it.", line)
check("no digits in it", not any(ch.isdigit() for ch in line))
CC.introspective_state = {}
line2 = get_unchanged_line(a)
check("without the compressor's phrase, its own repeated words are used (earliest run wins the tie)", "said red foam finger so often" in line2, line2)
check("nothing done → no line", get_unchanged_line(agent(3000, [])) == "")
check("under two minutes → no line", get_unchanged_line(agent(60, acts)) == "")
ph, k = _top_phrase(["the empty chair by the desk", "that empty chair again", "an empty chair, an empty desk"], T - 100)
check("top phrase fallback finds the repeated contiguous run", (ph, k) == ("empty chair", 3), (ph, k))
# 11:43 boot: "named chair has three times", "named i'm looking four times" — the thing named must end on a noun
for bad in ("chair has", "i'm looking", "still sitting"):
    CC.introspective_state = {"loop_notice": {"phrase": bad, "ts": T - 10}}
    ph, k = _top_phrase(["a sheet of paper, still sitting there", "sheet of paper, still sitting", "that sheet of paper is still sitting"], T - 100)
    check(f"compressor phrase {bad!r} refused (not a noun phrase); own run 'sheet of paper' instead", (ph, k) == ("sheet of paper", 3), (ph, k))
CC.introspective_state = {}
ph, k = _top_phrase(["still sitting by the black curtain", "still sitting, black curtain", "still sitting near the black curtain"], T - 100)
check("fallback skips the verb run that comes first and takes the noun run", (ph, k) == ("black curtain", 3), (ph, k))

# Sep 13 11:22, first live quarter-hour: the compressor's REPEATING phrase went in unchecked —
# "named 16 hours nine times", "named pointing four times", "named or a paper three times".
saids = ["The white sheet of paper is still there.", "That sheet of paper hasn't moved.", "A sheet of paper, 16 hours old.", "Nothing."]
for bad in ("16 hours", "pointing", "or a paper"):
    CC.introspective_state = {"loop_notice": {"phrase": bad, "ts": T - 10}}
    ph, k = _top_phrase(saids, T - 100)
    check(f"compressor phrase {bad!r} is refused; own contiguous run used instead", ph == "sheet of paper" and k == 3, (ph, k))
CC.introspective_state = {}
ph, k = _top_phrase(["the clock says 12 hours", "12 hours of this", "12 hours again"], T - 100)
check("no digit phrase from the fallback either", not any(ch.isdigit() for ch in ph), (ph, k))
ph, k = _top_phrase(["red foam finger up there", "the red foam finger again", "red foam finger, still"], T - 100)
check("edges must be content words: 'red foam finger', never 'the red foam'", ph == "red foam finger", (ph, k))
# Sep 13 evening (artist: "Mentioning something ten times is not the same as
# mentioning it 2000 times, but both fall under 'you keep coming back to'"). The
# count stays in the code; what crosses is what the repetition has become, on a
# log-spaced ladder — so it compounds, and no rung has a slot a numeral can fill.
check("no digits anywhere in any rung", not any(c.isdigit() for L in (_NAMED_LADDER, _LOOKED_LADDER, _DRIFTED_LADDER) for r in L for c in r))
named = [_ladder(_NAMED_LADDER, n, p="the finger") for n in (2, 5, 10, 25, 70, 200, 500, 2000)]
check("ten mentions and two thousand do not read the same", len(set(named)) == len(named), named)
check("and each one is a different observation, not a bigger number", "coming back" in named[0] and "worn" in named[-1], (named[0], named[-1]))
check("two thousand is eight rungs past ten", _rung(2000) - _rung(10) == 5, (_rung(10), _rung(2000)))
check("the ladder never runs out", _ladder(_NAMED_LADDER, 10**9, p="x") == _NAMED_LADDER[-1].format(p="x"))
big = agent(7200, [(T - 7000 + i, "said", "the red foam finger again") for i in range(300)] + [(T - 6000 + i * 10, "look", "") for i in range(200)])
line_big = get_unchanged_line(big)
check("a long stretch reads as erosion, not as a tally", "the way a clock says the hour" in line_big and not any(c.isdigit() for c in line_big), line_big)

q = agent(600, [(T - 400 + i * 8, "silence", "") for i in range(12)] + [(T - 300, "look", "")])
line3 = get_unchanged_line(q)
check("twelve quiet cycles (96 s) read as 'a minute of it', not 'just now of it'", line3.endswith("and been quiet for a minute of it."), line3)
q = agent(600, [(T - 400 + i * 8, "silence", "") for i in range(5)] + [(T - 300, "look", "")])
check("under a minute of quiet is not mentioned", "quiet" not in get_unchanged_line(q), get_unchanged_line(q))
print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED")); sys.exit(1 if fails else 0)
