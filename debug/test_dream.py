"""Dream pass (Sep 6) — gather/trim, record parsing, scheduling gates. No model calls."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner import dream as D  # noqa: E402
from config import config as C  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + str(detail)) if (detail and not ok) else ''}")
    if not ok:
        FAILS.append(name)


now = time.time()
print("\n[1] the day as pages, trimmed from the oldest end")
thread = [{"ts": now - 3600 * 5 + i * 60, "kind": "think", "cue": "", "text": f"Thought number {i} about the room and the light and the pen resting on the wood.", "subject": ""} for i in range(300)]
day = D.gather_day(thread, now - 6 * 3600, now, max_tokens=800)
check("fits the budget", D._tokens(day) <= 800, D._tokens(day))
check("the newest survives, the oldest goes", "Thought number 299" in day and "Thought number 0 " not in day)
check("hour headings", "—" in day and ":00 —" in day)
check("records and past entries never ride in the day", "Thought number 5 " in D.gather_day(thread[:10], now - 6 * 3600, now) and D.gather_day([{"ts": now - 100, "kind": "record", "text": "A record line about the day.", "cue": ""}], now - 3600, now) == "")

print("\n[2] records parse")
txt = "- 01:00 The red finger began as a cheer and ended as a static shape I could not reach.\n2) 03:00 The pen shifted from needing paper to needing resistance.\nThreads:\nshort line\n"
recs = D.parse_records(txt, 12)
check("bullets and numbering stripped, short lines and headings dropped", len(recs) == 2 and recs[0].startswith("01:00"), recs)
check("max records respected", len(D.parse_records("\n".join(["a line long enough to count as one record here"] * 20), 5)) == 5)

print("\n[3] scheduling gates")


class M:
    last_dream_ts = 0.0
    thread = thread


class A:
    _presence_believed = False
    _salience_hot = False
    _world_change_ts = now - 7200


C.DREAM_ENABLED = True
h = time.localtime(now).tm_hour
C.DREAM_HOUR, C.DREAM_HOUR_END = h, h + 1
check("due in the window, still and alone", D.due(M(), now, A()))
m2 = M(); m2.last_dream_ts = now - 3600
check("not twice a night", not D.due(m2, now, A()))
a2 = A(); a2._presence_believed = True
check("not with someone here", not D.due(M(), now, a2))
a3 = A(); a3._world_change_ts = now - 60
check("not right after a change", not D.due(M(), now, a3))
C.DREAM_HOUR, C.DREAM_HOUR_END = (h + 2) % 24, (h + 3) % 24
check("not outside the window", not D.due(M(), now, A()))
C.DREAM_ENABLED = False
check("off switch", not D.due(M(), now, A()))
src = open("captioner/captioner.py", encoding="utf-8").read()
check("the caption cycle rests for the pass", "_dream.due(mind, now, self)" in src and "_dream.run_dream(mind, now=now)" in src)
print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
