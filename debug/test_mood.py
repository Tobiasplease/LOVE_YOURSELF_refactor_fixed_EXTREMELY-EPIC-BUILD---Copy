"""Mood with dynamics (Sep 6) — utils/mood.py + its wiring. Run: python debug/test_mood.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config as C  # noqa: E402
from utils import mood as M  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + str(detail)) if (detail and not ok) else ''}")
    if not ok:
        FAILS.append(name)


def reset(now):
    M._STATE.update({"v": 0.0, "a": 0.35, "ts": now, "scare_ts": 0.0, "felt": "", "felt_since": 0.0, "label": "neutral"})


now = time.time()
print("\n[1] targets follow the situation")
reset(now)
v, a = M.targets({"valence": 0.0, "arousal": 0.35, "felt": ""}, {})
check("no situation → the read", abs(v) < 1e-9 and abs(a - 0.35) < 1e-9)
v2, a2 = M.targets({"valence": 0.0, "arousal": 0.35}, {"awake_h": 8, "night": True, "still_h": 3})
check("fatigue + night + stillness pull arousal down", a2 < a - 0.3, a2)
v3, a3 = M.targets({"valence": 0.0, "arousal": 0.35}, {"refusals": 4})
check("refusals: valence down, arousal up", v3 < -0.25 and a3 > a, (v3, a3))
v4, a4 = M.targets({"valence": 0.0, "arousal": 0.35}, {"settled": True})
check("a settled reflection: valence up, arousal down", v4 > 0.1 and a4 < a)

print("\n[2] inertia and labels")
reset(now)
for i in range(1, 7):
    M.tick(now + i * 300, {"valence": 0.0, "arousal": 0.35}, {"awake_h": 9, "night": True, "still_h": 3})
st = M.state()
check("half an hour of night fatigue → flat", st["label"] == "flat", st)
_, a_day = M.targets({"valence": 0.0, "arousal": 0.35}, {"awake_h": 9, "still_h": 3})
check("daytime fatigue is a pressure, not a verdict", a_day > 0.05, a_day)
reset(now)
M.tick(now + 60, {"valence": 0.5, "arousal": 0.3}, {"settled": True, "awake_h": 9, "still_h": 3, "night": True})
for i in range(2, 8):
    M.tick(now + i * 300, {"valence": 0.5, "arousal": 0.3}, {"settled": True, "awake_h": 9, "still_h": 3, "night": True})
check("settled + low arousal → serene", M.state()["label"] == "serene", M.state())
reset(now)
for i in range(1, 6):
    M.tick(now + i * 300, {"valence": -0.5, "arousal": 0.55}, {"refusals": 3})
check("refusals + stirred → frustrated", M.state()["label"] == "frustrated", M.state())
reset(now)
M.tick(now + 60, {"valence": 0.0, "arousal": 0.35}, {"scare": True})
check("a scare jumps arousal at once", M.state()["a"] > 0.55, M.state())
M._STATE["scare_ts"] = time.time()
check("scare + high arousal → on_edge", M.label() == "on_edge")
reset(now)
M.tick(now + 1, {"valence": 0.0, "arousal": 0.35}, {})
check("nothing → neutral", M.state()["label"] == "neutral")

print("\n[3] the cadence map is the malleable part")
M._STATE["label"] = "flat"
c = M.cadence()
check("flat: fewer looks, cooler; length and rate untouched", c["interval_mult"] == 1.0 and c["budget_scale"] == 1.0 and c["look_mult"] < 1.0 and c["temp_delta"] < 0)
M._STATE["label"] = "on_edge"
c = M.cadence()
check("on edge: more looks, hotter; length and rate untouched", c["interval_mult"] == 1.0 and c["look_mult"] >= 1.5 and c["temp_delta"] > 0)
C.MOOD_CADENCE_MAP = {"on_edge": {"look_mult": 3.0}}
check("config override per label", M.cadence()["look_mult"] == 3.0)
C.MOOD_CADENCE_MAP = {}
from utils import felt_loop as F  # noqa: E402

M._STATE["label"] = "flat"
C.MOOD_CADENCE_MAP = {"flat": {"interval_mult": 1.6, "budget_scale": 0.7}}
check("felt_loop sources the mood (when the artist sets length/rate in the map)", F.cadence_mult() > 1.5 and F.budget_scale() < 0.8)
C.MOOD_CADENCE_MAP = {}

print("\n[4] felt held + wiring (source-level)")
M._STATE.update({"felt": "tired", "felt_since": now - 4000})
check("felt held duration", 3990 < M.felt_held_s(now) < 4010)
src = open("captioner/mind.py", encoding="utf-8").read()
check("mind ticks the mood each build", "self.tick_mood(now, agent)" in src)
check("frame says how long the felt word has held", 'P("monologue.felt-held")' in src)
check("look rate follows the mood", "look_mult" in src)
check("situation reaches the compressor's FELT ask", "_cc.situation_line = self.situation_words(inputs)" in src)
cc = open("captioner/context_compression.py", encoding="utf-8").read()
check("compression prompt carries the situation", "situation=(getattr(self, \"situation_line\"" in cc)
cap = open("captioner/captioner.py", encoding="utf-8").read()
check("phantom gate is a scare to the mood", "mind.note_scare(now)" in cap)
check("heat follows the mood", '_mc.get("temp_delta"' in cap)
from captioner.mind import Mind  # noqa: E402


class A:
    _salience_hot = False
    _presence_believed = False
    _salience_event = None
    _last_view_verdict = "unchanged"
    _world_change_ts = now - 7200
    _presence_dropped_at = now - 5 * 3600
    _loop_hits = [(now - 100, "x"), (now - 200, "y")]
    _skip_streak = 0
    boredom = 0.2
    true_session_start = now - 3600


m = Mind(A(), path="/tmp/mood_test_thread.json", backfill=False)
m._index = False
m.absorb("A thought.", "think", "c", now - 3000)
words = m.situation_words(m.situation(now, A()))
check("situation in words: durations and facts", "no one here for" in words and "circling" in words and "nothing changed" in words, words)

print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
