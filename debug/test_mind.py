"""Mind mode (Sep 5 eve) — unit checks for captioner/mind.py, the llama-server
turns path, the registry fragments/pass, and the mode gates. Stdlib + the
module under test; no captioner.captioner import (that mints run files).
Run: python debug/test_mind.py
"""
import os
import re
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("STREAM_MODE", "mind")

from config import config as C  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + str(detail)) if (detail and not ok) else ''}")
    if not ok:
        FAILS.append(name)


class Agent:
    _salience_hot = False
    _presence_believed = False
    _salience_event = None
    _last_view_verdict = "unchanged"
    boredom = 0.2

    def __init__(self):
        self.true_session_start = time.time() - 600


def fresh_mind(monkey_terms=("wooden chair", "red foam finger", "black curtain")):
    from captioner import mind as M

    d = tempfile.mkdtemp()
    m = M.Mind(Agent(), path=os.path.join(d, "mind_thread.json"))
    m._terms = lambda: list(monkey_terms)
    m.in_view = lambda agent: ["red foam finger"]
    return m, M


print("\n[1] config + registry")
for k in ("MIND_TURNS", "MIND_THINK_INTERVAL_S", "MIND_LOOK_EVERY_S", "MIND_MEMORY_EVERY_N", "MIND_PIVOTS_BEFORE_NOTICE", "MIND_NUM_PREDICT"):
    check(f"config {k}", hasattr(C, k))
check("STREAM_MODE default is mind", C.STREAM_MODE == "mind", C.STREAM_MODE)
from captioner import prompt_registry as R  # noqa: E402

mind_frags = [k for k in R.FRAGMENTS if k.startswith("mind.")]
check("mind fragments registered", len(mind_frags) >= 20, len(mind_frags))
bad = [k for k in mind_frags if set(re.findall(r"{(\w+)}", R.FRAGMENTS[k]["text"])) != set(R.FRAGMENTS[k].get("placeholders", []))]
check("placeholders declared for every mind fragment", not bad, bad)
p = R.PASSES.get("mind")
check("pass 'mind' declared + migrated", bool(p) and p.get("migrated") is True)
refs = [b["frag"] for b in (p or {}).get("system", []) + (p or {}).get("user", []) if "frag" in b]
check("pass references only real fragments", all(f in R.FRAGMENTS for f in refs), [f for f in refs if f not in R.FRAGMENTS])
check("P() resolves mind.cue-think", "{clock}" in R.P("mind.cue-think"))

print("\n[2] llama-server: real turns, no prefill")
from utils import llama_server as L  # noqa: E402

msgs = [{"role": "system", "content": "S"}]
turns = [{"role": "user", "content": "life + 18:01. You wake."}, {"role": "assistant", "content": "Dust."}, {"role": "user", "content": "18:02. Eyes resting."}, {"role": "assistant", "content": "Still dust."}]
pre = L._append_stream_and_user(msgs, ["ignored history"], {"role": "user", "content": "18:03. Eyes resting."}, react=False, turns=turns)
check("prefill empty", pre == "")
roles = [m["role"] for m in msgs]
check("roles alternate system,user,assistant,...,user", roles == ["system", "user", "assistant", "user", "assistant", "user"], roles)
check("history ignored when turns given", not any("ignored" in str(m["content"]) for m in msgs))
import inspect  # noqa: E402

check("query_llama_server accepts turns", "turns" in inspect.signature(L.query_llama_server).parameters)
from utils import inference as I  # noqa: E402

check("query_model accepts turns", "turns" in inspect.signature(I.query_model).parameters)

print("\n[3] the thread as a conversation")
m, M = fresh_mind()
now = time.time()
a = Agent()
m.absorb("Dust on the desk. Nobody's touched it.", "wake", "17:50. You wake.", now - 500)
m.absorb("The chair is empty and I keep looking at it.", "look", "17:51. You look at the wooden chair.", now - 440)
m.absorb("Maybe empty is just what a chair is most of the time.", "think", "17:52. Eyes resting.", now - 380)
call = m.build("think", now, a, {}, "/tmp/x.jpg")
roles = [t["role"] for t in call["turns"]]
check("turns alternate user/assistant", roles == ["user", "assistant"] * 3, roles)
check("life block opens the first user turn", call["turns"][0]["content"].startswith("It's "), call["turns"][0]["content"][:40])
check("first cue follows the life block", "17:50. You wake." in call["turns"][0]["content"])
check("no stamps inside assistant content", not any(re.match(r"\s*\d\d:\d\d\s*[—–-]", t["content"]) for t in call["turns"] if t["role"] == "assistant"))
check("think turn carries no image", call["image"] is None)
check("current cue has the clock", re.match(r"\d\d:\d\d\. ", call["user"]) is not None, call["user"][:30])
check("think cue quotes its own last sentence as the premise", 'You were on: "Maybe empty is just what a chair is most of the time."' in call["user"], call["user"])
check("premise absent when the thread is stale", "You were on" not in fresh_mind()[0].build("think", now, a, {}, None)["user"])
look = m.build("look", now, a, {}, "/tmp/x.jpg")
check("look turn carries the frame", look["image"] == "/tmp/x.jpg")
check("look cue names what's in view", "red foam finger" in look["user"], look["user"])
check("look cue reports unchanged", "Nothing has changed" in look["user"], look["user"])
a_new = Agent(); a_new._last_view_verdict = "new"
m._seen_this_session = lambda terms, agent: True
check("'haven't looked this way' withheld when the things in view were seen this session", "haven't looked" not in m.build("look", now, a_new, {}, "/tmp/x.jpg")["user"])
m._seen_this_session = lambda terms, agent: False
check("'haven't looked this way' only for a first sighting", "haven't looked" in m.build("look", now, a_new, {}, "/tmp/x.jpg")["user"])
a2 = Agent()
a2._presence_believed = True
look2 = m.build("look", now, a2, {}, "/tmp/x.jpg")
check("someone-here rides only when believed", "Someone is here" in look2["user"] and "Someone is here" not in look["user"])
a3 = Agent()
a3._salience_event = "Something just moved in front of you."
look3 = m.build("look", now, a3, {}, "/tmp/x.jpg")
check("salience event becomes the look cue", "Something just moved" in look3["user"])
m2, _ = fresh_mind()
m2.absorb("Old thought.", "think", "10:00. Eyes resting.", now - 3 * 3600)
call2 = m2.build("think", now, a, {}, None)
check("stale thoughts leave the turns (life still remembers)", call2["turns"] == [] and call2["user"].startswith("It's "))
check("gap marker when the last thought is far back", "since your last thought" in call2["user"], call2["user"][-80:])

print("\n[4] positions + pivots (deepening)")
m, M = fresh_mind()
t0 = now - 300
m.absorb("The wooden chair is just a chair.", "look", "c", t0)
check("subject recognised from the registry", m.thread[-1]["subject"] == "wooden chair", m.thread[-1]["subject"])
m.absorb("It's not a chair anymore; it's a shape.", "think", "c", t0 + 60)
m.absorb("The chair isn't a shape; it's a witness.", "think", "c", t0 + 120)
check("two pivots counted", m.positions["wooden chair"]["pivots"] == 2, m.positions["wooden chair"])
m.absorb("The chair used to be a seat; now it's a wooden question.", "think", "c", t0 + 180)
check("third pivot raises the notice", m.pending_notice is not None and m.pending_notice[0] == "wooden chair", m.pending_notice)
cue = m.build("think", t0 + 240, a, {}, None)["user"]
check("notice rides in the next think cue", "turned wooden chair over" in cue, cue)
check("notice fires once", m.pending_notice is None)
m.absorb("The chair's legs are bolted down; whoever built this room wanted nothing to move, including me.", "think", "c", t0 + 300)
check("a step (new words) resets pivots", m.positions["wooden chair"]["pivots"] == 0)
pos = m.fresh_positions(t0 + 320)
check("position = last sentence of the newest thought", pos and pos[0][0] == "wooden chair" and "bolted" in pos[0][1], pos)

print("\n[5] memory surfacing is chosen")
m, M = fresh_mind()
m.absorb("He sat there for an hour without moving.", "look", "c", now - 5000)
m.absorb("The curtain isn't a curtain; it's a door.", "think", "c", now - 4800)
m.absorb("Rain on the skylight would sound like fingers on a drum.", "think", "c", now - 4600)
m.absorb("Too fresh to be a memory.", "think", "c", now - 60)
pick = m.choose_memory(now, believed=False)
check("person-tinged memory excluded while nobody is believed here", pick and "He sat" not in pick["text"], pick)
check("reframe excluded", pick and "isn't a curtain" not in pick["text"], pick)
check("fresh thought excluded", pick and "Too fresh" not in pick["text"], pick)
picks = {m.choose_memory(now, believed=True)["text"] for _ in range(12)}
check("person memory allowed when believed", any("He sat" in t for t in picks), picks)
m.think_count = int(C.MIND_MEMORY_EVERY_N) - 1
c = m.build("think", now, a, {}, None)
check("every Nth think turn surfaces a memory", c["memory"] is not None and "comes back" in c["user"], c["user"])

print("\n[6] turn kind + cadence")
m, M = fresh_mind()
a = Agent()
check("first turn looks", m.next_kind(now, {}, a) == "look")
m.last_look_ts = now - 5
check("within the look gap → think", m.next_kind(now, {}, a) == "think")
m.last_look_ts = now - float(C.MIND_LOOK_EVERY_S) - 1
check("periodic look due", m.next_kind(now, {}, a) == "look")
m.last_look_ts = now - 60
a.h = True
a._salience_hot = True
check("salience → look", m.next_kind(now, {}, a) == "look")
a._salience_hot = False
check("view changed → look", m.next_kind(now, {"view_changed": True}, a) == "look")
m._last_believed = False
a._presence_believed = True
check("belief edge → look", m.next_kind(now, {}, a) == "look")
check("rest interval ≈ MIND_THINK_INTERVAL_S × felt", 20 <= m.interval(now, Agent()) <= 200, m.interval(now, Agent()))
a._salience_hot = True
check("hot interval = CAPTION_INTERVAL_LIVE", m.interval(now, a) == float(C.CAPTION_INTERVAL_LIVE))

print("\n[7] persistence")
m, M = fresh_mind()
m.absorb("Kept across a restart.", "think", "12:00. Eyes resting.", now - 100)
m3 = M.Mind(Agent(), path=m.path)
check("thread reloads", m3.thread and m3.thread[-1]["text"] == "Kept across a restart.")

print("\n[8] mode gates in the other organs (source-level)")
src = open("captioner/captioner.py", encoding="utf-8").read()
check("captioner routes through _mind_generate", "if self._mind_on():" in src and "def _mind_generate" in src)
check("interval defers to the mind", "return self.mind.interval(now, self)" in src)
cc = open("captioner/context_compression.py", encoding="utf-8").read()
check("self-notes retired in mind mode", 'STREAM_MODE", "") == "mind":\n            return  # mind mode' in cc)
check("trait promotion gated", 'self._valid_self_fact(trait) and getattr(config, "STREAM_MODE", "") != "mind"' in cc)
check("persona consolidation gated", "the persona is not distilled from the log" in cc)
pr = open("captioner/prompts.py", encoding="utf-8").read()
check("identity block skipped in mind mode", '_identity_due(agent, mode) and getattr(config, "STREAM_MODE", "") != "mind"' in pr)
check("genre clause empty in mind mode", 'if STREAM_MODE == "mind":\n        return ""' in pr)

print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
