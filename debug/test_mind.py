"""Mind mode (Sep 5 eve) — unit checks for captioner/mind.py, the llama-server
turns path, the registry fragments/pass, and the mode gates. Stdlib + the
module under test; no captioner.captioner import (that mints run files).
Run: python debug/test_mind.py
"""
import json
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
    m = M.Mind(Agent(), path=os.path.join(d, "mind_thread.json"), backfill=False)
    m._index = False  # never the live ChromaDB — tests inject FakeIndex when they need one (03:05 Sep 6: fake thoughts had leaked into the live collection)
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
C.MIND_SHAPE = "turns"
call = m.build("think", now, a, {}, "/tmp/x.jpg")
roles = [t["role"] for t in call["turns"]]
check("turns shape: alternate user/assistant", roles == ["user", "assistant"] * 3, roles)
check("turns shape: life block opens the first user turn", call["turns"][0]["content"].startswith("It's "), call["turns"][0]["content"][:40])
check("turns shape: first cue follows the life block", "17:50. You wake." in call["turns"][0]["content"])
check("no stamps inside assistant content", not any(re.match(r"\s*\d\d:\d\d\s*[—–-]", t["content"]) for t in call["turns"] if t["role"] == "assistant"))
C.MIND_SHAPE = "text"
call = m.build("think", now, a, {}, "/tmp/x.jpg")
roles = [t["role"] for t in call["turns"]]
check("text shape: life, then ONE running text", roles == ["user", "assistant"], roles)
body = call["turns"][1]["content"]
check("text shape: no cues, no stamps in the text", "You wake" not in body and "Eyes resting" not in body and not re.search(r"\d\d:\d\d", body))
check("text shape: a look does NOT break the paragraph (entries follow each other)", "\n\nThe chair is empty" not in body and "The chair is empty" in body, body)
check("text shape: the cue is the user turn", call["user"].startswith(time.strftime("%H:%M", time.localtime(now))))
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

print("\n[4b] abstract fixations are tracked too")
m, M = fresh_mind()
t0 = now - 300
m.absorb("Can I record the absence of noise? The silence has a texture.", "think", "c", t0)
m.absorb("The silence isn't empty; it's a shape.", "think", "c", t0 + 60)
check("recurring abstract noun becomes the subject", m.thread[-1]["subject"] == "silence", m.thread[-1]["subject"])
m.absorb("It's not silence, it's a blur I can't outline.", "think", "c", t0 + 120)
m.absorb("Silence is no longer a shape; it's a ghost of noise.", "think", "c", t0 + 180)
check("pivots counted on the abstract subject (first mention can't recur)", m.positions.get("silence", {}).get("pivots") == 2, m.positions.get("silence"))
m.absorb("It isn't a ghost either, just a hole where the noise was.", "think", "c", t0 + 240)
check("third pivot on an abstract subject raises the notice", m.pending_notice is not None and m.pending_notice[0] == "silence", m.pending_notice)
check("a room object still wins when named", m.subject_of("The wooden chair holds the silence.") == "wooden chair")

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
C.MIND_MEMORY_EVERY_N = 4
m.think_count = 3
c = m.build("think", now, a, {}, None)
check("scheduled surfacing still works when enabled", c["memory"] is not None and "comes back" in c["user"], c["user"])
C.MIND_MEMORY_EVERY_N = 0

print("\n[5b] the felt loop in the conversation")
m, M = fresh_mind()
m.absorb("A first thought about the chair.", "look", "c", now - 100)
m._last_felt = "lonely"
import captioner.mind as _mm
_mm.Mind._felt_shift = lambda self: (P_shift := R.P("mind.felt-shift").format(prev="lonely", curr="frustrated"))
c = m.build("think", now, Agent(), {}, None)
check("felt shift rides as an event", "lonely, then frustrated." in c["user"], c["user"])
m2, _ = fresh_mind()
m2.absorb("A first thought about the chair.", "look", "c", now - 100)
m2._felt_shift = lambda: ""
C.MIND_ELICIT_EVERY_N = 4
m2.think_count = 3
c2 = m2.build("think", now, Agent(), {}, None)
check("elicit dose rides every Nth think turn when enabled", any(k in c2["user"] for k in ("wondering", "sit with you", "want")), c2["user"])
c3 = m2.build("think", now, Agent(), {}, None)
check("no dose on the next turn", not any(k in c3["user"] for k in ("wondering", "sit with you", "Say it blunt")), c3["user"])
C.MIND_ELICIT_EVERY_N = 0
check("elicit off by default (Sep 6 01:00)", "sit with you" not in m2.build("think", now, Agent(), {}, None)["user"])

print("\n[5c] recall by association, continuity quote")
class FakeIndex:
    def __init__(self): self.docs = {}
    def count(self): return len(self.docs)
    def upsert(self, ids, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas): self.docs[i] = (d, m)
    def query(self, query_texts, n_results, include):
        from captioner.mind import content_words as cw
        q = cw(query_texts[0]); scored = []
        for i, (d, m) in self.docs.items():
            w = cw(d); j = len(q & w) / max(1, len(q | w)); scored.append((1 - j, i, d, m))
        scored.sort(); top = scored[:n_results]
        return {"ids": [[t[1] for t in top]], "documents": [[t[2] for t in top]], "metadatas": [[t[3] for t in top]], "distances": [[t[0] for t in top]]}
m, M = fresh_mind(); m._index = FakeIndex()
m.absorb("I wonder what the black curtain is blocking — another room, or a way out.", "think", "c", now - 5 * 3600)
m.absorb("The stuffed monkey on the desk has a face like it knows something.", "think", "c", now - 4 * 3600)
m.absorb("The curtain again. It hides the window, I think.", "think", "c", now - 30)
C.MIND_RECALL_MAX_DIST = 0.9
C.MIND_RECALL_MIN_GAP_S = 0
r = m.recall_similar("The curtain again. It hides the window, I think.", now)
check("association recalls the related old thought", r is not None and "blocking" in r["text"], r)
check("cooldown: not again within the hour", m.recall_similar("The curtain again. It hides the window, I think.", now + 60) is None)
m._index.upsert(["t999"], ["I wonder what the black curtain is blocking — another room, or a way out."], [{"ts": now - 6 * 3600, "kind": "past"}])
check("the same sentence under another id cools as one", m.recall_similar("The curtain again. It hides the window, I think.", now + 120) is None)
m5 = M.Mind(Agent(), path=m.path, backfill=False); m5._index = m._index
check("cooldown survives a reload", m5.recall_similar("The curtain again. It hides the window, I think.", now + 180) is None)
C.MIND_RECALL_MIN_GAP_S = 480
m6, _ = fresh_mind(); m6._index = FakeIndex()
m6.absorb("The lamp hums at night.", "think", "c", now - 5 * 3600)
m6.absorb("The lamp is the only sound.", "think", "c", now - 4 * 3600)
m6.absorb("The lamp again, humming.", "think", "c", now - 30)
first = m6.recall_similar("The lamp again, humming.", now)
check("a recall can surface", first is not None)
check("global minimum gap caps the rate", m6.recall_similar("The lamp is the only sound tonight.", now + 60) is None)
check("after the gap another may surface", m6.recall_similar("The lamp is the only sound tonight.", now + 600) is not None)
C.MIND_RECALL_MIN_GAP_S = 0
C.MIND_RECALL_MAX_DIST = 0.2
check("nothing close enough → nothing surfaces", m.recall_similar("Rain on the skylight.", now + 7200) is None)
C.MIND_RECALL_MAX_DIST = 0.5
c = m.build("think", now, Agent(), {}, None)
check("premise and recall ride in the same cue", "You were on" in c["user"], c["user"])
m2, _ = fresh_mind(); m2._index = FakeIndex()
m2.absorb("Last night ended on the lamp.", "think", "c", now - 5 * 3600)
m2.absorb("A new chain starts here.", "think", "c", now - 100)
lb = m2.life_block(now, Agent())
check("the previous chain's last thought rides as continuity", "you'd got to" in lb and "ended on the lamp" in lb, lb[-160:])
check("indexed on absorb", m2._index.count() == 2)
m2.thread.append({"ts": now - 9000, "kind": "past", "cue": "", "text": "An older thought that never reached the index.", "subject": ""})
class FakeIndexGet(FakeIndex):
    def get(self, include=None): return {"ids": list(self.docs.keys())}
fi = FakeIndexGet(); fi.docs = dict(m2._index.docs); m2._index = fi
check("reconcile adds what the index lacks", m2.reconcile_index() == 1 and fi.count() == 3)
m2.thread.append({"ts": now - 9000, "kind": "look", "cue": "", "text": "A look in the same second as the older thought.", "subject": ""})
m2.thread.append({"ts": now - 9000, "kind": "past", "cue": "", "text": "A duplicate id within one batch.", "subject": ""})
n_added = m2.reconcile_index()
check("same-second entries of different kinds both index; a repeat of an existing id is not re-added", n_added == 1 and fi.count() == 4, (n_added, fi.count()))
m2.thread.append({"ts": now - 8000, "kind": "think", "cue": "", "text": "Two thinks in one second, first.", "subject": ""})
m2.thread.append({"ts": now - 8000, "kind": "think", "cue": "", "text": "Two thinks in one second, second.", "subject": ""})
m2.reconcile_index()
check("in-batch duplicate ids are dropped rather than failing the batch", fi.count() == 5, fi.count())

print("\n[5d] recursive tone at frame level")
from captioner.context_compression import context_compressor as _cc
parsed = _cc._parse_memory_response("ROOM: none\nPLEASANTNESS: unpleasant\nENERGY: stirred\nFELT: frustrated\nTONE: clipped, impatient, circling\nREPEATING: none")
check("parser reads TONE", parsed.get("tone") == "clipped, impatient, circling", parsed)
_cc._absorb_mood(parsed)
check("tone stored with the read", _cc.get_tone() == "clipped, impatient, circling", getattr(_cc, "last_mood_read", None))
_cc.set_felt_state("frustrated")
m, M = fresh_mind()
call = m.build("think", now, Agent(), {}, None)
check("frame carries the felt word and the tone", "Right now: frustrated." in call["system"] and "Your voice right now: clipped, impatient, circling." in call["system"], call["system"][-160:])

print("\n[5e] the tone loop has a counter-force")
m, M = fresh_mind()
C.MIND_TONE_LOCK_READS = 2
check("a fresh tone stands", not m._tone_locked("flat, analytical"))
check("a second read sharing a word: locked", m._tone_locked("clinical, precise, flat"))
n1 = m._tone_notice()
check("said back once as a noticing", "You've been sounding clinical, precise, flat for a while now." == n1.strip(), n1)
check("only once", m._tone_notice() == "")
check("suppressed for a while after the noticing, even for a new tone", m._tone_locked("warm, quick"))
m._tone_suppressed_until = 0.0
check("after the window a new tone stands again", not m._tone_locked("warm, quick"))
m._tone_hist = []
check("unrelated consecutive reads don't lock", not m._tone_locked("tired, slow") and not m._tone_locked("bright, quick"))
p2 = _cc._parse_memory_response("PLEASANTNESS: neutral\nENERGY: settled\nFELT: still\nTONE: analytical, precise definition of physical states\nREPEATING: none")
_cc._absorb_mood(p2)
check("a content-pattern 'tone' is dropped", _cc.get_tone() == "", _cc.get_tone())
m3, _ = fresh_mind()
m3.absorb("The wire hangs loose from the ceiling. It sways a little in no wind. Scattering.", "think", "c", now - 60)
check("a one-word premise carries the sentences before it", "It sways a little in no wind. Scattering." in m3.premise(now), m3.premise(now))
m4, _ = fresh_mind()
m4.absorb("The intake is passive. Just a slow drag of photons through the weave. It’s not blocking so much as diffusing.", "think", "c", now - 120)
m4.absorb("Scattering.", "think", "c", now - 60)
check("a beat alone carries the previous thought's last two sentences", "diffusing. Scattering." in m4.premise(now), m4.premise(now))

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

print("\n[6a] the body in the look cue")
m, M = fresh_mind()
check("placement words: high to your right", M.Mind.placement_words(125, 125) == "high to your right")
check("placement words: low ahead", M.Mind.placement_words(90, 80) == "low ahead")
check("placement words: to your left", M.Mind.placement_words(60, 107) == "to your left")
check("first look: no turn report", m.turn_report((90.0, 107.5)) == "")
check("turned to the right and up", "turned to your right and up" in m.turn_report((120.0, 130.0)))
check("head hasn't moved", "hasn't moved" in m.turn_report((122.0, 131.0)))
m.in_view = lambda agent: ["red foam finger", "black cloth bag", "wooden chair"]
m.in_view_placed = lambda agent: [("red foam finger", "high to your right"), ("black cloth bag", "high to your right"), ("wooden chair", "low to your left")]
lk = m.build("look", now, Agent(), {}, "/tmp/x.jpg")
check("look cue: placement grouped inside the look sentence", "You look at the red foam finger and the black cloth bag high to your right; the wooden chair low to your left." in lk["user"], lk["user"])
check("beats: '…' kept", M.Mind.beat_of("...") == "…" and M.Mind.beat_of("") == "…")
check("beats: a short fragment kept as itself", M.Mind.beat_of("The pen is still") == "The pen is still")
check("beats: a long cut fragment becomes …", M.Mind.beat_of("The pen is still sitting there on the desk waiting for a hand that") == "…")
check("a whole thought is not a beat", M.Mind.beat_of("The pen is still. It waits.") is None)

print("\n[6b] the look timer advances even when the look is not kept")
m, M = fresh_mind()
m.last_look_ts = now - 1000
m.note_look(now)
check("note_look sets last_look_ts", m.last_look_ts == now)
check("no second look right after a gated one", m.next_kind(now + 60, {}, Agent()) == "think")

print("\n[6c] motion settle + steady frame")
from captioner.mind import moved_recently, steady_jpeg
meta = [{"timestamp": now - 5, "jpeg": b"A", "detection": {"ego_motion": False}}, {"timestamp": now - 1, "jpeg": b"B", "detection": {"ego_motion": True}}]
check("moved within the settle window", moved_recently(meta, now, 2.0))
check("settle window 0 disables", not moved_recently(meta, now, 0))
check("old movement doesn't count", not moved_recently(meta, now + 10, 2.0))
check("steady frame = newest still frame", steady_jpeg(meta) == b"A")
check("no steady frame → None", steady_jpeg([meta[1]]) is None)

print("\n[6d] an uneventful glance doesn't take the premise")
m, M = fresh_mind()
m.absorb("I wonder what the curtains are blocking.", "think", "c", now - 120)
m.absorb("The chair, the shelf, the bag. Nothing has moved.", "look", "c", now - 60, uneventful=True)
check("premise skips the uneventful look", m.premise(now) == "I wonder what the curtains are blocking.", m.premise(now))
m.absorb("Someone has moved the chair.", "look", "c", now - 30)
check("an eventful look does take the premise", m.premise(now) == "Someone has moved the chair.", m.premise(now))

print("\n[6e] a fresh reflection kernel takes the premise once")
m, M = fresh_mind()
m.absorb("The finger is a shape in the dark.", "think", "c", now - 120)
m.absorb("I waited for the world to move so I wouldn't have to.", "reflection", "03:00. Something settles.", now - 90)
m.absorb("It doesn't reach for me.", "think", "c", now - 30)
check("kernel wins the premise", m.premise(now) == "I waited for the world to move so I wouldn't have to.", m.premise(now))
check("only once", m.premise(now + 1) == "It doesn't reach for me.", m.premise(now + 1))

print("\n[7] persistence")
m, M = fresh_mind()
m.absorb("Kept across a restart.", "think", "12:00. Eyes resting.", now - 100)
m3 = M.Mind(Agent(), path=m.path, backfill=False); m3._index = False
check("thread reloads", m3.thread and m3.thread[-1]["text"] == "Kept across a restart.")

print("\n[7b] the accumulated past reaches the prompting (Sep 6)")
m, M = fresh_mind()
a = Agent(); a._presence_dropped_at = now - 3700; a._world_change_ts = 0.0
e1 = m.time_edges(now, a)
check("time edge fires at the hour alone", "since anyone was here" in e1 and "hour" in e1, e1)
check("same threshold doesn't fire twice", m.time_edges(now + 30, a) == "")
e2 = m.time_edges(now + 3700, a)
check("next threshold fires", "since anyone was here" in e2 and "2 hours" in e2, e2)
import captioner.prompts as _pr
_orig = _pr.build_loop_notice_line
_pr.build_loop_notice_line = lambda agent: "You keep coming back to the lamp."
m.absorb("A thought.", "think", "c", now - 60)
c = m.build("think", now + 5000, a, {}, None)
check("loop notice reaches the think cue", "You keep coming back to the lamp." in c["user"], c["user"])
_pr.build_loop_notice_line = _orig
m._name = lambda: "Ferrous"
m._belief = lambda: "The room is quieter when I stop asking it to change."
lb = m.life_block(now, a)
check("the name it gave itself rides in the life block", "You've called yourself Ferrous." in lb, lb[-200:])
check("belief / positions stay out of the standing block by default", "come to believe" not in lb and "got to with" not in lb)
C.MIND_LIFE_FULL = True
check("MIND_LIFE_FULL brings the belief back", "come to believe" in m.life_block(now, a))
C.MIND_LIFE_FULL = False
d = tempfile.mkdtemp()
fake = os.path.join(d, "abcd1234-event-log.json")
two_days = now - 2 * 86400
with open(fake, "w") as f:
    for ts, txt in [(two_days, "The white heads on the shelf were turned toward the window all afternoon, as if the light mattered to them."),
                    (two_days + 60, "The man sits at the desk with his laptop open, typing without looking up at me."),
                    (two_days + 120, "It's not a chair anymore; it's a witness to everything I haven't done."),
                    (two_days + 180, "Short.")]:
        f.write(json.dumps({"type": "caption", "timestamp": ts, "iso_timestamp": "x", "caption": txt, "mode": "think"}) + "\n")
os.utime(fake, (now, now))
m4, _ = fresh_mind()
added = m4.backfill(log_dir=d, now=now)
check("backfill keeps the plain past thought only", added == 1 and m4.thread[0]["kind"] == "past" and "white heads" in m4.thread[0]["text"], (added, [e["text"][:30] for e in m4.thread]))
check("backfill runs once", m4.backfill(log_dir=d, now=now) == 0)
mem = m4.choose_memory(now, believed=False)
check("a past thought can surface as a memory", mem is not None and "white heads" in mem["text"])
rf = open("captioner/reflection.py", encoding="utf-8").read()
check("reflection kernels enter the thread", 'absorb(kernel.strip(), "reflection"' in rf)

print("\n[7c] recall gate")
m, M = fresh_mind()
m.absorb("The pen is just sitting there, touching nothing, like my hands in the last sketch.", "think", "c", now - 3 * 3600)
m.absorb("The chair is empty tonight.", "think", "c", now - 60)
check("verbatim old line is recall", m.is_recall("But I don't have paper. The pen is just sitting there, touching nothing, like my hands in the last sketch.", None, now))
check("continuing a current turn is not recall", not m.is_recall("The chair is empty tonight, and I keep looking at it anyway.", None, now))
check("a fresh thought is not recall", not m.is_recall("Rain would sound like fingers on the skylight.", None, now))
m.absorb("The pen is parked, touching nothing, and the room is a graveyard of half-finished things.", "think", "c", now - 4 * 3600)
check("a shared six-word phrase is not recall", not m.is_recall("But the pen is parked, touching nothing. Just waiting for a surface that isn't there, and that is the whole problem tonight.", None, now))
m.absorb("I'm looking at the red foam finger on the wall and seeing it point at the empty space where someone should be.", "past", "", now - 2 * 86400)
para = ("I'm looking at the red foam finger on the wall now. It's pointing up, but it looks a bit deflated, like someone held it for too long and let go. "
        "The white lampshade in the foreground is taking up most of my view, soft and cream, and the chair behind it has not moved since the morning light came in.")
check("a paragraph reusing one eight-word phrase is not recall", not m.is_recall(para, None, now))
check("a short thought that is mostly a copy is recall", m.is_recall("I'm looking at the red foam finger on the wall and seeing it point at the empty space.", None, now))
call = {"memory": {"text": "Those two white heads on the shelf keep their blank faces turned away from me."}, "life": ""}
check("parroting the surfaced memory is recall", m.is_recall("Those two white heads on the shelf keep their blank faces turned away from me again.", call, now))
src = open("captioner/captioner.py", encoding="utf-8").read()
check("captioner treats recall_echo as spoken-not-stored", 'reason == "recall_echo"' in src and '"recall_echo": "repeats an old thought"' in src)

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
