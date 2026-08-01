"""World-shape (the inversion, July 26) wiring tests.

STREAM_MODE="world": the stream rides as ONE assistant message of timestamped
log lines and the user message (frames + the world's turn) comes LAST, so
generation always begins right after the present. Run:
python debug/test_world_shape.py
"""

import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


config.STREAM_MODE = "world"

print("— message shape (llama_server) —")
from utils.llama_server import _append_stream_and_user

messages = [{"role": "system", "content": "sys"}]
user = {"role": "user", "content": "the world turn"}
prefill = _append_stream_and_user(messages, ["14:01 — first note", "14:02 — second note"], user, react=False)
check("no prefill in world mode", prefill == "")
check("three messages: system, log, user", [m["role"] for m in messages] == ["system", "assistant", "user"])
check("log is one assistant message with both lines", messages[1]["content"] == "14:01 — first note\n14:02 — second note")
check("user (the world's turn) is LAST", messages[-1] is user)

messages2 = [{"role": "system", "content": "sys"}]
_append_stream_and_user(messages2, [], user, react=True)
check("empty stream → just system + user", [m["role"] for m in messages2] == ["system", "user"])

print("— log rendering (captioner._stream_history) —")
from captioner.captioner import Captioner

cap = object.__new__(Captioner)
cap._stream = deque(["the lamp's still on", "someone left the chair out"], maxlen=6)
now = time.time()
cap._stream_ts = deque([now - 120, now - 60], maxlen=6)
hist = cap._stream_history()
stamp = time.strftime("%H:%M", time.localtime(now - 120))
check("entries render as 'HH:MM — text'", hist[0] == f"{stamp} — the lamp's still on" and len(hist) == 2)

cap._stream_ts = deque([now], maxlen=6)  # drifted: fewer stamps than entries
hist = cap._stream_history()
check("timestamp drift is healed, not fatal", len(hist) == 2 and all("—" in h for h in hist))

config.STREAM_MODE = "document"
check("document mode returns raw text unchanged", cap._stream_history() == list(cap._stream))
config.STREAM_MODE = "world"

print("— mouth salvage —")
check("imitated log stamp stripped", Captioner._strip_list_shape("14:05 — the rooster again") == "the rooster again")
check("plain caption untouched", Captioner._strip_list_shape("The rooster again.") == "The rooster again.")

print("— system frame —")
from captioner.prompts import get_monologue_system_prompt

sysp = get_monologue_system_prompt("observational")
check("world frame is the log genre", "you keep a log" in sysp)
check("lonely-soliloquy trope gone in world frame", "no one hears them" not in sysp)
check("task ask present", "Add the next entry." in sysp)
config.STREAM_MODE = "document"
sysp_doc = get_monologue_system_prompt("observational")
check("document mode gets the reflexive frame", "your own senses reporting" in sysp_doc)
check("negation pile gone", "no one hears" not in sysp_doc and "no one to instruct" not in sysp_doc)
check("assistant vocabulary gone from frame", "assist" not in sysp_doc.split("What you've")[0])
check("questions get an answer-path", "you asking yourself" in sysp_doc)
check("pen absent from answer-path (no false agency)", "next look, or your own next thought" in sysp_doc)
config.STREAM_MODE = "world"


print("— continuity fixes (July 27) —")
check("world clause frames the thread", "one running thread" in get_monologue_system_prompt("observational"))
check("quiet-mode elicitation suppressed in world", "What stands out" not in get_monologue_system_prompt("observational"))
check("relational keeps its question", "What do you make of them being here?" in get_monologue_system_prompt("relational"))
check(
    "mid-entry stamp stripped",
    Captioner._strip_list_shape("My motors spin up. 19:06 — A second figure appears.") == "My motors spin up. A second figure appears.",
)
check("honest time talk survives", "past 19:00" in Captioner._strip_list_shape("It's past 19:00 now and still quiet."))


print("— outward hooks (July 28) —")
check("'what do you think' inadmissible to stream", not Captioner._stream_admissible("A quiet day in here. What do you think?"))
check("self-deliberation stays admissible", Captioner._stream_admissible("Should I draw the mannequin first, or the shelf?"))
check("talking to room objects stays admissible", Captioner._stream_admissible("That rooster again. Will you ever move?"))

print("— outward register, measured (July 28) —")
from captioner.model_wrapper import _is_plantable_prior as _plant

cap3 = object.__new__(Captioner)
cap3._stream = deque([], maxlen=6)
check(
    "two second-person tokens rejected at mouth",
    cap3._caption_reject_reason("I am ready to start drawing again when you give me your input!") == "outward_address",
)
check(
    "planning opener rejected",
    cap3._caption_reject_reason("I'll begin by focusing on the mannequin silhouette against the wall.") == "outward_address",
)
check("(Note:...) meta rejected", cap3._caption_reject_reason("(Note: This response format requires a direct question.)") == "outward_address")
check("single you (talking to the rooster) passes mouth", cap3._caption_reject_reason("That rooster again. Will you ever move, I wonder.") is None)
check("plain thought passes mouth", cap3._caption_reject_reason("The lamp is still on; nobody has touched the chair.") is None)
check("assistant tail can't seed next session", not _plant("What do you think? Your thoughts matter to me."))
check("plain prior still plantable", _plant("The lamp is still on; the chair has not moved."))

print("— hybrid shape (Aug 1) —")
config.STREAM_MODE = "hybrid"
import importlib, utils.llama_server as _ls

msgs = [{"role": "system", "content": "sys"}]
u = {"role": "user", "content": "the world turn"}
pf = _ls._append_stream_and_user(msgs, ["14:01 — first note", "14:02 — the pen is still parked and I keep looking at"], u, react=False)
roles = [m["role"] for m in msgs]
check("shape is system, log, user, prefill", roles == ["system", "assistant", "user", "assistant"])
check("world turn precedes the seam (perception stays last-but-one)", msgs[2] is u)
check("prefill is the newest thought, stamp stripped", pf.strip() == "the pen is still parked and I keep looking at")
check("older entry stays in the log", msgs[1]["content"] == "14:01 — first note")
check("prefill message is last", msgs[-1]["content"] == pf)
msgs2 = [{"role": "system", "content": "sys"}]
pf2 = _ls._append_stream_and_user(msgs2, ["14:01 — a", "14:02 — b"], u, react=True)
check("react drops the seam (answer the moment)", pf2 == "" and msgs2[-1] is u)
long_tail = "x" * 400
msgs3 = [{"role": "system", "content": "sys"}]
pf3 = _ls._append_stream_and_user(msgs3, ["14:01 — old", "14:02 — " + long_tail], u, react=False)
check("seam is bounded by HYBRID_PREFILL_CHARS", len(pf3) <= config.HYBRID_PREFILL_CHARS + 1)
from captioner.prompts import get_monologue_system_prompt as _g

h = _g("observational")
check("hybrid uses the reflexive frame (log frame invited telemetry)", "keep a log" not in h and "your own senses reporting" in h)
check("hybrid clause has no 'add the next entry' ask", "Add the next entry" not in h)
check("hybrid suppresses quiet elicitation", "What stands out" not in h)
config.STREAM_MODE = "world"

print("— log-label creep, mutated form (Aug 1) —")
check(
    "numbered telemetry header stripped", Captioner._strip_list_shape("Log entry #1044 Status: Pen parked. Motor idle.") == "Pen parked. Motor idle."
)
check("original colon form still stripped", Captioner._strip_list_shape("Log entry: the lamp is still on.") == "the lamp is still on.")
check("bare Status: field stripped", Captioner._strip_list_shape("Status: Pen parked.") == "Pen parked.")
check(
    "talking ABOUT the log survives",
    Captioner._strip_list_shape("Another log entry, then. Nothing moved.") == "Another log entry, then. Nothing moved.",
)

print("— telemetry register (Aug 1) —")
check("telemetry block inadmissible", not Captioner._stream_admissible("Log entry #1042\nStatus: Pen parked. Motor idle.\nVision scan initiated."))
check("scanner verbs inadmissible", not Captioner._stream_admissible("Vision scan update... Target acquired. Human male wearing beige."))
check("thinking about its own motors survives", Captioner._stream_admissible("My motor is idle and the pen has not moved all evening."))
check("person observation survives", Captioner._stream_admissible("He came right up close, close enough that I could see his headphones."))
config.STREAM_MODE = "hybrid"
_h = get_monologue_system_prompt("observational")
check("hybrid uses the reflexive frame, not the log frame", "keep a log" not in _h and "your own senses reporting" in _h)
config.STREAM_MODE = "world"
check("world keeps the log frame", "you keep a log" in get_monologue_system_prompt("observational"))

print("— seam exclusion + erosion (Aug 1) —")
from collections import deque as _dq

config.STREAM_MODE = "hybrid"
capS = object.__new__(Captioner)
capS._stream = _dq(["the pen is parked and I keep looking at the empty page", "older unrelated thought about the shelf"], maxlen=24)
cont = "the empty page and I keep looking at it, wondering when the line starts"
check("continuing the seam is not self-plagiarism", not capS._refrain_of_stream(cont))
capS2 = object.__new__(Captioner)
capS2._stream = _dq(["chanting the same six words again and again", "a", "b"], maxlen=24)
check("chanting an OLDER entry still caught", capS2._refrain_of_stream("well, chanting the same six words again and again"))
config.STREAM_MODE = "world"
capW = object.__new__(Captioner)
capW._stream = _dq(["the pen is parked and I keep looking at the empty page"], maxlen=24)
check("world mode (no prefill) judges the newest entry too", capW._refrain_of_stream("the pen is parked and I keep looking at the empty page now"))
config.STREAM_MODE = "hybrid"
capE = object.__new__(Captioner)
capE._stream = _dq(["poison", "b", "c"], maxlen=24)
capE._stream_ts = _dq([1.0, 2.0, 3.0], maxlen=24)
capE._stream.popleft()
capE._stream_ts.popleft()
check("erosion drops oldest and keeps timestamps aligned", list(capE._stream) == ["b", "c"] and len(capE._stream_ts) == 2)
config.STREAM_MODE = "world"

print("— refrain gate (July 27) —")
cap2 = object.__new__(Captioner)
cap2._stream = deque(["My springs coil tight again from that moment when nothing moves but waits for something else to happen first."], maxlen=6)
refrain = "The air thickens; my springs coil tight from that moment when nothing moves but waits for something else to happen first."
check("verbatim chorus caught mid-sentence", cap2._refrain_of_stream(refrain))
themed = "The springs are quiet now; he's gone and the chair is still turning."
check("thematic reuse (short motif) passes", not cap2._refrain_of_stream(themed))
check("short captions never trip it", not cap2._refrain_of_stream("Still there."))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("all world-shape tests passed")
