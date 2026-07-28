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
