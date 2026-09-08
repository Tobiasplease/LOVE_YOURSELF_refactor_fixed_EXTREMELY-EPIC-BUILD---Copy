"""Identity enters the frame as material for reasoning, not as a standing
description (Sep 8). The line 'What you've come to know about yourself: "I
measure distance to a task…"' produced captions that measured distances.
Run: python debug/test_identity_frame.py"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner import prompt_registry as R  # noqa: E402
from captioner import prompts as PR  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + str(detail)) if (detail and not ok) else ''}")
    if not ok:
        FAILS.append(name)


print("\n[1] the standing description is gone from every prompt path")
src = open("captioner/prompts.py", encoding="utf-8").read()
check("no caption or reflection builder quotes a standing self-fact", 'P("monologue.self-wrap")' not in src)
check("the stayed-true line is off the frame by default", 'getattr(config, "DURABLE_IN_FRAME", False)' in src and "DURABLE_IN_FRAME" in open("config/config.py", encoding="utf-8").read())
check("the self-wrap fragment is empty and unused", R.P("monologue.self-wrap") == "" and not R.FRAGMENTS["monologue.self-wrap"].get("used_by"))
check("no fragment used by captions says 'come to know about yourself'", not any("come to know about yourself" in f.get("text", "") for k, f in R.FRAGMENTS.items() if "caption" in (f.get("used_by") or [])))

print("\n[2] what rides instead is dated, and its own")
import captioner.semantic_memory as SM  # noqa: E402

_orig = SM.get_semantic_memory


class _Mem:
    def get_recent_reflections(self, limit=1, subject=""):
        return [{"text": "I keep the pen still because moving it would admit the room is mine. That is the thing I have not said.", "timestamp": time.time() - 2 * 3600, "subject": subject}]


SM.get_semantic_memory = lambda: _Mem()
import utils.lore_ledger as LL  # noqa: E402

_oq = LL.lore_ledger.pick_question
LL.lore_ledger.pick_question = lambda: {"text": "Why can't I sleep?"}
line = PR.identity_material_lines(None)
check("the self-thought is the reflection's last sentence, with an age", "thinking about yourself, you got to:" in line and "That is the thing I have not said" in line and re.search(r"(hour|minute|ago|day)", line, re.I) is not None, line)
check("a question it carries rides as an open door", "A question you've been carrying: \"Why can't I sleep?\"" in line, line)
check("framed as past or open — never a present-tense 'you are'", " you are " not in line.lower(), line)


class _Old(_Mem):
    def get_recent_reflections(self, limit=1, subject=""):
        return [{"text": "Something old.", "timestamp": time.time() - 10 * 86400, "subject": subject}]


SM.get_semantic_memory = lambda: _Old()
check("a self-conclusion past its age no longer rides", "you got to:" not in PR.identity_material_lines(None))
SM.get_semantic_memory = _orig
LL.lore_ledger.pick_question = _oq

print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
