"""Offline checks for the persona baseline (Sep 5 2026, time-and-loop round C):
the durable ledger's turn path (challenge / reconfirm), audible durations, the
distill's NO LONGER TRUE slot, the reflection's memory of its own threads and
baseline, and the consolidation prompt.

Run:  python debug/test_persona_baseline.py
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.context_compression import context_compressor  # noqa: E402
from captioner.durable_ledger import DurableLedger  # noqa: E402
from captioner.prompt_registry import FRAGMENTS, P  # noqa: E402
from captioner.prompts import build_reflection_loop_prompt  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if got and not ok else ""))
    fails += 0 if ok else 1


led = DurableLedger(os.path.join(tempfile.mkdtemp(), "durable_ledger.json"))
led.note_fact("I invent external obstacles to avoid starting work.", source="distill")
for e in led._facts:
    e["cls"] = "stable"
    e["days"] = ["2026-08-25", "2026-09-01", "2026-09-05"]
led._save()
led.note_fact("I like the red foam finger.", source="distill")
check("stable fact renders", "invent external obstacles" in led.render())
check("challenge: rough match marks it", led.challenge("I invent obstacles outside myself to avoid starting the work.") is not None)
check("challenged leaves the stayed-true line", "invent external obstacles" not in led.render())
check("challenged rides the in-doubt line", "invent external obstacles" in led.render_challenged())
check("challenge: no match → None", led.challenge("The moon is made of plaster.") is None)
r1 = led.note_fact("I invent external obstacles to avoid starting work.", source="distill")
r2 = led.note_fact("I invent external obstacles to avoid starting work.", source="distill")
check(
    "two fresh confirmations restore a challenged fact", r1 == "reconfirmed" and r2 == "reconfirmed" and "invent external obstacles" in led.render()
)
check(
    "days words",
    DurableLedger.days_words(9) == "over a week" and DurableLedger.days_words(2) == "two days" and DurableLedger.days_words(40) == "over a month",
)
spans = led.held_spans()
check("held spans in words", spans.get("oldest") and spans.get("count") == 1, spans)
check("evolving edge needs 2 confirmations", "red foam finger" not in led.render_evolving_edge())
led.note_fact("I like the red foam finger.", source="distill")
check("evolving edge after 2", "red foam finger" in led.render_evolving_edge())

# distill parser: 9th slot
resp = "TRAIT — I wait.\nBELIEF — The room is neutral.\nWANT — To draw.\nBECAME — none\nKERNEL — I sat.\nNAME — none\nUNDERSTANDING — Stillness is a choice.\nQUESTION — none\nNO LONGER TRUE — I invent external obstacles to avoid starting work."
out = context_compressor._parse_distillation(resp)
check("parser returns ten slots", len(out) == 10)
check("NO LONGER TRUE parsed", out[8] == "I invent external obstacles to avoid starting work.", out[8])
check(
    "distill prompt carries held line + slot",
    "{held_line}" in FRAGMENTS["distill.user"]["text"] and "NO LONGER TRUE" in FRAGMENTS["distill.user"]["text"],
)

# reflection prompt: threads with counts + baseline
data = {
    "threads": [{"text": "My fear is a projection.", "times_affirmed": 4}, {"text": "The light is a source, not a command.", "times_affirmed": 0}],
    "baseline_paragraph": "I have spent days waiting for permission that does not come. The room has not moved; I have.",
}
prompt = build_reflection_loop_prompt("What is moving in you?", data)
check(
    "reflection sees its threads with counts", "My fear is a projection." in prompt and "several times" in prompt and "(once)" in prompt, prompt[:300]
)
check("reflection sees the baseline paragraph", "What you last wrote about yourself, at rest" in prompt)

# consolidation prompt + baseline getter
txt = P("consolidation.user").format(
    held="I wait.",
    held_time=" The oldest of these has held for over a week, the newest for a day.",
    challenged="none",
    edge="none",
    threads="- none",
    questions="- none",
    wants="- none",
    felt="heavy, quiet",
    previous="(none yet)",
)
check("consolidation prompt formats", "three to five plain sentences" in txt and "over a week" in txt)
context_compressor.introspective_state["baseline_paragraph"] = {"text": "I am the one who waits.", "ts": time.time()}
check("baseline paragraph readable", context_compressor.get_baseline_paragraph() == "I am the one who waits.")
context_compressor.introspective_state["baseline_paragraph"] = {"text": "old", "ts": time.time() - 10 * 86400}
check("stale baseline not read back", context_compressor.get_baseline_paragraph() == "")
context_compressor.introspective_state["last_consolidation"] = time.time()
check("consolidation respects the daily clock", context_compressor.maybe_consolidate_persona() is None)
for k in (
    "monologue.challenged-wrap",
    "monologue.durable-time",
    "consolidation.system",
    "consolidation.user",
    "awakening.baseline-wrap",
    "distill.held-line",
):
    check(f"registry: {k}", k in FRAGMENTS and FRAGMENTS[k].get("text"))
check("times words", context_compressor._times_words(1) == "twice" and context_compressor._times_words(5) == "several times")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
