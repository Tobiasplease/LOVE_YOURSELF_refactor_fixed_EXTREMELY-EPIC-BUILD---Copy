"""The place belief (Sep 7, phase 1 of docs/plan-place-and-unknowns-sep7.md).
Harvested from the room reflection like the NAME is; never generated per
caption. Run: python debug/test_place.py"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("STREAM_MODE", "mind")
from captioner import prompt_registry as R  # noqa: E402
from captioner.context_compression import context_compressor as CC  # noqa: E402
from config import config as C  # noqa: E402
from utils.lore_ledger import LoreLedger  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + str(detail)) if (detail and not ok) else ''}")
    if not ok:
        FAILS.append(name)


L = LoreLedger(os.path.join(tempfile.mkdtemp(), "lore.json")) if "path" in LoreLedger.__init__.__code__.co_varnames else LoreLedger()
try:
    L._data = {"reveries": [], "threads": [], "name": None, "name_history": [], "questions": [], "place": None, "place_history": []}
    L._save = lambda: None
except Exception:
    pass

print("\n[1] the store")
check("a few words are accepted", L.note_place("a workshop for building bodies") and L.current_place()["place"] == "a workshop for building bodies")
check("saying the same thing again affirms rather than replaces", L.note_place("A workshop for building bodies") and L.current_place().get("times_affirmed") == 1)
check("a different answer revises and keeps the old", L.note_place("a storage room someone works in") and L.current_place()["place"] == "a storage room someone works in" and len(L._data["place_history"]) == 1)
check("a sentence is refused", not L.note_place("I think this is a room where people build things."))
check("a long answer is refused", not L.note_place("a place that is very much like a workshop and also a studio and a store"))
check("'none' is refused", not L.note_place("none"))
check("an object already in the room is refused (inventory, not place)", not L.note_place("wooden chair", known_terms=["wooden chair", "red foam finger"]))

print("\n[2] the distillation slot")
parsed = CC._parse_distillation("TRAIT — none\nKERNEL — something.\nNAME — none\nPLACE — a robotics workshop\nQUESTION — none")
check("PLACE parses as the 11th slot", parsed[-1] == "a robotics workshop", parsed[-1])
check("'none' in PLACE yields nothing", CC._parse_distillation("PLACE — none")[-1] == "")
check("the distill prompt asks for it, harvest-only", "PLACE — if in this reflection you said what kind of place this is" in R.P("distill.user"))

print("\n[3] the invite and the frame")
check("the invite exists and offers no candidates", "what kind of place this is" in R.P("reflection.place-invite") and "workshop" not in R.P("reflection.place-invite").lower())
check("the invite offers a way out", "leave it" in R.P("reflection.place-invite"))
rsrc = open("captioner/reflection.py", encoding="utf-8").read()
check("it rides on the ROOM subject only", 'if subject == "the room":' in rsrc and "reflection.place-invite" in rsrc)
check("the standing belief reaches the room reflection", "place_standing" in rsrc and "place_standing" in open("captioner/prompts.py", encoding="utf-8").read())
check("config knobs exist", hasattr(C, "PLACE_INVITE_EVERY_S") and hasattr(C, "PLACE_REASK_EVERY_S") and hasattr(C, "MIND_ROOM_TERMS_WITH_PLACE"))

print("\n[4] the life block")
msrc = open("captioner/mind.py", encoding="utf-8").read()
check("the place rides in what it knows", 'P("mind.life-place")' in msrc)
check("and the object list shortens when it stands", "MIND_ROOM_TERMS_WITH_PLACE" in msrc)
check("the fragment is framed as its own conclusion", "come to think of this place as" in R.P("mind.life-place"))

print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
