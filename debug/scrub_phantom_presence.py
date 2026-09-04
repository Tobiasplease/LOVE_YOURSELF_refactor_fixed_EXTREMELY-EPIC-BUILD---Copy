"""One-shot: report (and with --apply, remove) phantom-presence residue from
the persisted stores — reveries in the lore ledger that make a present-tense
third-person claim (the machine "imagining" the artist at the desk after they
had left; 25/40 reveries on Sep 4 evening) and the in-context seeds that
debug/fresh_stream.py clears. Threads, questions, identity, durable facts and
the episodic record are reported only, never touched.

Run BETWEEN machine-stop and machine-start. Refuses to apply while machine.py runs.

Run:  python debug/scrub_phantom_presence.py            (dry run)
      python debug/scrub_phantom_presence.py --apply
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.captioner import Captioner  # noqa: E402
from config.config import MOOD_SNAPSHOT_FOLDER  # noqa: E402

PERSON, ABSENT = Captioner._PHANTOM_PERSON_RE, Captioner._ABSENCE_MARK_RE
apply = "--apply" in sys.argv
LORE = os.path.join(MOOD_SNAPSHOT_FOLDER, "lore_ledger.json")


def phantom(text: str) -> bool:
    return bool(PERSON.search(text or "")) and not ABSENT.search(text or "")


if apply and subprocess.run(["pgrep", "-f", "python machine.py"], capture_output=True).stdout.strip():
    print("machine.py is running — stop it first (this edits its state files)")
    sys.exit(1)

if os.path.exists(LORE):
    lore = json.load(open(LORE))
    rev = lore.get("reveries", []) or []
    bad = [r for r in rev if phantom(r.get("text", ""))]
    print(f"lore reveries: {len(rev)} total, {len(bad)} phantom-presence")
    for r in bad[-5:]:
        print("   -", (r.get("text") or "")[:110].replace("\n", " "))
    for key in ("threads", "questions"):
        items = lore.get(key, []) or []
        print(f"lore {key}: {len(items)} total, {sum(1 for i in items if phantom(i.get('text', '')))} phantom-presence (reported only)")
    if apply and bad:
        lore["reveries"] = [r for r in rev if not phantom(r.get("text", ""))]
        json.dump(lore, open(LORE, "w"), indent=2, ensure_ascii=False)
        print(f"removed {len(bad)} reveries")
else:
    print("no lore ledger")

print("stream tail / last caption / recent_memory: run debug/fresh_stream.py" + (" (not applied here)" if apply else ""))
print("done" if apply else "dry run — add --apply to remove the reveries")
