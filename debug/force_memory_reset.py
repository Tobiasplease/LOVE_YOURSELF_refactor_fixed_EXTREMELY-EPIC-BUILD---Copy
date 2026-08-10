#!/usr/bin/env python3
"""Complete memory reset — wipe everything the machine has accumulated.

Covers ALL persistent memory (June 2026 audit; re-audited Aug 1 2026):
  - machine_identity.json   persona (core_facts.self), desires, beliefs, journal
  - chromadb/               concepts, observations, reflections
  - drawing_memory.json     past drawing summaries
  - durable_ledger.json     cross-day facts  (MISSED until Aug 1)
  - episodic_events.json    arrivals/departures/drew  (MISSED until Aug 1)
  - activation_snapshot.json visualizer state  (MISSED until Aug 1)
  - system_state.json, lifetime_state.json, last_session.txt, last_caption.txt
  - live_captions.txt       running caption mirror

The three MISSED stores all postdate the June audit. Leaving the durable
ledger behind was the consequential one — it survives a "complete" reset and
its facts re-promote into the reflection frame, so a supposedly blank machine
resumed asserting the identity it was wiped to forget.

STOP machine.py before running this — on shutdown it writes a journal entry
and re-saves identity, which would partially undo the reset.

Usage:
    python debug/force_memory_reset.py           # asks for confirmation
    python debug/force_memory_reset.py --yes     # no confirmation
    python debug/force_memory_reset.py --backup  # move aside instead of delete
"""

import os
import shutil
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER


def force_memory_reset(backup: bool = False):
    print("Complete Memory Reset" + (" (backup mode — moving aside, not deleting)" if backup else ""))
    print("=" * 50)
    print(f"Memory root: {MOOD_SNAPSHOT_FOLDER}\n")

    files = [
        "machine_identity.json",
        "drawing_memory.json",
        "durable_ledger.json",
        "episodic_events.json",
        "activation_snapshot.json",
        "system_state.json",
        "lifetime_state.json",
        "last_session.txt",
        "last_caption.txt",
        "live_captions.txt",
    ]
    dirs = [
        "chromadb",
    ]
    # The durable ledger resolves relative to the repo root, not the memory
    # root (captioner/durable_ledger.py) — check both so a non-default
    # MOOD_SNAPSHOT_FOLDER can't leave it behind.
    extra = [os.path.join("event_log", "durable_ledger.json")]

    suffix = time.strftime("wipe-bak-%Y%m%d_%H%M%S")
    removed = 0

    def _clear(path, label):
        nonlocal removed
        if backup:
            shutil.move(path, f"{path}.{suffix}")
            print(f"Moved aside: {label} -> {os.path.basename(path)}.{suffix}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {label}/")
        else:
            os.remove(path)
            print(f"Removed: {label}")
        removed += 1

    for name in files:
        path = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
        if os.path.exists(path):
            _clear(path, name)
        else:
            print(f"Not found (already clean): {name}")

    for name in dirs:
        path = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
        if os.path.isdir(path):
            _clear(path, name)
        else:
            print(f"Not found (already clean): {name}/")

    for path in extra:
        if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(os.path.join(MOOD_SNAPSHOT_FOLDER, os.path.basename(path))):
            _clear(path, path)

    print("\n" + "=" * 50)
    if removed:
        print(f"SUCCESS: cleared {removed} items. Next startup is a true first awakening:")
        print("- no persona ('What you've come to know about yourself' line absent)")
        print("- no desires, beliefs, journal, core facts")
        print("- no concepts/observations/reflections in ChromaDB")
        print("- no durable facts, no episodic arrivals, no drawing history")
        print("- no last thought; FIRST_AWAKENING_PROMPT path will be used")
    else:
        print("INFO: nothing to clear (already clean)")
    print("=" * 50)


if __name__ == "__main__":
    backup = "--backup" in sys.argv
    if "--yes" not in sys.argv:
        verb = "moves aside" if backup else "PERMANENTLY wipes"
        answer = input(f"This {verb} ALL accumulated memory. Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)
    force_memory_reset(backup=backup)
