#!/usr/bin/env python3
"""Complete memory reset — wipe everything the machine has accumulated.

Covers ALL persistent memory (June 2026 audit):
  - machine_identity.json   persona (core_facts.self), desires, beliefs, journal
  - chromadb/               concepts, observations, reflections
  - drawing_memory.json     past drawing summaries
  - system_state.json, lifetime_state.json, last_session.txt, last_caption.txt
  - live_captions.txt       running caption mirror

STOP machine.py before running this — on shutdown it writes a journal entry
and re-saves identity, which would partially undo the reset.

Usage:
    python debug/force_memory_reset.py          # asks for confirmation
    python debug/force_memory_reset.py --yes    # no confirmation
"""

import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MOOD_SNAPSHOT_FOLDER


def force_memory_reset():
    print("Complete Memory Reset")
    print("=" * 50)
    print(f"Memory root: {MOOD_SNAPSHOT_FOLDER}\n")

    files = [
        "machine_identity.json",
        "drawing_memory.json",
        "system_state.json",
        "lifetime_state.json",
        "last_session.txt",
        "last_caption.txt",
        "live_captions.txt",
    ]
    dirs = [
        "chromadb",
    ]

    removed = 0
    for name in files:
        path = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed: {name}")
            removed += 1
        else:
            print(f"Not found (already clean): {name}")

    for name in dirs:
        path = os.path.join(MOOD_SNAPSHOT_FOLDER, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {name}/")
            removed += 1
        else:
            print(f"Not found (already clean): {name}/")

    print("\n" + "=" * 50)
    if removed:
        print(f"SUCCESS: removed {removed} items. Next startup is a true first awakening:")
        print("- no persona ('What you've come to know about yourself' line absent)")
        print("- no desires, beliefs, journal, core facts")
        print("- no concepts/observations/reflections in ChromaDB")
        print("- no last thought; FIRST_AWAKENING_PROMPT path will be used")
    else:
        print("INFO: nothing to remove (already clean)")
    print("=" * 50)


if __name__ == "__main__":
    if "--yes" not in sys.argv:
        answer = input("This wipes ALL accumulated memory permanently. Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(1)
    force_memory_reset()
