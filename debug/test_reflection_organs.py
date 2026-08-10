"""Dry-run the five reflection organs and show that their bundles differ.

Piece 1 of docs/reflection-organs-handover.md turned each reflection subject
from a question over identical data into an organ with its own slice of
memory. This builds every subject's bundle and prompt WITHOUT querying the
model, so divergence is inspectable in seconds instead of the ~100 minutes a
full rotation takes live.

    python debug/test_reflection_organs.py            # summary + block diff
    python debug/test_reflection_organs.py --full     # print whole prompts
    python debug/test_reflection_organs.py --arc      # allow the artistic-arc
                                                      # LLM call (off by default)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.prompts import REFLECTION_SUBJECTS, build_reflection_loop_prompt, get_reflection_system_prompt  # noqa: E402
from captioner.reflection import ReflectionLoop  # noqa: E402


class _StubAgent:
    first_caption_done = True
    _last_scene_motion = False

    def _is_currently_drawing(self):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="print each prompt in full")
    ap.add_argument("--arc", action="store_true", help="allow the artistic-arc LLM call in the drawings diet")
    args = ap.parse_args()

    loop = ReflectionLoop(_StubAgent())

    if not args.arc:
        from drawing.drawing_memory import DrawingMemory

        DrawingMemory.get_artistic_arc = lambda self: ""
        print("(artistic arc stubbed out — pass --arc to include the real LLM call)\n")

    spine = loop._gather_spine()
    print("=== SPINE ===")
    print(f"  hour entries : {len(spine.get('hour') or [])}")
    print(f"  today entries: {len(spine.get('today') or [])}")
    print(f"  journal      : {len(spine.get('journal') or [])}")
    print(f"  prior reflections exist: {spine.get('lived')}")
    print(f"  has lived    : {loop._has_lived(spine)}\n")

    if not loop._has_lived(spine):
        print("Nothing lived yet — the live loop would postpone. Run the machine a while first.")
        return 1

    bundles = {}
    for subject, question in REFLECTION_SUBJECTS:
        loop._drawings_cache = None
        data = loop._gather_context(subject, spine)
        prompt = build_reflection_loop_prompt(question, data)
        system = get_reflection_system_prompt(subject)
        bundles[subject] = (data, prompt, system)

    print("=== ORGAN DIETS ===")
    shared = set.intersection(*(set(d) for d, _, _ in bundles.values())) if bundles else set()
    for subject, (data, prompt, system) in bundles.items():
        own = sorted(k for k in data if k not in shared and data[k])
        empty = sorted(k for k in data if not data[k])
        print(f"\n{subject}")
        print(f"  own keys with material : {', '.join(own) or '(none)'}")
        if empty:
            print(f"  keys that came up empty: {', '.join(empty)}")
        print(f"  own reflection thread  : {len(data.get('reflections') or [])} prior on this subject")
        print(f"  prompt length          : {len(prompt)} chars")
        print(f"  identity in system     : {'yes' if 'come to know about yourself' in system else 'no'}")
        print(f"  durable in system      : {'yes' if 'stayed true across days' in system else 'no'}")

    print(f"\nShared across all five: {', '.join(sorted(shared)) or '(none)'}")

    print("\n=== OVERLAP (how much of each prompt is identical to another's) ===")
    names = list(bundles)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            blocks_a = set(bundles[a][1].split("\n\n"))
            blocks_b = set(bundles[b][1].split("\n\n"))
            common = blocks_a & blocks_b
            chars = sum(len(c) for c in common)
            total = min(len(bundles[a][1]), len(bundles[b][1])) or 1
            print(f"  {a:14s} vs {b:14s}: {len(common)} identical blocks, {100 * chars // total}% of the shorter prompt")

    if args.full:
        for subject, (_, prompt, system) in bundles.items():
            print(f"\n\n{'=' * 70}\n{subject.upper()}\n{'=' * 70}")
            print(f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{prompt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
