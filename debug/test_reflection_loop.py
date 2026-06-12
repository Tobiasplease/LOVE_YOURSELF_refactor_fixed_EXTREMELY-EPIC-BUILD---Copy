"""End-to-end test of the reflection loop (captioner/reflection.py).

Exercises: subject rotation, context gathering, prompt build, the real model
call (starts llama-server if it isn't running), ChromaDB storage, and
retrieval back into a caption via the echo line.

Run with a scratch folder so nothing lands in real memory:

    MOOD_SNAPSHOT_FOLDER=/tmp/reflect_test python debug/test_reflection_loop.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from captioner.reflection import ReflectionLoop
from captioner.semantic_memory import get_semantic_memory


class FakeAgent:
    first_caption_done = True
    _last_scene_motion = False
    last_caption = "The cables under the desk again. They haven't moved."
    boredom = 0.3

    def _is_currently_drawing(self):
        return False


def main():
    agent = FakeAgent()
    loop = ReflectionLoop(agent)

    print("=== Subject rotation (6 picks, gated subjects may be skipped) ===")
    for _ in range(6):
        subject, _q = loop._next_subject()
        print(f"  -> {subject}")

    print("\n=== Trigger logic ===")
    print(f"  should_reflect right after boot (interval not elapsed): {loop._should_reflect()}")
    loop.last_reflection_time = time.time() - 10_000
    print(f"  should_reflect with elapsed interval + quiet scene:     {loop._should_reflect()}")
    agent._last_scene_motion = True
    print(f"  should_reflect with busy scene (should defer):          {loop._should_reflect()}")
    agent._last_scene_motion = False

    sm = get_semantic_memory()
    before = sm._reflections.count()
    print(f"\n=== Reflecting for real (reflections stored before: {before}) ===")
    loop._reflect()
    after = sm._reflections.count()
    print(f"reflections stored after: {after}")

    if after <= before:
        print("FAIL: no reflection was stored")
        return 1

    recent = sm.get_recent_reflections(limit=1)
    print(f"\n=== Stored reflection ({recent[0]['subject']}) ===\n{recent[0]['text']}\n")

    print("=== Echo retrieval into a quiet caption ===")
    from captioner.prompts import get_reflection_echo_line

    agent._reflection_echo_counter = 3  # next call hits the every-4th gate
    line = get_reflection_echo_line(agent)
    print(f"  {line or '(no echo — relevance below threshold)'}")

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
