"""Thought-shaped share of the stream, measured from a run's event log.

The Sep 3 target: ~15-20% of stream entries thought-shaped (drift turns +
reflection kernels; kernels alone were ~1% — no genre moves at that dose).
Stored ordinary captions are estimated as caption_generated minus the two
logged unstored outcomes (echo_spoken_not_stored, runon_not_stored) —
admissibility rejects are unlogged and rare, so the share is a slight
underestimate. Drift fire rate uses caption_request as the cycle count
(it logs before the drift roll).

Run: python debug/drift_share.py [event_log/<run>-event-log.json]
     (defaults to the newest run log)
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def newest_log():
    logs = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "event_log", "*-event-log.json")), key=os.path.getmtime)
    return logs[-1] if logs else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest_log()
    if not path or not os.path.exists(path):
        print("no event log found")
        sys.exit(1)

    counts = {}
    drift_stored = 0
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            a = e.get("action")
            if a:
                counts[a] = counts.get(a, 0) + 1
                if a == "drift_turn" and e.get("stored"):
                    drift_stored += 1

    cycles = counts.get("caption_request", 0)
    drifts = counts.get("drift_turn", 0)
    kernels = counts.get("kernel_to_stream", 0)
    stored_captions = max(0, counts.get("caption_generated", 0) - counts.get("echo_spoken_not_stored", 0) - counts.get("runon_not_stored", 0))
    thought = drift_stored + kernels
    total = stored_captions + thought

    print(f"run log: {os.path.basename(path)}")
    print(f"  cycles (caption_request):        {cycles}")
    print(f"  drift turns fired / stored:      {drifts} / {drift_stored}")
    print(f"  reflection kernels admitted:     {kernels}")
    print(f"  stored ordinary captions (est.): {stored_captions}")
    if cycles:
        print(f"  drift fire rate:                 {100 * drifts / cycles:.1f}% of cycles")
    if total:
        print(f"  thought-shaped stream share:     {100 * thought / total:.1f}%  (target ~15-20%)")
    else:
        print("  no stream entries counted yet")


if __name__ == "__main__":
    main()
