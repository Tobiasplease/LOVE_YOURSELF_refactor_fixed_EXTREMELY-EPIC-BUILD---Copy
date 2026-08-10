"""Standalone check of the Phase 3 spatial registry (no camera, no threads).

Feeds synthetic detections at known positions, verifies the box-center ->
absolute-angle math, EMA convergence, and the glance policy's revisit/explore
split.

Usage:
    python debug/test_spatial_registry.py
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRATCH_STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spatial_registry_test.json")
if os.path.exists(SCRATCH_STATE):
    os.remove(SCRATCH_STATE)

from perception.spatial_registry import SpatialRegistry

FRAME = (720, 1280, 3)


def det(term, box, pan, tilt, conf=0.8, settled=True):
    return {"term": term, "box": box, "pan": pan, "tilt": tilt, "conf": conf, "settled": settled}


def main():
    reg = SpatialRegistry(state_path=SCRATCH_STATE)

    # Object dead-center at pan 90 -> anchor must be exactly 90
    reg.update_from_detections([det("centered thing", (600, 320, 680, 400), 90.0, 90.0)], FRAME)
    a = reg.get_anchor("centered thing")
    print(f"centered box @ pan 90     -> anchor {a[0]:.1f}/{a[1]:.1f}  (want 90/90)")

    # Object at right edge of frame -> anchor ~ pan + 27 (HFOV 60, cx ~ 0.95)
    reg.update_from_detections([det("right-edge thing", (1180, 320, 1260, 400), 90.0, 90.0)], FRAME)
    a = reg.get_anchor("right-edge thing")
    print(f"right-edge box @ pan 90   -> anchor {a[0]:.1f}/{a[1]:.1f}  (want ~117/90)")

    # Object at top of frame -> higher tilt (image y down, tilt up)
    reg.update_from_detections([det("high thing", (600, 10, 680, 90), 90.0, 90.0)], FRAME)
    a = reg.get_anchor("high thing")
    print(f"top-edge box @ tilt 90    -> anchor {a[0]:.1f}/{a[1]:.1f}  (want ~90/106)")

    # Unsettled detections must not move anchors
    reg.update_from_detections([det("centered thing", (0, 0, 100, 100), 50.0, 70.0, settled=False)], FRAME)
    a = reg.get_anchor("centered thing")
    print(f"after unsettled detection -> anchor {a[0]:.1f}/{a[1]:.1f}  (must still be 90/90)")

    # EMA: same object re-seen 5 deg to the right converges toward the new spot
    for _ in range(6):
        reg.update_from_detections([det("centered thing", (600, 320, 680, 400), 95.0, 90.0)], FRAME)
    a = reg.get_anchor("centered thing")
    print(f"after 6 sightings @ 95    -> anchor {a[0]:.1f}/{a[1]:.1f}  (converging toward 95)")

    kinds = Counter()
    terms = Counter()
    for _ in range(400):
        pick = reg.pick_glance_target(explore_weight=0.25)
        kinds[pick["kind"]] += 1
        if pick["term"]:
            terms[pick["term"]] += 1
    print(f"\nglance policy over 400 picks: {dict(kinds)}")
    print(f"revisit distribution: {dict(terms)}")
    print("(all three known objects should appear; ~25% explore)")

    os.remove(SCRATCH_STATE) if os.path.exists(SCRATCH_STATE) else None


if __name__ == "__main__":
    main()
