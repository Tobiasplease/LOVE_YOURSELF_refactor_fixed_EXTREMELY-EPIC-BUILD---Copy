"""Seed + verify the two-factor body schema (place + appearance) on real frames.

Enrolls hand-boxed crops of the machine's own arms (drawing-watch pose, pan 90
tilt 67), then checks the full is_self verdicts: the arm re-seen in a later
frame must be self; the artist standing in the arm's region must not; the
hanging wooden figure (appearance-confusable, 0.87 similarity) must not.

Usage:
    python debug/test_body_schema.py          # dry run against a scratch gallery
    python debug/test_body_schema.py --seed   # verify AND write the real gallery
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.body_schema import BodySchema

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "event_log", "e9658bbb-images")
DRAW_POSE = (90.0, 67.0)  # set_drawing_mode default: pan 90, TILT_MIN + 2

ARM_REFS = [  # frame, box — drawing-watch views of the machine's own limbs
    ("mood_1786314404.jpg", (300, 545, 545, 700)),
    ("mood_1786314404.jpg", (590, 590, 1030, 720)),
    ("mood_1786314404.jpg", (0, 520, 110, 700)),
]
PROBES = [
    ("own hand, later frame (want SELF)", "mood_1786314407.jpg", (300, 545, 545, 700), DRAW_POSE),
    ("own arm, earlier frame (want SELF)", "mood_1786314383.jpg", (590, 590, 1030, 720), DRAW_POSE),
    ("the artist, in arm region (want not-self)", "mood_1786313539.jpg", (800, 270, 1070, 720), DRAW_POSE),
    ("hanging wooden figure (want not-self)", "mood_1786313539.jpg", (0, 240, 100, 530), DRAW_POSE),
    ("styrofoam heads (want not-self)", "mood_1786313539.jpg", (640, 25, 760, 120), DRAW_POSE),
]


def load(frame, box):
    img = cv2.imread(os.path.join(RUN, frame))
    x1, y1, x2, y2 = box
    h, w = img.shape[0], img.shape[1]
    return img[y1:y2, x1:x2], (x1 / w, y1 / h, x2 / w, y2 / h)


def main():
    seed = "--seed" in sys.argv
    state = None if seed else os.path.join(os.path.dirname(os.path.abspath(__file__)), "body_schema_test.json")
    if state and os.path.exists(state):
        os.remove(state)
    schema = BodySchema(state_path=state)

    for frame, box in ARM_REFS:
        crop, norm = load(frame, box)
        schema.add_reference(crop, norm, *DRAW_POSE)
    print(f"enrolled {schema.gallery_size()} arm references at drawing pose {DRAW_POSE}\n")

    ok = True
    for name, frame, box, (pan, tilt) in PROBES:
        crop, norm = load(frame, box)
        verdict, sim = schema.is_self(crop, norm, pan, tilt)
        want_self = "want SELF" in name
        passed = verdict is want_self
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  self={verdict!s:5s} sim={sim:.3f}  {name}")

    print(f"\n{'all verdicts correct' if ok else 'MISCLASSIFICATION — tune thresholds before enabling'}")
    if state and os.path.exists(state):
        os.remove(state)
    if seed:
        print(f"real gallery seeded: {schema.state_path}")


if __name__ == "__main__":
    main()
