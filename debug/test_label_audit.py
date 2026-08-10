"""Standalone check of the label-audit head-to-head (no VLM call, no threads).

Recreates the artist-reported mislabels on real saved frames — the styrofoam
heads claimed by "wooden mannequin torso", a cable bundle claimed by "wire
basket" — and runs the audit's CLIP head-to-head with plausible VLM candidate
names to see whether the correction would fire.

Usage:
    python debug/test_label_audit.py
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.label_audit import LabelAuditThread

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    {
        "name": "styrofoam heads (top shelf)",
        "frame": "event_log/ccd4cfbc-images/mood_1785928841.jpg",
        "box": (650, 30, 790, 140),
        "incumbent": "wooden mannequin torso",
        "vlm_candidates": ["white styrofoam head", "polystyrene head", "foam head"],
    },
    {
        "name": "cable bundle (wall, right)",
        "frame": "event_log/873a6770-images/mood_1785938232.jpg",
        "box": (990, 165, 1130, 300),
        "incumbent": "wire basket",
        "vlm_candidates": ["coiled black cable", "bundle of cables", "black cable"],
    },
]


def main():
    auditor = LabelAuditThread(detector=None)

    reply = "1. White styrofoam head\n2. Polystyrene head.\n- foam head\n"
    parsed = auditor._parse_candidates(reply, exclude="wooden mannequin torso")
    print(f"parse check: {parsed}\n")

    for case in CASES:
        img = cv2.imread(os.path.join(REPO, case["frame"]))
        x1, y1, x2, y2 = case["box"]
        crop = img[y1:y2, x1:x2]
        ok, jpg = cv2.imencode(".jpg", crop)
        out = os.path.join(REPO, "debug", f"audit_crop_{case['name'].split()[0]}.jpg")
        cv2.imwrite(out, crop)

        terms = [case["incumbent"]] + case["vlm_candidates"]
        scores = auditor._head_to_head(jpg.tobytes(), terms)
        print(f"{case['name']}  (crop saved to {os.path.basename(out)})")
        for t in terms:
            marker = " <- incumbent" if t == case["incumbent"] else ""
            print(f"   {scores.get(t, 0.0):.2f}  {t}{marker}")
        challenger = max(case["vlm_candidates"], key=lambda c: scores.get(c, 0.0))
        margin = scores.get(challenger, 0.0) - scores.get(case["incumbent"], 0.0)
        print(f"   verdict: {'RELABEL -> ' + challenger if margin > 0.08 else 'held'} (margin {margin:+.2f})\n")


if __name__ == "__main__":
    main()
