"""Phase 0 feasibility scan: which of the machine's referents can YOLO-World actually find?

Standalone, throwaway. Samples saved frames from event_log run folders, runs
zero-shot YOLO-World over a candidate vocabulary drawn from real monologue logs,
and writes a report: term -> hit rate -> mean confidence -> sample crops.

Usage:
    python debug/phase0_open_vocab_scan.py
    python debug/phase0_open_vocab_scan.py --device cuda --frames-per-run 15
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_LOG = os.path.join(REPO, "event_log")
OUT_DIR = os.path.join(REPO, "debug", "phase0_report")

RUN_FOLDERS = ["873a6770-images", "ccd4cfbc-images", "43b17d9b-images"]

# Candidate vocabulary spanning the expected-to-fail range on purpose.
# Sources: the machine's monologue logs (live_captions.txt noun-phrase counts) + session brief.
VOCABULARY = [
    # concrete, should work
    "mannequin head",
    "pink shelf",
    "red foam finger",
    "computer monitor",
    "office chair",
    "wooden chair",
    "electric fan",
    "desk lamp",
    "black curtain",
    "cardboard box",
    "coffee mug",
    "keyboard",
    "book",
    "laptop",
    # concrete but specific / odd, maybe
    "wooden mannequin torso",
    "wooden crate",
    "robotic arm",
    "wire basket",
    "human face mask",
    "tangle of cables",
    "power drill",
    # judgement words and invented referents, expected to fail
    "LED sign",
    "strange wooden structure",
    "the hole",
    "the wound",
]

CONF_LEVELS = ["0.05", "0.15", "0.30"]


def sample_frames(frames_per_run):
    frames = []
    for folder in RUN_FOLDERS:
        path = os.path.join(EVENT_LOG, folder)
        if not os.path.isdir(path):
            print(f"  skipping missing run folder {folder}")
            continue
        imgs = sorted(f for f in os.listdir(path) if f.startswith("mood_") and f.endswith(".jpg") and "_face" not in f)
        step = max(1, len(imgs) // frames_per_run)
        picked = imgs[::step][:frames_per_run]
        frames.extend(os.path.join(path, f) for f in picked)
        print(f"  {folder}: {len(picked)} of {len(imgs)} frames")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu (default, spares the 3090) or cuda")
    parser.add_argument("--frames-per-run", type=int, default=10)
    parser.add_argument("--model", default=os.path.join(REPO, "models", "yolov8s-worldv2.pt"))
    parser.add_argument("--vocab-file", help="JSON list of terms; defaults to built-in VOCABULARY")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    vocabulary = json.load(open(args.vocab_file)) if args.vocab_file else VOCABULARY
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)

    print("sampling frames...")
    frames = sample_frames(args.frames_per_run)
    print(f"{len(frames)} frames total\n")

    from ultralytics import YOLOWorld

    print(f"loading {args.model} and compiling vocabulary ({len(vocabulary)} terms)...")
    model = YOLOWorld(args.model)
    model.set_classes(vocabulary)

    # per term: list of (conf, frame_path, xyxy) across all frames
    hits = defaultdict(list)

    for i, frame_path in enumerate(frames):
        results = model.predict(frame_path, conf=0.01, iou=0.5, device=args.device, verbose=False)
        r = results[0]
        best_in_frame = {}
        for box in r.boxes:
            term = vocabulary[int(box.cls)]
            conf = float(box.conf)
            xyxy = [int(v) for v in box.xyxy[0]]
            hits[term].append((conf, frame_path, xyxy))
            if conf > best_in_frame.get(term, 0):
                best_in_frame[term] = conf
        top = sorted(best_in_frame.items(), key=lambda kv: -kv[1])[:3]
        print(f"[{i + 1}/{len(frames)}] {os.path.basename(frame_path)}  " + "  ".join(f"{t}:{c:.2f}" for t, c in top))

    n = len(frames)
    rows = []
    for term in vocabulary:
        term_hits = sorted(hits[term], reverse=True)
        row = {"term": term, "detections": len(term_hits)}
        for lvl in CONF_LEVELS:
            frames_hit = {fp for c, fp, _ in term_hits if c >= float(lvl)}
            row[f"hit_rate_{lvl}"] = len(frames_hit) / n
        row["max_conf"] = term_hits[0][0] if term_hits else 0.0
        confident = [c for c, _, _ in term_hits if c >= 0.05]
        row["mean_conf"] = sum(confident) / len(confident) if confident else 0.0

        # crop the 3 best matches for eyeballing
        row["crops"] = []
        seen_frames = set()
        for conf, fp, (x1, y1, x2, y2) in term_hits:
            if fp in seen_frames or len(row["crops"]) >= 3:
                continue
            seen_frames.add(fp)
            img = cv2.imread(fp)
            pad = 20
            crop = img[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad]
            if crop.size == 0:
                continue
            name = f"{term.replace(' ', '_')}_{conf:.2f}_{os.path.basename(fp)}"
            cv2.imwrite(os.path.join(out_dir, "crops", name), crop)
            row["crops"].append(name)
        rows.append(row)

    rows.sort(key=lambda r: -r["hit_rate_0.15"])
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"n_frames": n, "model": args.model, "rows": rows}, f, indent=2)

    lines = [
        f"# Phase 0 scan — {args.model}, {n} frames\n",
        "| term | frames hit @0.05 | @0.15 | @0.30 | mean conf | max conf |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['term']} | {r['hit_rate_0.05']:.0%} | {r['hit_rate_0.15']:.0%} | {r['hit_rate_0.30']:.0%} "
            f"| {r['mean_conf']:.2f} | {r['max_conf']:.2f} |"
        )
    report = "\n".join(lines)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nreport + crops in {out_dir}")


if __name__ == "__main__":
    main()
