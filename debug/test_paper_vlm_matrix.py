#!/usr/bin/env python3
"""
Rigorous VLM paper-check evaluation: labeled scenario matrix.

Interactive session. You physically stage a scenario on the drawing table
(paper down, bare table, clutter, etc.), tell the harness the ground truth,
and it captures several frames — with slight pan jitter, echoing the
production search sweep — running BOTH checks on every frame:
  - production-equivalent ArUco detection (marker visible = no paper)
  - the VLM yes/no question

Everything is appended to debug/paper_vlm_matrix/results.csv (safe to run
multiple sessions — lighting conditions, different days — it accumulates).
On exit it prints per-scenario results, a confusion matrix for each method,
and what a unanimous-vote rule would have decided per scenario.

Suggested scenario checklist (stage each, run it, name it accordingly):
  paper cases:    fresh sheet centered / sheet offset / sheet crooked /
                  already-drawn-on sheet
  no-paper cases: bare table / tools+clutter on table / white-ish non-paper
                  object (cloth, foam, box lid) / crumpled paper ball
  edge cases:     sheet half in drawing area / sheet with marker still
                  peeking out / hand resting on table
  repeat a few key scenarios under exhibition lighting vs studio lighting.

Requires camera free (stop machine.py) and llama-server (auto-starts).

Usage: python debug/test_paper_vlm_matrix.py
"""
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config.config import BAUD_RATE, CAMERA_INDEX, PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT, SERIAL_PORT
from debug.test_paper_vlm_check import PROMPT, aruco_paper_present, parse_verdict
from utils.inference import is_failed_response, query_model

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_vlm_matrix")
CSV_PATH = os.path.join(OUT_DIR, "results.csv")
CSV_FIELDS = ["timestamp", "session", "scenario", "ground_truth", "frame_idx", "pan_offset", "aruco", "vlm", "latency_s", "first_line", "frame_path"]

# Per-scenario capture pattern: pan offsets from the base check angle,
# echoing the production organic sweep so one fixed viewpoint can't flatter
# either method.
PAN_OFFSETS = [0, -4, 4, 0]


def capture_at(cap, servos, pan_offset):
    if servos is not None:
        servos.set_pan(PAPER_DETECTION_GAZE_PAN + pan_offset)
        time.sleep(0.6)
    for _ in range(3):
        cap.read()
    ret, frame = cap.read()
    return frame if ret else None


def ask_vlm(frame):
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None, 0.0, "ENCODE_FAILED"
    t0 = time.time()
    response = query_model(
        PROMPT,
        image=jpg.tobytes(),
        timeout=120,
        options={"temperature": 0.1, "num_predict": 120},
        prompt_type="paper_vlm_matrix",
        skip_generation_wait=True,
    )
    elapsed = time.time() - t0
    if is_failed_response(response):
        return None, elapsed, "QUERY_FAILED"
    return parse_verdict(response), elapsed, response.strip().split("\n")[0]


def run_scenario(cap, servos, session, scenario, ground_truth, writer, csv_file):
    print(f"\n=== '{scenario}'  (ground truth: paper={'YES' if ground_truth else 'NO'}) ===")
    verdicts = {"aruco": [], "vlm": []}
    for idx, offset in enumerate(PAN_OFFSETS):
        frame = capture_at(cap, servos, offset)
        if frame is None:
            print("  lost camera frame, skipping")
            continue
        frame_path = os.path.join(OUT_DIR, f"{session}_{scenario.replace(' ', '_')}_{idx}.jpg")
        cv2.imwrite(frame_path, frame)

        aruco_v = aruco_paper_present(frame)
        vlm_v, latency, first_line = ask_vlm(frame)
        verdicts["aruco"].append(aruco_v)
        verdicts["vlm"].append(vlm_v)

        def mark(v):
            return "?" if v is None else ("✓" if v == ground_truth else "✗ WRONG")

        print(f"  frame {idx} (pan{offset:+d}°): aruco={aruco_v} {mark(aruco_v)}   vlm={vlm_v} {mark(vlm_v)}  ({latency:.1f}s)  {first_line[:50]}")
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "session": session,
                "scenario": scenario,
                "ground_truth": ground_truth,
                "frame_idx": idx,
                "pan_offset": offset,
                "aruco": aruco_v,
                "vlm": vlm_v,
                "latency_s": f"{latency:.1f}",
                "first_line": first_line[:120],
                "frame_path": os.path.basename(frame_path),
            }
        )
        csv_file.flush()

    # What each method would decide for the whole check, fail-closed:
    # draw only if every frame said paper (None counts as no).
    for method, vs in verdicts.items():
        decision = bool(vs) and all(v is True for v in vs)
        ok = "✓ correct" if decision == ground_truth else "✗ WRONG"
        print(f"  {method} unanimous-vote decision: {'DRAW' if decision else 'BLOCK'}  {ok}")


def print_report():
    if not os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    print("\n" + "=" * 70)
    print(f"CUMULATIVE REPORT — {len(rows)} frames across all sessions")
    print("=" * 70)

    for method in ("aruco", "vlm"):
        cm = defaultdict(int)
        for r in rows:
            truth = r["ground_truth"] == "True"
            v = r[method]
            verdict = None if v not in ("True", "False") else v == "True"
            if verdict is None:
                cm["unparsed"] += 1
            elif verdict and truth:
                cm["true_pos"] += 1
            elif not verdict and not truth:
                cm["true_neg"] += 1
            elif verdict and not truth:
                cm["false_allow"] += 1  # says paper when none — the dangerous one
            else:
                cm["false_block"] += 1
        total = len(rows)
        correct = cm["true_pos"] + cm["true_neg"]
        print(f"\n{method.upper()}: {correct}/{total} correct")
        print(f"  false ALLOW (would draw on bare surface): {cm['false_allow']}")
        print(f"  false BLOCK (paper there, refused):       {cm['false_block']}")
        if cm["unparsed"]:
            print(f"  unparsed/failed:                          {cm['unparsed']}")

    lat = [float(r["latency_s"]) for r in rows if r["vlm"] in ("True", "False")]
    if lat:
        print(f"\nVLM latency: mean {sum(lat)/len(lat):.1f}s, max {max(lat):.1f}s")

    print("\nPer-scenario (frames correct, vlm | aruco):")
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[(r["scenario"], r["ground_truth"])].append(r)
    for (scenario, truth), srows in sorted(by_scenario.items()):
        def n_ok(method):
            return sum(1 for r in srows if r[method] == truth)

        print(f"  {scenario} (paper={truth}): vlm {n_ok('vlm')}/{len(srows)} | aruco {n_ok('aruco')}/{len(srows)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    session = datetime.now().strftime("%m%d_%H%M")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, _ = cap.read()
    if not ret:
        print("ERROR: cannot capture from camera (is machine.py running and holding it?)")
        sys.exit(1)

    servos = None
    try:
        from servo_control.servo_control import ServoController

        servos = ServoController(port=SERIAL_PORT, baudrate=BAUD_RATE)
        if servos.ser is None:
            raise RuntimeError(f"no serial connection on {SERIAL_PORT}")
        servos.set_tilt(PAPER_DETECTION_GAZE_TILT)
        time.sleep(0.5)
        print(f"✓ Camera at paper-check tilt ({PAPER_DETECTION_GAZE_TILT}°); pan sweeps around {PAPER_DETECTION_GAZE_PAN}°")
    except Exception as e:
        print(f"⚠ Servos unavailable ({e}) — using current camera angle, no sweep")

    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()

        print("\nStage a scenario on the table, then name it here. Empty name = finish.")
        while True:
            try:
                scenario = input("\nScenario name (or Enter to finish): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not scenario:
                break
            truth = ""
            while truth not in ("y", "n"):
                truth = input("Ground truth — is there paper on the table? [y/n]: ").strip().lower()
            run_scenario(cap, servos, session, scenario, truth == "y", writer, csv_file)

    cap.release()
    print_report()


if __name__ == "__main__":
    main()
