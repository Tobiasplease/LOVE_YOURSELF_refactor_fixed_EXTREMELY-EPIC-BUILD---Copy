"""Closed-loop simulation of face tracking: does the camera bob at close range?

Models the real feedback loop: servo pan/tilt moves the camera, which moves
where a stationary face appears in frame, which drives the next gaze update.
Bbox jitter proportional to face size stands in for detection noise + the
flimsy mount. Compares the pre-July-10 tuning (no dead zone, underdamped
physics) against the current config on a far face and a close face.

Bobbing shows up as path length: total degrees the servo travels while the
person stands still.

Usage:
    python debug/test_face_tracking_stability.py
"""

import importlib
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W, H = 1280, 720
FOV = 60.0
FPS = 30
SECONDS = 8
SETTLE = 4  # ignore first N seconds (approach), measure steady state


def simulate(gaze, face_w_px, person_angle=100.0, seed=7):
    rng = random.Random(seed)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    positions = []
    for i in range(SECONDS * FPS):
        pan = gaze.physics_state.pan
        # stationary person: where their face lands in frame depends on camera pan
        offset_norm = (person_angle - pan) / FOV
        cx = W * (0.5 + offset_norm)
        cx = W - cx  # inverse of the FLIP_X the gaze code applies
        jitter = 0.05 * face_w_px  # detection noise + mount wobble scale with face size
        cx += rng.uniform(-jitter, jitter)
        cy = H * 0.45 + rng.uniform(-jitter, jitter)
        box = (int(cx - face_w_px / 2), int(cy - face_w_px * 0.6), int(cx + face_w_px / 2), int(cy + face_w_px * 0.6))
        gaze.update_gaze(frame, box, "calm_observant")
        positions.append((gaze.physics_state.pan, gaze.physics_state.tilt))
        time.sleep(1 / FPS)
    steady = positions[SETTLE * FPS :]
    pans = [p for p, _ in steady]
    path = sum(abs(steady[i][0] - steady[i - 1][0]) + abs(steady[i][1] - steady[i - 1][1]) for i in range(1, len(steady)))
    return {"pan_std": float(np.std(pans)), "path_deg": path, "path_deg_per_s": path / (SECONDS - SETTLE)}


def run(label, patch_old):
    import vision.gaze as gaze

    results = {}
    for scenario, face_w in (("far (60px face)", 60), ("close (320px face)", 320)):
        importlib.reload(gaze)  # reset state machine + physics between runs
        if patch_old:
            gaze.FACE_TRACK_DEAD_ZONE = 0.0
            gaze.FACE_TRACK_DEAD_ZONE_FACE_SCALE = 0.0
            gaze.FACE_TRACK_MAX_STEP = 999.0
            gaze.TRACKING_PHYSICS["damping"] = 2.5
        results[scenario] = simulate(gaze, face_w)
    print(f"\n{label}")
    for scenario, r in results.items():
        print(f"  {scenario:20s} steady-state pan std {r['pan_std']:5.2f}°   servo travel {r['path_deg_per_s']:6.1f}°/s")
    return results


if __name__ == "__main__":
    old = run("OLD (no dead zone, damping 2.5)", patch_old=True)
    new = run("NEW (scaled dead zone + step cap + critical damping)", patch_old=False)
    close_improvement = old["close (320px face)"]["path_deg_per_s"] / max(0.01, new["close (320px face)"]["path_deg_per_s"])
    print(f"\nclose-range servo travel reduced {close_improvement:.0f}x")
