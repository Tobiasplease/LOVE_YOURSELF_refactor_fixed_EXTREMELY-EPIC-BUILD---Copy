"""A/B probe: does the drift turn need the image DROPPED, or does framing win
with vision supplied? (Artist's Sep 3 question — 'more a matter of prompt
ordering than omitting information?')

Both arms use the drift turn's exact call shape: drift.system + drift.ask,
DRIFT_TEMP/DRIFT_NUM_PREDICT, the stream as history. Only `image` varies:
  A — image=None (as built)
  B — a recent frame supplied; the single-image path already places the ask
      AFTER the image, so text sits closest to generation (the ordering law).

Seeds come from the machine's own live captions (register-true history, no
invented voice). LLM logs go to the dir given on the command line (default a
probe folder under debug/) — NEVER the live run's event log.

Run: python debug/probe_drift_image_ab.py [frame.jpg] [n_pairs]
"""

import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = os.path.join(os.path.dirname(__file__), "..")


def make_shell(seed_lines):
    from captioner.captioner import Captioner

    c = Captioner.__new__(Captioner)
    c._stream = deque(maxlen=24)
    c._stream_ts = deque(maxlen=24)
    now = time.time()
    for i, line in enumerate(seed_lines):
        c._stream.append(line.strip())
        c._stream_ts.append(now - (len(seed_lines) - i) * 30)
    return c


def newest_frame():
    import glob

    imgs = sorted(glob.glob(os.path.join(REPO, "event_log", "*-images", "*.jpg")), key=os.path.getmtime)
    return imgs[-1] if imgs else None


def seed_from_live(n=5):
    path = os.path.join(REPO, "event_log", "live_captions.txt")
    try:
        lines = [ln.strip() for ln in open(path) if len(ln.strip()) > 30]
        return lines[-n:] if len(lines) >= n else lines
    except OSError:
        return []


def main():
    from captioner.prompt_registry import P
    from config.config import DRIFT_NUM_PREDICT, DRIFT_TEMP, MODEL_NAME
    from utils.inference import query_model

    frame = sys.argv[1] if len(sys.argv) > 1 else newest_frame()
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    log_dir = os.path.join(REPO, "debug", "probe_drift_logs")
    os.makedirs(log_dir, exist_ok=True)

    seeds = seed_from_live()
    if not seeds:
        print("no live captions to seed from")
        sys.exit(1)
    if not frame:
        print("no frame found for arm B")
        sys.exit(1)

    shell = make_shell(seeds)
    history = shell._stream_history()
    print(f"seed: last {len(seeds)} live captions | frame: {os.path.basename(frame)}")
    print(f"temp {DRIFT_TEMP}, num_predict {DRIFT_NUM_PREDICT}\n")

    for i in range(n_pairs):
        for arm, img in (("A image-dropped", None), ("B image-supplied", frame)):
            t0 = time.time()
            try:
                text = query_model(
                    prompt=P("drift.ask"),
                    model=MODEL_NAME,
                    image=img,
                    system_prompt=P("drift.system"),
                    timeout=90,
                    log_dir=log_dir,
                    options={"temperature": DRIFT_TEMP, "num_predict": DRIFT_NUM_PREDICT},
                    prompt_type="drift_probe",
                    history=history,
                    skip_generation_wait=True,
                )
            except Exception as e:
                text = f"(call failed: {e})"
            dt = time.time() - t0
            print(f"--- pair {i + 1} arm {arm} ({dt:.1f}s) ---")
            print((text or "(empty)").strip())
            print()


if __name__ == "__main__":
    main()
