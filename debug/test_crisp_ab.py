"""A/B ComfyUI style knobs against the live template, without touching runtime.

Queues variant A (exact runtime settings) and variant B (candidate knob
changes) with the SAME seed, prompt, and input frame, then waits for both
and reports the output files. Compare by eye + Laplacian sharpness.

    python debug/test_crisp_ab.py
"""

import copy
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API = "http://localhost:8188"
TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drawing", "impostor-template-impostor-bot-svg.json")

SEED = 991784582164
INPUT_FRAME = "/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy/event_log/a7345101-images/mood_1784582156.jpg"
PROMPT = (
    "Black ink line drawing on white paper. A thick, chaotic spiral radiates from a central point where lines converge tightly. "
    "Jagged shards extend outward along the curve before dissolving into scattered short dashes and dots representing dust motes. "
    "The composition uses bold, expressive linework with no shading or fills to depict motion and decay through simple geometric "
    "fragmentation against stark white paper."
)

RUNTIME_TRIGGER = "impostor black and white sketch line art "
CRISP_TRIGGER = "impostor black ink line art, sharp clean lines, high contrast, stark white background "


def build(variant, trigger, end_percent):
    wf = json.load(open(TEMPLATE))
    wf["607"]["inputs"]["image"] = INPUT_FRAME
    wf["616"]["inputs"]["value"] = trigger
    wf["723"]["inputs"]["String"] = PROMPT
    wf["294"]["inputs"].update(scheduler="beta", steps=25, denoise=1)
    wf["295"]["inputs"]["noise_seed"] = SEED
    wf["300"]["inputs"]["guidance"] = 4.0
    wf["711"]["inputs"].update(strength=0.3, start_percent=0.0, end_percent=end_percent)
    wf["5"]["inputs"].update(width=1024, height=1024, batch_size=1)
    wf["30"]["inputs"]["filename_prefix"] = f"crisp-{variant}"
    return wf


def queue(wf):
    data = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(f"{API}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)["prompt_id"]


def wait(pid, label, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{API}/history/{pid}") as r:
            hist = json.load(r)
        if pid in hist:
            outputs = hist[pid].get("outputs", {})
            files = [o["filename"] for node in outputs.values() for o in node.get("images", []) if o.get("type") == "output"]
            status = hist[pid].get("status", {})
            if status.get("completed") or files:
                print(f"[{label}] done in {time.time() - t0:.0f}s: {files}")
                return files
            if status.get("status_str") == "error":
                print(f"[{label}] ERROR: {json.dumps(status)[:500]}")
                return []
        time.sleep(5)
    print(f"[{label}] timed out")
    return []


def main():
    variants = [
        ("A-runtime", RUNTIME_TRIGGER, 1.0),
        ("B-crisp", CRISP_TRIGGER, 0.6),
    ]
    pids = []
    for name, trigger, endp in variants:
        pid = queue(build(name, trigger, endp))
        print(f"queued {name}: {pid}")
        pids.append((pid, name))
    for pid, name in pids:
        wait(pid, name)


if __name__ == "__main__":
    main()
