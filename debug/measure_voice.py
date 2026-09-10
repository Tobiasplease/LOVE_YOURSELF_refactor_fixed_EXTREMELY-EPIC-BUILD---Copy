"""Measure the voice (Sep 9 2026): six numbers over a window of captions, so a
change to the prompt, the model or the seam has a before and an after instead of
an impression. Written after the Sep 4-8 stretch, where every result that held
came from a probe with a number and every reverted experiment came from an
argument (docs/where-we-are-sep9.md).

The two failures it separates:
  - the TIC (form)    "it's not X, it's just Y" — flat across a run means a model
                      prior, not feedback; no gate can touch it.
  - the CHANT (content) the same sentence 3-16x — that IS feedback, from the
                      running-text window (docs/architecture-diagnosis-sep5.md).

Run:  python debug/measure_voice.py                  (current run log)
      python debug/measure_voice.py --hours 6
      python debug/measure_voice.py --log event_log/<run>-event-log.json
      python debug/measure_voice.py --hourly          (drift table: tic amplifying?)
"""

import argparse
import collections
import glob
import json
import os
import re
import time

REPO = os.path.join(os.path.dirname(__file__), "..")
EV = os.path.join(REPO, "event_log")

TIC = re.compile(
    r"\b(it|that|this|they)('s| is|’s| are|'re)\s+not\b.{0,60}?"
    r"\b(it|that|this|they)?('s| is|’s| just|,? just)\b|"
    r"\bnot a .{1,30}\.\s*(it'?s|just)\b",
    re.I,
)
JUST = re.compile(r"\b(it'?s|its|that'?s)\s+just\b", re.I)
PIVOT = re.compile(r"\bused to think\b", re.I)
PERSON = re.compile(
    r"\b(person|someone|somebody|the guy|the man|the woman|camo)\b", re.I
)


def load(path, hours):
    """Captions from a run log (JSONL), newest window first."""
    cutoff = time.time() - hours * 3600 if hours else 0
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "caption":
                continue
            if d.get("timestamp", 0) < cutoff:
                continue
            out.append((d.get("timestamp", 0), d.get("iso_timestamp", ""),
                        (d.get("caption") or "").strip()))
    return out


def report(caps, hourly=False):
    n = len(caps)
    if not n:
        print("no captions in window")
        return
    print(f"RUN  {caps[0][1]} -> {caps[-1][1]}   captions={n}")

    def row(label, k):
        print(f"  {label:<34} {k:5d}  {100 * k / n:5.1f}%")

    print("\nTIC (form — expect flat over time if it is a model prior)")
    row("'it's not X, it's just Y'", sum(1 for _, _, t in caps if TIC.search(t)))
    row("'it's just ...'", sum(1 for _, _, t in caps if JUST.search(t)))
    row("'I used to think ...'", sum(1 for _, _, t in caps if PIVOT.search(t)))

    print("\nCHANT (content — this one is feedback)")
    sent = collections.Counter()
    for _, _, t in caps:
        for s in re.split(r"(?<=[.!?])\s+", t):
            s = s.strip()
            if len(s.split()) >= 4:
                sent[s] += 1
    rep = sorted([(k, v) for k, v in sent.items() if v >= 3], key=lambda x: -x[1])
    emissions = sum(v for _, v in rep)
    print(f"  sentences said >=3x                {len(rep):5d}"
          f"   ({emissions} emissions, {100 * emissions / n:.1f}% of output)")
    dup = collections.Counter(t for _, _, t in caps if t)
    row("exact duplicate captions", sum(v - 1 for v in dup.values() if v > 1))
    for k, v in rep[:8]:
        print(f"     {v:3d}x  {k[:96]}")

    print("\nOTHER")
    row("cut mid-sentence (no end punct)",
        sum(1 for _, _, t in caps if t and t[-1] not in ".!?…\"'”’"))
    row("person mentioned", sum(1 for _, _, t in caps if PERSON.search(t)))
    gaps = sorted(caps[i + 1][0] - caps[i][0] for i in range(n - 1))
    gaps = [g for g in gaps if 0 <= g < 600]
    if gaps:
        print(f"  {'caption gap median / p90':<34} {gaps[len(gaps) // 2]:5.0f}s"
              f"  {gaps[9 * len(gaps) // 10]:.0f}s")

    if hourly:
        print("\nDRIFT  (a rising tic = amplification; flat = model prior)")
        b = collections.defaultdict(lambda: [0, 0, 0])
        for _, iso, t in caps:
            h = iso[11:13]
            b[h][0] += 1
            b[h][1] += bool(JUST.search(t))
            b[h][2] += bool(PIVOT.search(t))
        print("  hour     n   'just'   'used to think'")
        for h in sorted(b):
            k, j, u = b[h]
            print(f"   {h}    {k:4d}   {100 * j / k:5.1f}%   {100 * u / k:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="run log (default: most recent)")
    ap.add_argument("--hours", type=float, default=0, help="window (default: all)")
    ap.add_argument("--hourly", action="store_true", help="per-hour drift table")
    a = ap.parse_args()
    path = a.log or max(glob.glob(os.path.join(EV, "*-event-log.json")),
                        key=os.path.getmtime)
    print(f"log: {os.path.basename(path)}")
    report(load(path, a.hours), a.hourly)


if __name__ == "__main__":
    main()
