"""Prove the movement-vector gaze bias: the walk drifts toward the gaze
without ever leaving the demonstrated vocabulary.

A chain is trained on a NOISY full-range shoulder wander (60-120) so
states have branching continuations — bias reweights choice, and a
deterministic chain leaves no choice to reweight (a clean triangle sweep
is momentum-locked under second-order transitions; noticing that is
itself useful recording guidance: vary your gestures if you want gaze to
matter). A +bias walk must live clearly higher than a -bias walk, and
every emitted value must still be a demonstrated state (bias never
invents or offsets poses).

    python debug/test_gaze_bias.py
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel import arms_markov as engine

LO, HI = 60, 120
RATE = 20


def make_wander_take(n=4000, seed=7):
    rng = random.Random(seed)
    v, out = 90.0, []
    for i in range(n):
        v = max(LO, min(HI, v + rng.choice([-2, -1, 1, 2])))
        out.append({"t": i / RATE, "dt": 1 / RATE, "shoulder": v})
    return out


def run(chain, bias, seconds=6.0):
    seen = []
    gen = engine.Generator(
        chain,
        send_ease=lambda d: seen.append(d["shoulder"]),
        send_plan=lambda d, dt: None,
        speed=3.0,
        bias=(lambda: bias) if bias else None,
        bias_strength=3.0,
    )
    gen.start({"shoulder": 90.0})
    time.sleep(seconds)
    gen.stop()
    return seen


def main():
    failures = []
    chain = engine.train(make_wander_take(), ["shoulder"])
    print(f"chain: {chain['unique_states']} states over shoulder {LO}-{HI}")

    flat = run(chain, None)
    up = run(chain, {"shoulder": +1.0})
    down = run(chain, {"shoulder": -1.0})
    mean_flat, mean_up, mean_down = (sum(x) / len(x) for x in (flat, up, down))
    print(f"means — unbiased {mean_flat:.1f}°, gaze-up {mean_up:.1f}°, gaze-down {mean_down:.1f}°")

    if mean_up - mean_down < 20:
        failures.append(f"bias barely separates the walks (up-down = {mean_up - mean_down:.1f}°, expected >= 20)")
    for name, seq in (("up", up), ("down", down)):
        bad = [v for v in seq if not (LO - 1 <= v <= HI + 1)]
        if bad:
            failures.append(f"{name}-biased walk left the demonstrated range: {bad[:5]}")

    print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
