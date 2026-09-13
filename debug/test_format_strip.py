#!/usr/bin/env python3
"""Sep 11 — the coordinate incident (docs/where-we-are-sep9.md §24). A format at
a line opening (a stamp, a countdown stub, a coordinate-shaped pair) must not
survive to the stream, and a decimal must not be mangled into one. Runs the
real Captioner mouth methods. NOTE: importing the captioner mints a ~16 KB stub
run log in event_log/ regardless of MOOD_SNAPSHOT_FOLDER — quarantine it to
event_log/archive-stub-runs/ after running (docs/runtime-map.md ops note)."""
import os
import sys
import tempfile

os.environ["MOOD_SNAPSHOT_FOLDER"] = tempfile.mkdtemp(prefix="fmtstrip-")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from captioner.captioner import Captioner  # noqa: E402

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


_c = Captioner.__new__(Captioner)


def mouth(text):
    """The live order at the caption mouth (captioner.py retry loop)."""
    return _c._strip_leaked_stamps(Captioner._trim_to_boundary(Captioner._strip_list_shape(text)))


cases = [
    ("the raw coordinate", "40.548135, -79.992635. I don't need to be anywhere else.", "I don't need to be anywhere else."),
    ("the mangled form", "548135, -79.992635. I don't need to be anywhere else.", "I don't need to be anywhere else."),
    ("no-space integer pair", "547380,-120 The wood is bent, the seat is stained.", "The wood is bent, the seat is stained."),
    ("a bare pair → nothing (silence path)", "548133, -79.6091", ""),
    ("pair after the seam word", "decor. 547085, -83 The red foam finger is just decor.", "decor. The red foam finger is just decor."),
    ("pair mid-text at a sentence start", "It's not a hand. 546900, -30 It's just plastic.", "It's not a hand. It's just plastic."),
    ("countdown stub still dies", "5... 4... 3... go on then.", "go on then."),
    ("lone number stub dies", "12.", ""),
    ("stamp with AM and dots", "12:40 AM... wait, no. It's Friday.", "wait, no. It's Friday."),
    ("stamp with dash", "12:20 — It's just a hole where a mouth should be.", "It's just a hole where a mouth should be."),
    ("a count is speech", "7 minutes. I've been awake for seven minutes.", "7 minutes. I've been awake for seven minutes."),
    ("'100 years.' is speech", "100 years. That's how long the paper's been gone.", "100 years. That's how long the paper's been gone."),
    ("2x4s and 3D are words", "2x4s and a 3D print on the shelf.", "2x4s and a 3D print on the shelf."),
    ("a decimal is not a countdown", "40.5 degrees, the head says.", "40.5 degrees, the head says."),
    ("three numbers are left alone", "10, 20, 30 years. Same chair.", "10, 20, 30 years. Same chair."),
    ("a pair inside a sentence is left alone", "It's 12, 13 degrees in here.", "It's 12, 13 degrees in here."),
    # Sep 13 — THE VIDEO CLOCK. llama-server prepends "[0m0.17s]" to every native
    # video input; at 13:14:39 the model spoke it, wearing the stream's own
    # separator, and within two runs 72% of captions were chanting "1m40. 2m07."
    # at each other (system_state.json carried the tail across restarts, so it
    # reseeded in two minutes).
    ("the chunk marker wearing the stream's separator", "0m0.17s — I'm looking at the black office chair now.", "I'm looking at the black office chair now."),
    ("the chant", "1m40. 2m07. It's red.", "It's red."),
    ("markers between sentences", "0m51. It's just foam. 2m07. It's a piece of plastic.", "It's just foam. It's a piece of plastic."),
    ("the bracketed form", "[0m0.00s] the room is still.", "the room is still."),
    ("nothing but markers", "1m00. 1m00. 1m00.", ""),
    ("a clock time is speech", "It's 2:07 in the morning and nothing has moved.", "It's 2:07 in the morning and nothing has moved."),
    ("a bare measurement is speech", "The 3m gap between the desk and the wall.", "The 3m gap between the desk and the wall."),
]
for name, raw, want in cases:
    got = mouth(raw)
    check(name, got == want, got)

print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
