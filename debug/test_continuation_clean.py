"""The continuation cleaner (Sep 8): with a whole-document prefill the model
re-types the tail before continuing. If we fail to strip it, the re-typed
passage is stored as the new thought and the machine reads as repeating itself
when it was actually continuing. Run: python debug/test_continuation_clean.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llama_server import _clean_continuation as C  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + str(detail)) if (detail and not ok) else ''}")
    if not ok:
        FAILS.append(name)


tail = "the last thought ends here, 2m away: the white sheer curtain ripples gently in a draft"
check("exact re-typing is stripped", C(tail + " that seems to come from nowhere.", tail) == "that seems to come from nowhere.")
check("a clean continuation is untouched", C("that seems to come from nowhere.", tail) == "that seems to come from nowhere.")
retyped = "2m away: the white sheer curtain ripples gently in a draft, that seems to come from nowhere."
check("re-typing with changed punctuation is stripped", C(retyped, tail).startswith("that seems"), C(retyped, tail))
retyped2 = "2m  away:  the white sheer curtain ripples gently  in a draft that seems to come from nowhere."
check("re-typing with changed spacing is stripped", C(retyped2, tail).startswith("that seems"), C(retyped2, tail))
check("no prefill, no change", C("A whole new thought.", "") == "A whole new thought.")
curly = "waiting to see if it’s going to fall over. It hasn’t."
check("a curly apostrophe in the seam vs a straight one in the output is still stripped", C("It hasn't. The red bucket is still upside down.", curly) == "The red bucket is still upside down.", C("It hasn't. The red bucket is still upside down.", curly))
check("an unrelated reply is not truncated", C("The lamp is on.", tail) == "The lamp is on.")
print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
