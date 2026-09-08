"""The storage law (Sep 8): a caption refused by the echo/phantom gates is
SPOKEN but stored nowhere — not the stream, not the compressor. A phantom that
leaves by one door and returns as a standing room fact through another is the
mechanism that turned one misread into an hour of invented company (Sep 7).
Run: python debug/test_storage_law.py"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + str(detail)) if (detail and not ok) else ''}")
    if not ok:
        FAILS.append(name)


src = open("captioner/captioner.py", encoding="utf-8").read()
print("\n[1] every store the caption path writes to is gated on the verdict")
add = re.search(r"if context_compressor and caption and caption\.strip\(\)([^\n:]*):", src)
check("the compressor is gated on _stream_store_ok", bool(add) and "_stream_store_ok" in add.group(1), add.group(0) if add else "call not found")
stream = re.search(r"if caption and getattr\(self, \"_stream_store_ok\", True\) and self\._stream_admissible", src)
check("the stream window is gated (unchanged)", bool(stream))
check("the verdict is set true at the top of each cycle", "self._stream_store_ok = True" in src)
check("and false when an echo-class gate fires", "self._stream_store_ok = False" in src)

print("\n[2] the feed still says a refused caption was spoken")
check("the not-kept marker survives", "_kept" in src and "not kept" in src)

print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
