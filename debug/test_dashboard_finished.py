#!/usr/bin/env python3
"""Sep 13 — the dashboard's finished-sheet listing (dashboard/server.py).
The artist: "Do we have access to the actual images of the paper post drawing
for the last 3 drawings? We added a system for it but it doesn't show up in the
mobile UI." Checks the grouping (one row per capture, the _sheet crop
preferred, the model-sized _t1024 copy never listed), the paging, and the name
guard that keeps the route inside the folder. No server, no camera."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fails = 0


def check(name, ok, got=""):
    global fails
    print(("PASS  " if ok else "FAIL  ") + name + (f"   → {got!r}" if not ok else ""))
    fails += 0 if ok else 1


tmp = tempfile.mkdtemp(prefix="finished-")
import dashboard.server as D  # noqa: E402

D.FINISHED_FOLDER = tmp

NOW = time.time()
files = {
    "finished_20260913_025838_0.jpg": NOW - 300,
    "finished_20260913_025838_0_sheet.jpg": NOW - 299,
    "finished_20260913_025838_1.jpg": NOW - 298,
    "finished_20260913_025838_1_sheet.jpg": NOW - 297,
    "finished_20260913_025838_1_sheet_t1024.jpg": NOW - 296,  # the model's reading copy
    "finished_20260913_021353_0.jpg": NOW - 9000,  # a capture with no sheet crop
    "notes.txt": NOW,
    "finished_bogus.jpg": NOW,
}
for n, mt in files.items():
    open(os.path.join(tmp, n), "wb").write(b"x")
    os.utime(os.path.join(tmp, n), (mt, mt))

d = D.finished_list({})
names = [r["name"] for r in d["images"]]
check("one row per capture, newest first", names == ["finished_20260913_025838_1_sheet.jpg", "finished_20260913_025838_0_sheet.jpg", "finished_20260913_021353_0.jpg"], names)
check("the model-sized copy is never listed", not any("t1024" in n for n in names))
check("stray files are ignored", not any(n in ("notes.txt", "finished_bogus.jpg") for n in names))
check("the wide shot stands in when there is no crop", "finished_20260913_021353_0.jpg" in names)

one = D.finished_list({"limit": ["1"]})
check("limit and truncated", [r["name"] for r in one["images"]] == names[:1] and one["truncated"], one)
older = D.finished_list({"before": [str(d["images"][1]["mtime"])]})
check("before pages backwards", [r["name"] for r in older["images"]] == ["finished_20260913_021353_0.jpg"], [r["name"] for r in older["images"]])

good = ["finished_20260913_025838_1.jpg", "finished_20260913_025838_1_sheet.jpg"]
bad = ["../../etc/passwd", "finished_20260913_025838_1_sheet_t1024.jpg", "finished_x.jpg", "", "finished_20260913_025838_1.png", "finished_20260913_025838_1.jpg\n"]
check("the name guard accepts real captures", all(D.FINISHED_NAME_RE.match(n) for n in good))
check("the name guard refuses everything else", not any(D.FINISHED_NAME_RE.match(n) for n in bad), [n for n in bad if D.FINISHED_NAME_RE.match(n)])

D.FINISHED_FOLDER = os.path.join(tmp, "gone")
check("a missing folder is a message, not a crash", D.finished_list({}) == {"images": [], "error": "no finished-drawing captures yet"}, D.finished_list({}))

for n in files:
    os.remove(os.path.join(tmp, n))
os.rmdir(tmp)
print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
