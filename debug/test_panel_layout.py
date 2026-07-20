"""Verify the motor panel layout fits the screen — no clipped buttons.

Builds the full panel UI in a withdrawn (invisible) window, measures the
requested size of every top-level region and each workspace tab, and
checks the total against the actual screen. Run after any panel layout
change:

    python debug/test_panel_layout.py
"""

import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motor_panel.panel import build_ui


def main():
    root = tk.Tk()
    root.withdraw()
    ui = build_ui(root)
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    usable_h = sh - 70  # window chrome + a taskbar allowance
    req_w, req_h = root.winfo_reqwidth(), root.winfo_reqheight()
    print(f"screen {sw}x{sh}  (usable height budget {usable_h})")
    print(f"panel requested {req_w}x{req_h}")

    body = ui["body"]
    for name, widget in (("session frame", body), ("bed", body.bed), ("linkage", body.linkage), ("hand", body.hand_space), ("lung", body.lung_strip)):
        print(f"  {name:>14}: {widget.winfo_reqwidth()}x{widget.winfo_reqheight()}")

    ok = req_w <= sw and req_h <= usable_h
    print("FIT OK" if ok else f"DOES NOT FIT — over by {max(0, req_w - sw)}px wide, {max(0, req_h - usable_h)}px tall")
    root.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
