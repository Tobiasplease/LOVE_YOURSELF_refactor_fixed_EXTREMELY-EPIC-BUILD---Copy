"""Warp calibration operator tool — measure the arm's real distortion field.

Workflow (machine.py stopped):

  1. python debug/warp_calibrate.py --generate           # writes dot-grid gcode
     python debug/warp_calibrate.py --run                # or streams it raw to GRBL
     -> the machine dots a 5x5 grid of KNOWN command coords on paper.
        Dot #1 has a short dash to its right (orientation mark).

  2. Photograph the sheet flat-on (keep it oriented as it sat in the
     machine), then:
     python debug/warp_calibrate.py --measure photo.jpg [--paper-w 210 --paper-h 148]

     Click the 4 PAPER corners in order: top-left, top-right, bottom-right,
     bottom-left ("top" = far from robot). Then click each dot in the
     prompted serpentine order (dot #1 = the one with the dash).
     Right-click skips an unreadable dot.

  3. The tool fits the inverse thin-plate spline, reports residuals, and
     writes grbl/warp_calibration.json — from then on ALL drawing gcode
     routes through the measured map (delete the JSON to revert).

Verify: python debug/warp_calibrate.py --square  writes a test-square gcode
file THROUGH the new calibration; draw it and check with a ruler.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from grbl import warp_calibration as wc

GCODE_PATH = "/tmp/warp_calibration_grid.gcode"
SQUARE_PATH = "/tmp/warp_test_square.gcode"


def homography(src, dst):
    """4+ point DLT homography src->dst."""
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
    _, _, V = np.linalg.svd(np.asarray(A, dtype=float))
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H, pt):
    v = H @ np.array([pt[0], pt[1], 1.0])
    return v[0] / v[2], v[1] / v[2]


def cmd_generate(args):
    lines = wc.generate_calibration_gcode(args.n)
    with open(GCODE_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {GCODE_PATH} ({args.n}x{args.n} dots over command domain {wc.DEFAULT_DOMAIN})")
    print("stream it RAW (no transforms) — easiest: python debug/warp_calibrate.py --run")


def cmd_run(args):
    from grbl.grbl_utils import ensure_homed, find_grbl_port, send_cmd
    lines = wc.generate_calibration_gcode(args.n)
    ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
    if ser is None:
        print("no GRBL found")
        return
    ensure_homed(ser)
    print(f"dotting {args.n}x{args.n} grid ({len(lines)} lines, raw command coords)…")
    for line in lines:
        send_cmd(ser, line, timeout=30.0)
    ser.close()
    print("done — photograph the sheet, then: python debug/warp_calibrate.py --measure photo.jpg")


def cmd_measure(args):
    import tkinter as tk
    from PIL import Image, ImageTk

    cmd_pts = wc.grid_points(args.n)
    img = Image.open(args.measure)
    scale = min(1400 / img.width, 900 / img.height, 1.0)
    disp = img.resize((int(img.width * scale), int(img.height * scale)))

    root = tk.Tk()
    root.title("warp calibration — click corners, then dots")
    cv = tk.Canvas(root, width=disp.width, height=disp.height)
    cv.pack()
    photo = ImageTk.PhotoImage(disp)
    cv.create_image(0, 0, anchor="nw", image=photo)
    status = tk.Label(root, text="", font=("monospace", 11))
    status.pack(fill="x")

    corners_mm = [(0, args.paper_h), (args.paper_w, args.paper_h), (args.paper_w, 0), (0, 0)]
    corner_names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    state = {"phase": "corners", "corners_px": [], "i": 0, "H": None, "paper_pts": [], "kept": []}

    def prompt():
        if state["phase"] == "corners":
            status.config(text=f"click paper corner {len(state['corners_px']) + 1}/4: "
                               f"{corner_names[len(state['corners_px'])]}  (top = far from robot)")
        else:
            status.config(text=f"click dot {state['i'] + 1}/{len(cmd_pts)} "
                               f"(command {cmd_pts[state['i']]}) — dot #1 has the dash. right-click = skip")

    def click(ev):
        px = (ev.x / scale, ev.y / scale)
        cv.create_oval(ev.x - 3, ev.y - 3, ev.x + 3, ev.y + 3, outline="#e94560", width=2)
        if state["phase"] == "corners":
            state["corners_px"].append(px)
            if len(state["corners_px"]) == 4:
                state["H"] = homography(state["corners_px"], corners_mm)
                state["phase"] = "dots"
        else:
            mm = apply_h(state["H"], px)
            state["paper_pts"].append(mm)
            state["kept"].append(state["i"])
            state["i"] += 1
            if state["i"] >= len(cmd_pts):
                finish()
                return
        prompt()

    def skip(ev):
        if state["phase"] == "dots":
            state["i"] += 1
            if state["i"] >= len(cmd_pts):
                finish()
                return
            prompt()

    def finish():
        kept_cmd = [cmd_pts[i] for i in state["kept"]]
        if len(kept_cmd) < 8:
            status.config(text=f"only {len(kept_cmd)} dots measured — too few, not saving")
            return
        cal = wc.WarpCalibration.fit(kept_cmd, state["paper_pts"])
        rms, mx = cal.residuals_mm()
        path = cal.save()
        status.config(text=f"saved {path} — {len(kept_cmd)} points, fit residual rms {rms:.2f}mm "
                           f"max {mx:.2f}mm. Draw the --square test next.")
        print(f"calibration saved: {path}")
        print(f"paper area for drawings: {['%.1f' % v for v in cal.paper_area]} mm")

    cv.bind("<Button-1>", click)
    cv.bind("<Button-3>", skip)
    prompt()
    root.mainloop()


def cmd_square(args):
    cal = wc.WarpCalibration.load()
    if cal is None:
        print("no calibration saved yet")
        return
    ideal = 40.0
    pts = [(5, 5), (35, 5), (35, 35), (5, 35), (5, 5)]
    lines = ["G21", "G90", "M3 S34", "G4 P0.3", ]
    x, y = pts[0]
    first = cal.apply_to_line(f"G0 X{x:.2f} Y{y:.2f}", ideal, ideal)
    lines.append(first)
    lines.append("M3 S52")
    lines.append("G4 P0.3")
    for x, y in pts[1:]:
        lines.append(cal.apply_to_line(f"G1 X{x:.2f} Y{y:.2f} F1000", ideal, ideal))
    lines += ["M3 S34", "G4 P0.3", "G0 X0 Y0"]
    with open(SQUARE_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {SQUARE_PATH} — a 30x30mm square through the measured calibration.")
    print("Stream it raw (or --run-file it) and put a ruler on the result.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true", help="write calibration dot-grid gcode")
    ap.add_argument("--run", action="store_true", help="stream the dot grid to GRBL raw")
    ap.add_argument("--measure", metavar="PHOTO", help="click-measure a photo of the dotted sheet")
    ap.add_argument("--square", action="store_true", help="write a test square through the calibration")
    ap.add_argument("--n", type=int, default=5, help="grid size (default 5x5)")
    ap.add_argument("--paper-w", type=float, default=210.0, help="paper width mm (default A5 landscape 210)")
    ap.add_argument("--paper-h", type=float, default=148.0, help="paper height mm (default 148)")
    args = ap.parse_args()
    if args.generate:
        cmd_generate(args)
    elif args.run:
        cmd_run(args)
    elif args.measure:
        cmd_measure(args)
    elif args.square:
        cmd_square(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
