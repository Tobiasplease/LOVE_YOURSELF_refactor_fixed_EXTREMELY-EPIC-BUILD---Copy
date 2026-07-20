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


def _survey_points(args):
    """Dot list for this survey. Priority: measured-boundary polygon grid
    (the default since July 20), or a rectangular grid if --domain given.
    The list is persisted to warp_survey.json at generate/run time and
    reloaded at measure time so the two can never disagree."""
    if args.domain:
        return wc.grid_points(args.n, tuple(args.domain))
    return wc.polygon_grid(spacing=args.spacing)


def cmd_generate(args):
    pts = _survey_points(args)
    lines = wc.generate_calibration_gcode(points=pts)
    with open(GCODE_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    wc.save_survey(pts, {"spacing": args.spacing})
    print(f"wrote {GCODE_PATH} ({len(pts)} dots inside the measured reach polygon)")
    print(f"dot list persisted to {wc.SURVEY_PATH}")
    print("stream it RAW (no transforms) — easiest: python debug/warp_calibrate.py --run")


def cmd_run(args):
    from grbl.grbl_utils import ensure_homed, find_grbl_port, send_cmd
    pts = _survey_points(args)
    lines = wc.generate_calibration_gcode(points=pts)
    wc.save_survey(pts, {"spacing": args.spacing})
    ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
    if ser is None:
        print("no GRBL found")
        return
    ensure_homed(ser)
    print(f"dotting {len(pts)} survey points ({len(lines)} lines, raw command coords)…")
    for line in lines:
        send_cmd(ser, line, timeout=30.0)
    ser.close()
    print("done — photograph the sheet, then: python debug/warp_calibrate.py --measure photo.jpg")


def cmd_measure(args):
    import tkinter as tk
    from PIL import Image, ImageTk

    cmd_pts = wc.load_survey()
    if cmd_pts is None:
        print("no saved survey (warp_survey.json) — falling back to grid from flags")
        cmd_pts = _survey_points(args)
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
    """An EXACT-size square in paper millimeters, centered in the calibrated
    drawing area, mapped point-by-point through the inverse TPS. Edges are
    subdivided in paper space — a single straight command-space segment
    would land as an arc (the disease itself)."""
    cal = wc.WarpCalibration.load()
    if cal is None:
        print("no calibration saved yet")
        return
    size = args.square_size
    x0, y0, x1, y1 = cal.paper_area
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = size / 2
    if size > min(x1 - x0, y1 - y0):
        print(f"warning: {size}mm exceeds the drawing area ({x1-x0:.0f}x{y1-y0:.0f}mm) — clipping risk")
    corners = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
    SUBDIV = 16
    path = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        for i in range(SUBDIV):
            f = i / SUBDIV
            path.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    path.append(corners[0])
    lines = ["G21", "G90", "M3 S34", "G4 P0.3"]
    mx, my = cal.to_command(*path[0])
    lines.append(f"G0 X{mx:.3f} Y{my:.3f}")
    lines.append("M3 S52")
    lines.append("G4 P0.3")
    for px, py in path[1:]:
        mx, my = cal.to_command(px, py)
        lines.append(f"G1 X{mx:.3f} Y{my:.3f} F1000")
    lines += ["M3 S34", "G4 P0.3", "G0 X0 Y0"]
    with open(SQUARE_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    import math as _m
    print(f"wrote {SQUARE_PATH} — an exactly {size:.0f}x{size:.0f}mm square (diagonals {size*_m.sqrt(2):.1f}mm).")
    print("Stream: python debug/warp_calibrate.py --run-file " + SQUARE_PATH)


def cmd_run_file(args):
    """Stream a gcode file raw over the same isolated path as --run —
    no bCNC, no pipeline, no transforms (the file is already final)."""
    from grbl.grbl_utils import ensure_homed, find_grbl_port, send_cmd
    with open(args.run_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith(";")]
    ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
    if ser is None:
        print("no GRBL found")
        return
    ensure_homed(ser)
    print(f"streaming {args.run_file} ({len(lines)} lines)…")
    for line in lines:
        send_cmd(ser, line, timeout=30.0)
    ser.close()
    print("done.")


def cmd_fit_paper(args):
    """Survey -> placement: find the largest W×H-aspect window inside the
    measured region, print where to tape the real sheet (in survey-sheet
    coordinates), and set it as the drawing area."""
    cal = wc.WarpCalibration.load()
    if cal is None:
        print("no calibration saved yet — measure first")
        return
    w_req, h_req = args.fit_paper
    aspect = w_req / h_req
    cx, cy, w_ach, h_ach, ang = wc.best_rect_rotated(cal.paper_pts, aspect)
    pct = 100 * h_ach / h_req
    corners = wc.window_corners(cx, cy, w_ach, h_ach, ang)
    print(f"requested window : {w_req:.0f} x {h_req:.0f} mm")
    print(f"achievable window: {w_ach:.1f} x {h_ach:.1f} mm  ({pct:.0f}% of requested), rotated {ang:+.0f}°")
    print(f"window corners — mm from the SURVEY sheet's bottom-left corner")
    print(f"(the corner marks on the bed; 'bottom' = near robot; order BL, BR, TR, TL")
    print(f"of the WINDOW, i.e. tape the sheet's corners at these points):")
    for name, (px, py) in zip(("BL", "BR", "TR", "TL"), corners):
        print(f"  {name}: ({px:.1f}, {py:.1f})")
    if pct >= 99:
        print("full requested size fits — tape the sheet there, at that angle.")
    else:
        print("the measured region cannot serve the full size; this is the largest")
        print("honest window. Tape a sheet there at that angle (or accept scaled output).")
    cal.paper_window = (cx, cy, w_ach, h_ach, ang)
    cal.save()
    print(f"drawing window updated in {wc.CALIBRATION_PATH} — all gcode now maps into it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true", help="write calibration dot-grid gcode")
    ap.add_argument("--run", action="store_true", help="stream the dot grid to GRBL raw")
    ap.add_argument("--measure", metavar="PHOTO", help="click-measure a photo of the dotted sheet")
    ap.add_argument("--square", action="store_true", help="write a test square through the calibration")
    ap.add_argument("--run-file", metavar="GCODE", help="stream a gcode file raw (isolated, no bCNC/pipeline)")
    ap.add_argument("--square-size", type=float, default=60.0, help="test square side length in mm (default 60)")
    ap.add_argument("--fit-paper", nargs=2, type=float, metavar=("W", "H"),
                    help="compute + set the largest W×H-aspect drawing window inside the measured region (mm)")
    ap.add_argument("--n", type=int, default=5, help="grid size for --domain rectangular mode")
    ap.add_argument("--spacing", type=float, default=10.0,
                    help="dot spacing (command units) for the polygon survey (default 10)")
    ap.add_argument("--domain", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1"),
                    help="OVERRIDE: rectangular survey domain instead of the measured reach polygon")
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
    elif args.run_file:
        cmd_run_file(args)
    elif args.fit_paper:
        cmd_fit_paper(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
