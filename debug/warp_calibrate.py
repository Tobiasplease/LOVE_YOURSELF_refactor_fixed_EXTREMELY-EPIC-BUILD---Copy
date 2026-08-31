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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from grbl import warp_calibration as wc

GCODE_PATH = "/tmp/warp_calibration_grid.gcode"
SQUARE_PATH = "/tmp/warp_test_square.gcode"


def fit_frame_transform(src, dst):
    """Best photo->frame transform from anchor pairs: affine (robust, 6 DOF)
    vs homography (perspective-true, 8 DOF), chosen by leave-one-out error —
    5 noisy clicks can't afford a full homography's noise amplification
    unless the perspective actually warrants it."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)

    def fit_affine(s, d):
        A = np.hstack([s, np.ones((len(s), 1))])
        M, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
        return lambda p: np.hstack([np.atleast_2d(p), np.ones((len(np.atleast_2d(p)), 1))]) @ M

    def loo(fitter, apply_one):
        errs = []
        for i in range(len(src)):
            keep = [j for j in range(len(src)) if j != i]
            model = fitter(src[keep], dst[keep])
            errs.append(np.linalg.norm(apply_one(model, src[i]) - dst[i]))
        return float(np.mean(errs))

    aff_err = loo(fit_affine, lambda m, p: m(p)[0])
    hom_err = loo(lambda s, d: homography(s, d), lambda H, p: np.array(apply_h(H, p)))
    if aff_err <= hom_err:
        m = fit_affine(src, dst)
        return (lambda p: m(p)[0]), "affine", aff_err
    H = homography(src, dst)
    return (lambda p: np.array(apply_h(H, p))), "homography", hom_err


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
    state = {"phase": "corners", "corners_px": [], "i": 0, "H": None, "paper_pts": [], "kept": [], "history": []}
    survey_full = wc.load_survey_full() or {}
    ring_mode = survey_full.get("mode") == "ring"
    expected_old = None
    if ring_mode:
        _, expected_old, _ = _expected_layout()
        # Ring mode needs NO paper corners: the 5 anchors (exactly-known
        # positions) carry the full photo->frame homography themselves —
        # more accurate than corner clicks, and immune to occluded corners.
        state["phase"] = "dots"
    ghost_items = []

    def _anchor_transform():
        n_a = survey_full["n_anchors"]
        anchors = [(k, p) for k, p in zip(state["kept"], state["paper_pts"]) if k < n_a]
        if len(anchors) < 4:
            return None, None, None
        src = [p for _, p in anchors]  # photo px (full res)
        dst = [expected_old[k] for k, _ in anchors]  # original frame mm
        return fit_frame_transform(src, dst)

    def _anchor_loo_errors():
        """Per-anchor leave-one-out disagreement — a misidentified anchor
        stands out as the one whose omission fixes everyone else."""
        n_a = survey_full["n_anchors"]
        anchors = [(k, p) for k, p in zip(state["kept"], state["paper_pts"]) if k < n_a]
        if len(anchors) < 4:
            return []
        out = []
        for i, (ki, pi) in enumerate(anchors):
            rest = [a for j, a in enumerate(anchors) if j != i]
            T, _, _ = fit_frame_transform([p for _, p in rest], [expected_old[k] for k, _ in rest])
            err = float(np.linalg.norm(T(np.array(pi)) - np.array(expected_old[ki])))
            out.append((ki + 1, err))
        return out

    def _anchor_feedback():
        errs = _anchor_loo_errors()
        if not errs:
            return ""
        worst = max(errs, key=lambda e: e[1])
        mean = sum(e for _, e in errs) / len(errs)
        note = f"   [anchors: mean {mean:.1f}mm"
        if worst[1] > 4.0:
            note += f" — ANCHOR {worst[0]} OFF BY {worst[1]:.0f}mm, press u to undo/redo it"
        return note + "]"

    def draw_ghosts():
        """After 4 anchors: project expected dots onto the photo as numbered
        ghosts; the currently prompted one glows. Click the ink nearest it."""
        for item in ghost_items:
            cv.delete(item)
        ghost_items.clear()
        if not ring_mode or expected_old is None:
            return
        n_a = survey_full["n_anchors"]
        anchors = [(k, p) for k, p in zip(state["kept"], state["paper_pts"]) if k < n_a]
        if len(anchors) < 4:
            return
        # reverse fit (old frame -> photo px) purely for ghost projection
        T_rev, _, _ = fit_frame_transform([expected_old[k] for k, _ in anchors], [p for _, p in anchors])
        for i in range(state["i"], len(cmd_pts)):
            gpx = T_rev(np.array(expected_old[i]))
            px, py = gpx[0] * scale, gpx[1] * scale
            if i == state["i"]:  # NEXT dot: unmissable magenta bullseye
                ghost_items.append(cv.create_oval(px - 16, py - 16, px + 16, py + 16, outline="#ff00d0", width=3))
                ghost_items.append(cv.create_line(px - 22, py, px + 22, py, fill="#ff00d0", width=1))
                ghost_items.append(cv.create_line(px, py - 22, px, py + 22, fill="#ff00d0", width=1))
                ghost_items.append(cv.create_text(px, py - 26, text=str(i + 1), fill="#ff00d0", font=("monospace", 12, "bold")))
            else:
                ghost_items.append(cv.create_oval(px - 9, py - 9, px + 9, py + 9, outline="#4ecdc4", width=2))
                ghost_items.append(cv.create_text(px, py - 17, text=str(i + 1), fill="#4ecdc4", font=("monospace", 10)))

    def prompt():
        if state["phase"] == "corners":
            status.config(
                text=f"click paper corner {len(state['corners_px']) + 1}/4: " f"{corner_names[len(state['corners_px'])]}  (top = far from robot)"
            )
        else:
            status.config(
                text=f"click dot {state['i'] + 1}/{len(cmd_pts)} "
                f"(command {cmd_pts[state['i']]}) — dot #1 has the dash. "
                f"right-click = skip, u = undo" + (_anchor_feedback() if ring_mode else "")
            )

    click_marks = []

    def click(ev):
        if state["phase"] == "dots" and state["i"] >= len(cmd_pts):
            return  # session complete (or merge refused) — only u works now
        px = (ev.x / scale, ev.y / scale)
        click_marks.append(cv.create_oval(ev.x - 3, ev.y - 3, ev.x + 3, ev.y + 3, outline="#e94560", width=2))
        if state["phase"] == "corners":
            state["corners_px"].append(px)
            if len(state["corners_px"]) == 4:
                state["H"] = homography(state["corners_px"], corners_mm)
                state["phase"] = "dots"
        else:
            # ring mode stores raw photo px (anchors define the frame);
            # base mode converts via the paper-corner homography
            mm = px if ring_mode else apply_h(state["H"], px)
            state["paper_pts"].append(mm)
            state["kept"].append(state["i"])
            state["history"].append("click")
            state["i"] += 1
            if state["i"] >= len(cmd_pts):
                finish()
                return
        draw_ghosts()
        prompt()

    def skip(ev):
        if state["phase"] == "dots":
            state["history"].append("skip")
            state["i"] += 1
            if state["i"] >= len(cmd_pts):
                finish()
                return
            draw_ghosts()
            prompt()

    def undo(ev=None):
        if state["phase"] != "dots" or not state["history"]:
            return
        action = state["history"].pop()
        state["i"] -= 1
        if action == "click":
            state["paper_pts"].pop()
            state["kept"].pop()
            if click_marks:
                cv.delete(click_marks.pop())
        draw_ghosts()
        prompt()

    def finish():
        survey = wc.load_survey_full()
        if survey and survey.get("mode") == "ring":
            _finish_ring(survey)
            return
        kept_cmd = [cmd_pts[i] for i in state["kept"]]
        if len(kept_cmd) < 8:
            status.config(text=f"only {len(kept_cmd)} dots measured — too few, not saving")
            return
        cal = wc.WarpCalibration.fit(kept_cmd, state["paper_pts"])
        rms, mx = cal.residuals_mm()
        path = cal.save()
        status.config(text=f"saved {path} — {len(kept_cmd)} points, fit residual rms {rms:.2f}mm " f"max {mx:.2f}mm. Draw the --square test next.")
        print(f"calibration saved: {path}")
        print(f"paper area for drawings: {['%.1f' % v for v in cal.paper_area]} mm")

    def _finish_ring(survey):
        n_a = survey["n_anchors"]
        anchors = [(k, p) for k, p in zip(state["kept"], state["paper_pts"]) if k < n_a]
        ring = [(k, p) for k, p in zip(state["kept"], state["paper_pts"]) if k >= n_a]
        if len(anchors) < 4:
            status.config(text=f"only {len(anchors)} anchors clicked — need 4+ to register the photo")
            return
        T, kind, reg_err = _anchor_transform()
        if reg_err > 2.5:
            per = ", ".join(f"anchor {k}: {e:.1f}mm" for k, e in _anchor_loo_errors())
            status.config(
                text=f"MERGE REFUSED — registration {reg_err:.1f}mm (need <2.5). {per}. " f"Press u repeatedly to rewind and re-click the bad anchor."
            )
            print(f"merge refused: registration {reg_err:.2f}mm — per-anchor: {per}")
            return
        cal_old = wc.WarpCalibration.load()
        merged_cmd = [tuple(c) for c in cal_old.command_pts] + [cmd_pts[k] for k, _ in ring]
        merged_paper = [tuple(p) for p in cal_old.paper_pts] + [tuple(map(float, T(np.array(p)))) for _, p in ring]
        print(f"frame registration: {kind}, leave-one-out error {reg_err:.2f}mm")
        new_cal = wc.WarpCalibration.fit(merged_cmd, merged_paper)
        new_cal.paper_window = cal_old.paper_window
        new_cal.save()
        status.config(
            text=f"ring merged: +{len(ring)} points (now {len(merged_cmd)}), " f"anchor registration err {reg_err:.2f}mm. Redraw --outline to verify."
        )
        print(f"ring survey merged: {len(ring)} new points, anchor registration {reg_err:.2f}mm mean")
        print("the grown window now rests on measured ground — redraw the outline to verify the corner")

    cv.bind("<Button-1>", click)
    cv.bind("<Button-3>", skip)
    root.bind("u", undo)
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


def cmd_outline(args):
    """Trace the calibrated drawing window's perimeter on the bed — the
    machine draws its own taping guide. A short inward tick marks the
    window's bottom-left corner (portrait 'down')."""
    cal = wc.WarpCalibration.load()
    if cal is None:
        print("no calibration saved yet")
        return
    if cal.paper_window:
        corners = wc.window_corners(*cal.paper_window)
    else:
        x0, y0, x1, y1 = cal.paper_area
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    SUBDIV = 14
    path = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        for i in range(SUBDIV):
            f = i / SUBDIV
            path.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
    path.append(corners[0])
    lines = ["G21", "G90", "M3 S34", "G4 P0.3"]
    mx, my = cal.to_command(*path[0])
    lines.append(f"G0 X{mx:.3f} Y{my:.3f}")
    lines += ["M3 S52", "G4 P0.3"]
    for px, py in path[1:]:
        mx, my = cal.to_command(px, py)
        lines.append(f"G1 X{mx:.3f} Y{my:.3f} F1000")
    # BL tick: 12mm toward the window centre from the bottom-left corner
    blx, bly = corners[0]
    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4
    d = ((cx - blx) ** 2 + (cy - bly) ** 2) ** 0.5
    tx, ty = blx + (cx - blx) / d * 12, bly + (cy - bly) / d * 12
    for px, py in ((blx, bly), (tx, ty)):
        mx, my = cal.to_command(px, py)
        lines.append(f"G1 X{mx:.3f} Y{my:.3f} F1000")
    lines += ["M3 S34", "G4 P0.3", "G0 X0 Y0"]
    out = "/tmp/warp_window_outline.gcode"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if cal.paper_window:
        cxw, cyw, w, h, ang = cal.paper_window
        print(f"wrote {out} — the {w:.0f}x{h:.0f}mm window at {ang:+.0f}°, with a tick at its bottom-left corner.")
    print("Lay a large sheet (or draw on the backing), then:")
    print(f"  python debug/warp_calibrate.py --run-file {out}")
    print("Tape your A4 aligned to the drawn rectangle; the tick corner is the sheet's bottom-left.")


def cmd_ring(args):
    """Dot the RING (unmeasured territory the grown window leans on) plus 5
    anchor re-dots from the original survey. Photograph and --measure as
    usual: anchors register the new sheet to the old frame, ring dots merge
    into the calibration, extrapolation guessing ends."""
    from grbl.grbl_utils import ensure_homed, find_grbl_port, send_cmd

    cal = wc.WarpCalibration.load()
    if cal is None:
        print("no calibration to extend — run the base survey first")
        return
    anchors = wc.pick_anchors(cal)
    ring = wc.ring_points(cal, spacing=args.spacing)
    if not ring:
        print("no ring points — the measured hull already covers the walked boundary")
        return
    anchor_cmd = [tuple(cal.command_pts[i]) for i in anchors]
    pts = anchor_cmd + ring
    lines = wc.generate_calibration_gcode(points=pts)
    wc.save_survey(
        pts,
        {
            "mode": "ring",
            "n_anchors": len(anchor_cmd),
            "anchor_paper_old": [list(cal.paper_pts[i]) for i in anchors],
        },
    )
    print(f"{len(anchor_cmd)} anchor re-dots + {len(ring)} ring dots " f"(dash marks anchor #1). Click order: anchors first, then ring.")
    ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
    if ser is None:
        print("no GRBL found")
        return
    ensure_homed(ser)
    print(f"dotting {len(pts)} points…")
    for line in lines:
        send_cmd(ser, line, timeout=30.0)
    ser.close()
    print("done — photograph (all four sheet corners in frame), then --measure as usual.")


def _expected_layout():
    """Predicted paper positions (original frame) for every survey point:
    anchors from stored truth, ring dots via a forward fit (command->paper)."""
    survey = wc.load_survey_full()
    cal = wc.WarpCalibration.load()
    if not survey or cal is None:
        return None, None, None
    pts = [tuple(p) for p in survey["points"]]
    n_a = survey.get("n_anchors", 0)
    from scipy.interpolate import RBFInterpolator

    fwd = RBFInterpolator(cal.command_pts, cal.paper_pts, kernel="thin_plate_spline", smoothing=1e-3)
    expected = []
    for i, p in enumerate(pts):
        if i < n_a:
            expected.append(tuple(survey["anchor_paper_old"][i]))
        else:
            expected.append(tuple(fwd(np.array([p]))[0]))
    return pts, expected, n_a


def cmd_guide(args):
    """Render the click-order map for the current survey (/tmp/ring_guide.png)."""
    pts, expected, n_a = _expected_layout()
    if pts is None:
        print("no survey/calibration to render")
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))
    for i, (ex, ey) in enumerate(expected):
        if i < n_a:
            ax.scatter([ex], [ey], marker="s", s=130, c="#e94560", zorder=3)
        else:
            ax.scatter([ex], [ey], marker="o", s=90, c="#2a9d8f", zorder=3)
        ax.annotate(str(i + 1), (ex, ey), fontsize=11, ha="center", va="center", color="white", zorder=4)
    ax.annotate("dash", expected[0], xytext=(12, 12), textcoords="offset points", color="#e94560")
    ax.set_title(f"click order — {n_a} anchors (red squares) then ring dots (green), as laid on the ORIGINAL sheet")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("/tmp/ring_guide.png", dpi=100)
    print("wrote /tmp/ring_guide.png — the constellation with click numbers")
    print("(your sheet may sit shifted/rotated: match the PATTERN, not absolute position)")


def cmd_corner_dots(args):
    """Targeted top-up for an existing (undotted-photo) ring session: dots
    placed ON the drawing window's command track wherever it runs outside
    measured territory — the exact curve the outline draws. Appends to the
    saved survey so ONE photo + click session covers ring + corners."""
    from scipy.spatial import Delaunay

    from grbl.grbl_utils import ensure_homed, find_grbl_port, send_cmd

    cal = wc.WarpCalibration.load()
    survey = wc.load_survey_full()
    if cal is None or not survey or survey.get("mode") != "ring":
        print("needs an active ring survey (run --ring first)")
        return
    hull = Delaunay(cal.command_pts)
    walked = wc._inset_polygon(wc.MEASURED_BOUNDARY, 0.97)
    corners = wc.window_corners(*cal.paper_window)
    track = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        for i in range(24):
            f = i / 24
            px, py = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            mx, my = cal.to_command(px, py)
            if hull.find_simplex(np.array([(mx, my)]))[0] < 0 and wc._point_in_polygon(mx, my, walked):
                track.append((round(mx, 2), round(my, 2)))
    targeted = []
    for p in track:  # thin to >=5 units apart
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= 25 for q in targeted):
            targeted.append(p)
    if not targeted:
        print("window track is fully inside measured territory — nothing to add")
        return
    existing = [tuple(p) for p in survey["points"]]
    wc.save_survey(existing + targeted, {k: v for k, v in survey.items() if k != "points"})
    lines = wc.generate_calibration_gcode(points=targeted, dash=False)
    print(f"{len(targeted)} targeted dots on the window's command track (click order: after the ring dots)")
    ser = find_grbl_port(preferred_port=os.getenv("GRBL_PORT", "/dev/arduino_cnc"))
    if ser is None:
        print("no GRBL found")
        return
    ensure_homed(ser)
    for line in lines:
        send_cmd(ser, line, timeout=30.0)
    ser.close()
    print("done — NOW photograph once and --measure; the guide (--guide) includes the new dots.")


def cmd_grow_window(args):
    """Grow the drawing window into the negotiable ring: beyond the dotted
    (interpolated) region but never beyond the physically walked boundary.
    Accuracy softens only in the strip outside the dot hull — the middle of
    every drawing stays fully calibrated. Verify with --outline after."""
    import numpy as np
    from scipy.spatial import Delaunay

    cal = wc.WarpCalibration.load()
    if cal is None or not cal.paper_window:
        print("need a calibration with a fitted window (--fit-paper) first")
        return
    cx, cy, w, h, ang = cal.paper_window
    walked = wc._inset_polygon(wc.MEASURED_BOUNDARY, 0.97)  # 3% off the true walls
    hull = Delaunay(cal.paper_pts)

    def perimeter(f, per_edge=15):
        corners = wc.window_corners(cx, cy, w * f, h * f, ang)
        pts = []
        for a, b in zip(corners, corners[1:] + corners[:1]):
            for i in range(per_edge):
                t = i / per_edge
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        return pts

    def safe(f):
        for px, py in perimeter(f):
            mx, my = cal.to_command(px, py)
            if not wc._point_in_polygon(mx, my, walked):
                return False
        return True

    lo, hi = 1.0, 1.8
    if not safe(1.0):
        print("current window already touches the walked boundary — no room")
        return
    for _ in range(22):
        mid = (lo + hi) / 2
        if safe(mid):
            lo = mid
        else:
            hi = mid
    fmax = lo
    f = fmax if args.grow_window <= 0 else min(args.grow_window, fmax)
    outside = sum(1 for p in perimeter(f) if hull.find_simplex(np.array([p]))[0] < 0)
    total = len(perimeter(f))
    print(f"max reach-safe factor: {fmax:.3f}  (window {w*fmax:.0f} x {h*fmax:.0f} mm)")
    print(f"applying factor {f:.3f}: window {w*f:.0f} x {h*f:.0f} mm at {ang:+.0f}°")
    print(
        f"extrapolation exposure: {outside}/{total} perimeter samples outside the dotted "
        f"region — expect curvature to creep back in those stretches, middle stays true"
    )
    cal.paper_window = (cx, cy, w * f, h * f, ang)
    cal.save()
    print("saved. Draw it and judge: python debug/warp_calibrate.py --outline")
    print("(revert anytime: --fit-paper 210 297 restores the fully-calibrated window)")


def cmd_auto_measure(args):
    """Fully automatic ring measurement: blob-detect dots in the photo,
    match the constellation to the expected layout by exhaustive similarity
    search (any rotation/scale, mirrored photos included), register the
    frame on the anchors, merge under the 2.5mm guard. No clicking."""
    import cv2 as cv2mod
    from PIL import Image
    from scipy.spatial import cKDTree

    cal = wc.WarpCalibration.load()
    if cal is None:
        print("no calibration")
        return

    # Constellation for THIS sheet = the saved survey (what was actually
    # inked). Each point is KNOWN (its command already measured in the
    # calibration — anchors and previously-merged dots, exact positions) or
    # UNKNOWN (position only forward-predicted; these are what we measure).
    # Registration uses ALL matched knowns — after a partial merge that can
    # be a dozen points, making leave-one-out statistically honest.
    from scipy.interpolate import RBFInterpolator

    survey_pts = wc.load_survey()
    if survey_pts is None:
        print("no saved survey (warp_survey.json)")
        return
    fwd = RBFInterpolator(cal.command_pts, cal.paper_pts, kernel="thin_plate_spline", smoothing=1e-3)
    E_list, known, all_cmd = [], [], []
    for p in survey_pts:
        d = np.linalg.norm(cal.command_pts - np.asarray(p), axis=1)
        j = int(np.argmin(d))
        if d[j] < 1.0:
            E_list.append(tuple(cal.paper_pts[j]))
            known.append(True)
        else:
            E_list.append(tuple(fwd(np.array([p]))[0]))
            known.append(False)
        all_cmd.append(tuple(p))
    E = np.array(E_list)
    known = np.array(known)
    n_known = int(known.sum())
    print(f"survey: {len(E)} inked dots — {n_known} known (registration), {len(E) - n_known} to measure")
    if n_known < 4:
        print("fewer than 4 known dots — need a base survey first")
        return
    # bases for the search: the 5 most-spread knowns
    kidx = np.nonzero(known)[0]
    Ek = E[kidx]
    spread = [
        int(kidx[i])
        for i in (
            np.argmin(Ek[:, 0]),
            np.argmax(Ek[:, 0]),
            np.argmin(Ek[:, 1]),
            np.argmax(Ek[:, 1]),
            np.argmin(np.linalg.norm(Ek - Ek.mean(axis=0), axis=1)),
        )
    ]
    base_idx = list(dict.fromkeys(spread))
    n_a = len(base_idx)  # naming kept for downstream code

    # detect candidate dots in the photo. Document-filtered photos bleach
    # pen dots to faint gray (~200-230 vs 250+ paper), so: sweep HIGH
    # thresholds and let the anchor pentagon itself pick the detection set
    # that contains it. px/mm is estimated from the sheet's own extent.
    arr = np.array(Image.open(args.auto_measure).convert("L"))
    sheet = (arr > 120).astype(np.uint8)
    n, lab, stats, cents = cv2mod.connectedComponentsWithStats(sheet)
    big = 1 + int(np.argmax(stats[1:, cv2mod.CC_STAT_AREA]))
    mask = cv2mod.erode((lab == big).astype(np.uint8), np.ones((31, 31), np.uint8))
    mask_edge = cv2mod.erode((lab == big).astype(np.uint8), np.ones((7, 7), np.uint8))
    sheet_w_px = float(max(stats[big, cv2mod.CC_STAT_WIDTH], stats[big, cv2mod.CC_STAT_HEIGHT]))
    paper_long = max(args.paper_w, args.paper_h)
    if paper_long < 300:  # default flags are A5 — surveys are shot on A3
        paper_long = 420.0
        print("assuming A3 (420mm long side) — pass --paper-w/--paper-h to override")
    px_per_mm = sheet_w_px / paper_long
    print(f"sheet {sheet_w_px:.0f}px long side -> ~{px_per_mm:.2f} px/mm")

    # Dots can be ghost-faint (5-15 gray levels above paper) under an
    # illumination gradient — global thresholds can't isolate them. Flatten
    # the background with a heavy blur and detect compact local DIPS.
    bg = cv2mod.GaussianBlur(arr, (0, 0), 15)
    dip = bg.astype(np.int16) - arr.astype(np.int16)  # ink = positive dip

    def detect(delta, m=None):
        dark = ((dip > delta) & ((mask if m is None else m) > 0)).astype(np.uint8)
        nn, ll, st, ce = cv2mod.connectedComponentsWithStats(dark)
        raw = []
        for i in range(1, nn):
            a = st[i, cv2mod.CC_STAT_AREA]
            w, h = st[i, cv2mod.CC_STAT_WIDTH], st[i, cv2mod.CC_STAT_HEIGHT]
            if 3 <= a <= 2500 and w <= 70 and h <= 70 and max(w, h) / max(1, min(w, h)) <= 8:
                raw.append((float(ce[i][0]), float(ce[i][1]), float(a)))
        merged = []
        for x, y, a in sorted(raw, key=lambda r: -r[2]):
            for m in merged:
                if (m[0] - x) ** 2 + (m[1] - y) ** 2 < (2.2 * px_per_mm) ** 2:
                    tot = m[2] + a
                    m[0] = (m[0] * m[2] + x * a) / tot
                    m[1] = (m[1] * m[2] + y * a) / tot
                    m[2] = tot
                    break
            else:
                merged.append([x, y, a])
        return np.array([(m[0], m[1]) for m in merged]) if merged else np.zeros((0, 2))

    d2 = np.sum((E[None, :, :] - E[:, None, :]) ** 2, axis=2)
    Ea = E[known]

    def try_match(D):
        """Anchor-pentagon similarity search + affine keystone refinement.
        Returns (refine_fn, D) or None."""
        if len(D) < n_a:
            return None
        tree = cKDTree(D)
        best = (0, 0, None)
        for ia in base_idx:
            for ib in base_idx:
                if ia == ib:
                    continue
                Lmm = float(np.sqrt(d2[ia, ib]))
                for i in range(len(D)):
                    dists = np.hypot(D[:, 0] - D[i, 0], D[:, 1] - D[i, 1])
                    for j in np.nonzero(np.abs(dists - Lmm * px_per_mm) < 0.12 * Lmm * px_per_mm)[0]:
                        if i == j:
                            continue
                        s = dists[j] / Lmm
                        for refl in (1, -1):
                            ea, eb = E[ia], E[ib]
                            va, vb = eb - ea, D[j] - D[i]
                            ca = np.array([va[0], va[1] * refl])
                            ang = np.arctan2(vb[1], vb[0]) - np.arctan2(ca[1], ca[0])
                            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
                            F = np.diag([1, refl])
                            proj_a = (Ea - ea) @ F.T @ R.T + D[i]
                            dist_a, _ = tree.query(proj_a)
                            inl_a = int((dist_a < 7.0 * s).sum())
                            if inl_a < max(4, int(0.6 * len(Ea))):
                                continue
                            proj = (E - ea) @ F.T @ R.T + D[i]
                            dist, _ = tree.query(proj)
                            inl = int((dist < 12.0 * s).sum())
                            if (inl_a, inl) > (best[0], best[1]):
                                best = (inl_a, inl, (ea, R, F, D[i]))
        if best[2] is None:
            return None
        ea, R, F, off = best[2]
        s0 = float(np.sqrt(abs(np.linalg.det(R))))
        tree = cKDTree(D)
        proj_a = (Ea - ea) @ F.T @ R.T + off
        dist_a, near_a = tree.query(proj_a)
        a_pairs = [(k, int(near_a[k])) for k in range(len(Ea)) if dist_a[k] < 7.0 * s0]
        A = np.hstack([Ea[[k for k, _ in a_pairs]], np.ones((len(a_pairs), 1))])
        M, _, _, _ = np.linalg.lstsq(A, D[[d for _, d in a_pairs]], rcond=None)

        def refine(pts):
            return np.hstack([pts, np.ones((len(pts), 1))]) @ M

        dist_a, _ = tree.query(refine(Ea))
        tight = int((dist_a < 3.5 * s0).sum())
        if tight < max(4, int(0.6 * len(Ea))):
            return None
        # a TRUE pentagon drags the rest of the constellation with it;
        # coincidental quadrilaterals don't. Require the entourage.
        dist_all, _ = tree.query(refine(E))
        entourage = int((dist_all < 8.0 * s0).sum())
        # scale with the constellation: after partial merges few expecteds
        # remain (and some sit in the edge band, invisible at match stage)
        need = n_a + max(3, int(0.35 * (len(E) - n_a)))
        if entourage < need:
            return None
        print(f"knowns locked: {best[0]}/{len(Ea)} (sim), {tight}/{len(Ea)} within 3.5mm after affine, " f"{best[1]}/{len(E)} total")
        return refine, D, s0

    result = None
    won_delta = None
    for delta in (14, 11, 9, 7, 5, 4):
        D = detect(delta)
        if not (n_a <= len(D) <= 800):
            continue
        result = try_match(D)
        if result:
            print(f"matched at dip depth {delta} ({len(D)} candidates)")
            won_delta = delta
            break
    if result is None:
        print("constellation match failed at every threshold — photo may need retaking (flatter/sharper)")
        return
    refine, D, s0 = result
    # harvest liberally: frame is locked, so edge-band dots (eaten by the
    # conservative mask) can now be collected by proximity to expectation
    D_edge = detect(won_delta, mask_edge)
    if len(D_edge):
        both = np.vstack([D, D_edge])
        keep = []
        for p in both:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= (2.2 * px_per_mm) ** 2 for q in keep):
                keep.append(p)
        D = np.array(keep)
        print(f"harvest: {len(D)} candidates including edge band")
    tree = cKDTree(D)
    s = s0

    proj = refine(E)
    dist2, nearest2 = tree.query(proj, k=2)
    s = s0
    matched = {}
    for k in range(len(E)):
        d1, d2 = dist2[k][0], dist2[k][1]
        # knowns: tight gate. Unknowns: loose gate (their predictions are
        # extrapolated — the very reason they're being measured) but only
        # when the identification is unambiguous (clear nearest-vs-second).
        if known[k]:
            if d1 < 8.0 * s:
                matched[k] = int(nearest2[k][0])
        else:
            if d1 < 16.0 * s and (d2 - d1) > 5.0 * s:
                matched[k] = int(nearest2[k][0])
    anchors_matched = [k for k in matched if known[k]]
    print(f"constellation locked: {len(matched)}/{len(E)} dots matched, {len(anchors_matched)}/{n_known} knowns")
    if len(anchors_matched) < 4:
        print("not enough known dots matched — aborting")
        return

    # final frame: anchor-fitted photo->old-frame transform (LOO-guarded)
    Tfin, kind, loo = fit_frame_transform([D[matched[k]] for k in anchors_matched], [E[k] for k in anchors_matched])
    print(f"frame registration: {kind}, leave-one-out {loo:.2f}mm")
    # Acceptance: strict LOO alone, OR moderate LOO backed by the entourage —
    # LOO over 5 points amplifies honest re-dot scatter ~2-3x (true locks
    # measured ~4-5mm; misidentifications 8-12mm), while a wrong frame can
    # never place a full anchor set AND a dozen ring dots simultaneously.
    entourage_ok = len(anchors_matched) >= max(5, int(0.7 * n_known)) and len(matched) >= len(anchors_matched) + 2
    if not (loo <= 2.5 or (loo <= 6.0 and entourage_ok)):
        print("MERGE REFUSED — registration too poor. Inspect /tmp/automatch.png")
    else:
        import shutil

        shutil.copy(wc.CALIBRATION_PATH, wc.CALIBRATION_PATH + ".bak")
        new_cmd = [tuple(c) for c in cal.command_pts]
        new_paper = [tuple(p) for p in cal.paper_pts]
        added = 0
        for k, di in matched.items():
            if known[k]:
                continue
            new_cmd.append(tuple(map(float, all_cmd[k])))
            new_paper.append(tuple(map(float, Tfin(D[di]))))
            added += 1
        new_cal = wc.WarpCalibration.fit(new_cmd, new_paper)
        new_cal.paper_window = cal.paper_window
        new_cal.save()
        print(f"MERGED: +{added} measured points (now {len(new_cmd)}). Backup at warp_calibration.json.bak")

    # overlay for human eyes
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(arr, cmap="gray")
    ax.scatter(D[:, 0], D[:, 1], s=14, c="#4ecdc4", label="detected")
    for k, di in matched.items():
        c = "#e94560" if k < n_a else "#ffb400"
        ax.plot([proj[k][0], D[di][0]], [proj[k][1], D[di][1]], c=c, lw=1)
        ax.scatter([proj[k][0]], [proj[k][1]], s=30, c=c)
    ax.set_title(f"auto-match: {len(matched)}/{len(E)} (red=anchors, yellow=new points)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/automatch.png", dpi=90)
    print("overlay: /tmp/automatch.png")


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
    angles = [args.angle] if args.angle is not None else None
    cx, cy, w_ach, h_ach, ang = wc.best_rect_rotated(cal.paper_pts, aspect, **({"angles": angles} if angles else {}))
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
    ap.add_argument("--outline", action="store_true", help="trace the calibrated drawing window on the bed (taping guide)")
    ap.add_argument("--grow-window", type=float, metavar="FACTOR", help="scale the window into the reach-safe extrapolation ring (0 = max safe)")
    ap.add_argument("--angle", type=float, metavar="DEG", help="pin the window rotation for --fit-paper (default: search all angles)")
    ap.add_argument("--ring", action="store_true", help="dot the unmeasured ring + anchor re-dots, to be merged via --measure")
    ap.add_argument("--guide", action="store_true", help="render the click-order constellation map for the current survey")
    ap.add_argument(
        "--corner-dots", action="store_true", help="append targeted dots on the window's command track (same sheet, before photographing)"
    )
    ap.add_argument("--auto-measure", metavar="PHOTO", help="fully automatic ring measurement from a photo — no clicking")
    ap.add_argument(
        "--fit-paper",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        help="compute + set the largest W×H-aspect drawing window inside the measured region (mm)",
    )
    ap.add_argument("--n", type=int, default=5, help="grid size for --domain rectangular mode")
    ap.add_argument("--spacing", type=float, default=10.0, help="dot spacing (command units) for the polygon survey (default 10)")
    ap.add_argument(
        "--domain",
        nargs=4,
        type=float,
        metavar=("X0", "X1", "Y0", "Y1"),
        help="OVERRIDE: rectangular survey domain instead of the measured reach polygon",
    )
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
    elif args.outline:
        cmd_outline(args)
    elif args.grow_window is not None:
        cmd_grow_window(args)
    elif args.ring:
        cmd_ring(args)
    elif args.guide:
        cmd_guide(args)
    elif args.corner_dots:
        cmd_corner_dots(args)
    elif args.auto_measure:
        cmd_auto_measure(args)
    elif args.fit_paper:
        cmd_fit_paper(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
