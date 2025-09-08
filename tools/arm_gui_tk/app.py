import json
import os
import sys
from pathlib import Path
from tkinter import BOTH, BOTTOM, LEFT, RIGHT, TOP, Button, Canvas, Frame, Label, Tk, filedialog, messagebox

import numpy as np

# Support running both as a module and as a script
try:
    from .calibration import apply_homography, compute_homography, invert_homography, load_calibration, save_calibration
    from .correct import prewarp_points, simulate_print
    from .grbl_link import GrblLink
    from .gcode_utils import parse_path_for_preview, transform_gcode_file
except Exception:
    # Fallback when executed as a script: add repo root to sys.path and import absolute package
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.arm_gui_tk.calibration import apply_homography, compute_homography, invert_homography, load_calibration, save_calibration
    from tools.arm_gui_tk.correct import prewarp_points, simulate_print
    from tools.arm_gui_tk.grbl_link import GrblLink
    from tools.arm_gui_tk.gcode_utils import parse_path_for_preview, transform_gcode_file
from grbl.grbl_utils import execute_gcode_file


class HomographyApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title("CNC GUI – 4‑Point Skew Calibration (MVP)")

        # Layout
        self.top = Frame(root)
        self.top.pack(side=TOP, fill=BOTH)

        self.canvas = Canvas(root, width=900, height=650, bg="#fffef8")
        self.canvas.pack(fill=BOTH, expand=True)

        self.status = Label(root, text="Click 'New 4‑pt Calib' to start", anchor="w")
        self.status.pack(side=BOTTOM, fill=BOTH)

        # Modes
        Button(self.top, text="Mode: Arm", command=self.set_mode_arm).pack(side=LEFT)
        Button(self.top, text="Mode: Calib", command=self.set_mode_calib).pack(side=LEFT)

        # Calib (compute only)
        Button(self.top, text="New 4‑pt Calib", command=self.start_calib).pack(side=LEFT)
        Button(self.top, text="Compute", command=self.compute).pack(side=LEFT)

        # GRBL
        Button(self.top, text="Connect+Home", command=self.connect_home).pack(side=LEFT)
        # Mapping anchors (click canvas first, then press these)
        Button(self.top, text="Map: (0,0)=Here", command=self.map_set_origin_here).pack(side=LEFT)
        Button(self.top, text="Map: (Xmax,0)=Here", command=self.map_set_xmax_here).pack(side=LEFT)
        Button(self.top, text="Map: (0,Ymax)=Here", command=self.map_set_ymax_here).pack(side=LEFT)
        Button(self.top, text="Sync Pose", command=self.arm_lock_from_grbl).pack(side=LEFT)
        Button(self.top, text="Engage Control", command=self.engage_control).pack(side=LEFT)
        Button(self.top, text="Disengage", command=self.disengage_control).pack(side=LEFT)
        Button(self.top, text="Pen Up", command=self.pen_up).pack(side=LEFT)
        Button(self.top, text="Pen Down", command=self.pen_down).pack(side=LEFT)
        Button(self.top, text="Flip Y", command=self.toggle_flip_y).pack(side=LEFT)

        # G-code
        Button(self.top, text="Load G-code", command=self.load_gcode).pack(side=LEFT)
        Button(self.top, text="Run Loaded", command=self.run_loaded_gcode).pack(side=LEFT)
        Button(self.top, text="Run Corrected", command=self.run_corrected_gcode).pack(side=LEFT)

        # Arm
        Button(self.top, text="Arm: Lock From GRBL", command=self.arm_lock_from_grbl).pack(side=LEFT)
        Button(self.top, text="Arm: Toggle Elbow", command=self.arm_toggle_elbow).pack(side=LEFT)

        # Reference square (ideal) in pixels (treat px as mm for MVP)
        # Order: TL, TR, BR, BL (counter-clockwise)
        self.ideal_square = [(120, 120), (420, 120), (420, 420), (120, 420)]

        # State
        self.measured_points: list[tuple[float, float]] = []
        self.calib = None  # dict with H, H_inv
        self.calib_path = Path("calibration/homography_calibration.json")
        self.collecting = False

        # Live control
        self.grbl = GrblLink()
        self.mm_per_px = 1.0  # MVP: treat 1 px as 1 mm
        # Default Y-flip follows repo config if available
        flip_default = False
        try:
            # Prefer a CNC-specific flag if present; else fall back to FLIP_Y
            from config.config import CNC_Y_FLIP  # type: ignore

            flip_default = bool(CNC_Y_FLIP)
        except Exception:
            try:
                from config.config import FLIP_Y  # type: ignore

                flip_default = bool(FLIP_Y)
            except Exception:
                flip_default = False
        self.flip_y = flip_default
        self.origin_px = (0.0, 0.0)  # canvas origin that maps to GRBL (0,0)
        # Work bounds (safe area). Use repo GRBL_IDLE_ZONE if present.
        try:
            from config.config import GRBL_IDLE_ZONE
            self.bounds = GRBL_IDLE_ZONE  # (x_min, x_max, y_min, y_max)
        except Exception:
            # Default safe region
            self.bounds = (0.0, 50.0, 0.0, 50.0)
        self.live_pen_down = False
        self.last_click_px = (0.0, 0.0)
        self.loaded_gcode_path: Path | None = None
        self.loaded_gcode_pts: list[tuple[float, float]] = []
        self.machine_pos_mm: tuple[float, float] | None = None

        # 2-link arm model (mm)
        self.arm_L1 = 25.0
        self.arm_L2 = 25.0
        self.arm_base_mm = (0.0, 0.0)
        self.arm_elbow_up = True
        self.arm_theta1 = 0.0
        self.arm_theta2 = 0.0
        self.mode = "arm"  # or "calib"
        self.drag_target = None  # 'base' or 'tip' or None
        self.control_engaged = False  # Only send moves to GRBL when engaged
        # Mapping anchors (pixels)
        self.map_p00_px: tuple[float, float] | None = None
        self.map_pX_px: tuple[float, float] | None = None
        self.map_pY_px: tuple[float, float] | None = None

        # Canvas bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Initial draw
        self.redraw()
        # Place base at bottom-center and scale to full reach
        self.root.after(50, self._place_base_default)
        # Poll GRBL position for live correlation
        self.root.after(200, self.poll_machine_pos)

    # ---------- Canvas→WPos mapping (three-point) ----------
    def _recompute_mapping(self):
        if self.map_p00_px is None:
            return
        x_min, x_max, y_min, y_max = self.bounds
        origin_px = self.map_p00_px
        sx = None
        sy = None
        if self.map_pX_px is not None and x_max > x_min:
            dx_px = self.map_pX_px[0] - origin_px[0]
            if abs(dx_px) > 1e-3:
                sx = (x_max - x_min) / dx_px
        if self.map_pY_px is not None and y_max > y_min:
            dy_px = self.map_pY_px[1] - origin_px[1]
            if abs(dy_px) > 1e-3:
                sy = (y_max - y_min) / abs(dy_px)
        # Choose mm_per_px
        if sx and sy:
            self.mm_per_px = float((sx + sy) / 2.0)
        elif sx:
            self.mm_per_px = float(sx)
        elif sy:
            self.mm_per_px = float(sy)
        # Set origin so that (0,0) maps to origin_px
        self.origin_px = (origin_px[0], origin_px[1])
        self.update_status("Mapping updated from anchors")
        self.redraw()

    def map_set_origin_here(self):
        self.map_p00_px = (self.last_click_px[0], self.last_click_px[1])
        self._recompute_mapping()

    def map_set_xmax_here(self):
        self.map_pX_px = (self.last_click_px[0], self.last_click_px[1])
        self._recompute_mapping()

    def map_set_ymax_here(self):
        self.map_pY_px = (self.last_click_px[0], self.last_click_px[1])
        self._recompute_mapping()

    def _place_base_default(self):
        """Place base at bottom-center and scale so full reach fits on canvas."""
        w = float(self.canvas.winfo_width() or 900)
        h = float(self.canvas.winfo_height() or 650)
        margin = 30.0
        base_px = (w * 0.5, h - margin)

        # Compute allowable radius to top/left/right edges
        r_left = base_px[0] - margin
        r_right = (w - margin) - base_px[0]
        r_top = base_px[1] - margin  # distance to top edge
        r_px_allow = max(10.0, min(r_left, r_right, r_top))

        R_mm = float(self.arm_L1 + self.arm_L2)
        # Choose mm_per_px so that R_mm maps to ~95% of r_px_allow
        self.mm_per_px = R_mm / (r_px_allow * 0.95)

        # Keep current arm_base_mm in mm but set origin so that it lands at base_px
        x_mm, y_mm = self.arm_base_mm
        y_mm_draw = -y_mm if self.flip_y else y_mm
        self.origin_px = (
            base_px[0] - (x_mm / self.mm_per_px),
            base_px[1] - (y_mm_draw / self.mm_per_px),
        )
        self.redraw()

    # -------- UI Actions --------
    def start_calib(self):
        self.collecting = True
        self.measured_points.clear()
        self.calib = None
        self.update_status("Click 4 measured corners in order: TL, TR, BR, BL")
        self.redraw()

    def compute(self):
        if len(self.measured_points) != 4:
            messagebox.showwarning("Need 4 points", "Please click 4 measured corners first.")
            return
        try:
            H = compute_homography(self.ideal_square, self.measured_points)
            H_inv = invert_homography(H)
            self.calib = {"H": H, "H_inv": H_inv}
            self.collecting = False
            self.update_status("Calibration computed. Click Preview or Save.")
        except Exception as e:
            messagebox.showerror("Calibration failed", str(e))

        self.redraw()

    def preview(self):
        if not self.calib:
            messagebox.showinfo("No calibration", "Compute or load a calibration first.")
            return
        self.redraw(preview=True)
        self.update_status("Preview: green=ideal, red=measured, blue=pre‑warp to send, cyan=simulated print")

    def save(self):
        if not self.calib:
            messagebox.showinfo("No calibration", "Compute a calibration first.")
            return
        try:
            save_calibration(
                self.calib_path,
                H=self.calib["H"],
                H_inv=self.calib["H_inv"],
                ideal_pts=self.ideal_square,
                measured_pts=self.measured_points or [],
            )
            messagebox.showinfo("Saved", f"Calibration saved to {self.calib_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def load(self):
        try:
            data = load_calibration(self.calib_path)
            self.calib = {"H": data["H"], "H_inv": data["H_inv"]}
            # ideal/measured points are optional for display
            self.measured_points = [tuple(p) for p in data.get("measured_points", [])]
            self.update_status("Calibration loaded. Click Preview.")
        except FileNotFoundError:
            messagebox.showwarning("Not found", f"No file at {self.calib_path}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
        self.redraw()

    def clear(self):
        self.collecting = False
        self.measured_points.clear()
        self.calib = None
        self.update_status("Cleared. Click 'New 4‑pt Calib' to start")
        self.redraw()

    # -------- GRBL Integration --------
    def connect_home(self):
        preferred = None
        retries = -1
        try:
            from config.config import GRBL_CNC_PORT, GRBL_HOMING_MAX_RETRIES

            preferred = GRBL_CNC_PORT
            retries = GRBL_HOMING_MAX_RETRIES if GRBL_HOMING_MAX_RETRIES != -1 else -1
        except Exception:
            pass

        ok = self.grbl.connect_and_home(preferred_port=preferred, max_homing_retries=retries)
        if ok:
            # Apply repo mapping automatically on connect for consistent defaults
            try:
                self.apply_repo_mapping()
                # Enforce base at bottom-center and scale to full reach
                self._place_base_default()
            except Exception:
                pass
            self.update_status("Connected and homed. Sync Pose, then Engage Control to move.")
        else:
            messagebox.showerror("GRBL connect/home failed", self.grbl.state.last_error or "Unknown error")

    def disconnect(self):
        self.grbl.disconnect()
        self.update_status("Disconnected.")

    def pen_up(self):
        self.live_pen_down = False
        self.grbl.pen_up()

    def pen_down(self):
        self.live_pen_down = True
        self.grbl.pen_down()

    # -------- Event Handlers --------
    def on_click(self, event):
        # Calibration point collection
        if self.mode == "calib" and self.collecting:
            self.measured_points.append((float(event.x), float(event.y)))
            if len(self.measured_points) < 4:
                self.update_status(f"Selected {len(self.measured_points)}/4. Keep clicking corners (TL → TR → BR → BL)")
            else:
                self.collecting = False
                self.update_status("4 points collected. Click Compute.")
            self.redraw()
            return

        # Arm mode: decide whether user clicked near base or tip
        x_mm, y_mm = self.canvas_to_mm(event.x, event.y)
        if self.mode == "arm":
            bx, by = self.mm_to_canvas(*self.arm_base_mm)
            # Determine if click is near base handle
            if (event.x - bx) ** 2 + (event.y - by) ** 2 <= 12 ** 2:
                self.drag_target = "base"
                self.arm_set_base_px(event.x, event.y)
            else:
                self.drag_target = "tip"
                if self.control_engaged:
                    self._rel_last_px = (float(event.x), float(event.y))
                else:
                    self.arm_move_to_tip(x_mm, y_mm)
        self.last_click_px = (float(event.x), float(event.y))

    def on_drag_move(self, event):
        x_mm, y_mm = self.canvas_to_mm(event.x, event.y)
        if self.mode == "arm":
            if self.drag_target == "base":
                self.arm_set_base_px(event.x, event.y)
            else:
                if self.control_engaged:
                    if self._rel_last_px is None:
                        self._rel_last_px = (float(event.x), float(event.y))
                    dx_px = float(event.x) - self._rel_last_px[0]
                    dy_px = float(event.y) - self._rel_last_px[1]
                    self._rel_last_px = (float(event.x), float(event.y))
                    dy_sign = -1.0 if self.flip_y else 1.0
                    dx_mm = dx_px * self.mm_per_px
                    dy_mm = dy_sign * dy_px * self.mm_per_px
                    # Use current position if available to compute delta; do not clamp
                    cur = self.grbl.get_machine_position()
                    if cur is not None:
                        pass
                    self.grbl.move_delta(dx_mm, dy_mm, rapid=not self.live_pen_down)
                else:
                    self.arm_move_to_tip(x_mm, y_mm)
        self.last_click_px = (float(event.x), float(event.y))

    def on_release(self, _event):
        self.drag_target = None
        self._rel_last_px = None

    # -------- Drawing --------
    def redraw(self, preview: bool = False):
        c = self.canvas
        c.delete("all")

        # Calibration UI: show the ideal square only in calibration mode
        if self.mode == "calib":
            self._draw_poly(self.ideal_square, outline="#2e8b57", width=2)
            self._draw_points(self.ideal_square, color="#2e8b57", labels=["1", "2", "3", "4"])  # TL,TR,BR,BL

        # Draw measured clicks (red)
        if self.measured_points:
            self._draw_poly(self.measured_points, outline="#cc3b3b", width=2)
            self._draw_points(self.measured_points, color="#cc3b3b", labels=["1", "2", "3", "4"])  # as clicked

        # If preview and we have a calibration, show pre-warp and simulated print
        if preview and self.calib is not None and self.mode == "calib":
            H = self.calib["H"]
            H_inv = self.calib["H_inv"]

            # Pre-warp the ideal square (what we would send)
            to_send = prewarp_points(self.ideal_square, H_inv)
            self._draw_poly(to_send, outline="#1f77b4", width=2)  # blue

            # Simulate what would print if the world applies H to the sent points
            printed = simulate_print(to_send, H)
            self._draw_poly(printed, outline="#17becf", width=1, dash=(4, 4))  # cyan dashed

        # Draw a simple mm grid for live control context
        self._draw_grid()

        # Draw loaded G-code path (gray)
        if self.loaded_gcode_pts and self.mode == "arm":
            path_px = [self.mm_to_canvas(x, y) for (x, y) in self.loaded_gcode_pts]
            self._draw_poly_px(path_px, outline="#bbbbbb", width=1)

        # Draw current machine position (red dot)
        if self.machine_pos_mm is not None:
            x_px, y_px = self.mm_to_canvas(*self.machine_pos_mm)
            r = 4
            self.canvas.create_oval(x_px - r, y_px - r, x_px + r, y_px + r, fill="#d62728", outline="", width=0)

        # Draw the 2-link arm (base, elbow, tip)
        if self.mode == "arm":
            self._draw_arm()

    def _draw_poly(self, pts, outline="#000", width=1, dash=None):
        if len(pts) < 2:
            return
        flat = []
        for x, y in pts:
            flat.extend([x, y])
        # close the polygon
        flat.extend([pts[0][0], pts[0][1]])
        self.canvas.create_line(*flat, fill=outline, width=width, dash=dash)

    def _draw_points(self, pts, color="#000", labels=None):
        r = 4
        for i, (x, y) in enumerate(pts):
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, width=2)
            if labels and i < len(labels):
                self.canvas.create_text(x + 10, y - 10, text=str(labels[i]), fill=color, anchor="w")

    def update_status(self, text: str):
        self.status.config(text=text)

    # -------- Helpers --------
    def canvas_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        # MVP mapping: 1 px = 1 mm, origin at (origin_px)
        x_mm = (x_px - self.origin_px[0]) * self.mm_per_px
        y_mm = (y_px - self.origin_px[1]) * self.mm_per_px
        if self.flip_y:
            y_mm = -y_mm
        return (x_mm, y_mm)

    def prewarp_if_needed(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        if not self.calib:
            return x_mm, y_mm
        pts = prewarp_points([(x_mm, y_mm)], self.calib["H_inv"])  # type: ignore
        return pts[0]

    def mm_to_canvas(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        y = -y_mm if self.flip_y else y_mm
        return (
            x_mm / self.mm_per_px + self.origin_px[0],
            y / self.mm_per_px + self.origin_px[1],
        )

    def _draw_grid(self):
        # Light 50 px grid, darker 100 px
        w = int(self.canvas.winfo_width() or 900)
        h = int(self.canvas.winfo_height() or 650)
        for x in range(0, w + 1, 50):
            color = "#f0f0f0" if x % 100 else "#e0e0e0"
            self.canvas.create_line(x, 0, x, h, fill=color)
        for y in range(0, h + 1, 50):
            color = "#f0f0f0" if y % 100 else "#e0e0e0"
            self.canvas.create_line(0, y, w, y, fill=color)

    def _draw_arm(self):
        try:
            from .kinematics import fk_2link
        except Exception:
            from tools.arm_gui_tk.kinematics import fk_2link

        elbow, tip = fk_2link(self.arm_theta1, self.arm_theta2, self.arm_base_mm, self.arm_L1, self.arm_L2)
        bx, by = self.mm_to_canvas(*self.arm_base_mm)
        ex, ey = self.mm_to_canvas(*elbow)
        tx, ty = self.mm_to_canvas(*tip)

        # Reach annulus
        Rmax = (self.arm_L1 + self.arm_L2) / self.mm_per_px
        Rmin = abs(self.arm_L1 - self.arm_L2) / self.mm_per_px
        self.canvas.create_oval(bx - Rmax, by - Rmax, bx + Rmax, by + Rmax, outline="#ddd")
        self.canvas.create_oval(bx - Rmin, by - Rmin, bx + Rmin, by + Rmin, outline="#eee")

        # Links
        self.canvas.create_line(bx, by, ex, ey, fill="#333", width=4)
        self.canvas.create_line(ex, ey, tx, ty, fill="#333", width=4)
        # Joints with labels
        for (cx, cy, lbl, color) in [
            (bx, by, "Base", "#2ca02c"),
            (ex, ey, "Elbow", "#ff7f0e"),
            (tx, ty, "Tip", "#1f77b4"),
        ]:
            r = 7 if lbl == "Tip" else 6
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#ffffff", outline=color, width=2)
            self.canvas.create_text(cx + 10, cy - 10, text=lbl, fill=color, anchor="w")

    # -------- 2-link arm helpers --------
    def arm_move_to_tip(self, x_mm: float, y_mm: float):
        try:
            from .kinematics import clamp_to_reach, ik_2link
        except Exception:
            from tools.arm_gui_tk.kinematics import clamp_to_reach, ik_2link

        x_mm, y_mm = clamp_to_reach(x_mm, y_mm, self.arm_base_mm, self.arm_L1, self.arm_L2)
        sol = ik_2link(x_mm, y_mm, self.arm_base_mm, self.arm_L1, self.arm_L2, elbow_up=self.arm_elbow_up)
        if sol is None:
            return
        self.arm_theta1, self.arm_theta2 = sol
        # Live dragging sends raw XY in mm (no calibration) to avoid surprises
        if self.grbl.state.connected and self.control_engaged:
            self.grbl.move_xy_safe(x_mm, y_mm, rapid=not self.live_pen_down)
        self.redraw()

    def arm_lock_from_grbl(self):
        pos = self.grbl.get_machine_position()
        if pos is None:
            self.update_status("No GRBL position available")
            return
        # Set digital arm pose to GRBL position without moving the machine
        try:
            from .kinematics import clamp_to_reach, ik_2link
        except Exception:
            from tools.arm_gui_tk.kinematics import clamp_to_reach, ik_2link
        x_mm, y_mm = pos
        x_mm, y_mm = clamp_to_reach(x_mm, y_mm, self.arm_base_mm, self.arm_L1, self.arm_L2)
        sol = ik_2link(x_mm, y_mm, self.arm_base_mm, self.arm_L1, self.arm_L2, elbow_up=self.arm_elbow_up)
        if sol is not None:
            self.arm_theta1, self.arm_theta2 = sol
            self.redraw()
            self.update_status("Arm pose synced to GRBL (no move)")
        else:
            self.update_status("Failed to sync pose: IK invalid for current base/lengths")

    def arm_set_base_here(self):
        x_mm, y_mm = self.canvas_to_mm(self.last_click_px[0], self.last_click_px[1])
        self.arm_base_mm = (x_mm, y_mm)
        self.update_status(f"Arm base set to ({x_mm:.1f},{y_mm:.1f})")
        self.redraw()

    def arm_set_base_px(self, x_px: float, y_px: float):
        x_mm, y_mm = self.canvas_to_mm(x_px, y_px)
        self.arm_base_mm = (x_mm, y_mm)
        self.redraw()

    def arm_toggle_elbow(self):
        self.arm_elbow_up = not self.arm_elbow_up
        self.update_status(f"Elbow mode: {'UP' if self.arm_elbow_up else 'DOWN'}")
        try:
            from .kinematics import fk_2link
        except Exception:
            from tools.arm_gui_tk.kinematics import fk_2link
        _, tip = fk_2link(self.arm_theta1, self.arm_theta2, self.arm_base_mm, self.arm_L1, self.arm_L2)
        self.arm_move_to_tip(*tip)

    # Control gating (safety): user must explicitly engage before moves are sent
    def engage_control(self):
        if not self.grbl.state.connected:
            messagebox.showwarning("Not connected", "Connect+Home first.")
            return
        # Force pen up for safety
        try:
            self.grbl.pen_up()
        except Exception:
            pass
        # Switch GRBL to relative for soft, small steps
        try:
            self.grbl.set_relative_mode(True)
            # Cap speed/feed conservatively
            self.grbl.set_feed(600)
            self.grbl.max_speed_mm_s = 4.0
            self.grbl.max_step_mm = 0.6
        except Exception:
            pass
        self.control_engaged = True
        self.update_status("Control engaged. Drag the tip to move. Pen is UP.")

    def disengage_control(self):
        self.control_engaged = False
        try:
            self.grbl.pen_up()
            self.grbl.set_relative_mode(False)
        except Exception:
            pass
        self.update_status("Control disengaged. Dragging only moves the virtual arm.")

    # Quick origin/scale helpers
    def origin_center(self):
        w = float(self.canvas.winfo_width() or 900)
        h = float(self.canvas.winfo_height() or 650)
        self.origin_px = (w / 2.0, h / 2.0)
        if self.grbl.state.connected:
            self.grbl.set_work_origin_here()
            self.update_status("Origin set to canvas center and GRBL origin set to current")
        else:
            self.update_status("Origin set to canvas center (not connected)")

    def reset_scale(self):
        self.mm_per_px = 1.0
        self.update_status("Scale reset: 1 mm per px")

    def toggle_flip_y(self):
        self.flip_y = not self.flip_y
        self.update_status(f"Flip Y is now {'ON' if self.flip_y else 'OFF'}")
        self._rel_last_px = None

    def scale_down(self):
        self.mm_per_px *= 0.8
        self.update_status(f"Scale set: {self.mm_per_px:.3f} mm/px (smaller)")

    def scale_up(self):
        self.mm_per_px *= 1.25
        self.update_status(f"Scale set: {self.mm_per_px:.3f} mm/px (larger)")

    def apply_repo_mapping(self):
        """Align GUI and GRBL to the repo's existing mapping:
        - Idle zone bounds from config (GRBL_IDLE_ZONE) → canvas scale
        - Optional origin offset like svg_to_gcode (-40,-40,0) via G92
        - Flip Y if needed
        """
        # 1) Read idle zone from config for scale/bounds
        try:
            from config.config import GRBL_IDLE_ZONE

            x_min, x_max, y_min, y_max = GRBL_IDLE_ZONE
            # Fit this mm area into 70% of canvas size
            w_px = float(self.canvas.winfo_width() or 900)
            h_px = float(self.canvas.winfo_height() or 650)
            area_w = x_max - x_min
            area_h = y_max - y_min
            # Keep aspect, choose uniform mm_per_px to fit
            mm_per_px_x = area_w / (w_px * 0.7)
            mm_per_px_y = area_h / (h_px * 0.7)
            self.mm_per_px = max(mm_per_px_x, mm_per_px_y)
            # Center origin visually
            self.origin_center()
            # Place base default after mapping
            self._place_base_default()
        except Exception:
            pass

        # Note: do NOT apply G92 origin offsets in live control to avoid surprises.
        self.update_status("Applied repo mapping: scaled to idle zone; no G92 offset in live mode")

    # ---------- G-code features ----------
    def load_gcode(self):
        path = filedialog.askopenfilename(filetypes=[("G-code", "*.gcode *.ngc *.nc *.tap"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            self.loaded_gcode_pts = parse_path_for_preview(lines)
            self.loaded_gcode_path = Path(path)
            self.update_status(f"Loaded G-code ({len(self.loaded_gcode_pts)} points): {Path(path).name}")
            self.redraw()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def save_corrected_gcode(self):
        if not self.calib:
            messagebox.showwarning("No calibration", "Compute or load a calibration first.")
            return
        if not self.loaded_gcode_path:
            messagebox.showwarning("No G-code", "Load a G-code file first.")
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".gcode", initialfile=f"{self.loaded_gcode_path.stem}_corrected.gcode")
        if not out_path:
            return
        try:
            H_inv = self.calib["H_inv"]

            def map_xy(x: float, y: float):
                return prewarp_points([(x, y)], H_inv)[0]

            transform_gcode_file(self.loaded_gcode_path, out_path, map_xy)
            messagebox.showinfo("Saved", f"Corrected G-code saved: {out_path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _draw_poly_px(self, pts, outline="#999", width=1, dash=None):
        if len(pts) < 2:
            return
        flat = []
        for x, y in pts:
            flat.extend([x, y])
        flat.extend([pts[0][0], pts[0][1]])
        self.canvas.create_line(*flat, fill=outline, width=width, dash=dash)

    def poll_machine_pos(self):
        pos = self.grbl.get_machine_position()
        if pos is not None:
            # Debounce small jitter to avoid flicker
            changed = (
                self.machine_pos_mm is None
                or abs(self.machine_pos_mm[0] - pos[0]) > 0.05
                or abs(self.machine_pos_mm[1] - pos[1]) > 0.05
            )
            self.machine_pos_mm = pos
            if changed:
                # Keep the virtual arm matched to the real tip position (no motion commands)
                try:
                    from .kinematics import clamp_to_reach, ik_2link
                except Exception:
                    from tools.arm_gui_tk.kinematics import clamp_to_reach, ik_2link
                x_mm, y_mm = pos
                x_mm, y_mm = clamp_to_reach(x_mm, y_mm, self.arm_base_mm, self.arm_L1, self.arm_L2)
                sol = ik_2link(x_mm, y_mm, self.arm_base_mm, self.arm_L1, self.arm_L2, elbow_up=self.arm_elbow_up)
                if sol is not None:
                    self.arm_theta1, self.arm_theta2 = sol
                self.redraw()
        self.root.after(200, self.poll_machine_pos)

    # Modes
    def set_mode_arm(self):
        self.mode = "arm"
        self.update_status("Mode: Arm. Drag the blue tip to move. Drag the green base to reposition the pivot.")
        self.redraw()

    def set_mode_calib(self):
        self.mode = "calib"
        self.update_status("Mode: Calibration. Click 4 printed corners in order TL,TR,BR,BL.")
        self.redraw()

    def run_loaded_gcode(self):
        if not self.loaded_gcode_path:
            messagebox.showwarning("No G-code", "Load a G-code file first.")
            return
        if not self.grbl.state.connected:
            messagebox.showwarning("Not connected", "Connect to GRBL first.")
            return
        try:
            # Safety: pen up
            self.grbl.pen_up()
            execute_gcode_file(self.grbl.ser, str(self.loaded_gcode_path))
            messagebox.showinfo("Done", "Loaded G-code executed.")
        except Exception as e:
            messagebox.showerror("Execution failed", str(e))

    def run_corrected_gcode(self):
        if not self.calib:
            messagebox.showwarning("No calibration", "Compute or load a calibration first.")
            return
        if not self.loaded_gcode_path:
            messagebox.showwarning("No G-code", "Load a G-code file first.")
            return
        if not self.grbl.state.connected:
            messagebox.showwarning("Not connected", "Connect to GRBL first.")
            return
        try:
            import tempfile

            H_inv = self.calib["H_inv"]

            def map_xy(x: float, y: float):
                return prewarp_points([(x, y)], H_inv)[0]

            with tempfile.NamedTemporaryFile(delete=False, suffix="_corrected.gcode") as tf:
                temp_path = tf.name
            transform_gcode_file(self.loaded_gcode_path, temp_path, map_xy)
            # Safety: pen up
            self.grbl.pen_up()
            execute_gcode_file(self.grbl.ser, temp_path)
            messagebox.showinfo("Done", "Corrected G-code executed.")
        except Exception as e:
            messagebox.showerror("Execution failed", str(e))


def main():
    root = Tk()
    app = HomographyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
