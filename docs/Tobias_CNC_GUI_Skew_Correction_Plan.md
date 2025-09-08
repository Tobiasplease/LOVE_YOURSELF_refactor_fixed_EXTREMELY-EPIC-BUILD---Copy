# Tobias Plan for CNC GUI and Skew Correction

This document outlines a practical, end‑to‑end plan to build a Tkinter GUI that models a 2‑link (~300 mm each) drawing arm from a top‑down view, enables cursor‑drag control of the pen, defines a large test area, draws squares/grids, and performs calibration to pre‑distort motion so printed shapes are straight and square on paper. It integrates with the existing GRBL utilities in this repository.

---

## Goals

- Visual 2D arm simulator with fixed link lengths and realistic constraints.
- Mouse‑drag end effector; elbow auto‑solves via inverse kinematics (IK).
- “Lock Home” (zero offsets) + “Lock Limits” (large soft‑limit box).
- Square/grid tool with ideal vs. corrected preview.
- Skew correction: start with homography (4 points), then thin‑plate spline (TPS) or 2D polynomial from a sparse grid.
- Optional live send to GRBL using repo helpers; safe feed/pen handling.
- Persist calibration and machine state to JSON; reload across sessions.

## What The App Can Correct

- Global skew (rotation/scale/shear): affine/homography.
- Smooth curvature (like the consistent bow in the reference photo): TPS or low‑order polynomial fit.
- Workspace “truthing”: only correct within a user‑defined soft‑limit rectangle for stable accuracy.

Limits: Direction‑dependent backlash and non‑repeatable pen compliance need mechanical mitigation (approach direction lock, speed consistency, stiff pen mount).

## Geometry & Kinematics

- Links: `L1 ≈ 300 mm`, `L2 ≈ 300 mm` (editable in UI).
- Base pivot: fixed world position `(x0, y0)` (settable; drawn on canvas).
- Forward kinematics (FK):
  - `x = x0 + L1 cos θ1 + L2 cos(θ1+θ2)`
  - `y = y0 + L1 sin θ1 + L2 sin(θ1+θ2)`
- Inverse kinematics (IK): closed‑form for 2‑link, elbow‑up/down toggle; unreachable targets clamped to reachable boundary; solution continuity (choose nearest to last θ).
- Angle offsets: store θ_home; runtime commands use `(θ − θ_home)`; show live θ and (x,y).

## Coordinate Systems & Scale

- Canvas pixels ↔ mm: fixed scale per document or calibrated by a drag‑to‑measure tool (e.g., validate that 300 mm link ≙ displayed segment).
- World origin near base; draw axes and mm grid. Large workspace (≥ 1000×1000 mm view) with zoom/pan.

## UI Design (Tkinter)

- Canvas (center):
  - Arm: base, elbow, tip, link segments, reach annulus.
  - Draggable tip handle (primary), optional draggable base (locked by default).
  - Soft‑limit rectangle (draggable/resize) to "lock" safe area.
  - Square/grid overlay: move (Ctrl‑drag), rotate (Shift‑drag), resize via handles.
- Right panel controls:
  - Link lengths, base position, elbow mode.
  - Home: Capture current θ, Set/Clear, show offsets.
  - Limits: Capture from drawn box; enable/disable enforcement; save.
  - Square/Grid: side length, count/spacing, rotation; preview vs. corrected.
  - Calibration: start 4‑pt homography; advanced grid (TPS); error metrics; Save/Load calibration JSON.
  - GRBL: port (auto), connect, home, feed, pen S‑values, send preview, stop.
- Status bar: θ1/θ2, (x,y) mm, reach/limit warnings, serial status.

## Skew Correction Workflow

1) Quick: Homography (4 points)
   - Draw a reference square (e.g., 200×200 mm) with current calibration (or none).
   - Click the four printed corners in the GUI (ideal → measured). Fit H.
   - For drawing, apply pre‑distortion with `H⁻¹` to target points. Preview corrected vs. ideal.

2) Accurate: Grid‑based TPS / Polynomial
   - Draw a sparse grid (e.g., 5×5) over the soft‑limit area.
   - For each ideal grid node, click its printed location (or type measured offsets). Fit:
     - TPS (preferred): smooth, non‑rigid fit for curvature.
     - Or 2D polynomial (order 2–3) if TPS dependencies are undesirable.
   - Save calibration JSON (type, params, extents). Apply inverse mapping during path generation.

3) Verification
   - Draw a fresh square and centerlines using the calibration.
   - Compute RMS/max error from user‑clicked measurements; display heatmap (optional).

Expected outcome: with consistent, repeatable mechanics the TPS calibration should reduce line bow to ≤ 0.5–1.0 mm over a 200–300 mm field, rising near edges; use soft‑limits to constrain work area.

## GRBL Integration (reuse repo utilities)

- Utilities referenced:
  - `grbl/grbl_utils.py`: `find_grbl_port`, `ensure_homed`, `setup_basic_grbl`, `send_cmd`, `wait_until_idle`, `get_status`, `DEFAULT_FEED_RATE`, `PEN_UP_CMD`.
  - Patterns from `grbl/run_idle_movements.py` and `grbl/idle_movements.py` for pacing and safety.
- Live mode flow:
  1. Connect: `find_grbl_port(preferred=GRBL_CNC_PORT)`.
  2. Home and basic setup (absolute positioning, feed).
  3. Pen up; jog to a safe start inside soft‑limits.
  4. Generate target polyline (square/grid) in mm → pre‑distort via calibration → emit `G1 X.. Y.. F..` with pacing similar to idle sender.
  5. Pen down/up via configurable `M3 Sxx` values; throttle; periodic `wait_until_idle`.
  6. Stop: feed hold `!`, pen up, jog to safe point, close serial.
- Dry‑run: optionally export `.ngc` without sending.

## Module Structure

```
tools/arm_gui_tk/
  app.py                # Tk root, Canvas, bindings, layout
  kinematics.py         # FK/IK, limits, continuity
  pathgen.py            # square/grid generation, resampling
  calibration.py        # homography + TPS/polynomial; save/load JSON
  correct.py            # apply inverse calibration to paths
  grbl_link.py          # wraps grbl_utils; streaming, safety
  state.py              # save/load home, limits, defaults
configs/arm_default.json
```

## Core Algorithms (notes)

- IK: for target (x, y) relative to base:
  - `c2 = (r² − L1² − L2²) / (2 L1 L2)`; `s2 = ±sqrt(1−c2²)` (elbow mode);
  - `θ2 = atan2(s2, c2)`; `θ1 = atan2(y, x) − atan2(L2 s2, L1 + L2 c2)`.
  - Clamp to joint limits; choose solution nearest last θ for smoothness.
- Homography: 4 point pairs → DLT → `H` (normalize; invert for pre‑distort).
- TPS: fit radial basis mapping `(x,y) → (X,Y)` using control points; compute inverse mapping numerically or fit reverse TPS.

## Milestones

1. Simulator (M1)
   - Large canvas with mm grid, zoom/pan.
   - FK/IK drag of tip; elbow toggle; reach boundary.
   - Home capture; soft‑limit rectangle; status bar.

2. Square Tool (M2)
   - Draw/preview square; warnings for reach/limits; save session.

3. GRBL Live (M3)
   - Connect/home; pen up/down; send preview path; stop safely.

4. Calibration (M4)
   - Homography wizard; corrected preview/send; save/load calibration.

5. Advanced Calibration (M5)
   - Grid TPS/polynomial; verification metrics; heatmap.

6. QoL (M6)
   - Approach‑direction lock; backlash take‑up; export `.ngc`.

## Safety & Diagnostics

- Always render reach feasibility and soft‑limit checks separately.
- Status shows serial port, last `OK`, queued moves.
- Emergency stop button: send `!`, pen up, return to safe point.
- Dry‑run and slow‑mode toggles.

## Defaults & Config

- Link lengths: `L1=L2=300 mm` (editable).
- Soft‑limits: start generous (e.g., 0..400 mm square) then lock your measured range.
- Pen S‑values: default `up=S15`, `down=S50` (match your setup).
- Feeds: default from `DEFAULT_FEED_RATE`, override per path.

## Open Questions

- Exact pen up/down S‑values for your Robottini setup?
- Do we ever want arcs (G2/G3) for long edges, or keep all linear?
- Preferred verification square size (e.g., 200 mm) and grid density for TPS?

## Expected Result

Given the very consistent skew you report, the homography+TPS calibration workflow should reliably straighten lines inside the locked area. You’ll be able to iterate quickly: draw → measure/click → fit → preview → send corrected. Persist the calibration JSON and reuse it as long as the mechanical setup remains unchanged.

