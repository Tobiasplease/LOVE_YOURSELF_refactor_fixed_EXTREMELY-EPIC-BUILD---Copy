# Skew Calibration Quickstart (4‑Point Homography MVP)

This minimal tool helps you correct paper/sketch skew by measuring 4 printed corners and generating a pre‑distortion mapping.

## Run

1) Ensure Python + Tkinter + NumPy are installed (see `requirements.txt`).
2) From repo root, run:

```
python -m tools.arm_gui_tk.app  # or: python3 -m tools.arm_gui_tk.app
# or from repo root:
python skew_calibration.py
```

## Workflow

- Click `New 4‑pt Calib`.
- Click the four printed corners in order: TL → TR → BR → BL.
- Click `Compute`.
- Click `Preview`: 
  - Green: ideal square (target)
  - Red: your measured printed quadrilateral
  - Blue: pre‑warped square you would send to the machine
  - Cyan dashed: simulated print if the world applies your skew to the blue path
- Click `Save` to store `calibration/homography_calibration.json`.

## Live Control (GRBL)

- Click `Connect+Home` to connect to your GRBL controller and home it.
- Use `Pen Up` / `Pen Down` to control the pen servo.
- Drag on the canvas to jog in realtime (throttled ~20 Hz). If a calibration is loaded, drag destinations are pre‑warped automatically.
- Use `Origin=Center` to set the canvas origin to center and set GRBL work origin (G54) to the current position (so center ≙ X0 Y0). Use `Reset Scale(1mm/px)` to reset scale.
- If your Y moves the opposite way, click `Flip Y` to invert the Y axis mapping.
- Click `Disconnect` to close the serial port.

## How To Use The Calibration

- Pre‑warp any toolpath points `(x, y)` with `H_inv` from the JSON. 
- If you use the provided code, call `prewarp_points(points, H_inv)` in `tools/arm_gui_tk/correct.py`.

## Notes

- Point order matters. Use TL, TR, BR, BL consistently.
- Keep clicks inside your stable drawing area; outside extrapolation may be inaccurate.
- You can `Load` the saved JSON later and preview again.
- Live mapping is simplified (1 px = 1 mm, Y not inverted). We can refine together to match your exact machine coordinates.
