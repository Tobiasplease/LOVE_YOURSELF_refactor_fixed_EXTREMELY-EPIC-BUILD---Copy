"""Arm studio — side-by-side calibration of both arms into one shared frame.

    python -m motor_panel.arm_studio        (machine.py must be OFF)

LEFT: the machine's own camera (aim it down at the arms with the gaze
sliders — the view is oblique, that's fine, your eye does the projection).
RIGHT: an abstract SQUARE canvas. No workspace geometry is assumed — the
calibration decides all real limits. Two elbowed skeletons hang into
frame, matching what the camera sees: the gantry-driven arm on the
canvas-left, the servo arm on the canvas-right (as in the camera view).

Workflow, per arm:
  servo arm — drive it (sliders / P1-P5), drag its base/elbow/hand to
    match the camera, [capture servo arm]; 3+ poses, then [fit].
  gantry arm — jog it (⌂/mid/far or connect+home first), drag its
    skeleton to match, [capture gantry arm]; 3+ spread positions, [fit].
Then flip on VERIFY: both skeletons follow their models while you drive
the machines — if canvas tracks camera, the model is real. [save] writes
motor_panel/arm_model.json (the future collision floor reads it); the
live arm-vs-arm separation shows whenever both models exist.

Everything degrades gracefully with no hardware attached. Trust-test
build — deliberately separate from panel.py until proven.
"""

import json
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CAMERA_INDEX, LEFT_ARM_ELBOW_LIMITS, LEFT_ARM_SHOULDER_LIMITS, LEFT_ARM_WRIST_LIMITS
from motor_panel.arm_model import HAND_RADIUS, ArmModel, GantryArmModel, _two_link_elbow, arms_separation, load_models, save_models

VIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_studio_view.json")
WORLD = 100.0  # the square: 0..100 studio units each axis
MARGIN = 8.0  # separation floor drawn in verify (studio units)
POSES = [(70, 70), (70, 115), (95, 90), (115, 70), (110, 115)]  # (shoulder, elbow) spread
JOGS = [("⌂", (2, 2)), ("mid", (45, 18)), ("far", (95, -8))]  # gantry command units


class Camera:
    def __init__(self, label):
        self.label = label
        self.frame = None
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        try:
            import cv2

            cap = cv2.VideoCapture(CAMERA_INDEX)
        except Exception:
            cap = None
        while self.running:
            if cap is not None and cap.isOpened():
                ok, frm = cap.read()
                if ok:
                    self.frame = frm
            time.sleep(0.07)
        if cap is not None:
            cap.release()

    def tick(self):
        if self.frame is not None:
            try:
                import cv2
                from PIL import Image, ImageTk

                frm = cv2.cvtColor(cv2.resize(self.frame, (560, 315)), cv2.COLOR_BGR2RGB)
                self._img = ImageTk.PhotoImage(Image.fromarray(frm))
                self.label.config(image=self._img, text="")
            except Exception:
                pass


class Studio:
    def __init__(self, root):
        self.root = root
        root.title("arm studio — two arms, one frame")
        self.left_model, self.right_model = load_models()
        self.caps_left, self.caps_right = [], []
        self.verify = tk.BooleanVar(value=False)
        self.servo = {"shoulder": tk.DoubleVar(value=90.0), "elbow": tk.DoubleVar(value=90.0), "wrist": tk.DoubleVar(value=90.0)}
        self.gantry_cmd = [40.0, 15.0]
        # two draggable elbowed skeletons: gantry arm canvas-left, servo arm canvas-right
        self.skel = {
            "right": {"base": [22.0, 92.0], "elbow": [32.0, 62.0], "hand": [44.0, 38.0]},  # gantry arm (canvas-left)
            "left": {"base": [78.0, 92.0], "elbow": [68.0, 62.0], "hand": [56.0, 38.0]},  # servo arm (canvas-right)
        }
        self.hand_dev = None
        self.gaze = None
        self.gantry = None
        self._drag = None  # (arm_key, joint_key)
        self._grip = {"l1": 30.0, "l2": 30.0, "sign": 1.0, "rel": 0.0}  # rigid-drag state, cached at press
        self._load_view()
        self._build()
        self.cam = Camera(self.cam_label)
        self.cam.start()
        self._tick()

    # --- layout ---------------------------------------------------------------
    def _build(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.LabelFrame(main, text="machine view (oblique — your eye does the projection)")
        left.pack(side="left", fill="y", padx=4)
        self.cam_label = tk.Label(left, text="\n\n  no camera  \n\n", bg="#222", fg="#888", width=70, height=18)
        self.cam_label.pack(padx=4, pady=4)
        gz = ttk.Frame(left)
        gz.pack(fill="x", padx=4, pady=2)
        ttk.Button(gz, text="gaze", width=6, command=self._connect_gaze).pack(side="left")
        self.pan_v, self.tilt_v = tk.IntVar(value=self.view.get("pan", 80)), tk.IntVar(value=self.view.get("tilt", 65))
        for name, var in (("pan", self.pan_v), ("tilt", self.tilt_v)):
            ttk.Label(gz, text=name).pack(side="left", padx=(8, 2))
            ttk.Scale(gz, from_=0, to=180, variable=var, command=lambda _v, n=name: self._send_gaze(n)).pack(side="left", fill="x", expand=True)
        ttk.Button(gz, text="save aim", width=9, command=self._save_view).pack(side="left", padx=4)

        right = ttk.LabelFrame(main, text="canvas — abstract square, your calibration sets the limits")
        right.pack(side="left", fill="both", expand=True, padx=4)
        self.canvas = tk.Canvas(right, width=620, height=460, bg="#101014", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=4)
        dev = ttk.Frame(bar)
        dev.pack(fill="x")
        ttk.Button(dev, text="servo arm", width=10, command=self._connect_hand).pack(side="left")
        limits = {"shoulder": LEFT_ARM_SHOULDER_LIMITS, "elbow": LEFT_ARM_ELBOW_LIMITS, "wrist": LEFT_ARM_WRIST_LIMITS}
        for name in ("shoulder", "elbow", "wrist"):
            lo, hi = limits[name][0], limits[name][1]
            ttk.Label(dev, text=name).pack(side="left", padx=(10, 2))
            ttk.Scale(dev, from_=lo, to=hi, variable=self.servo[name], command=lambda _v, n=name: self._send_servo(n)).pack(
                side="left", fill="x", expand=True
            )
        pose_row = ttk.Frame(bar)
        pose_row.pack(fill="x", pady=3)
        for i, (s, e) in enumerate(POSES):
            ttk.Button(pose_row, text=f"P{i + 1}", width=4, command=lambda s=s, e=e: self._ease_to(s, e)).pack(side="left", padx=2)
        ttk.Button(pose_row, text="gantry connect+home", command=self._connect_gantry).pack(side="left", padx=(16, 2))
        for label, (gx, gy) in JOGS:
            ttk.Button(pose_row, text=label, width=4, command=lambda gx=gx, gy=gy: self._goto(gx, gy)).pack(side="left", padx=2)
        cal = ttk.Frame(bar)
        cal.pack(fill="x", pady=3)
        ttk.Button(cal, text="capture servo arm", command=self._capture_left).pack(side="left", padx=2)
        self.cap_l_lbl = ttk.Label(cal, text="0")
        self.cap_l_lbl.pack(side="left", padx=(2, 8))
        ttk.Button(cal, text="capture gantry arm", command=self._capture_right).pack(side="left", padx=2)
        self.cap_r_lbl = ttk.Label(cal, text="0")
        self.cap_r_lbl.pack(side="left", padx=(2, 8))
        ttk.Button(cal, text="fit", command=self._fit).pack(side="left", padx=2)
        self.fit_lbl = ttk.Label(cal, text=self._fit_state())
        self.fit_lbl.pack(side="left", padx=6)
        ttk.Checkbutton(cal, text="verify (skeletons follow the machines)", variable=self.verify).pack(side="left", padx=10)
        ttk.Button(cal, text="save", command=self._save_model).pack(side="left", padx=2)
        self.status = ttk.Label(bar, text="drive an arm, drag its skeleton to match the camera, capture. 3+ each, fit, verify, save.")
        self.status.pack(fill="x", pady=2)

    def _fit_state(self):
        return f"models: servo {'✓' if self.left_model else '—'}  gantry {'✓' if self.right_model else '—'}"

    # --- world <-> px (fixed square) ----------------------------------------------
    def _bounds(self):
        w = max(self.canvas.winfo_width(), 200)
        h = max(self.canvas.winfo_height(), 200)
        s = min(w, h) / (WORLD + 10)
        return -5.0, -5.0, s, h

    def _to_px(self, x, y):
        x0, y0, s, h = self._bounds()
        return (x - x0) * s, h - (y - y0) * s

    def _unproject(self, px, py):
        x0, y0, s, h = self._bounds()
        return px / s + x0, (h - py) / s + y0

    # --- drawing ----------------------------------------------------------------
    def _draw(self):
        c = self.canvas
        c.delete("all")
        _x0, _y0, s, _h = self._bounds()
        tl, br = self._to_px(0, WORLD), self._to_px(WORLD, 0)
        c.create_rectangle(*tl, *br, outline="#345", width=2)
        for g in range(10, 100, 10):  # faint grid
            a, b = self._to_px(g, 0), self._to_px(g, WORLD)
            c.create_line(*a, *b, fill="#1a1a22")
            a, b = self._to_px(0, g), self._to_px(WORLD, g)
            c.create_line(*a, *b, fill="#1a1a22")

        verify = self.verify.get()
        sep = None
        if verify and self.left_model is not None and self.right_model is not None:
            sep = arms_separation(self.left_model, self.right_model, self.servo["shoulder"].get(), self.servo["elbow"].get(), *self.gantry_cmd)

        for arm, color in (("right", "#8ac"), ("left", "#ddd")):
            if verify:
                model = self.left_model if arm == "left" else self.right_model
                if model is None:
                    continue
                if arm == "left":
                    elbow, hand = model.fk(self.servo["shoulder"].get(), self.servo["elbow"].get())
                else:
                    elbow, hand = model.fk(*self.gantry_cmd)
                base = model.base
            else:
                sk = self.skel[arm]
                base, elbow, hand = sk["base"], sk["elbow"], sk["hand"]
            b, e, hd = self._to_px(*base), self._to_px(*elbow), self._to_px(*hand)
            hand_color = color if sep is None else ("#4d4" if sep > MARGIN else "#d44")
            c.create_line(*b, *e, fill=color, width=4)
            c.create_line(*e, *hd, fill=color, width=4)
            c.create_rectangle(b[0] - 6, b[1] - 6, b[0] + 6, b[1] + 6, outline=color, width=2)
            c.create_oval(e[0] - 6, e[1] - 6, e[0] + 6, e[1] + 6, outline=color, width=2)
            r = HAND_RADIUS * s
            c.create_oval(hd[0] - r, hd[1] - r, hd[0] + r, hd[1] + r, outline=hand_color, width=2)
        c.create_text(*self._to_px(22, 96), fill="#568", text="gantry arm", font=("TkFixedFont", 9))
        c.create_text(*self._to_px(78, 96), fill="#888", text="servo arm", font=("TkFixedFont", 9))
        if sep is not None:
            col = "#4d4" if sep > MARGIN else "#d44"
            c.create_text(8, 8, anchor="nw", fill=col, text=f"arm separation {sep:.1f} (floor {MARGIN:.0f})", font=("TkFixedFont", 10))
        elif not verify:
            c.create_text(
                8,
                8,
                anchor="nw",
                fill="#888",
                text="drag: hand ◯ = IK · elbow ○ = swing · base □ = move arm · shift-drag stretches links",
                font=("TkFixedFont", 9),
            )

    def _press(self, ev):
        if self.verify.get():
            return
        best, hit = 18.0, None
        for arm in ("left", "right"):
            for joint in ("hand", "elbow", "base"):
                px, py = self._to_px(*self.skel[arm][joint])
                d = math.hypot(px - ev.x, py - ev.y)
                if d < best:
                    best, hit = d, (arm, joint)
        self._drag = hit
        if hit:
            sk = self.skel[hit[0]]
            b, e, h = sk["base"], sk["elbow"], sk["hand"]
            cross = (h[0] - b[0]) * (e[1] - b[1]) - (h[1] - b[1]) * (e[0] - b[0])
            self._grip = {
                "l1": max(1.0, math.hypot(e[0] - b[0], e[1] - b[1])),
                "l2": max(1.0, math.hypot(h[0] - e[0], h[1] - e[1])),
                "sign": 1.0 if cross >= 0 else -1.0,
                "rel": math.atan2(h[1] - e[1], h[0] - e[0]) - math.atan2(e[1] - b[1], e[0] - b[0]),
            }

    def _motion(self, ev):
        if not self._drag:
            return
        arm, joint = self._drag
        wx, wy = self._unproject(ev.x, ev.y)
        wx, wy = max(0.0, min(WORLD, wx)), max(0.0, min(WORLD, wy))
        sk, g = self.skel[arm], self._grip
        if ev.state & 0x0001:  # shift-drag: free stretch — the only way lengths change
            sk[joint][:] = [wx, wy]
            return
        if joint == "base":  # the whole arm translates rigidly
            dx, dy = wx - sk["base"][0], wy - sk["base"][1]
            for j in ("base", "elbow", "hand"):
                sk[j][:] = [sk[j][0] + dx, sk[j][1] + dy]
        elif joint == "hand":  # IK: target clamped to the reach annulus, elbow follows
            b = sk["base"]
            d = math.hypot(wx - b[0], wy - b[1])
            lo, hi = abs(g["l1"] - g["l2"]) + 0.5, g["l1"] + g["l2"] - 0.01
            if d > 1e-6 and not (lo <= d <= hi):
                f = max(lo, min(hi, d)) / d
                wx, wy = b[0] + (wx - b[0]) * f, b[1] + (wy - b[1]) * f
            sk["hand"][:] = [wx, wy]
            sk["elbow"][:] = list(_two_link_elbow(b, (wx, wy), g["l1"], g["l2"], g["sign"]))
        elif joint == "elbow":  # swing: elbow orbits the base, forearm keeps its bend
            b = sk["base"]
            d = math.hypot(wx - b[0], wy - b[1])
            if d < 1e-6:
                return
            ux, uy = (wx - b[0]) / d, (wy - b[1]) / d
            sk["elbow"][:] = [b[0] + g["l1"] * ux, b[1] + g["l1"] * uy]
            t2 = math.atan2(uy, ux) + g["rel"]
            sk["hand"][:] = [sk["elbow"][0] + g["l2"] * math.cos(t2), sk["elbow"][1] + g["l2"] * math.sin(t2)]

    # --- calibration actions ------------------------------------------------------
    def _capture_left(self):
        sk = self.skel["left"]
        self.caps_left.append(
            {
                "servo_shoulder": self.servo["shoulder"].get(),
                "servo_elbow": self.servo["elbow"].get(),
                "base": tuple(sk["base"]),
                "elbow": tuple(sk["elbow"]),
                "hand": tuple(sk["hand"]),
            }
        )
        self.cap_l_lbl.config(text=str(len(self.caps_left)))
        self.status.config(text=f"servo arm captured at shoulder {self.servo['shoulder'].get():.0f} / elbow {self.servo['elbow'].get():.0f}")

    def _capture_right(self):
        sk = self.skel["right"]
        self.caps_right.append({"cmd": tuple(self.gantry_cmd), "base": tuple(sk["base"]), "elbow": tuple(sk["elbow"]), "hand": tuple(sk["hand"])})
        self.cap_r_lbl.config(text=str(len(self.caps_right)))
        self.status.config(text=f"gantry arm captured at command ({self.gantry_cmd[0]:.0f}, {self.gantry_cmd[1]:.0f})")

    def _fit(self):
        msgs = []
        if len(self.caps_left) >= 3:
            try:
                self.left_model, r = ArmModel.fit(self.caps_left)
                msgs.append(f"servo residual {r:.1f}")
            except Exception as e:
                msgs.append(f"servo fit failed: {e}")
        else:
            msgs.append(f"servo needs {3 - len(self.caps_left)} more")
        if len(self.caps_right) >= 3:
            try:
                self.right_model, r = GantryArmModel.fit(self.caps_right)
                msgs.append(f"gantry residual {r:.1f}")
            except Exception as e:
                msgs.append(f"gantry fit failed: {e}")
        else:
            msgs.append(f"gantry needs {3 - len(self.caps_right)} more")
        self.fit_lbl.config(text=self._fit_state())
        self.status.config(text=" · ".join(msgs) + " — verify: do the skeletons track the camera?")

    def _save_model(self):
        if self.left_model is None and self.right_model is None:
            self.status.config(text="nothing to save — fit first")
            return
        save_models(self.left_model, self.right_model, extra={"captures_left": self.caps_left, "captures_right": self.caps_right})
        self._save_view()
        self.status.config(text="saved motor_panel/arm_model.json")

    # --- hardware (all optional) --------------------------------------------------
    def _connect_hand(self):
        def go():
            try:
                from motor_panel.devices import build_devices

                self.hand_dev = build_devices()[1]
                self.status.config(text=f"servo arm: {self.hand_dev.connect()}")
            except Exception as e:
                self.status.config(text=f"servo arm connect failed: {e}")

        threading.Thread(target=go, daemon=True).start()

    def _send_servo(self, name):
        if self.hand_dev is not None:
            try:
                self.hand_dev.set_channel(name, self.servo[name].get())
            except Exception:
                pass

    def _ease_to(self, s, e, seconds=2.0):
        def go():
            s0, e0 = self.servo["shoulder"].get(), self.servo["elbow"].get()
            steps = int(seconds / 0.05)
            for i in range(1, steps + 1):
                f = i / steps
                self.servo["shoulder"].set(s0 + (s - s0) * f)
                self.servo["elbow"].set(e0 + (e - e0) * f)
                self._send_servo("shoulder")
                self._send_servo("elbow")
                time.sleep(0.05)

        threading.Thread(target=go, daemon=True).start()

    def _connect_gaze(self):
        def go():
            try:
                from servo_control.servo_control import ServoController

                self.gaze = ServoController(port="/dev/arduino_lunggaze")
                self._send_gaze("pan")
                self._send_gaze("tilt")
                self.status.config(text="gaze connected — aim the camera at the arms")
            except Exception as e:
                self.status.config(text=f"gaze connect failed: {e}")

        threading.Thread(target=go, daemon=True).start()

    def _send_gaze(self, which):
        if self.gaze is not None:
            try:
                (self.gaze.set_pan if which == "pan" else self.gaze.set_tilt)(int((self.pan_v if which == "pan" else self.tilt_v).get()))
            except Exception:
                pass

    def _connect_gantry(self):
        def go():
            try:
                from motor_panel.gantry import GantryLink

                self.gantry = GantryLink()
                self.gantry.on_log = lambda m: self.status.config(text=m)
                self.gantry.connect_and_home()
            except Exception as e:
                self.status.config(text=f"gantry failed: {e}")

        threading.Thread(target=go, daemon=True).start()

    def _goto(self, x, y):
        self.gantry_cmd[:] = [x, y]
        if self.gantry is not None and self.gantry.alive:
            self.gantry.goto(x, y, dt=1.2)

    # --- view persistence ----------------------------------------------------------
    def _load_view(self):
        try:
            with open(VIEW_PATH) as f:
                self.view = json.load(f)
            for arm in ("left", "right"):
                if arm in self.view.get("skel", {}):
                    for j, v in self.view["skel"][arm].items():
                        self.skel[arm][j][:] = v
        except Exception:
            self.view = {}

    def _save_view(self):
        with open(VIEW_PATH, "w") as f:
            json.dump({"pan": int(self.pan_v.get()), "tilt": int(self.tilt_v.get()), "skel": self.skel}, f, indent=1)

    def _tick(self):
        self.cam.tick()
        self._draw()
        self.root.after(80, self._tick)


def main():
    root = tk.Tk()
    root.geometry("1380x720")
    Studio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
