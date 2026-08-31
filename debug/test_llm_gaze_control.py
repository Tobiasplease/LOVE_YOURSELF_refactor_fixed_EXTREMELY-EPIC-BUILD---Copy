#!/usr/bin/env python3
"""
LLM-DIRECTED ZONE-BASED GAZE CONTROL

The LLM decides WHICH ZONE to look at (left/right/up/down/ahead).
Organic Perlin movement adds natural variance WITHIN that zone.
Face tracking OVERRIDES zone control when a person is detected.

Zone System:
- Pan zones: left (45-75°), ahead (75-105°), right (105-135°)
- Tilt zones: up (65-95°), level (95-125°), down (125-150°)

Flow:
1. LLM sees frame → outputs "LOOK: left" (or right/up/down/ahead/person)
2. Gaze system moves toward that zone with organic Perlin variance
3. If face detected → tracking overrides zone control
4. LLM queries every 3 seconds

Run: python debug/test_llm_gaze_control.py

Controls:
  q - Quit
  r - Reset to center, disable LLM zone
  l - Manual: set zone LEFT
  a - Manual: set zone AHEAD
  h - Manual: set zone RIGHT
  u - Manual: set zone UP
  d - Manual: set zone DOWN
  p - Force person detection (debug)
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CAMERA_INDEX, PAN_MAX, PAN_MIN, TILT_MAX, TILT_MIN
from perception.person_detection_state import get_person_detection_state
from vision import gaze
from vision.spatial_awareness import get_spatial_awareness_engine

# YOLO (optional)
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[!] YOLO not available - running without person detection")

# Face detection
import numpy as np

# === CONFIGURATION ===
LLM_QUERY_INTERVAL = 3.0  # Seconds between LLM zone decisions
SHOW_VIDEO = True

# Face detection model paths
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"


class GazeController:
    def __init__(self, use_hardware: bool = True, llm_interval: float = 3.0):
        self.use_hardware = use_hardware
        self.servos = None
        self.camera = None
        self.yolo = None
        self.face_net = None
        self.person_state = get_person_detection_state()
        self.spatial_engine = get_spatial_awareness_engine()
        self.spatial_engine.query_interval = llm_interval
        self.spatial_engine.debug = True

        # Gaze state
        self.pan = 90
        self.tilt = 90

        # State change tracking
        self._last_gaze_state = "idle"

    def init_hardware(self):
        """Initialize camera and servos."""
        # Camera
        print("[📷] Initializing camera...")
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print("[!] Camera failed - trying index 0")
            self.camera = cv2.VideoCapture(0)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[✓] Camera initialized")

        # Servos
        if self.use_hardware:
            try:
                from config.config import SERIAL_PORT
                from servo_control.servo_control import ServoController

                self.servos = ServoController(port=SERIAL_PORT)
                if self.servos.ser is None:
                    print("[!] Servo serial connection failed")
                    self.use_hardware = False
                else:
                    print("[✓] Servos connected")
            except Exception as e:
                print(f"[!] Servo init failed: {e}")
                self.use_hardware = False

        # YOLO
        if YOLO_AVAILABLE:
            print("[🔍] Loading YOLO model...")
            self.yolo = YOLO("yolov8n.pt")
            print("[✓] YOLO loaded")

        # Face detection
        print("[👤] Loading face detection model...")
        try:
            self.face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
            print("[✓] Face detection loaded")
        except Exception as e:
            print(f"[!] Face detection failed: {e}")
            self.face_net = None

        # Reset gaze to center
        gaze.servo_x = 90
        gaze.servo_y = 90
        gaze.state = "idle"
        gaze.llm_zone_active = False

        return True

    def detect_face(self, frame) -> tuple:
        """Run face detection on frame. Returns (face_box, confidence) or (None, 0)."""
        if self.face_net is None:
            return None, 0

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        best_box = None
        best_conf = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_box = box.astype("int")
                best_conf = confidence

        return best_box, best_conf

    def move_servos(self, pan: int, tilt: int):
        """Move servos to position."""
        self.pan = pan
        self.tilt = tilt
        if self.servos:
            try:
                self.servos.set_pan(pan)
                self.servos.set_tilt(tilt)
            except Exception as e:
                print(f"[SERVO ERROR] {e}")

    def detect_person(self, frame):
        """Run YOLO detection on frame."""
        if not YOLO_AVAILABLE or self.yolo is None:
            return False, None

        results = self.yolo(frame, verbose=False, classes=[0])

        for r in results:
            for box in r.boxes:
                if box.cls[0] == 0:
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        return True, box.xyxy[0].cpu().numpy()

        return False, None

    def draw_overlay(self, frame, person_detected: bool, person_box, face_box=None):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        ps = self.person_state.get_person_state()
        status = self.spatial_engine.get_status()

        # Draw person box if detected (YOLO - green)
        if person_box is not None:
            x1, y1, x2, y2 = map(int, person_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "YOLO", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw face box if detected (cyan)
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "FACE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Status panel background
        cv2.rectangle(frame, (5, 5), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (400, 200), (0, 255, 0), 1)

        # Title
        cv2.putText(frame, "LLM-DIRECTED ZONE GAZE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Current zone (where gaze is looking)
        zone_text = gaze.get_current_zone_text()
        cv2.putText(frame, f"Looking: {zone_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # LLM target zone
        llm_zone = f"{gaze.llm_target_zone_pan}/{gaze.llm_target_zone_tilt}"
        active = "ACTIVE" if gaze.llm_zone_active else "inactive"
        zone_color = (0, 255, 0) if gaze.llm_zone_active else (100, 100, 100)
        cv2.putText(frame, f"LLM zone: {llm_zone} ({active})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)

        # Gaze state
        gaze_state = gaze.state.upper()
        gaze_colors = {
            "IDLE": (150, 150, 150),
            "TRACKING": (0, 255, 255),
            "GRACE": (200, 200, 0),
            "AWARE": (255, 165, 0),
        }
        gaze_color = gaze_colors.get(gaze_state, (255, 255, 255))
        cv2.putText(frame, f"Gaze state: {gaze_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)

        # Servo position
        cv2.putText(frame, f"Servos: pan={self.pan:.0f} tilt={self.tilt:.0f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Person state
        person_state_str = ps.get("person_state", "absent")
        person_color = (0, 255, 0) if ps["is_present"] else (100, 100, 100)
        cv2.putText(frame, f"Person: {person_state_str}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 1)

        # Tracking context (if tracking)
        if gaze.state == "tracking":
            tracking_ctx = gaze.get_tracking_context()
            cv2.putText(frame, f"Tracking: {tracking_ctx}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Last LLM response
        if status["last_response"]:
            response = status["last_response"][:50]
            cv2.putText(frame, f"LLM: {response}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)

        # Last LLM direction
        cv2.putText(frame, f"LLM says: {status['last_direction'].upper()}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)

        # Zone indicator bar at bottom
        zone_width = w // 3
        zones = [("left", 0), ("ahead", 1), ("right", 2)]
        for zone_name, idx in zones:
            x_start = idx * zone_width
            color = (0, 100, 0)
            if gaze.llm_target_zone_pan == zone_name and gaze.llm_zone_active:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x_start, h - 30), (x_start + zone_width, h - 5), color, -1)
            cv2.putText(frame, zone_name.upper(), (x_start + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def run(self):
        """Main loop."""
        print("\n" + "=" * 60)
        print("  LLM-DIRECTED ZONE-BASED GAZE CONTROL")
        print("=" * 60)
        print("\nHOW IT WORKS:")
        print("  1. LLM queries Ollama every few seconds with current frame")
        print("  2. LLM decides zone: left / right / ahead / up / down / person")
        print("  3. Gaze moves toward zone with organic Perlin variance")
        print("  4. Face tracking OVERRIDES zone control when detected")
        print("\nZONES:")
        print(f"  Pan:  left(45-75) ahead(75-105) right(105-135)")
        print(f"  Tilt: up(65-95) level(95-125) down(125-150)")
        print("\nControls:")
        print("  q=quit  r=reset  l=left  a=ahead  h=right  u=up  d=down  p=force person")
        print("=" * 60 + "\n")

        if not self.init_hardware():
            print("[FATAL] Hardware init failed")
            return

        # Center servos
        self.move_servos(90, 90)
        time.sleep(0.5)

        # Start spatial awareness engine
        self.spatial_engine.start()
        print(f"[🧠] Spatial awareness started - LLM queries every {self.spatial_engine.query_interval}s\n")

        frame_count = 0

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                frame_count += 1

                # Feed frame to spatial awareness engine
                self.spatial_engine.update_frame(frame)

                # Run YOLO detection
                person_detected, person_box = self.detect_person(frame)

                # Run face detection (for tracking)
                face_box, face_conf = self.detect_face(frame)
                face_detected = face_box is not None

                # Update person detection state
                self.person_state.update_yolo_detection(person_detected, 0.9 if person_detected else 0.0)
                if face_detected:
                    self.person_state.update_face_detection(face_conf, tuple(face_box))

                # Get smoothed person state
                ps = self.person_state.get_person_state()
                smoothed_person = ps.get("is_present", False)

                # Update spatial awareness with person info
                if smoothed_person and person_box is not None:
                    x1, y1, x2, y2 = map(int, person_box)
                    frame_width = frame.shape[1]
                    center_x = (x1 + x2) / 2
                    frame_center = frame_width / 2

                    if center_x < frame_center - 100:
                        position = "left"
                    elif center_x > frame_center + 100:
                        position = "right"
                    else:
                        position = "center"

                    self.spatial_engine.update_person_state(True, position, gaze.tracking_person_movement)
                else:
                    self.spatial_engine.update_person_state(False)

                # Update servo position in person state
                self.person_state.update_servo_position(self.pan, self.tilt)

                # Update gaze
                _, new_pan, new_tilt = gaze.update_gaze(
                    frame,
                    face_box=tuple(face_box) if face_detected else None,
                    current_emotion_state="calm_observant",
                    yolo_person_detected=smoothed_person,
                    person_direction=ps.get("direction"),
                )

                # Move physical servos
                self.move_servos(new_pan, new_tilt)

                # Print status on state change
                current_gaze_state = gaze.state
                if current_gaze_state != self._last_gaze_state:
                    zone_text = gaze.get_current_zone_text()
                    status = self.spatial_engine.get_status()
                    print(f"[GAZE] {self._last_gaze_state} → {current_gaze_state} | Looking: {zone_text} | LLM→{status['last_direction']}")
                    self._last_gaze_state = current_gaze_state

                # Periodic status
                if frame_count % 90 == 0:
                    zone_text = gaze.get_current_zone_text()
                    status = self.spatial_engine.get_status()
                    llm_zone = f"{gaze.llm_target_zone_pan}/{gaze.llm_target_zone_tilt}"
                    active = "ON" if gaze.llm_zone_active else "off"
                    print(
                        f"[STATUS] {gaze.state:8s} | Looking: {zone_text:12s} | LLM zone: {llm_zone} ({active}) | pan={new_pan:.0f}° tilt={new_tilt:.0f}°"
                    )

                # Draw overlay and show
                if SHOW_VIDEO:
                    display = self.draw_overlay(frame.copy(), person_detected, person_box, face_box)
                    cv2.imshow("LLM Zone Gaze", display)

                # Handle keys
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    print("[RESET] Returning to center, disabling LLM zone")
                    gaze.servo_x = 90
                    gaze.servo_y = 90
                    gaze.state = "idle"
                    gaze.llm_zone_active = False
                    gaze.llm_target_zone_pan = "ahead"
                    gaze.llm_target_zone_tilt = "level"
                    self.move_servos(90, 90)
                elif key == ord("l"):
                    print("[MANUAL] Zone → LEFT")
                    gaze.set_llm_zone("left")
                elif key == ord("a"):
                    print("[MANUAL] Zone → AHEAD")
                    gaze.set_llm_zone("ahead")
                elif key == ord("h"):
                    print("[MANUAL] Zone → RIGHT")
                    gaze.set_llm_zone("right")
                elif key == ord("u"):
                    print("[MANUAL] Zone → UP")
                    gaze.set_llm_zone("ahead", "up")
                elif key == ord("d"):
                    print("[MANUAL] Zone → DOWN")
                    gaze.set_llm_zone("ahead", "down")
                elif key == ord("p"):
                    print("[DEBUG] Forcing person detection")
                    self.person_state.update_yolo_detection(True, 0.95)
                    self.person_state.update_face_detection(0.9, (280, 200, 360, 280))

                time.sleep(0.03)

        except KeyboardInterrupt:
            print("\n[INTERRUPTED]")
        finally:
            self.spatial_engine.stop()
            print("[CLEANUP] Returning to center...")
            self.move_servos(90, 90)
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("[DONE]")


def main():
    parser = argparse.ArgumentParser(description="LLM-directed zone-based gaze control")
    parser.add_argument("--no-hardware", action="store_true", help="Run without servo hardware")
    parser.add_argument("--no-video", action="store_true", help="Don't show video window")
    parser.add_argument("--interval", type=float, default=3.0, help="LLM query interval in seconds (default: 3.0)")
    args = parser.parse_args()

    global SHOW_VIDEO
    SHOW_VIDEO = not args.no_video

    controller = GazeController(use_hardware=not args.no_hardware, llm_interval=args.interval)
    controller.run()


if __name__ == "__main__":
    main()
