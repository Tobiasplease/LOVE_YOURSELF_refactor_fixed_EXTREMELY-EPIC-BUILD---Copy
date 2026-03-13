#!/usr/bin/env python3
"""
Live test of LLM-directed gaze with zone system.

The LLM decides where to look (left/right/up/down/ahead/person)
and the gaze system moves organically within that zone.

Run: python debug/test_spatial_awareness_live.py --hardware
"""

import sys
import os
import time
import cv2
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CAMERA_INDEX, PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX
from vision.spatial_awareness import get_spatial_awareness_engine
from vision import gaze
from perception.person_detection_state import get_person_detection_state

# Optional YOLO
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False
    print("[!] YOLO not available - running without person detection")

# Hardware servo controller
servo_controller = None


def init_hardware():
    """Initialize servo hardware."""
    global servo_controller
    try:
        from servo_control.arduino_servos import ServoController
        servo_controller = ServoController()
        print("[✓] Servos connected")
        return True
    except Exception as e:
        print(f"[✗] Servo init failed: {e}")
        return False


def move_servos(pan: int, tilt: int):
    """Move physical servos."""
    if servo_controller:
        try:
            servo_controller.set_pan(int(pan))
            servo_controller.set_tilt(int(tilt))
        except Exception as e:
            print(f"[SERVO ERROR] {e}")


def detect_person(frame):
    """Simple YOLO person detection."""
    if not YOLO_AVAILABLE:
        return False, 0.0, None

    results = yolo_model(frame, classes=[0], verbose=False)
    for r in results:
        for box in r.boxes:
            if box.cls[0] == 0 and box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                return True, float(box.conf[0]), (x1, y1, x2, y2)
    return False, 0.0, None


def draw_hud(frame, spatial_engine, person_state, pan, tilt):
    """Draw heads-up display with zone and awareness info."""
    h, w = frame.shape[:2]

    # Background panel
    cv2.rectangle(frame, (5, 5), (400, 200), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (400, 200), (0, 255, 0), 1)

    # Title
    cv2.putText(frame, "LLM-DIRECTED GAZE", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Current zone
    zone_text = gaze.get_current_zone_text()
    cv2.putText(frame, f"Looking: {zone_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # LLM zone (what LLM wants)
    llm_zone = f"{gaze.llm_target_zone_pan}/{gaze.llm_target_zone_tilt}"
    active = "ACTIVE" if gaze.llm_zone_active else "inactive"
    cv2.putText(frame, f"LLM zone: {llm_zone} ({active})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Gaze position
    cv2.putText(frame, f"Servos: pan={pan:.0f} tilt={tilt:.0f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"State: {gaze.state}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Person state
    ps = person_state.get_person_state()
    person_color = (0, 255, 0) if ps["is_present"] else (100, 100, 100)
    cv2.putText(frame, f"Person: {ps['person_state']}", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 1)

    # Tracking awareness
    if gaze.state == "tracking":
        tracking_ctx = gaze.get_tracking_context()
        cv2.putText(frame, f"Tracking: person {tracking_ctx}", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    # LLM response
    status = spatial_engine.get_status()
    if status["last_response"]:
        response = status["last_response"][:50]
        cv2.putText(frame, f"LLM: {response}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)

    # Draw zone indicator (simple bar at top)
    zone_width = w // 3
    zones = [("left", 0), ("ahead", 1), ("right", 2)]
    for zone_name, idx in zones:
        x_start = idx * zone_width
        color = (0, 100, 0)
        if gaze.llm_target_zone_pan == zone_name and gaze.llm_zone_active:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x_start, h-30), (x_start + zone_width, h-5), color, -1)
        cv2.putText(frame, zone_name.upper(), (x_start + 10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def print_status(spatial_engine, person_state, pan, tilt):
    """Print compact status line."""
    ps = person_state.get_person_state()
    status = spatial_engine.get_status()

    zone_text = gaze.get_current_zone_text()
    llm_dir = status["last_direction"]

    print(f"\r[{gaze.state:10s}] Looking: {zone_text:15s} LLM→{llm_dir:8s} "
          f"pan={pan:3.0f}° tilt={tilt:3.0f}° person={ps['person_state']:10s}   ",
          end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true", help="Enable servo hardware")
    parser.add_argument("--interval", type=float, default=3.0, help="LLM query interval (seconds)")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LLM-DIRECTED GAZE TEST")
    print("  The LLM decides where to look, organic movement follows")
    print("=" * 60)

    if args.hardware:
        if not init_hardware():
            print("[!] Running without hardware")
    else:
        print("[!] Hardware disabled - use --hardware to enable servos")

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize systems
    person_state = get_person_detection_state()
    spatial_engine = get_spatial_awareness_engine()
    spatial_engine.query_interval = args.interval
    spatial_engine.debug = True

    # Start spatial awareness
    spatial_engine.start()

    # Reset gaze
    gaze.servo_x = 90
    gaze.servo_y = 90
    gaze.state = "idle"
    gaze.llm_zone_active = False
    move_servos(90, 90)

    print(f"\nLLM queries every {args.interval}s. Press 'q' to quit, 'r' to reset\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Feed frame to spatial awareness
            spatial_engine.update_frame(frame)

            # Detect person
            person_detected, confidence, person_box = detect_person(frame)
            person_state.update_yolo_detection(person_detected, confidence)

            # Get smoothed person state
            ps = person_state.get_person_state()
            smoothed_person = ps.get("is_present", False)

            # Update spatial awareness with person info
            if smoothed_person and person_box:
                # Calculate person position in frame
                x1, y1, x2, y2 = person_box
                frame_width = frame.shape[1]
                center_x = (x1 + x2) / 2
                frame_center = frame_width / 2

                if center_x < frame_center - 100:
                    position = "left"
                elif center_x > frame_center + 100:
                    position = "right"
                else:
                    position = "center"

                spatial_engine.update_person_state(True, position, gaze.tracking_person_movement)
            else:
                spatial_engine.update_person_state(False)

            # Get face box for tracking (use person box center as face approximation)
            face_box = None
            if person_box and smoothed_person:
                x1, y1, x2, y2 = person_box
                # Use upper portion as "face"
                face_h = (y2 - y1) // 3
                face_box = (x1, y1, x2, y1 + face_h)

            # Update gaze
            _, pan, tilt = gaze.update_gaze(
                frame,
                face_box=face_box,
                current_emotion_state="calm_observant",
                yolo_person_detected=smoothed_person,
                person_direction=ps.get("direction")
            )

            # Feed gaze position back
            person_state.update_servo_position(pan, tilt)

            # Move physical servos
            move_servos(pan, tilt)

            # Print status
            print_status(spatial_engine, person_state, pan, tilt)

            # Display
            if not args.no_display:
                display_frame = frame.copy()

                # Draw person box if detected
                if person_box:
                    x1, y1, x2, y2 = person_box
                    color = (0, 255, 0) if smoothed_person else (0, 100, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Draw HUD
                display_frame = draw_hud(display_frame, spatial_engine, person_state, pan, tilt)

                cv2.imshow("LLM Gaze Test", display_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\n[RESET] Returning to center, disabling LLM zone")
                gaze.servo_x = 90
                gaze.servo_y = 90
                gaze.state = "idle"
                gaze.llm_zone_active = False
                gaze.llm_target_zone_pan = "ahead"
                gaze.llm_target_zone_tilt = "level"
                move_servos(90, 90)
            elif key == ord('l'):
                # Manual test: set zone to left
                print("\n[MANUAL] Zone → LEFT")
                gaze.set_llm_zone("left")
            elif key == ord('a'):
                # Manual test: set zone to ahead
                print("\n[MANUAL] Zone → AHEAD")
                gaze.set_llm_zone("ahead")

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
    finally:
        spatial_engine.stop()
        cap.release()
        cv2.destroyAllWindows()
        move_servos(90, 90)
        print("\n[EXIT] Done")


if __name__ == "__main__":
    main()
