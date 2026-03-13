#!/usr/bin/env python3
"""
Comprehensive test of the intelligent gaze system with searching behavior.

This script tests:
1. Searching state activation when person is "remembered"
2. Zone scanning with goal-directed movement
3. LLM interest point tracking
4. Integration with PersonDetectionState
5. State machine transitions: tracking → grace → aware → searching → idle

Run with hardware: python debug/test_intelligent_gaze.py --hardware
Run simulation only: python debug/test_intelligent_gaze.py
"""

import sys
import os
import time
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we're testing
from perception.person_detection_state import PersonDetectionState
from vision import gaze
from config.config import PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX

# Hardware controller (optional)
servo_controller = None


def init_hardware():
    """Initialize real servo hardware."""
    global servo_controller
    try:
        from servo_control.arduino_servos import ServoController
        servo_controller = ServoController()
        print("[✓] Hardware initialized - servos connected")
        return True
    except Exception as e:
        print(f"[✗] Hardware init failed: {e}")
        return False


def move_servos(pan: int, tilt: int):
    """Move physical servos (if connected) or just print."""
    if servo_controller:
        try:
            servo_controller.set_pan(pan)
            servo_controller.set_tilt(tilt)
        except Exception as e:
            print(f"[SERVO ERROR] {e}")
    else:
        pass  # Simulation mode - no physical movement


class MockFrame:
    """Mock frame for testing."""
    def __init__(self, width=640, height=480):
        self.shape = (height, width, 3)


def visualize_gaze_state(person_state: PersonDetectionState, pan: float, tilt: float) -> str:
    """Create ASCII visualization of gaze zones and current position."""
    # Pan range divided into 6 zones (0-5)
    zones = []
    for zone_num in range(6):
        zone_start = zone_num * 30
        zone_end = zone_start + 30

        # Check if this zone has been visited (from person detection state)
        visited = zone_num in person_state.scan_zones_visited

        # Check if this zone is a search target
        is_search_target = zone_num in gaze.searching_zones_to_visit if gaze.searching_zones_to_visit else False

        # Current position zone
        current_zone = int(pan // 30) if pan else 3
        is_current = (zone_num == current_zone)

        # Last seen zone
        last_seen_zone = None
        if person_state.last_seen_servo_pan is not None:
            last_seen_zone = int(person_state.last_seen_servo_pan // 30)
        is_last_seen = (zone_num == last_seen_zone)

        # Build display
        if is_current and is_last_seen:
            marker = "[◉]"  # Current AND last seen
        elif is_current:
            marker = "[●]"  # Current position
        elif is_last_seen:
            marker = "[◎]"  # Last seen
        elif is_search_target:
            marker = "[→]"  # Search target
        elif visited:
            marker = "[✓]"  # Visited
        else:
            marker = "[ ]"  # Not visited

        zones.append(marker)

    return " ".join(zones)


def print_compact_status(pan: float, tilt: float, gaze_state: str, search_info: str = ""):
    """Print compact status line for real-time feedback."""
    bar_width = 30
    pan_norm = (pan - PAN_MIN) / (PAN_MAX - PAN_MIN) if PAN_MAX > PAN_MIN else 0.5
    pos = int(pan_norm * bar_width)
    bar = "─" * pos + "●" + "─" * (bar_width - pos - 1)
    print(f"\r  [{bar}] pan={pan:3.0f}° tilt={tilt:3.0f}° state={gaze_state:12s} {search_info}", end="", flush=True)


def print_full_status(person_state: PersonDetectionState, pan: float, tilt: float, scenario: str = ""):
    """Print comprehensive status of both person detection and gaze systems."""
    print("\n" + "=" * 70)
    if scenario:
        print(f"  {scenario}")
        print("=" * 70)

    # Zone visualization
    print(f"\n  ZONES:  {visualize_gaze_state(person_state, pan, tilt)}")
    print(f"          0°    30°   60°   90°   120°  150°")
    print(f"          LEFT ←─────── CENTER ───────→ RIGHT")

    # Person detection state
    ps = person_state.get_person_state()
    print(f"\n  PERSON DETECTION:")
    print(f"    State:        {ps['person_state'].upper()}")
    print(f"    is_present:   {ps['is_present']}")
    print(f"    Direction:    {ps.get('direction', 'N/A')}")
    print(f"    Last seen:    pan={person_state.last_seen_servo_pan}°")
    print(f"    Zones visited: {len(person_state.scan_zones_visited)}/4 ({sorted(person_state.scan_zones_visited)})")

    # Gaze state
    print(f"\n  GAZE STATE:")
    print(f"    State:        {gaze.state.upper()}")
    print(f"    Position:     pan={pan:.0f}°, tilt={tilt:.0f}°")
    print(f"    Searching:    {gaze.searching_active}")

    if gaze.searching_active:
        target_pan, target_tilt, goal_type = gaze.get_search_target()
        print(f"    Search goal:  {goal_type}")
        print(f"    Search target: pan={target_pan}°")
        print(f"    Zones to scan: {gaze.searching_zones_to_visit}")

    if gaze.searching_interest_points:
        print(f"    Interest pts: {len(gaze.searching_interest_points)}")
        for i, (p, t, pri, exp) in enumerate(gaze.searching_interest_points):
            print(f"      {i+1}. pan={p:.0f}°, tilt={t:.0f}°, priority={pri:.1f}")


def simulate_gaze_update_realtime(person_state: PersonDetectionState,
                                   face_detected: bool = False,
                                   yolo_detected: bool = False,
                                   duration: float = 3.0,
                                   update_interval: float = 0.05) -> tuple:
    """
    Run gaze updates in real-time, moving physical servos.
    Shows live progress bar as camera moves.
    """
    frame = MockFrame()
    face_box = (280, 200, 360, 280) if face_detected else None

    pan, tilt = gaze.servo_x, gaze.servo_y
    start_time = time.time()

    print()  # New line for progress bar
    while time.time() - start_time < duration:
        person_direction = person_state.get_person_direction()
        _, pan, tilt = gaze.update_gaze(
            frame,
            face_box,
            current_emotion_state="calm_observant",
            yolo_person_detected=yolo_detected,
            person_direction=person_direction
        )

        # Feed position back to person detection for zone tracking
        person_state.update_servo_position(pan, tilt)

        # Move physical servos
        move_servos(int(pan), int(tilt))

        # Update search progress if searching
        if gaze.searching_active:
            gaze.update_search_progress(pan, tilt, person_found=face_detected)

        # Show progress
        search_info = ""
        if gaze.searching_active:
            target_pan, _, goal_type = gaze.get_search_target()
            if target_pan:
                search_info = f"→ {goal_type}@{target_pan:.0f}°"
        print_compact_status(pan, tilt, gaze.state, search_info)

        time.sleep(update_interval)

    print()  # End progress line
    return pan, tilt


def run_interactive_demo(use_hardware: bool):
    """Run an interactive demo with real hardware (if available)."""
    print("\n" + "=" * 70)
    print("  INTELLIGENT GAZE DEMO - INTERACTIVE MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  d - Detect person (simulate face detection)")
    print("  l - Lose person (trigger search mode)")
    print("  i - Add interest point at random location")
    print("  r - Reset to center")
    print("  s - Show full status")
    print("  q - Quit")
    print("=" * 70)

    if use_hardware and not init_hardware():
        print("[!] Running in simulation mode (no hardware)")

    # Reset state
    person_state = PersonDetectionState()
    gaze.servo_x = 90
    gaze.servo_y = 90
    gaze.state = "idle"
    gaze.searching_active = False
    gaze.searching_zones_to_visit = []
    gaze.searching_interest_points = []

    pan, tilt = 90, 90
    move_servos(90, 90)

    import random

    while True:
        print_full_status(person_state, pan, tilt, "Current State")

        try:
            cmd = input("\nCommand> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if cmd == 'q':
            break

        elif cmd == 'd':
            print("\n[ACTION] Detecting person...")
            person_state.update_yolo_detection(True, 0.9)

            # Deactivate search if active
            if gaze.is_search_mode_active():
                gaze.deactivate_search_mode()

            pan, tilt = simulate_gaze_update_realtime(
                person_state, face_detected=True, yolo_detected=True, duration=2.0
            )

        elif cmd == 'l':
            print("\n[ACTION] Losing person - triggering search mode...")

            # Age the detection
            person_state.update_yolo_detection(False)
            if person_state.yolo_detection:
                person_state.yolo_detection.timestamp -= 15
            if person_state.face_detection:
                person_state.face_detection.timestamp -= 15
            with person_state._lock:
                person_state._update_person_state()

            # Activate search mode
            ps = person_state.get_person_state()
            if ps["person_state"] == "remembered" and not gaze.is_search_mode_active():
                gaze.activate_search_mode(
                    last_seen_pan=person_state.last_seen_servo_pan,
                    last_seen_tilt=person_state.last_seen_servo_tilt,
                    zones_visited=person_state.scan_zones_visited
                )

            # Run search for a while
            print("\n[SEARCHING] Camera searching for person...")
            pan, tilt = simulate_gaze_update_realtime(
                person_state, face_detected=False, yolo_detected=False, duration=10.0
            )

        elif cmd == 'i':
            # Add random interest point
            interest_pan = random.choice([30, 60, 90, 120, 150])
            interest_tilt = random.randint(int(TILT_MIN + 10), int(TILT_MAX - 10))
            priority = random.uniform(0.6, 1.0)

            print(f"\n[ACTION] Adding interest point at pan={interest_pan}°, tilt={interest_tilt}°, priority={priority:.2f}")
            gaze.add_interest_point(interest_pan, interest_tilt, priority)

            # Activate search mode if not already
            if not gaze.is_search_mode_active():
                gaze.activate_search_mode(
                    last_seen_pan=None,
                    last_seen_tilt=None,
                    zones_visited=set()
                )

            # Move toward interest
            print("\n[MOVING] Camera moving toward interest point...")
            pan, tilt = simulate_gaze_update_realtime(
                person_state, face_detected=False, yolo_detected=False, duration=8.0
            )

        elif cmd == 'r':
            print("\n[ACTION] Resetting to center...")
            gaze.servo_x = 90
            gaze.servo_y = 90
            gaze.state = "idle"
            gaze.deactivate_search_mode()
            pan, tilt = 90, 90
            move_servos(90, 90)

        elif cmd == 's':
            pass  # Status already shown at top of loop

        else:
            print("[?] Unknown command")

    # Cleanup
    print("\n[EXIT] Returning to center...")
    move_servos(90, 90)
    print("Done.")


def run_automated_demo(use_hardware: bool):
    """Run automated demo sequence."""
    print("\n" + "=" * 70)
    print("  INTELLIGENT GAZE DEMO - AUTOMATED SEQUENCE")
    print("=" * 70)

    if use_hardware and not init_hardware():
        print("[!] Running in simulation mode (no hardware)")

    # Reset state
    person_state = PersonDetectionState()
    gaze.servo_x = 90
    gaze.servo_y = 90
    gaze.state = "idle"
    gaze.searching_active = False
    gaze.searching_zones_to_visit = []
    gaze.searching_interest_points = []

    move_servos(90, 90)
    time.sleep(1)

    print("\n--- PHASE 1: Person Detection ---")
    print("Simulating person appearing in front of camera...")
    person_state.update_yolo_detection(True, 0.9)
    pan, tilt = simulate_gaze_update_realtime(
        person_state, face_detected=True, yolo_detected=True, duration=3.0
    )
    print_full_status(person_state, pan, tilt, "Person detected")

    print("\n--- PHASE 2: Person Lost → Search Mode ---")
    print("Person disappears... camera should start searching...")
    time.sleep(1)

    # Age detection to trigger remembered state
    person_state.update_yolo_detection(False)
    if person_state.yolo_detection:
        person_state.yolo_detection.timestamp -= 15
    with person_state._lock:
        person_state._update_person_state()

    # Activate search
    ps = person_state.get_person_state()
    if ps["person_state"] == "remembered":
        gaze.activate_search_mode(
            last_seen_pan=person_state.last_seen_servo_pan,
            last_seen_tilt=person_state.last_seen_servo_tilt,
            zones_visited=person_state.scan_zones_visited
        )

    print("\nSearching... watch the camera scan zones!")
    pan, tilt = simulate_gaze_update_realtime(
        person_state, face_detected=False, yolo_detected=False, duration=15.0
    )
    print_full_status(person_state, pan, tilt, "After search")

    print("\n--- PHASE 3: Interest Point ---")
    print("LLM finds something interesting on the right...")
    gaze.add_interest_point(150, 90, priority=0.9)

    if not gaze.is_search_mode_active():
        gaze.activate_search_mode(None, None, set())

    print("\nMoving to interest point...")
    pan, tilt = simulate_gaze_update_realtime(
        person_state, face_detected=False, yolo_detected=False, duration=8.0
    )
    print_full_status(person_state, pan, tilt, "At interest point")

    print("\n--- PHASE 4: Person Returns ---")
    print("Person comes back!")
    time.sleep(1)
    person_state.update_yolo_detection(True, 0.9)
    gaze.deactivate_search_mode()

    pan, tilt = simulate_gaze_update_realtime(
        person_state, face_detected=True, yolo_detected=True, duration=3.0
    )
    print_full_status(person_state, pan, tilt, "Person returned!")

    print("\n--- DEMO COMPLETE ---")
    print("Returning to center...")
    move_servos(90, 90)


def main():
    parser = argparse.ArgumentParser(description="Test intelligent gaze system")
    parser.add_argument("--hardware", action="store_true", help="Enable real servo hardware")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive mode")
    parser.add_argument("--auto", "-a", action="store_true", help="Run automated demo (default)")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_demo(args.hardware)
    else:
        run_automated_demo(args.hardware)


if __name__ == "__main__":
    main()
