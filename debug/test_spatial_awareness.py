#!/usr/bin/env python3
"""
Comprehensive test of the spatial awareness system for person detection.

This script:
1. Visualizes zone coverage in real-time
2. Shows spatial memory state (where person was last seen)
3. Demonstrates LLM-driven gaze decisions
4. Tests departure logic with simulated scenarios

Run: python debug/test_spatial_awareness.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.person_detection_state import PersonDetectionState


def visualize_zones(state: PersonDetectionState, current_pan: float) -> str:
    """Create ASCII visualization of zone coverage."""
    # Pan range: typically 45-135 degrees (90 center)
    # Zones are 30 degrees each: 0-30, 30-60, 60-90, 90-120, 120-150, 150-180

    zones = []
    for zone_num in range(6):  # 0-5 covering 0-180 degrees
        zone_start = zone_num * 30
        zone_end = zone_start + 30
        zone_mid = zone_start + 15

        # Check if this zone has been visited
        visited = zone_num in state.scan_zones_visited

        # Check if current position is in this zone
        current_zone = int(current_pan // 30)
        is_current = (zone_num == current_zone)

        # Check if last seen position is in this zone
        last_seen_zone = None
        if state.last_seen_servo_pan is not None:
            last_seen_zone = int(state.last_seen_servo_pan // 30)
        is_last_seen = (zone_num == last_seen_zone)

        # Build zone display
        if is_current and is_last_seen:
            marker = "[◉]"  # Current AND last seen
        elif is_current:
            marker = "[●]"  # Current position
        elif is_last_seen:
            marker = "[◎]"  # Last seen position
        elif visited:
            marker = "[✓]"  # Visited
        else:
            marker = "[ ]"  # Not visited

        zones.append(marker)

    return " ".join(zones)


def get_llm_gaze_suggestion(state: PersonDetectionState, current_pan: float) -> dict:
    """
    Generate a gaze suggestion based on spatial awareness.

    This simulates what an LLM could decide given the current state.
    Returns a dict with suggested action and reasoning.
    """
    person_state = state.person_state
    direction = state.get_person_direction()
    zones_visited = len(state.scan_zones_visited)
    sweep_complete = state._has_completed_sweep()
    looking_at_last = state.is_looking_at_last_known_location()

    # Decision tree for gaze behavior
    if person_state == "visible":
        return {
            "action": "hold",
            "target_pan": None,
            "reasoning": "Person is visible - maintain current gaze",
            "urgency": "low"
        }

    elif person_state == "remembered":
        if direction == "to my left":
            return {
                "action": "look_left",
                "target_pan": current_pan - 30,
                "reasoning": f"Person was to my left. I should check there. ({zones_visited}/4 zones scanned)",
                "urgency": "medium"
            }
        elif direction == "to my right":
            return {
                "action": "look_right",
                "target_pan": current_pan + 30,
                "reasoning": f"Person was to my right. I should check there. ({zones_visited}/4 zones scanned)",
                "urgency": "medium"
            }
        elif not sweep_complete:
            # Need to scan more zones
            unvisited = [z for z in range(6) if z not in state.scan_zones_visited]
            if unvisited:
                target_zone = unvisited[0]
                target_pan = target_zone * 30 + 15  # Center of zone
                return {
                    "action": "scan",
                    "target_pan": target_pan,
                    "reasoning": f"Looking for them. Scanning zone {target_zone}. ({zones_visited}/4 zones)",
                    "urgency": "medium"
                }
        elif not looking_at_last and state.last_seen_servo_pan is not None:
            return {
                "action": "return_to_last",
                "target_pan": state.last_seen_servo_pan,
                "reasoning": "Sweep complete. Checking where I last saw them.",
                "urgency": "high"
            }
        else:
            return {
                "action": "wait",
                "target_pan": None,
                "reasoning": "Completed sweep and checked last location. They may have left.",
                "urgency": "low"
            }

    else:  # absent
        return {
            "action": "idle",
            "target_pan": None,
            "reasoning": "No one here. Free to wander.",
            "urgency": "none"
        }


def print_status(state: PersonDetectionState, pan: float, scenario: str = ""):
    """Print comprehensive status."""
    person_state = state.get_person_state()

    print("\n" + "="*60)
    if scenario:
        print(f"  SCENARIO: {scenario}")
        print("="*60)

    # Zone visualization
    print(f"\n  ZONES:  {visualize_zones(state, pan)}")
    print(f"          0°    30°   60°   90°   120°  150°")
    print(f"          LEFT ←─────── CENTER ───────→ RIGHT")

    # Current state
    print(f"\n  Current pan:     {pan:.0f}°")
    print(f"  Last seen at:    {state.last_seen_servo_pan or 'N/A'}°")
    print(f"  Person state:    {person_state['person_state'].upper()}")
    print(f"  is_present:      {person_state['is_present']}")
    print(f"  Direction:       {person_state.get('direction', 'N/A')}")
    print(f"  Zones visited:   {len(state.scan_zones_visited)}/4 ({sorted(state.scan_zones_visited)})")
    print(f"  Sweep complete:  {state._has_completed_sweep()}")
    print(f"  At last loc:     {state.is_looking_at_last_known_location()}")

    # LLM suggestion
    suggestion = get_llm_gaze_suggestion(state, pan)
    print(f"\n  LLM SUGGESTION:")
    print(f"    Action:    {suggestion['action']}")
    print(f"    Target:    {suggestion['target_pan']}°" if suggestion['target_pan'] else "    Target:    (stay)")
    print(f"    Reasoning: {suggestion['reasoning']}")
    print(f"    Urgency:   {suggestion['urgency']}")


def run_scenario(title: str, steps: list):
    """Run a test scenario with multiple steps."""
    print("\n" + "#"*60)
    print(f"# {title}")
    print("#"*60)

    state = PersonDetectionState()

    for i, step in enumerate(steps):
        action = step.get("action")
        pan = step.get("pan", 90)
        description = step.get("description", "")

        if action == "detect":
            # Simulate person detection
            state.update_servo_position(pan, 90)
            state.update_yolo_detection(True, 0.8)
            print_status(state, pan, f"Step {i+1}: {description}")

        elif action == "lose":
            # Person no longer detected - age the detection to simulate time passing
            # Since we don't clear detection anymore, we need to age it
            state.update_servo_position(pan, 90)
            state.update_yolo_detection(False)
            # Age the YOLO detection by 15 seconds to make it expire (threshold is 10s)
            if state.yolo_detection:
                state.yolo_detection.timestamp -= 15
            if state.face_detection:
                state.face_detection.timestamp -= 15
            # Re-run state update with aged detection
            with state._lock:
                state._update_person_state()
            print_status(state, pan, f"Step {i+1}: {description}")

        elif action == "move":
            # Just move camera (no detection update)
            state.update_servo_position(pan, 90)
            # Force state update by calling internal method
            with state._lock:
                state._update_person_state()
            print_status(state, pan, f"Step {i+1}: {description}")

        elif action == "wait":
            # Wait for time to pass (simulate timestamp aging)
            wait_time = step.get("seconds", 5)
            print(f"\n  ... waiting {wait_time} seconds ...")
            # Hack: directly modify last_detection_time to simulate time passing
            if state.last_detection_time:
                state.last_detection_time -= wait_time
            state.update_servo_position(pan, 90)
            with state._lock:
                state._update_person_state()
            print_status(state, pan, f"Step {i+1}: {description}")

        time.sleep(0.3)  # Brief pause for readability


def main():
    print("\n" + "="*60)
    print("  SPATIAL AWARENESS SYSTEM TEST")
    print("  Testing zone tracking, departure logic, and LLM suggestions")
    print("="*60)

    # Scenario 1: Basic detection and loss
    run_scenario("SCENARIO 1: Person detected, then camera looks away", [
        {"action": "detect", "pan": 90, "description": "Person detected at center"},
        {"action": "lose", "pan": 90, "description": "Person leaves frame (camera still at center)"},
        {"action": "move", "pan": 60, "description": "Camera pans left - zone 2 visited"},
        {"action": "move", "pan": 30, "description": "Camera pans more left - zone 1 visited"},
        {"action": "move", "pan": 120, "description": "Camera pans right - zone 4 visited"},
        {"action": "move", "pan": 150, "description": "Camera pans far right - zone 5 visited (sweep complete!)"},
        {"action": "move", "pan": 90, "description": "Camera returns to last known location"},
        {"action": "wait", "pan": 90, "seconds": 10, "description": "Wait 10 seconds at last location"},
    ])

    # Scenario 2: Person detected on left, camera sweeps
    run_scenario("SCENARIO 2: Person detected on LEFT, camera must search", [
        {"action": "detect", "pan": 45, "description": "Person detected on LEFT side"},
        {"action": "lose", "pan": 45, "description": "Person disappears"},
        {"action": "move", "pan": 90, "description": "Camera moves to center (away from last seen)"},
        {"action": "move", "pan": 120, "description": "Camera moves right"},
        {"action": "move", "pan": 150, "description": "Camera moves far right"},
    ])

    # Scenario 3: Person reappears during sweep
    run_scenario("SCENARIO 3: Person REAPPEARS during sweep", [
        {"action": "detect", "pan": 90, "description": "Person detected at center"},
        {"action": "lose", "pan": 90, "description": "Person disappears"},
        {"action": "move", "pan": 60, "description": "Camera pans left"},
        {"action": "move", "pan": 30, "description": "Camera pans more left"},
        {"action": "detect", "pan": 120, "description": "Person REAPPEARS on right!"},
        {"action": "lose", "pan": 120, "description": "Person disappears again"},
    ])

    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)
    print("\nKey insights:")
    print("  [●] = Current gaze position")
    print("  [◎] = Where person was last seen")
    print("  [✓] = Zone has been scanned")
    print("  [ ] = Zone not yet scanned")
    print("\nThe LLM can use this state to decide:")
    print("  - Which direction to look when searching")
    print("  - When to return to last known location")
    print("  - When enough area has been scanned to conclude person left")


if __name__ == "__main__":
    main()
