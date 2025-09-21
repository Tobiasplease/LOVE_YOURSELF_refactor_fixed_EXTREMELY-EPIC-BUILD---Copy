#!/usr/bin/env python3
"""
Test person-responsive CNC drawing functionality - pause and resume based on face detection.
"""
import os
import sys
import time
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.person_detection_state import get_person_detection_state
from grbl.segmented_executor import SegmentedExecutor

class MockSerial:
    """Mock serial interface for testing without actual GRBL connection"""
    def __init__(self):
        self.sent_commands = []
        self.is_open = True

    def write(self, data):
        command = data.decode('utf-8').strip()
        self.sent_commands.append(command)
        print(f"[GRBL] -> {command}")

    def readline(self):
        # Simulate GRBL OK response
        return b"ok\r\n"

    def close(self):
        self.is_open = False

def create_test_gcode_file():
    """Create a test G-code file for demonstration"""
    test_file = "/tmp/test_drawing.gcode"

    gcode_content = """
; Test Drawing - Simple Square
G21 ; mm units
G90 ; absolute positioning
G0 Z5 ; lift pen
G0 X0 Y0 ; go to start

; Draw square
G1 Z-1 F500 ; pen down
G1 X10 Y0 F300  ; line 1
G1 X10 Y10 F300 ; line 2
G1 X0 Y10 F300  ; line 3
G1 X0 Y0 F300   ; line 4
G0 Z5 ; pen up

; Draw another square
G0 X15 Y0 ; move to next position
G1 Z-1 F500 ; pen down
G1 X25 Y0 F300  ; line 1
G1 X25 Y10 F300 ; line 2
G1 X15 Y10 F300 ; line 3
G1 X15 Y0 F300  ; line 4
G0 Z5 ; pen up

; Final position
G0 X0 Y0 Z10
M30 ; end program
"""

    with open(test_file, 'w') as f:
        f.write(gcode_content.strip())

    return test_file

def simulate_person_detection_sequence():
    """Simulate a person detection sequence in a separate thread"""
    person_detection = get_person_detection_state()

    # Wait a bit before person arrives
    time.sleep(2)

    print("\n👤 [SIMULATION] Person approaches...")
    person_detection.update_face_detection(0.85, (100, 100, 200, 200))

    # Person stays for a while
    time.sleep(4)

    print("👤 [SIMULATION] Person moves away...")
    person_detection.update_face_detection(0.0)

    # Brief gap
    time.sleep(1)

    print("👤 [SIMULATION] Person returns briefly...")
    person_detection.update_face_detection(0.75, (90, 90, 190, 190))

    time.sleep(2)

    print("👤 [SIMULATION] Person leaves for good...")
    person_detection.update_face_detection(0.0)

def test_person_responsive_cnc():
    print("🎨 Testing Person-Responsive CNC Drawing")
    print("=" * 60)

    # Create test G-code file
    test_file = create_test_gcode_file()
    print(f"📄 Created test G-code: {test_file}")

    # Create mock serial connection
    mock_ser = MockSerial()
    print("🔌 Mock GRBL connection established")

    # Create segmented executor with person detection enabled
    executor = SegmentedExecutor(max_segment_size=5, enable_person_detection_pause=True)
    print("🤖 Segmented executor ready with person detection")

    # Start person detection simulation in background
    detection_thread = threading.Thread(target=simulate_person_detection_sequence, daemon=True)
    detection_thread.start()

    print("\n🎬 Starting execution simulation...")
    print("Expected behavior:")
    print("  - Drawing starts normally")
    print("  - Pauses when person detected")
    print("  - Resumes 3 seconds after person leaves")
    print("  - Continues until completion")

    start_time = time.time()

    try:
        # Execute the G-code with person detection pausing
        success = executor.execute_gcode_file_segmented(mock_ser, test_file, move_timeout=5.0)

        total_time = time.time() - start_time
        progress_info = executor.get_progress_info()

        print(f"\n🎯 EXECUTION COMPLETE")
        print(f"   Success: {success}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Effective drawing time: {progress_info.get('effective_elapsed_time', 0):.1f}s")
        print(f"   Total pause time: {progress_info.get('total_pause_time', 0):.1f}s")
        print(f"   Commands sent: {len(mock_ser.sent_commands)}")

        # Show some sample commands
        print(f"\n📋 Sample commands sent:")
        for i, cmd in enumerate(mock_ser.sent_commands[:5]):
            print(f"   {i+1}: {cmd}")
        if len(mock_ser.sent_commands) > 5:
            print(f"   ... and {len(mock_ser.sent_commands) - 5} more")

    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted by user")

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

    print(f"\n✅ Person-responsive CNC test complete!")

def test_person_detection_states():
    """Test different person detection scenarios"""
    print("\n🧪 Testing Person Detection State Scenarios")
    print("-" * 40)

    person_detection = get_person_detection_state()

    scenarios = [
        {"name": "No person", "confidence": 0.0, "expected_pause": False},
        {"name": "Low confidence", "confidence": 0.3, "expected_pause": False},
        {"name": "Medium confidence", "confidence": 0.6, "expected_pause": True},
        {"name": "High confidence", "confidence": 0.9, "expected_pause": True},
    ]

    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")

        # Update detection
        if scenario['confidence'] > 0:
            person_detection.update_face_detection(scenario['confidence'], (100, 100, 200, 200))
        else:
            person_detection.update_face_detection(0.0)

        # Check state
        state = person_detection.get_person_state()
        print(f"   Person present: {state['is_present']}")
        print(f"   Confidence: {state['confidence']:.2f}")
        print(f"   Expected to pause: {scenario['expected_pause']}")
        print(f"   Would actually pause: {state['is_present']}")

        # Test breathing response
        from breathing.breathing import update_lung_position
        breath_mod, pause_mod = person_detection.get_breathing_modifiers("alert_curious")
        print(f"   Breathing modifiers: speed={breath_mod:.2f}, pause={pause_mod:.2f}")

if __name__ == "__main__":
    # Test person detection states first
    test_person_detection_states()

    # Then test the full CNC integration
    test_person_responsive_cnc()