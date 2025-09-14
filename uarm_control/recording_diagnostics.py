#!/usr/bin/env python3
"""
Diagnose recording quality and position accuracy issues
"""

import time
import json
import os
from uarm_controller import UarmController


def test_position_accuracy_and_speed():
    """Test how accurately and quickly we can read positions"""
    print("Testing Position Reading Accuracy and Speed")
    print("===========================================")

    controller = UarmController(auto_home=False)

    if not controller.is_connected():
        print("ERROR: Failed to connect to uArm")
        return

    print("✅ Connected without auto-homing")

    # Test 1: Position reading frequency
    print("\n=== Test 1: Position Reading Frequency ===")
    positions = []
    start_time = time.time()
    test_duration = 5.0  # 5 seconds

    print(f"Reading positions for {test_duration} seconds as fast as possible...")

    while time.time() - start_time < test_duration:
        try:
            pos = controller.robot.get_position()
            timestamp = time.time() - start_time
            if pos:
                positions.append({"time": timestamp, "pos": pos})
        except Exception as e:
            print(f"Position read error: {e}")

    total_readings = len(positions)
    actual_hz = total_readings / test_duration
    print(f"📊 Results: {total_readings} readings in {test_duration}s = {actual_hz:.1f} Hz")

    if actual_hz < 10:
        print("⚠️ WARNING: Very low sampling rate - recordings will be choppy")
    elif actual_hz < 50:
        print("⚠️ CAUTION: Moderate sampling rate - may miss fine movements")
    else:
        print("✅ Good sampling rate for high-quality recording")

    # Test 2: Position precision
    print("\n=== Test 2: Position Precision Analysis ===")
    if len(positions) >= 10:
        recent_positions = positions[-10:]  # Last 10 readings
        x_values = [p["pos"][0] for p in recent_positions if len(p["pos"]) >= 3]
        y_values = [p["pos"][1] for p in recent_positions if len(p["pos"]) >= 3]
        z_values = [p["pos"][2] for p in recent_positions if len(p["pos"]) >= 3]

        if x_values:
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)
            z_range = max(z_values) - min(z_values)

            print(f"Position stability (stationary arm):")
            print(f"  X-axis variation: {x_range:.3f}mm")
            print(f"  Y-axis variation: {y_range:.3f}mm")
            print(f"  Z-axis variation: {z_range:.3f}mm")

            total_variation = x_range + y_range + z_range
            if total_variation < 0.5:
                print("✅ Excellent position precision")
            elif total_variation < 2.0:
                print("✅ Good position precision")
            else:
                print("⚠️ Position readings may be noisy")

    # Test 3: Movement detection sensitivity
    print("\n=== Test 3: Movement Detection Test ===")
    print("Please move the arm slightly and I'll measure the sensitivity...")

    baseline_pos = controller.robot.get_position()
    if baseline_pos and len(baseline_pos) >= 3:
        print(f"Baseline position: X={baseline_pos[0]:.2f}, Y={baseline_pos[1]:.2f}, Z={baseline_pos[2]:.2f}")
        print("Move the arm gently and watch for detection...")

        movement_detected = False
        for i in range(50):  # 5 seconds at 10Hz
            try:
                current_pos = controller.robot.get_position()
                if current_pos and len(current_pos) >= 3:
                    dx = abs(current_pos[0] - baseline_pos[0])
                    dy = abs(current_pos[1] - baseline_pos[1])
                    dz = abs(current_pos[2] - baseline_pos[2])
                    total_movement = dx + dy + dz

                    if total_movement > 1.0:  # 1mm threshold
                        print(f"📍 Movement detected: ΔX={dx:.2f}, ΔY={dy:.2f}, ΔZ={dz:.2f} (total={total_movement:.2f}mm)")
                        movement_detected = True

                time.sleep(0.1)
            except Exception as e:
                print(f"Movement detection error: {e}")

        if not movement_detected:
            print("No significant movement detected")

    controller.disconnect()
    return actual_hz, positions


def analyze_existing_recording():
    """Analyze an existing recording file to see what's wrong"""
    print("\n=== Analyzing Existing Recording Files ===")

    # Check for recorded motions
    storage_path = "movement_recordings/uarm"
    if not os.path.exists(storage_path):
        print(f"No recordings found in {storage_path}")
        return

    recording_files = [f for f in os.listdir(storage_path) if f.endswith('.json')]

    if not recording_files:
        print("No recording files found")
        return

    print(f"Found {len(recording_files)} recording files:")
    for file in recording_files:
        print(f"  - {file}")

    # Analyze the first recording
    latest_file = recording_files[0]
    file_path = os.path.join(storage_path, latest_file)

    try:
        with open(file_path, 'r') as f:
            recording_data = json.load(f)

        print(f"\n📊 Analysis of {latest_file}:")
        print(f"  Name: {recording_data.get('name', 'Unknown')}")
        print(f"  Type: {recording_data.get('type', 'Unknown')}")
        print(f"  Duration: {recording_data.get('total_duration', 0):.1f}s")

        sequence = recording_data.get('sequence', [])
        print(f"  Sample count: {len(sequence)}")

        if sequence:
            sampling_rate = len(sequence) / recording_data.get('total_duration', 1)
            print(f"  Sampling rate: {sampling_rate:.1f} Hz")

            # Analyze position changes
            positions = [s['position'] for s in sequence if 'position' in s and s['position']]
            if len(positions) >= 2:
                start_pos = positions[0]
                end_pos = positions[-1]

                if len(start_pos) >= 3 and len(end_pos) >= 3:
                    dx = abs(end_pos[0] - start_pos[0])
                    dy = abs(end_pos[1] - start_pos[1])
                    dz = abs(end_pos[2] - start_pos[2])
                    total_distance = (dx**2 + dy**2 + dz**2)**0.5

                    print(f"  Movement distance:")
                    print(f"    ΔX: {dx:.1f}mm")
                    print(f"    ΔY: {dy:.1f}mm")
                    print(f"    ΔZ: {dz:.1f}mm")
                    print(f"    Total: {total_distance:.1f}mm")

                    if total_distance < 10:
                        print("    ⚠️ Very small movement recorded - may not be visible during playback")
                    elif total_distance < 50:
                        print("    ⚠️ Small movement - check if this matches your intended motion")
                    else:
                        print("    ✅ Significant movement recorded")

            # Check for timing issues
            timestamps = [s['time'] for s in sequence if 'time' in s]
            if len(timestamps) >= 2:
                time_intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = sum(time_intervals) / len(time_intervals)
                print(f"  Average time between samples: {avg_interval:.3f}s ({1/avg_interval:.1f} Hz)")

                # Check for timing consistency
                max_gap = max(time_intervals)
                min_gap = min(time_intervals)
                if max_gap > min_gap * 3:
                    print(f"    ⚠️ Inconsistent timing: {min_gap:.3f}s to {max_gap:.3f}s")
                else:
                    print(f"    ✅ Consistent timing")

    except Exception as e:
        print(f"Error analyzing recording: {e}")


def main():
    print("uArm Recording Quality Diagnostics")
    print("==================================\n")

    # Test current system capabilities
    hz, positions = test_position_accuracy_and_speed()

    # Analyze existing recordings
    analyze_existing_recording()

    # Provide recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR HIGH-QUALITY RECORDING")
    print("="*60)

    if hz < 20:
        print("❌ CRITICAL: Sampling rate too low")
        print("   - Need at least 50Hz for smooth playback")
        print("   - Current rate may miss rapid movements")

    print("\n📋 Recommended Fixes:")
    print("1. Increase sampling frequency to 50-100Hz")
    print("2. Use more aggressive position polling")
    print("3. Implement interpolation for smooth playback")
    print("4. Add movement velocity calculation")
    print("5. Use encoder-based position tracking if available")

    print("\n🎯 Target Performance:")
    print("- Sampling: 50-100Hz minimum")
    print("- Position accuracy: <1mm")
    print("- Timing precision: <10ms jitter")
    print("- Movement fidelity: Capture all motions >2mm")


if __name__ == "__main__":
    main()