#!/usr/bin/env python3
"""
Angle-Based High-Definition Recorder

Records servo angles directly (which work when motors are detached)
instead of relying on get_position() which returns stale data.
"""

import time
import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional
from uarm_controller import UarmController


class AngleBasedRecorder:
    def __init__(self, controller: UarmController, storage_path: str = "movement_recordings/uarm"):
        self.controller = controller
        self.storage_path = storage_path
        self.recording = False
        self.recording_thread = None
        self.motion_data = []
        self.start_time = 0

        # High-frequency recording settings
        self.target_frequency = 100  # 100Hz target
        self.sample_interval = 1.0 / self.target_frequency  # 10ms intervals

        os.makedirs(storage_path, exist_ok=True)

    def start_recording(self, motion_name: str) -> bool:
        """Start high-frequency angle-based recording"""
        if self.recording:
            print("❌ Already recording!")
            return False

        if not self.controller or not self.controller.is_connected():
            print("❌ Controller not connected")
            return False

        print(f"🎬 Starting ANGLE-BASED recording: {motion_name}")
        print(f"📊 Target frequency: {self.target_frequency}Hz")

        # Reset recording state
        self.motion_data = []
        self.start_time = time.time()
        self.recording = True

        # Clear button events
        self.controller.get_button_events()

        # Start high-frequency recording thread
        self.recording_thread = threading.Thread(
            target=self._angle_recording_loop,
            args=(motion_name,),
            daemon=True
        )
        self.recording_thread.start()

        print("✅ Angle-based recording started")
        print("🔘 Press PLAY button to toggle suction during motion")
        return True

    def stop_recording(self, motion_name: str) -> bool:
        """Stop recording and save motion data"""
        if not self.recording:
            print("❌ Not currently recording")
            return False

        self.recording = False

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)

        total_time = time.time() - self.start_time
        sample_count = len(self.motion_data)
        actual_frequency = sample_count / total_time if total_time > 0 else 0

        print(f"⏹️ Recording stopped")
        print(f"📊 Captured {sample_count} samples in {total_time:.1f}s")
        print(f"📊 Actual frequency: {actual_frequency:.1f}Hz")

        if not self.motion_data:
            print("❌ No motion data captured!")
            return False

        # Save motion data
        return self._save_motion_data(motion_name, total_time, actual_frequency)

    def _angle_recording_loop(self, motion_name: str):
        """High-frequency angle recording loop"""
        print(f"🚀 Starting angle recording loop for: {motion_name}")

        suction_state = False
        sample_count = 0
        missed_samples = 0
        last_angles = None

        while self.recording:
            loop_start = time.time()

            try:
                # Get current servo angles (works when motors detached!)
                current_angles = self.controller.robot.get_servo_angle()
                current_time = time.time() - self.start_time

                # Check for button events (suction control)
                button_events = self.controller.get_button_events()
                button_pressed = False

                for event in button_events:
                    if event["pressed"] and event["button"] == "play":
                        suction_state = not suction_state
                        self.controller.set_pump(suction_state)
                        button_pressed = True
                        print(f"🔘 Suction {'ON' if suction_state else 'OFF'} at {current_time:.2f}s")

                # Record angles if valid
                if current_angles and len(current_angles) >= 3:
                    should_record = True

                    # Only record if angles changed significantly OR suction changed OR periodic sample
                    if last_angles is not None:
                        angle_change = sum(abs(current_angles[i] - last_angles[i]) for i in range(3))
                        if angle_change < 0.5 and not button_pressed and sample_count % 5 != 0:
                            should_record = False

                    if should_record:
                        self.motion_data.append({
                            "time": current_time,
                            "angles": [float(a) for a in current_angles],  # [servo0, servo1, servo2]
                            "suction": suction_state,
                            "sample": sample_count,
                            "button_event": button_pressed
                        })
                        last_angles = current_angles.copy() if hasattr(current_angles, 'copy') else list(current_angles)

                sample_count += 1

            except Exception as e:
                print(f"⚠️ Recording sample error: {e}")
                missed_samples += 1

            # Precise timing control
            loop_time = time.time() - loop_start
            sleep_time = max(0, self.sample_interval - loop_time)

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                missed_samples += 1

        # Final statistics
        total_samples = sample_count
        success_rate = ((total_samples - missed_samples) / total_samples * 100) if total_samples > 0 else 0

        print(f"📊 Recording statistics:")
        print(f"   Total samples attempted: {total_samples}")
        print(f"   Missed samples: {missed_samples}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Data points saved: {len(self.motion_data)}")

    def _save_motion_data(self, motion_name: str, total_time: float, actual_frequency: float) -> bool:
        """Save recorded angle-based motion data to file"""
        try:
            # Calculate movement statistics
            if len(self.motion_data) >= 2:
                start_angles = self.motion_data[0]["angles"]
                end_angles = self.motion_data[-1]["angles"]

                total_angle_change = sum(abs(end_angles[i] - start_angles[i]) for i in range(3))
            else:
                total_angle_change = 0

            # Create motion data structure
            motion_data = {
                "name": motion_name,
                "type": "angle_based_sequence",
                "description": f"Angle-based recorded motion at {actual_frequency:.1f}Hz",
                "recorded_at": datetime.now().isoformat(),
                "total_duration": total_time,
                "actual_frequency": actual_frequency,
                "target_frequency": self.target_frequency,
                "sample_count": len(self.motion_data),
                "total_angle_change": total_angle_change,
                "sequence": self.motion_data,
                "metadata": {
                    "recording_method": "servo_angles",
                    "sample_interval": self.sample_interval,
                    "recording_quality": "high" if actual_frequency >= 50 else "medium"
                }
            }

            # Save to file
            filename = f"{motion_name}_angles.json"
            filepath = os.path.join(self.storage_path, filename)

            with open(filepath, 'w') as f:
                json.dump(motion_data, f, indent=2)

            print(f"💾 Motion saved: {filepath}")
            print(f"📊 Angle change: {total_angle_change:.1f}°, Quality: {motion_data['metadata']['recording_quality']}")

            return True

        except Exception as e:
            print(f"❌ Failed to save motion data: {e}")
            return False

    def play_motion(self, motion_name: str, speed_multiplier: float = 1.0) -> bool:
        """Play back angle-based recorded motion"""
        if not self.controller or not self.controller.is_connected():
            print("❌ Controller not connected")
            return False

        # Load motion data
        filename = f"{motion_name}_angles.json"
        filepath = os.path.join(self.storage_path, filename)

        if not os.path.exists(filepath):
            print(f"❌ Motion file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                motion_data = json.load(f)

            sequence = motion_data.get("sequence", [])
            if not sequence:
                print("❌ No sequence data found")
                return False

            print(f"▶️ Playing angle-based motion: {motion_name}")
            print(f"📊 Sequence: {len(sequence)} points, {motion_data.get('total_duration', 0):.1f}s")
            print(f"⚡ Speed: {speed_multiplier}x")

            return self._execute_angle_playback(sequence, speed_multiplier)

        except Exception as e:
            print(f"❌ Playback error: {e}")
            return False

    def _execute_angle_playback(self, sequence: List[Dict], speed_multiplier: float) -> bool:
        """Execute angle-based playback"""
        try:
            start_time = time.time()
            current_suction = False

            print("🎬 Starting angle-based playback...")

            for i, waypoint in enumerate(sequence):
                target_time = waypoint["time"] / speed_multiplier
                angles = waypoint["angles"]
                suction = waypoint.get("suction", False)

                # Wait for correct timing
                elapsed = time.time() - start_time
                if target_time > elapsed:
                    time.sleep(target_time - elapsed)

                # Handle suction changes
                if suction != current_suction:
                    self.controller.set_pump(suction)
                    current_suction = suction

                # Move to angles
                if len(angles) >= 3:
                    try:
                        # Set each servo angle individually for precise control
                        for servo_id, angle in enumerate(angles[:3]):
                            self.controller.robot.set_servo_angle(servo=servo_id, angle=float(angle), wait=False)

                        # Optional: Show progress
                        if i % 20 == 0:
                            print(f"  Waypoint {i}/{len(sequence)}: [{angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°]")

                    except Exception as e:
                        print(f"⚠️ Movement error at waypoint {i}: {e}")

            print("✅ Angle-based playback complete")
            return True

        except Exception as e:
            print(f"❌ Playback execution error: {e}")
            return False


def test_angle_recording():
    """Test the angle-based recording system"""
    print("Testing Angle-Based Recording System")
    print("===================================")

    controller = UarmController(auto_home=False)
    if not controller.is_connected():
        print("❌ Failed to connect to uArm")
        return

    recorder = AngleBasedRecorder(controller)

    print("✅ Connected and ready")
    print("\n🎬 Starting 5-second test recording...")
    print("Release motors and move the arm to test angle capture")

    # Release motors first
    controller.release_motors()
    time.sleep(1)

    recorder.start_recording("test_angles")
    time.sleep(5)  # Record for 5 seconds
    success = recorder.stop_recording("test_angles")

    if success:
        print("\n▶️ Testing playback...")

        # Re-enable motors for playback
        controller.enable_motors()
        time.sleep(1)

        recorder.play_motion("test_angles")

    controller.disconnect()


if __name__ == "__main__":
    test_angle_recording()