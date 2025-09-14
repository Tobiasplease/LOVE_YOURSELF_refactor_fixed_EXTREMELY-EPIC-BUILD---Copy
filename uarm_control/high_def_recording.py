#!/usr/bin/env python3
"""
High-Definition Motion Recording System for uArm Swift Pro

This implements high-frequency position sampling and accurate playback
to capture and reproduce exact movements with proper timing.
"""

import time
import json
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional
from uarm_controller import UarmController


class HighDefRecorder:
    def __init__(self, controller: UarmController, storage_path: str = "movement_recordings/uarm"):
        self.controller = controller
        self.storage_path = storage_path
        self.recording = False
        self.recording_thread = None
        self.motion_data = []
        self.start_time = 0

        # High-frequency recording settings
        self.target_frequency = 100  # 100Hz target
        self.min_frequency = 50     # 50Hz minimum acceptable
        self.sample_interval = 1.0 / self.target_frequency  # 10ms intervals

        # Movement detection settings
        self.position_threshold = 0.5  # mm - minimum movement to record
        self.last_recorded_pos = None

        os.makedirs(storage_path, exist_ok=True)

    def start_recording(self, motion_name: str) -> bool:
        """Start high-frequency motion recording"""
        if self.recording:
            print("❌ Already recording!")
            return False

        if not self.controller or not self.controller.is_connected():
            print("❌ Controller not connected")
            return False

        print(f"🎬 Starting HIGH-DEF recording: {motion_name}")
        print(f"📊 Target frequency: {self.target_frequency}Hz ({self.sample_interval*1000:.1f}ms intervals)")

        # Reset recording state
        self.motion_data = []
        self.start_time = time.time()
        self.recording = True
        self.last_recorded_pos = None

        # Clear button events
        self.controller.get_button_events()

        # Start high-frequency recording thread
        self.recording_thread = threading.Thread(
            target=self._high_frequency_recording_loop,
            args=(motion_name,),
            daemon=True
        )
        self.recording_thread.start()

        print("✅ High-frequency recording started")
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

        if actual_frequency < self.min_frequency:
            print(f"⚠️ WARNING: Low sampling rate may cause choppy playback")

        if not self.motion_data:
            print("❌ No motion data captured!")
            return False

        # Save motion data
        return self._save_motion_data(motion_name, total_time, actual_frequency)

    def _high_frequency_recording_loop(self, motion_name: str):
        """High-frequency recording loop running in separate thread"""
        print(f"🚀 Starting recording loop for: {motion_name}")

        suction_state = False
        sample_count = 0
        missed_samples = 0

        while self.recording:
            loop_start = time.time()

            try:
                # Get current position
                current_pos = self.controller.robot.get_position()
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

                # Record position if valid and movement detected
                if current_pos and len(current_pos) >= 3:
                    should_record = True

                    # Check if significant movement occurred (optimization)
                    if self.last_recorded_pos and self.position_threshold > 0:
                        dx = abs(current_pos[0] - self.last_recorded_pos[0])
                        dy = abs(current_pos[1] - self.last_recorded_pos[1])
                        dz = abs(current_pos[2] - self.last_recorded_pos[2])
                        movement = (dx**2 + dy**2 + dz**2)**0.5

                        # Only record if movement is significant OR suction changed OR it's been a while
                        if movement < self.position_threshold and not button_pressed and sample_count % 10 != 0:
                            should_record = False

                    if should_record:
                        self.motion_data.append({
                            "time": current_time,
                            "position": current_pos.copy() if hasattr(current_pos, 'copy') else list(current_pos),
                            "suction": suction_state,
                            "sample": sample_count,
                            "button_event": button_pressed
                        })
                        self.last_recorded_pos = current_pos

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
        """Save recorded motion data to file"""
        try:
            # Calculate movement statistics
            if len(self.motion_data) >= 2:
                start_pos = self.motion_data[0]["position"]
                end_pos = self.motion_data[-1]["position"]

                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                dz = end_pos[2] - start_pos[2]
                total_distance = (dx**2 + dy**2 + dz**2)**0.5
            else:
                total_distance = 0

            # Create motion data structure
            motion_data = {
                "name": motion_name,
                "type": "high_def_sequence",
                "description": f"High-definition recorded motion at {actual_frequency:.1f}Hz",
                "recorded_at": datetime.now().isoformat(),
                "total_duration": total_time,
                "actual_frequency": actual_frequency,
                "target_frequency": self.target_frequency,
                "sample_count": len(self.motion_data),
                "total_distance": total_distance,
                "sequence": self.motion_data,
                "metadata": {
                    "position_threshold": self.position_threshold,
                    "sample_interval": self.sample_interval,
                    "recording_quality": "high" if actual_frequency >= self.min_frequency else "medium"
                }
            }

            # Save to file
            filename = f"{motion_name}_hd.json"
            filepath = os.path.join(self.storage_path, filename)

            with open(filepath, 'w') as f:
                json.dump(motion_data, f, indent=2)

            print(f"💾 Motion saved: {filepath}")
            print(f"📊 Distance: {total_distance:.1f}mm, Quality: {motion_data['metadata']['recording_quality']}")

            return True

        except Exception as e:
            print(f"❌ Failed to save motion data: {e}")
            return False

    def play_motion(self, motion_name: str, speed_multiplier: float = 1.0) -> bool:
        """Play back recorded motion with high fidelity"""
        if not self.controller or not self.controller.is_connected():
            print("❌ Controller not connected")
            return False

        # Load motion data
        filename = f"{motion_name}_hd.json"
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

            print(f"▶️ Playing motion: {motion_name}")
            print(f"📊 Sequence: {len(sequence)} points, {motion_data.get('total_duration', 0):.1f}s")
            print(f"⚡ Speed: {speed_multiplier}x")

            return self._execute_high_fidelity_playback(sequence, speed_multiplier)

        except Exception as e:
            print(f"❌ Playback error: {e}")
            return False

    def _execute_high_fidelity_playback(self, sequence: List[Dict], speed_multiplier: float) -> bool:
        """Execute high-fidelity playback with proper timing and movement"""
        try:
            start_time = time.time()
            current_suction = False

            for i, waypoint in enumerate(sequence):
                target_time = waypoint["time"] / speed_multiplier
                position = waypoint["position"]
                suction = waypoint.get("suction", False)

                # Wait for correct timing
                elapsed = time.time() - start_time
                if target_time > elapsed:
                    time.sleep(target_time - elapsed)

                # Handle suction changes
                if suction != current_suction:
                    self.controller.set_pump(suction)
                    current_suction = suction

                # Move to position with calculated speed
                if len(position) >= 3:
                    # Calculate movement speed based on time to next waypoint
                    if i < len(sequence) - 1:
                        next_waypoint = sequence[i + 1]
                        next_time = next_waypoint["time"] / speed_multiplier
                        next_pos = next_waypoint["position"]

                        if len(next_pos) >= 3:
                            # Calculate distance and time for speed
                            dx = next_pos[0] - position[0]
                            dy = next_pos[1] - position[1]
                            dz = next_pos[2] - position[2]
                            distance = (dx**2 + dy**2 + dz**2)**0.5
                            time_diff = next_time - target_time

                            # Calculate speed (mm/s to speed parameter)
                            if time_diff > 0 and distance > 0:
                                velocity = distance / time_diff  # mm/s
                                # Convert to uArm speed parameter (rough approximation)
                                speed = min(500, max(10, int(velocity * 2)))
                            else:
                                speed = 100
                        else:
                            speed = 100
                    else:
                        speed = 100

                    # Move to position
                    try:
                        self.controller.robot.set_position(
                            x=position[0],
                            y=position[1],
                            z=position[2],
                            speed=speed,
                            wait=False  # Don't wait - let timing control handle it
                        )
                    except Exception as e:
                        print(f"⚠️ Movement error at waypoint {i}: {e}")

            print("✅ High-fidelity playback complete")
            return True

        except Exception as e:
            print(f"❌ Playback execution error: {e}")
            return False


def test_high_def_recording():
    """Test the high-definition recording system"""
    print("Testing High-Definition Recording System")
    print("=======================================")

    controller = UarmController(auto_home=False)
    if not controller.is_connected():
        print("❌ Failed to connect to uArm")
        return

    recorder = HighDefRecorder(controller)

    print("✅ Connected and ready")
    print("\n🎬 Starting 5-second test recording...")
    print("Move the arm to test high-frequency capture")

    recorder.start_recording("test_motion")
    time.sleep(5)  # Record for 5 seconds
    success = recorder.stop_recording("test_motion")

    if success:
        print("\n▶️ Testing playback...")
        recorder.play_motion("test_motion")

    controller.disconnect()


if __name__ == "__main__":
    test_high_def_recording()