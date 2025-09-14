#!/usr/bin/env python3
"""
High-Frequency Motion Manager - Using Proven Servo Angle Method
==============================================================

Uses get_servo_angle() for live encoder readings during manual movement.
Provides 50Hz recording frequency with proven accuracy.
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

try:
    from .uarm_controller import UarmController
except ImportError:
    from uarm_controller import UarmController


class HighFrequencyMotionManager:
    def __init__(self, storage_path: str = "movement_recordings/uarm", controller: Optional[UarmController] = None):
        self.storage_path = storage_path
        self.controller = controller
        self.motion_slots = {
            1: {"name": "pickup", "description": "Primary pickup motion"},
            2: {"name": "place", "description": "Primary placement motion"},
            3: {"name": "gesture", "description": "Gestural expression motion"}
        }

        # Recording state
        self.recording_in_progress = False
        self.recording_start_time = 0
        self.current_recording = []
        self.record_thread = None

        self._ensure_storage_path()
        self.load_motion_metadata()

    def _ensure_storage_path(self):
        os.makedirs(self.storage_path, exist_ok=True)

    def load_motion_metadata(self) -> Dict:
        metadata_file = os.path.join(self.storage_path, "motion_metadata.json")

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.motion_slots.update(metadata.get("motion_slots", {}))
                    return metadata
            except Exception as e:
                print(f"[HighFreqMotionManager] Error loading metadata: {e}")

        return {}

    def save_motion_metadata(self) -> bool:
        metadata_file = os.path.join(self.storage_path, "motion_metadata.json")

        try:
            metadata = {
                "motion_slots": self.motion_slots,
                "last_updated": datetime.now().isoformat(),
                "version": "2.0"  # Mark as high-frequency version
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"[HighFreqMotionManager] Error saving metadata: {e}")
            return False

    def start_recording(self, slot: int, name: Optional[str] = None, description: Optional[str] = None) -> bool:
        """Start high-frequency angle-based recording"""
        if self.recording_in_progress:
            print("[HighFreqMotionManager] Recording already in progress")
            return False

        if slot not in self.motion_slots:
            print(f"[HighFreqMotionManager] Invalid motion slot: {slot}")
            return False

        if not self.controller or not self.controller.is_connected():
            print("[HighFreqMotionManager] uArm controller not connected")
            return False

        motion_name = name or self.motion_slots[slot]["name"]
        motion_description = description or self.motion_slots[slot]["description"]

        print(f"[HighFreqMotionManager] Starting high-frequency recording for: {motion_name}")

        try:
            # Release motors for manual movement
            if not self.controller.release_motors():
                print("[HighFreqMotionManager] Warning: Could not release motors")

            # Clear button events and prepare recording state
            self.controller.get_button_events()
            self.current_recording = []
            self.recording_start_time = time.time()
            self.recording_in_progress = True

            # Store recording context
            self.current_slot = slot
            self.current_name = motion_name
            self.current_description = motion_description

            # Start background recording thread using proven servo angle method
            self.record_thread = threading.Thread(target=self._record_angles_background, daemon=True)
            self.record_thread.start()

            print(f"[HighFreqMotionManager] 🎬 Recording started - move the arm manually!")
            print(f"[HighFreqMotionManager] 📊 Recording at 50Hz using servo angles")
            print(f"[HighFreqMotionManager] 🔘 Use physical buttons to control suction during recording")

            return True

        except Exception as e:
            print(f"[HighFreqMotionManager] Error starting recording: {e}")
            self.recording_in_progress = False
            return False

    def _record_angles_background(self):
        """Background thread for high-frequency servo angle recording"""
        print(f"[HighFreqMotionManager] Background recording thread started")
        sample_count = 0
        last_progress_time = time.time()

        while self.recording_in_progress:
            try:
                # Use get_servo_angle() - proven to work with released motors
                current_angles = self.controller.robot.get_servo_angle()
                button_events = self.controller.get_button_events()

                if current_angles and len(current_angles) >= 3:
                    timestamp = time.time()
                    relative_time = timestamp - self.recording_start_time

                    # Handle button events for suction control
                    suction_changed = False
                    for event in button_events:
                        if event["pressed"] and event["button"] == "play":
                            self.controller.toggle_pump()
                            suction_changed = True
                            print(f"[Recording] 🔘 Suction toggled at {relative_time:.1f}s")

                    motion_point = {
                        "time": relative_time,
                        "angles": list(current_angles),  # Live encoder angles
                        "suction": self.controller.suction_enabled,
                        "sample": sample_count,
                        "button_event": len(button_events) > 0
                    }

                    self.current_recording.append(motion_point)
                    sample_count += 1

                    # Show progress every 2 seconds
                    if timestamp - last_progress_time >= 2.0:
                        actual_frequency = sample_count / relative_time if relative_time > 0 else 0
                        print(f"[HighFreqMotionManager] 📊 Recorded {sample_count} samples at {actual_frequency:.1f}Hz ({relative_time:.1f}s)")
                        last_progress_time = timestamp

                time.sleep(0.02)  # 50Hz target frequency

            except Exception as e:
                print(f"[Recording] Error: {e}")
                break

        print(f"[HighFreqMotionManager] Recording thread completed - {sample_count} samples captured")

    def stop_recording(self) -> bool:
        """Stop recording and save the motion data"""
        if not self.recording_in_progress:
            print("[HighFreqMotionManager] No recording in progress")
            return False

        print(f"[HighFreqMotionManager] Stopping recording...")

        # Stop the recording thread
        self.recording_in_progress = False
        if self.record_thread:
            self.record_thread.join(timeout=2.0)

        # Re-enable motors
        try:
            self.controller.enable_motors()
        except Exception as e:
            print(f"[HighFreqMotionManager] Warning: Could not re-enable motors: {e}")

        # Calculate recording statistics
        total_duration = time.time() - self.recording_start_time
        sample_count = len(self.current_recording)
        actual_frequency = sample_count / total_duration if total_duration > 0 else 0

        print(f"[HighFreqMotionManager] 📊 Recording complete:")
        print(f"  ✅ Duration: {total_duration:.2f}s")
        print(f"  ✅ Samples: {sample_count}")
        print(f"  ✅ Frequency: {actual_frequency:.1f}Hz")

        if sample_count == 0:
            print("[HighFreqMotionManager] ❌ No data captured!")
            return False

        try:
            # Calculate total angle change for quality assessment
            if sample_count >= 2:
                first_angles = self.current_recording[0]["angles"]
                last_angles = self.current_recording[-1]["angles"]
                total_angle_change = sum(abs(a - b) for a, b in zip(first_angles, last_angles))
            else:
                total_angle_change = 0.0

            # Save high-frequency motion data
            motion_data = {
                "name": self.current_name,
                "type": "angle_based_sequence",
                "description": f"Angle-based recorded motion at {actual_frequency:.1f}Hz",
                "recorded_at": datetime.now().isoformat(),
                "total_duration": total_duration,
                "actual_frequency": actual_frequency,
                "target_frequency": 50,
                "sample_count": sample_count,
                "total_angle_change": total_angle_change,
                "sequence": self.current_recording,
                "metadata": {
                    "recording_method": "servo_angles",
                    "sample_interval": 0.02,
                    "recording_quality": "high" if actual_frequency >= 25 else "medium" if actual_frequency >= 10 else "low"
                }
            }

            # Save to file
            motion_file = os.path.join(self.storage_path, f"{self.current_name}_angles.json")
            with open(motion_file, 'w') as f:
                json.dump(motion_data, f, indent=2)

            # Update metadata
            self.motion_slots[self.current_slot] = {
                "name": self.current_name,
                "description": self.current_description,
                "recorded_at": datetime.now().isoformat(),
                "file_path": motion_file
            }

            self.save_motion_metadata()

            print(f"[HighFreqMotionManager] ✅ Motion saved: {motion_file}")
            print(f"[HighFreqMotionManager] 📈 Recording quality: {motion_data['metadata']['recording_quality']}")

            return True

        except Exception as e:
            print(f"[HighFreqMotionManager] ❌ Error saving recording: {e}")
            return False

    def play_motion(self, slot: int, speed_multiplier: float = 1.0) -> bool:
        """Play back angle-based recorded motion"""
        if slot not in self.motion_slots:
            print(f"[HighFreqMotionManager] Invalid motion slot: {slot}")
            return False

        if not self.controller or not self.controller.is_connected():
            print("[HighFreqMotionManager] No connected uArm controller")
            return False

        motion_data = self.motion_slots[slot]
        if "recorded_at" not in motion_data:
            print(f"[HighFreqMotionManager] Motion slot {slot} has no recording")
            return False

        try:
            # Load motion file
            motion_file = motion_data.get("file_path")
            if not motion_file or not os.path.exists(motion_file):
                print(f"[HighFreqMotionManager] Motion file not found: {motion_file}")
                return False

            with open(motion_file, 'r') as f:
                recorded_motion = json.load(f)

            if recorded_motion.get("type") != "angle_based_sequence":
                print("[HighFreqMotionManager] Motion is not angle-based - cannot play")
                return False

            sequence = recorded_motion["sequence"]
            motion_name = recorded_motion["name"]

            print(f"[HighFreqMotionManager] 🎬 Playing {motion_name}: {len(sequence)} angle waypoints")

            robot = self.controller.robot
            start_time = time.time()
            current_suction_state = False

            # Direct servo angle playback - NO conversion, NO skipping waypoints
            for i, waypoint in enumerate(sequence):
                target_time = waypoint["time"] / speed_multiplier
                angles = waypoint["angles"]
                suction = waypoint.get("suction", False)

                # Wait for precise timing
                elapsed = time.time() - start_time
                if target_time > elapsed:
                    time.sleep(target_time - elapsed)

                # Handle suction changes
                if suction != current_suction_state:
                    self.controller.set_pump(suction)
                    current_suction_state = suction

                # Direct servo angle control - play ALL waypoints
                if len(angles) >= 3:
                    try:
                        # Set all three servos directly with recorded angles
                        robot.set_servo_angle(servo_id=0, angle=angles[0], wait=False)  # Base
                        robot.set_servo_angle(servo_id=1, angle=angles[1], wait=False)  # Left
                        robot.set_servo_angle(servo_id=2, angle=angles[2], wait=False)  # Right

                        # Show progress every 50 waypoints
                        if i % 50 == 0:
                            progress = (i / len(sequence)) * 100
                            print(f"[Playback] {progress:.1f}% - T:{target_time:.2f}s - Angles: [{angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°]")

                    except Exception as e:
                        if i % 50 == 0:
                            print(f"[Playback] Waypoint {i} failed: {e}")
                        continue

            # Ensure suction is off at the end
            if current_suction_state:
                self.controller.set_pump(False)

            total_playback_time = time.time() - start_time
            print(f"[HighFreqMotionManager] ✅ Motion {motion_name} completed in {total_playback_time:.1f}s")

            return True

        except Exception as e:
            print(f"[HighFreqMotionManager] ❌ Playback failed: {e}")
            return False

    def is_recording(self) -> bool:
        """Check if recording is currently in progress"""
        return self.recording_in_progress

    def get_recording_status(self) -> Dict:
        """Get current recording status and statistics"""
        if not self.recording_in_progress:
            return {"recording": False}

        elapsed = time.time() - self.recording_start_time
        sample_count = len(self.current_recording)
        frequency = sample_count / elapsed if elapsed > 0 else 0

        return {
            "recording": True,
            "elapsed_time": elapsed,
            "sample_count": sample_count,
            "current_frequency": frequency,
            "target_frequency": 50
        }

    def get_motion_info(self, slot: int) -> Optional[Dict]:
        """Get motion information including recording quality"""
        if slot not in self.motion_slots:
            return None

        motion_data = self.motion_slots[slot].copy()

        # Add recording quality info if available
        if "file_path" in motion_data:
            try:
                with open(motion_data["file_path"], 'r') as f:
                    recorded_motion = json.load(f)
                    if "metadata" in recorded_motion:
                        motion_data.update(recorded_motion["metadata"])
                        motion_data["actual_frequency"] = recorded_motion.get("actual_frequency", 0)
                        motion_data["sample_count"] = recorded_motion.get("sample_count", 0)
            except:
                pass

        return motion_data

    def list_all_motions(self) -> Dict[int, Dict]:
        return {slot: self.get_motion_info(slot) for slot in self.motion_slots.keys()}

    def is_motion_recorded(self, slot: int) -> bool:
        if slot not in self.motion_slots:
            return False
        return "recorded_at" in self.motion_slots[slot]