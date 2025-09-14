import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

try:
    from .uarm_controller import UarmController
except ImportError:
    from uarm_controller import UarmController


class MotionManager:
    def __init__(self, storage_path: str = "movement_recordings/uarm", controller: Optional[UarmController] = None):
        self.storage_path = storage_path
        self.controller = controller
        self.motion_slots = {
            1: {"name": "pickup", "description": "Primary pickup motion"},
            2: {"name": "place", "description": "Primary placement motion"},
            3: {"name": "gesture", "description": "Gestural expression motion"}
        }

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
                print(f"[MotionManager] Error loading metadata: {e}")

        return {}

    def save_motion_metadata(self) -> bool:
        metadata_file = os.path.join(self.storage_path, "motion_metadata.json")

        try:
            metadata = {
                "motion_slots": self.motion_slots,
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True
        except Exception as e:
            print(f"[MotionManager] Error saving metadata: {e}")
            return False

    def record_motion(self, slot: int, name: Optional[str] = None, description: Optional[str] = None) -> bool:
        if slot not in self.motion_slots:
            print(f"[MotionManager] Invalid motion slot: {slot}")
            return False

        motion_name = name or self.motion_slots[slot]["name"]
        motion_description = description or self.motion_slots[slot]["description"]

        print(f"[MotionManager] Starting recording for motion {slot}: {motion_name}")

        if not self.controller or not self.controller.is_connected():
            print("[MotionManager] uArm controller not connected for recording")
            return False

        try:
            # Direct recording using the uArm API instead of subprocess
            print("="*50)
            print("RECORDING MODE INSTRUCTIONS:")
            print("1. Physically move the uArm to demonstrate the motion")
            print("2. The uArm motors will be disabled for manual movement")
            print("3. Press ENTER when you've finished the motion sequence")
            print("4. The motion will be saved and can be replayed")
            print("="*50)

            # Get the robot instance from controller
            robot = self.controller.robot
            if not robot:
                print("[MotionManager] Robot instance not available")
                return False

            # Start recording process
            print(f"\nRecording motion: {motion_name}")

            # Release motors so arm can be moved manually
            print("Releasing motors for manual movement...")
            if not self.controller.release_motors():
                print("Warning: Could not release motors - may be difficult to move")

            # Enhanced recording: capture full motion sequence with timing
            print("Press ENTER to START recording...")
            input()

            print("Recording motion sequence...")
            print("Move the uArm through your desired motion at the speed you want it replayed.")
            print("Press the PLAY button on the uArm base to toggle suction cup ON/OFF during motion.")
            print("Press ENTER when motion is COMPLETE...")

            # Record motion sequence with timing and suction states
            motion_sequence = []
            start_time = time.time()
            recording = True
            suction_state = False

            print("\nBUTTON CONTROLS DURING RECORDING:")
            print("- PLAY button: Toggle suction cup ON/OFF (firmware override active)")
            print("- MENU button: Reserved for future use")
            print("- Move the arm smoothly at the desired playback speed")
            print("- Button callbacks will override firmware play functions\n")

            # Clear any existing button events
            self.controller.get_button_events()

            # Start background thread to capture positions and button events
            def capture_positions():
                nonlocal recording, motion_sequence, suction_state
                while recording:
                    try:
                        current_pos = robot.get_position()
                        current_time = time.time() - start_time

                        # Check for button events (using callback system)
                        button_events = self.controller.get_button_events()

                        for event in button_events:
                            if event["pressed"]:  # Only respond to button presses, not releases
                                if event["button"] == "play":
                                    suction_state = not suction_state  # Toggle suction
                                    # Apply suction change immediately
                                    self.controller.set_pump(suction_state)
                                    print(f"[Recording] 🔘 PLAY button pressed - Suction {'ON' if suction_state else 'OFF'} at {current_time:.1f}s")

                                elif event["button"] == "menu":
                                    print(f"[Recording] 🔘 MENU button pressed at {current_time:.1f}s (reserved for future use)")

                        if current_pos:
                            motion_sequence.append({
                                "position": current_pos,
                                "time": current_time,
                                "suction": suction_state,
                                "button_events": button_events  # Store all events at this timestamp
                            })
                        time.sleep(0.1)  # Sample every 100ms (10Hz)
                    except Exception as e:
                        print(f"[Recording] Capture error: {e}")
                        pass

            # Start recording thread
            import threading
            record_thread = threading.Thread(target=capture_positions, daemon=True)
            record_thread.start()

            # Wait for user to finish motion
            input()
            recording = False
            total_time = time.time() - start_time

            print(f"Motion recorded! Duration: {total_time:.1f} seconds")
            print(f"Captured {len(motion_sequence)} waypoints")

            # Re-enable motors
            print("Re-enabling motors...")
            self.controller.enable_motors()

            if not motion_sequence:
                print("No motion data captured!")
                return False

            # Save motion data as full sequence with timing
            motion_data = {
                "name": motion_name,
                "description": motion_description,
                "sequence": motion_sequence,
                "total_duration": total_time,
                "recorded_at": datetime.now().isoformat(),
                "type": "full_sequence"
            }

            # Save to file
            motion_file = os.path.join(self.storage_path, f"{motion_name}.json")
            with open(motion_file, 'w') as f:
                json.dump(motion_data, f, indent=2)

            # Update metadata
            self.motion_slots[slot] = {
                "name": motion_name,
                "description": motion_description,
                "recorded_at": datetime.now().isoformat(),
                "file_path": motion_file
            }

            self.save_motion_metadata()
            print(f"[MotionManager] Successfully recorded motion: {motion_name}")
            return True

        except Exception as e:
            print(f"[MotionManager] Recording error: {e}")
            return False

    def play_motion(self, slot: int, relative: bool = False) -> bool:
        if slot not in self.motion_slots:
            print(f"[MotionManager] Invalid motion slot: {slot}")
            return False

        if not self.controller or not self.controller.is_connected():
            print("[MotionManager] No connected uArm controller available")
            return False

        motion_data = self.motion_slots[slot]
        motion_name = motion_data["name"]

        if "recorded_at" not in motion_data:
            print(f"[MotionManager] Motion slot {slot} ({motion_name}) has no recording")
            return False

        try:
            # Load the recorded motion data
            motion_file = motion_data.get("file_path")
            if not motion_file or not os.path.exists(motion_file):
                print(f"[MotionManager] Motion file not found: {motion_file}")
                return False

            with open(motion_file, 'r') as f:
                recorded_motion = json.load(f)

            print(f"[MotionManager] Playing motion {slot}: {motion_name}")

            # Execute the motion using the recorded sequence
            robot = self.controller.robot
            if not robot:
                print("[MotionManager] Robot instance not available")
                return False

            motion_type = recorded_motion.get("type", "simple_sequence")

            if motion_type == "full_sequence":
                # Play back the full recorded sequence with timing
                sequence = recorded_motion["sequence"]
                total_duration = recorded_motion["total_duration"]

                print(f"[MotionManager] Playing {len(sequence)} waypoints over {total_duration:.1f}s")

                start_time = time.time()
                last_time = 0
                current_suction_state = False

                for waypoint in sequence:
                    target_time = waypoint["time"]
                    position = waypoint["position"]
                    suction = waypoint.get("suction", False)

                    # Wait for the correct time
                    elapsed = time.time() - start_time
                    if target_time > elapsed:
                        time.sleep(target_time - elapsed)

                    # Handle suction changes
                    if suction != current_suction_state:
                        self.controller.set_pump(suction)
                        current_suction_state = suction
                        print(f"[Playback] Suction {'ON' if suction else 'OFF'} at {target_time:.1f}s")

                    # Move to position
                    if len(position) >= 3:
                        try:
                            # Calculate speed based on time between waypoints
                            time_diff = target_time - last_time
                            speed = min(250, max(50, int(100 / max(0.1, time_diff))))  # Adaptive speed

                            robot.set_position(x=position[0], y=position[1], z=position[2],
                                             speed=speed, wait=False)
                            last_time = target_time
                        except Exception as e:
                            print(f"[WARNING] Waypoint failed: {e}")
                            continue

                # Ensure suction is off at the end
                if current_suction_state:
                    self.controller.set_pump(False)
                    print("[Playback] Suction OFF - motion complete")

                print(f"[MotionManager] Full sequence motion {motion_name} completed")

            else:
                # Fallback: simple start/end position playback for older recordings
                print("[MotionManager] Playing simple start/end motion (legacy format)")

                # Get current position if using relative movement
                if relative:
                    current_pos = robot.get_position()
                    print(f"[MotionManager] Starting from current position: {current_pos}")

                # Move to start position
                start_pos = recorded_motion.get("start_position", recorded_motion["sequence"][0]["position"])
                if relative and current_pos and len(current_pos) >= 3:
                    # Calculate relative offset - this is complex, skip for now
                    print("[MotionManager] Relative mode not supported for legacy recordings")

                if isinstance(start_pos, list) and len(start_pos) >= 3:
                    robot.set_position(x=start_pos[0], y=start_pos[1], z=start_pos[2], speed=100, wait=True)

                # Move to end position
                end_pos = recorded_motion.get("end_position", recorded_motion["sequence"][-1]["position"])
                if isinstance(end_pos, list) and len(end_pos) >= 3:
                    robot.set_position(x=end_pos[0], y=end_pos[1], z=end_pos[2], speed=100, wait=True)

            print(f"[MotionManager] Motion {motion_name} completed successfully")
            return True

        except Exception as e:
            print(f"[MotionManager] Motion playback failed: {e}")
            return False

    def get_motion_info(self, slot: int) -> Optional[Dict]:
        if slot not in self.motion_slots:
            return None
        return self.motion_slots[slot].copy()

    def list_all_motions(self) -> Dict[int, Dict]:
        return {slot: self.get_motion_info(slot) for slot in self.motion_slots.keys()}

    def is_motion_recorded(self, slot: int) -> bool:
        if slot not in self.motion_slots:
            return False
        return "recorded_at" in self.motion_slots[slot]

    def delete_motion(self, slot: int) -> bool:
        if slot not in self.motion_slots:
            return False

        motion_data = self.motion_slots[slot]
        motion_name = motion_data["name"]

        try:
            # Remove the recorded file if it exists
            file_path = motion_data.get("file_path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

            # Also try removing the motion file by name convention
            motion_file = os.path.join(self.storage_path, f"{motion_name}.json")
            if os.path.exists(motion_file):
                os.remove(motion_file)

            # Reset motion slot to default state
            default_names = {1: "pickup", 2: "place", 3: "gesture"}
            default_descriptions = {
                1: "Primary pickup motion",
                2: "Primary placement motion",
                3: "Gestural expression motion"
            }

            self.motion_slots[slot] = {
                "name": default_names[slot],
                "description": default_descriptions[slot]
            }

            self.save_motion_metadata()
            print(f"[MotionManager] Deleted motion from slot {slot}")
            return True

        except Exception as e:
            print(f"[MotionManager] Error deleting motion: {e}")
            return False

    def update_motion_info(self, slot: int, name: Optional[str] = None, description: Optional[str] = None) -> bool:
        if slot not in self.motion_slots:
            return False

        if name:
            self.motion_slots[slot]["name"] = name
        if description:
            self.motion_slots[slot]["description"] = description

        return self.save_motion_metadata()

    def get_status(self) -> Dict:
        return {
            "storage_path": self.storage_path,
            "total_slots": len(self.motion_slots),
            "recorded_motions": sum(1 for slot in self.motion_slots.keys() if self.is_motion_recorded(slot)),
            "controller_connected": self.controller.is_connected() if self.controller else False,
            "motions": self.list_all_motions()
        }