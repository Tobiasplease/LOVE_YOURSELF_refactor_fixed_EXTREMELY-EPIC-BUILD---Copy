#!/usr/bin/env python3

import os
import sys
import time
from typing import Optional

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uarm_control.uarm_controller import UarmController
from uarm_control.motion_manager import MotionManager

try:
    from config.config import (
        UARM_MOTION_STORAGE,
        UARM_MOVEMENT_NAMES,
        UARM_PORT,
        UARM_DEFAULT_SPEED
    )
except ImportError:
    UARM_MOTION_STORAGE = "movement_recordings/uarm"
    UARM_MOVEMENT_NAMES = {1: "pickup", 2: "place", 3: "gesture"}
    UARM_PORT = None
    UARM_DEFAULT_SPEED = 100


class UarmRecordingUI:
    def __init__(self):
        print("=== uArm Swift Pro Recording UI ===")
        print("Teach and Play Functionality for 3 Movements")
        print()

        self.controller = None
        self.motion_manager = None
        self.initialize_system()

    def initialize_system(self):
        try:
            print("Initializing uArm controller...")
            self.controller = UarmController(port=UARM_PORT, connect_on_init=False)

            print("Setting up motion manager...")
            self.motion_manager = MotionManager(
                storage_path=UARM_MOTION_STORAGE,
                controller=self.controller
            )

            print("Attempting to connect to uArm...")
            if self.controller.connect():
                print("✅ uArm connected successfully!")
            else:
                print("❌ uArm connection failed. Some features may be limited.")
                print(f"   Error: {self.controller.last_error}")

        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            sys.exit(1)

    def show_main_menu(self):
        while True:
            print("\n" + "="*50)
            print("uArm Swift Pro Recording UI - Main Menu")
            print("="*50)

            # Show current status
            status = self.motion_manager.get_status()
            print(f"Controller: {'Connected' if status['controller_connected'] else 'Disconnected'}")
            print(f"Recorded Motions: {status['recorded_motions']}/{status['total_slots']}")
            print()

            # Show motion status
            motions = status['motions']
            for slot_num in [1, 2, 3]:
                motion = motions[slot_num]
                recorded = "✅" if self.motion_manager.is_motion_recorded(slot_num) else "❌"
                print(f"  {slot_num}. {motion['name']}: {motion['description']} {recorded}")

            print()
            print("Options:")
            print("  1-3: Record motion for slot 1-3")
            print("  p1-p3: Play motion from slot 1-3")
            print("  t1-t3: Test motion from slot 1-3")
            print("  d1-d3: Delete motion from slot 1-3")
            print("  s: Show detailed status")
            print("  c: Check connection")
            print("  h: Home uArm to safe position")
            print("  q: Quit")

            choice = input("\nEnter your choice: ").lower().strip()

            if choice == 'q':
                break
            elif choice in ['1', '2', '3']:
                self.record_motion(int(choice))
            elif choice.startswith('p') and len(choice) == 2 and choice[1] in ['1', '2', '3']:
                self.play_motion(int(choice[1]))
            elif choice.startswith('t') and len(choice) == 2 and choice[1] in ['1', '2', '3']:
                self.test_motion(int(choice[1]))
            elif choice.startswith('d') and len(choice) == 2 and choice[1] in ['1', '2', '3']:
                self.delete_motion(int(choice[1]))
            elif choice == 's':
                self.show_detailed_status()
            elif choice == 'c':
                self.check_connection()
            elif choice == 'h':
                self.home_uarm()
            else:
                print("Invalid choice. Please try again.")

    def record_motion(self, slot: int):
        print(f"\n=== Recording Motion {slot} ===")

        motion_info = self.motion_manager.get_motion_info(slot)
        if not motion_info:
            print("Invalid motion slot!")
            return

        print(f"Recording motion: {motion_info['name']}")
        print(f"Description: {motion_info['description']}")

        # Check if motion already exists
        if self.motion_manager.is_motion_recorded(slot):
            confirm = input("\nThis slot already has a recording. Overwrite? (y/N): ")
            if confirm.lower() != 'y':
                print("Recording cancelled.")
                return

        # Optional: Update name/description
        new_name = input(f"\nMotion name [{motion_info['name']}]: ").strip()
        if new_name:
            motion_info['name'] = new_name

        new_desc = input(f"Description [{motion_info['description']}]: ").strip()
        if new_desc:
            motion_info['description'] = new_desc

        print(f"\nPreparing to record '{motion_info['name']}'...")
        print("You will now physically move the uArm to teach it the motion.")
        print("The recording will start in 3 seconds...")

        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        success = self.motion_manager.record_motion(
            slot=slot,
            name=motion_info['name'],
            description=motion_info['description']
        )

        if success:
            print(f"✅ Motion {slot} recorded successfully!")
        else:
            print(f"❌ Recording failed for motion {slot}")

    def play_motion(self, slot: int):
        print(f"\n=== Playing Motion {slot} ===")

        if not self.controller.is_connected():
            print("❌ uArm not connected! Cannot play motion.")
            return

        if not self.motion_manager.is_motion_recorded(slot):
            print(f"❌ No motion recorded in slot {slot}")
            return

        motion_info = self.motion_manager.get_motion_info(slot)
        print(f"Playing: {motion_info['name']}")

        confirm = input("Execute motion? (y/N): ")
        if confirm.lower() != 'y':
            print("Playback cancelled.")
            return

        success = self.motion_manager.play_motion(slot, relative=False)

        if success:
            print("✅ Motion played successfully!")
        else:
            print("❌ Motion playback failed!")

    def test_motion(self, slot: int):
        print(f"\n=== Testing Motion {slot} (Relative) ===")

        if not self.controller.is_connected():
            print("❌ uArm not connected! Cannot test motion.")
            return

        if not self.motion_manager.is_motion_recorded(slot):
            print(f"❌ No motion recorded in slot {slot}")
            return

        motion_info = self.motion_manager.get_motion_info(slot)
        print(f"Testing: {motion_info['name']} (relative positioning)")

        success = self.motion_manager.play_motion(slot, relative=True)

        if success:
            print("✅ Motion test completed!")
        else:
            print("❌ Motion test failed!")

    def delete_motion(self, slot: int):
        print(f"\n=== Deleting Motion {slot} ===")

        if not self.motion_manager.is_motion_recorded(slot):
            print(f"❌ No motion recorded in slot {slot} to delete.")
            return

        motion_info = self.motion_manager.get_motion_info(slot)
        print(f"Motion to delete: {motion_info['name']}")
        print(f"Description: {motion_info['description']}")

        confirm = input("\nAre you sure you want to delete this motion? (y/N): ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return

        success = self.motion_manager.delete_motion(slot)

        if success:
            print("✅ Motion deleted successfully!")
        else:
            print("❌ Motion deletion failed!")

    def show_detailed_status(self):
        print("\n=== Detailed Status ===")

        # Controller status
        controller_status = self.controller.get_status()
        print(f"Controller Connected: {controller_status['connected']}")
        print(f"Port: {controller_status['port'] or 'Auto-detect'}")
        if controller_status['last_error']:
            print(f"Last Error: {controller_status['last_error']}")
        if controller_status['position']:
            pos = controller_status['position']
            print(f"Current Position: {pos}")

        print()

        # Motion manager status
        manager_status = self.motion_manager.get_status()
        print(f"Storage Path: {manager_status['storage_path']}")
        print(f"Total Motion Slots: {manager_status['total_slots']}")
        print(f"Recorded Motions: {manager_status['recorded_motions']}")

        print("\nDetailed Motion Information:")
        for slot_num, motion in manager_status['motions'].items():
            print(f"\n  Slot {slot_num}: {motion['name']}")
            print(f"    Description: {motion['description']}")
            if 'recorded_at' in motion:
                print(f"    Recorded: {motion['recorded_at']}")
                if 'file_path' in motion:
                    exists = "✅" if os.path.exists(motion['file_path']) else "❌"
                    print(f"    File: {motion['file_path']} {exists}")
            else:
                print(f"    Status: Not recorded")

    def check_connection(self):
        print("\n=== Connection Check ===")

        if self.controller.is_connected():
            print("✅ uArm is connected and ready")

            # Try to get position
            pos = self.controller.get_position()
            if pos:
                print(f"Current position: {pos}")
            else:
                print("⚠️  Could not read current position")
        else:
            print("❌ uArm is not connected")
            print("Attempting to reconnect...")

            if self.controller.connect():
                print("✅ Reconnection successful!")
            else:
                print("❌ Reconnection failed")
                print(f"Error: {self.controller.last_error}")

    def home_uarm(self):
        print("\n=== Homing uArm ===")

        if not self.controller.is_connected():
            print("❌ uArm not connected!")
            return

        print("Moving uArm to home position...")
        success = self.controller.home()

        if success:
            print("✅ uArm homed successfully!")
        else:
            print("❌ Homing failed!")

    def cleanup(self):
        print("\nCleaning up...")
        if self.controller:
            self.controller.disconnect()
        print("Goodbye!")


def main():
    ui = UarmRecordingUI()

    try:
        ui.show_main_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        ui.cleanup()


if __name__ == "__main__":
    main()