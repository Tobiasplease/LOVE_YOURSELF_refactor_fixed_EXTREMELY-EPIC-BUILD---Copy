"""
Simple API for uArm Swift Pro - 3 Utilitarian Motions

This provides a clean interface to trigger the 3 pre-recorded motions
from anywhere in the codebase without emotion logic.
"""

from typing import Optional


class UarmSimpleAPI:
    def __init__(self, controller=None, motion_manager=None):
        self.controller = controller
        self.motion_manager = motion_manager

    def is_available(self) -> bool:
        """Check if uArm is connected and ready"""
        return (self.controller and self.controller.is_connected() and
                self.motion_manager is not None)

    def execute_pickup(self) -> bool:
        """Execute the pickup motion (slot 1)"""
        if not self.is_available():
            print("[uArm] Not available for pickup motion")
            return False

        if not self.motion_manager.is_motion_recorded(1):
            print("[uArm] Pickup motion not recorded - use recording UI first")
            return False

        try:
            print("[uArm] Executing pickup motion...")
            return self.motion_manager.play_motion(1, relative=False)
        except Exception as e:
            print(f"[uArm] Pickup motion failed: {e}")
            return False

    def execute_place(self) -> bool:
        """Execute the place motion (slot 2)"""
        if not self.is_available():
            print("[uArm] Not available for place motion")
            return False

        if not self.motion_manager.is_motion_recorded(2):
            print("[uArm] Place motion not recorded - use recording UI first")
            return False

        try:
            print("[uArm] Executing place motion...")
            return self.motion_manager.play_motion(2, relative=False)
        except Exception as e:
            print(f"[uArm] Place motion failed: {e}")
            return False

    def execute_gesture(self) -> bool:
        """Execute the gesture motion (slot 3)"""
        if not self.is_available():
            print("[uArm] Not available for gesture motion")
            return False

        if not self.motion_manager.is_motion_recorded(3):
            print("[uArm] Gesture motion not recorded - use recording UI first")
            return False

        try:
            print("[uArm] Executing gesture motion...")
            return self.motion_manager.play_motion(3, relative=True)  # Gestures are relative
        except Exception as e:
            print(f"[uArm] Gesture motion failed: {e}")
            return False

    def get_status(self) -> dict:
        """Get current status of uArm and recorded motions"""
        if not self.motion_manager:
            return {"available": False, "error": "Motion manager not initialized"}

        status = self.motion_manager.get_status()
        status["available"] = self.is_available()
        status["motions_ready"] = {
            "pickup": self.motion_manager.is_motion_recorded(1),
            "place": self.motion_manager.is_motion_recorded(2),
            "gesture": self.motion_manager.is_motion_recorded(3)
        }

        return status

    def home(self) -> bool:
        """Move uArm to home position"""
        if not self.controller:
            print("[uArm] Controller not available for homing")
            return False

        try:
            print("[uArm] Moving to home position...")
            return self.controller.home()
        except Exception as e:
            print(f"[uArm] Homing failed: {e}")
            return False

    def set_home_position(self, x: float, y: float, z: float) -> bool:
        """Set a custom home position"""
        if not self.controller:
            print("[uArm] Controller not available")
            return False

        try:
            print(f"[uArm] Setting home position to ({x}, {y}, {z})...")
            return self.controller.set_home_position(x, y, z)
        except Exception as e:
            print(f"[uArm] Set home position failed: {e}")
            return False

    def save_current_as_home(self) -> bool:
        """Save the current position as the new home position"""
        if not self.controller:
            print("[uArm] Controller not available")
            return False

        try:
            print("[uArm] Saving current position as home...")
            return self.controller.save_current_as_home()
        except Exception as e:
            print(f"[uArm] Save home position failed: {e}")
            return False

    def get_home_position(self) -> dict:
        """Get the current home position"""
        if not self.controller:
            return {"error": "Controller not available"}

        return self.controller.get_home_position()


# Global API instance (will be set in machine.py)
uarm_api: Optional[UarmSimpleAPI] = None


# Convenience functions for easy access from anywhere
def pickup() -> bool:
    """Execute pickup motion - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.execute_pickup()
    print("[uArm] API not initialized")
    return False


def place() -> bool:
    """Execute place motion - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.execute_place()
    print("[uArm] API not initialized")
    return False


def gesture() -> bool:
    """Execute gesture motion - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.execute_gesture()
    print("[uArm] API not initialized")
    return False


def home() -> bool:
    """Move to home position - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.home()
    print("[uArm] API not initialized")
    return False


def status() -> dict:
    """Get uArm status - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.get_status()
    return {"available": False, "error": "API not initialized"}


def is_available() -> bool:
    """Check if uArm is ready - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.is_available()
    return False


def set_home_position(x: float, y: float, z: float) -> bool:
    """Set custom home position - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.set_home_position(x, y, z)
    print("[uArm] API not initialized")
    return False


def save_current_as_home() -> bool:
    """Save current position as home - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.save_current_as_home()
    print("[uArm] API not initialized")
    return False


def get_home_position() -> dict:
    """Get home position - call from anywhere in the codebase"""
    if uarm_api:
        return uarm_api.get_home_position()
    return {"error": "API not initialized"}


def suction_on() -> bool:
    """Activate suction cup - call from anywhere in the codebase"""
    if uarm_api and uarm_api.controller:
        return uarm_api.controller.set_pump(True)
    print("[uArm] API not initialized")
    return False


def suction_off() -> bool:
    """Deactivate suction cup - call from anywhere in the codebase"""
    if uarm_api and uarm_api.controller:
        return uarm_api.controller.set_pump(False)
    print("[uArm] API not initialized")
    return False


def suction_toggle() -> bool:
    """Toggle suction cup state - call from anywhere in the codebase"""
    if uarm_api and uarm_api.controller:
        # This would require tracking current state - for now just return False
        print("[uArm] Use suction_on() or suction_off() for explicit control")
        return False
    print("[uArm] API not initialized")
    return False