import threading
import time
from typing import Optional

try:
    # Use original uFactory SDK (works with v3.2.0 firmware)
    from uarm.wrapper.swift_api import SwiftAPI
    UARM_AVAILABLE = True

    def uarm_scan_and_connect():
        """Scan and connect to uArm using original uFactory SDK"""
        try:
            swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
            swift.waiting_ready(timeout=3)
            return swift
        except Exception as e:
            print(f"uArm connection failed: {e}")
            return None

except ImportError as e:
    print(f"Warning: uArm Python wrapper not available: {e}")
    UARM_AVAILABLE = False
    uarm_scan_and_connect = None


class UarmController:
    def __init__(self, port: Optional[str] = None, connect_on_init: bool = True, auto_home: bool = True):
        self.robot = None
        self.port = port
        self.connected = False
        self.last_error = None
        self.connection_lock = threading.Lock()
        self.auto_home_on_connect = auto_home  # Control automatic homing

        # Default folded/discrete home position (low, folded back)
        self.home_position = {"x": 90, "y": -120, "z": 50}

        # Button state tracking using callbacks (3 buttons: on/off, menu, play)
        self.button_states = {"onoff": False, "menu": False, "play": False}  # key0=menu, key1=play, key2=onoff
        self.button_events = []  # Store button press events
        self.callbacks_registered = False

        # Suction pump state tracking
        self.suction_enabled = False

        if connect_on_init and UARM_AVAILABLE:
            self.connect()

    def connect(self) -> bool:
        if not UARM_AVAILABLE:
            self.last_error = "uArm Python wrapper not available"
            return False

        with self.connection_lock:
            try:
                print("[UarmController] Scanning for uArm devices...")
                self.robot = uarm_scan_and_connect()

                if self.robot:
                    self.connected = True
                    self.last_error = None
                    print(f"[UarmController] Connected successfully")

                    # Register button callbacks to override firmware functions
                    self._register_button_callbacks()

                    # Perform initial homing only if enabled
                    if self.auto_home_on_connect:
                        print("[UarmController] Performing automatic homing...")
                        self.home()
                    else:
                        print("[UarmController] Skipping automatic homing (disabled)")

                    return True
                else:
                    self.last_error = "No uArm devices found"
                    self.connected = False
                    return False

            except Exception as e:
                self.last_error = f"Connection failed: {e}"
                self.connected = False
                print(f"[ERROR] UarmController connection failed: {e}")
                return False

    def disconnect(self):
        with self.connection_lock:
            if self.robot:
                try:
                    # Release button callbacks first
                    self._release_button_callbacks()

                    # Only move to safe position if auto_home was enabled
                    if self.auto_home_on_connect:
                        print("[UarmController] Moving to safe position before disconnect")
                        self.home()
                        time.sleep(1)
                    else:
                        print("[UarmController] Skipping safe disconnect movement (auto_home disabled)")
                except Exception as e:
                    print(f"[WARNING] Error during safe disconnect: {e}")

                self.robot = None
                self.connected = False
                print("[UarmController] Disconnected")

    def is_connected(self) -> bool:
        return self.connected and self.robot is not None

    def home(self) -> bool:
        if not self.is_connected():
            return False

        try:
            # Move to the configured home position (default: folded/discrete)
            pos = self.home_position
            self.robot.set_position(x=pos["x"], y=pos["y"], z=pos["z"], speed=100, wait=True)
            print(f"[UarmController] Moved to home position ({pos['x']}, {pos['y']}, {pos['z']})")
            return True
        except Exception as e:
            self.last_error = f"Homing failed: {e}"
            print(f"[ERROR] UarmController homing failed: {e}")
            return False

    def set_home_position(self, x: float, y: float, z: float) -> bool:
        """Set a custom home position"""
        if not self.is_connected():
            print("[UarmController] Not connected - cannot validate position")
            return False

        try:
            # Test if the position is reachable
            current_pos = self.robot.get_position()
            self.robot.set_position(x=x, y=y, z=z, speed=100, wait=True)

            # If successful, save as home position
            self.home_position = {"x": x, "y": y, "z": z}
            print(f"[UarmController] Home position set to ({x}, {y}, {z})")
            return True
        except Exception as e:
            self.last_error = f"Invalid home position: {e}"
            print(f"[ERROR] Could not set home position ({x}, {y}, {z}): {e}")
            return False

    def get_home_position(self) -> dict:
        """Get the current home position"""
        return self.home_position.copy()

    def save_current_as_home(self) -> bool:
        """Save the current position as the new home position"""
        if not self.is_connected():
            return False

        try:
            current_pos = self.robot.get_position()
            if current_pos and len(current_pos) >= 3:
                self.home_position = {"x": current_pos[0], "y": current_pos[1], "z": current_pos[2]}
                print(f"[UarmController] Current position saved as home: {self.home_position}")
                return True
            else:
                print("[ERROR] Could not read current position")
                return False
        except Exception as e:
            self.last_error = f"Failed to save current position: {e}"
            print(f"[ERROR] Could not save current position as home: {e}")
            return False

    def move_to(self, x: float, y: float, z: float, speed: int = 100) -> bool:
        if not self.is_connected():
            return False

        try:
            # Original uFactory API method
            self.robot.set_position(x=x, y=y, z=z, speed=speed)
            return True
        except Exception as e:
            self.last_error = f"Move failed: {e}"
            print(f"[ERROR] UarmController move_to failed: {e}")
            return False

    def set_pump(self, on: bool) -> bool:
        if not self.is_connected():
            return False

        try:
            # Original uFactory API method
            self.robot.set_pump(on)
            self.suction_enabled = on  # Track state
            action = "activated" if on else "deactivated"
            print(f"[UarmController] Suction pump {action}")
            return True
        except Exception as e:
            self.last_error = f"Pump control failed: {e}"
            print(f"[ERROR] UarmController pump control failed: {e}")
            return False

    def toggle_pump(self) -> bool:
        """Toggle suction pump state"""
        return self.set_pump(not self.suction_enabled)

    def move_to_home(self) -> bool:
        """Move to home position"""
        return self.home()

    def set_gripper(self, catch: bool) -> bool:
        if not self.is_connected():
            return False

        try:
            # Original uFactory API method
            if catch:
                self.robot.set_gripper(True)
            else:
                self.robot.set_gripper(False)
            action = "closed" if catch else "opened"
            print(f"[UarmController] Gripper {action}")
            return True
        except Exception as e:
            self.last_error = f"Gripper control failed: {e}"
            print(f"[ERROR] UarmController gripper control failed: {e}")
            return False

    def get_position(self) -> Optional[dict]:
        if not self.is_connected():
            return None

        try:
            pos = self.robot.get_position()
            return pos
        except Exception as e:
            self.last_error = f"Position read failed: {e}"
            print(f"[ERROR] UarmController position read failed: {e}")
            return None

    def _register_button_callbacks(self):
        """Register callback functions to intercept button presses"""
        if not self.robot or self.callbacks_registered:
            return

        try:
            # Enable key reporting first
            if hasattr(self.robot, 'set_report_keys'):
                self.robot.set_report_keys(True)
                print("[UarmController] Key reporting enabled")

            # Register callbacks for all three keys
            if hasattr(self.robot, 'register_key0_callback'):
                self.robot.register_key0_callback(self._key0_callback)
                print("[UarmController] Key0 (menu) callback registered")

            if hasattr(self.robot, 'register_key1_callback'):
                self.robot.register_key1_callback(self._key1_callback)
                print("[UarmController] Key1 (play) callback registered")

            if hasattr(self.robot, 'register_key2_callback'):
                self.robot.register_key2_callback(self._key2_callback)
                print("[UarmController] Key2 (on/off) callback registered")

            self.callbacks_registered = True
            print("[UarmController] Button callbacks registered - firmware functions overridden")

        except Exception as e:
            print(f"[UarmController] Button callback registration failed: {e}")

    def _key0_callback(self, value):
        """Callback for key0 (menu button)"""
        pressed = bool(value)
        self.button_states["menu"] = pressed
        self.button_events.append({"button": "menu", "pressed": pressed, "time": time.time()})
        print(f"[UarmController] Menu button {'PRESSED' if pressed else 'RELEASED'}")

    def _key1_callback(self, value):
        """Callback for key1 (play button)"""
        pressed = bool(value)
        self.button_states["play"] = pressed
        self.button_events.append({"button": "play", "pressed": pressed, "time": time.time()})
        print(f"[UarmController] Play button {'PRESSED' if pressed else 'RELEASED'}")

    def _key2_callback(self, value):
        """Callback for key2 (on/off button)"""
        pressed = bool(value)
        self.button_states["onoff"] = pressed
        self.button_events.append({"button": "onoff", "pressed": pressed, "time": time.time()})
        print(f"[UarmController] On/Off button {'PRESSED' if pressed else 'RELEASED'}")

    def _release_button_callbacks(self):
        """Release button callbacks"""
        if not self.robot or not self.callbacks_registered:
            return

        try:
            if hasattr(self.robot, 'release_key0_callback'):
                self.robot.release_key0_callback()

            if hasattr(self.robot, 'release_key1_callback'):
                self.robot.release_key1_callback()

            if hasattr(self.robot, 'release_key2_callback'):
                self.robot.release_key2_callback()

            if hasattr(self.robot, 'set_report_keys'):
                self.robot.set_report_keys(False)

            self.callbacks_registered = False
            print("[UarmController] Button callbacks released")

        except Exception as e:
            print(f"[UarmController] Button callback release failed: {e}")

    def get_button_state(self) -> dict:
        """Get the state of all three base buttons on the uArm (onoff, menu, play)"""
        if not self.is_connected():
            return {"onoff": False, "menu": False, "play": False}

        # Return current button states from callback system
        return self.button_states.copy()

    def get_button_events(self) -> list:
        """Get list of button events and clear the event buffer"""
        events = self.button_events.copy()
        self.button_events.clear()
        return events

    def wait_for_button_press(self, timeout: float = 10.0) -> Optional[str]:
        """Wait for any button press and return which button ('menu' or 'play')"""
        start_time = time.time()
        initial_events = len(self.button_events)

        while time.time() - start_time < timeout:
            if len(self.button_events) > initial_events:
                # Check for press events
                for event in self.button_events[initial_events:]:
                    if event["pressed"]:  # Button press (not release)
                        return event["button"]
            time.sleep(0.1)

        return None  # Timeout

    def get_any_button_pressed(self) -> bool:
        """Check if any button is currently pressed"""
        states = self.get_button_state()
        return states["menu"] or states["play"]

    def get_menu_button_pressed(self) -> bool:
        """Check if menu button is currently pressed"""
        states = self.get_button_state()
        return states["menu"]

    def get_play_button_pressed(self) -> bool:
        """Check if play button is currently pressed"""
        states = self.get_button_state()
        return states["play"]

    def release_motors(self) -> bool:
        """Release/disable motors so arm can be moved manually"""
        if not self.is_connected():
            return False

        try:
            # Disable servo motors using the correct API
            self.robot.set_servo_detach()  # Detach all servos at once
            print("[UarmController] Motors released - arm can be moved manually")
            return True
        except Exception as e:
            self.last_error = f"Motor release failed: {e}"
            print(f"[ERROR] UarmController motor release failed: {e}")
            # Try alternative method
            try:
                # Alternative: set mode to manual
                self.robot.set_mode(0)  # Mode 0 = manual
                print("[UarmController] Motors released via mode setting")
                return True
            except Exception as e2:
                print(f"[ERROR] Alternative motor release also failed: {e2}")
                return False

    def enable_motors(self) -> bool:
        """Enable/attach motors for powered movement"""
        if not self.is_connected():
            return False

        try:
            # Re-attach servo motors
            self.robot.set_servo_attach()  # Attach all servos
            print("[UarmController] Motors enabled - arm will hold position")
            return True
        except Exception as e:
            self.last_error = f"Motor enable failed: {e}"
            print(f"[ERROR] UarmController motor enable failed: {e}")
            # Try alternative method
            try:
                # Alternative: set mode back to normal
                self.robot.set_mode(1)  # Mode 1 = normal/servo
                print("[UarmController] Motors enabled via mode setting")
                return True
            except Exception as e2:
                print(f"[ERROR] Alternative motor enable also failed: {e2}")
                return False

    def playback_motion(self, motion_name: str, relative: bool = False) -> bool:
        if not self.is_connected():
            return False

        try:
            self.robot.playback(motion_name, relative=relative)
            print(f"[UarmController] Played back motion: {motion_name} (relative={relative})")
            return True
        except Exception as e:
            self.last_error = f"Playback failed: {e}"
            print(f"[ERROR] UarmController playback failed for {motion_name}: {e}")
            return False

    def get_status(self) -> dict:
        return {
            "connected": self.connected,
            "port": self.port,
            "last_error": self.last_error,
            "position": self.get_position() if self.is_connected() else None
        }

    def __del__(self):
        try:
            self.disconnect()
        except:
            pass