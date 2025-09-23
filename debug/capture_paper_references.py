#!/usr/bin/env python3
"""
Paper Detection Reference Image Capture Tool

Interactive camera preview tool for capturing paper present/absent reference images.
Features:
- Live camera preview window
- Automatic servo positioning for paper detection angle
- Hotkey-based snapshot capture
- Visual feedback and instructions

Controls:
- P: Capture "Paper Present" reference image
- A: Capture "Paper Absent" reference image
- Q: Quit
"""

import os
import sys
import threading
import time

import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT
from safety.paper_detection import capture_paper_absent_reference, capture_paper_present_reference, get_paper_detection_status

# Always align with runtime drawing lock pose: use drawing tilt (TILT_MIN+2)
try:
    from config.config import TILT_MIN
except Exception:
    TILT_MIN = 50


class PaperReferenceCaptureTool:
    """Interactive tool for capturing paper detection reference images."""

    def __init__(self):
        self.camera = None
        self.servos = None
        self.preview_running = False
        self.last_frame = None
        self.status_text = "Initializing..."

        # Detection area parameters - MUST MATCH safety/paper_detection.py coordinates exactly
        self.detection_x = 0.6  # X center as fraction of width (right of center)
        self.detection_y = 0.52  # Y center as fraction of height
        self.detection_width = 0.25  # Width as fraction of frame width
        self.detection_height = 0.2  # Height as fraction of frame height

    def init_hardware(self):
        """Initialize camera and servo hardware."""
        try:
            print("🔧 Initializing hardware...")

            import os

            import cv2

            from config.config import BAUD_RATE, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, USE_SERVO
            from servo_control.servo_control import ServoController

            # Initialize camera like machine.py does
            self.camera = cv2.VideoCapture(CAMERA_INDEX if CAMERA_INDEX else 0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")

            # Match machine.py capture resolution and basic properties for consistent FOV/appearance
            try:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                # Load persisted camera settings if available
                import json

                settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "camera_settings.json")
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    # Apply basic properties if present
                    if "brightness" in s:
                        self.camera.set(cv2.CAP_PROP_BRIGHTNESS, float(s["brightness"]))
                    if "contrast" in s:
                        self.camera.set(cv2.CAP_PROP_CONTRAST, float(s["contrast"]))
                    if "saturation" in s:
                        self.camera.set(cv2.CAP_PROP_SATURATION, float(s["saturation"]))
                    if "sharpness" in s:
                        self.camera.set(cv2.CAP_PROP_SHARPNESS, float(s["sharpness"]))
            except Exception:
                pass

            # Initialize servos like machine.py does
            if not USE_SERVO:
                raise Exception("USE_SERVO is disabled in config")

            # Get Arduino devices mapping
            try:
                from config.config import SERIAL_PORT  # This should be the servo port

                servo_port = SERIAL_PORT
            except:
                servo_port = "/dev/arduino_lunggaze"  # Fallback to default

            if not os.path.exists(servo_port):
                raise Exception(f"Servo port not found: {servo_port}")

            print(f"🔧 Connecting to servos on {servo_port}...")
            self.servos = ServoController(port=servo_port, baudrate=BAUD_RATE)

            # Wait for Arduino to initialize like machine.py does
            time.sleep(0.5)

            # Don't test servos during init - let them attach naturally when positioned
            print("🔧 Servo controller ready (servos will attach when first commanded)")

            print("✅ Hardware initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Hardware initialization failed: {e}")
            print("Make sure camera and servos are connected and accessible")
            return False

    def position_for_paper_detection(self):
        """Position servos to look down at drawing area."""
        try:
            from config.config import PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT

            print(f"🎯 Positioning servos for paper detection...")
            # Force the same tilt as runtime drawing lock/paper check
            det_tilt = TILT_MIN + 2
            print(f"   Pan: {PAPER_DETECTION_GAZE_PAN}°")
            print(f"   Tilt: {det_tilt}° (drawing lock tilt)")

            # Send commands carefully - Arduino will attach servos on first command
            print(f"🔧 Positioning servos (Pan: {PAPER_DETECTION_GAZE_PAN}°, Tilt: {det_tilt}°)")

            # Debug: Check servo controller state
            print(f"🔧 Servo controller state:")
            print(f"   Serial port: {self.servos.ser.port if self.servos.ser else 'None'}")
            print(f"   Serial open: {self.servos.ser.is_open if self.servos.ser else 'None'}")
            print(f"   Disabled: {self.servos.disabled}")

            # Smooth pan movement - send intermediate positions
            print(f"🔧 Moving pan smoothly to {PAPER_DETECTION_GAZE_PAN}°")
            current_pan = 90  # Assume starting at center
            target_pan = PAPER_DETECTION_GAZE_PAN

            # Send a few intermediate positions for smooth movement
            steps = 3
            for i in range(steps + 1):
                intermediate_pan = current_pan + (target_pan - current_pan) * i / steps
                pan_command = f"PAN:{int(intermediate_pan)}"
                try:
                    self.servos.send(pan_command, key=f"pan_smooth_{i}")
                    print(f"   Step {i+1}: {pan_command}")
                    time.sleep(0.3)  # Small delay between steps
                except Exception as e:
                    print(f"   Pan step {i+1} failed: {e}")

            time.sleep(1.0)  # Final settling time

            # Smooth tilt movement - send intermediate positions
            print(f"🔧 Moving tilt smoothly to {det_tilt}°")
            current_tilt = 90  # Assume starting at center
            target_tilt = det_tilt

            # Send a few intermediate positions for smooth movement
            for i in range(steps + 1):
                intermediate_tilt = current_tilt + (target_tilt - current_tilt) * i / steps
                # Arduino constrains tilt to 65-115°, so make sure we're in range
                constrained_tilt = max(65, min(115, int(intermediate_tilt)))
                tilt_command = f"TILT:{constrained_tilt}"
                try:
                    self.servos.send(tilt_command, key=f"tilt_smooth_{i}")
                    print(f"   Step {i+1}: {tilt_command}")
                    time.sleep(0.3)  # Small delay between steps
                except Exception as e:
                    print(f"   Tilt step {i+1} failed: {e}")

            time.sleep(1.0)  # Final settling time

            # Check what was actually sent
            print(f"🔧 Last sent commands:")
            for key, value in self.servos.last_sent.items():
                print(f"   {key}: {value}")

            time.sleep(1.5)  # Extra time to allow servos to settle

            print("✅ Servos positioned (or commands sent)")
            self.status_text = "Servos positioned - Ready for capture"
            return True

        except Exception as e:
            print(f"❌ Servo positioning failed: {e}")
            import traceback

            traceback.print_exc()
            self.status_text = f"Servo error: {e}"
            return False

    def update_preview(self):
        """Update camera preview in separate thread."""
        while self.preview_running:
            try:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    self.last_frame = frame.copy()
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Preview update error: {e}")
                time.sleep(0.1)

    def draw_overlay(self, frame):
        """Draw status overlay on preview frame."""
        height, width = frame.shape[:2]

        # Dark semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Status text
        cv2.putText(frame, "Paper Detection Reference Capture", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, self.status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Controls
        cv2.putText(frame, "Controls: [P] Paper Present  [A] Paper Absent  [Q] Quit", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Adjust Area: Arrow Keys=Move  +/-=Size  R=Reset", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Adjustable detection area
        det_center_x = int(self.detection_x * width)
        det_center_y = int(self.detection_y * height)
        det_w = int(self.detection_width * width)
        det_h = int(self.detection_height * height)

        # Detection area rectangle
        x1 = det_center_x - det_w // 2
        y1 = det_center_y - det_h // 2
        x2 = det_center_x + det_w // 2
        y2 = det_center_y + det_h // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Paper Detection Area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Center crosshair
        cv2.line(frame, (det_center_x - 10, det_center_y), (det_center_x + 10, det_center_y), (0, 255, 255), 1)
        cv2.line(frame, (det_center_x, det_center_y - 10), (det_center_x, det_center_y + 10), (0, 255, 255), 1)

        return frame

    def capture_paper_present(self):
        """Capture paper present reference image."""
        try:
            print("\n📄 Capturing PAPER PRESENT reference...")
            self.status_text = "Capturing Paper Present reference..."

            # Create a simple camera wrapper that matches the expected interface
            class CameraWrapper:
                def __init__(self, cv2_camera):
                    self.cv2_camera = cv2_camera

                def read_frame(self):
                    ret, frame = self.cv2_camera.read()
                    return frame if ret else None

            camera_wrapper = CameraWrapper(self.camera)
            success = capture_paper_present_reference(camera_wrapper, self.servos)

            if success:
                print("✅ Paper Present reference captured!")
                self.status_text = "✅ Paper Present reference captured!"
            else:
                print("❌ Failed to capture Paper Present reference")
                self.status_text = "❌ Paper Present capture failed"

            return success

        except Exception as e:
            print(f"❌ Error capturing Paper Present reference: {e}")
            self.status_text = f"❌ Error: {e}"
            return False

    def capture_paper_absent(self):
        """Capture paper absent reference image."""
        try:
            print("\n📄 Capturing PAPER ABSENT reference...")
            self.status_text = "Capturing Paper Absent reference..."

            # Create a simple camera wrapper that matches the expected interface
            class CameraWrapper:
                def __init__(self, cv2_camera):
                    self.cv2_camera = cv2_camera

                def read_frame(self):
                    ret, frame = self.cv2_camera.read()
                    return frame if ret else None

            camera_wrapper = CameraWrapper(self.camera)
            success = capture_paper_absent_reference(camera_wrapper, self.servos)

            if success:
                print("✅ Paper Absent reference captured!")
                self.status_text = "✅ Paper Absent reference captured!"
            else:
                print("❌ Failed to capture Paper Absent reference")
                self.status_text = "❌ Paper Absent capture failed"

            return success

        except Exception as e:
            print(f"❌ Error capturing Paper Absent reference: {e}")
            self.status_text = f"❌ Error: {e}"
            return False

    def show_current_status(self):
        """Display current paper detection system status."""
        print("\n📋 Current Paper Detection Status:")
        print("=" * 50)

        status = get_paper_detection_status()
        print(f"Paper Present Ref: {'✅' if status['paper_present_exists'] else '❌'} ({status['paper_present_path']})")
        print(f"Paper Absent Ref:  {'✅' if status['paper_absent_exists'] else '❌'} ({status['paper_absent_path']})")
        print(f"System Enabled: {'✅' if status['enabled'] else '❌'}")
        print(f"Method: {status['method']}")
        print(f"Confidence Threshold: {status['confidence_threshold']}")

    def run(self):
        """Run the interactive reference capture tool."""
        print("📄 Paper Detection Reference Capture Tool")
        print("=" * 50)

        if not self.init_hardware():
            return False

        if not self.position_for_paper_detection():
            return False

        # Show current status
        self.show_current_status()

        print("\n🎥 Starting camera preview...")
        print("📋 Instructions:")
        print("   1. Position paper properly in the detection area (yellow box)")
        print("   2. Press 'P' to capture Paper Present reference")
        print("   3. Remove paper completely")
        print("   4. Press 'A' to capture Paper Absent reference")
        print("   5. Press 'Q' to quit")

        # Start preview thread
        self.preview_running = True
        preview_thread = threading.Thread(target=self.update_preview, daemon=True)
        preview_thread.start()

        # Wait for first frame
        time.sleep(1)

        # Main preview loop
        try:
            while True:
                if self.last_frame is not None:
                    # Draw overlay and show frame (unflipped to match capture orientation)
                    display_frame = self.draw_overlay(self.last_frame.copy())
                    cv2.imshow("Paper Detection Reference Capture", display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # Q or ESC
                    print("\n👋 Exiting...")
                    break
                elif key == ord("p"):
                    self.capture_paper_present()
                elif key == ord("a"):
                    self.capture_paper_absent()
                elif key == ord("r"):  # Reset detection area
                    self.detection_x = 0.5
                    self.detection_y = 0.5
                    self.detection_width = 0.4
                    self.detection_height = 0.3
                    print("🔄 Detection area reset to default")
                # Arrow keys for position
                elif key == 82:  # Up arrow
                    self.detection_y = max(0.1, self.detection_y - 0.02)
                elif key == 84:  # Down arrow
                    self.detection_y = min(0.9, self.detection_y + 0.02)
                elif key == 81:  # Left arrow
                    self.detection_x = max(0.1, self.detection_x - 0.02)
                elif key == 83:  # Right arrow
                    self.detection_x = min(0.9, self.detection_x + 0.02)
                # +/- for size
                elif key == ord("+") or key == ord("="):  # Increase size
                    self.detection_width = min(0.8, self.detection_width + 0.02)
                    self.detection_height = min(0.8, self.detection_height + 0.02)
                elif key == ord("-"):  # Decrease size
                    self.detection_width = max(0.1, self.detection_width - 0.02)
                    self.detection_height = max(0.1, self.detection_height - 0.02)

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")

        finally:
            # Cleanup
            self.preview_running = False
            preview_thread.join(timeout=1)
            cv2.destroyAllWindows()

            # Release camera
            if self.camera:
                self.camera.release()

            # Show final status
            print("\n📋 Final Status:")
            self.show_current_status()

        return True


def main():
    """Main entry point."""
    tool = PaperReferenceCaptureTool()
    success = tool.run()

    if success:
        print("\n✅ Reference capture session completed")
    else:
        print("\n❌ Reference capture session failed")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
