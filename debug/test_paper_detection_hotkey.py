#!/usr/bin/env python3
"""
Interactive paper detection test tool with hotkey support.
Press SPACE to test paper detection on current camera view.
Press 'q' to quit.
"""

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import requests

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.config import CAMERA_INDEX, MOOD_SNAPSHOT_FOLDER
from utils.ollama import query_ollama


class PaperDetectionTester:
    def __init__(self):
        self.camera = None
        self.running = False
        self.testing = False
        self.test_count = 0

        # Ensure test images directory exists
        self.test_dir = os.path.join(MOOD_SNAPSHOT_FOLDER, "paper_detection_tests")
        Path(self.test_dir).mkdir(parents=True, exist_ok=True)

    def crop_detection_area(self, image_path: str) -> str:
        """Crop image to detection area using same coordinates as paper detection system."""
        # Use exact same coordinates as safety/paper_detection.py
        detection_x = 0.25  # X center as fraction of width (center-left of white paper)
        detection_y = 0.47  # Y center as fraction of height (center of white paper)
        detection_width = 0.22  # Width as fraction of frame width
        detection_height = 0.18  # Height as fraction of frame height

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return image_path  # Return original if crop fails

        h, w = img.shape[:2]

        # Calculate crop coordinates
        det_center_x = int(detection_x * w)
        det_center_y = int(detection_y * h)
        det_w = int(detection_width * w)
        det_h = int(detection_height * h)

        # Crop area
        x1 = det_center_x - det_w // 2
        y1 = det_center_y - det_h // 2
        x2 = det_center_x + det_w // 2
        y2 = det_center_y + det_h // 2

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop the image
        cropped = img[y1:y2, x1:x2]

        # Save cropped version
        cropped_path = image_path.replace(".jpg", "_CROPPED.jpg")
        cv2.imwrite(cropped_path, cropped)

        return cropped_path

    def query_ollama_multi_image(self, prompt: str, image_paths: list, model: str = "llava:7b-v1.6-mistral-q5_1") -> str:
        """Send multiple images to ollama for comparison."""
        payload = {"model": model, "prompt": prompt, "stream": False, "images": []}

        # Add all images to payload
        for image_path in image_paths:
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    payload["images"].append(img_b64)

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json().get("response", "")
            return result
        except Exception as e:
            return f"Error: {e}"

    def init_camera(self):
        """Initialize camera."""
        print(f"🎥 Initializing camera {CAMERA_INDEX}...")
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print(f"❌ Failed to open camera {CAMERA_INDEX}")
            return False

        # Set camera resolution for consistency
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✅ Camera initialized")
        return True

    def capture_frame(self):
        """Capture current frame from camera."""
        if not self.camera or not self.camera.isOpened():
            return None

        ret, frame = self.camera.read()
        if not ret:
            print("❌ Failed to capture frame")
            return None

        return frame

    def test_paper_detection(self):
        """Run paper detection test on current camera view."""
        if self.testing:
            print("⏳ Test already in progress...")
            return

        self.testing = True
        self.test_count += 1

        print(f"\n🔍 Running paper detection test #{self.test_count}...")

        try:
            # Capture frame
            frame = self.capture_frame()
            if frame is None:
                print("❌ Failed to capture frame")
                return

            # Save test image
            timestamp = int(time.time())
            test_image_path = os.path.join(self.test_dir, f"test_{timestamp}_{self.test_count:03d}.jpg")
            cv2.imwrite(test_image_path, frame)
            print(f"📸 Captured: {test_image_path}")

            # Text reading test
            prompt = """Do you see the text saying "NO PAPER"?

Answer: YES or NO"""

            print("🤖 Querying Ollama with direct contextual approach...")
            start_time = time.time()

            # Query ollama with deterministic parameters for consistency
            import base64

            import requests

            # Read and encode image
            with open(test_image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": "llava:7b-v1.6-mistral-q5_1",
                "prompt": prompt,
                "stream": False,
                "images": [img_b64],
                "options": {
                    "temperature": 0.1,  # Very low for consistency
                    "top_p": 0.9,
                    "top_k": 10,
                    "repeat_penalty": 1.0,
                    "seed": 42,  # Fixed seed for deterministic results
                },
            }

            try:
                response_obj = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
                response_obj.raise_for_status()
                response = response_obj.json().get("response", "")
            except Exception as e:
                response = f"Error: {e}"

            duration = time.time() - start_time

            # Parse response
            paper_present, confidence, reason = self.parse_response(response)

            # Display results
            print(f"\n📄 RESULTS (took {duration:.1f}s):")
            print(f"   Paper Present: {'✅ YES' if paper_present else '❌ NO'}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Reason: {reason}")
            print(f"   Image saved: {test_image_path}")

            if confidence < 0.65:  # Current threshold
                print(f"⚠️  Below confidence threshold (0.65)")

        except Exception as e:
            print(f"❌ Test failed: {e}")
        finally:
            self.testing = False
            print("\n" + "=" * 60)
            print("Press SPACE to test again, 'q' to quit")

    def parse_response(self, response):
        """Parse YES/NO response about 'NO PAPER' text visibility."""
        # Parse YES/NO response - if it sees "NO PAPER" text, that means no paper present
        response_upper = response.strip().upper()
        if "YES" in response_upper:
            paper_present = False  # YES = sees "NO PAPER" text = no paper present
            confidence = 0.8
            reason = "LLM detected 'NO PAPER' text (no paper present)"
        elif "NO" in response_upper:
            paper_present = True  # NO = doesn't see "NO PAPER" text = paper present
            confidence = 0.8
            reason = "LLM did not detect 'NO PAPER' text (paper present)"
        else:
            paper_present = False
            confidence = 0.0
            reason = f"Unclear response: {response.strip()}"

        return paper_present, confidence, reason

    def show_camera_feed(self):
        """Show live camera feed with instructions."""
        print("\n🎥 Live Camera Feed")
        print("Controls:")
        print("  SPACE - Run paper detection test")
        print("  'q' - Quit")
        print("=" * 60)

        while self.running:
            frame = self.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Add instructions overlay
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, f"Tests run: {self.test_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(overlay_frame, "SPACE: Test paper detection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(overlay_frame, "Q: Quit", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.testing:
                cv2.putText(overlay_frame, "TESTING...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("Paper Detection Tester", overlay_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):  # Space key
                threading.Thread(target=self.test_paper_detection, daemon=True).start()
            elif key == ord("q"):  # Quit
                break

        cv2.destroyAllWindows()

    def run(self):
        """Main run loop."""
        if not self.init_camera():
            return False

        self.running = True

        try:
            self.show_camera_feed()
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.running = False
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("🔄 Camera released, windows closed")

        return True


def main():
    print("🔧 Paper Detection Interactive Tester")
    print("This tool lets you test paper detection with live camera feed")

    tester = PaperDetectionTester()
    tester.run()


if __name__ == "__main__":
    main()
