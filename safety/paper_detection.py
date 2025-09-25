#!/usr/bin/env python3
"""
Paper Detection Safety System

Prevents drawing on bare surfaces by checking for paper presence before execution.
Uses vision-based detection with LLM analysis for reliable paper verification.
"""

import os
import time
import cv2
import base64
import json
import requests
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from config.config import (
    ENABLE_PAPER_DETECTION,
    PAPER_CHECK_METHOD,
    PAPER_DETECTION_CONFIDENCE_THRESHOLD,
    PAPER_DETECTION_TEXT,
    PAPER_REFERENCE_IMAGE_PATH,
    PAPER_PRESENT_REFERENCE_PATH,
    PAPER_ABSENT_REFERENCE_PATH,
    PAPER_DETECTION_GAZE_PAN,
    PAPER_DETECTION_GAZE_TILT,
    PAPER_CHECK_TIMEOUT,
    ALLOW_PAPER_DETECTION_OVERRIDE,
    MOOD_SNAPSHOT_FOLDER
)
from utils.ollama import query_ollama
from event_logging.event_logger import log_json_entry, LogType
from event_logging.run_manager import get_run_image_path


@dataclass
class PaperCheckResult:
    """Result of paper detection check."""
    paper_present: bool
    confidence: float
    method_used: str
    check_image_path: str
    timestamp: float
    llm_response: str
    error_message: Optional[str] = None


class PaperDetector:
    """Vision-based paper detection for drawing safety."""

    def __init__(self):
        self.reference_image_path = PAPER_REFERENCE_IMAGE_PATH
        self.present_image_path = PAPER_PRESENT_REFERENCE_PATH
        self.absent_image_path = PAPER_ABSENT_REFERENCE_PATH
        self.calibration_dir = os.path.dirname(self.reference_image_path)

        # Ensure calibration directory exists
        os.makedirs(self.calibration_dir, exist_ok=True)

        # Track detection history for debugging
        self.detection_history = []

    def capture_reference_image(self, camera, servos) -> bool:
        """
        Capture reference image of paper properly positioned in drawing area.

        Args:
            camera: Camera instance for image capture
            servos: Servo controller for gaze positioning

        Returns:
            bool: True if reference image captured successfully
        """
        try:
            log_json_entry(
                LogType.DEBUG,
                {"action": "capture_reference_image", "component": "paper_detection"},
                print_message="[📄] Capturing paper reference image..."
            )

            # Position gaze to look down at drawing area
            self._position_gaze_for_detection(servos)
            time.sleep(1.0)  # Allow gaze to settle

            # Capture image
            frame = camera.read_frame()
            if frame is None:
                raise Exception("Failed to capture frame from camera")

            # Save reference image
            cv2.imwrite(self.reference_image_path, frame)

            log_json_entry(
                LogType.DEBUG,
                {
                    "action": "reference_image_saved",
                    "path": self.reference_image_path,
                    "component": "paper_detection"
                },
                print_message=f"[📄] ✓ Reference image saved: {self.reference_image_path}"
            )

            return True

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {
                    "action": "capture_reference_failed",
                    "error": str(e),
                    "component": "paper_detection"
                },
                print_message=f"[📄] ✗ Failed to capture reference image: {e}"
            )
            return False

    def capture_present_image(self, camera, servos) -> bool:
        """Capture 'paper present' reference image to configured path."""
        try:
            self._position_gaze_for_detection(servos)
            time.sleep(1.0)
            frame = camera.read_frame()
            if frame is None:
                raise Exception("Failed to capture frame from camera")
            Path(os.path.dirname(self.present_image_path)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(self.present_image_path, frame)
            log_json_entry(LogType.DEBUG, {"action": "present_reference_saved", "path": self.present_image_path}, print_message=f"[📄] ✓ Paper PRESENT saved: {self.present_image_path}")
            return True
        except Exception as e:
            log_json_entry(LogType.ERROR, {"action": "present_reference_failed", "error": str(e)}, print_message=f"[📄] ✗ Failed PRESENT capture: {e}")
            return False

    def capture_absent_image(self, camera, servos) -> bool:
        """Capture 'paper absent' reference image to configured path."""
        try:
            self._position_gaze_for_detection(servos)
            time.sleep(1.0)
            frame = camera.read_frame()
            if frame is None:
                raise Exception("Failed to capture frame from camera")
            Path(os.path.dirname(self.absent_image_path)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(self.absent_image_path, frame)
            log_json_entry(LogType.DEBUG, {"action": "absent_reference_saved", "path": self.absent_image_path}, print_message=f"[📄] ✓ Paper ABSENT saved: {self.absent_image_path}")
            return True
        except Exception as e:
            log_json_entry(LogType.ERROR, {"action": "absent_reference_failed", "error": str(e)}, print_message=f"[📄] ✗ Failed ABSENT capture: {e}")
            return False

    def check_paper_present(self, camera, servos, captioner=None) -> PaperCheckResult:
        """
        Check if paper is present in the drawing area.

        Args:
            camera: Camera instance for image capture
            servos: Servo controller for gaze positioning
            captioner: Captioner instance for LLM queries (optional)

        Returns:
            PaperCheckResult: Detection result with confidence and details
        """
        start_time = time.time()

        try:
            if not ENABLE_PAPER_DETECTION:
                return PaperCheckResult(
                    paper_present=True,
                    confidence=1.0,
                    method_used="disabled",
                    check_image_path="",
                    timestamp=start_time,
                    llm_response="Paper detection disabled in config"
                )

            log_json_entry(
                LogType.DEBUG,
                {"action": "paper_check_start", "method": PAPER_CHECK_METHOD},
                print_message=f"[📄] Checking for paper using {PAPER_CHECK_METHOD} method..."
            )

            # Position gaze to look down at drawing area
            self._position_gaze_for_detection(servos)
            time.sleep(1.0)  # Allow gaze to settle

            # Capture current view
            frame = camera.read_frame()
            if frame is None:
                raise Exception("Failed to capture frame for paper detection")

            # Save check image with timestamp
            timestamp_str = str(int(start_time))
            check_image_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"paper_check_{timestamp_str}.jpg")
            cv2.imwrite(check_image_path, frame)

            # Perform detection based on configured method
            if PAPER_CHECK_METHOD == "reference":
                result = self._check_with_reference_image(check_image_path, captioner)
            else:  # "direct"
                result = self._check_direct_detection(check_image_path, captioner)

            # Update result with common fields
            result.check_image_path = check_image_path
            result.timestamp = start_time

            # Log result
            log_json_entry(
                LogType.DECISION,
                {
                    "action": "paper_check_complete",
                    "paper_present": result.paper_present,
                    "confidence": result.confidence,
                    "method": result.method_used,
                    "check_image": check_image_path,
                    "duration": time.time() - start_time
                },
                print_message=f"[📄] {'✓' if result.paper_present else '✗'} Paper detection: {result.confidence:.2f} confidence"
            )

            # Store in history for debugging
            self.detection_history.append(result)
            if len(self.detection_history) > 10:  # Keep last 10 results
                self.detection_history.pop(0)

            return result

        except Exception as e:
            error_result = PaperCheckResult(
                paper_present=False,
                confidence=0.0,
                method_used=PAPER_CHECK_METHOD,
                check_image_path="",
                timestamp=start_time,
                llm_response="",
                error_message=str(e)
            )

            log_json_entry(
                LogType.ERROR,
                {
                    "action": "paper_check_failed",
                    "error": str(e),
                    "component": "paper_detection"
                },
                print_message=f"[📄] ✗ Paper detection failed: {e}"
            )

            return error_result

    def _position_gaze_for_detection(self, servos):
        """Position servos to look down at drawing area."""
        # Use direct servo control instead of gaze locking to prevent stuck states
        servos.set_pan(PAPER_DETECTION_GAZE_PAN)
        time.sleep(0.1)
        servos.set_tilt(PAPER_DETECTION_GAZE_TILT)

    def _crop_detection_area(self, image_path: str) -> str:
        """Crop image to focus on the paper detection area."""
        # Coordinates calculated from actual screenshot positioning
        detection_x = 0.25     # X center as fraction of width (center-left of white paper)
        detection_y = 0.47     # Y center as fraction of height (center of white paper)
        detection_width = 0.22 # Width as fraction of frame width
        detection_height = 0.18 # Height as fraction of frame height

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
        cropped_path = image_path.replace('.jpg', '_detection_area.jpg')
        cv2.imwrite(cropped_path, cropped)

        return cropped_path

    def _query_ollama_multi_image(self, prompt: str, image_paths: list, model: str = "llava:7b-v1.6-mistral-q5_1") -> str:
        """Send multiple images to ollama for comparison."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "images": []
        }

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

    def _check_with_reference_image(self, check_image_path: str, captioner=None) -> PaperCheckResult:
        """Check paper presence by comparing to reference images using multi-image."""
        if not os.path.exists(self.present_image_path) or not os.path.exists(self.absent_image_path):
            raise Exception(f"Reference images not found: {self.present_image_path}, {self.absent_image_path}")

        # Text reading test - step-by-step reasoning for better accuracy
        prompt = f"""Please examine this image carefully using these steps:

1. First, scan the entire image for any visible text, words, or writing
2. Look specifically for the text "{PAPER_DETECTION_TEXT}" written anywhere in the image
3. Answer YES if you can see "{PAPER_DETECTION_TEXT}" text, NO if you cannot see it

Answer: YES or NO"""

        # Use single-image query with deterministic parameters for consistency
        from utils.ollama import query_ollama_raw
        import base64

        # Read and encode image
        with open(check_image_path, "rb") as f:
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
                "seed": 42  # Fixed seed for deterministic results
            }
        }

        import requests
        try:
            response_obj = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
            response_obj.raise_for_status()
            response = response_obj.json().get("response", "")
        except Exception as e:
            response = f"Error: {e}"

        # Parse YES/NO response - if it sees "EMPTY" text, that means no paper present
        response_upper = response.strip().upper()
        if 'YES' in response_upper:
            paper_present = False  # YES = sees "EMPTY" text = no paper present
            confidence = 0.8
        elif 'NO' in response_upper:
            paper_present = True   # NO = doesn't see "EMPTY" text = paper present
            confidence = 0.8
        else:
            paper_present = False
            confidence = 0.0

        return PaperCheckResult(
            paper_present=paper_present,
            confidence=confidence,
            method_used="direct_contextual_single_image",
            check_image_path=check_image_path,
            timestamp=time.time(),
            llm_response=response
        )

    def _check_direct_detection(self, check_image_path: str, captioner=None) -> PaperCheckResult:
        """Check paper presence by direct LLM analysis."""
        return self._perform_double_check_detection(check_image_path, "direct")

    def _perform_double_check_detection(self, check_image_path: str, method_name: str) -> PaperCheckResult:
        """Perform paper detection with double-check mechanism to reduce false positives."""
        # First check
        first_result = self._single_paper_check(check_image_path, "primary")

        # If first check says NO PAPER (blocking), do a second check with different phrasing
        if not first_result['paper_present']:
            time.sleep(0.5)  # Brief pause
            second_result = self._single_paper_check(check_image_path, "confirmation")

            # Only block if BOTH checks agree there's no paper
            if not second_result['paper_present']:
                # Both checks agree - no paper
                return PaperCheckResult(
                    paper_present=False,
                    confidence=min(first_result['confidence'], second_result['confidence']),
                    method_used=f"{method_name}_double_check",
                    check_image_path=check_image_path,
                    timestamp=time.time(),
                    llm_response=f"First: {first_result['response']} | Second: {second_result['response']}"
                )
            else:
                # Disagreement - allow drawing (favor allowing over blocking)
                return PaperCheckResult(
                    paper_present=True,
                    confidence=0.6,  # Lower confidence due to disagreement
                    method_used=f"{method_name}_double_check_disagreement",
                    check_image_path=check_image_path,
                    timestamp=time.time(),
                    llm_response=f"Disagreement - First: {first_result['response']} | Second: {second_result['response']}"
                )
        else:
            # First check says paper present - no need for double check
            return PaperCheckResult(
                paper_present=True,
                confidence=first_result['confidence'],
                method_used=f"{method_name}_single_check",
                check_image_path=check_image_path,
                timestamp=time.time(),
                llm_response=first_result['response']
            )

    def _single_paper_check(self, check_image_path: str, check_type: str = "primary") -> dict:
        """Perform a single paper detection check."""
        if check_type == "primary":
            prompt = f'''Please examine this image carefully using these steps:

1. First, scan the entire image for any visible text, words, or writing
2. Look specifically for the text "{PAPER_DETECTION_TEXT}" written anywhere in the image
3. Answer YES if you can see "{PAPER_DETECTION_TEXT}" text, NO if you cannot see it

Answer: YES or NO'''
        else:  # confirmation check with different phrasing
            prompt = f'''Look at this image step by step:

1. Search the image thoroughly for any written text or words
2. Check if the specific phrase "{PAPER_DETECTION_TEXT}" appears anywhere
3. Respond YES if "{PAPER_DETECTION_TEXT}" text is visible, NO if it is not visible

Answer: YES or NO'''

        # Read and encode image
        import base64
        import requests

        with open(check_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": "llava:7b-v1.6-mistral-q5_1",
            "prompt": prompt,
            "stream": False,
            "images": [img_b64],
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 10,
                "repeat_penalty": 1.0,
                "seed": None  # Remove fixed seed to allow variation between checks
            }
        }

        try:
            response_obj = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
            response_obj.raise_for_status()
            response = response_obj.json().get("response", "")
        except Exception as e:
            response = f"Error: {e}"

        # Parse response
        response_upper = response.strip().upper()
        if 'YES' in response_upper:
            paper_present = False  # YES = sees "EMPTY" text = no paper present
            confidence = 0.8
        elif 'NO' in response_upper:
            paper_present = True   # NO = doesn't see "EMPTY" text = paper present
            confidence = 0.8
        else:
            paper_present = False
            confidence = 0.0

        return {
            'paper_present': paper_present,
            'confidence': confidence,
            'response': response
        }


    def _parse_detection_response(self, response: str) -> Tuple[bool, float]:
        """Parse LLM response to extract paper presence and confidence."""
        try:
            lines = response.strip().split('\n')
            paper_present = False
            confidence = 0.0

            for line in lines:
                line = line.strip().upper()
                if line.startswith('PAPER:'):
                    paper_value = line.split(':', 1)[1].strip()
                    paper_present = 'YES' in paper_value
                elif line.startswith('CONFIDENCE:'):
                    conf_value = line.split(':', 1)[1].strip()
                    try:
                        confidence = float(conf_value)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
                    except ValueError:
                        confidence = 0.5  # Default if parsing fails

            # Apply confidence threshold
            if confidence < PAPER_DETECTION_CONFIDENCE_THRESHOLD:
                paper_present = False

            return paper_present, confidence

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"action": "parse_response_failed", "error": str(e), "response": response},
                print_message=f"[📄] Failed to parse detection response: {e}"
            )
            return False, 0.0

    def get_detection_status(self) -> Dict[str, Any]:
        """Get current paper detection system status."""
        return {
            "enabled": ENABLE_PAPER_DETECTION,
            "method": PAPER_CHECK_METHOD,
            "confidence_threshold": PAPER_DETECTION_CONFIDENCE_THRESHOLD,
            "reference_image_exists": os.path.exists(self.reference_image_path),
            "reference_image_path": self.reference_image_path,
            "paper_present_exists": os.path.exists(self.present_image_path),
            "paper_absent_exists": os.path.exists(self.absent_image_path),
            "paper_present_path": self.present_image_path,
            "paper_absent_path": self.absent_image_path,
            "recent_checks": len(self.detection_history),
            "last_check": self.detection_history[-1].__dict__ if self.detection_history else None
        }


# Global instance for easy access
paper_detector = PaperDetector()


# Convenience functions for external use
def check_paper_before_drawing(camera, servos, captioner=None) -> bool:
    """
    Convenience function to check paper presence before drawing.

    Returns:
        bool: True if paper is present and safe to draw, False otherwise
    """
    result = paper_detector.check_paper_present(camera, servos, captioner)
    return result.paper_present and result.error_message is None


def capture_paper_reference(camera, servos) -> bool:
    """
    Convenience function to capture reference image for paper detection.

    Returns:
        bool: True if reference captured successfully
    """
    return paper_detector.capture_reference_image(camera, servos)


def capture_paper_present_reference(camera, servos) -> bool:
    return paper_detector.capture_present_image(camera, servos)


def capture_paper_absent_reference(camera, servos) -> bool:
    return paper_detector.capture_absent_image(camera, servos)


def get_paper_detection_status() -> Dict[str, Any]:
    """Get current paper detection system status."""
    return paper_detector.get_detection_status()
