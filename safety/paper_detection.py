#!/usr/bin/env python3
"""
Paper Detection Safety System

Prevents drawing on bare surfaces by checking for paper presence before execution.
Uses ArUco marker detection for reliable paper verification.
"""

import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config.config import (
    ENABLE_PAPER_DETECTION,
    PAPER_DETECTION_GAZE_PAN,
    PAPER_DETECTION_GAZE_TILT,
)
from event_logging.event_logger import log_json_entry, LogType


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
    """ArUco marker-based paper detection for drawing safety."""

    def check_paper_present(self, camera, servos, captioner=None) -> PaperCheckResult:
        """Check if paper is present in the drawing area."""
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
                {"action": "paper_check_start", "method": "aruco"},
                print_message="[📄] Checking for paper using aruco method..."
            )

            result = self._check_aruco_detection_continuous(camera)
            result.timestamp = start_time

            log_json_entry(
                LogType.DECISION,
                {
                    "action": "paper_check_complete",
                    "paper_present": result.paper_present,
                    "confidence": result.confidence,
                    "method": result.method_used,
                    "duration": time.time() - start_time
                },
                print_message=f"[📄] {'✓' if result.paper_present else '✗'} Paper detection: {result.confidence:.2f} confidence"
            )

            return result

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {
                    "action": "paper_check_failed",
                    "error": str(e),
                    "component": "paper_detection",
                    "default_behavior": "allow_drawing"
                },
                print_message=f"[📄] ⚠️ Paper detection failed: {e} → Defaulting to ALLOW drawing"
            )
            return PaperCheckResult(
                paper_present=True,
                confidence=0.0,
                method_used="aruco",
                check_image_path="",
                timestamp=start_time,
                llm_response="",
                error_message=str(e)
            )

    def _check_aruco_detection_continuous(self, camera) -> PaperCheckResult:
        """
        Check paper presence using organic searching movement with ArUco detection.

        Instead of holding at a fixed position, gaze moves organically around the
        paper detection area for ~6 seconds. If the ArUco marker is detected at ANY
        point during the search, it means no paper is present (marker visible = no paper).
        """
        print(f"[📄] Starting organic paper search (~6 seconds)...")

        try:
            from safety.aruco_detector import get_aruco_detector
            from vision.gaze import set_paper_search_mode, update_paper_search_target

            detector = get_aruco_detector()

            # Search configuration - centered around paper detection position
            search_center_pan = PAPER_DETECTION_GAZE_PAN
            search_center_tilt = PAPER_DETECTION_GAZE_TILT
            search_range_pan = 20.0  # ±20° pan range
            search_range_tilt = 8.0   # ±8° tilt range — tighter, biased toward bottom in gaze.py
            search_duration = 12.0  # Total search time in seconds
            check_interval = 0.1  # Check ArUco every 100ms

            # Activate organic search mode
            set_paper_search_mode(
                active=True,
                center_pan=search_center_pan,
                center_tilt=search_center_tilt,
                range_pan=search_range_pan,
                range_tilt=search_range_tilt
            )

            # Give gaze time to reach search area before starting detection
            time.sleep(1.5)
            # Clear stale detections — rolling window is 2s, so anything from before gaze moved is gone
            detector.reset_detection_state()

            search_start = time.time()
            marker_ever_detected = False
            detection_count = 0
            total_checks = 0

            # Search loop - continuously move and check for marker
            while time.time() - search_start < search_duration:
                # Update search target (creates organic movement)
                update_paper_search_target()

                # Check ArUco detector status
                status = detector.get_status()
                total_checks += 1

                if status["marker_visible"]:
                    marker_ever_detected = True
                    detection_count += 1
                    elapsed = time.time() - search_start
                    print(f"[📄] 🎯 Marker DETECTED at {elapsed:.1f}s (IDs={status['detected_ids']}) → No paper!")
                    # Early exit - marker detected means no paper
                    break

                time.sleep(check_interval)

            # Ensure search mode is deactivated
            set_paper_search_mode(active=False)

            elapsed_total = time.time() - search_start

            if marker_ever_detected:
                print(f"[📄] ✓ Paper marker VISIBLE during search → No paper present → BLOCKING draw")
                return PaperCheckResult(
                    paper_present=False,
                    confidence=1.0,
                    method_used="aruco_search",
                    check_image_path="",
                    timestamp=time.time(),
                    llm_response=f"ArUco marker detected during {elapsed_total:.1f}s organic search - no paper covering surface"
                )
            else:
                print(f"[📄] ✗ Paper marker NOT VISIBLE during {elapsed_total:.1f}s search → Paper present → ALLOWING draw")
                return PaperCheckResult(
                    paper_present=True,
                    confidence=1.0,
                    method_used="aruco_search",
                    check_image_path="",
                    timestamp=time.time(),
                    llm_response=f"ArUco marker not detected during {elapsed_total:.1f}s search ({total_checks} checks) - paper present"
                )

        except Exception as e:
            # Ensure search mode is deactivated on error
            try:
                from vision.gaze import set_paper_search_mode
                set_paper_search_mode(active=False)
            except:
                pass
            print(f"[📄] ⚠️ Paper search failed: {e} → Defaulting to ALLOW drawing")
            return PaperCheckResult(
                paper_present=True,
                confidence=0.0,
                method_used="aruco_search",
                check_image_path="",
                timestamp=time.time(),
                llm_response="",
                error_message=f"Paper search error: {str(e)} (defaulting to allow drawing)"
            )

    def get_detection_status(self) -> Dict[str, Any]:
        """Get current paper detection system status."""
        return {
            "enabled": ENABLE_PAPER_DETECTION,
            "method": "aruco",
        }


# Global instance for easy access
paper_detector = PaperDetector()


def check_paper_before_drawing(camera, servos, captioner=None) -> bool:
    """
    Convenience function to check paper presence before drawing.

    Returns:
        bool: True if paper is present and safe to draw, False otherwise
    """
    result = paper_detector.check_paper_present(camera, servos, captioner)
    if result.error_message is not None:
        print(f"[📄] Paper check had error - defaulting to ALLOW: {result.error_message}")
        return True
    return result.paper_present


def get_paper_detection_status() -> Dict[str, Any]:
    """Get current paper detection system status."""
    return paper_detector.get_detection_status()
