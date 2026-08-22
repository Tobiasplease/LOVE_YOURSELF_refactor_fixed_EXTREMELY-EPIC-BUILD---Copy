#!/usr/bin/env python3
"""
Paper Detection Safety System

Prevents drawing on bare or already-drawn-on surfaces by checking the table
before execution. Two methods, selected by config.PAPER_CHECK_METHOD:

  "vlm"   (default since Aug 20) — the loaded model looks at the table and
          answers structurally; only a BLANK sheet allows drawing. Bare
          surface, clutter, a drawn-on sheet, or any model failure all
          block (fails CLOSED).
  "aruco" — legacy marker search: marker visible = no paper. Occlusion by
          anything reads as paper, and errors fail OPEN (allow).
"""

import os
import re
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from config import config as _cfg
from config.config import (
    ENABLE_PAPER_DETECTION,
    PAPER_DETECTION_GAZE_PAN,
    PAPER_DETECTION_GAZE_TILT,
)
from event_logging.event_logger import log_json_entry, LogType

_PAPER_RE = re.compile(r"paper:\s*(yes|no)", re.I)
_MARKS_RE = re.compile(r"marks:\s*(yes|no|n/?a)", re.I)


def _parse_paper_state(text: str) -> str:
    """Structural parse of the model's answer lines.

    Returns one of: blank_paper / drawn_paper / no_paper / unclear.
    Only blank_paper allows drawing.
    """
    p = _PAPER_RE.search(text or "")
    if not p:
        return "unclear"
    if p.group(1).lower() == "no":
        return "no_paper"
    m = _MARKS_RE.search(text)
    if not m:
        return "unclear"
    return "drawn_paper" if m.group(1).lower() == "yes" else "blank_paper"


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
    # blank_paper / drawn_paper / no_paper / unclear — feeds the monologue's
    # paper-state line via state_manager (aruco can only infer blank/no).
    paper_state: str = ""


class PaperDetector:
    """Paper detection for drawing safety (VLM or ArUco, per config)."""

    def check_paper_present(self, camera, servos, captioner=None) -> PaperCheckResult:
        """Check if paper is present in the drawing area."""
        start_time = time.time()
        method = str(getattr(_cfg, "PAPER_CHECK_METHOD", "aruco")).lower()

        try:
            if not ENABLE_PAPER_DETECTION:
                return PaperCheckResult(
                    paper_present=True,
                    confidence=1.0,
                    method_used="disabled",
                    check_image_path="",
                    timestamp=start_time,
                    llm_response="Paper detection disabled in config",
                )

            log_json_entry(
                LogType.DEBUG, {"action": "paper_check_start", "method": method}, print_message=f"[📄] Checking for paper using {method} method..."
            )

            # Kinetic bus: both arms play their recorded get-clear move so
            # they don't occlude the marker; released again in the finally.
            _clear_wait = 0.0
            try:
                from utils import hooks as _kin_hooks

                if _kin_hooks.on_paper_check_start:
                    _clear_wait = min(float(_kin_hooks.on_paper_check_start() or 0.0), 20.0)
            except Exception:
                pass
            if _clear_wait > 0:
                print(f"[📄] Arms clearing the view ({_clear_wait:.1f}s)…")
                time.sleep(_clear_wait)

            try:
                if method == "vlm":
                    result = self._check_vlm(camera)
                else:
                    result = self._check_aruco_detection_continuous(camera)
            finally:
                try:
                    from utils import hooks as _kin_hooks

                    if _kin_hooks.on_paper_check_done:
                        _kin_hooks.on_paper_check_done()
                except Exception:
                    pass
            result.timestamp = start_time

            log_json_entry(
                LogType.DECISION,
                {
                    "action": "paper_check_complete",
                    "paper_present": result.paper_present,
                    "confidence": result.confidence,
                    "method": result.method_used,
                    "duration": time.time() - start_time,
                },
                print_message=f"[📄] {'✓' if result.paper_present else '✗'} Paper detection: {result.confidence:.2f} confidence",
            )

            return result

        except Exception as e:
            # VLM path fails CLOSED (no draw); legacy aruco path keeps its fail-open default.
            fail_open = method != "vlm"
            log_json_entry(
                LogType.ERROR,
                {
                    "action": "paper_check_failed",
                    "error": str(e),
                    "component": "paper_detection",
                    "method": method,
                    "default_behavior": "allow_drawing" if fail_open else "block_drawing",
                },
                print_message=f"[📄] ⚠️ Paper detection failed: {e} → {'Defaulting to ALLOW drawing' if fail_open else 'Failing CLOSED — blocking draw'}",
            )
            return PaperCheckResult(
                paper_present=fail_open,
                confidence=0.0,
                method_used=method,
                check_image_path="",
                timestamp=start_time,
                llm_response="",
                error_message=str(e),
            )

    def _grab_frame(self, camera):
        """Freshest table view: the aruco thread's shared frame (fed every loop
        by machine.py) first, then whatever read method the camera object has."""
        try:
            from safety.aruco_detector import get_aruco_detector

            det = get_aruco_detector()
            with det.lock:
                if det.shared_frame is not None:
                    return det.shared_frame.copy()
        except Exception:
            pass
        for attr in ("read", "read_frame"):
            fn = getattr(camera, attr, None)
            if fn is None:
                continue
            try:
                out = fn()
            except Exception:
                continue
            if isinstance(out, tuple):
                ok, frame = out
                return frame if ok else None
            return out
        return None

    def _check_vlm(self, camera) -> PaperCheckResult:
        """Ask the loaded model whether a blank sheet is on the table.

        Captures PAPER_VLM_FRAMES frames at the paper-check angle and queries
        the model per frame. Every frame must read as blank_paper to allow;
        drawn_paper / no_paper / unclear / any failure blocks (fail closed).
        """
        import cv2

        from captioner.prompts import PAPER_CHECK_PROMPT
        from utils.inference import is_failed_response, query_model

        n_frames = max(1, int(getattr(_cfg, "PAPER_VLM_FRAMES", 2)))

        # Park gaze on the table via the same search plumbing as the aruco
        # sweep, just with a tight range — no-op when the gaze loop isn't
        # running (standalone tests position servos themselves).
        gaze_parked = False
        try:
            from vision.gaze import set_paper_search_mode

            set_paper_search_mode(
                active=True, center_pan=PAPER_DETECTION_GAZE_PAN, center_tilt=PAPER_DETECTION_GAZE_TILT, range_pan=4.0, range_tilt=2.0
            )
            gaze_parked = True
            # The live gaze EASES toward the target — 1.5s (bench-calibrated,
            # direct servo writes) shot frame 1 mid-travel at the workbench.
            time.sleep(float(getattr(_cfg, "PAPER_VLM_SETTLE_S", 4.0)))
        except Exception:
            pass

        img_dir = os.path.join(_cfg.MOOD_SNAPSHOT_FOLDER, "paper_checks")
        os.makedirs(img_dir, exist_ok=True)

        try:
            states = []
            responses = []
            last_image_path = ""
            for i in range(n_frames):
                if i > 0:
                    time.sleep(0.8)
                frame = self._grab_frame(camera)
                if frame is None:
                    states.append("no_frame")
                    responses.append("(no camera frame)")
                    continue

                stamp = time.strftime("%H%M%S")
                last_image_path = os.path.join(img_dir, f"paper_check_{stamp}_{i}.jpg")
                cv2.imwrite(last_image_path, frame)

                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    states.append("no_frame")
                    responses.append("(jpeg encode failed)")
                    continue

                response = query_model(
                    PAPER_CHECK_PROMPT,
                    image=jpg.tobytes(),
                    timeout=90,
                    options={"temperature": 0.1, "num_predict": 120},
                    prompt_type="paper_check",
                    skip_generation_wait=True,
                )
                if is_failed_response(response):
                    states.append("query_failed")
                    responses.append(response)
                    continue
                states.append(_parse_paper_state(response))
                responses.append(response.strip())

            allow = bool(states) and all(s == "blank_paper" for s in states)
            summary = "+".join(states) if states else "no_frames"
            # Consensus must never claim more than every frame agrees on:
            # a mixed no_paper+blank_paper vote (first live fire, 17:14 —
            # frame 1 shot mid-gaze-travel) spoke "no paper on the desk"
            # with a sheet in plain view. Mixed → unclear: still blocks,
            # but the monologue stays silent instead of asserting a false
            # absence. Any drawn sighting wins — one frame seeing marks is
            # positive evidence even if the other missed them.
            if allow:
                consensus = "blank_paper"
            elif "drawn_paper" in states:
                consensus = "drawn_paper"
            elif states and all(s == "no_paper" for s in states):
                consensus = "no_paper"
            else:
                consensus = "unclear"
            print(f"[📄] VLM paper check: {summary} → {'ALLOW (blank sheet)' if allow else 'BLOCK'}")
            return PaperCheckResult(
                paper_present=allow,
                confidence=1.0 if allow or all(s in ("no_paper", "drawn_paper") for s in states) else 0.5,
                method_used=f"vlm:{summary}",
                check_image_path=last_image_path,
                timestamp=time.time(),
                llm_response=" | ".join(responses),
                paper_state=consensus,
            )
        finally:
            if gaze_parked:
                try:
                    from vision.gaze import set_paper_search_mode

                    set_paper_search_mode(active=False)
                except Exception:
                    pass

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
            search_range_tilt = 8.0  # ±8° tilt range — tighter, biased toward bottom in gaze.py
            search_duration = 12.0  # Total search time in seconds
            check_interval = 0.1  # Check ArUco every 100ms

            # Activate organic search mode
            set_paper_search_mode(
                active=True, center_pan=search_center_pan, center_tilt=search_center_tilt, range_pan=search_range_pan, range_tilt=search_range_tilt
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
                    llm_response=f"ArUco marker detected during {elapsed_total:.1f}s organic search - no paper covering surface",
                )
            else:
                print(f"[📄] ✗ Paper marker NOT VISIBLE during {elapsed_total:.1f}s search → Paper present → ALLOWING draw")
                return PaperCheckResult(
                    paper_present=True,
                    confidence=1.0,
                    method_used="aruco_search",
                    check_image_path="",
                    timestamp=time.time(),
                    llm_response=f"ArUco marker not detected during {elapsed_total:.1f}s search ({total_checks} checks) - paper present",
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
                error_message=f"Paper search error: {str(e)} (defaulting to allow drawing)",
            )

    def get_detection_status(self) -> Dict[str, Any]:
        """Get current paper detection system status."""
        return {
            "enabled": ENABLE_PAPER_DETECTION,
            "method": str(getattr(_cfg, "PAPER_CHECK_METHOD", "aruco")).lower(),
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

    # Publish the verdict so the monologue can speak it (TTL-gated there).
    # Errors publish "unclear" — a server hiccup must not become "no paper".
    try:
        from utils.state_manager import state_manager as _sm

        if result.error_message is not None:
            state = "unclear"
        else:
            state = result.paper_state or ("blank_paper" if result.paper_present else "no_paper")
        _sm.paper_state = state
        _sm.paper_present = state == "blank_paper"
        _sm.last_paper_check_ts = time.time()
        _sm.last_paper_check_reason = result.method_used
    except Exception:
        pass

    if result.error_message is not None:
        if result.method_used.startswith("vlm"):
            print(f"[📄] Paper check had error - failing CLOSED (no draw): {result.error_message}")
            return False
        print(f"[📄] Paper check had error - defaulting to ALLOW: {result.error_message}")
        return True
    return result.paper_present


def get_paper_detection_status() -> Dict[str, Any]:
    """Get current paper detection system status."""
    return paper_detector.get_detection_status()
