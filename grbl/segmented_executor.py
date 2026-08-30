"""
Segmented G-code Execution
Executes G-code in segments to prevent GRBL buffer overload while preserving image quality
"""

import os
import sys
import time
import threading
from typing import List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from event_logging.event_logger import log_json_entry
    from event_logging.log_type import LogType
    from grbl.gcode_segmenter import GCodeSegmenter, GCodeSegment
    from grbl.grbl_utils import send_cmd, wait_until_idle, DEFAULT_CMD_TIMEOUT, DEFAULT_MOVE_TIMEOUT
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Fallback for testing
    def log_json_entry(log_type, data, print_message=""):
        print(print_message)
    class LogType:
        INFO = "info"
        ERROR = "error"
        GRBL = "grbl"


class SegmentedExecutor:
    """Executes G-code in segments with progress tracking, error recovery, and person detection pausing"""

    def __init__(self, max_segment_size: int = 150, enable_person_detection_pause: bool = False):
        self.segmenter = GCodeSegmenter(max_segment_size)
        self.current_segment = 0
        self.total_segments = 0
        self.execution_start_time = None

        # Person detection pause functionality
        self.enable_person_detection_pause = enable_person_detection_pause
        self.is_paused_for_person = False
        self.person_pause_start_time = None
        self.pause_resume_delay = 3.0  # Seconds to wait after person leaves before resuming
        self.max_pause_duration = 300.0  # Maximum 5 minutes pause to prevent infinite blocking
        self.total_pause_time = 0.0

    def execute_gcode_file_segmented(self, ser, gcode_file: str, move_timeout=DEFAULT_MOVE_TIMEOUT) -> bool:
        """
        Execute G-code file using segmentation for optimal GRBL performance

        Args:
            ser: Serial connection to GRBL
            gcode_file: Path to G-code file
            move_timeout: Timeout for movement commands

        Returns:
            bool: True if execution successful, False if failed
        """
        try:
            log_json_entry(
                LogType.GRBL,
                {"action": "segmented_execution_start", "file": gcode_file, "max_segment_size": self.segmenter.max_segment_size},
                print_message=f"[📋] Starting segmented execution: {os.path.basename(gcode_file)}"
            )

            # Segment the G-code file
            try:
                header_lines, segments = self.segmenter.segment_gcode_file(gcode_file)
            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {"action": "segmentation_failed", "error": str(e), "file": gcode_file},
                    print_message=f"[❌] Segmentation failed, falling back to original method: {e}"
                )
                return self._fallback_to_original_execution(ser, gcode_file, move_timeout)

            if not segments:
                log_json_entry(
                    LogType.ERROR,
                    {"action": "no_segments_found", "file": gcode_file},
                    print_message="[❌] No segments found in G-code file"
                )
                return False

            self.total_segments = len(segments)
            self.execution_start_time = time.time()

            log_json_entry(
                LogType.GRBL,
                {
                    "action": "segmentation_complete",
                    "total_segments": self.total_segments,
                    "header_lines": len(header_lines),
                    "avg_segment_size": sum(len(seg) for seg in segments) / len(segments)
                },
                print_message=f"[✅] Segmented into {self.total_segments} segments (avg {sum(len(seg) for seg in segments) / len(segments):.0f} lines each)"
            )

            # Send header commands once at the beginning
            if header_lines:
                log_json_entry(
                    LogType.GRBL,
                    {"action": "sending_header", "header_count": len(header_lines)},
                    print_message=f"[📋] Sending {len(header_lines)} header commands..."
                )
                for line in header_lines:
                    if line.strip():
                        send_cmd(ser, line.strip(), timeout=DEFAULT_CMD_TIMEOUT)

            # Execute segments with progress tracking and person detection pausing
            for segment_idx, segment in enumerate(segments):
                self.current_segment = segment_idx + 1

                # Check for person detection before executing segment
                if self.enable_person_detection_pause:
                    self._check_person_detection_pause()

                success = self._execute_segment(ser, segment, move_timeout)
                if not success:
                    log_json_entry(
                        LogType.ERROR,
                        {"action": "segment_execution_failed", "segment": self.current_segment, "total": self.total_segments},
                        print_message=f"[❌] Segment {self.current_segment}/{self.total_segments} failed"
                    )
                    return False

                # Small pause between segments to let GRBL buffer clear
                if segment_idx < len(segments) - 1:  # Don't pause after last segment
                    time.sleep(0.1)

            total_time = time.time() - self.execution_start_time
            log_json_entry(
                LogType.GRBL,
                {
                    "action": "segmented_execution_complete",
                    "total_segments": self.total_segments,
                    "total_time": total_time,
                    "avg_time_per_segment": total_time / self.total_segments
                },
                print_message=f"[✅] Segmented execution complete: {self.total_segments} segments in {total_time:.1f}s"
            )
            return True

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"action": "segmented_execution_error", "error": str(e), "file": gcode_file},
                print_message=f"[❌] Segmented execution failed: {e}"
            )
            return False

    def _execute_segment(self, ser, segment: GCodeSegment, move_timeout: float) -> bool:
        """Execute a single G-code segment with progress tracking"""

        try:
            # Progress reporting
            elapsed = time.time() - self.execution_start_time if self.execution_start_time else 0
            percent_complete = ((self.current_segment - 1) / self.total_segments) * 100

            log_json_entry(
                LogType.GRBL,
                {
                    "action": "segment_execution_start",
                    "segment": self.current_segment,
                    "total": self.total_segments,
                    "segment_lines": len(segment),
                    "progress_percent": percent_complete,
                    "elapsed_time": elapsed
                },
                print_message=f"[📋] Segment {self.current_segment}/{self.total_segments} ({percent_complete:.0f}%) - {len(segment)} lines"
            )

            # Execute all lines in this segment
            for line_idx, line in enumerate(segment.lines):
                line = line.strip()

                if not line or line.startswith(';'):
                    continue

                try:
                    # Apply warp transform if needed
                    transformed_line = self._apply_transforms(line)

                    # Determine timeout based on command type
                    timeout = move_timeout if line.startswith(('G0', 'G1', 'G00', 'G01')) else DEFAULT_CMD_TIMEOUT

                    send_cmd(ser, transformed_line, timeout=timeout)

                except Exception as line_error:
                    log_json_entry(
                        LogType.ERROR,
                        {
                            "action": "line_execution_failed",
                            "segment": self.current_segment,
                            "line_in_segment": line_idx + 1,
                            "command": line,
                            "error": str(line_error)
                        },
                        print_message=f"[❌] Line failed in segment {self.current_segment}: {line} - {line_error}"
                    )
                    return False

            # Wait for segment to complete before moving to next
            wait_until_idle(ser, max_wait=30.0)

            return True

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"action": "segment_execution_error", "segment": self.current_segment, "error": str(e)},
                print_message=f"[❌] Segment {self.current_segment} execution error: {e}"
            )
            return False

    def _apply_transforms(self, line: str) -> str:
        """Apply coordinate transforms if enabled"""
        try:
            # Check if warp transform is enabled
            from config.config import GRBL_WARP_TRANSFORM
            if GRBL_WARP_TRANSFORM and line.startswith(('G0', 'G1', 'G00', 'G01')):
                try:
                    from grbl.warp_transform import warp_transform_line
                    return warp_transform_line(line)
                except ImportError:
                    pass
        except ImportError:
            pass

        return line

    def _fallback_to_original_execution(self, ser, gcode_file: str, move_timeout: float) -> bool:
        """Fallback to original line-by-line execution if segmentation fails"""
        log_json_entry(
            LogType.GRBL,
            {"action": "fallback_execution_start", "file": gcode_file},
            print_message="[⚠️] Using fallback execution method..."
        )

        try:
            from grbl.grbl_utils import execute_gcode_file
            execute_gcode_file(ser, gcode_file, move_timeout)
            return True
        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"action": "fallback_execution_failed", "error": str(e)},
                print_message=f"[❌] Fallback execution also failed: {e}"
            )
            return False

    def _check_person_detection_pause(self):
        """Check for person detection and pause/resume execution accordingly"""
        try:
            # Use simple face detection from DetectionMemory (same as machine.py)
            from perception.detection_memory import DetectionMemory
            person_present = "person" in DetectionMemory.get_labels()

            if person_present and not self.is_paused_for_person:
                # Person detected - pause execution
                self.is_paused_for_person = True
                self.person_pause_start_time = time.time()

                log_json_entry(
                    LogType.GRBL,
                    {
                        "action": "drawing_paused_for_person",
                        "segment": self.current_segment,
                        "total": self.total_segments,
                        "person_detected": True
                    },
                    print_message=f"[👤] Drawing paused - person detected (segment {self.current_segment}/{self.total_segments})"
                )

            # Auto-resume after max pause duration to prevent infinite blocking
            elif self.is_paused_for_person and self.person_pause_start_time:
                pause_duration = time.time() - self.person_pause_start_time
                if pause_duration >= self.max_pause_duration:
                    # Force resume after maximum pause time
                    self.total_pause_time += pause_duration
                    self.is_paused_for_person = False
                    log_json_entry(
                        LogType.GRBL,
                        {"action": "drawing_auto_resumed_timeout", "pause_duration": pause_duration},
                        print_message=f"[⏰] Drawing auto-resumed after {pause_duration:.1f}s timeout"
                    )

            elif not person_present and self.is_paused_for_person:
                # Person left - check if enough time has passed before resuming
                pause_duration = time.time() - self.person_pause_start_time if self.person_pause_start_time else 0

                if pause_duration >= self.pause_resume_delay:
                    # Resume execution
                    self.total_pause_time += pause_duration
                    self.is_paused_for_person = False

                    log_json_entry(
                        LogType.GRBL,
                        {
                            "action": "drawing_resumed_after_person",
                            "segment": self.current_segment,
                            "total": self.total_segments,
                            "pause_duration": pause_duration,
                            "total_pause_time": self.total_pause_time
                        },
                        print_message=f"[✅] Drawing resumed - person left (paused {pause_duration:.1f}s, resuming at segment {self.current_segment})"
                    )

            # If paused, wait a bit before checking again (NON-BLOCKING)
            if self.is_paused_for_person:
                time.sleep(0.1)  # Brief check, but don't block indefinitely
                # Return without recursion to prevent blocking - will be checked again on next segment

        except Exception as e:
            # If person detection fails, continue execution
            log_json_entry(
                LogType.ERROR,
                {"action": "person_detection_check_failed", "error": str(e)},
                print_message=f"[⚠️] Person detection check failed: {e}"
            )

    def get_progress_info(self) -> dict:
        """Get current execution progress information including pause state"""
        if self.total_segments == 0:
            return {"progress_percent": 0, "current_segment": 0, "total_segments": 0}

        progress_percent = (self.current_segment / self.total_segments) * 100
        elapsed = time.time() - self.execution_start_time if self.execution_start_time else 0

        # Adjust elapsed time to exclude pause time
        effective_elapsed = elapsed - self.total_pause_time
        if self.is_paused_for_person and self.person_pause_start_time:
            current_pause = time.time() - self.person_pause_start_time
            effective_elapsed = elapsed - self.total_pause_time - current_pause

        return {
            "progress_percent": progress_percent,
            "current_segment": self.current_segment,
            "total_segments": self.total_segments,
            "elapsed_time": elapsed,
            "effective_elapsed_time": effective_elapsed,
            "total_pause_time": self.total_pause_time,
            "is_paused_for_person": self.is_paused_for_person,
            "estimated_remaining": (effective_elapsed / max(self.current_segment, 1)) * (self.total_segments - self.current_segment)
        }


def test_segmented_execution_dry_run(gcode_file: str):
    """Test segmented execution without actually sending commands to GRBL"""
    print(f"🧪 DRY RUN: Testing segmented execution on {os.path.basename(gcode_file)}")

    executor = SegmentedExecutor(max_segment_size=150)

    try:
        header_lines, segments = executor.segmenter.segment_gcode_file(gcode_file)

        print(f"✅ Would execute:")
        print(f"   Header commands: {len(header_lines)}")
        print(f"   Segments: {len(segments)}")
        print(f"   Average segment size: {sum(len(seg) for seg in segments) / len(segments):.0f} lines")
        print(f"   Largest segment: {max(len(seg) for seg in segments)} lines")
        print(f"   Total commands: {sum(len(seg) for seg in segments)} lines")

        # Simulate execution timing
        estimated_time = len(segments) * 2  # ~2 seconds per segment average
        print(f"   Estimated execution time: {estimated_time // 60}m {estimated_time % 60}s")

        return True

    except Exception as e:
        print(f"❌ Dry run failed: {e}")
        return False


if __name__ == "__main__":
    # Test on most recent G-code file
    import glob

    output_dir = "/home/impostor/ComfyUI/output"
    gcode_files = glob.glob(os.path.join(output_dir, "*.gcode"))

    if gcode_files:
        recent_file = max(gcode_files, key=os.path.getmtime)
        test_segmented_execution_dry_run(recent_file)
    else:
        print("No G-code files found for testing")