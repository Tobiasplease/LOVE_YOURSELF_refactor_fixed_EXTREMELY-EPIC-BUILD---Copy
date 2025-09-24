import glob
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from bcnc import raster_to_centerline_svg
from config.config import CENTER_LINE_SVG, COMFY_OUTPUT_FOLDER, EXECUTE_GRBL_GCODE, MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from grbl import svg_to_grbl
from grbl.idle_movement_manager import pause_for_drawing, resume_after_drawing
from utils.state_manager import state_manager
import numpy as np
import cv2


class ImageMonitor:
    """Monitor a folder for new images and log them when they appear."""

    def __init__(self, monitor_folder=None, log_folder=None, check_interval=1.0, on_image_complete: Optional[Callable[[str], None]] = None, camera=None, servos=None, captioner=None):
        self.monitor_folder = monitor_folder or COMFY_OUTPUT_FOLDER
        self.log_folder = log_folder or MOOD_SNAPSHOT_FOLDER
        self.check_interval = check_interval
        self.image_extensions = {".png"}
        self.monitored_images = set()
        self.running = False
        self.thread = None
        self.on_image_complete = on_image_complete
        self.session_start_time = time.time()  # Track when this session started
        self.camera = camera
        self.servos = servos
        self.captioner = captioner

    def set_dependencies(self, camera, servos, captioner):
        """Set camera, servos, and captioner dependencies after initialization."""
        self.camera = camera
        self.servos = servos
        self.captioner = captioner

    def start(self):
        """Start the image monitoring thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

        log_json_entry(
            LogType.INFO,
            {"message": f"Image monitor started for folder: {self.monitor_folder}"},
            print_message=f"[👁️] Image monitor started: {self.monitor_folder}",
        )

    def stop(self):
        """Stop the image monitoring thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _initialize_existing_images(self):
        """Initialize the set with existing images in the folder."""
        Path(self.monitor_folder).mkdir(parents=True, exist_ok=True)

        for ext in self.image_extensions:
            existing_images = glob.glob(os.path.join(self.monitor_folder, f"*{ext}"))
            existing_images.extend(glob.glob(os.path.join(self.monitor_folder, f"*{ext.upper()}")))
            self.monitored_images.update(existing_images)

        log_json_entry(
            LogType.INFO,
            {"message": f"Image monitor initialized with {len(self.monitored_images)} existing images"},
            print_message=f"[📁] Found {len(self.monitored_images)} existing images",
        )

    def _get_current_images(self):
        """Get all current image files in the monitored folder."""
        current_images = set()

        for ext in self.image_extensions:
            pattern = os.path.join(self.monitor_folder, f"*{ext}")
            current_images.update(glob.glob(pattern))
            pattern_upper = os.path.join(self.monitor_folder, f"*{ext.upper()}")
            current_images.update(glob.glob(pattern_upper))

        return current_images

    def _find_latest_svg_in_folder(self, folder_path):
        """Find the most recently created SVG file in the given folder."""
        svg_pattern = os.path.join(folder_path, "*.svg")
        svg_files = glob.glob(svg_pattern)

        if not svg_files:
            return None

        # Sort by modification time, newest first
        svg_files.sort(key=os.path.getmtime, reverse=True)
        return svg_files[0]

    def _process_png_to_gcode(self, png_path):
        """Process a PNG file to G-code based on CENTER_LINE_SVG config."""

        # Pause idle movements to free the serial port
        pause_for_drawing()

        try:
            base_name = os.path.splitext(os.path.basename(png_path))[0]
            output_folder = os.path.dirname(png_path)

            # End any lingering 'generation' phase now that we have the PNG
            try:
                if getattr(state_manager, 'is_generating_drawing', False):
                    state_manager.finish_drawing_generation()
                    state_manager.clear_expected_output_prefix()
            except Exception:
                pass

            # --- Minimal Paper Presence Check (non-blocking, safe fallbacks) ---
            try:
                from config.config import ENABLE_PAPER_DETECTION, ALLOW_PAPER_DETECTION_OVERRIDE
            except Exception:
                ENABLE_PAPER_DETECTION = True
                ALLOW_PAPER_DETECTION_OVERRIDE = True

            def _quick_paper_check() -> bool:
                """Fast, simple check using a single camera frame.
                - Returns True to proceed with drawing; False to block (only if override disabled).
                - Never raises; logs decisions.
                """
                if not ENABLE_PAPER_DETECTION:
                    log_json_entry(
                        LogType.DECISION,
                        {"action": "paper_check", "enabled": False, "decision": "proceed"},
                        print_message="[📄] Paper detection disabled — proceeding",
                    )
                    return True

                # Proactively position gaze to look down for the check (if servos available)
                try:
                    from config.config import USE_SERVO, PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT
                    if USE_SERVO:
                        try:
                            from vision.gaze import set_drawing_mode
                            log_json_entry(
                                LogType.DEBUG,
                                {"action": "paper_check_gaze", "step": "engage", "pan": PAPER_DETECTION_GAZE_PAN, "tilt": PAPER_DETECTION_GAZE_TILT},
                                print_message="[👁️] Positioning gaze for paper check...",
                            )
                            set_drawing_mode(active=True, drawing_pan=PAPER_DETECTION_GAZE_PAN, drawing_tilt=PAPER_DETECTION_GAZE_TILT)
                            time.sleep(1.2)  # brief settle; shorter than drawing settle
                        except Exception as e:
                            log_json_entry(
                                LogType.WARNING,
                                {"action": "paper_check_gaze", "step": "fallback", "error": str(e)},
                                print_message=f"[👁️] Could not use gaze system ({e})",
                            )
                            # Optional minimal fallback via direct servos if provided
                            try:
                                if self.servos is not None:
                                    self.servos.set_pan(PAPER_DETECTION_GAZE_PAN)
                                    time.sleep(0.2)
                                    self.servos.set_tilt(PAPER_DETECTION_GAZE_TILT)
                                    time.sleep(0.6)
                            except Exception:
                                pass
                except Exception:
                    # Missing config or gaze system — continue without repositioning
                    pass

                # No camera available — proceed by override
                if self.camera is None or not hasattr(self.camera, "read_frame"):
                    log_json_entry(
                        LogType.WARNING,
                        {"action": "paper_check", "camera": "unavailable", "decision": "proceed"},
                        print_message="[📄] No camera — proceeding (safety override)",
                    )
                    return True

                try:
                    frame = self.camera.read_frame()
                except Exception as e:
                    frame = None
                    log_json_entry(
                        LogType.ERROR,
                        {"action": "paper_check", "error": str(e), "decision": "proceed"},
                        print_message=f"[📄] Camera read failed ({e}) — proceeding",
                    )

                if frame is None or not isinstance(frame, np.ndarray):
                    log_json_entry(
                        LogType.WARNING,
                        {"action": "paper_check", "frame": "none", "decision": "proceed"},
                        print_message="[📄] No camera frame — proceeding",
                    )
                    return True

                # Use centralized LLM-based paper detection (same as reliable hotkey test)
                try:
                    from safety.paper_detection import check_paper_before_drawing
                    # Create minimal camera/servos objects for the centralized function
                    camera_obj = type('Camera', (), {'read_frame': lambda: frame})()
                    servos_obj = self.servos if hasattr(self, 'servos') and self.servos is not None else None

                    paper_present = check_paper_before_drawing(camera_obj, servos_obj, None)
                    reason = "centralized_llm_detection"

                    log_json_entry(
                        LogType.DEBUG,
                        {"action": "paper_check_delegated", "result": paper_present, "method": "centralized"},
                        print_message=f"[📄] Delegated to centralized detection: {'YES' if paper_present else 'NO'}"
                    )
                except Exception as e:
                    log_json_entry(
                        LogType.ERROR,
                        {"action": "paper_check_fallback", "error": str(e)},
                        print_message=f"[📄] Centralized detection failed ({e}) — defaulting to PROCEED"
                    )
                    paper_present = True
                    reason = f"centralized_error_fallback({str(e)})"

                log_json_entry(
                    LogType.DECISION,
                    {
                        "action": "paper_check",
                        "enabled": True,
                        "method": "centralized_llm",
                        "paper_present": paper_present,
                        "reason": reason,
                        "override": ALLOW_PAPER_DETECTION_OVERRIDE,
                    },
                    print_message=f"[📄] Paper check → {'YES' if paper_present else 'NO'} ({reason})",
                )

                # Release gaze lock after check
                try:
                    from vision.gaze import set_drawing_mode
                    set_drawing_mode(active=False)
                except Exception:
                    pass

                if paper_present:
                    return True

                # If not present: proceed only if override is allowed
                if ALLOW_PAPER_DETECTION_OVERRIDE:
                    log_json_entry(
                        LogType.WARNING,
                        {"action": "paper_check", "result": "no_paper", "decision": "proceed_override"},
                        print_message="[📄] No paper detected — proceeding (override)",
                    )
                    return True

                log_json_entry(
                    LogType.DECISION,
                    {"action": "paper_check", "result": "no_paper", "decision": "block"},
                    print_message="[⛔] No paper detected — blocking draw",
                )
                return False

            # Paper check moved to after GRBL homing (in grbl_utils.process_svg_to_grbl)
            # to ensure the arm is not blocking the view. Pre-check intentionally disabled here.

            if CENTER_LINE_SVG:
                # Convert PNG to centerline SVG, then to G-code
                centerline_svg_path = os.path.join(output_folder, f"{base_name}_center_lined.svg")
                gcode_path = os.path.join(output_folder, f"{base_name}_center_lined.gcode")

                log_json_entry(
                    LogType.INFO,
                    {"message": f"Converting PNG to centerline SVG: {png_path}"},
                    print_message=f"[🔄] Converting PNG to centerline SVG: {base_name}",
                )

                # Run svg_centerliner
                raster_to_centerline_svg(
                    input_path=png_path,
                    output_path=centerline_svg_path,
                    threshold_value=0,  # Testa 160–200 beroende på bild
                    blur_kernel=(1, 1),  # (1,1) = ingen blur, (3,3) = mild
                    do_dilate=False,  # Sätt till False om det tar med för mycket
                    dilation_iterations=1,  # Testa 0–2
                    scale=1.0,  # SVG-skalning
                )

                # Convert SVG to G-code (CNC execution tracking will start when GRBL actually begins)
                result_path = svg_to_grbl(svg_input=centerline_svg_path, output_gcode=gcode_path, execute_grbl=EXECUTE_GRBL_GCODE)

                if result_path:
                    log_json_entry(
                        LogType.INFO,
                        {"message": f"G-code generated: {gcode_path}"},
                        print_message=f"[🔧] G-code generated: {os.path.basename(gcode_path)}",
                    )

                    # Trigger self-critique AFTER physical drawing completes
                    if self.on_image_complete:
                        self.on_image_complete(png_path)
                    
                    # Note: CNC execution state is cleared by grbl_utils.py after physical drawing completes
                else:
                    # Determine if this was a deliberate skip due to no paper
                    no_paper_skip = False
                    try:
                        now_ts = time.time()
                        if getattr(state_manager, 'last_paper_check_ts', 0) > 0 and not getattr(state_manager, 'paper_present', True):
                            if now_ts - state_manager.last_paper_check_ts < 5.0:
                                no_paper_skip = True
                    except Exception:
                        no_paper_skip = False

                    # Clear execution state to allow recovery/next attempt
                    state_manager.finish_cnc_execution()

                    if no_paper_skip:
                        # Reset cooldown so next drawing attempt isn't throttled
                        try:
                            if self.captioner and hasattr(self.captioner, 'drawing'):
                                # Start cooldown now to encompass prompt+generation+execution attempt
                                self.captioner.drawing.last_drawing_time = time.time()
                        except Exception:
                            pass
                        # If a generation cycle was active, end it now to avoid long timeouts blocking captions
                        try:
                            if getattr(state_manager, 'is_generating_drawing', False):
                                state_manager.finish_drawing_generation()
                                state_manager.clear_expected_output_prefix()
                        except Exception:
                            pass

                        # Record context for LLM/memory
                        try:
                            if self.captioner and hasattr(self.captioner, 'observe'):
                                reason = getattr(state_manager, 'last_paper_check_reason', 'no_paper')
                                self.captioner.observe(f"Skipped drawing: no paper detected ({reason}).", getattr(self.captioner, 'current_mood', 0.5), png_path, memory_type="environment")
                        except Exception:
                            pass

                        log_json_entry(
                            LogType.DECISION,
                            {"decision": "skip_drawing_no_paper", "reason": getattr(state_manager, 'last_paper_check_reason', '')},
                            print_message="[🛑] Skipped drawing: no paper detected",
                        )
                    else:
                        log_json_entry(
                            LogType.ERROR,
                            {"message": "CNC execution failed; clearing execution state for retry", "gcode_path": gcode_path},
                            print_message="[❌] CNC execution failed; clearing state for retry",
                        )

            else:
                # Find latest SVG in output folder and convert to G-code
                latest_svg = self._find_latest_svg_in_folder(output_folder)

                if latest_svg:
                    svg_base_name = os.path.splitext(os.path.basename(latest_svg))[0]
                    gcode_path = os.path.join(output_folder, f"{svg_base_name}.gcode")

                    log_json_entry(
                        LogType.INFO,
                        {"message": f"Converting latest SVG to G-code: {latest_svg}"},
                        print_message=f"[🔄] Converting latest SVG to G-code: {os.path.basename(latest_svg)}",
                    )

                    # Convert SVG to G-code and execute (CNC execution tracking will start when GRBL actually begins)
                    result_path = svg_to_grbl(svg_input=latest_svg, output_gcode=gcode_path, execute_grbl=EXECUTE_GRBL_GCODE)

                    if result_path:
                        log_json_entry(
                            LogType.INFO,
                            {"message": f"G-code generated: {gcode_path}"},
                            print_message=f"[🔧] G-code generated: {os.path.basename(gcode_path)}",
                        )

                        # Trigger self-critique AFTER physical drawing completes
                        if self.on_image_complete:
                            self.on_image_complete(png_path)
                        
                        # Note: CNC execution state is cleared by grbl_utils.py after physical drawing completes
                    else:
                        # Determine if skipped due to no paper and log accordingly
                        no_paper_skip = False
                        try:
                            now_ts = time.time()
                            if getattr(state_manager, 'last_paper_check_ts', 0) > 0 and not getattr(state_manager, 'paper_present', True):
                                if now_ts - state_manager.last_paper_check_ts < 5.0:
                                    no_paper_skip = True
                        except Exception:
                            no_paper_skip = False

                        state_manager.finish_cnc_execution()
                        # If generation was active, end it to prevent captioner waiting for timeout
                        try:
                            if getattr(state_manager, 'is_generating_drawing', False):
                                state_manager.finish_drawing_generation()
                                state_manager.clear_expected_output_prefix()
                        except Exception:
                            pass
                        if no_paper_skip:
                            try:
                                if self.captioner and hasattr(self.captioner, 'drawing'):
                                    # Start cooldown now since full cycle was attempted but skipped
                                    self.captioner.drawing.last_drawing_time = time.time()
                            except Exception:
                                pass
                            try:
                                if self.captioner and hasattr(self.captioner, 'observe'):
                                    reason = getattr(state_manager, 'last_paper_check_reason', 'no_paper')
                                    self.captioner.observe(f"Skipped drawing: no paper detected ({reason}).", getattr(self.captioner, 'current_mood', 0.5), png_path, memory_type="environment")
                            except Exception:
                                pass
                            log_json_entry(
                                LogType.DECISION,
                                {"decision": "skip_drawing_no_paper", "reason": getattr(state_manager, 'last_paper_check_reason', '')},
                                print_message="[🛑] Skipped drawing: no paper detected",
                            )
                        else:
                            log_json_entry(
                                LogType.ERROR,
                                {"message": "CNC execution failed; clearing execution state for retry", "gcode_path": gcode_path},
                                print_message="[❌] CNC execution failed; clearing state for retry",
                            )
                else:
                    log_json_entry(
                        LogType.ERROR,
                        {"message": f"No SVG files found in output folder: {output_folder}"},
                        print_message=f"[❌] No SVG files found in output folder",
                    )

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"error": f"PNG to G-code conversion failed: {str(e)}"},
                print_message=f"[❌] PNG to G-code conversion failed: {str(e)}",
            )
        finally:
            # Resume idle movements after execution attempt completes (success or failure)
            resume_after_drawing()

    def _log_new_image(self, image_path) -> bool:
        """Log a newly detected image and return True if processed/accepted.

        Returns False when the drawing pipeline is busy so the monitor can retry later.
        """
        filename = os.path.basename(image_path)
        file_size = os.path.getsize(image_path)

        # Skip only if CNC is actively executing (avoid stacking physical draws)
        try:
            if getattr(state_manager, "is_executing_cnc", False):
                log_json_entry(
                    LogType.DECISION,
                    {"message": "Skipping new image while CNC executing", "filename": filename, "is_executing_cnc": True},
                    print_message=f"[⏳] Skipping {filename}: CNC busy",
                )
                return False
        except Exception:
            pass

        # Only process images created after this session started
        try:
            file_creation_time = os.path.getmtime(image_path)
            if file_creation_time < self.session_start_time:
                log_json_entry(
                    LogType.INFO,
                    {"message": f"Skipping old image from previous session: {filename}"},
                    print_message=f"[⏭️] Skipping old image: {filename}",
                )
                return False
        except OSError:
            # If we can't get file time, assume it's old and skip
            log_json_entry(
                LogType.WARNING,
                {"message": f"Could not get creation time for {filename}, skipping"},
                print_message=f"[⚠️] Cannot check age of {filename}, skipping",
            )
            return False

        log_json_entry(
            LogType.NEW_DRAWING,
            {"event": "new_image_detected", "filename": filename, "image_path": image_path, "file_size": file_size, "timestamp": time.time()},
            print_message=f"[🖼️] New drawing: {filename} ({file_size} bytes)",
        )

        # Process PNG to G-code and execute it (if configured)
        if image_path.lower().endswith(".png"):
            self._process_png_to_gcode(image_path)
        if self.on_image_complete:
            self.on_image_complete(image_path)
        # Clear the expected prefix after accepting one matching image
        try:
            state_manager.clear_expected_output_prefix()
        except Exception:
            pass
        return True

    def _monitor_loop(self):
        """Main monitoring loop that runs in the background thread."""
        self._initialize_existing_images()

        while self.running:
            try:
                current_images = self._get_current_images()
                new_images = current_images - self.monitored_images

                for new_image in new_images:
                    if self._log_new_image(new_image):
                        self.monitored_images.add(new_image)

                time.sleep(self.check_interval)

            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {"error": f"Image monitor error: {str(e)}"},
                    print_message=f"[❌] Image monitor error: {str(e)}",
                )
                time.sleep(5.0)  # Wait longer on error
