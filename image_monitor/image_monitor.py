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

            # DON'T end generation phase yet - keep blocking flag active until GRBL starts
            # to prevent gap where both is_generating_drawing and is_executing_cnc are False
            # We'll clear it right before svg_to_grbl starts CNC execution

            # Paper check happens after GRBL homing (grbl_utils.process_svg_to_grbl)
            # to ensure the arm is not blocking the view. The old pre-check that
            # lived here (_quick_paper_check, disabled and unreachable since the
            # move) was deleted Aug 30 2026 — it only wrapped the same
            # check_paper_before_drawing the post-homing path calls.

            if CENTER_LINE_SVG:
                # Convert PNG to centerline SVG, then to G-code
                centerline_svg_path = os.path.join(output_folder, f"{base_name}_center_lined.svg")
                gcode_path = os.path.join(output_folder, f"{base_name}_center_lined.gcode")

                log_json_entry(
                    LogType.INFO,
                    {"message": f"Converting PNG to centerline SVG: {png_path}"},
                    print_message=f"[🔄] Converting PNG to centerline SVG: {base_name}",
                )

                # Re-arm the generation window for the vectorize phase: the
                # 5-minute timeout from queue time was expiring mid-centerline
                # (ComfyUI ~3min + DSV fallback ~3min), force-clearing the flag
                # and letting llama-server reload the 27B WHILE DSV needed the
                # GPU (Aug 12). llama must never run alongside the vectorizer;
                # a fresh window keeps the flag held until finish below.
                try:
                    state_manager.start_drawing_generation(
                        getattr(state_manager, "current_drawing_prompt", None) or "vectorizing")
                except Exception:
                    pass

                raster_to_centerline_svg(
                    input_path=png_path,
                    output_path=centerline_svg_path,
                )

                # Generation phase is NOT released here anymore (Aug 12): the
                # flag now drops inside process_svg_to_grbl, immediately after
                # the servo g-code is written — so llama-server stays down
                # through vectorize + gcode conversion and reloads alongside
                # GRBL execution, per the artist's rule.
                try:
                    state_manager.clear_expected_output_prefix()
                except Exception:
                    pass

                # Convert SVG to G-code (svg_to_grbl will call start_cnc_execution immediately)
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

        # Provenance gate (Aug 19): only draw what THIS session queued. The
        # mtime gate below misses generations a PREVIOUS session queued that
        # finish rendering after boot (17:27 queue → quit → reboot 17:30 →
        # file lands 17:31, 20s into the new run — the DSV CUDA pipeline
        # collided with llama-server's boot allocation burst). Also closes
        # the July 21 hazard where a manually queued test gen got physically
        # drawn: no expected prefix, no pen.
        try:
            expected = state_manager.get_expected_output_prefix()
        except Exception:
            expected = None
        if not expected or not filename.startswith(expected):
            log_json_entry(
                LogType.INFO,
                {"message": f"Skipping image this session did not queue: {filename}", "expected_prefix": expected},
                print_message=f"[⏭️] Skipping unqueued image: {filename}",
            )
            # Dismiss permanently (False alone means "retry every poll" —
            # that's for the busy case; an unqueued file stays unqueued)
            self.monitored_images.add(image_path)
            return False

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
