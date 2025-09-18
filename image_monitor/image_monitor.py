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


class ImageMonitor:
    """Monitor a folder for new images and log them when they appear."""

    def __init__(self, monitor_folder=None, log_folder=None, check_interval=1.0, on_image_complete: Optional[Callable[[str], None]] = None):
        self.monitor_folder = monitor_folder or COMFY_OUTPUT_FOLDER
        self.log_folder = log_folder or MOOD_SNAPSHOT_FOLDER
        self.check_interval = check_interval
        self.image_extensions = {".png"}
        self.monitored_images = set()
        self.running = False
        self.thread = None
        self.on_image_complete = on_image_complete
        self.session_start_time = time.time()  # Track when this session started

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
        try:
            base_name = os.path.splitext(os.path.basename(png_path))[0]
            output_folder = os.path.dirname(png_path)

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

                # Start CNC execution tracking
                original_prompt = state_manager.current_drawing_prompt or "Unknown drawing"
                state_manager.start_cnc_execution(gcode_path, original_prompt)

                # Only pause idle movements if we're actually executing G-code
                if EXECUTE_GRBL_GCODE:
                    pause_for_drawing()

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
                    # Execution failed - must clear execution state to allow recovery
                    state_manager.finish_cnc_execution()
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

                    # Start CNC execution tracking
                    original_prompt = state_manager.current_drawing_prompt or "Unknown drawing"
                    state_manager.start_cnc_execution(gcode_path, original_prompt)

                    # Only pause idle movements if we're actually executing G-code
                    if EXECUTE_GRBL_GCODE:
                        pause_for_drawing()

                    # Convert SVG to G-code and execute
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
                        # Execution failed - must clear execution state to allow recovery
                        state_manager.finish_cnc_execution()
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
            # CRITICAL: This is where drawing actually completes - trigger uArm hook here
            if EXECUTE_GRBL_GCODE:
                print(f"🎯 [DEBUG] DRAWING EXECUTION COMPLETE - TRIGGERING UARM HOOK AND RESETTING COOLDOWN")

                # Reset drawing cooldown timer since physical drawing is now complete
                try:
                    import time
                    # Get access to the captioner to reset its drawing timer
                    # This is a bit hacky but necessary since image_monitor doesn't have direct access
                    import sys
                    if hasattr(sys.modules.get('__main__'), 'captioner'):
                        captioner = sys.modules['__main__'].captioner
                        captioner.last_drawing_time = time.time()
                        print(f"⏰ [DEBUG] DRAWING COOLDOWN RESET - next drawing can trigger in 60s")
                    else:
                        print(f"❌ [DEBUG] Could not reset drawing cooldown - captioner not accessible")
                except Exception as e:
                    print(f"❌ [DEBUG] DRAWING COOLDOWN RESET FAILED: {e}")

                # CRITICAL: Perform homing sequence BEFORE uArm hook
                try:
                    print(f"🏠 [DEBUG] PERFORMING FINAL HOMING BEFORE UARM TRIGGER")
                    from grbl.grbl_utils import find_grbl_port, ensure_homed

                    # Open serial connection and home the machine
                    try:
                        from config.config import GRBL_CNC_PORT
                        ser = find_grbl_port(preferred_port=GRBL_CNC_PORT, continuous_retry=False)
                    except ImportError:
                        ser = find_grbl_port(continuous_retry=False)

                    if ser:
                        ensure_homed(ser, max_retries=3)
                        print(f"🏠 [DEBUG] HOMING COMPLETE - CNC IS NOW AT HOME POSITION")
                        ser.close()
                    else:
                        print(f"❌ [DEBUG] Could not establish GRBL connection for homing")

                except Exception as e:
                    print(f"❌ [DEBUG] FINAL HOMING FAILED: {e}")

                # Now trigger uArm hook AFTER homing is complete
                try:
                    from utils.hooks import on_grbl_drawing_complete
                    if callable(on_grbl_drawing_complete):
                        print(f"🔥 [DEBUG] CALLING UARM HOOK FROM IMAGE MONITOR (AFTER HOMING)")
                        on_grbl_drawing_complete()
                        print(f"🔥 [DEBUG] UARM HOOK COMPLETED")
                    else:
                        print(f"❌ [DEBUG] NO UARM HOOK REGISTERED")
                except Exception as e:
                    print(f"❌ [DEBUG] UARM HOOK FAILED: {e}")

                # Resume idle movements after uArm completes
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
            # Strict gating: only process when a drawing job is active AND prefix matches
            try:
                expected = state_manager.get_expected_output_prefix()
                generating = getattr(state_manager, "is_generating_drawing", False)
            except Exception:
                expected = None
                generating = False

            if not generating:
                log_json_entry(
                    LogType.DECISION,
                    {"message": "Ignoring PNG: no active drawing generation", "filename": filename},
                    print_message=f"[⏭️] Ignoring {filename}: no active drawing",
                )
                return False

            if not expected:
                log_json_entry(
                    LogType.DECISION,
                    {"message": "Ignoring PNG: no expected output prefix set", "filename": filename},
                    print_message=f"[⏭️] Ignoring {filename}: no expected prefix",
                )
                return False

            if not os.path.basename(image_path).startswith(expected):
                log_json_entry(
                    LogType.DECISION,
                    {"message": "Ignoring PNG that does not match expected prefix", "expected_prefix": expected, "filename": filename},
                    print_message=f"[⏭️] Ignoring {filename}: does not match expected prefix",
                )
                return False

            # Stop the generation timer BEFORE starting CNC execution
            if state_manager.is_generating_drawing:
                state_manager.finish_drawing_generation()
                log_json_entry(
                    LogType.INFO,
                    {"message": "Drawing generation completed", "image_path": image_path},
                    print_message="[✅] Drawing generation completed",
                )

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
