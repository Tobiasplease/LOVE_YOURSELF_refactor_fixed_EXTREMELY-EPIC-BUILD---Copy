import time
import os
import glob
import threading
from pathlib import Path
from typing import Callable, Optional
from config.config import COMFY_OUTPUT_FOLDER, MOOD_SNAPSHOT_FOLDER, CENTER_LINE_SVG
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.state_manager import state_manager

from bcnc import raster_to_centerline_svg, svg_to_gcode


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
            print_message=f"👁️ Image monitor started: {self.monitor_folder}",
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
            print_message=f"📁 Found {len(self.monitored_images)} existing images",
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
                    print_message=f"🔄 Converting PNG to centerline SVG: {base_name}",
                )

                # Run svg_centerliner
                raster_to_centerline_svg(
                    input_path=png_path,
                    output_path=centerline_svg_path,
                    threshold_value=180,
                    blur_kernel=(3, 3),
                    do_dilate=True,
                    dilation_iterations=1,
                    scale=1.0,
                )

                # Convert SVG to G-code
                svg_to_gcode(svg_input=centerline_svg_path, output_gcode=gcode_path, auto_run=True)

                log_json_entry(
                    LogType.INFO,
                    {"message": f"G-code generated: {gcode_path}"},
                    print_message=f"🔧 G-code generated: {os.path.basename(gcode_path)}",
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
                        print_message=f"🔄 Converting latest SVG to G-code: {os.path.basename(latest_svg)}",
                    )

                    # Convert SVG to G-code
                    svg_to_gcode(svg_input=latest_svg, output_gcode=gcode_path, auto_run=True)

                    log_json_entry(
                        LogType.INFO,
                        {"message": f"G-code generated: {gcode_path}"},
                        print_message=f"🔧 G-code generated: {os.path.basename(gcode_path)}",
                    )
                else:
                    log_json_entry(
                        LogType.ERROR,
                        {"message": f"No SVG files found in output folder: {output_folder}"},
                        print_message=f"⚠️ No SVG files found in output folder",
                    )

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"error": f"PNG to G-code conversion failed: {str(e)}"},
                print_message=f"❌ PNG to G-code conversion failed: {str(e)}",
            )

    def _log_new_image(self, image_path):
        """Log a newly detected image."""
        filename = os.path.basename(image_path)
        file_size = os.path.getsize(image_path)

        log_json_entry(
            LogType.NEW_DRAWING,
            {"event": "new_image_detected", "filename": filename, "image_path": image_path, "file_size": file_size, "timestamp": time.time()},
            print_message=f"🖼 New drawing: {filename} ({file_size} bytes)",
        )

        if state_manager.is_generating_drawing:
            state_manager.finish_drawing_generation()
            log_json_entry(
                LogType.INFO, {"message": "Drawing generation completed", "image_path": image_path}, print_message="✅ Drawing generation completed"
            )

        # Process PNG to G-code if it's a PNG file
        if image_path.lower().endswith(".png"):
            self._process_png_to_gcode(image_path)

        if self.on_image_complete:
            self.on_image_complete(image_path)

    def _monitor_loop(self):
        """Main monitoring loop that runs in the background thread."""
        self._initialize_existing_images()

        while self.running:
            try:
                current_images = self._get_current_images()
                new_images = current_images - self.monitored_images

                for new_image in new_images:
                    self._log_new_image(new_image)
                    self.monitored_images.add(new_image)

                time.sleep(self.check_interval)

            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {"error": f"Image monitor error: {str(e)}"},
                    print_message=f"❌ Image monitor error: {str(e)}",
                )
                time.sleep(5.0)  # Wait longer on error
