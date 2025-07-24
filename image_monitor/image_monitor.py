import time
import os
import glob
import threading
from pathlib import Path
from config.config import COMFY_OUTPUT_FOLDER, MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry, LogType


class ImageMonitor:
    """Monitor a folder for new images and log them when they appear."""

    def __init__(self, monitor_folder=None, log_folder=None, check_interval=1.0):
        self.monitor_folder = monitor_folder or COMFY_OUTPUT_FOLDER
        self.log_folder = log_folder or MOOD_SNAPSHOT_FOLDER
        self.check_interval = check_interval
        self.image_extensions = {".png"}
        self.monitored_images = set()
        self.running = False
        self.thread = None

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
            str(self.log_folder),
            auto_print=True,
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
            self.log_folder,
            auto_print=True,
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

    def _log_new_image(self, image_path):
        """Log a newly detected image."""
        filename = os.path.basename(image_path)
        file_size = os.path.getsize(image_path)

        log_json_entry(
            LogType.NEW_DRAWING,
            {"event": "new_image_detected", "filename": filename, "image_path": image_path, "file_size": file_size, "timestamp": time.time()},
            self.log_folder,
            auto_print=True,
            print_message=f"🖼 New drawing: {filename} ({file_size} bytes)",
        )

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
                    self.log_folder,
                    auto_print=True,
                    print_message=f"❌ Image monitor error: {str(e)}",
                )
                time.sleep(5.0)  # Wait longer on error
