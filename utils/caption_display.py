import threading
import time
from typing import Optional

import serial


class CaptionDisplay:
    def __init__(self, port: str = "/dev/arduino_lcd", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.last_caption = ""
        self.send_lock = threading.Lock()
        self.current_brightness = 255  # Default full brightness

        # Slower timing for better readability
        self.chunk_delay = 1000  # 1000ms (1 second) between chunks for readability

        # Simple rate limiting with skip-ahead logic
        self.last_sent_time = 0
        self.min_interval = 3.0  # 3 seconds between captions for readability
        self.caption_queue = []
        self.skip_counter = 0

        self._connect()


    def _connect(self):
        try:
            print(f"[DISPLAY] Attempting to connect to {self.port}")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"[DISPLAY] Successfully connected to {self.port}")

            # Test if Arduino responds to anything
            print(f"[DEBUG] Testing Arduino communication...")
            self.ser.write(b"BRIGHTNESS:255\n")
            self.ser.flush()
            time.sleep(0.5)
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode().strip()
                print(f"[DEBUG] Arduino brightness response: {response}")
            else:
                print(f"[DEBUG] No response to brightness command")
        except Exception as e:
            print(f"[DISPLAY] Failed to connect to caption display: {e}")
            print(f"[DISPLAY] Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            self.connected = False

    def _monitor_arduino_status(self):
        """Background thread to continuously monitor Arduino status."""
        while True:
            try:
                if self.connected and self.ser:
                    self._check_arduino_status()
                time.sleep(0.1)  # Check every 100ms
            except Exception as e:
                time.sleep(1)


    def _check_arduino_status(self):
        """Check for any pending Arduino responses and update LCD state."""
        if not self.connected or not self.ser:
            return

        try:
            while self.ser.in_waiting > 0:
                response = self.ser.readline().decode().strip()
                if response == "CAPTION_COMPLETE":
                    self.lcd_busy = False
                    # Send latest caption if we have one
                    if self.latest_caption:
                        caption, priority = self.latest_caption
                        self.latest_caption = None
                        self._try_send_caption(caption, priority)
                elif response == "CAPTION_ACCEPTED":
                    self.lcd_busy = True
        except Exception:
            pass

    def _try_send_caption(self, caption: str, priority: str) -> bool:
        """Try to send caption, return True if sent, False if LCD busy."""
        if not self.connected or not self.ser:
            return False

        try:
            # Use single-row format (16 chars max per chunk)
            priority_flag = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}.get(priority, "M")
            message = f"{priority_flag}:2000:{caption}\n"  # 2 second delay for readability

            print(f"[DEBUG] Sending to Arduino: {message.strip()}")
            with self.send_lock:
                self.ser.write(message.encode())
                self.ser.flush()

            # Don't wait for response - just force send every caption
            return True

        except Exception:
            return False

    def _detect_priority(self, caption: str) -> str:
        """Detect caption priority based on content."""
        caption_lower = caption.lower()

        # High priority: movement, interaction, changes
        high_priority_words = ['moving', 'walking', 'entering', 'leaving', 'appears', 'disappears',
                              'person', 'face', 'hand', 'gesture', 'looking', 'turns', 'stops']

        # Medium priority: emotions, expressions
        medium_priority_words = ['smiling', 'frowning', 'surprised', 'focused', 'tired', 'happy']

        # Low priority: static observations
        low_priority_words = ['sitting', 'standing', 'room', 'wall', 'table', 'chair']

        if any(word in caption_lower for word in high_priority_words):
            return "HIGH"
        elif any(word in caption_lower for word in medium_priority_words):
            return "MEDIUM"
        elif any(word in caption_lower for word in low_priority_words):
            return "LOW"

        return "MEDIUM"  # Default priority


    def send_caption(self, caption: str, priority: str = None):
        if not self.connected or not self.ser:
            return

        clean_caption = caption.strip()
        import time
        now = time.time()

        # Add to queue
        self.caption_queue.append((clean_caption, priority))

        # Rate limit captions for readability
        if hasattr(self, 'last_sent_time') and now - self.last_sent_time < self.min_interval:
            return

        # Every 4th caption, skip to the latest
        self.skip_counter += 1
        if self.skip_counter >= 4 and len(self.caption_queue) > 1:
            # Skip to latest caption
            clean_caption, priority = self.caption_queue[-1]
            self.caption_queue.clear()
            self.skip_counter = 0
        else:
            # Send oldest caption
            if self.caption_queue:
                clean_caption, priority = self.caption_queue.pop(0)
            else:
                return

        if priority is None:
            priority = self._detect_priority(clean_caption)

        # Send this caption
        success = self._try_send_caption(clean_caption, priority)
        if success:
            self.last_caption = clean_caption
            self.last_sent_time = now

    def set_brightness(self, brightness: int):
        """Set LCD brightness (0-255, 0=off, 255=full brightness)."""
        if not self.connected or not self.ser:
            return

        brightness = max(0, min(255, brightness))
        self.current_brightness = brightness

        with self.send_lock:
            try:
                message = f"BRIGHTNESS:{brightness}\n"
                self.ser.write(message.encode())
                self.ser.flush()
            except Exception:
                pass

    def _reconnect(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.connected = False
        time.sleep(1)
        self._connect()

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.connected = False


# Global instance
_caption_display: Optional[CaptionDisplay] = None


def init_caption_display(port: str = "/dev/arduino_lcd"):
    global _caption_display
    if _caption_display is None:
        _caption_display = CaptionDisplay(port)


def send_caption_to_display(caption: str):
    global _caption_display
    if _caption_display:
        threading.Thread(target=_caption_display.send_caption, args=(caption,), daemon=True).start()


def set_lcd_brightness(brightness: int):
    """Set LCD brightness globally (0-255)."""
    global _caption_display
    if _caption_display:
        _caption_display.set_brightness(brightness)


def close_caption_display():
    global _caption_display
    if _caption_display:
        _caption_display.close()
        _caption_display = None