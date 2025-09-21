import threading
import time
from typing import Optional

import serial


class CaptionDisplay:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.last_caption = ""
        self.send_lock = threading.Lock()
        self.current_brightness = 255  # Default full brightness

        # Adaptive timing system - readable speeds
        self.caption_history = []  # Track caption timestamps
        self.base_chunk_delay = 1500  # Base 1.5s for comfortable reading
        self.min_chunk_delay = 800    # Minimum 800ms even when frequent
        self.max_chunk_delay = 2500   # Maximum 2.5s for slow reading

        self._connect()

    def _connect(self):
        try:
            print(f"[DISPLAY] Attempting to connect to {self.port}")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"[DISPLAY] Successfully connected to {self.port}")
        except (serial.SerialException, FileNotFoundError) as e:
            print(f"[DISPLAY] Failed to connect to caption display: {e}")
            self.connected = False

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

    def _calculate_adaptive_timing(self) -> int:
        """Calculate chunk delay based on recent caption frequency."""
        import time
        now = time.time()

        # Clean old history (keep last 5 minutes)
        self.caption_history = [t for t in self.caption_history if now - t < 300]

        if len(self.caption_history) < 2:
            return self.base_chunk_delay

        # Calculate average time between captions
        intervals = []
        for i in range(1, len(self.caption_history)):
            intervals.append(self.caption_history[i] - self.caption_history[i-1])

        avg_interval = sum(intervals) / len(intervals)

        # Adaptive logic: much faster chunks when captions come frequently
        if avg_interval < 20:  # Very frequent (< 20s apart)
            return self.min_chunk_delay  # 100ms - super fast
        elif avg_interval < 40:  # Moderate (20-40s apart)
            return int(self.base_chunk_delay * 0.7)  # 140ms - fast
        elif avg_interval < 60:  # Less frequent (40-60s apart)
            return self.base_chunk_delay  # 200ms
        else:  # Sparse (> 60s apart)
            return self.max_chunk_delay  # 500ms

    def send_caption(self, caption: str, priority: str = None, max_retries: int = 3):
        if not self.connected or not self.ser:
            print(f"[DISPLAY] Cannot send - not connected (connected={self.connected}, ser={self.ser is not None})")
            return

        clean_caption = caption.strip()

        # Strict deduplication - skip if exactly the same
        if clean_caption == self.last_caption:
            return

        import time
        now = time.time()

        # Auto-detect priority if not provided
        if priority is None:
            priority = self._detect_priority(clean_caption)

        # Track caption timing for adaptive system
        self.caption_history.append(now)

        # Simplified interruption logic - less aggressive skipping
        if hasattr(self, '_last_send_time'):
            time_since_last = now - self._last_send_time

            # Only skip if very recent (< 2 seconds) and not high priority
            if priority != "HIGH" and time_since_last < 2.0:
                return  # Skip silently

        self._last_send_time = now
        self._current_priority = priority

        # Send adaptive timing to Arduino
        adaptive_delay = self._calculate_adaptive_timing()
        priority_flag = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}.get(priority, "M")

        with self.send_lock:
            for attempt in range(max_retries):
                try:
                    # Send caption with timing and priority info
                    message = f"{priority_flag}:{adaptive_delay}:{clean_caption}\n"
                    self.ser.write(message.encode())
                    self.ser.flush()
                    self.last_caption = clean_caption
                    break  # Send silently
                except Exception as e:
                    print(f"[DISPLAY] Send attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        print(f"[DISPLAY] Failed to send caption after {max_retries} attempts")
                        self._reconnect()

    def set_brightness(self, brightness: int):
        """Set LCD brightness (0-255, 0=off, 255=full brightness)."""
        if not self.connected or not self.ser:
            print(f"[DISPLAY] Cannot set brightness - not connected")
            return

        brightness = max(0, min(255, brightness))
        self.current_brightness = brightness

        with self.send_lock:
            try:
                message = f"BRIGHTNESS:{brightness}\n"
                self.ser.write(message.encode())
                self.ser.flush()
                print(f"[DISPLAY] Brightness set to {brightness}")
            except Exception as e:
                print(f"[DISPLAY] Failed to set brightness: {e}")

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


def init_caption_display(port: str = "/dev/ttyUSB0"):
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
    else:
        print("[DISPLAY] Caption display not initialized - cannot set brightness")


def close_caption_display():
    global _caption_display
    if _caption_display:
        _caption_display.close()
        _caption_display = None