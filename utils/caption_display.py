import threading
import time
from typing import Optional, List
from collections import deque

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

        # Caption queuing system
        self.caption_queue = deque()
        self.lcd_busy = False
        self.queue_lock = threading.Lock()


        # Fixed timing system - responsive but readable
        self.caption_history = []  # Track caption timestamps
        self.base_chunk_delay = 300   # Fixed 300ms for readable chunks
        self.min_chunk_delay = 300    # Always 300ms chunks
        self.max_chunk_delay = 300    # Always 300ms chunks

        self._connect()

        # Start queue processing thread
        threading.Thread(target=self._process_queue, daemon=True).start()

    def _connect(self):
        try:
            print(f"[DISPLAY] Attempting to connect to {self.port}")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"[DISPLAY] Successfully connected to {self.port}")
        except Exception as e:
            print(f"[DISPLAY] Failed to connect to caption display: {e}")
            print(f"[DISPLAY] Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            self.connected = False

    def _process_queue(self):
        """Process caption queue, ensuring each caption displays fully before the next."""
        while True:
            try:
                # Check if we have captions to send and LCD is not busy
                with self.queue_lock:
                    if not self.caption_queue or self.lcd_busy:
                        time.sleep(0.1)
                        continue

                    caption, priority = self.caption_queue.popleft()
                    self.lcd_busy = True

                # Send caption and wait for completion
                success = self._send_caption_and_wait(caption, priority)

                with self.queue_lock:
                    self.lcd_busy = False

                if not success:
                    time.sleep(1)  # Wait before trying next caption if this one failed

            except Exception as e:
                print(f"[DISPLAY] Queue processing error: {e}")
                with self.queue_lock:
                    self.lcd_busy = False
                time.sleep(1)

    def _send_caption_and_wait(self, caption: str, priority: str, timeout: int = 30) -> bool:
        """Send caption and wait for Arduino to confirm it's fully displayed."""
        if not self.connected or not self.ser:
            return False

        try:
            adaptive_delay = self._calculate_adaptive_timing()
            priority_flag = {"HIGH": "H", "MEDIUM": "M", "LOW": "L"}.get(priority, "M")
            message = f"{priority_flag}:{adaptive_delay}:{caption}\n"

            # Send caption
            with self.send_lock:
                self.ser.write(message.encode())
                self.ser.flush()

            # Wait for Arduino response
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.ser.in_waiting > 0:
                    response = self.ser.readline().decode().strip()
                    if response == "CAPTION_ACCEPTED":
                        print(f"[DISPLAY] Caption accepted: {caption[:40]}...")
                        # Now wait for completion
                        while time.time() - start_time < timeout:
                            if self.ser.in_waiting > 0:
                                response = self.ser.readline().decode().strip()
                                if response == "CAPTION_COMPLETE":
                                    print(f"[DISPLAY] Caption completed")
                                    return True
                            time.sleep(0.1)
                    elif response == "CAPTION_BUSY":
                        print(f"[DISPLAY] LCD busy, caption queued")
                        return False
                time.sleep(0.1)

            print(f"[DISPLAY] Caption send timeout")
            return False

        except Exception as e:
            print(f"[DISPLAY] Send error: {e}")
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
        import time
        now = time.time()

        # Relaxed deduplication - only skip exact duplicates within 30 seconds
        if clean_caption == self.last_caption and hasattr(self, '_last_send_time') and (now - self._last_send_time) < 30:
            return

        # Auto-detect priority if not provided
        if priority is None:
            priority = self._detect_priority(clean_caption)

        # Track caption timing for adaptive system
        self.caption_history.append(now)

        # Minimal interruption logic - only skip if truly spamming
        if hasattr(self, '_last_send_time'):
            time_since_last = now - self._last_send_time

            # Only skip if extremely recent (< 0.5 seconds) to prevent spam
            if time_since_last < 0.5:
                return  # Skip silently

        self._last_send_time = now
        self._current_priority = priority

        # Add to queue instead of sending directly
        with self.queue_lock:
            # Limit queue size to prevent memory issues
            if len(self.caption_queue) > 10:
                self.caption_queue.popleft()  # Remove oldest if queue is full

            self.caption_queue.append((clean_caption, priority))
            print(f"[DISPLAY] Caption queued: {clean_caption[:40]}... (queue size: {len(self.caption_queue)})")

        self.last_caption = clean_caption

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
    else:
        print("[DISPLAY] Caption display not initialized - cannot set brightness")


def close_caption_display():
    global _caption_display
    if _caption_display:
        _caption_display.close()
        _caption_display = None