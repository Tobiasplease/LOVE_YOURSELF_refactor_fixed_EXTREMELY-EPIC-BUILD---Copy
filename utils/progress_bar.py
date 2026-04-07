"""
Simple dot animation for caption generation.
"""

import threading

# import sys
import time
from typing import Optional


class ProgressBar:
    """Simple dot animation that cycles through " ", ".", "..", "..."."""

    def __init__(self, width: int = 30, description: str = ""):
        self.active = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start the dot animation."""
        with self._lock:
            if self.active:
                return
            self.active = True
            self.thread = threading.Thread(target=self._animate, daemon=True)
            self.thread.start()

    def stop(self, success: bool = True):
        """Stop the animation and clear the line."""
        with self._lock:
            if not self.active:
                return
            self.active = False

        if self.thread:
            self.thread.join(timeout=0.1)

        # Clear the line completely
        print(f"\r{' ' * 10}\r", end="", flush=True)

    def _animate(self):
        """Animate with simple dots: " ", ".", "..", "..."."""
        dots = [" ", ".", "..", "..."]
        idx = 0

        while self.active:
            print(f"\r{dots[idx % len(dots)]}", end="", flush=True)
            idx += 1
            time.sleep(0.5)  # Slower animation for dots


