#!/usr/bin/env python3
"""
Non-blocking Lightbulb Controller
================================
Ultra-simple lightbulb controller that never blocks the main thread.
Uses fire-and-forget approach with minimal error handling.
"""

import threading
import time
from typing import Optional

import serial


class NonBlockingLightbulbController:
    """Fire-and-forget lightbulb controller that never blocks."""

    def __init__(self, port: str, baudrate: int = 9600, debug: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.debug = debug
        self.ser: Optional[serial.Serial] = None
        self.last_brightness = -1
        self.connection_attempted = False
        self._try_connect()

    def _try_connect(self) -> None:
        """Try to connect once - if it fails, give up quietly."""
        if self.connection_attempted:
            return

        self.connection_attempted = True
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.01,  # Extremely short timeout
                write_timeout=0.01,  # Extremely short write timeout
                dsrdtr=False,
                rtscts=False,
            )
            # Don't wait for Arduino boot - just send commands anyway
            if self.debug:
                print(f"[NonBlockingLightbulb] Connected to {self.port}")
        except Exception as e:
            if self.debug:
                print(f"[NonBlockingLightbulb] Connection failed: {e}")
            self.ser = None

    def _fire_and_forget_command(self, command: str) -> None:
        """Send command without waiting for response or checking errors."""
        try:
            if self.ser and self.ser.is_open:
                self.ser.write((command + "\n").encode())
                # No flush, no wait, no error checking
        except:
            # Ignore all errors - this is fire and forget
            pass

    def set_frame_diff_brightness(self, brightness: int):
        """Set brightness - fire and forget."""
        brightness = max(0, min(255, int(brightness)))

        # Skip if same as last sent value (reduce serial traffic)
        if brightness == self.last_brightness:
            return

        self.last_brightness = brightness
        self._fire_and_forget_command(f"B:{brightness}")

        if self.debug and brightness % 50 == 0:  # Only debug every 50 brightness levels
            print(f"[NonBlockingLightbulb] Brightness: {brightness}")

    def caption_flash(self):
        """Flash for caption - fire and forget."""
        self._fire_and_forget_command("F")
        if self.debug:
            print("[NonBlockingLightbulb] Flash triggered")

    def close(self):
        """Close connection if it exists."""
        try:
            if self.ser:
                self.ser.close()
        except:
            pass
