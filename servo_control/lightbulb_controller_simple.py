#!/usr/bin/env python3
"""
Ultra-Simple Lightbulb Controller
=================================
Only two functions:
1. Set brightness based on frame difference (0-255)
2. Flash when new caption is generated

Matches lightbulb_simple.ino Arduino firmware exactly.
"""

import serial
import time
import threading
from typing import Optional

class SimpleLightbulbController:
    """Minimal lightbulb controller - frame diff brightness + caption flash only."""
    
    def __init__(self, port: str, baudrate: int = 9600, debug: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.debug = debug
        self.ser: Optional[serial.Serial] = None
        self.last_brightness = -1  # Track last sent value to avoid duplicates
        self.connection_lock = threading.Lock()
        self.connect()
    
    def connect(self) -> bool:
        """Connect to Arduino."""
        try:
            with self.connection_lock:
                if self.ser:
                    self.ser.close()
                
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=0.1,
                    write_timeout=0.1
                )
                time.sleep(0.5)  # Arduino boot time
                
                # Clear any startup messages
                if self.ser.in_waiting:
                    self.ser.read(self.ser.in_waiting)
                    
                if self.debug:
                    print(f"[SimpleLightbulb] Connected to {self.port}")
                return True
                
        except Exception as e:
            if self.debug:
                print(f"[SimpleLightbulb] Connection failed: {e}")
            self.ser = None
            return False
    
    def _send_command(self, command: str) -> bool:
        """Send command to Arduino with error handling."""
        try:
            with self.connection_lock:
                if not self.ser or not self.ser.is_open:
                    if not self.connect():
                        return False
                
                self.ser.write((command + '\n').encode())
                if self.debug:
                    print(f"[SimpleLightbulb] Sent: {command}")
                return True
                
        except Exception as e:
            if self.debug:
                print(f"[SimpleLightbulb] Send error: {e}")
            self.ser = None
            return False
    
    def set_frame_diff_brightness(self, brightness: int):
        """Set brightness based on frame difference (0-255)."""
        brightness = max(0, min(255, int(brightness)))
        
        # Skip if same as last sent value
        if brightness == self.last_brightness:
            return
        
        self.last_brightness = brightness
        self._send_command(f"B:{brightness}")
    
    def caption_flash(self):
        """Flash lightbulb when new caption is generated."""
        self._send_command("F")
        if self.debug:
            print("[SimpleLightbulb] Caption flash triggered")
    
    # Legacy compatibility methods
    def set_brightness(self, brightness: int):
        """Alias for set_frame_diff_brightness."""
        self.set_frame_diff_brightness(brightness)
    
    def set_base_brightness(self, brightness: int):
        """Alias for set_frame_diff_brightness."""
        self.set_frame_diff_brightness(brightness)
    
    def set_pwm(self, value: int):
        """Alias for set_frame_diff_brightness."""
        self.set_frame_diff_brightness(value)
    
    def caption_boost(self, duration=None):
        """Alias for caption_flash (duration ignored)."""
        self.caption_flash()
    
    def close(self):
        """Close serial connection."""
        try:
            with self.connection_lock:
                if self.ser:
                    self.ser.close()
                    self.ser = None
        except:
            pass

# For backward compatibility
LightbulbController = SimpleLightbulbController