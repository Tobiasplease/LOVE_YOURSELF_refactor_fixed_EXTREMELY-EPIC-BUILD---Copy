#!/usr/bin/env python3
"""
Ultra-Robust Lightbulb Controller
=================================
Thread-safe Arduino lightbulb controller with comprehensive error handling
and crash prevention for multi-threaded environments.
"""

import serial
import time
import threading
import queue
from typing import Optional
from contextlib import contextmanager


class RobustLightbulbController:
    """Crash-proof lightbulb controller with command queuing and error isolation."""
    
    def __init__(self, port: str, baudrate: int = 9600, debug: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.debug = debug
        self.ser: Optional[serial.Serial] = None
        self.last_brightness = -1
        self.is_connected = False
        self.connection_lock = threading.RLock()
        self.command_queue = queue.Queue(maxsize=100)
        self.shutdown_event = threading.Event()
        self.worker_thread = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self._start_worker()
    
    def _start_worker(self):
        """Start background worker thread for command processing."""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="LightbulbWorker")
        self.worker_thread.start()
    
    def _worker_loop(self):
        """Background worker that processes commands safely."""
        while not self.shutdown_event.is_set():
            try:
                command = self.command_queue.get(timeout=0.5)
                if command is None:
                    break
                self._execute_command_safe(command)
                self.command_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                if self.debug:
                    print(f"[RobustLightbulb] Worker error: {e}")
    
    @contextmanager
    def _safe_connection(self):
        """Context manager for safe serial connection handling."""
        try:
            with self.connection_lock:
                if not self.is_connected:
                    self._connect_internal()
                if self.ser and self.ser.is_open:
                    yield self.ser
                else:
                    yield None
        except Exception as e:
            if self.debug:
                print(f"[RobustLightbulb] Connection context error: {e}")
            yield None
    
    def _connect_internal(self) -> bool:
        """Internal connection method with retry logic."""
        if self.connection_attempts >= self.max_connection_attempts:
            return False
        
        try:
            if self.ser:
                self.ser.close()
                time.sleep(0.1)
            
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.1, write_timeout=0.1)
            time.sleep(0.3)
            
            if self.ser.in_waiting:
                self.ser.read(self.ser.in_waiting)
            
            self.is_connected = True
            self.connection_attempts = 0
            if self.debug:
                print(f"[RobustLightbulb] Connected to {self.port}")
            return True
            
        except Exception as e:
            if self.debug:
                print(f"[RobustLightbulb] Connection failed (attempt {self.connection_attempts + 1}): {e}")
            self.connection_attempts += 1
            self.is_connected = False
            self.ser = None
            return False
    
    def _execute_command_safe(self, command: str) -> bool:
        """Execute command with comprehensive error handling."""
        try:
            with self._safe_connection() as conn:
                if conn is None:
                    return False
                
                conn.write((command + '\n').encode())
                if self.debug:
                    print(f"[RobustLightbulb] Sent: {command}")
                return True
                
        except Exception as e:
            if self.debug:
                print(f"[RobustLightbulb] Command error: {e}")
            self.is_connected = False
            return False
    
    def _queue_command(self, command: str):
        """Queue command for background processing."""
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            if self.debug:
                print("[RobustLightbulb] Command queue full, dropping command")
    
    def set_frame_diff_brightness(self, brightness: int):
        """Set brightness based on frame difference (0-255) - queued execution."""
        brightness = max(0, min(255, int(brightness)))
        if brightness == self.last_brightness:
            return
        self.last_brightness = brightness
        self._queue_command(f"B:{brightness}")
    
    def caption_flash(self):
        """Flash lightbulb when new caption is generated - queued execution."""
        self._queue_command("F")
    
    def set_brightness(self, brightness: int):
        """Alias for set_frame_diff_brightness."""
        self.set_frame_diff_brightness(brightness)
    
    def close(self):
        """Clean shutdown of controller."""
        self.shutdown_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        try:
            with self.connection_lock:
                if self.ser:
                    self.ser.close()
                    self.ser = None
        except:
            pass
        
        self.is_connected = False


class ThreadSafeLightbulbWrapper:
    """Ultra-safe wrapper that prevents any crashes from lightbulb issues."""
    
    def __init__(self, port: str, debug: bool = False):
        self.controller = None
        self.debug = debug
        self.initialization_lock = threading.Lock()
        try:
            self.controller = RobustLightbulbController(port, debug=debug)
        except Exception as e:
            if debug:
                print(f"[LightbulbWrapper] Init failed: {e}")
    
    def _safe_call(self, method_name: str, *args, **kwargs):
        """Execute any method call with complete crash protection."""
        try:
            if self.controller and hasattr(self.controller, method_name):
                method = getattr(self.controller, method_name)
                return method(*args, **kwargs)
        except Exception as e:
            if self.debug:
                print(f"[LightbulbWrapper] Method {method_name} failed: {e}")
        return None
    
    def set_frame_diff_brightness(self, brightness: int):
        """Crash-proof brightness setting."""
        self._safe_call('set_frame_diff_brightness', brightness)
    
    def caption_flash(self):
        """Crash-proof caption flash."""
        self._safe_call('caption_flash')
    
    def close(self):
        """Crash-proof cleanup."""
        self._safe_call('close')