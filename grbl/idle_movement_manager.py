#!/usr/bin/env python3
"""
Idle Movement Manager - Simple subprocess management for idle movements
========================================================================
Starts idle movements as a subprocess and provides pause/resume capability
to allow drawing execution to use the serial port.
"""

import os
import signal
import subprocess
import sys
import time
from typing import Optional

# Prevent idle from starting while a CNC drawing is executing
try:
    from utils.state_manager import state_manager
except Exception:
    state_manager = None


class IdleMovementManager:
    """Manages idle movements as a subprocess that can be paused for drawing"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.is_paused = False
        self.emotion = "calm_observant"

    def start(self, emotion: str = "calm_observant") -> bool:
        """Start idle movements subprocess"""
        # Do not start if a CNC drawing is currently executing
        if state_manager and getattr(state_manager, "is_executing_cnc", False):
            print("[INFO] Skipping idle start: CNC execution in progress")
            self.is_paused = True
            return False
        # Always stop any existing process first to avoid conflicts
        if self.process and self.process.poll() is None:
            print("[INFO] Stopping existing idle movements before starting new ones")
            self.stop()

        # Kill any orphaned idle movement processes
        try:
            subprocess.run(["pkill", "-f", "run_idle_movements.py"], capture_output=True, timeout=5)
            time.sleep(0.5)  # Brief pause for cleanup
        except:
            pass  # Ignore errors if no processes to kill

        self.emotion = emotion
        script_path = os.path.join(os.path.dirname(__file__), "run_idle_movements.py")

        try:
            # Use the same Python interpreter as the parent process (respects virtualenv)
            self.process = subprocess.Popen(
                [sys.executable, script_path, "--emotion", emotion],
                stdout=None,
                stderr=None,
            )
            try:
                from config.config import PRINT_CLEAN_CAPTIONS
                if not PRINT_CLEAN_CAPTIONS:
                    print(f"[🌊] Started idle movements with emotion: {emotion}")
            except ImportError:
                print(f"[🌊] Started idle movements with emotion: {emotion}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start idle movements: {e}")
            return False

    def pause_for_drawing(self) -> bool:
        """Pause idle movements to free serial port for drawing"""
        if not self.process or self.process.poll() is not None:
            print("[INFO] No idle movements to pause")
            return True

        try:
            # Send interrupt signal to gracefully stop idle movements
            self.process.send_signal(signal.SIGINT)
            self.is_paused = True

            # Wait for process to stop (max 5 seconds)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it didn't stop gracefully
                self.process.kill()
                self.process.wait()

            print("[⏸️] Idle movements paused for drawing")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to pause idle movements: {e}")
            return False

    def resume_after_drawing(self) -> bool:
        """Resume idle movements after drawing completes"""
        # If CNC is still executing, do not resume yet
        if state_manager and getattr(state_manager, "is_executing_cnc", False):
            print("[INFO] Not resuming idle: CNC execution still active")
            return False

        # If process not running, start regardless of paused flag (self-heal)
        proc_running = self.process is not None and self.process.poll() is None
        if not proc_running:
            time.sleep(1.0)  # allow port release
            success = self.start(self.emotion)
            if success:
                self.is_paused = False
                print("[▶️] Idle movements resumed")
            return success

        # Process is running already
        if not self.is_paused:
            return True

        # Paused flag set but process still running (edge) — restart cleanly
        try:
            self.stop()
        except Exception:
            pass
        time.sleep(1.0)
        success = self.start(self.emotion)
        if success:
            self.is_paused = False
            print("[▶️] Idle movements resumed")
        return success

    def stop(self):
        """Stop idle movements completely"""
        if self.process and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5)
            except:
                self.process.kill()
                self.process.wait()
            print("[🛑] Idle movements stopped")

    def update_emotion(self, emotion: str):
        """Update emotion by restarting with new emotion"""
        self.emotion = emotion
        if self.process and self.process.poll() is None:
            self.pause_for_drawing()
            time.sleep(0.5)
            self.resume_after_drawing()


# Global instance
_manager: Optional[IdleMovementManager] = None


def get_manager() -> IdleMovementManager:
    global _manager
    if _manager is None:
        _manager = IdleMovementManager()
    return _manager


def start_idle_movements(emotion: str = "calm_observant") -> bool:
    """Start idle movements"""
    return get_manager().start(emotion)


def pause_for_drawing() -> bool:
    """Pause idle movements for drawing"""
    return get_manager().pause_for_drawing()


def resume_after_drawing() -> bool:
    """Resume idle movements after drawing"""
    return get_manager().resume_after_drawing()


def stop_idle_movements():
    """Stop idle movements"""
    get_manager().stop()


def update_emotion(emotion: str):
    """Update emotion for idle movements"""
    # Skip emotion updates to avoid disruptive homing/restart cycles
    # Idle movements are organic and varied even with static emotion state
    pass
