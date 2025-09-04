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
import time
from typing import Optional

class IdleMovementManager:
    """Manages idle movements as a subprocess that can be paused for drawing"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.is_paused = False
        self.emotion = "calm_observant"
        
    def start(self, emotion: str = "calm_observant") -> bool:
        """Start idle movements subprocess"""
        # Always stop any existing process first to avoid conflicts
        if self.process and self.process.poll() is None:
            print("[INFO] Stopping existing idle movements before starting new ones")
            self.stop()
        
        # Kill any orphaned idle movement processes
        try:
            subprocess.run(["pkill", "-f", "run_idle_movements.py"], 
                         capture_output=True, timeout=5)
            time.sleep(0.5)  # Brief pause for cleanup
        except:
            pass  # Ignore errors if no processes to kill
            
        self.emotion = emotion
        script_path = os.path.join(
            os.path.dirname(__file__), 
            "run_idle_movements.py"
        )
        
        try:
            self.process = subprocess.Popen(
                ["python", script_path, "--emotion", emotion],
                stdout=None,
                stderr=None
            )
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
        if not self.is_paused:
            return True
            
        # Wait a moment for serial port to be fully released
        time.sleep(1.0)
        
        # Restart with same emotion
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