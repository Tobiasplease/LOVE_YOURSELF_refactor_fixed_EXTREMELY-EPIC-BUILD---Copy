#!/usr/bin/env python3
"""
    def __init__(self, sensitivity=1.0, smoothing_factor=0.8, pause_threshold=0.5, pause_duration=3.0):amera Reactivity System for Hand Controller
===========================================

Real-time camera-based reactivity that affects Markov chain generation.
Detects frame differences, activity levels, and motion to create
organic behavioral multipliers.

Features:
- Frame difference analysis for activity detection
- Smoothed reactivity scores to avoid jitter
- Multiple reactivity channels (movement, appearance, disappearance)
- Integration with hand controller Markov generation

Author: Camera Reactivity System
"""
import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any
import json
from config.config import REACTIVITY_PAUSE_THRESHOLD


class CameraReactivityEngine:
    """
    Real-time camera analysis for behavioral reactivity.
    """
    
    def __init__(self, sensitivity=1.8, smoothing_factor=0.85, pause_threshold=None, pause_duration=3.0):
        """
        Initialize camera reactivity engine.
        
        Args:
            sensitivity: How sensitive to frame changes (0.0-2.0, higher = more sensitive)
            smoothing_factor: Temporal smoothing (0.0=no smoothing, 1.0=max smoothing)
            pause_threshold: Activity level that triggers pause (0.0-1.0, lower = easier to trigger)
            pause_duration: How long to pause in seconds
        """
        self.sensitivity = sensitivity
        self.smoothing_factor = smoothing_factor
        self.pause_threshold = pause_threshold if pause_threshold is not None else REACTIVITY_PAUSE_THRESHOLD
        self.pause_duration = pause_duration
        
        # Frame analysis
        self.previous_frame = None
        self.frame_count = 0
        
        # Single unified activity level for pause triggering
        self.activity_level = 0.0        # Overall movement/change (0.0-1.0)
        
        # Pause system for high activity
        self.is_paused = False
        self.pause_start_time = 0.0
        self.pause_cooldown = 10.0        # 10 second cooldown between pauses
        self.last_pause_time = 0.0
        
        # Historical tracking for better detection
        self.activity_history = []
        self.max_history = 15  # Keep 0.5 seconds at 30fps
        
        # Reactivity state
        self.last_update_time = time.time()
        self.reactivity_active = True
        
        # Debug info
        self.debug_enabled = True  # Enable by default for better feedback
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Process a camera frame and return reactivity metrics.
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            Dict with reactivity metrics:
            - activity_level: Overall scene activity (0.0-1.0)
            - sudden_change: Spike in activity (0.0-1.0) 
            - motion_intensity: Movement strength (0.0-1.0)
            - chaos_multiplier: Suggested chaos factor for Markov (0.5-2.0)
            - speed_multiplier: Suggested speed factor (0.5-3.0)
        """
        if frame is None or not self.reactivity_active:
            return self._get_neutral_metrics()
            
        current_time = time.time()
        self.frame_count += 1
        
        # Convert to grayscale for analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        if self.previous_frame is None:
            self.previous_frame = blurred_frame.copy()
            return self._get_neutral_metrics()
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.previous_frame, blurred_frame)
        
        # Apply threshold to get binary difference
        threshold_value = int(15 * self.sensitivity)  # Balanced threshold
        _, thresh_diff = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Calculate activity metrics - balanced sensitivity
        raw_activity = np.sum(thresh_diff) / (frame.shape[0] * frame.shape[1] * 255)  # Normalize 0-1
        
        # Boost activity by sensitivity multiplier but more conservatively
        boosted_activity = min(1.0, raw_activity * self.sensitivity * 3.5)  # Reduced multiplier
        
        # Smooth the activity level but keep it responsive
        self.activity_level = (self.smoothing_factor * self.activity_level + 
                              (1 - self.smoothing_factor) * boosted_activity)
        
        # Track activity history for stability
        self.activity_history.append(boosted_activity)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
        
        # Detect sudden changes (spikes) - simplified
        recent_peak = max(self.activity_history[-5:]) if len(self.activity_history) >= 5 else 0.0
        
        # MEMORY FIX: Reuse previous frame buffer instead of copying
        np.copyto(self.previous_frame, blurred_frame)
        
        # Clean up temporary arrays to prevent memory leaks
        del gray_frame, blurred_frame, frame_diff, thresh_diff
        
        # Check for pause conditions (high activity triggers freeze)
        pause_active = self._check_pause_conditions(current_time)
        
        # Debug output - show activity and pause proximity (disabled by default)
        # if self.debug_enabled and self.frame_count % 15 == 0:  # Every 0.5 seconds
        #     progress_to_pause = (self.activity_level / self.pause_threshold) * 100
        #     print(f"🎥 Activity: {self.activity_level:.3f} | Pause at: {self.pause_threshold:.3f} | Progress: {progress_to_pause:.1f}% | Paused: {pause_active}")
        
        return {
            'activity_level': self.activity_level,
            'pause_threshold': self.pause_threshold,
            'progress_to_pause': min(100.0, (self.activity_level / self.pause_threshold) * 100),
            'is_paused': pause_active,
            'pause_remaining': max(0.0, (self.pause_start_time + self.pause_duration) - current_time) if pause_active else 0.0,
            'cooldown_remaining': max(0.0, self.pause_cooldown - (current_time - self.last_pause_time)),
            'timestamp': current_time,
            # Legacy compatibility for machine.py
            'chaos_multiplier': min(3.5, 0.3 + (self.activity_level * 3.2)),  # Convert activity to chaos range
            'speed_multiplier': min(2.0, 1.0 + self.activity_level)  # Convert activity to speed range
        }
    
    def _calculate_motion_intensity(self, frame_diff: np.ndarray) -> float:
        """Calculate motion intensity using gradient analysis."""
        # Calculate gradients to detect directional movement
        grad_x = cv2.Sobel(frame_diff, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame_diff, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude of gradient vectors
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-1 range
        max_possible = frame_diff.shape[0] * frame_diff.shape[1] * 255
        intensity = np.sum(magnitude) / max_possible
        
        # Smooth the intensity
        self.motion_intensity = (self.smoothing_factor * self.motion_intensity + 
                                (1 - self.smoothing_factor) * intensity)
        
        return min(1.0, self.motion_intensity)
    
    def _calculate_chaos_multiplier(self) -> float:
        """
        Calculate chaos multiplier for Markov chain generation.
        Higher activity = more chaotic/random behavior.
        """
        # Base multiplier is 1.0 (normal behavior)
        base = 1.0
        
        # Activity contribution (0.0-1.0 becomes 0.0-1.5 bonus)
        activity_bonus = self.activity_level * 1.5
        
        # Sudden change contribution (spikes add extra chaos)
        spike_bonus = self.sudden_change * 0.8
        
        # Total chaos multiplier: 1.0-3.5 range
        chaos = base + activity_bonus + spike_bonus
        
        return min(3.5, max(0.3, chaos))
    
    def _calculate_speed_multiplier(self) -> float:
        """
        Calculate speed multiplier for movement timing.
        Higher motion = faster movements.
        """
        # Base speed is 1.0 (normal timing)
        base = 1.0
        
        # Motion intensity affects speed (0.0-1.0 becomes 0.0-2.5 bonus)
        motion_bonus = self.motion_intensity * 2.5
        
        # Sudden changes trigger immediate fast responses
        sudden_bonus = self.sudden_change * 1.0
        
        # Total speed multiplier: 0.2-4.5 range
        speed = base + motion_bonus + sudden_bonus
        
        return min(4.5, max(0.2, speed))
    
    def _check_pause_conditions(self, current_time: float) -> bool:
        """
        Check if high activity should trigger a pause/freeze reaction.
        Returns True if the system should pause movement.
        """
        # Check if we're currently in a pause
        if self.is_paused:
            # Check if pause duration has elapsed
            if current_time - self.pause_start_time >= self.pause_duration:
                self.is_paused = False
                # print(f"SUCCESS CAMERA PAUSE ENDED - Resuming hand movement after {self.pause_duration}s")
                return False
            else:
                return True  # Still paused
        
        # Check if activity level exceeds pause threshold
        activity_spike = (self.activity_level >= self.pause_threshold)
        
        if (activity_spike and 
            current_time - self.last_pause_time >= self.pause_cooldown):
            
            # Trigger pause
            self.is_paused = True
            self.pause_start_time = current_time
            self.last_pause_time = current_time
            
            # Pause feedback (commented out to reduce terminal output)
            # print(f"🛑 CAMERA PAUSE TRIGGERED! Activity: {self.activity_level:.3f} > {self.pause_threshold:.3f} - Pausing for {self.pause_duration}s")
            
            return True
        
        return False
    
    def _get_neutral_metrics(self) -> Dict[str, float]:
        """Return neutral/baseline metrics when no processing possible."""
        return {
            'activity_level': 0.0,
            'pause_threshold': self.pause_threshold,
            'progress_to_pause': 0.0,
            'is_paused': False,
            'pause_remaining': 0.0,
            'cooldown_remaining': 0.0,
            'timestamp': time.time(),
            # Legacy compatibility
            'chaos_multiplier': 0.3,  # Baseline chaos
            'speed_multiplier': 1.0   # Baseline speed
        }
    
    def set_sensitivity(self, sensitivity: float):
        """Update sensitivity (0.0-1.0)."""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
    
    def set_smoothing(self, smoothing_factor: float):
        """Update temporal smoothing (0.0-1.0)."""
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
    
    def enable_debug(self, enabled: bool = True):
        """Enable/disable debug output."""
        self.debug_enabled = enabled
    
    def reset(self):
        """Reset all state for fresh start."""
        self.previous_frame = None
        self.frame_count = 0
        self.activity_level = 0.0
        self.sudden_change = 0.0
        self.motion_intensity = 0.0
        self.activity_history = []
        
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status and metrics for debugging."""
        return {
            'frame_count': self.frame_count,
            'activity_level': self.activity_level,
            'sudden_change': self.sudden_change,
            'motion_intensity': self.motion_intensity,
            'sensitivity': self.sensitivity,
            'smoothing_factor': self.smoothing_factor,
            'history_length': len(self.activity_history),
            'active': self.reactivity_active,
            'last_update': self.last_update_time
        }


def test_camera_reactivity():
    """Test the camera reactivity system with webcam."""
    print("🎥 Testing Camera Reactivity System")
    print("Press 'q' to quit, 's' to toggle sensitivity, 'd' for debug")
    
    engine = CameraReactivityEngine(sensitivity=0.5, smoothing_factor=0.7)
    engine.enable_debug(True)
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR Could not open camera")
        return
    
    cv2.namedWindow('Camera Reactivity Test', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR Could not read frame")
                break
            
            # Process frame for reactivity
            metrics = engine.process_frame(frame)
            
            # Draw metrics on frame
            height, width = frame.shape[:2]
            
            # Activity level bar
            activity_bar_width = int(metrics['activity_level'] * 200)
            cv2.rectangle(frame, (10, 10), (210, 30), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (10 + activity_bar_width, 30), (0, 255, 0), -1)
            cv2.putText(frame, f"Activity: {metrics['activity_level']:.3f}", 
                       (220, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Chaos multiplier
            cv2.putText(frame, f"Chaos: {metrics['chaos_multiplier']:.2f}x", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Speed multiplier
            cv2.putText(frame, f"Speed: {metrics['speed_multiplier']:.2f}x", 
                       (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Sudden change indicator
            if metrics['sudden_change'] > 0.1:
                cv2.circle(frame, (width - 30, 30), 20, (0, 0, 255), -1)
                cv2.putText(frame, "!", (width - 35, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Camera Reactivity Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Toggle sensitivity
                engine.set_sensitivity(0.8 if engine.sensitivity < 0.6 else 0.3)
                print(f"🎚️ Sensitivity: {engine.sensitivity}")
            elif key == ord('d'):
                # Toggle debug
                engine.enable_debug(not engine.debug_enabled)
                print(f"🐛 Debug: {engine.debug_enabled}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Camera reactivity test complete")


if __name__ == "__main__":
    test_camera_reactivity()
