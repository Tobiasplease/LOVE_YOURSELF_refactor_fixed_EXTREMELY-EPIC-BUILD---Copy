"""
servo_control/hand_expression.py
-------------------------------
Consciousness-driven robotic hand expression system.

Maps AI emotional states to organic finger movements while preserving
the existing tapping behavior and mirrored servo mechanics.
"""

import time
import random
import math
from typing import Dict, Tuple, Optional
from enum import Enum


class HandGesture(Enum):
    """Predefined hand gestures for different emotional states."""
    IDLE = "idle"                    # Random gentle movements
    CURIOUS = "curious"              # Index finger extended, others relaxed
    CONTEMPLATIVE = "contemplative"  # Fingers slightly curled, gentle motion
    AGITATED = "agitated"           # Rapid, jerky movements
    FOCUSED = "focused"             # Fingers still, slight tension
    EXPRESSIVE = "expressive"       # Dynamic gestures, finger dancing
    WITHDRAWN = "withdrawn"         # Fingers curled inward, minimal motion
    AWAKENING = "awakening"         # Slow uncurling from rest position


class HandExpressionController:
    """Controls robotic hand based on AI consciousness states."""
    
    def __init__(self, port: str = "COM3", baudrate: int = 9600, clean_output: bool = False):
        # Serial connection for hand communication
        self.port = port
        self.baudrate = baudrate
        self.clean_output = clean_output
        self.serial_connection = None
        self._init_serial()
        
        # Match your Arduino servo setup exactly
        self.num_fingers = 4
        self.finger_names = ["index", "middle", "ring", "pinky"]  # pins 8,9,10,11
        self.mirrored_fingers = [False, False, True, True]  # pins 10,11 are mirrored
        
        # Your proven ranges
        self.min_angle = 40
        self.max_angle = 130
        self.tap_curled = 130      # 30° when mirrored
        self.tap_outstretched = 10 # 170° when mirrored
        
        # Current state tracking
        self.current_gesture = HandGesture.IDLE
        self.finger_targets = [70, 70, 70, 70]  # Mid-range default
        self.finger_speeds = [30, 30, 30, 30]   # Default speeds
        self.gesture_start_time = time.time()
        self.gesture_duration = 0.0
        
        # Special behaviors
        self.tap_finger_index = 3  # Pinky (pin 11) remains the tapper
        self.enable_tapping = True
        self.last_mood_change_time = 0.0
        
    def _init_serial(self):
        """Initialize serial connection to hand controller."""
        try:
            import serial
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            if not self.clean_output:
                print(f"[HAND] Connected to hand controller on {self.port}")
        except ImportError:
            if not self.clean_output:
                print("[HAND] PySerial not available - hand control disabled")
            self.serial_connection = None
        except Exception as e:
            if not self.clean_output:
                print(f"[HAND] Failed to connect to {self.port}: {e}")
            self.serial_connection = None
    
    def _send_hand_command(self, finger_positions: Dict[str, int]):
        """Send finger positions to Arduino hand controller."""
        if not self.serial_connection or not self.serial_connection.is_open:
            return
            
        try:
            # Format: "HAND,f0,f1,f2,f3\n" 
            pos_list = [finger_positions.get(f"finger{i}", 70) for i in range(4)]
            command = f"HAND,{','.join(map(str, pos_list))}\n"
            self.serial_connection.write(command.encode())
        except Exception as e:
            if not self.clean_output:
                print(f"[HAND] Serial communication error: {e}")
        
    def update_from_consciousness(self, mood: float, novelty: float, 
                                boredom: float, person_present: bool,
                                temporal_context: Optional[Dict] = None) -> Dict[str, int]:
        """
        Update hand expression based on AI consciousness state.
        Returns finger positions as dict: {"finger0": angle, "finger1": angle, ...}
        """
        current_time = time.time()
        
        # Determine appropriate gesture based on consciousness state
        new_gesture = self._select_gesture(mood, novelty, boredom, person_present, temporal_context)
        
        # Change gesture if needed
        if new_gesture != self.current_gesture:
            self._transition_to_gesture(new_gesture, current_time)
        
        # Update finger positions based on current gesture
        finger_positions = self._update_gesture_motion(current_time)
        
        # Send commands to Arduino hand controller
        self._send_hand_command(finger_positions)
        
        return finger_positions
    
    def _select_gesture(self, mood: float, novelty: float, boredom: float, 
                       person_present: bool, temporal_context: Optional[Dict] = None) -> HandGesture:
        """Select appropriate gesture based on consciousness state."""
        
        # Handle special temporal states
        if temporal_context:
            consciousness_state = temporal_context.get('consciousness_state', '')
            if 'freshly awakened' in consciousness_state:
                return HandGesture.AWAKENING
        
        # Mood-driven gesture selection
        if mood < 0.2:
            return HandGesture.WITHDRAWN
        elif mood > 0.8 and novelty > 0.6:
            return HandGesture.EXPRESSIVE
        elif person_present and novelty > 0.5:
            return HandGesture.CURIOUS
        elif boredom > 0.7:
            return HandGesture.AGITATED
        elif mood > 0.6 and boredom < 0.3:
            return HandGesture.CONTEMPLATIVE
        elif novelty < 0.2 and boredom < 0.5:
            return HandGesture.FOCUSED
        else:
            return HandGesture.IDLE
    
    def _transition_to_gesture(self, new_gesture: HandGesture, current_time: float):
        """Smoothly transition to a new gesture."""
        self.current_gesture = new_gesture
        self.gesture_start_time = current_time
        self.gesture_duration = random.uniform(3.0, 8.0)  # Variable gesture duration
        
        # Set gesture-specific parameters
        gesture_configs = {
            HandGesture.WITHDRAWN: ([110, 105, 100, 95], [20, 20, 20, 20]),  # Slow, closed
            HandGesture.EXPRESSIVE: ([30, 35, 40, 45], [15, 15, 15, 15]),   # Fast, open
            HandGesture.CURIOUS: ([50, 80, 85, 90], [25, 25, 25, 25]),      # Index extended
            HandGesture.AGITATED: ([60, 70, 65, 75], [10, 12, 10, 12]),     # Fast, restless
            HandGesture.CONTEMPLATIVE: ([90, 85, 80, 75], [30, 30, 30, 30]), # Slow, curved
            HandGesture.FOCUSED: ([70, 75, 80, 85], [40, 40, 40, 40]),      # Very slow, steady
            HandGesture.AWAKENING: ([120, 115, 110, 105], [35, 35, 35, 35]), # Slow uncurling
            HandGesture.IDLE: ([70, 70, 70, 70], [25, 25, 25, 25])          # Default
        }
        
        if new_gesture in gesture_configs:
            targets, speeds = gesture_configs[new_gesture]
            self.finger_targets = targets[:]
            self.finger_speeds = speeds[:]
    
    def _update_gesture_motion(self, current_time: float) -> Dict[str, int]:
        """Update finger positions based on current gesture and timing."""
        gesture_elapsed = current_time - self.gesture_start_time
        positions = {}
        
        for i in range(self.num_fingers):
            # Base angle from gesture
            base_angle = self.finger_targets[i]
            
            # Add gesture-specific motion patterns
            if self.current_gesture == HandGesture.AGITATED:
                # Random jerky movements
                jitter = random.randint(-15, 15)
                base_angle = max(self.min_angle, min(self.max_angle, base_angle + jitter))
                
            elif self.current_gesture == HandGesture.EXPRESSIVE:
                # Flowing, wave-like motion
                wave = 10 * math.sin(gesture_elapsed * 2 + i * 0.5)
                base_angle = max(self.min_angle, min(self.max_angle, base_angle + wave))
                
            elif self.current_gesture == HandGesture.CONTEMPLATIVE:
                # Subtle breathing motion
                breath = 5 * math.sin(gesture_elapsed * 0.5 + i * 0.3)
                base_angle = max(self.min_angle, min(self.max_angle, base_angle + breath))
            
            positions[f"finger{i}"] = int(base_angle)
        
        return positions
    
    def get_current_gesture_description(self) -> str:
        """Get human-readable description of current hand gesture."""
        descriptions = {
            HandGesture.IDLE: "relaxed idle movement",
            HandGesture.CURIOUS: "pointing with interest",
            HandGesture.CONTEMPLATIVE: "thoughtful gentle motion",
            HandGesture.AGITATED: "restless rapid movements",
            HandGesture.FOCUSED: "still concentration",
            HandGesture.EXPRESSIVE: "animated gesturing",
            HandGesture.WITHDRAWN: "closed protective posture",
            HandGesture.AWAKENING: "slowly awakening motion"
        }
        return descriptions.get(self.current_gesture, "unknown gesture")
    
    def cleanup(self):
        """Clean up serial connection."""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
                if not self.clean_output:
                    print("[HAND] Serial connection closed")
            except Exception as e:
                if not self.clean_output:
                    print(f"[HAND] Error closing serial connection: {e}")
