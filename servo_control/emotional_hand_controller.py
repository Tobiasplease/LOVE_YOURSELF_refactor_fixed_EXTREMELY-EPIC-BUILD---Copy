#!/usr/bin/env python3
"""
Emotional Hand Controller - Autonomous Integration
================================================

Autonomous hand control that reads mood states from the main script.
Similar pattern to breathing.py - no UI, just pure emotional movement.

Integration with machine.py:
- Reads current mood, novelty, boredom from main script
- Maps emotional states to movement patterns
- Smooth transitions between emotional states
- Maintains learned movement signatures for each emotion

Author: Autonomous Hand Control System
"""
import time
import math
import random
import threading
from typing import Dict, Optional

# Import the movement patterns from your working interface
try:
    from servo_control.hand_expression import HandExpressionController
    HAND_CONTROLLER_AVAILABLE = True
except ImportError:
    HAND_CONTROLLER_AVAILABLE = False

# Import mood state mapping (matches captioner/prompts.py)
def map_mood_to_emotional_state(mood: float, novelty: float, boredom: float) -> str:
    """Map numerical mood values to emotional state names (matches main script)."""
    if mood > 0.7:
        return 'energized_engaged'  # "energized and deeply engaged"
    elif mood > 0.5:
        return 'alert_curious'      # "alert and curious"  
    elif mood > 0.3:
        return 'calm_observant'     # "calm and observant"
    elif mood > -0.1:
        return 'calm_observant'     # "neutral and watchful" -> use calm
    elif mood > -0.5:
        return 'quiet_detached'     # "quiet and detached"
    else:
        return 'withdrawn_distant'  # "withdrawn and distant"


class AutonomousHandController:
    """Autonomous hand controller that follows emotional state from main script."""
    
    def __init__(self):
        self.hand_controller: Optional[HandExpressionController] = None
        self.connected = False
        
        # Hand state
        self.num_fingers = 4
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_targets = [90.0] * self.num_fingers
        
        # Emotional state tracking
        self.current_emotion = 'calm_observant'
        self.last_emotion = 'calm_observant'
        self.emotion_change_time = 0.0
        self.transition_duration = 2.0  # 2 seconds to transition between emotions
        
        # Person detection and freeze behavior
        self.last_person_detection_time = 0.0
        self.person_detection_cooldown = 60.0  # 60 seconds cooldown as requested
        self.is_frozen = False
        self.freeze_start_time = 0.0
        self.freeze_duration = 0.0
        self.freeze_position = (0.5, 0.5)
        
        # Idle and conservation mode
        self.no_person_start_time = 0.0
        self.idle_threshold = 120.0  # 2 minutes before entering idle mode
        self.idle_mode = False
        self.idle_intensity = 0.2  # 20% intensity when idle
        self.servo_conservation_mode = False
        
        # Movement parameters for each emotional state (from working interface)
        self.emotional_patterns = {
            'energized_engaged': {
                'cursor_sensitivity': 4.0,
                'wave_strength': 3.0,
                'movement_speed': 0.8,
                'jitter_amount': 0.02,
                'boundary_preference': 'exploratory',
                'base_intensity': 1.0
            },
            'alert_curious': {
                'cursor_sensitivity': 3.2,
                'wave_strength': 2.2,
                'movement_speed': 0.6,
                'jitter_amount': 0.015,
                'boundary_preference': 'moderate',
                'base_intensity': 0.8
            },
            'calm_observant': {
                'cursor_sensitivity': 2.5,
                'wave_strength': 1.5,
                'movement_speed': 0.3,
                'jitter_amount': 0.008,
                'boundary_preference': 'centered',
                'base_intensity': 0.6
            },
            'quiet_detached': {
                'cursor_sensitivity': 1.8,
                'wave_strength': 1.0,
                'movement_speed': 0.2,
                'jitter_amount': 0.005,
                'boundary_preference': 'withdrawn',
                'base_intensity': 0.4
            },
            'withdrawn_distant': {
                'cursor_sensitivity': 1.2,
                'wave_strength': 0.6,
                'movement_speed': 0.1,
                'jitter_amount': 0.002,
                'boundary_preference': 'minimal',
                'base_intensity': 0.2
            }
        }
        
        # Autonomous movement state
        self.virtual_cursor_x = 0.5
        self.virtual_cursor_y = 0.5
        self.movement_direction = 0.0
        self.movement_phase = 0.0
        self.last_update_time = time.time()
        
        # Initialize connection
        self.initialize_connection()
    
    def initialize_connection(self):
        """Initialize connection to hand controller (like breathing.py does)."""
        if not HAND_CONTROLLER_AVAILABLE:
            print("⚠️ Hand controller not available - autonomous hand control in simulation mode")
            return
        
        try:
            self.hand_controller = HandExpressionController(
                port="COM3",
                baudrate=9600,
                clean_output=True
            )
            if self.hand_controller.serial_connection:
                self.hand_controller.enable_manual_override()
                self.connected = True
                print("✅ Autonomous hand control connected to COM3")
            else:
                print("❌ Failed to connect autonomous hand control to COM3")
        except Exception as e:
            print(f"❌ Autonomous hand control connection error: {e}")
    
    def update_emotional_state(self, mood: float, novelty: float, boredom: float):
        """Update emotional state based on mood values from main script."""
        new_emotion = map_mood_to_emotional_state(mood, novelty, boredom)
        
        if new_emotion != self.current_emotion:
            print(f"🎭 Hand emotion transition: {self.current_emotion} → {new_emotion}")
            self.last_emotion = self.current_emotion
            self.current_emotion = new_emotion
            self.emotion_change_time = time.time()
    
    def handle_person_detection(self, person_present: bool):
        """Handle person detection with freeze behavior and cooldown."""
        current_time = time.time()
        
        if person_present:
            # Person detected
            if not self.is_frozen:
                # Check cooldown
                time_since_last = current_time - self.last_person_detection_time
                if time_since_last >= self.person_detection_cooldown:
                    # Trigger freeze
                    self.trigger_freeze(current_time)
                # else: Still on cooldown, ignore this detection
            
            # Reset no-person timer
            self.no_person_start_time = 0.0
            if self.idle_mode:
                print("👤 Person detected - exiting idle mode")
                self.idle_mode = False
        else:
            # No person detected
            if self.no_person_start_time == 0.0:
                self.no_person_start_time = current_time
            elif not self.idle_mode:
                # Check if we should enter idle mode
                time_alone = current_time - self.no_person_start_time
                if time_alone >= self.idle_threshold:
                    print(f"😴 No person for {time_alone:.0f}s - entering idle mode")
                    self.idle_mode = True
    
    def trigger_freeze(self, current_time: float):
        """Trigger freeze response with random duration 1-4 seconds."""
        self.freeze_position = (self.virtual_cursor_x, self.virtual_cursor_y)
        self.freeze_duration = random.uniform(1.0, 4.0)  # 1-4 seconds as requested
        self.freeze_start_time = current_time
        self.is_frozen = True
        self.last_person_detection_time = current_time
        
        print(f"❄️ FREEZE triggered for {self.freeze_duration:.1f}s (cooldown: {self.person_detection_cooldown}s)")
    
    def update_freeze_state(self, current_time: float):
        """Update freeze state and handle thawing."""
        if not self.is_frozen:
            return
        
        elapsed = current_time - self.freeze_start_time
        if elapsed >= self.freeze_duration:
            # End freeze
            self.is_frozen = False
            print("🔄 Freeze ended - resuming movement")
        else:
            # Maintain frozen position
            self.virtual_cursor_x = self.freeze_position[0]
            self.virtual_cursor_y = self.freeze_position[1]
    
    def calculate_intensity_multiplier(self, novelty: float, boredom: float, person_present: bool):
        """Calculate movement intensity based on engagement and conservation needs."""
        base_intensity = 1.0
        
        # Person present boosts intensity
        if person_present:
            base_intensity *= 1.2
        
        # Idle mode reduces intensity significantly for servo conservation
        if self.idle_mode:
            base_intensity *= self.idle_intensity
        
        # Novelty increases intensity (curiosity drives movement)
        novelty_boost = 1.0 + (novelty * 0.5)
        base_intensity *= novelty_boost
        
        # Boredom can increase OR decrease intensity depending on level
        if boredom > 0.7:
            # High boredom = frustration = more twitchy movement
            boredom_factor = 1.0 + ((boredom - 0.7) * 1.5)
        elif boredom > 0.4:
            # Medium boredom = restlessness = slightly more movement  
            boredom_factor = 1.0 + ((boredom - 0.4) * 0.3)
        else:
            # Low boredom = content = normal movement
            boredom_factor = 1.0
        
        base_intensity *= boredom_factor
        
        # Clamp to reasonable range
        return max(0.05, min(2.0, base_intensity))
    
    def get_transition_blend(self) -> float:
        """Get blend factor for smooth emotional transitions."""
        if self.current_emotion == self.last_emotion:
            return 1.0
        
        elapsed = time.time() - self.emotion_change_time
        if elapsed >= self.transition_duration:
            return 1.0
        
        # Smooth easing function
        progress = elapsed / self.transition_duration
        return 0.5 * (1.0 - math.cos(progress * math.pi))
    
    def blend_patterns(self, pattern1: dict, pattern2: dict, blend: float) -> dict:
        """Blend between two emotional patterns for smooth transitions."""
        result = {}
        for key in pattern1:
            if isinstance(pattern1[key], (int, float)):
                result[key] = pattern1[key] * (1.0 - blend) + pattern2[key] * blend
            else:
                result[key] = pattern2[key] if blend > 0.5 else pattern1[key]
        return result
    
    def update_autonomous_movement(self, delta: float, intensity_multiplier: float):
        """Update virtual cursor position autonomously based on emotional state."""
        if self.is_frozen:
            return  # No movement during freeze
        
        current_time = time.time()
        
        # Get blended pattern for smooth transitions
        blend = self.get_transition_blend()
        current_pattern = self.emotional_patterns[self.current_emotion]
        
        if blend < 1.0:
            last_pattern = self.emotional_patterns[self.last_emotion]
            current_pattern = self.blend_patterns(last_pattern, current_pattern, blend)
        
        # Apply intensity multiplier to movement parameters
        movement_speed = current_pattern['movement_speed'] * intensity_multiplier
        wave_strength = current_pattern['wave_strength'] * intensity_multiplier
        jitter_amount = current_pattern['jitter_amount'] * intensity_multiplier
        
        # Update movement phase
        self.movement_phase += delta * movement_speed
        
        # Generate autonomous movement based on emotional state
        if current_pattern['boundary_preference'] == 'exploratory':
            # Large, bold movements
            base_x = 0.5 + 0.3 * math.sin(self.movement_phase * 0.7)
            base_y = 0.5 + 0.25 * math.cos(self.movement_phase * 0.9)
        elif current_pattern['boundary_preference'] == 'moderate':
            # Medium-range movements
            base_x = 0.5 + 0.2 * math.sin(self.movement_phase * 0.8)
            base_y = 0.5 + 0.15 * math.cos(self.movement_phase * 1.1)
        elif current_pattern['boundary_preference'] == 'centered':
            # Small, controlled movements around center
            base_x = 0.5 + 0.1 * math.sin(self.movement_phase * 1.0)
            base_y = 0.5 + 0.08 * math.cos(self.movement_phase * 1.3)
        elif current_pattern['boundary_preference'] == 'withdrawn':
            # Slow drift with minimal movement
            base_x = 0.4 + 0.1 * math.sin(self.movement_phase * 0.3)
            base_y = 0.4 + 0.08 * math.cos(self.movement_phase * 0.4)
        else:  # minimal
            # Almost no movement
            base_x = 0.35 + 0.05 * math.sin(self.movement_phase * 0.2)
            base_y = 0.35 + 0.03 * math.cos(self.movement_phase * 0.25)
        
        # Add jitter/micro-movements
        jitter_x = random.uniform(-jitter_amount, jitter_amount)
        jitter_y = random.uniform(-jitter_amount, jitter_amount)
        
        # Update virtual cursor position
        self.virtual_cursor_x = max(0.0, min(1.0, base_x + jitter_x))
        self.virtual_cursor_y = max(0.0, min(1.0, base_y + jitter_y))
    
    def calculate_finger_targets(self):
        """Calculate finger targets from virtual cursor (same logic as working interface)."""
        # Use the same wave-based calculation as your working interface
        current_pattern = self.emotional_patterns[self.current_emotion]
        wave_strength = current_pattern['wave_strength']
        gravity_width = 0.4  # Fixed value from working interface
        default_pos = 90.0   # Center position
        sensitivity = current_pattern['cursor_sensitivity']
        
        for i in range(self.num_fingers):
            # Same condensed mapping as working interface (25%-75% of screen)
            condensed_start = 0.25
            condensed_width = 0.5
            finger_x = condensed_start + ((i + 0.5) / self.num_fingers) * condensed_width
            
            # Calculate influence
            distance = abs(self.virtual_cursor_x - finger_x)
            if distance < gravity_width:
                influence = 1.0 - (distance / gravity_width)
                y_offset = (self.virtual_cursor_y - 0.5) * sensitivity * wave_strength * influence
                target = default_pos + (y_offset * 45.0)  # ±45 degrees range
                self.finger_targets[i] = max(0, min(180, target))
            else:
                self.finger_targets[i] = default_pos
        
        # Direct control - immediate response (like working interface)
        self.finger_positions = self.finger_targets.copy()
    
    def send_to_controller(self):
        """Send positions to hand controller."""
        if not self.connected or not self.hand_controller:
            return
        
        try:
            positions = [int(pos) for pos in self.finger_positions]
            self.hand_controller.set_hand_positions(positions)
        except Exception as e:
            print(f"❌ Autonomous hand control error: {e}")


# Global instance for integration with machine.py
_autonomous_hand_controller = None

def update_hand_position(current_mood: float, novelty: float, boredom: float, 
                        person_present: bool, delta: float, servo_controller=None):
    """
    Main integration function - called from machine.py like breathing.py
    
    Args:
        current_mood: Current mood value (-1.0 to 1.0)
        novelty: Novelty level (0.0 to 1.0) 
        boredom: Boredom level (0.0 to 1.0)
        person_present: Whether person is detected
        delta: Time since last update
        servo_controller: Servo controller instance (for compatibility)
    
    Returns:
        dict: Current hand positions and state info
    """
    global _autonomous_hand_controller
    
    # Initialize on first call
    if _autonomous_hand_controller is None:
        _autonomous_hand_controller = AutonomousHandController()
        print("🤖 Autonomous hand control system initialized")
    
    current_time = time.time()
    
    # Handle person detection with freeze behavior
    _autonomous_hand_controller.handle_person_detection(person_present)
    
    # Update freeze state
    _autonomous_hand_controller.update_freeze_state(current_time)
    
    # Update emotional state based on mood
    _autonomous_hand_controller.update_emotional_state(current_mood, novelty, boredom)
    
    # Calculate intensity multiplier for conservation and engagement
    intensity_multiplier = _autonomous_hand_controller.calculate_intensity_multiplier(
        novelty, boredom, person_present
    )
    
    # Update autonomous movement (respects freeze state internally)
    _autonomous_hand_controller.update_autonomous_movement(delta, intensity_multiplier)
    
    # Calculate finger positions
    _autonomous_hand_controller.calculate_finger_targets()
    
    # Send to hardware
    _autonomous_hand_controller.send_to_controller()
    
    # Return comprehensive state info
    state_info = {
        'finger_positions': _autonomous_hand_controller.finger_positions,
        'emotional_state': _autonomous_hand_controller.current_emotion,
        'virtual_cursor': (_autonomous_hand_controller.virtual_cursor_x, _autonomous_hand_controller.virtual_cursor_y),
        'connected': _autonomous_hand_controller.connected,
        'is_frozen': _autonomous_hand_controller.is_frozen,
        'idle_mode': _autonomous_hand_controller.idle_mode,
        'intensity_multiplier': intensity_multiplier,
        'freeze_cooldown_remaining': max(0, _autonomous_hand_controller.person_detection_cooldown - 
                                       (current_time - _autonomous_hand_controller.last_person_detection_time))
    }
    
    return state_info


# Cleanup function for graceful shutdown
def cleanup_hand_controller():
    """Cleanup function for graceful shutdown."""
    global _autonomous_hand_controller
    if _autonomous_hand_controller and _autonomous_hand_controller.hand_controller:
        _autonomous_hand_controller.hand_controller.cleanup()
        print("🤖 Autonomous hand control cleaned up")
