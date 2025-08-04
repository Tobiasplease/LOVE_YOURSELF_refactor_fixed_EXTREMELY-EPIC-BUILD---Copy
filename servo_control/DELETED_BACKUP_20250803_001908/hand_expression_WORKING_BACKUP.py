"""
servo_control/hand_expression.py
-------------------------------
Consciousness-driven robotic hand expression system with continuous physics simulation.

Creates organic, ever-moving finger gestures that flow naturally with AI emotional states,
using spring-damper physics similar to the gaze system.
"""

import time
import random
import math
import serial
from typing import Dict, Tuple, Optional
from enum import Enum


class HandGesture(Enum):
    """Hand expression modes - now more like fluid states than discrete gestures."""
    NEUTRAL = "neutral"              # Gentle idle motion
    CURIOUS = "curious"              # Index finger explores, others follow
    EXPRESSIVE = "expressive"        # Dynamic, flowing movements
    CONTEMPLATIVE = "contemplative"  # Deep, slow breathing motions
    AGITATED = "agitated"           # Rapid, chaotic micro-movements
    WITHDRAWN = "withdrawn"         # Fingers curl inward, minimal motion
    FOCUSED = "focused"             # Still but alive, subtle tension


class HandExpressionController:
    """Physics-based continuous hand expression system."""
    
    def __init__(self, port: str = "COM3", baudrate: int = 9600, clean_output: bool = True):
        # Serial connection for hand communication
        self.port = port
        self.baudrate = baudrate
        self.clean_output = clean_output  # Controls verbose output
        self.serial_connection = None
        self._init_serial()
        
        # Hardware configuration
        self.num_fingers = 4
        self.finger_names = ["index", "middle", "ring", "pinky"]  # pins 8,9,10,11
        self.mirrored_fingers = [False, False, True, True]  # pins 10,11 are mirrored
        
        # Full dramatic range
        self.min_angle = 10        # Full extension (170° when mirrored)
        self.max_angle = 130       # Full curl (30° when mirrored)
        self.center_angle = 70     # Neutral middle position
        
        # Physics state for each finger
        self.finger_positions = [70.0, 70.0, 70.0, 70.0]  # Current positions
        self.finger_velocities = [0.0, 0.0, 0.0, 0.0]     # Current velocities
        self.finger_targets = [70.0, 70.0, 70.0, 70.0]    # Target positions
        
        # Physics parameters (mood will modulate these) - REFINED BASED ON FEEDBACK
        self.base_spring_force = 30.0     # Increased for more responsiveness
        self.base_friction = 1.0          # Reduced for smoother motion
        self.base_noise_amplitude = 12.0  # Increased for more expression
        self.base_frequency = 1.5         # Slower for smoother base movement
        
        # Current state
        self.current_mode = HandGesture.NEUTRAL
        self.physics_time = 0.0
        self.last_update_time = time.time()
        
        # Special state tracking for complex behaviors
        self.withdrawn_state = {"curling": False, "hold_time": 0, "pause_time": 0}
        self.expressive_pauses = {"finger_active": [True, True, True, True], "pause_timers": [0, 0, 0, 0]}
        self.focused_twitch_timers = [0, 0, 0, 0]  # For occasional twitches
        
        # Startle reaction system with cooldown
        self.is_startled = False
        self.startle_start_time = 0
        self.startle_duration = 2.5  # Extended for multiple pulses
        self.startle_targets = [70.0, 70.0, 70.0, 70.0]  # Random startle positions
        self.startle_oscillation_phase = 0.0  # For back-and-forth snapping
        self.startle_oscillation_speed = 15.0  # Speed of oscillation
        self.person_was_present = False  # Track previous state for face detection events
        self.last_startle_time = 0.0  # Track when last startle occurred
        self.startle_cooldown = 8.0   # 8 second cooldown between startles
        
        # Mood-influenced parameters (updated each cycle)
        self.spring_force = self.base_spring_force
        self.friction = self.base_friction
        self.noise_amplitude = self.base_noise_amplitude
        self.frequency_multiplier = 1.0
        self.movement_range = 30.0  # How far from center fingers can drift
        
        # Manual override system for interface control
        self.manual_override = False  # When True, disable automatic consciousness updates
        
        # Command throttling system (prevents overwhelming Arduino)
        self.last_command_time = 0.0
        self.min_command_interval = 0.05  # 20Hz max (instead of 100Hz)
        self.last_sent_positions = {}
        self.position_change_threshold = 3.0  # Only send if position changes >3 degrees
        
        if not self.clean_output:
            print(f"[HAND] Physics-based hand controller initialized")

    def _init_serial(self):
        """Initialize serial connection to Arduino hand controller."""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino boot time
            if not self.clean_output:
                print(f"✅ [HAND] Connected to hand controller on {self.port} at {self.baudrate} baud")
            # Send test command to verify connection
            test_command = "HAND,90,90,90,90\n"
            self.serial_connection.write(test_command.encode())
            if not self.clean_output:
                print(f"📤 [HAND] Test command sent: {test_command.strip()}")
        except Exception as e:
            if not self.clean_output:
                print(f"❌ [HAND] Failed to connect to {self.port}: {e}")
            self.serial_connection = None

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))

    def _send_hand_command(self, finger_positions: Dict[str, int]):
        """Send finger positions to Arduino hand controller with throttling."""
        if not self.serial_connection:
            return
        
        current_time = time.time()
        
        # Rate limiting: Don't send more than 20 commands per second
        if current_time - self.last_command_time < self.min_command_interval:
            return
        
        # Position change detection: Only send if positions changed significantly
        if self.last_sent_positions:
            position_changed = False
            for finger_name, new_pos in finger_positions.items():
                old_pos = self.last_sent_positions.get(finger_name, 0)
                if abs(new_pos - old_pos) > self.position_change_threshold:
                    position_changed = True
                    break
            
            if not position_changed:
                return  # Skip sending - positions haven't changed enough
        
        try:
            # Convert to list format expected by Arduino: "HAND,f0,f1,f2,f3\n"
            pos_list = [finger_positions.get(f"finger{i}", 70) for i in range(4)]
            command = f"HAND,{','.join(map(str, pos_list))}\n"
            self.serial_connection.write(command.encode())
            
            # Update tracking variables
            self.last_command_time = current_time
            self.last_sent_positions = finger_positions.copy()
            
            # Temporary debug - show what's being sent occasionally
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            self._debug_count += 1
            if self._debug_count % 20 == 0:  # Every 20 commands instead of 100
                print(f"📤 SERIAL: {command.strip()} (throttled)")
                
        except Exception as e:
            if not self.clean_output:
                print(f"[HAND] Serial write error: {e}")

    def update_from_consciousness(self, mood: float, novelty: float, boredom: float, 
                                person_present: bool, temporal_context: Dict = None) -> Dict[str, int]:
        """
        Main update function - continuously evolving physics simulation.
        Returns finger positions as dict: {"finger0": angle, "finger1": angle, ...}
        """
        # Skip automatic updates if manual override is enabled
        if self.manual_override:
            return {}  # Return empty dict - manual control is active
        
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        delta_time = self._clamp(delta_time, 0.001, 0.1)  # Prevent physics explosions
        self.last_update_time = current_time
        self.physics_time += delta_time

        # Check for new face detection (startle trigger) - DISABLED: Using machine.py cooldown system instead  
        # if person_present and not self.person_was_present:
        #     self._trigger_startle()
        self.person_was_present = person_present

        # Update mood-influenced physics parameters
        self._update_physics_parameters(mood, novelty, boredom, person_present)
        
        # Generate organic target positions based on current mode (or startle override)
        if self.is_startled:
            if not self.clean_output:
                print(f"[HAND] 🔥 STARTLE MODE ACTIVE - elapsed: {current_time - self.startle_start_time:.2f}s")
            self._update_startle(current_time, delta_time)
        else:
            self._generate_organic_targets(delta_time)
        
        # Run physics simulation
        finger_positions = self._update_physics(delta_time)
        
        # Send to hardware
        self._send_hand_command(finger_positions)
        
        return finger_positions

    def _update_physics_parameters(self, mood: float, novelty: float, boredom: float, person_present: bool):
        """Update physics parameters based on consciousness state."""
        
        # Determine current expression mode - ORDER MATTERS!
        if mood < 0.2 and novelty < 0.2:  # Very low mood + low novelty = withdrawn
            self.current_mode = HandGesture.WITHDRAWN
        elif boredom > 0.7:  # High boredom = agitated (but check withdrawn first)
            self.current_mode = HandGesture.AGITATED
        elif novelty > 0.7 and person_present:
            self.current_mode = HandGesture.CURIOUS
        elif mood > 0.8 and novelty > 0.6:
            self.current_mode = HandGesture.EXPRESSIVE
        elif boredom < 0.2 and novelty < 0.3:
            self.current_mode = HandGesture.FOCUSED
        elif mood > 0.5 and boredom < 0.4:
            self.current_mode = HandGesture.CONTEMPLATIVE
        else:
            self.current_mode = HandGesture.NEUTRAL

        # Modulate physics based on mode and mood - REFINED TUNING
        if self.current_mode == HandGesture.AGITATED:
            self.spring_force = self.base_spring_force * 2.0      # Reduced from 3.0
            self.friction = self.base_friction * 0.5              # Increased from 0.2
            self.noise_amplitude = self.base_noise_amplitude * 3.0 # Reduced from 6.0
            self.frequency_multiplier = 3.0                       # Reduced from 5.0
            self.movement_range = 60.0                            # Reduced from 80.0
            
        elif self.current_mode == HandGesture.EXPRESSIVE:
            self.spring_force = self.base_spring_force * 1.5      # Reduced from 2.5
            self.friction = self.base_friction * 0.8              # Increased from 0.4
            self.noise_amplitude = self.base_noise_amplitude * 2.0 # Reduced from 4.0
            self.frequency_multiplier = 1.8                       # Reduced from 3.0
            self.movement_range = 45.0                            # Reduced from 70.0
            
        elif self.current_mode == HandGesture.CURIOUS:
            self.spring_force = self.base_spring_force * 1.2      # Reduced from 2.0
            self.friction = self.base_friction * 1.0              # Increased from 0.5
            self.noise_amplitude = self.base_noise_amplitude * 1.5 # Reduced from 3.0
            self.frequency_multiplier = 1.2                       # Reduced from 2.5
            self.movement_range = 35.0                            # Reduced from 60.0
            
        elif self.current_mode == HandGesture.WITHDRAWN:
            self.spring_force = self.base_spring_force * 0.3      # Much reduced from 0.6
            self.friction = self.base_friction * 4.0              # Much increased from 2.0
            self.noise_amplitude = self.base_noise_amplitude * 0.1 # Much reduced from 0.3
            self.frequency_multiplier = 0.2                       # Much reduced from 0.4
            self.movement_range = 15.0                            # Reduced from 25.0
            
        elif self.current_mode == HandGesture.FOCUSED:
            self.spring_force = self.base_spring_force * 1.5      # More responsive for twitches
            self.friction = self.base_friction * 1.8              # Good damping
            self.noise_amplitude = self.base_noise_amplitude * 1.2 # More pronounced twitches
            self.frequency_multiplier = 0.8                       # Medium speed
            self.movement_range = 25.0                            # Medium range
            
        elif self.current_mode == HandGesture.CONTEMPLATIVE:
            self.spring_force = self.base_spring_force * 1.2      # More responsive
            self.friction = self.base_friction * 0.7              # Less damping for flow
            self.noise_amplitude = self.base_noise_amplitude * 0.5 # Less noise, more rhythm
            self.frequency_multiplier = 0.6                       # Breathing pace
            self.movement_range = 45.0                            # Wider breathing range
            
        else:  # NEUTRAL - WIDER RANGE BUT SMOOTHER AND SLOWER
            self.spring_force = self.base_spring_force * 0.8      # Slower response
            self.friction = self.base_friction * 1.5              # More smoothness  
            self.noise_amplitude = self.base_noise_amplitude * 1.2 # Slightly more expression
            self.frequency_multiplier = 0.7                       # Slower movement
            self.movement_range = 50.0                            # Wider range as requested

    def trigger_startle(self):
        """Trigger smooth clench-hold-release sequence with cooldown check."""
        current_time = time.time()
        
        # Check cooldown - prevent continuous reactions (increased to 10 seconds)
        if current_time - self.last_startle_time < 10.0:
            time_remaining = 10.0 - (current_time - self.last_startle_time)
            if not self.clean_output:
                print(f"[HAND] 🚫 Clench on cooldown - {time_remaining:.1f}s remaining")
            return
        
        if not self.clean_output:
            print("[HAND] ✊ SMOOTH CLENCH! Face detected - beginning clench sequence")
            print("[HAND] 🤲 CLENCH → HOLD → RELEASE → EASE")
        self.is_startled = True
        self.startle_start_time = current_time
        self.last_startle_time = current_time  # Update cooldown timer
        
        self._trigger_startle()
        
    def _trigger_startle(self):
        """Internal startle trigger - smooth clench-hold-release sequence."""
        # SMOOTH COORDINATED CLENCH: All fingers smoothly curl, hold, then release
        # Phase 1: Smooth clench (1.0s)
        # Phase 2: Hold closed (1.5s) 
        # Phase 3: Slow release (2.0s)
        # Phase 4: Ease to normal (1.5s)
        
        # Initialize clench state
        self.startle_phase = 'clench'  # clench → hold → release → ease
        self.startle_phase_start = self.startle_start_time
        self.startle_duration = 6.0  # Total sequence duration
        
        # Smooth clench - ALL fingers to closed position with gentle velocity
        for i in range(self.num_fingers):
            self.startle_targets[i] = 110  # Closed/clenched position (less extreme)
            self.finger_velocities[i] = 30  # Smooth, controlled velocity
        
        if not self.clean_output:
            print(f"[HAND] 🟢 Phase 1: CLENCH - smooth curl to closed position")

    def _update_startle(self, current_time: float, delta_time: float):
        """Update smooth clench-hold-release sequence."""
        elapsed = current_time - self.startle_start_time
        phase_elapsed = current_time - self.startle_phase_start
        
        if elapsed > self.startle_duration:
            # Clench sequence finished - return to normal behavior
            self.is_startled = False
            if not self.clean_output:
                print("[HAND] 🤲 Smooth clench sequence complete - resuming normal behavior")
            return
        
        # Phase transitions with smooth timing
        if self.startle_phase == 'clench' and phase_elapsed >= 1.0:
            # Phase 1 → 2: Clench complete, start hold
            self.startle_phase = 'hold'
            self.startle_phase_start = current_time
            if not self.clean_output:
                print(f"[HAND] 🟡 Phase 2: HOLD - maintaining clenched position")
            
        elif self.startle_phase == 'hold' and phase_elapsed >= 1.5:
            # Phase 2 → 3: Hold complete, start release
            self.startle_phase = 'release'
            self.startle_phase_start = current_time
            for i in range(self.num_fingers):
                self.startle_targets[i] = 50  # Open position
                self.finger_velocities[i] = 20  # Slow, gentle release
            if not self.clean_output:
                print(f"[HAND] 🟠 Phase 3: RELEASE - slow opening to relaxed position")
            
        elif self.startle_phase == 'release' and phase_elapsed >= 2.0:
            # Phase 3 → 4: Release complete, start easing
            self.startle_phase = 'ease'
            self.startle_phase_start = current_time
            for i in range(self.num_fingers):
                self.startle_targets[i] = 90  # Neutral position
                self.finger_velocities[i] = 15  # Very gentle ease
            if not self.clean_output:
                print(f"[HAND] 🔵 Phase 4: EASE - gentle return to neutral position")

    def _update_startle(self, current_time: float, delta_time: float):
        """Update smooth clench-hold-release sequence."""
        elapsed = current_time - self.startle_start_time
        phase_elapsed = current_time - self.startle_phase_start
        
        if elapsed > self.startle_duration:
            # Clench sequence finished - return to normal behavior
            self.is_startled = False
            if not self.clean_output:
                print("[HAND] 🤲 Smooth clench sequence complete - resuming normal behavior")
            return
        
        # Phase transitions with smooth timing
        if self.startle_phase == 'clench' and phase_elapsed >= 1.0:
            # Phase 1 → 2: Clench complete, start hold
            self.startle_phase = 'hold'
            self.startle_phase_start = current_time
            if not self.clean_output:
                print(f"[HAND] 🟡 Phase 2: HOLD - maintaining clenched position")
            
        elif self.startle_phase == 'hold' and phase_elapsed >= 1.5:
            # Phase 2 → 3: Hold complete, start release
            self.startle_phase = 'release'
            self.startle_phase_start = current_time
            for i in range(self.num_fingers):
                self.startle_targets[i] = 50  # Open position
                self.finger_velocities[i] = 20  # Slow, gentle release
            if not self.clean_output:
                print(f"[HAND] 🟠 Phase 3: RELEASE - slow opening to relaxed position")
            
        elif self.startle_phase == 'release' and phase_elapsed >= 2.0:
            # Phase 3 → 4: Release complete, start easing
            self.startle_phase = 'ease'
            self.startle_phase_start = current_time
            for i in range(self.num_fingers):
                self.startle_targets[i] = 90  # Neutral position
                self.finger_velocities[i] = 15  # Very gentle ease
            if not self.clean_output:
                print(f"[HAND] 🔵 Phase 4: EASE - gentle return to neutral position")
        
        # Apply smooth spring forces during clench sequence
        for i in range(self.num_fingers):
            if self.startle_phase == 'clench':
                # Smooth clench with moderate force
                spring_force = (self.startle_targets[i] - self.finger_positions[i]) * 25.0
                self.finger_velocities[i] += spring_force * delta_time
                self.finger_velocities[i] *= (1.0 - 2.0 * delta_time)  # Moderate friction
                
            elif self.startle_phase == 'hold':
                # Hold position with strong spring to maintain clench
                spring_force = (self.startle_targets[i] - self.finger_positions[i]) * 40.0
                self.finger_velocities[i] += spring_force * delta_time
                self.finger_velocities[i] *= (1.0 - 4.0 * delta_time)  # High friction for stability
                
            elif self.startle_phase == 'release':
                # Slow, controlled release
                spring_force = (self.startle_targets[i] - self.finger_positions[i]) * 15.0
                self.finger_velocities[i] += spring_force * delta_time
                self.finger_velocities[i] *= (1.0 - 3.0 * delta_time)  # High friction for slow movement
                
            elif self.startle_phase == 'ease':
                # Very gentle ease to neutral
                spring_force = (self.startle_targets[i] - self.finger_positions[i]) * 10.0
                self.finger_velocities[i] += spring_force * delta_time
                self.finger_velocities[i] *= (1.0 - 3.5 * delta_time)  # Very high friction for smoothness
        
        if not self.clean_output:
            print(f"[HAND] 🤲 {self.startle_phase.upper()} phase | Elapsed: {phase_elapsed:.1f}s | Positions: {[int(p) for p in self.finger_positions]}")

    def _generate_organic_targets(self, delta_time: float):
        """Generate continuously evolving target positions using organic patterns."""
        
        # Special behaviors for specific modes
        if self.current_mode == HandGesture.WITHDRAWN:
            self._generate_withdrawn_behavior(delta_time)
            return
        elif self.current_mode == HandGesture.EXPRESSIVE:
            self._generate_expressive_behavior(delta_time)
            return
        elif self.current_mode == HandGesture.FOCUSED:
            self._generate_focused_behavior(delta_time)
            return
        
        # Standard organic motion for other modes
        self._generate_standard_organic_motion(delta_time)

    def _generate_withdrawn_behavior(self, delta_time: float):
        """Special withdrawn behavior: slow curl up, hold, extend, pause, repeat."""
        state = self.withdrawn_state
        
        # State machine: curling -> holding -> extending -> pausing
        if not state["curling"] and state["pause_time"] <= 0:
            # Start curling phase
            state["curling"] = True
            state["hold_time"] = 0
            state["pause_time"] = 0
            
        if state["curling"]:
            # Slowly curl all fingers inward
            for i in range(self.num_fingers):
                self.finger_targets[i] = 110.0  # Curled position
            
            # Check if we've reached curl position (simple timeout)
            state["hold_time"] += delta_time
            if state["hold_time"] > 3.0:  # 3 seconds to curl
                state["curling"] = False
                state["hold_time"] = random.uniform(2.0, 4.0)  # Hold for 2-4 seconds
                
        elif state["hold_time"] > 0:
            # Hold curled position
            state["hold_time"] -= delta_time
            for i in range(self.num_fingers):
                self.finger_targets[i] = 110.0  # Stay curled
                
        else:
            # Extend and prepare for pause
            for i in range(self.num_fingers):
                self.finger_targets[i] = 50.0   # Extended position
            state["pause_time"] = random.uniform(3.0, 8.0)  # Pause 3-8 seconds
            state["pause_time"] -= delta_time

    def _generate_expressive_behavior(self, delta_time: float):
        """Special expressive behavior: sometimes still, sometimes bouncy, varied finger activity."""
        
        # Update pause timers and decide which fingers are active
        for i in range(self.num_fingers):
            if self.expressive_pauses["pause_timers"][i] > 0:
                self.expressive_pauses["pause_timers"][i] -= delta_time
                self.expressive_pauses["finger_active"][i] = False
            else:
                # Random chance to become active or pause
                if random.random() < 0.02:  # 2% chance per frame to change state
                    if self.expressive_pauses["finger_active"][i]:
                        # Go to pause
                        self.expressive_pauses["finger_active"][i] = False
                        self.expressive_pauses["pause_timers"][i] = random.uniform(1.0, 4.0)
                    else:
                        # Become active
                        self.expressive_pauses["finger_active"][i] = True
        
        # Generate motion only for active fingers
        for i in range(self.num_fingers):
            if self.expressive_pauses["finger_active"][i]:
                # Bouncy, smaller range movement
                freq = self.frequency_multiplier
                t = self.physics_time
                
                # Bouncy wave (more spring-like)
                bounce = math.sin(t * freq * 2.0 + i * 0.8) ** 2 * self.movement_range * 0.4
                variation = math.sin(t * freq * 0.6 + i * 1.2) * self.movement_range * 0.2
                noise = (random.random() - 0.5) * self.noise_amplitude * 0.5
                
                target = 70.0 + bounce + variation + noise  # Around neutral
                self.finger_targets[i] = self._clamp(target, self.min_angle, self.max_angle)
            else:
                # Stay still at current position (or drift slowly to neutral)
                current = self.finger_targets[i]
                drift_to_neutral = (70.0 - current) * 0.01  # Very slow drift
                self.finger_targets[i] = current + drift_to_neutral

    def _generate_focused_behavior(self, delta_time: float):
        """Special focused behavior: mostly still with occasional pronounced finger twitches."""
        
        # Update twitch timers
        for i in range(self.num_fingers):
            self.focused_twitch_timers[i] -= delta_time
            
            if self.focused_twitch_timers[i] <= 0:
                # Random chance for a pronounced twitch
                if random.random() < 0.01:  # 1% chance per frame (increased from 0.5%)
                    # More pronounced twitch movement
                    twitch_amount = random.uniform(-25, 25)  # Increased from -15,15
                    target = 70.0 + twitch_amount
                    self.finger_targets[i] = self._clamp(target, self.min_angle, self.max_angle)
                    self.focused_twitch_timers[i] = random.uniform(1.5, 6.0)  # More frequent
                else:
                    # Gentle drift back to neutral
                    current = self.finger_targets[i]
                    drift = (70.0 - current) * 0.03  # Slightly faster drift
                    self.finger_targets[i] = current + drift

    def _generate_standard_organic_motion(self, delta_time: float):
        """Standard organic motion for modes that don't need special behavior."""
        
        # Base centers for each finger (can be different per mode)
        if self.current_mode == HandGesture.CURIOUS:
            base_centers = [40.0, 80.0, 90.0, 100.0]  # Index extended, others curled
        else:
            base_centers = [70.0, 70.0, 70.0, 70.0]     # Neutral center
        
        # Generate organic motion for each finger
        for i in range(self.num_fingers):
            # Multiple sine waves for organic, non-repetitive motion
            freq = self.frequency_multiplier
            t = self.physics_time
            
            # Primary wave (main rhythm)
            wave1 = math.sin(t * freq * 0.7 + i * 1.2) * self.movement_range * 0.6
            
            # Secondary wave (adds complexity)
            wave2 = math.sin(t * freq * 1.3 + i * 0.8) * self.movement_range * 0.3
            
            # Tertiary wave (micro-variations)
            wave3 = math.sin(t * freq * 2.1 + i * 0.5) * self.movement_range * 0.1
            
            # Noise component (random micro-movements)
            noise = (random.random() - 0.5) * self.noise_amplitude
            
            # Combine all components
            target = base_centers[i] + wave1 + wave2 + wave3 + noise
            
            # Clamp to valid servo range
            self.finger_targets[i] = self._clamp(target, self.min_angle, self.max_angle)

    def _update_physics(self, delta_time: float) -> Dict[str, int]:
        """Run spring-damper physics simulation for each finger."""
        finger_positions = {}
        
        for i in range(self.num_fingers):
            # Calculate spring force toward target
            force = (self.finger_targets[i] - self.finger_positions[i]) * self.spring_force
            
            # Apply force to velocity
            self.finger_velocities[i] += force * delta_time
            
            # Apply friction (damping)
            self.finger_velocities[i] *= (1.0 - self.friction * delta_time)
            
            # Update position (with startle debug)
            old_pos = self.finger_positions[i]
            self.finger_positions[i] += self.finger_velocities[i] * delta_time
            
            # Debug startle physics
            if self.is_startled and not self.clean_output:
                print(f"[HAND] 🔥 Finger {i}: {old_pos:.1f}→{self.finger_positions[i]:.1f} (vel:{self.finger_velocities[i]:.1f}, force:{force:.1f})")
            
            # Clamp to servo limits
            self.finger_positions[i] = self._clamp(self.finger_positions[i], self.min_angle, self.max_angle)
            
            # Store integer position for servo
            finger_positions[f"finger{i}"] = int(round(self.finger_positions[i]))
        
        return finger_positions

    def get_current_gesture_description(self) -> str:
        """Get human-readable description of current hand expression."""
        descriptions = {
            HandGesture.NEUTRAL: "gentle organic movement",
            HandGesture.CURIOUS: "exploring with index finger",
            HandGesture.EXPRESSIVE: "flowing dynamic gestures",
            HandGesture.CONTEMPLATIVE: "deep rhythmic breathing",
            HandGesture.AGITATED: "rapid chaotic micro-movements",
            HandGesture.WITHDRAWN: "curled inward with minimal motion",
            HandGesture.FOCUSED: "still concentration with subtle life"
        }
        return descriptions.get(self.current_mode, "unknown gesture")

    def close(self):
        """Alias for cleanup() for compatibility."""
        self.cleanup()

    def cleanup(self):
        """Clean shutdown of hand controller."""
        if self.serial_connection:
            try:
                self.serial_connection.close()
                print("[HAND] Serial connection closed")
            except Exception as e:
                print(f"[HAND] Error closing serial: {e}")
    
    # ===== MANUAL CONTROL METHODS FOR INTERFACE =====
    
    def set_hand_positions(self, positions: list):
        """
        Direct manual control method for the hand interface.
        positions: list of 4 angles [index, middle, ring, pinky] (0-180 degrees)
        """
        if len(positions) != 4:
            raise ValueError("Must provide exactly 4 positions for 4 fingers")
        
        # Convert list to finger dictionary using finger0, finger1, etc. format
        finger_dict = {}
        for i, angle in enumerate(positions):
            # Clamp to safe range
            clamped_angle = self._clamp(angle, self.min_angle, self.max_angle)
            finger_dict[f"finger{i}"] = int(clamped_angle)
        
        # Send directly to hardware (now with throttling!)
        self._send_hand_command(finger_dict)
    
    def enable_manual_override(self):
        """Enable manual override mode - stops automatic consciousness updates."""
        self.manual_override = True
        if not self.clean_output:
            print("[HAND] Manual override ENABLED - automatic control disabled")
    
    def disable_manual_override(self):
        """Disable manual override mode - resumes automatic consciousness updates."""
        self.manual_override = False
        if not self.clean_output:
            print("[HAND] Manual override DISABLED - automatic control resumed")
