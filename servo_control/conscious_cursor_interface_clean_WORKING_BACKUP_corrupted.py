#!/usr/bin/env python3
"""
Conscious Cursor Interface - CLEAN BASELINE
===========================================

Simplified hand control with 5 core emotional states.
Focus: Working cursor→servo control + simple emotional switching.

Core Features:
- Direct cursor→4 servo control (WORKING)
- Physics-based movement with parameter control (WORKING)  
- 5 basic emotional states with manual switching
- Simple movement logging for each emotional state
- Clean baseline for future integration

Author: Simplified Emotional Control System
"""
import tkinter as tk
from tkinter import ttk
import time
import math
import random
import os
import json
import traceback
from typing import Optional

# Import hand controller - should work since it was working before
try:
    from hand_expression import HandExpressionController
    HAND_CONTROLLER_AVAILABLE = True
    print("✅ Hand controller available")
except ImportError:
    print("⚠️ Hand controller not available - simulation mode")
    HAND_CONTROLLER_AVAILABLE = False

# Import the working Markov system
try:
    from markov_movement import MarkovMovementSystem
    MARKOV_SYSTEM_AVAILABLE = True
    print("✅ Markov movement system available")
except ImportError:
    print("⚠️ Markov movement system not available - using fallback")
    MARKOV_SYSTEM_AVAILABLE = False


class EmotionalState:
    """Simple emotional state with movement characteristics."""
    def __init__(self, name, mood_factor=0.0, energy_factor=0.5, focus_factor=0.5):
        self.name = name
        self.mood_factor = mood_factor      # -1.0 to 1.0 (sad to happy)
        self.energy_factor = energy_factor  # 0.0 to 1.0 (calm to energetic)
        self.focus_factor = focus_factor    # 0.0 to 1.0 (scattered to focused)
        self.movement_data = []             # Log of movements in this state
    
    def get_movement_params(self):
        """Get movement parameters based on emotional factors."""
        # Base parameters - these work well from your current system
        base_spring = 500.0
        base_damping = 0.1
        base_velocity = 1000.0
        base_sensitivity = 3.0
        
        # Emotional modulation
        spring_mod = 1.0 + (self.energy_factor * 0.5)  # More energetic = springier
        damping_mod = 1.0 - (self.focus_factor * 0.3)  # More focused = less damping
        velocity_mod = 1.0 + (self.energy_factor * 0.8) # More energetic = faster
        sensitivity_mod = 1.0 + (self.mood_factor * 0.2) # Happier = more sensitive
        
        return {
            'spring_force': base_spring * spring_mod,
            'damping': base_damping * damping_mod,
            'max_velocity': base_velocity * velocity_mod,
            'cursor_sensitivity': base_sensitivity * sensitivity_mod
        }


class CleanCursorInterface:
    """Clean baseline cursor interface with 5 emotional states."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 Clean Emotional Hand Control - Baseline")
        
        # FIXED WINDOW SIZE - No more resizing!
        self.root.geometry("650x650")  # Fixed square dimensions
        self.root.resizable(False, False)  # Disable resizing completely
        self.root.minsize(650, 650)  # Enforce minimum
        self.root.maxsize(650, 650)  # Enforce maximum
        # Color scheme similar to original
        self.colors = {
            'bg_main': '#FFE4E6',        # Light pink background
            'bg_frame': '#F8BBD9',       # Medium pink for frames
            'bg_accent': '#E4A5FF',      # Light purple accent
            'text_main': '#4A4A4A',      # Dark gray text
            'text_accent': '#8B5CF6',    # Purple accent text
            'button_bg': '#DDA0DD',      # Plum button background
            'button_active': '#FF69B4',  # Hot pink for active buttons
            'success': '#98FB98',        # Light green for success
            'warning': '#FFB347',        # Peach for warnings
            'error': '#FFB6C1'          # Light pink for errors
        }
        
        # Apply main background color
        self.root.configure(bg=self.colors['bg_main'])
        
        # FIXED SCROLLABLE INTERFACE - No more auto-resizing
        # Create main canvas with scrollbar for scrollable content
        self.main_canvas = tk.Canvas(self.root, bg=self.colors['bg_main'], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg=self.colors['bg_main'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Hand controller - FIXED CONNECTION LOGIC
        self.hand_controller: Optional[HandExpressionController] = None
        self.connected = False
        
        # Direct control state - clean and simple!
        self.num_fingers = 4
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_targets = [90.0] * self.num_fingers
        
        print(f"🎯 Initialized finger positions: {self.finger_positions}")
        print(f"🎯 Initialized finger targets: {self.finger_targets}")
        
        # Wave control parameters - the good stuff!
        self.cursor_sensitivity = tk.DoubleVar(value=3.0)
        
        # Wave control parameters - KEEP - working values for smooth wave-based control
        self.wave_strength = tk.DoubleVar(value=2.0)
        self.gravity_width = tk.DoubleVar(value=0.4)
        self.default_position = tk.DoubleVar(value=90.0)
        
        # Control toggles - simplified
        self.reverse_vertical = tk.BooleanVar(value=False)
        
        # Mouse tracking (KEEP - this works)
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        
        # Animation state (KEEP - this works)
        self.running = False
        self.last_time = time.time()
        self.last_send_time = 0
        self.send_interval = 0.016  # 60 Hz
        self.position_threshold = 1.0
        
        # 5 CORE EMOTIONAL STATES - Simple and clear
        self.emotional_states = {
            'neutral': EmotionalState('neutral', mood_factor=0.0, energy_factor=0.5, focus_factor=0.5),
            'happy': EmotionalState('happy', mood_factor=0.8, energy_factor=0.8, focus_factor=0.6),
            'sad': EmotionalState('sad', mood_factor=-0.7, energy_factor=0.2, focus_factor=0.4),
            'excited': EmotionalState('excited', mood_factor=0.6, energy_factor=1.0, focus_factor=0.3),
            'focused': EmotionalState('focused', mood_factor=0.2, energy_factor=0.4, focus_factor=1.0)
        }
        
        self.current_emotional_state = 'neutral'
        self.logging_movement = False
        
        # Recording/playback state - VECTOR-BASED!
        self.recording = False
        self.playing_back = False
        self.recorded_movements = {}  # emotion_name -> list of movements (positions for exact playback)
        self.recorded_vectors = {}   # emotion_name -> list of movement vectors (for generation)
        self.playback_start_time = 0
        self.current_playback = []
        self.record_start_time = 0
        
        # Vector recording state
        self.last_record_pos = (0.5, 0.5)
        self.last_record_time = 0
        
        # Generative playback state - MARKOV-BASED!
        self.generating = False
        self.generation_start_time = 0
        self.learned_patterns = {}  # emotion_name -> vector pattern analysis
        self.vector_generator = None  # Active vector generator
        
        # Initialize Markov movement system
        if MARKOV_SYSTEM_AVAILABLE:
            self.markov_system = MarkovMovementSystem()
            print("🧬 Markov movement system initialized")
        else:
            self.markov_system = None
            print("⚠️ Markov system not available - generation disabled")
        
        # Advanced generative state
        self.gen_direction = 0
        self.gen_persistence = 0
        self.gen_macro_target_x = 0.5
        self.gen_macro_target_y = 0.5
        self.gen_macro_change_time = 0
        self.gen_favorite_positions = []  # Positions we like to return to
        self.gen_rhythm_phase = 0
        self.gen_complexity_level = 0
        
        self.setup_ui()
        self.start_control_loop()
        
        print("🎯 Clean Emotional Hand Control initialized")
        print("🎮 Direct wave-based cursor→servo control ready")
        print("😊 5 emotional states available for testing")
        print(f"📐 FIXED canvas dimensions: 480x200 (no more resizing)")
        print(f"🎯 Condensed control area: 25%-75% of canvas width for precise movement")
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling in the interface."""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def setup_ui(self):
        """Create clean, focused UI."""
        
        # === CONNECTION FRAME ===
        conn_frame = ttk.LabelFrame(self.scrollable_frame, text="🔌 Connection")
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect to Hand Controller", 
                                     command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(conn_frame, text="❌ Disconnected", width=20)
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # === EMOTIONAL STATE CONTROL ===
        emotion_frame = ttk.LabelFrame(self.scrollable_frame, text="😊 Emotional State Control")
        emotion_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Current state display
        self.current_state_label = ttk.Label(emotion_frame, text=f"Current: {self.current_emotional_state}", 
                                           font=('Arial', 12, 'bold'))
        self.current_state_label.pack(pady=5)
        
        # Emotion buttons
        emotion_buttons_frame = ttk.Frame(emotion_frame)
        emotion_buttons_frame.pack(pady=5)
        
        for emotion_name in self.emotional_states.keys():
            btn = ttk.Button(emotion_buttons_frame, text=emotion_name.title(), width=10,
                           command=lambda e=emotion_name: self.switch_emotional_state(e))
            btn.pack(side=tk.LEFT, padx=5)
        
        # Movement recording/playback control - SIMPLE AND DIRECT!
        record_frame = ttk.Frame(emotion_frame)
        record_frame.pack(pady=5)
        
        # Record button
        self.record_btn = ttk.Button(record_frame, text="🎬 Record Movement (45s)", 
                                   command=self.start_recording, width=20)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        # Playback button
        self.playback_btn = ttk.Button(record_frame, text="▶️ Play Back", 
                                     command=self.start_playback, width=15)
        self.playback_btn.pack(side=tk.LEFT, padx=5)
        
        # MARKOV CHAIN playbook button - THE REAL THING!
        self.generate_btn = ttk.Button(record_frame, text="� Markov Generate", 
                                     command=self.start_markov_generation, width=18)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Status - MORE INFORMATIVE
        self.record_status = ttk.Label(record_frame, text="Ready to record", foreground="gray")
        self.record_status.pack(side=tk.LEFT, padx=10)
        
        # Vector status display
        self.vector_status = ttk.Label(record_frame, text="No vectors recorded", foreground="gray", font=("Arial", 8))
        self.vector_status.pack(side=tk.LEFT, padx=5)
        
        # Recording state
        self.recording = False
        self.playing_back = False
        self.recorded_movements = {}  # emotion_name -> list of movements
        self.playback_start_time = 0
        self.current_playback = []
        
        # === HAND CONTROL AREA - FIXED DIMENSIONS ===
        control_frame = ttk.LabelFrame(self.scrollable_frame, text="🎯 Hand Control Area")
        control_frame.pack(fill=tk.X, padx=10, pady=5)  # Changed from fill=tk.BOTH to fill=tk.X
        
        # FIXED CANVAS SIZE - No more resizing, consistent proportions!
        self.canvas = tk.Canvas(control_frame, bg="black", height=200, width=480)  # Much smaller and fixed
        self.canvas.pack(padx=5, pady=5)  # Removed fill and expand
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        
        # === CONTROL MODES - Simplified without physics ===
        mode_frame = ttk.LabelFrame(self.scrollable_frame, text="🎛️ Control Modes")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        reverse_cb = ttk.Checkbutton(mode_frame, text="🔄 Reverse Vertical", 
                                   variable=self.reverse_vertical)
        reverse_cb.pack(side=tk.LEFT, padx=5)
        
        reset_btn = ttk.Button(mode_frame, text="🎯 Reset to Center", 
                             command=self.reset_to_center)
        reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # === WAVE CONTROL PARAMETERS - The core functionality! ===
        wave_frame = ttk.LabelFrame(self.scrollable_frame, text="🌊 Wave Control Parameters")
        wave_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Cursor Sensitivity  
        ttk.Label(wave_frame, text="Cursor Sensitivity:").grid(row=0, column=0, sticky=tk.W, padx=5)
        sensitivity_scale = ttk.Scale(wave_frame, from_=0.5, to=10.0, variable=self.cursor_sensitivity, orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        sensitivity_label = ttk.Label(wave_frame, text="3.0")
        sensitivity_label.grid(row=0, column=2, padx=5)
        self.cursor_sensitivity.trace_add("write", lambda *args: sensitivity_label.config(text=f"{self.cursor_sensitivity.get():.1f}"))
        
        # Wave Strength
        ttk.Label(wave_frame, text="Wave Strength:").grid(row=1, column=0, sticky=tk.W, padx=5)
        wave_scale = ttk.Scale(wave_frame, from_=0.0, to=5.0, variable=self.wave_strength, orient=tk.HORIZONTAL)
        wave_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        wave_label = ttk.Label(wave_frame, text="2.0")
        wave_label.grid(row=1, column=2, padx=5)
        self.wave_strength.trace_add("write", lambda *args: wave_label.config(text=f"{self.wave_strength.get():.1f}"))
        
        # Gravity Width
        ttk.Label(wave_frame, text="Gravity Width:").grid(row=2, column=0, sticky=tk.W, padx=5)
        gravity_scale = ttk.Scale(wave_frame, from_=0.1, to=1.0, variable=self.gravity_width, orient=tk.HORIZONTAL)
        gravity_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        gravity_label = ttk.Label(wave_frame, text="0.4")
        gravity_label.grid(row=2, column=2, padx=5)
        self.gravity_width.trace_add("write", lambda *args: gravity_label.config(text=f"{self.gravity_width.get():.1f}"))
        
        # Default Position
        ttk.Label(wave_frame, text="Default Position:").grid(row=3, column=0, sticky=tk.W, padx=5)
        default_scale = ttk.Scale(wave_frame, from_=0, to=180, variable=self.default_position, orient=tk.HORIZONTAL)
        default_scale.grid(row=3, column=1, sticky=tk.EW, padx=5)
        default_label = ttk.Label(wave_frame, text="90")
        default_label.grid(row=3, column=2, padx=5)
        self.default_position.trace_add("write", lambda *args: default_label.config(text=f"{self.default_position.get():.0f}"))
        
        wave_frame.columnconfigure(1, weight=1)
    
    def switch_emotional_state(self, emotion_name):
        """Switch to a different emotional state and update movement parameters."""
        if emotion_name not in self.emotional_states:
            return
        
        self.current_emotional_state = emotion_name
        self.current_state_label.config(text=f"Current: {emotion_name}")
        
        # Update movement parameters based on emotional state
        state = self.emotional_states[emotion_name]
        params = state.get_movement_params()
        
        self.cursor_sensitivity.set(params['cursor_sensitivity'])
        
        print(f"🎭 Switched to {emotion_name} emotional state")
        print(f"📊 Parameters: sensitivity={params['cursor_sensitivity']:.1f}")
    
    def toggle_connection(self):
        """Toggle hand controller connection - FIXED LOGIC."""
        if not HAND_CONTROLLER_AVAILABLE:
            self.status_label.config(text="❌ Controller unavailable")
            print("⚠️ Hand controller not available - simulation mode")
            return
        
        if not self.connected:
            try:
                # CORRECT CONNECTION: Initialize HandExpressionController like the original
                self.hand_controller = HandExpressionController(
                    port="COM3",  # Your Arduino port
                    baudrate=9600,
                    clean_output=True
                )
                # The connection is established in __init__, check if serial connection exists
                if self.hand_controller.serial_connection:
                    # Enable manual override to ensure our commands are processed
                    self.hand_controller.enable_manual_override()
                    self.connected = True
                    self.status_label.config(text="✅ Connected")
                    self.connect_btn.config(text="Disconnect")
                    print("✅ Connected to hand controller on COM3")
                    print("🎮 Manual override enabled - ready for cursor control")
                else:
                    self.status_label.config(text="❌ Connection failed")
                    print("❌ Failed to connect to COM3")
            except Exception as e:
                self.status_label.config(text="❌ Connection error")
                print(f"❌ Connection error: {e}")
        else:
            # Disconnect
            if self.hand_controller:
                if hasattr(self.hand_controller, 'disable_manual_override'):
                    self.hand_controller.disable_manual_override()
                if hasattr(self.hand_controller, 'cleanup'):
                    self.hand_controller.cleanup()
            self.hand_controller = None
            self.connected = False
            self.status_label.config(text="❌ Disconnected")
            self.connect_btn.config(text="Connect to Hand Controller")
            print("🔌 Disconnected from hand controller")
    
    def on_mouse_move(self, event):
        """Handle mouse movement in canvas - FIXED with debug output."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            old_x, old_y = self.mouse_x, self.mouse_y
            self.mouse_x = event.x / canvas_width
            self.mouse_y = event.y / canvas_height
            
            # Debug output for first few movements
            if hasattr(self, 'move_count'):
                self.move_count += 1
            else:
                self.move_count = 1
            
            if self.move_count < 5:
                print(f"🎯 Mouse move {self.move_count}: ({self.mouse_x:.3f}, {self.mouse_y:.3f}) canvas: {canvas_width}x{canvas_height}")
            
            # Record movement if recording - VECTOR-BASED APPROACH!
            if self.recording:
                current_time = time.time()
                relative_time = current_time - self.record_start_time
                
                # Store position for exact playback
                movement_point = {
                    'time': current_time,
                    'relative_time': relative_time,
                    'x': self.mouse_x,
                    'y': self.mouse_y,
                    'finger_positions': self.finger_positions.copy()
                }
                if self.current_emotional_state not in self.recorded_movements:
                    self.recorded_movements[self.current_emotional_state] = []
                self.recorded_movements[self.current_emotional_state].append(movement_point)
                
                # Calculate movement vector for generation - ENHANCED for fast movements
                dt = current_time - self.last_record_time
                if dt > 0.008 and self.last_record_time > 0:  # Sample at ~120fps max for fast movements
                    dx = self.mouse_x - self.last_record_pos[0]
                    dy = self.mouse_y - self.last_record_pos[1]
                    
                    # ADAPTIVE movement threshold - lower for fast emotions
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Dynamic threshold based on current emotional state
                    if self.current_emotional_state == 'excited':
                        min_threshold = 0.0005  # Very sensitive for excited
                    elif self.current_emotional_state == 'happy':
                        min_threshold = 0.0008  # Moderately sensitive
                    else:
                        min_threshold = 0.001   # Standard threshold
                    
                    if distance > min_threshold:
                        # Calculate movement vector - ENHANCED with better speed detection
                        direction = math.atan2(dy, dx)
                        speed = distance / dt
                        
                        # IMPROVED acceleration calculation for jerky/fast movements
                        if hasattr(self, 'last_record_speed'):
                            acceleration = (speed - self.last_record_speed) / dt
                            # Track acceleration magnitude for jerkiness detection
                            accel_magnitude = abs(acceleration)
                        else:
                            acceleration = 0.0
                            accel_magnitude = 0.0
                        
                        # ENHANCED vector with more movement characteristics
                        vector = {
                            'time': current_time,
                            'relative_time': relative_time,
                            'direction': direction,  # Angle in radians
                            'speed': speed,         # Units per second
                            'acceleration': acceleration,
                            'accel_magnitude': accel_magnitude,  # For jerkiness detection
                            'distance': distance,
                            'dt': dt,
                            'start_x': self.last_record_pos[0],
                            'start_y': self.last_record_pos[1],
                            'end_x': self.mouse_x,
                            'end_y': self.mouse_y,
                            'is_fast_movement': speed > 2.0,  # Flag fast movements
                            'is_micro_movement': distance < 0.005  # Flag tiny movements
                        }
                        
                        if self.current_emotional_state not in self.recorded_vectors:
                            self.recorded_vectors[self.current_emotional_state] = []
                        self.recorded_vectors[self.current_emotional_state].append(vector)
                        
                        self.last_record_speed = speed
                        
                # Update tracking
                self.last_record_pos = (self.mouse_x, self.mouse_y)
                self.last_record_time = current_time
            
            # IMPORTANT: Don't interfere with generative movement!
            elif self.generating:
                # Ignore mouse input during generation to prevent interference
                return
    
    def on_mouse_click(self, event):
        """Handle mouse click in canvas."""
        self.on_mouse_move(event)  # Update position
    
    def reset_to_center(self):
        """Reset cursor and servos to center position."""
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_targets = [90.0] * self.num_fingers
        print("🎯 Reset to center position")
    
    def start_control_loop(self):
        """Start the main control loop - clean and direct."""
        self.running = True
        print("🎯 Starting direct control loop...")
        self.control_loop()
    
    def control_loop(self):
        """Main control loop - direct wave-based control without physics."""
        if not self.running:
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Debug output for first few loops
        if hasattr(self, 'loop_count'):
            self.loop_count += 1
        else:
            self.loop_count = 1
            
        if self.loop_count < 5:
            print(f"🔄 Control loop {self.loop_count}: dt={dt:.3f}, mouse=({self.mouse_x:.3f}, {self.mouse_y:.3f})")
        
        # Update canvas visualization
        self.update_canvas()
        
        # Update playback if active
        if self.playing_back:
            self.update_playback()
        
        # Update Markov generation if active
        if self.generating:
            self.update_markov_generation()
        
        # Calculate finger targets from cursor position
        self.calculate_finger_targets()
        
        # Direct control - immediate response!
        self.finger_positions = self.finger_targets.copy()
        
        if hasattr(self, 'direct_count'):
            self.direct_count += 1
        else:
            self.direct_count = 1
        
        if self.direct_count < 5 or self.direct_count % 30 == 0:
            print(f"🎯 Direct control {self.direct_count}: positions={[f'{p:.1f}' for p in self.finger_positions]} from targets={[f'{t:.1f}' for t in self.finger_targets]}")
        
        # Send to hand controller
        self.send_to_hand_controller()
        
        # Schedule next update
        self.root.after(16, self.control_loop)  # ~60 FPS
    
    def calculate_finger_targets(self):
        """Calculate servo targets from cursor position - Pure wave-based control."""
        # Wave-based finger control (this was working well)
        wave_strength = self.wave_strength.get()
        gravity_width = self.gravity_width.get()
        default_pos = self.default_position.get()
        sensitivity = self.cursor_sensitivity.get()
        
        # Debug output for first few calculations
        if hasattr(self, 'calc_count'):
            self.calc_count += 1
        else:
            self.calc_count = 1
            
        targets_changed = False
        
        for i in range(self.num_fingers):
            # TIGHTENED MAPPING - same condensed area as visual (25%-75% of screen)
            condensed_start = 0.25  # 25% from left (matches visualization)
            condensed_width = 0.5   # 50% of screen width (matches visualization)
            finger_x = condensed_start + ((i + 0.5) / self.num_fingers) * condensed_width
            
            # Calculate influence of cursor on this finger
            distance = abs(self.mouse_x - finger_x)
            if distance < gravity_width:
                influence = 1.0 - (distance / gravity_width)
                
                # Calculate vertical influence
                y_offset = (self.mouse_y - 0.5) * sensitivity * wave_strength * influence
                if self.reverse_vertical.get():
                    y_offset = -y_offset
                
                target = default_pos + (y_offset * 45.0)  # ±45 degrees range
                new_target = max(0, min(180, target))
                
                if abs(new_target - self.finger_targets[i]) > 1.0:
                    targets_changed = True
                    
                self.finger_targets[i] = new_target
            else:
                # Return to default position
                if abs(default_pos - self.finger_targets[i]) > 1.0:
                    targets_changed = True
                self.finger_targets[i] = default_pos
        
        # Debug output for first few calculations or when targets change significantly
        if self.calc_count < 5 or (targets_changed and self.calc_count % 30 == 0):
            print(f"🎯 Targets {self.calc_count}: {[f'{t:.1f}' for t in self.finger_targets]} (mouse: {self.mouse_x:.3f}, {self.mouse_y:.3f})")
    
    def update_canvas(self):
        """Update canvas visualization - FIXED for better visibility."""
        self.canvas.delete("all")
        
        # Draw cursor
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        cursor_x = self.mouse_x * canvas_width
        cursor_y = self.mouse_y * canvas_height
        
        self.canvas.create_oval(cursor_x-8, cursor_y-8, cursor_x+8, cursor_y+8, 
                               fill="red", outline="white", width=2)
        
        # Draw finger indicators - TIGHTENED and CONDENSED for precise control
        for i in range(self.num_fingers):
            # Much tighter spacing - only center 50% of canvas for focused control
            condensed_area_start = canvas_width * 0.25  # Start at 25% from left
            condensed_area_width = canvas_width * 0.5   # Use only 50% of canvas width
            finger_x = condensed_area_start + ((i + 0.5) / self.num_fingers) * condensed_area_width
            finger_y = canvas_height - 80  # Adjusted for smaller canvas
            
            # Smaller bars for tighter control
            bar_height = 60   # Smaller bars for the smaller canvas
            bar_width = 25    # Narrower bars for tighter spacing
            self.canvas.create_rectangle(finger_x-bar_width//2, finger_y, 
                                       finger_x+bar_width//2, finger_y+bar_height,
                                       fill="gray30", outline="white", width=2)
            
            # Position indicator (current position)
            pos_ratio = self.finger_positions[i] / 180.0
            pos_height = int(pos_ratio * bar_height)
            self.canvas.create_rectangle(finger_x-bar_width//2, finger_y+bar_height-pos_height, 
                                       finger_x+bar_width//2, finger_y+bar_height,
                                       fill="lime", outline="yellow", width=1)
            
            # Target indicator (where it's trying to go)
            target_ratio = self.finger_targets[i] / 180.0
            target_height = int(target_ratio * bar_height)
            target_y = finger_y + bar_height - target_height
            self.canvas.create_line(finger_x-bar_width//2-3, target_y, 
                                  finger_x+bar_width//2+3, target_y,
                                  fill="red", width=2)
            
            # Labels - smaller text for smaller canvas
            self.canvas.create_text(finger_x, finger_y+bar_height+10, text=f"F{i+1}", 
                                  fill="white", font=("Arial", 8, "bold"))
            self.canvas.create_text(finger_x, finger_y+bar_height+22, 
                                  text=f"{self.finger_positions[i]:.0f}°", 
                                  fill="white", font=("Arial", 7))
        
        # Mode indicator
        mode_text = f"Mode: Direct Wave Control"
        self.canvas.create_text(10, 10, text=mode_text, fill="white", anchor="nw", 
                              font=("Arial", 12, "bold"))
        
        # Emotional state indicator
        emotion_text = f"Emotion: {self.current_emotional_state}"
        self.canvas.create_text(10, 30, text=emotion_text, fill="yellow", anchor="nw",
                              font=("Arial", 12, "bold"))
        
        # Cursor position indicator
        cursor_text = f"Cursor: ({self.mouse_x:.2f}, {self.mouse_y:.2f})"
        self.canvas.create_text(10, 50, text=cursor_text, fill="cyan", anchor="nw",
                              font=("Arial", 10))
        
        # Recording/playback indicators
        if self.recording:
            self.canvas.create_text(canvas_width-10, 10, text="🔴 RECORDING", fill="red", anchor="ne",
                                  font=("Arial", 14, "bold"))
        elif self.playing_back:
            self.canvas.create_text(canvas_width-10, 10, text="▶️ PLAYING", fill="blue", anchor="ne",
                                  font=("Arial", 14, "bold"))
        elif self.generating:
            self.canvas.create_text(canvas_width-10, 10, text="🎨 LEARNING", fill="purple", anchor="ne",
                                  font=("Arial", 14, "bold"))
    
    def send_to_hand_controller(self):
        """Send positions to hand controller - FIXED INTERFACE."""
        if not self.connected or not self.hand_controller:
            return
        
        current_time = time.time()
        if current_time - self.last_send_time < self.send_interval:
            return
        
        # Check if positions changed significantly
        changed = False
        for i in range(self.num_fingers):
            if abs(self.finger_positions[i] - getattr(self, 'last_sent_positions', [90]*4)[i]) > self.position_threshold:
                changed = True
                break
        
        if changed or current_time - getattr(self, 'last_any_send_time', 0) > 1.0:
            try:
                # CORRECT INTERFACE: Use set_hand_positions with simple list like the working version
                positions = [
                    int(self.finger_positions[0]),  # index
                    int(self.finger_positions[1]),  # middle  
                    int(self.finger_positions[2]),  # ring
                    int(self.finger_positions[3])   # pinky
                ]
                
                # Debug output to see what we're actually sending
                if hasattr(self, 'send_count'):
                    self.send_count += 1
                else:
                    self.send_count = 1
                
                if self.send_count < 5 or self.send_count % 20 == 0:
                    print(f"📤 Sending {self.send_count}: positions={positions} from finger_positions={[f'{p:.1f}' for p in self.finger_positions]}")
                
                # Use the WORKING method from the working version
                self.hand_controller.set_hand_positions(positions)
                
                self.last_sent_positions = self.finger_positions.copy()
                self.last_send_time = current_time
                self.last_any_send_time = current_time
                
            except Exception as e:
                print(f"❌ Error sending to hand controller: {e}")
                import traceback
                traceback.print_exc()
    
    def start_recording(self):
        """Start recording cursor movements AND movement vectors for current emotional state."""
        if self.recording:
            # Stop recording
            self.stop_recording()
            return
            
        # Start recording
        self.recording = True
        self.record_start_time = time.time()
        self.recorded_movements[self.current_emotional_state] = []
        self.recorded_vectors[self.current_emotional_state] = []
        
        # Initialize vector tracking
        self.last_record_pos = (self.mouse_x, self.mouse_y)
        self.last_record_time = time.time()
        if hasattr(self, 'last_record_speed'):
            delattr(self, 'last_record_speed')
        
        self.record_btn.config(text="⏹️ Stop Recording")
        self.record_status.config(text=f"Recording {self.current_emotional_state}...", foreground="red")
        self.vector_status.config(text="Capturing vectors...", foreground="orange")
        
        print(f"🎬 Started recording movements AND vectors for {self.current_emotional_state}")
        
        # Auto-stop after 45 seconds
        self.root.after(45000, self.auto_stop_recording)
    
    def auto_stop_recording(self):
        """Auto-stop recording after 45 seconds."""
        if self.recording:
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and save movements + vectors."""
        if not self.recording:
            return
            
        self.recording = False
        duration = time.time() - self.record_start_time
        movement_count = len(self.recorded_movements.get(self.current_emotional_state, []))
        vector_count = len(self.recorded_vectors.get(self.current_emotional_state, []))
        
        self.record_btn.config(text="🎬 Record Movement (45s)")
        self.record_status.config(text=f"Recorded {movement_count} points in {duration:.1f}s", foreground="green")
        self.vector_status.config(text=f"{vector_count} vectors captured", foreground="green")
        
        print(f"🎬 Stopped recording. Captured {movement_count} movements and {vector_count} vectors in {duration:.1f} seconds")
        
        # Save to file immediately
        self.save_recording()
        
        # Train Markov chain with recorded movements
        self.train_markov_chain()
    
    def save_recording(self):
        """Save recorded movements AND vectors to file."""
        if self.current_emotional_state not in self.recorded_movements:
            return
            
        os.makedirs("movement_recordings", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"movement_recordings/{self.current_emotional_state}_{timestamp}.json"
        
        movements = self.recorded_movements[self.current_emotional_state]
        vectors = self.recorded_vectors.get(self.current_emotional_state, [])
        
        data = {
            'emotion': self.current_emotional_state,
            'timestamp': timestamp,
            'movement_count': len(movements),
            'vector_count': len(vectors),
            'duration': movements[-1]['time'] - movements[0]['time'] if len(movements) > 1 else 0,
            'movements': movements,
            'vectors': vectors  # Include vector data for generation
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"💾 Saved recording with {len(vectors)} vectors to {filename}")
    
    def train_markov_chain(self):
        """Train Markov chain with recorded movements."""
        if not self.markov_system:
            print("⚠️ Markov system not available - skipping training")
            return
            
        if self.current_emotional_state not in self.recorded_movements:
            print("❌ No recorded movements to train on")
            return
            
        movements = self.recorded_movements[self.current_emotional_state]
        if len(movements) < 10:
            print(f"❌ Need at least 10 movement points, got {len(movements)}")
            return
        
        print(f"🧬 Training Markov chain for {self.current_emotional_state}...")
        
        # Train the Markov chain
        success = self.markov_system.learn_emotion(self.current_emotional_state, movements)
        
        if success:
            self.record_status.config(text=f"Markov chain learned for {self.current_emotional_state}!", foreground="green")
            print(f"✅ Successfully trained Markov chain for {self.current_emotional_state}")
        else:
            self.record_status.config(text="Failed to learn Markov chain!", foreground="red")
            print(f"❌ Failed to train Markov chain for {self.current_emotional_state}")
    
    def start_playback(self):
        """Start playing back recorded movements for current emotional state."""
        if self.playing_back:
            # Stop playback
            self.stop_playback()
            return
            
        if self.current_emotional_state not in self.recorded_movements:
            self.record_status.config(text="No recording for this emotion!", foreground="orange")
            return
            
        # Start playback
        self.playing_back = True
        self.playback_start_time = time.time()
        self.current_playback = self.recorded_movements[self.current_emotional_state].copy()
        
        self.playback_btn.config(text="⏹️ Stop Playback")
        self.record_status.config(text=f"Playing back {self.current_emotional_state}...", foreground="blue")
        
        print(f"▶️ Started playback of {len(self.current_playback)} movements for {self.current_emotional_state}")
    
    def stop_playback(self):
        """Stop playback."""
        self.playing_back = False
        self.playback_btn.config(text="▶️ Play Back")
        self.record_status.config(text="Playback stopped", foreground="gray")
        print("⏹️ Playback stopped")
    
    def update_playback(self):
        """Update cursor position during playback."""
        if not self.playing_back or not self.current_playback:
            return
            
        current_time = time.time()
        playback_elapsed = current_time - self.playback_start_time
        
        # Find the movement to play at this time
        for movement in self.current_playback:
            movement_time = movement['time'] - self.current_playback[0]['time']  # Relative time
            
            if movement_time <= playback_elapsed:
                # Update cursor position to recorded position
                self.mouse_x = movement['x']
                self.mouse_y = movement['y']
            else:
                break
        
        # Check if playback finished
        if self.current_playback:
            total_duration = self.current_playback[-1]['time'] - self.current_playback[0]['time']
            if playback_elapsed >= total_duration:
                self.stop_playback()
    
    def analyze_movement_vectors(self):
        """SPATIAL-AWARE ANALYSIS: Deep analysis with proper spatial understanding and boundary awareness."""
        if self.current_emotional_state not in self.recorded_vectors:
            print(f"⚠️ No vectors found for {self.current_emotional_state}")
            return
            
        vectors = self.recorded_vectors[self.current_emotional_state]
        if len(vectors) < 5:  # Need some data to analyze
            self.vector_status.config(text="Need more movement", foreground="orange")
            print(f"⚠️ Only {len(vectors)} vectors for {self.current_emotional_state}, need at least 5")
            return
            
        print(f"🔬 SPATIAL-AWARE ANALYSIS: Analyzing {len(vectors)} vectors for {self.current_emotional_state}...")
        
        try:
            # Extract movement data with spatial context - ENHANCED for fast movements
            directions = [v['direction'] for v in vectors]
            speeds = [v['speed'] for v in vectors]
            distances = [v['distance'] for v in vectors]
            accelerations = [v.get('acceleration', 0) for v in vectors]
            accel_magnitudes = [v.get('accel_magnitude', 0) for v in vectors]
            fast_movements = [v.get('is_fast_movement', False) for v in vectors]
            micro_movements = [v.get('is_micro_movement', False) for v in vectors]
            
            # Get ALL positions (both start and end)
            all_positions = []
            for v in vectors:
                all_positions.append((v['start_x'], v['start_y']))
                all_positions.append((v['end_x'], v['end_y']))
            
            x_positions = [pos[0] for pos in all_positions]
            y_positions = [pos[1] for pos in all_positions]
            
            # SPATIAL CHARACTERISTICS - The key improvement!
            min_x, max_x = min(x_positions), max(x_positions)
            min_y, max_y = min(y_positions), max(y_positions)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Calculate actual surface area used
            x_range = max_x - min_x
            y_range = max_y - min_y
            movement_area = x_range * y_range
            
            # Calculate movement density (how much you move within your area)
            total_distance = sum(distances)
            movement_density = total_distance / max(movement_area, 0.001)  # Avoid division by zero
            
            # Speed analysis with proper context - ENHANCED
            avg_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds) if speeds else 0
            speed_std = (sum((s - avg_speed)**2 for s in speeds) / len(speeds)) ** 0.5
            speed_variance_ratio = speed_std / max(avg_speed, 0.001)
            
            # ENHANCED: Fast movement detection
            fast_movement_count = sum(fast_movements)
            fast_movement_ratio = fast_movement_count / len(vectors)
            
            # ENHANCED: Acceleration analysis for jerkiness
            avg_accel_magnitude = sum(accel_magnitudes) / len(accel_magnitudes) if accel_magnitudes else 0
            high_accel_count = sum(1 for a in accel_magnitudes if a > avg_accel_magnitude * 2)
            jerkiness_ratio = high_accel_count / len(vectors)
            
            # Micro-movement analysis for precision
            micro_movement_count = sum(micro_movements)
            micro_movement_ratio = micro_movement_count / len(vectors)
            
            # Direction analysis
            direction_vectors_x = [math.cos(d) for d in directions]
            direction_vectors_y = [math.sin(d) for d in directions]
            avg_direction_x = sum(direction_vectors_x) / len(direction_vectors_x)
            avg_direction_y = sum(direction_vectors_y) / len(direction_vectors_y)
            direction_consistency = math.sqrt(avg_direction_x**2 + avg_direction_y**2)
            
            # IMPROVED BOUNDARY ANALYSIS - Check how close you get to edges
            edge_proximity_count = 0
            for x, y in all_positions:
                if x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9:  # Near edges
                    edge_proximity_count += 1
            edge_usage_ratio = edge_proximity_count / len(all_positions)
            
            # BETTER STYLE DETECTION with fast movement awareness
            # Area-based classification (much more accurate)
            is_exploratory = movement_area > 0.15  # Uses more than 15% of canvas
            is_focused = movement_area < 0.02       # Uses less than 2% of canvas
            is_contained = edge_usage_ratio < 0.1   # Stays away from edges
            is_boundary_pushing = edge_usage_ratio > 0.3  # Often near edges
            
            # ENHANCED Speed-based classification with better fast detection
            is_jerky = jerkiness_ratio > 0.3 or speed_variance_ratio > 2.0     # High acceleration changes OR high speed variation
            is_smooth = jerkiness_ratio < 0.1 and speed_variance_ratio < 0.5   # Low acceleration changes AND low speed variation
            is_fast = avg_speed > 0.8 or fast_movement_ratio > 0.4             # High average speed OR many fast movements
            is_slow = avg_speed < 0.15 and fast_movement_ratio < 0.1           # Low average speed AND few fast movements
            is_explosive = fast_movement_ratio > 0.6 and jerkiness_ratio > 0.4  # NEW: Lots of fast, jerky movements
            is_precise = micro_movement_ratio > 0.3                            # NEW: Lots of tiny precise movements
            
            # Direction-based classification
            is_directional = direction_consistency > 0.7  # Consistent direction
            is_chaotic = direction_consistency < 0.3      # Random directions
            
            # Density-based classification (NEW!)
            is_dense = movement_density > 20.0        # Lots of movement in small area
            is_sparse = movement_density < 5.0        # Little movement in large area
            
            # Create comprehensive movement signature - ENHANCED
            signature = {
                'vector_count': len(vectors),
                'emotion': self.current_emotional_state,
                'ready_to_generate': True,
                
                # Core measurements - ENHANCED
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'speed_std': speed_std,
                'speed_variance_ratio': speed_variance_ratio,
                'fast_movement_ratio': fast_movement_ratio,
                'avg_accel_magnitude': avg_accel_magnitude,
                'jerkiness_ratio': jerkiness_ratio,
                'micro_movement_ratio': micro_movement_ratio,
                'movement_area': movement_area,
                'movement_density': movement_density,
                'direction_consistency': direction_consistency,
                
                # Spatial boundaries
                'center_x': center_x,
                'center_y': center_y,
                'x_range': x_range,
                'y_range': y_range,
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'edge_usage_ratio': edge_usage_ratio,
                
                # Style characteristics - ENHANCED
                'is_jerky': is_jerky,
                'is_smooth': is_smooth,
                'is_fast': is_fast,
                'is_slow': is_slow,
                'is_explosive': is_explosive,  # NEW
                'is_precise': is_precise,      # NEW
                'is_exploratory': is_exploratory,
                'is_focused': is_focused,
                'is_contained': is_contained,
                'is_boundary_pushing': is_boundary_pushing,
                'is_directional': is_directional,
                'is_chaotic': is_chaotic,
                'is_dense': is_dense,
                'is_sparse': is_sparse,
                
                # Pattern flags (simplified for now)
                'is_circular': False,
                'is_spiraling': False,
                'has_micro_patterns': is_dense or is_precise  # Dense movement or precise movements
            }
            
            self.learned_patterns[self.current_emotional_state] = signature
            
            # Create more accurate style tags - ENHANCED
            style_tags = []
            
            # Speed characteristics - ENHANCED
            if is_explosive: style_tags.append("EXPLOSIVE")  # Most distinctive
            elif is_jerky: style_tags.append("jerky")
            if is_smooth: style_tags.append("smooth")
            if is_fast: style_tags.append("fast")
            if is_slow: style_tags.append("slow")
            if is_precise: style_tags.append("precise")
            
            # Spatial characteristics
            if is_exploratory: style_tags.append("exploratory")
            if is_focused: style_tags.append("focused")
            if is_contained: style_tags.append("contained")
            if is_boundary_pushing: style_tags.append("boundary-pushing")
            
            # Direction characteristics
            if is_directional: style_tags.append("directional")
            if is_chaotic: style_tags.append("chaotic")
            
            # Density characteristics
            if is_dense: style_tags.append("dense")
            if is_sparse: style_tags.append("sparse")
            
            style_text = f"✅ Style: {', '.join(style_tags) if style_tags else 'balanced'}"
            self.vector_status.config(text=style_text, foreground="green")
            
            print(f"✅ ENHANCED SPATIAL ANALYSIS COMPLETE for {self.current_emotional_state}!")
            print(f"   📊 Speed: avg={avg_speed:.3f}, max={max_speed:.3f}, fast_ratio={fast_movement_ratio:.2f}")
            print(f"   ⚡ Jerkiness: ratio={jerkiness_ratio:.2f}, accel_mag={avg_accel_magnitude:.3f}")
            print(f"   📍 Area: {movement_area:.3f} ({x_range:.2f}x{y_range:.2f}), center=({center_x:.2f},{center_y:.2f})")
            print(f"   🎯 Density: {movement_density:.1f}, edge_usage: {edge_usage_ratio:.2f}")
            print(f"   🔬 Precision: micro_ratio={micro_movement_ratio:.2f}")
            print(f"   🏷️ Style: {', '.join(style_tags) if style_tags else 'balanced movement'}")
            print(f"   ✅ Enhanced fast-movement aware pattern learned and ready for generation!")
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            # Create basic fallback signature
            signature = {
                'vector_count': len(vectors),
                'emotion': self.current_emotional_state,
                'ready_to_generate': True,
                'avg_speed': 0.1,
                'speed_std': 0.05,
                'movement_area': 0.2,
                'center_x': 0.5, 'center_y': 0.5,
                'x_range': 0.4, 'y_range': 0.4,
                'min_x': 0.3, 'max_x': 0.7, 'min_y': 0.3, 'max_y': 0.7,
                'direction_consistency': 0.5,
                'is_jerky': False, 'is_smooth': False, 'is_exploratory': True,
                'is_focused': False, 'is_directional': False, 'is_chaotic': False,
                'is_contained': True, 'is_boundary_pushing': False,
                'is_circular': False, 'is_spiraling': False, 'has_micro_patterns': False
            }
            self.learned_patterns[self.current_emotional_state] = signature
            self.vector_status.config(text="✅ Basic spatial analysis", foreground="orange")
    
    def start_markov_generation(self):
        """Start Markov chain-based generative movement using learned patterns."""
        if self.generating:
            # Stop generation
            self.stop_markov_generation()
            return
            
        if not self.markov_system:
            self.record_status.config(text="Markov system not available!", foreground="red")
            return
            
        # Check if we have a trained chain for this emotion
        available_emotions = self.markov_system.get_available_emotions()
        if self.current_emotional_state not in available_emotions:
            self.record_status.config(text=f"No Markov chain for {self.current_emotional_state}!", foreground="orange")
            return
            
        # Switch to the emotion's chain
        success = self.markov_system.apply_emotion(self.current_emotional_state)
        if not success:
            self.record_status.config(text="Failed to load Markov chain!", foreground="red")
            return
            
        # Start generation
        self.generating = True
        self.generation_start_time = time.time()
        
        self.generate_btn.config(text="⏹️ Stop Markov")
        self.record_status.config(text=f"🧬 Generating {self.current_emotional_state} patterns...", foreground="purple")
        
        print(f"🧬 Started Markov generation for {self.current_emotional_state}")
        
        # Auto-stop after 45 seconds
        self.root.after(45000, self.auto_stop_generation)
    
    def auto_stop_generation(self):
        """Auto-stop generation after 45 seconds."""
        if self.generating:
            self.stop_markov_generation()
    
    def stop_markov_generation(self):
        """Stop Markov generation."""
        self.generating = False
        self.generate_btn.config(text="� Markov Generate")
        self.record_status.config(text="Markov generation stopped", foreground="gray")
        print("⏹️ Markov generation stopped")
    
    def update_markov_generation(self):
        """Update cursor position during Markov generation."""
        if not self.generating or not self.markov_system:
            return
            
        current_time = time.time()
        dt = current_time - getattr(self, 'last_generation_time', current_time)
        self.last_generation_time = current_time
        
        # Generate next movement from Markov chain
        if dt > 0:
            current_pos = (self.mouse_x, self.mouse_y)
            dx, dy = self.markov_system.get_movement(current_pos, dt)
            
            # Update cursor position with generated movement
            self.mouse_x = max(0.0, min(1.0, self.mouse_x + dx))
            self.mouse_y = max(0.0, min(1.0, self.mouse_y + dy))
        
        # Check if generation finished
        generation_elapsed = current_time - self.generation_start_time
        if generation_elapsed >= 45.0:
            self.stop_markov_generation()


if __name__ == "__main__":
    print("🎯 Starting Clean Emotional Hand Control with Markov Chains")
    app = CleanCursorInterface()
    app.root.mainloop()
    
    def __init__(self, signature):
        self.signature = signature
        
        # Basic movement characteristics - ENHANCED
        self.avg_speed = signature.get('avg_speed', 0.1)
        self.max_speed = signature.get('max_speed', 0.5)
        self.speed_std = signature.get('speed_std', 0.05)
        self.direction_consistency = signature.get('direction_consistency', 0.5)
        self.fast_movement_ratio = signature.get('fast_movement_ratio', 0.0)
        self.jerkiness_ratio = signature.get('jerkiness_ratio', 0.0)
        self.avg_accel_magnitude = signature.get('avg_accel_magnitude', 0.0)
        
        # SPATIAL AWARENESS - The key improvement!
        self.center_x = signature.get('center_x', 0.5)
        self.center_y = signature.get('center_y', 0.5)
        self.x_range = signature.get('x_range', 0.4)
        self.y_range = signature.get('y_range', 0.4)
        self.min_x = signature.get('min_x', 0.3)
        self.max_x = signature.get('max_x', 0.7)
        self.min_y = signature.get('min_y', 0.3)
        self.max_y = signature.get('max_y', 0.7)
        
        # Movement style characteristics - ENHANCED
        self.is_jerky = signature.get('is_jerky', False)
        self.is_smooth = signature.get('is_smooth', False)
        self.is_fast = signature.get('is_fast', False)
        self.is_slow = signature.get('is_slow', False)
        self.is_explosive = signature.get('is_explosive', False)  # NEW
        self.is_precise = signature.get('is_precise', False)     # NEW
        self.is_exploratory = signature.get('is_exploratory', False)
        self.is_focused = signature.get('is_focused', False)
        self.is_contained = signature.get('is_contained', True)
        self.is_boundary_pushing = signature.get('is_boundary_pushing', False)
        self.is_directional = signature.get('is_directional', False)
        self.is_chaotic = signature.get('is_chaotic', False)
        self.is_dense = signature.get('is_dense', False)
        self.is_sparse = signature.get('is_sparse', False)
        
        # Current state - start at learned center with enhanced tracking
        self.x = self.center_x
        self.y = self.center_y
        self.current_direction = random.uniform(0, 2 * math.pi)
        self.current_speed = self.avg_speed
        self.direction_drift = 0.0
        self.time_since_direction_change = 0.0
        self.time_since_speed_burst = 0.0  # NEW: Track speed bursts
        self.in_speed_burst = False        # NEW: Are we in a fast burst?
        
        # Boundary handling - respect learned spatial preferences
        if self.is_contained:
            self.boundary_padding = 0.08  # Stay well away from edges
        elif self.is_boundary_pushing:
            self.boundary_padding = 0.02  # Get close to edges
        else:
            self.boundary_padding = 0.05  # Moderate boundary avoidance
            
        self.boundary_repel_strength = 0.4
        
        print(f"🎨 Created ENHANCED spatial-aware generator for {signature['emotion']}")
        print(f"   📍 Spatial bounds: ({self.min_x:.2f},{self.min_y:.2f}) to ({self.max_x:.2f},{self.max_y:.2f})")
        print(f"   🎯 Center: ({self.center_x:.2f},{self.center_y:.2f}), range: {self.x_range:.2f}x{self.y_range:.2f}")
        print(f"   ⚡ Speed: avg={self.avg_speed:.3f}, max={self.max_speed:.3f}, fast_ratio={self.fast_movement_ratio:.2f}")
        print(f"   🏷️ Style: explosive={self.is_explosive}, jerky={self.is_jerky}, fast={self.is_fast}")
        print(f"   🔧 Jerkiness: {self.jerkiness_ratio:.2f}, accel_mag: {self.avg_accel_magnitude:.3f}")
    
    def generate_next_step(self, dt):
        """Generate next movement step with proper spatial awareness."""
        # Update direction based on style
        self.update_direction(dt)
        
        # Calculate base speed with enhanced style modulation
        base_speed = self.calculate_speed(dt)
        
        # Calculate movement vector
        dx = math.cos(self.current_direction) * base_speed * dt
        dy = math.sin(self.current_direction) * base_speed * dt
        
        # Apply boundary awareness - CRITICAL IMPROVEMENT!
        dx, dy = self.apply_boundary_constraints(dx, dy)
        
        # Update position with boundary clamping
        new_x = max(self.boundary_padding, min(1.0 - self.boundary_padding, self.x + dx))
        new_y = max(self.boundary_padding, min(1.0 - self.boundary_padding, self.y + dy))
        
        # If we hit a boundary, adjust direction appropriately
        if new_x <= self.boundary_padding or new_x >= 1.0 - self.boundary_padding:
            if not self.is_boundary_pushing:
                self.current_direction = math.pi - self.current_direction  # Reflect horizontally
        if new_y <= self.boundary_padding or new_y >= 1.0 - self.boundary_padding:
            if not self.is_boundary_pushing:
                self.current_direction = -self.current_direction  # Reflect vertically
        
        self.x = new_x
        self.y = new_y
        
        return self.x, self.y
    
    def update_direction(self, dt):
        """Update movement direction based on learned style with enhanced explosive handling."""
        self.time_since_direction_change += dt
        
        # ENHANCED: Different direction change patterns based on style
        if self.is_explosive:
            # Explosive movement: sudden direction changes and bursts
            change_interval = random.uniform(0.1, 0.8)  # Very quick changes
            if self.time_since_direction_change > change_interval:
                self.current_direction += random.uniform(-math.pi, math.pi)  # Big direction changes
                self.time_since_direction_change = 0
                
        elif self.is_chaotic:
            # Chaotic movement: frequent random direction changes
            if random.random() < 0.3:  # 30% chance per frame to change direction
                self.current_direction += random.uniform(-math.pi/2, math.pi/2)
        elif self.is_directional:
            # Directional movement: maintain direction with small adjustments
            if random.random() < 0.05:  # 5% chance to adjust
                self.current_direction += random.uniform(-math.pi/8, math.pi/8)
        else:
            # Balanced movement: moderate direction changes
            if random.random() < 0.1:  # 10% chance
                self.current_direction += random.uniform(-math.pi/4, math.pi/4)
        
        # Add style-specific behaviors
        if self.is_exploratory and not self.is_boundary_pushing:
            # Exploratory but contained: avoid edges, seek new areas within learned bounds
            center_pull = self.get_center_pull_force()
            if abs(center_pull) > math.pi/6:  # Only apply if significantly off-center
                self.current_direction += center_pull * 0.1
                
        elif self.is_focused:
            # Focused movement: tend to return to center area
            center_pull = self.get_center_pull_force()
            self.current_direction += center_pull * 0.3
        
        # Normalize direction
        while self.current_direction > 2 * math.pi:
            self.current_direction -= 2 * math.pi
        while self.current_direction < 0:
            self.current_direction += 2 * math.pi
    
    def calculate_speed(self, dt):
        """Calculate movement speed based on learned characteristics with enhanced fast handling."""
        self.time_since_speed_burst += dt
        
        # Base speed from learned patterns
        base_speed = self.avg_speed
        
        # ENHANCED: Handle explosive/fast movement patterns
        if self.is_explosive:
            # Explosive: alternate between normal and very fast bursts
            if not self.in_speed_burst and self.time_since_speed_burst > random.uniform(0.5, 2.0):
                self.in_speed_burst = True
                self.time_since_speed_burst = 0
                burst_duration = random.uniform(0.1, 0.5)  # Short bursts
                self.burst_end_time = burst_duration
                
            if self.in_speed_burst:
                if self.time_since_speed_burst < getattr(self, 'burst_end_time', 0.3):
                    # In burst: use maximum speed
                    base_speed = self.max_speed * random.uniform(1.5, 3.0)
                else:
                    # End burst
                    self.in_speed_burst = False
                    self.time_since_speed_burst = 0
                    
        elif self.is_fast:
            # Fast: consistently higher speeds with variation
            base_speed = self.avg_speed * random.uniform(1.5, 2.5)
            
        elif self.is_jerky:
            # Jerky: big speed variations
            base_speed = self.avg_speed * random.uniform(0.2, 3.0)
            
        elif self.is_smooth:
            # Smooth: consistent speed with small variations
            base_speed = self.avg_speed * random.uniform(0.8, 1.2)
            
        elif self.is_slow:
            # Slow: consistently lower speeds
            base_speed = self.avg_speed * random.uniform(0.3, 0.8)
        else:
            # Normal: moderate variation
            base_speed = self.avg_speed * random.uniform(0.6, 1.8)
            
        # Apply jerkiness factor for additional speed variation
        if self.jerkiness_ratio > 0.2:
            jerk_factor = 1.0 + (random.random() - 0.5) * self.jerkiness_ratio * 2.0
            base_speed *= jerk_factor

        # Dense movement patterns tend to be slower and more controlled
        if self.is_dense:
            base_speed *= 0.7
        elif self.is_sparse:
            base_speed *= 1.2

        return max(0.01, min(2.0, base_speed))  # Increased max for explosive movements
    
    def apply_boundary_constraints(self, dx, dy):
        """Apply boundary awareness to movement vector."""
        # Check if movement would take us too close to boundaries
        future_x = self.x + dx
        future_y = self.y + dy
        
        # Apply repelling force from boundaries
        boundary_force_x = 0.0
        boundary_force_y = 0.0
        
        # Canvas boundaries (hard limits)
        edge_margin = self.boundary_padding * 2
        
        # Left boundary
        if future_x < edge_margin:
            boundary_force_x += self.boundary_repel_strength * (edge_margin - future_x)
        
        # Right boundary  
        if future_x > 1.0 - edge_margin:
            boundary_force_x -= self.boundary_repel_strength * (future_x - (1.0 - edge_margin))
        
        # Top boundary
        if future_y < edge_margin:
            boundary_force_y += self.boundary_repel_strength * (edge_margin - future_y)
        
        # Bottom boundary
        if future_y > 1.0 - edge_margin:
            boundary_force_y -= self.boundary_repel_strength * (future_y - (1.0 - edge_margin))
        
        # Apply learned spatial constraints
        if self.is_contained:
            # Stay within learned movement area with some expansion
            learned_margin = 0.08  # Allow some expansion beyond learned area
            learned_min_x = max(0.1, self.min_x - learned_margin)
            learned_max_x = min(0.9, self.max_x + learned_margin)
            learned_min_y = max(0.1, self.min_y - learned_margin)
            learned_max_y = min(0.9, self.max_y + learned_margin)
            
            constraint_strength = 0.3
            
            if future_x < learned_min_x:
                boundary_force_x += constraint_strength * (learned_min_x - future_x)
            if future_x > learned_max_x:
                boundary_force_x -= constraint_strength * (future_x - learned_max_x)
            if future_y < learned_min_y:
                boundary_force_y += constraint_strength * (learned_min_y - future_y)
            if future_y > learned_max_y:
                boundary_force_y -= constraint_strength * (future_y - learned_max_y)
        
        return dx + boundary_force_x, dy + boundary_force_y
    
    def get_center_pull_force(self):
        """Calculate a force that pulls towards the learned center."""
        # Vector from current position to learned center
        to_center_x = self.center_x - self.x
        to_center_y = self.center_y - self.y
        
        # Calculate angle to center
        if to_center_x != 0 or to_center_y != 0:
            angle_to_center = math.atan2(to_center_y, to_center_x)
            # Calculate the angle difference
            angle_diff = angle_to_center - self.current_direction
            
            # Normalize to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            return angle_diff
        
        return 0.0
    
    def _detect_circular_patterns(self, positions, directions):
        """Detect circular and looping patterns in movement."""
        if len(positions) < 10:
            return {'circular_tendency': 0.0, 'avg_turn_rate': 0.0, 'circle_sizes': [], 'circle_completeness': 0.0, 'clockwise_preference': 0.5}
        
        # Analyze direction changes for circular motion
        direction_changes = []
        for i in range(1, len(directions)):
            change = directions[i] - directions[i-1]
            # Normalize to [-pi, pi]
            while change > math.pi:
                change -= 2*math.pi
            while change < -math.pi:
                change += 2*math.pi
            direction_changes.append(change)
        
        # Calculate turning tendency
        clockwise_turns = len([c for c in direction_changes if c < -0.1])
        counterclockwise_turns = len([c for c in direction_changes if c > 0.1])
        total_turns = clockwise_turns + counterclockwise_turns
        
        if total_turns > 0:
            clockwise_preference = clockwise_turns / total_turns
        else:
            clockwise_preference = 0.5
        
        # Calculate average turn rate (how much we turn per step)
        avg_turn_rate = abs(sum(direction_changes)) / len(direction_changes) if direction_changes else 0
        
        # Detect potential circles by looking for position returns
        circle_detections = []
        window_size = min(20, len(positions) // 3)
        
        for i in range(window_size, len(positions) - window_size):
            current_pos = positions[i]
            
            # Look for returns to similar positions
            for j in range(i + window_size, min(i + window_size * 3, len(positions))):
                other_pos = positions[j]
                distance = math.sqrt((current_pos[0] - other_pos[0])**2 + (current_pos[1] - other_pos[1])**2)
                
                if distance < 0.05:  # Close return
                    # Calculate the "circle size" from the path taken
                    path_positions = positions[i:j+1]
                    if len(path_positions) > 5:
                        # Calculate bounding box of the circular path
                        x_coords = [p[0] for p in path_positions]
                        y_coords = [p[1] for p in path_positions]
                        circle_size = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
                        
                        # Calculate completeness (how much of circle was completed)
                        if j < len(direction_changes):
                            total_angle_change = sum(abs(direction_changes[i:j]) if i < len(direction_changes) else [0])
                            completeness = min(1.0, total_angle_change / (2 * math.pi))
                        else:
                            completeness = 0.5
                        
                        circle_detections.append({
                            'size': circle_size,
                            'completeness': completeness,
                            'steps': j - i
                        })
                    break  # Found a return, stop looking for this starting point
        
        # Analyze detected circles
        if circle_detections:
            avg_circle_size = sum(c['size'] for c in circle_detections) / len(circle_detections)
            avg_completeness = sum(c['completeness'] for c in circle_detections) / len(circle_detections)
            circular_tendency = min(1.0, len(circle_detections) / (len(positions) / 20))  # Normalize
        else:
            avg_circle_size = 0.0
            avg_completeness = 0.0
            circular_tendency = 0.0
        
        return {
            'circular_tendency': circular_tendency,
            'avg_turn_rate': avg_turn_rate,
            'circle_sizes': [c['size'] for c in circle_detections],
            'circle_completeness': avg_completeness,
            'clockwise_preference': clockwise_preference
        }
    
    def _detect_spiral_patterns(self, positions, speeds):
        """Detect spiral patterns where movement curves inward or outward."""
        if len(positions) < 15:
            return {'spiral_tendency': 0.0, 'spiral_tightness': 0.0, 'spiral_direction': 0}
        
        spiral_segments = []
        window_size = 10
        
        for i in range(len(positions) - window_size):
            segment = positions[i:i+window_size]
            
            # Calculate center of segment
            center_x = sum(p[0] for p in segment) / len(segment)
            center_y = sum(p[1] for p in segment) / len(segment)
            
            # Calculate distances from center for each point
            distances = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in segment]
            
            # Check for spiral pattern (increasing or decreasing distance)
            if len(distances) > 5:
                # Linear regression to see if distance changes consistently
                n = len(distances)
                sum_x = sum(range(n))
                sum_y = sum(distances)
                sum_xy = sum(i * distances[i] for i in range(n))
                sum_x2 = sum(i*i for i in range(n))
                
                if n * sum_x2 - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    
                    # Significant slope indicates spiral
                    if abs(slope) > 0.001:
                        spiral_segments.append({
                            'slope': slope,
                            'tightness': abs(slope),
                            'direction': 1 if slope > 0 else -1  # 1 = expanding, -1 = contracting
                        })
        
        if spiral_segments:
            avg_tightness = sum(s['tightness'] for s in spiral_segments) / len(spiral_segments)
            avg_direction = sum(s['direction'] for s in spiral_segments) / len(spiral_segments)
            spiral_tendency = min(1.0, len(spiral_segments) / (len(positions) / 15))
        else:
            avg_tightness = 0.0
            avg_direction = 0.0
            spiral_tendency = 0.0
        
        return {
            'spiral_tendency': spiral_tendency,
            'spiral_tightness': avg_tightness,
            'spiral_direction': avg_direction
        }
    
    def _analyze_curvature(self, directions):
        """Analyze the curvature (sharpness of turns) in the movement."""
        if len(directions) < 3:
            return {'avg_curvature': 0.0, 'curvature_variance': 0.0, 'sharp_turn_frequency': 0.0}
        
        curvatures = []
        for i in range(1, len(directions) - 1):
            # Calculate curvature as change in direction change
            prev_change = directions[i] - directions[i-1]
            next_change = directions[i+1] - directions[i]
            
            # Normalize angles to [-pi, pi]
            while prev_change > math.pi:
                prev_change -= 2*math.pi
            while prev_change < -math.pi:
                prev_change += 2*math.pi
            while next_change > math.pi:
                next_change -= 2*math.pi
            while next_change < -math.pi:
                next_change += 2*math.pi
            
            # Curvature is the second derivative of direction
            curvature = abs(next_change - prev_change)
            curvatures.append(curvature)
        
        if curvatures:
            avg_curvature = sum(curvatures) / len(curvatures)
            curvature_variance = sum((c - avg_curvature)**2 for c in curvatures) / len(curvatures)
            
            # Sharp turns are high curvature values
            sharp_threshold = avg_curvature + (curvature_variance ** 0.5) if avg_curvature > 0 else math.pi/4
            sharp_turns = len([c for c in curvatures if c > sharp_threshold])
            sharp_turn_frequency = sharp_turns / len(curvatures)
        else:
            avg_curvature = 0.0
            curvature_variance = 0.0
            sharp_turn_frequency = 0.0
            while prev_change > math.pi: prev_change -= 2*math.pi
            while prev_change < -math.pi: prev_change += 2*math.pi
            while next_change > math.pi: next_change -= 2*math.pi
            while next_change < -math.pi: next_change += 2*math.pi
            
            curvature = abs(next_change - prev_change)
            curvatures.append(curvature)
        
        if curvatures:
            avg_curvature = sum(curvatures) / len(curvatures)
            curvature_variance = sum((c - avg_curvature)**2 for c in curvatures) / len(curvatures)
            
            # Sharp turn threshold (more than 45 degrees change)
            sharp_turn_threshold = math.pi / 4
            sharp_turns = len([c for c in curvatures if c > sharp_turn_threshold])
            sharp_turn_frequency = sharp_turns / len(curvatures)
        else:
            avg_curvature = curvature_variance = sharp_turn_frequency = 0.0
        
        return {
            'avg_curvature': avg_curvature,
            'curvature_variance': curvature_variance,
            'sharp_turn_frequency': sharp_turn_frequency
        }
    
    def _analyze_micro_movements(self, vectors):
        """Analyze small-scale movement patterns and jitter."""
        if len(vectors) < 10:
            return {'intensity': 0.0, 'frequency': 0.0, 'regularity': 0.0, 'jitter_level': 0.0}
        
        # Analyze very small movements
        micro_movements = [v for v in vectors if v['distance'] < 0.01]  # Very small movements
        
        if len(micro_movements) < 3:
            return {'intensity': 0.0, 'frequency': 0.0, 'regularity': 0.0, 'jitter_level': 0.0}
        
        # Micro movement frequency
        micro_frequency = len(micro_movements) / len(vectors)
        
        # Micro movement intensity (average size of micro movements)
        micro_intensity = sum(m['distance'] for m in micro_movements) / len(micro_movements)
        
        # Regularity (how consistent are the micro movements)
        micro_speeds = [m['speed'] for m in micro_movements]
        if len(micro_speeds) > 1:
            micro_speed_avg = sum(micro_speeds) / len(micro_speeds)
            micro_speed_variance = sum((s - micro_speed_avg)**2 for s in micro_speeds) / len(micro_speeds)
            regularity = 1.0 / (1.0 + micro_speed_variance)  # Lower variance = higher regularity
        else:
            regularity = 0.0
        
        # Jitter level (rapid direction changes in micro movements)
        micro_directions = [m['direction'] for m in micro_movements]
        jitter_changes = []
        for i in range(1, len(micro_directions)):
            change = abs(micro_directions[i] - micro_directions[i-1])
            if change > math.pi:
                change = 2*math.pi - change
            jitter_changes.append(change)
        
        jitter_level = sum(jitter_changes) / len(jitter_changes) if jitter_changes else 0.0
        
        return {
            'intensity': min(1.0, micro_intensity * 100),  # Scale to 0-1
            'frequency': micro_frequency,
            'regularity': regularity,
            'jitter_level': min(1.0, jitter_level / math.pi)  # Normalize to 0-1
        }
    
    def _analyze_complex_rhythm(self, speeds, dt_values):
        """Analyze complex rhythmic patterns in speed changes."""
        if len(speeds) < 10:
            return {'complexity': 0.0, 'periodicity': 0.0, 'speed_oscillation': 0.0}
        
        # Speed oscillation analysis
        speed_changes = []
        for i in range(1, len(speeds)):
            change = speeds[i] - speeds[i-1]
            speed_changes.append(change)
        
        # Look for oscillatory patterns
        positive_changes = len([c for c in speed_changes if c > 0.001])
        negative_changes = len([c for c in speed_changes if c < -0.001])
        total_changes = len(speed_changes)
        
        if total_changes > 0:
            oscillation_balance = 1.0 - abs(positive_changes - negative_changes) / total_changes
        else:
            oscillation_balance = 0.0
        
        # Rhythm complexity (how varied are the speed patterns)
        if len(speeds) > 5:
            speed_avg = sum(speeds) / len(speeds)
            speed_variance = sum((s - speed_avg)**2 for s in speeds) / len(speeds)
            complexity = min(1.0, speed_variance * 10)  # Scale appropriately
        else:
            complexity = 0.0
        
        # Look for periodic patterns in speed
        periodicity = 0.0
        if len(speeds) > 20:
            # Simple periodicity detection - look for recurring patterns
            window_size = min(10, len(speeds) // 4)
            correlations = []
            
            for offset in range(1, min(window_size, len(speeds) // 2)):
                correlation = 0.0
                count = 0
                for i in range(len(speeds) - offset):
                    if i + offset < len(speeds):
                        correlation += abs(speeds[i] - speeds[i + offset])
                        count += 1
                if count > 0:
                    correlations.append(1.0 / (1.0 + correlation / count))
            
            periodicity = max(correlations) if correlations else 0.0
        
        return {
            'complexity': complexity,
            'periodicity': periodicity,
            'speed_oscillation': oscillation_balance
        }
    
    def _analyze_quadrant_preferences(self, x_positions, y_positions):
        """Analyze which quadrants of the canvas are preferred."""
        quadrants = [0, 0, 0, 0]  # TL, TR, BL, BR
        for x, y in zip(x_positions, y_positions):
            if x < 0.5 and y < 0.5:
                quadrants[0] += 1  # Top-left
            elif x >= 0.5 and y < 0.5:
                quadrants[1] += 1  # Top-right
            elif x < 0.5 and y >= 0.5:
                quadrants[2] += 1  # Bottom-left
            else:
                quadrants[3] += 1  # Bottom-right
        
        total = sum(quadrants)
        return [q / total if total > 0 else 0.25 for q in quadrants]
    
    def _analyze_speed_rhythm(self, speeds):
        """Analyze rhythmic patterns in speed changes."""
        if len(speeds) < 10:
            return {'rhythm_factor': 1.0, 'burst_tendency': 0.5}
        
        # Look for speed bursts vs steady movement
        speed_changes = []
        for i in range(1, len(speeds)):
            change = abs(speeds[i] - speeds[i-1])
            speed_changes.append(change)
        
        avg_change = sum(speed_changes) / len(speed_changes)
        max_change = max(speed_changes)
        
        return {
            'rhythm_factor': avg_change / (max_change + 0.001),  # How rhythmic vs random
            'burst_tendency': len([c for c in speed_changes if c > avg_change * 2]) / len(speed_changes)
        }
    
    def _find_favorite_positions(self, x_positions, y_positions):
        """Find positions where the cursor lingered or returned to."""
        if len(x_positions) < 20:
            return []
        
        favorites = []
        position_clusters = {}
        
        # Cluster nearby positions
        for x, y in zip(x_positions, y_positions):
            # Round to create position clusters
            cluster_x = round(x * 10) / 10
            cluster_y = round(y * 10) / 10
            key = (cluster_x, cluster_y)
            
            if key not in position_clusters:
                position_clusters[key] = 0
            position_clusters[key] += 1
        
        # Find clusters with significant dwell time
        total_points = len(x_positions)
        threshold = max(3, total_points * 0.05)  # At least 5% of time or 3 points
        
        for (x, y), count in position_clusters.items():
            if count >= threshold:
                favorites.append({'x': x, 'y': y, 'strength': count / total_points})
        
        # Sort by strength and keep top 5
        favorites.sort(key=lambda f: f['strength'], reverse=True)
        return favorites[:5]
    
    def _analyze_directional_preferences(self, directions):
        """Analyze preferred movement directions."""
        if not directions:
            return 0.0
        
        # Calculate average direction (circular mean)
        sum_x = sum(math.cos(d) for d in directions)
        sum_y = sum(math.sin(d) for d in directions)
        
        if sum_x == 0 and sum_y == 0:
            return 0.0
        
        return math.atan2(sum_y, sum_x)
    
    def _analyze_pause_patterns(self, movements):
        """Analyze where and how often pauses occur."""
        if len(movements) < 10:
            return {'pause_frequency': 0.1, 'pause_duration': 0.5}
        
        pauses = []
        last_pos = (movements[0]['x'], movements[0]['y'])
        pause_start = None
        
        for i, movement in enumerate(movements[1:], 1):
            pos = (movement['x'], movement['y'])
            distance = math.sqrt((pos[0] - last_pos[0])**2 + (pos[1] - last_pos[1])**2)
            
            if distance < 0.01:  # Very small movement = pause
                if pause_start is None:
                    pause_start = i
            else:
                if pause_start is not None:
                    pause_duration = i - pause_start
                    pauses.append(pause_duration)
                    pause_start = None
            
            last_pos = pos
        
        if not pauses:
            return {'pause_frequency': 0.1, 'pause_duration': 0.5}
        
        return {
            'pause_frequency': len(pauses) / len(movements),
            'pause_duration': sum(pauses) / len(pauses) / len(movements)  # Normalized
        }
    
    def start_vector_generation(self):
        """Start vector-based movement generation that creates new movements in your style."""
        if self.generating:
            # Stop generation
            self.stop_vector_generation()
            return
            
        if self.current_emotional_state not in self.learned_patterns:
            self.record_status.config(text="No movement pattern learned for this emotion!", foreground="orange")
            print(f"⚠️ No learned pattern for {self.current_emotional_state}. Record some movement first!")
            return
            
        # Check if we have vectors to work with
        if self.current_emotional_state not in self.recorded_vectors:
            self.record_status.config(text="No vectors recorded for this emotion!", foreground="orange")
            return
            
        # Initialize vector-based generation
        self.generating = True
        self.generation_start_time = time.time()
        
        # Create vector generator
        self.vector_generator = VectorMovementGenerator(
            self.recorded_vectors[self.current_emotional_state],
            self.learned_patterns[self.current_emotional_state]
        )
        
        # Set starting position
        self.vector_generator.set_position(self.mouse_x, self.mouse_y)
        
        self.generate_btn.config(text="⏹️ Stop Generating")
        self.record_status.config(text=f"Generating {self.current_emotional_state} movement...", foreground="purple")
        
        print(f"🧠 Started vector-based movement generation for {self.current_emotional_state}")
        print(f"🎯 Using learned patterns to create new movements in your style!")
        print(f"� Will preserve natural timing and pacing with gentle variations")
        print(f"🌊 Continuous loop with smooth interpolation - no jumping between segments!")
    
    def stop_vector_generation(self):
        """Stop vector generation and clean up."""
        self.generating = False
        self.vector_generator = None
        self.generate_btn.config(text="🧠 Generate Movement")
        self.record_status.config(text="Generation stopped", foreground="gray")
        
        print("⏹️ Vector-based movement generation stopped")
    
    def update_vector_generation(self):
        """Update cursor position using vector-based generation."""
        if not self.generating or not self.vector_generator:
            return
            
        current_time = time.time()
        dt = current_time - getattr(self, 'last_generation_time', current_time)
        self.last_generation_time = current_time
        
        # Generate next movement step
        new_x, new_y = self.vector_generator.generate_next_step(dt)
        
        # Update cursor position
        self.mouse_x = new_x
        self.mouse_y = new_y
if __name__ == "__main__":
    print("🚀 Starting Clean Emotional Hand Control...")
    app = CleanCursorInterface()
    app.root.mainloop()
