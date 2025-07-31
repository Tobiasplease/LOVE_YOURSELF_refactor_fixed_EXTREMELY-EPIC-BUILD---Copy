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
import glob
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
        
        # SPACEBAR HOTKEY for recording - REQUESTED FEATURE!
        self.root.bind("<KeyPress-space>", self.on_spacebar_press)
        self.root.focus_set()  # Ensure window can receive keyboard events
        
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
        
        # Person detection simulation for testing integration behavior
        self.person_detected = tk.BooleanVar(value=False)
        self.last_detection_time = 0
        self.detection_cooldown = 60.0  # 60 seconds between detections
        self.freeze_duration = 0  # Current freeze remaining
        self.min_freeze_duration = tk.DoubleVar(value=2.0)  # Adjustable minimum
        self.max_freeze_duration = tk.DoubleVar(value=6.0)  # Adjustable maximum
        self.is_frozen = False
        self.freeze_start_time = 0
        self.pre_freeze_mouse_pos = (0.5, 0.5)  # Store position before freeze
        self.is_thawing = False  # Smooth transition back
        self.thaw_start_time = 0
        self.thaw_duration = 2.0  # 2 seconds to smoothly transition back
        
        # Mouse tracking (KEEP - this works)
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        
        # Animation state (KEEP - this works)
        self.running = False
        self.last_time = time.time()
        self.last_send_time = 0
        self.send_interval = 0.016  # 60 Hz
        self.position_threshold = 1.0
        
        # 5 CORE EMOTIONAL STATES - Matching main script's mood system
        self.emotional_states = {
            'energized_engaged': EmotionalState('Energized & Deeply Engaged', mood_factor=0.8, energy_factor=1.0, focus_factor=0.8),
            'alert_curious': EmotionalState('Alert & Curious', mood_factor=0.6, energy_factor=0.8, focus_factor=0.6),
            'calm_observant': EmotionalState('Calm & Observant', mood_factor=0.3, energy_factor=0.5, focus_factor=0.7),
            'quiet_detached': EmotionalState('Quiet & Detached', mood_factor=-0.3, energy_factor=0.2, focus_factor=0.4),
            'withdrawn_distant': EmotionalState('Withdrawn & Distant', mood_factor=-0.7, energy_factor=0.1, focus_factor=0.3)
        }
        
        self.current_emotional_state = 'calm_observant'  # Default to neutral middle state
        self.logging_movement = False
        
        # Recording/playback state - MARKOV CHAIN BASED!
        self.recording = False
        self.playing_back = False
        self.recorded_movements = {}  # emotion_name -> list of movements (positions for exact playback)
        self.markov_chains = {}      # emotion_name -> position transition chains
        self.playback_start_time = 0
        self.current_playback = []
        self.record_start_time = 0
        
        # Time-based recording state (captures stillness!)
        self.recording_timer = None
        self.record_interval = 0.025  # 40 Hz - much higher resolution for easing capture
        self.recorded_positions = []  # Current recording session positions
        
        # Markov chain generation state
        self.generating = False
        self.generation_start_time = 0
        self.current_markov_state = None  # Current state in generation
        self.generation_timer = None
        self.generation_speed = 0.02  # Much faster: ~50 Hz for ultra-smooth movement
        self.generation_smoothing = True  # Enable position smoothing
        self.last_generated_pos = (0.5, 0.5)  # For smooth interpolation
        self.target_generated_pos = (0.5, 0.5)  # Target position for easing
        self.generation_easing_factor = 0.3  # Smooth easing between positions
        
        self.setup_ui()
        self.start_control_loop()
        
        # Load any previously saved movements for persistence
        self.load_saved_movements()
        
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
        
        # Record button - FIXED WIDTH to prevent layout shifts
        self.record_btn = ttk.Button(record_frame, text="🎬 Record Movement (2min)", 
                                   command=self.start_recording, width=22)  # Increased to accommodate longer text
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        # Playback button
        self.playback_btn = ttk.Button(record_frame, text="▶️ Play Back", 
                                     command=self.start_playback, width=15)
        self.playback_btn.pack(side=tk.LEFT, padx=5)
        
        # GENERATIVE playback button - MARKOV CHAIN MAGIC!
        self.generate_btn = ttk.Button(record_frame, text="🧠 Generate (Markov)", 
                                     command=self.start_markov_generation, width=18)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Status - FIXED WIDTH to prevent layout shifts
        self.record_status = ttk.Label(record_frame, text="Ready to record (Spacebar)", 
                                     foreground="gray", width=35)  # Fixed width for longest expected text
        self.record_status.pack(side=tk.LEFT, padx=10)
        
        # Markov status display - FIXED WIDTH to prevent shifting
        self.markov_status = ttk.Label(record_frame, text="No chains built", 
                                     foreground="gray", font=("Arial", 8), width=20)
        self.markov_status.pack(side=tk.LEFT, padx=5)
        
        # Recording state
        self.recording = False
        self.playing_back = False
        self.recorded_movements = {}  # emotion_name -> list of movements
        self.playback_start_time = 0
        self.current_playback = []
        
        # === HAND CONTROL AREA - ABSOLUTELY FIXED DIMENSIONS ===
        control_frame = ttk.LabelFrame(self.scrollable_frame, text="🎯 Hand Control Area")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a stable container frame for the canvas - PROPERLY CENTERED!
        canvas_container = tk.Frame(control_frame, bg=self.colors['bg_frame'])
        canvas_container.pack(pady=5, fill=tk.X)  # Fill horizontally to center content
        
        # ABSOLUTELY FIXED CANVAS - No shifting or resizing allowed!
        self.canvas_width = 480  # Store as instance variables for consistency
        self.canvas_height = 200
        self.canvas = tk.Canvas(canvas_container, 
                               bg="black", 
                               height=self.canvas_height, 
                               width=self.canvas_width,
                               highlightthickness=2,
                               highlightbackground="white")
        # Use grid for proper centering instead of pack
        canvas_container.grid_columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, pady=10)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        
        # Prevent any canvas resizing by binding to configure events
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # === CONTROL MODES - Simplified without physics ===
        mode_frame = ttk.LabelFrame(self.scrollable_frame, text="🎛️ Control Modes")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        reverse_cb = ttk.Checkbutton(mode_frame, text="🔄 Reverse Vertical", 
                                   variable=self.reverse_vertical)
        reverse_cb.pack(side=tk.LEFT, padx=5)
        
        # Person detection simulation toggle
        person_cb = ttk.Checkbutton(mode_frame, text="👤 Person Detected (Sim)", 
                                  variable=self.person_detected,
                                  command=self.on_person_detection_toggle)
        person_cb.pack(side=tk.LEFT, padx=10)
        
        reset_btn = ttk.Button(mode_frame, text="🎯 Reset to Center", 
                             command=self.reset_to_center)
        reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # === FREEZE BEHAVIOR CONTROLS ===
        freeze_frame = ttk.LabelFrame(self.scrollable_frame, text="❄️ Freeze Behavior")
        freeze_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Min Freeze Duration
        ttk.Label(freeze_frame, text="Min Freeze Duration:").grid(row=0, column=0, sticky=tk.W, padx=5)
        min_freeze_scale = ttk.Scale(freeze_frame, from_=1.0, to=10.0, variable=self.min_freeze_duration, orient=tk.HORIZONTAL)
        min_freeze_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        min_freeze_label = ttk.Label(freeze_frame, text="2.0s")
        min_freeze_label.grid(row=0, column=2, padx=5)
        self.min_freeze_duration.trace_add("write", lambda *args: min_freeze_label.config(text=f"{self.min_freeze_duration.get():.1f}s"))
        
        # Max Freeze Duration
        ttk.Label(freeze_frame, text="Max Freeze Duration:").grid(row=1, column=0, sticky=tk.W, padx=5)
        max_freeze_scale = ttk.Scale(freeze_frame, from_=2.0, to=15.0, variable=self.max_freeze_duration, orient=tk.HORIZONTAL)
        max_freeze_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        max_freeze_label = ttk.Label(freeze_frame, text="6.0s")
        max_freeze_label.grid(row=1, column=2, padx=5)
        self.max_freeze_duration.trace_add("write", lambda *args: max_freeze_label.config(text=f"{self.max_freeze_duration.get():.1f}s"))
        
        freeze_frame.columnconfigure(1, weight=1)
        
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
    
    def on_canvas_configure(self, event):
        """Prevent canvas from being resized and maintain fixed dimensions."""
        # Force canvas to maintain exact dimensions
        if event.width != self.canvas_width or event.height != self.canvas_height:
            self.canvas.config(width=self.canvas_width, height=self.canvas_height)
            print(f"🔒 Canvas size locked: {self.canvas_width}x{self.canvas_height}")
    
    def on_mouse_move(self, event):
        """Handle mouse movement in canvas - ABSOLUTELY FIXED coordinates."""
        # Use stored canvas dimensions for perfect consistency
        canvas_width = self.canvas_width
        canvas_height = self.canvas_height
        
        # Always track raw mouse position for thaw transitions
        raw_mouse_x = max(0, min(event.x, canvas_width)) / canvas_width  # Clamp to canvas bounds
        raw_mouse_y = max(0, min(event.y, canvas_height)) / canvas_height
        self._current_raw_mouse_x = raw_mouse_x
        self._current_raw_mouse_y = raw_mouse_y
        
        # Only update actual mouse position if not frozen or during thaw
        if not self.is_frozen or self.is_thawing:
            old_x, old_y = self.mouse_x, self.mouse_y
            # During thaw, the control loop handles smooth blending
            if not self.is_thawing:
                self.mouse_x = raw_mouse_x
                self.mouse_y = raw_mouse_y
        
        # Debug output for first few movements
        if hasattr(self, 'move_count'):
            self.move_count += 1
        else:
            self.move_count = 1
        
        if self.move_count < 5:
            freeze_status = " [FROZEN]" if self.is_frozen else " [THAWING]" if self.is_thawing else ""
            print(f"🎯 Mouse move {self.move_count}: ({self.mouse_x:.3f}, {self.mouse_y:.3f}) canvas: {canvas_width}x{canvas_height} event: ({event.x}, {event.y}){freeze_status}")
        
        # Record movement if recording - TIME-BASED CONTINUOUS SAMPLING!
        if self.recording and not self.is_frozen:
            current_time = time.time()
            relative_time = current_time - self.record_start_time
            
            # Store position for exact playback (keep this for compatibility)
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
            
            # Note: Time-based recording happens in record_position_sample() called by timer
        
        # IMPORTANT: Don't interfere with generative movement!
        elif self.generating:
            # Ignore mouse input during generation to prevent interference
            return
    
    def on_mouse_click(self, event):
        """Handle mouse click in canvas."""
        self.on_mouse_move(event)  # Update position
    
    def on_spacebar_press(self, event):
        """Handle spacebar press for recording toggle - REQUESTED HOTKEY!"""
        if self.recording:
            self.stop_recording()
            print("⏹️ Spacebar: Stopped recording")
        else:
            self.start_recording()
            print("🎬 Spacebar: Started recording")
    
    def on_person_detection_toggle(self):
        """Handle person detection toggle for testing freeze behavior."""
        current_time = time.time()
        
        if self.person_detected.get():
            # Check cooldown
            if current_time - self.last_detection_time < self.detection_cooldown:
                remaining = self.detection_cooldown - (current_time - self.last_detection_time)
                print(f"👤 Person detection on cooldown - {remaining:.1f}s remaining")
                self.person_detected.set(False)  # Reset toggle
                return
            
            # Start freeze
            self.trigger_freeze()
        else:
            # Manual unfreeze
            if self.is_frozen:
                self.end_freeze()
    
    def trigger_freeze(self):
        """Trigger a freeze response to person detection."""
        current_time = time.time()
        
        # Store current position
        self.pre_freeze_mouse_pos = (self.mouse_x, self.mouse_y)
        
        # Start freeze with adjustable duration
        self.is_frozen = True
        self.is_thawing = False
        self.freeze_start_time = current_time
        min_duration = self.min_freeze_duration.get()
        max_duration = self.max_freeze_duration.get()
        self.freeze_duration = random.uniform(min_duration, max_duration)
        self.last_detection_time = current_time
        
        print(f"❄️ FREEZE triggered for {self.freeze_duration:.1f}s at position ({self.mouse_x:.2f}, {self.mouse_y:.2f})")
        
        # Schedule auto-unfreeze
        freeze_duration_ms = int(self.freeze_duration * 1000)
        self.root.after(freeze_duration_ms, self.start_thaw)
    
    def start_thaw(self):
        """Begin smooth transition back to movement."""
        if not self.is_frozen:
            return
            
        self.is_thawing = True
        self.thaw_start_time = time.time()
        
        print(f"🔄 Starting smooth thaw transition back to {self.current_emotional_state}")
        
        # Schedule end of thaw
        thaw_duration_ms = int(self.thaw_duration * 1000)
        self.root.after(thaw_duration_ms, self.end_freeze)
    
    def end_freeze(self):
        """End freeze and begin transition back to movement."""
        if not self.is_frozen:
            return
            
        self.is_frozen = False
        self.is_thawing = False
        self.person_detected.set(False)  # Reset toggle
        
        print(f"✅ Freeze complete - resumed normal {self.current_emotional_state} movement")
    
    def reset_to_center(self):
        """Reset cursor and servos to center position."""
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_targets = [90.0] * self.num_fingers
        
        # End any freeze state
        if self.is_frozen:
            self.end_freeze()
            
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
        
        # Handle freeze state
        if self.is_frozen:
            if self.is_thawing:
                # Smooth transition back to movement
                thaw_elapsed = current_time - self.thaw_start_time
                thaw_progress = min(1.0, thaw_elapsed / self.thaw_duration)
                
                # Smooth easing back to pre-freeze position (or allow new movement)
                if not self.generating and not self.playing_back:
                    # Gradually allow manual mouse control again
                    # Start from frozen position and smoothly enable input
                    freeze_x, freeze_y = self.pre_freeze_mouse_pos
                    
                    # Smooth transition - gradually blend from freeze position to current mouse
                    current_mouse_x = getattr(self, '_current_raw_mouse_x', self.mouse_x)
                    current_mouse_y = getattr(self, '_current_raw_mouse_y', self.mouse_y)
                    
                    blend_factor = thaw_progress * thaw_progress  # Ease-in curve
                    self.mouse_x = freeze_x + (current_mouse_x - freeze_x) * blend_factor
                    self.mouse_y = freeze_y + (current_mouse_y - freeze_y) * blend_factor
            else:
                # Hard freeze - maintain exact position
                self.mouse_x, self.mouse_y = self.pre_freeze_mouse_pos
        else:
            # Normal movement updates
            
            # Update playback if active
            if self.playing_back:
                self.update_playback()
            
            # Update generative playback if active - MARKOV CHAIN!
            if self.generating:
                self.update_markov_generation()
        
        # Update canvas visualization (always show current state)
        self.update_canvas()
        
        # Calculate finger targets from cursor position (even during freeze - maintains position)
        self.calculate_finger_targets()
        
        # Direct control - immediate response!
        self.finger_positions = self.finger_targets.copy()
        
        if hasattr(self, 'direct_count'):
            self.direct_count += 1
        else:
            self.direct_count = 1
        
        if self.direct_count < 5 or self.direct_count % 30 == 0:
            freeze_status = " [FROZEN]" if self.is_frozen else ""
            print(f"🎯 Direct control {self.direct_count}: positions={[f'{p:.1f}' for p in self.finger_positions]} from targets={[f'{t:.1f}' for t in self.finger_targets]}{freeze_status}")
        
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
        """Update canvas visualization - ABSOLUTELY FIXED coordinates."""
        # FORCE complete canvas clearing to prevent dots/trails
        self.canvas.delete("all")
        self.canvas.update_idletasks()  # Force immediate clearing
        
        # Use stored canvas dimensions for perfect consistency
        canvas_width = self.canvas_width
        canvas_height = self.canvas_height
        
        cursor_x = self.mouse_x * canvas_width
        cursor_y = self.mouse_y * canvas_height
        
        # Draw main cursor - single red dot
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
        
        # Cursor position indicator with pixel coordinates for debugging
        cursor_text = f"Cursor: ({self.mouse_x:.2f}, {self.mouse_y:.2f}) px:({cursor_x:.0f}, {cursor_y:.0f})"
        self.canvas.create_text(10, 50, text=cursor_text, fill="cyan", anchor="nw",
                              font=("Arial", 10))
        
        # Canvas size indicator for debugging
        size_text = f"Canvas: {canvas_width}x{canvas_height}"
        self.canvas.create_text(10, 70, text=size_text, fill="lightgray", anchor="nw",
                              font=("Arial", 8))
        
        # Recording/playback indicators
        if self.is_frozen:
            if self.is_thawing:
                thaw_elapsed = time.time() - self.thaw_start_time
                thaw_progress = min(1.0, thaw_elapsed / self.thaw_duration)
                self.canvas.create_text(canvas_width-10, 10, text=f"🔄 THAWING ({thaw_progress*100:.0f}%)", 
                                      fill="lightblue", anchor="ne", font=("Arial", 14, "bold"))
            else:
                remaining_freeze = self.freeze_duration - (time.time() - self.freeze_start_time)
                if remaining_freeze > 0:
                    self.canvas.create_text(canvas_width-10, 10, text=f"❄️ FROZEN ({remaining_freeze:.1f}s)", 
                                          fill="cyan", anchor="ne", font=("Arial", 14, "bold"))
        elif self.recording:
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
        """Start time-based recording that captures both movement AND stillness."""
        if self.recording:
            # Stop recording
            self.stop_recording()
            return
            
        # Start recording
        self.recording = True
        self.record_start_time = time.time()
        self.recorded_movements[self.current_emotional_state] = []
        self.recorded_positions = []  # Reset current session
        
        self.record_btn.config(text="⏹️ Stop Recording")
        self.record_status.config(text=f"Recording {self.current_emotional_state}... (Spacebar to stop)", foreground="red")
        self.markov_status.config(text="Capturing positions...", foreground="orange")
        
        print(f"🎬 Started TIME-BASED recording for {self.current_emotional_state}")
        print(f"⏰ Sampling at {1/self.record_interval:.0f} Hz (captures easing motions!)")
        
        # Start time-based sampling timer
        self.start_recording_timer()
        
        # Auto-stop after 2 minutes for much more training data
        self.root.after(120000, self.auto_stop_recording)
    
    def start_recording_timer(self):
        """Start the time-based recording timer that samples positions continuously."""
        if self.recording:
            self.record_position_sample()
            # Schedule next sample
            interval_ms = int(self.record_interval * 1000)
            self.recording_timer = self.root.after(interval_ms, self.start_recording_timer)
    
    def record_position_sample(self):
        """Record a single position sample - captures both movement and stillness."""
        if not self.recording or self.is_frozen:
            return
            
        current_time = time.time()
        relative_time = current_time - self.record_start_time
        
        # Create position state for Markov chain with ULTRA HIGH RESOLUTION
        # Use 80x80 grid for capturing subtle easing motions (6400 possible states!)
        grid_size = 80  # Ultra high resolution for easing capture
        grid_x = int(self.mouse_x * grid_size)
        grid_y = int(self.mouse_y * grid_size)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        
        # Also capture velocity information for better easing reproduction
        velocity_x = 0.0
        velocity_y = 0.0
        if len(self.recorded_positions) > 0:
            prev_pos = self.recorded_positions[-1]
            dt = current_time - prev_pos['time']
            if dt > 0:
                velocity_x = (self.mouse_x - prev_pos['x']) / dt
                velocity_y = (self.mouse_y - prev_pos['y']) / dt
        
        position_state = {
            'time': current_time,
            'relative_time': relative_time,
            'x': self.mouse_x,
            'y': self.mouse_y,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_state': (grid_x, grid_y),  # This becomes our Markov state
            'velocity_x': velocity_x,  # Capture movement velocity for easing
            'velocity_y': velocity_y,
            'speed': math.sqrt(velocity_x**2 + velocity_y**2),  # Overall speed
            'finger_positions': self.finger_positions.copy()
        }
        
        self.recorded_positions.append(position_state)
        
        # Debug first few samples
        if len(self.recorded_positions) <= 5:
            print(f"📍 Sample {len(self.recorded_positions)}: ({self.mouse_x:.3f}, {self.mouse_y:.3f}) -> grid ({grid_x}, {grid_y}) [80x80 grid] speed: {math.sqrt(velocity_x**2 + velocity_y**2):.3f}")
    
    def auto_stop_recording(self):
        """Auto-stop recording after 45 seconds."""
        if self.recording:
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and build Markov chain from positions."""
        if not self.recording:
            return
            
        self.recording = False
        
        # Stop the recording timer
        if self.recording_timer:
            self.root.after_cancel(self.recording_timer)
            self.recording_timer = None
            
        duration = time.time() - self.record_start_time
        sample_count = len(self.recorded_positions)
        
        self.record_btn.config(text="🎬 Record Movement (45s)")
        self.record_status.config(text=f"Recorded {sample_count} samples in {duration:.1f}s", foreground="green")
        
        print(f"🎬 Stopped recording. Captured {sample_count} position samples in {duration:.1f} seconds")
        print(f"📊 Sample rate: {sample_count/duration:.1f} Hz")
        
        # Build Markov chain from recorded positions
        self.build_markov_chain()
        
        # Save to file
        self.save_recording()
    
    def build_markov_chain(self):
        """Build Markov chain from recorded position samples."""
        if len(self.recorded_positions) < 2:
            print("⚠️ Not enough samples to build Markov chain")
            return
            
        print(f"🔗 Building Markov chain from {len(self.recorded_positions)} samples...")
        
        # Initialize transition matrix
        transitions = {}
        
        # Build state transitions
        for i in range(len(self.recorded_positions) - 1):
            current_state = self.recorded_positions[i]['grid_state']
            next_state = self.recorded_positions[i + 1]['grid_state']
            
            if current_state not in transitions:
                transitions[current_state] = {}
                
            if next_state not in transitions[current_state]:
                transitions[current_state][next_state] = 0
                
            transitions[current_state][next_state] += 1
        
        # Convert counts to probabilities
        for state in transitions:
            total = sum(transitions[state].values())
            for next_state in transitions[state]:
                transitions[state][next_state] /= total
        
        # Store the Markov chain with ultra high resolution
        self.markov_chains[self.current_emotional_state] = {
            'transitions': transitions,
            'sample_count': len(self.recorded_positions),
            'duration': self.recorded_positions[-1]['time'] - self.recorded_positions[0]['time'],
            'grid_size': 80,  # Ultra high resolution grid
            'unique_states': len(transitions),
            'velocity_data': [p for p in self.recorded_positions if 'velocity_x' in p]  # Store velocity info
        }
        
        unique_states = len(transitions)
        avg_transitions = sum(len(t) for t in transitions.values()) / len(transitions) if transitions else 0
        
        self.markov_status.config(text=f"{unique_states} states, {avg_transitions:.1f} avg transitions", foreground="green")
        
        print(f"✅ Built Markov chain: {unique_states} unique states, {avg_transitions:.1f} average transitions per state")
        print(f"🎯 Chain covers {unique_states}/{80*80} possible grid positions ({100*unique_states/6400:.1f}%)")
    
    def save_recording(self):
        """Save recorded movements AND Markov chain to file."""
        if self.current_emotional_state not in self.recorded_movements:
            return
            
        os.makedirs("movement_recordings", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"movement_recordings/{self.current_emotional_state}_{timestamp}.json"
        
        movements = self.recorded_movements[self.current_emotional_state]
        positions = self.recorded_positions
        markov_chain = self.markov_chains.get(self.current_emotional_state, {})
        
        data = {
            'emotion': self.current_emotional_state,
            'timestamp': timestamp,
            'movement_count': len(movements),
            'position_count': len(positions),
            'duration': positions[-1]['time'] - positions[0]['time'] if len(positions) > 1 else 0,
            'sample_rate': len(positions) / (positions[-1]['time'] - positions[0]['time']) if len(positions) > 1 else 0,
            'movements': movements,  # Keep for compatibility
            'positions': positions,  # Time-based samples
            'markov_chain': markov_chain  # The learned chain
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"💾 Saved recording with Markov chain to {filename}")
        print(f"📈 Chain has {markov_chain.get('unique_states', 0)} states from {len(positions)} samples")
    
    def post_process_rhythm_analysis(self):
        """Post-process recorded vectors to detect rhythm, stillness patterns, and temporal dynamics."""
        if self.current_emotional_state not in self.recorded_vectors:
            return
            
        vectors = self.recorded_vectors[self.current_emotional_state]
        if len(vectors) < 10:  # Need at least 10 vectors for pattern analysis
            return
            
        print(f"🎵 ENHANCED: Analyzing temporal rhythm patterns in {len(vectors)} vectors...")
        
        # First pass: Detect stillness periods and micro-movements
        self.detect_stillness_patterns(vectors)
        
        # Calculate direction changes between consecutive vectors
        for i in range(1, len(vectors)):
            prev_dir = vectors[i-1]['direction']
            curr_dir = vectors[i]['direction']
            
            # Calculate angular difference (shortest path)
            diff = curr_dir - prev_dir
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi
                
            vectors[i]['direction_change'] = abs(diff)
            
            # Speed change rate
            if i > 0:
                speed_diff = vectors[i]['speed'] - vectors[i-1]['speed']
                vectors[i]['speed_change_rate'] = speed_diff / vectors[i]['dt']
        
        # Detect movement impulses (sudden movement after stillness)
        self.detect_movement_impulses(vectors)
        
        # Detect circular patterns by looking for consistent direction changes
        circular_threshold = math.pi / 6  # 30 degrees
        for i in range(2, len(vectors) - 2):
            # Look at direction changes over a small window
            window_changes = [vectors[j]['direction_change'] for j in range(i-2, i+3)]
            avg_change = sum(window_changes) / len(window_changes)
            
            # If direction is changing consistently and moderately, it's circular
            if circular_threshold / 2 < avg_change < circular_threshold * 1.5:
                vectors[i]['circular_component'] = avg_change
        
        # Detect easing (gradual acceleration/deceleration)
        for i in range(5, len(vectors) - 5):  # Need buffer for trend analysis
            # Look at speed trend over nearby vectors
            before_speeds = [vectors[j]['speed'] for j in range(i-5, i)]
            after_speeds = [vectors[j]['speed'] for j in range(i, i+5)]
            
            before_avg = sum(before_speeds) / len(before_speeds)
            after_avg = sum(after_speeds) / len(after_speeds)
            current_speed = vectors[i]['speed']
            
            # Easing in: speed gradually increasing into this point
            if before_avg < current_speed < after_avg * 1.2:
                vectors[i]['easing_in'] = True
            
            # Easing out: speed gradually decreasing from this point  
            if before_avg * 1.2 > current_speed > after_avg:
                vectors[i]['easing_out'] = True
        
        # Detect rhythmic patterns by analyzing time intervals between similar movements
        self.detect_temporal_rhythms(vectors)
        
        # Detect breathing-like patterns (expansion/contraction with pauses)
        self.detect_breathing_patterns(vectors)
        
        print(f"✅ Enhanced temporal rhythm analysis complete - stillness and movement patterns detected")
    
    def detect_stillness_patterns(self, vectors):
        """Detect periods of stillness, micro-movements, and position holding."""
        stillness_threshold = 0.002  # Very low movement threshold
        micro_movement_threshold = 0.008  # Slightly higher for micro-movements
        
        # Track stillness periods
        current_stillness_start = None
        
        for i, vector in enumerate(vectors):
            # Check if we're in a period of very low movement
            if vector['speed'] < stillness_threshold:
                if current_stillness_start is None:
                    current_stillness_start = vector['time']
                
                # Calculate how long we've been still
                stillness_duration = vector['time'] - current_stillness_start
                vector['stillness_duration'] = stillness_duration
                vector['position_hold'] = stillness_duration > 0.5  # Holding for 0.5+ seconds
                
                # Check for micro-tremors within stillness
                if stillness_threshold < vector['speed'] < micro_movement_threshold:
                    vector['micro_tremor'] = True
                    
            else:
                # Movement detected - end stillness period
                if current_stillness_start is not None:
                    stillness_duration = vector['time'] - current_stillness_start
                    # Mark this as a movement impulse if coming out of stillness
                    if stillness_duration > 0.3:  # After 0.3+ seconds of stillness
                        vector['movement_impulse'] = True
                        
                current_stillness_start = None
                vector['stillness_duration'] = 0.0
        
        print(f"   📍 Detected stillness patterns: {sum(1 for v in vectors if v.get('position_hold', False))} position holds")
        print(f"   🤏 Detected micro-movements: {sum(1 for v in vectors if v.get('micro_tremor', False))} micro tremors")
        print(f"   ⚡ Detected movement impulses: {sum(1 for v in vectors if v.get('movement_impulse', False))} impulses")
    
    def detect_movement_impulses(self, vectors):
        """Detect sudden movements that break stillness patterns."""
        for i in range(5, len(vectors)):
            # Look back for recent stillness
            recent_vectors = vectors[i-5:i]
            recent_avg_speed = sum(v['speed'] for v in recent_vectors) / len(recent_vectors)
            
            # If recent movement was very low but current is higher
            if recent_avg_speed < 0.01 and vectors[i]['speed'] > recent_avg_speed * 3:
                vectors[i]['movement_impulse'] = True
                
                # Calculate rhythmic interval (time since last impulse)
                for j in range(i-1, -1, -1):
                    if vectors[j].get('movement_impulse', False):
                        vectors[i]['rhythmic_interval'] = vectors[i]['time'] - vectors[j]['time']
                        break
    
    def detect_temporal_rhythms(self, vectors):
        """Detect rhythmic timing patterns - intervals between movements."""
        # Find significant movement moments (impulses or speed peaks)
        movement_moments = []
        
        for i, vector in enumerate(vectors):
            if (vector.get('movement_impulse', False) or 
                vector['speed'] > sum(v['speed'] for v in vectors) / len(vectors) * 1.5):
                movement_moments.append((i, vector['time']))
        
        if len(movement_moments) > 3:
            # Calculate intervals between movement moments
            intervals = []
            for i in range(1, len(movement_moments)):
                interval = movement_moments[i][1] - movement_moments[i-1][1]
                intervals.append(interval)
            
            # Look for consistent timing patterns
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                consistent_intervals = [iv for iv in intervals if abs(iv - avg_interval) < avg_interval * 0.4]
                
                # If 60%+ of intervals are consistent, mark as rhythmic
                if len(consistent_intervals) > len(intervals) * 0.6:
                    print(f"   🎵 Detected rhythmic timing: {avg_interval:.2f}s average interval")
                    
                    # Mark vectors that are part of rhythmic pattern
                    for moment_idx, moment_time in movement_moments:
                        if moment_idx < len(vectors):
                            vectors[moment_idx]['rhythm_beat'] = True
    
    def detect_breathing_patterns(self, vectors):
        """Detect breathing-like expansion/contraction patterns with pauses."""
        if len(vectors) < 20:
            return
            
        # Look for cyclical movement away from and back to center positions
        center_returns = []
        
        for i in range(10, len(vectors) - 10):
            # Calculate distance from center over time
            before_distances = []
            after_distances = []
            
            for j in range(i-10, i):
                dist = math.sqrt((vectors[j]['end_x'] - 0.5)**2 + (vectors[j]['end_y'] - 0.5)**2)
                before_distances.append(dist)
            
            for j in range(i, i+10):
                dist = math.sqrt((vectors[j]['end_x'] - 0.5)**2 + (vectors[j]['end_y'] - 0.5)**2)
                after_distances.append(dist)
            
            before_avg = sum(before_distances) / len(before_distances)
            after_avg = sum(after_distances) / len(after_distances)
            
            # If we moved away from center and then back
            current_dist = math.sqrt((vectors[i]['end_x'] - 0.5)**2 + (vectors[i]['end_y'] - 0.5)**2)
            
            if (before_avg < current_dist > after_avg and 
                vectors[i].get('stillness_duration', 0) > 0.2):  # With a pause
                vectors[i]['breathing_like'] = True
                center_returns.append(i)
        
        if center_returns:
            print(f"   🫁 Detected breathing-like patterns: {len(center_returns)} expansion/contraction cycles")
    
    def load_saved_movements(self):
        """Load previously saved movements from disk for persistence."""
        if not os.path.exists("movement_recordings"):
            print("📁 No saved movements found - starting fresh")
            return
        
        loaded_count = 0
        for emotion_key in self.emotional_states.keys():
            # Find the most recent file for this emotion
            pattern = f"movement_recordings/{emotion_key}_*.json"
            files = []
            try:
                import glob
                files = glob.glob(pattern)
                if files:
                    # Get the most recent file
                    latest_file = max(files, key=os.path.getmtime)
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    # Load movements and vectors
                    if 'movements' in data and data['movements']:
                        self.recorded_movements[emotion_key] = data['movements']
                        loaded_count += 1
                    
                    if 'vectors' in data and data['vectors']:
                        self.recorded_vectors[emotion_key] = data['vectors']
                    
                    print(f"📂 Loaded {len(data.get('movements', []))} movements for {emotion_key}")
            except Exception as e:
                print(f"⚠️ Error loading {emotion_key}: {e}")
        
        if loaded_count > 0:
            print(f"✅ Loaded movements for {loaded_count} emotional states")
        else:
            print("📁 No valid saved movements found")

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
    
    def start_markov_generation(self):
        """Start Markov chain generation for current emotional state."""
        if self.generating:
            self.stop_markov_generation()
            return
            
        if self.current_emotional_state not in self.markov_chains:
            print(f"❌ No Markov chain available for {self.current_emotional_state}")
            self.markov_status.config(text="No chain for this emotion", foreground="red")
            return
            
        chain = self.markov_chains[self.current_emotional_state]
        if not chain.get('transitions'):
            print(f"❌ Empty Markov chain for {self.current_emotional_state}")
            return
            
        # Start generation
        self.generating = True
        self.generation_start_time = time.time()
        
        # Pick a random starting state from available states
        transitions = chain['transitions']
        start_state = random.choice(list(transitions.keys()))
        self.current_markov_state = start_state
        
        # Convert grid state back to mouse position
        grid_size = chain.get('grid_size', 80)  # Default to ultra high resolution
        grid_x, grid_y = start_state
        self.mouse_x = (grid_x + 0.5) / grid_size  # Center of grid cell
        self.mouse_y = (grid_y + 0.5) / grid_size
        
        self.generate_btn.config(text="⏹️ Stop Generation")
        self.markov_status.config(text="Generating movement...", foreground="purple")
        
        print(f"🎨 Started Markov generation for {self.current_emotional_state}")
        print(f"🎯 Starting from state {start_state} -> position ({self.mouse_x:.3f}, {self.mouse_y:.3f})")
        print(f"📊 Chain has {len(transitions)} states available")
        
        # Start generation timer
        self.start_generation_timer()
        
        # Auto-stop after 30 seconds
        self.root.after(30000, self.auto_stop_generation)
    
    def start_generation_timer(self):
        """Start the generation timer for Markov chain steps."""
        if self.generating:
            self.step_markov_generation()
            # Schedule next step
            interval_ms = int(self.generation_speed * 1000)
            self.generation_timer = self.root.after(interval_ms, self.start_generation_timer)
    
    def step_markov_generation(self):
        """Take one step in the Markov chain generation."""
        if not self.generating or self.current_emotional_state not in self.markov_chains:
            return
            
        chain = self.markov_chains[self.current_emotional_state]
        transitions = chain['transitions']
        
        if self.current_markov_state not in transitions:
            # Dead end - pick a new random state
            self.current_markov_state = random.choice(list(transitions.keys()))
            print(f"🔄 Dead end reached, jumping to {self.current_markov_state}")
            return
            
        # Get possible next states and their probabilities
        next_states = transitions[self.current_markov_state]
        
        # Choose next state based on probabilities
        states = list(next_states.keys())
        probabilities = list(next_states.values())
        
        # Weighted random choice
        next_state = random.choices(states, weights=probabilities)[0]
        
        # Update current state
        self.current_markov_state = next_state
        
        # Convert grid state to mouse position
        grid_size = chain.get('grid_size', 80)  # Default to ultra high resolution
        grid_x, grid_y = next_state
        new_x = (grid_x + 0.5) / grid_size
        new_y = (grid_y + 0.5) / grid_size
        
        # Enhanced smooth interpolation with easing for natural movement
        self.target_generated_pos = (new_x, new_y)
        easing_factor = self.generation_easing_factor  # Use configurable easing
        self.mouse_x = self.mouse_x * (1 - easing_factor) + new_x * easing_factor
        self.mouse_y = self.mouse_y * (1 - easing_factor) + new_y * easing_factor
    
    def update_markov_generation(self):
        """Update method called from control loop during generation."""
        # The actual generation happens in step_markov_generation() via timer
        # This method exists for compatibility with the control loop structure
        pass
    
    def stop_markov_generation(self):
        """Stop Markov chain generation."""
        if not self.generating:
            return
            
        self.generating = False
        
        # Stop the generation timer
        if self.generation_timer:
            self.root.after_cancel(self.generation_timer)
            self.generation_timer = None
            
        duration = time.time() - self.generation_start_time
        
        self.generate_btn.config(text="🧠 Generate (Markov)")
        self.markov_status.config(text="Generation stopped", foreground="gray")
        
        print(f"🎨 Stopped Markov generation after {duration:.1f} seconds")
    
    def auto_stop_generation(self):
        """Auto-stop generation after 30 seconds."""
        if self.generating:
            self.stop_markov_generation()
    
    def load_saved_movements(self):
        """Load previously saved movements and Markov chains from disk."""
        if not os.path.exists("movement_recordings"):
            print("📁 No movement_recordings directory found")
            return
        
        loaded_count = 0
        chain_count = 0
        
        for emotion_key in self.emotional_states.keys():
            # Find most recent recording for this emotion
            pattern = f"movement_recordings/{emotion_key}_*.json"
            files = glob.glob(pattern)
            if not files:
                continue
                
            # Get most recent file
            latest_file = max(files, key=os.path.getctime)
            
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    
                # Load movements (for playback compatibility)
                if 'movements' in data:
                    self.recorded_movements[emotion_key] = data['movements']
                    loaded_count += 1
                    
                # Load Markov chain
                if 'markov_chain' in data and data['markov_chain']:
                    self.markov_chains[emotion_key] = data['markov_chain']
                    chain_count += 1
                    print(f"🔗 Loaded Markov chain for {emotion_key}: {data['markov_chain'].get('unique_states', 0)} states")
                    
            except Exception as e:
                print(f"❌ Error loading {latest_file}: {e}")
                continue
        
        if loaded_count > 0:
            print(f"✅ Loaded {loaded_count} movement recordings and {chain_count} Markov chains")
            self.markov_status.config(text=f"{chain_count} chains loaded", foreground="green")
        else:
            print("📁 No saved recordings found")
            self.markov_status.config(text="No saved chains", foreground="gray")

    def start_playback(self):
        """Start playing back recorded movements for current emotional state."""
        if self.playing_back:
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
    
    def stop_playback(self):
        """Stop current playback."""
        self.playing_back = False
        self.current_playback = []
        print(f"⏹️ Stopped playback")
    
    def start_markov_generation(self):
        """Start Markov chain generation of movement patterns with smooth character preservation."""
        if self.current_emotional_state not in self.markov_chains:
            print(f"❌ No Markov chain available for {self.current_emotional_state}")
            self.markov_status.config(text="No chain for this emotion", foreground="red")
            return
        
        if self.generating:
            self.stop_markov_generation()
            return
        
        chain = self.markov_chains[self.current_emotional_state]
        transitions = chain['transitions']
        
        if not transitions:
            print(f"❌ Empty Markov chain for {self.current_emotional_state}")
            return
        
        # Start generation with better initialization
        self.generating = True
        self.generation_start_time = time.time()
        
        # Pick starting state near current position for smoother start
        current_grid_x = int(self.mouse_x * chain['grid_size'])
        current_grid_y = int(self.mouse_y * chain['grid_size'])
        current_grid_x = max(0, min(chain['grid_size'] - 1, current_grid_x))
        current_grid_y = max(0, min(chain['grid_size'] - 1, current_grid_y))
        
        # Try to find a nearby state that exists in the chain, otherwise pick random
        candidate_state = (current_grid_x, current_grid_y)
        if candidate_state in transitions:
            self.current_markov_state = candidate_state
        else:
            # Find closest existing state
            min_dist = float('inf')
            best_state = None
            for state in transitions.keys():
                dist = ((state[0] - current_grid_x) ** 2 + (state[1] - current_grid_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_state = state
            self.current_markov_state = best_state or random.choice(list(transitions.keys()))
        
        # Initialize smoothing
        self.last_generated_pos = (self.mouse_x, self.mouse_y)
        
        self.generate_btn.config(text="⏹️ Stop Generation")
        self.record_status.config(text=f"Generating {self.current_emotional_state} patterns...", foreground="purple")
        
        print(f"🎨 Started ENHANCED Markov generation for {self.current_emotional_state}")
        print(f"🎯 Starting from state {self.current_markov_state} (near current position)")
        print(f"⚡ Generation rate: {1/self.generation_speed:.1f} Hz for smooth movement")
        
        # Start generation timer
        self.start_generation_timer()
    
    def start_generation_timer(self):
        """Timer for Markov generation steps."""
        if self.generating:
            self.step_markov_generation()
            # Schedule next step
            interval_ms = int(self.generation_speed * 1000)
            self.generation_timer = self.root.after(interval_ms, self.start_generation_timer)
    
    def step_markov_generation(self):
        """Take one step in Markov chain generation with smooth interpolation."""
        if not self.generating or self.current_markov_state is None:
            return
        
        chain = self.markov_chains[self.current_emotional_state]
        transitions = chain['transitions']
        
        if self.current_markov_state not in transitions:
            # Find a nearby state instead of stopping
            print(f"⚠️ Dead end state {self.current_markov_state}, finding nearby state...")
            grid_x, grid_y = self.current_markov_state
            
            # Look for nearby states in a small radius
            for radius in range(1, 5):
                found_state = None
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        candidate = (grid_x + dx, grid_y + dy)
                        if candidate in transitions:
                            found_state = candidate
                            break
                    if found_state:
                        break
                if found_state:
                    self.current_markov_state = found_state
                    print(f"🔄 Recovered with nearby state {found_state}")
                    break
            else:
                # Last resort: pick any random state
                self.current_markov_state = random.choice(list(transitions.keys()))
                print(f"🎲 Jumped to random state {self.current_markov_state}")
        
        # Choose next state based on probabilities with bias toward smoother movement
        next_states = list(transitions[self.current_markov_state].keys())
        probabilities = list(transitions[self.current_markov_state].values())
        
        # Add smoothness bias: prefer states closer to current position
        current_x = self.mouse_x * chain['grid_size']
        current_y = self.mouse_y * chain['grid_size']
        
        # Weight probabilities by distance (closer states get higher probability)
        smoothed_probs = []
        for i, state in enumerate(next_states):
            grid_x, grid_y = state
            distance = ((grid_x - current_x) ** 2 + (grid_y - current_y) ** 2) ** 0.5
            # Smooth movement bias: closer states get boosted probability
            distance_weight = 1.0 / (1.0 + distance * 0.1)  # Gentle bias toward closer states
            smoothed_prob = probabilities[i] * distance_weight
            smoothed_probs.append(smoothed_prob)
        
        # Normalize smoothed probabilities
        total_prob = sum(smoothed_probs)
        if total_prob > 0:
            smoothed_probs = [p / total_prob for p in smoothed_probs]
        else:
            smoothed_probs = probabilities  # Fallback
        
        # Weighted random choice with smoothing
        self.current_markov_state = random.choices(next_states, weights=smoothed_probs)[0]
        
        # Convert grid state back to mouse position with HIGH PRECISION
        grid_x, grid_y = self.current_markov_state
        grid_size = chain['grid_size']
        
        # Better coordinate conversion with smoothing
        target_x = (grid_x + 0.5) / grid_size  # Center of grid cell
        target_y = (grid_y + 0.5) / grid_size
        
        # Smooth interpolation from last position to target
        last_x, last_y = self.last_generated_pos
        smoothing_factor = 0.3  # How much to smooth (0=instant, 1=no change)
        
        self.mouse_x = last_x + (target_x - last_x) * (1 - smoothing_factor)
        self.mouse_y = last_y + (target_y - last_y) * (1 - smoothing_factor)
        
        # Clamp to valid range
        self.mouse_x = max(0.0, min(1.0, self.mouse_x))
        self.mouse_y = max(0.0, min(1.0, self.mouse_y))
        
        # Update smoothing state
        self.last_generated_pos = (self.mouse_x, self.mouse_y)
    
    def update_markov_generation(self):
        """Update method called from control loop during generation."""
        if not self.generating:
            return
        
        # Generation happens in step_markov_generation() called by timer
        # This method can be used for any additional updates needed during generation
        pass
    
    def stop_markov_generation(self):
        """Stop Markov chain generation."""
        if not self.generating:
            return
        
        self.generating = False
        
        # Stop the generation timer
        if self.generation_timer:
            self.root.after_cancel(self.generation_timer)
            self.generation_timer = None
        
        self.generate_btn.config(text="🧠 Generate (Markov)")
        self.record_status.config(text="Generation stopped", foreground="gray")
        
        generation_duration = time.time() - self.generation_start_time
        print(f"🎨 Stopped Markov generation after {generation_duration:.1f} seconds")
    
    def load_saved_movements(self):
        """Load previously saved movements from disk for persistence."""
        if not os.path.exists("movement_recordings"):
            print("📂 No movement recordings directory found")
            return
        
        loaded_count = 0
        for emotion_key in self.emotional_states.keys():
            # Find most recent recording for this emotion
            pattern = f"movement_recordings/{emotion_key}_*.json"
            files = glob.glob(pattern)
            
            if files:
                # Get most recent file
                latest_file = max(files, key=os.path.getmtime)
                
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    # Load movements
                    if 'movements' in data:
                        self.recorded_movements[emotion_key] = data['movements']
                        print(f"📂 Loaded {len(data['movements'])} movements for {emotion_key}")
                    
                    # Load Markov chain
                    if 'markov_chain' in data:
                        self.markov_chains[emotion_key] = data['markov_chain']
                        chain = data['markov_chain']
                        print(f"🔗 Loaded Markov chain for {emotion_key}: {chain.get('unique_states', 0)} states")
                    
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"❌ Error loading {latest_file}: {e}")
        
        if loaded_count > 0:
            print(f"✅ Loaded movements for {loaded_count} emotional states")
            # Update status if we have chains
            if self.markov_chains:
                total_states = sum(chain.get('unique_states', 0) for chain in self.markov_chains.values())
                self.markov_status.config(text=f"Loaded: {total_states} total states", foreground="blue")
        else:
            print("📂 No saved movements found")


def main():
    """Main function to start the interface."""
    print("🚀 Starting Clean Emotional Hand Control...")
    
    # Create and run the interface
    interface = CleanCursorInterface()
    
    try:
        # Start the tkinter main loop
        interface.root.mainloop()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if hasattr(interface, 'hand_controller') and interface.hand_controller:
            try:
                interface.hand_controller.cleanup()
            except:
                pass
        print("🔌 Clean shutdown complete")


if __name__ == "__main__":
    main()