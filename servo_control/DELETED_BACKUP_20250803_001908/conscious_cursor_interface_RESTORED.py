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
import tkinter.messagebox
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
        self.root.title("🎯 Hand Control")
        
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
        
        # Focus management for text fields vs keyboard finger control
        self.text_field_has_focus = False
        
        # KEYBOARD HOTKEYS for recording and finger control - ENHANCED SYSTEM!
        self.root.bind("<KeyPress-space>", self.on_spacebar_press)
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.focus_set()  # Ensure window can receive keyboard events
        
        # Hand controller - FIXED CONNECTION LOGIC
        self.hand_controller: Optional[HandExpressionController] = None
        self.connected = False
        
        # Direct control state - clean and simple!
        self.num_fingers = 4
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_targets = [90.0] * self.num_fingers
        
        # SIMPLIFIED KEYBOARD FINGER CONTROL - NO TIMERS, NO ACCUMULATION!
        self.finger_locks = [False] * self.num_fingers      # Which fingers are keyboard-locked
        self.finger_lock_targets = [90.0] * self.num_fingers # Target positions for locked fingers
        self.pressed_keys = set()                           # Currently pressed keys
        
        # ULTRA SIMPLE keyboard control parameters - NO TIMERS!
        self.keyboard_step_size = 2.0        # SMALLER steps for smoother control
        
        # SIMPLIFIED Key mappings for individual finger control
        self.key_mappings = {
            'w': (0, 'up'),    's': (0, 'down'),    # Index finger (F1)
            'e': (1, 'up'),    'd': (1, 'down'),    # Middle finger (F2)
            'r': (2, 'up'),    'f': (2, 'down'),    # Ring finger (F3)
            't': (3, 'up'),    'g': (3, 'down')     # Pinky finger (F4)
        }
        
        print(f"🎯 Initialized finger positions: {self.finger_positions}")
        print(f"🎯 Initialized finger targets: {self.finger_targets}")
        
        # Wave control parameters - the good stuff!
        self.cursor_sensitivity = tk.DoubleVar(value=3.0)
        
        # Wave control parameters - KEEP - working values for smooth wave-based control
        self.wave_strength = tk.DoubleVar(value=2.0)
        self.gravity_width = tk.DoubleVar(value=0.4)
        self.default_position = tk.DoubleVar(value=90.0)
        self.servo_range = tk.DoubleVar(value=45.0)  # ±45 degrees from default (current range)
        
        # Control toggles - simplified
        self.reverse_vertical = tk.BooleanVar(value=True)  # Default to reversed (better usability)
        
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
        
        # PERFORMANCE OPTIMIZATION - Canvas rendering state
        self.last_canvas_update = 0
        self.canvas_update_interval = 0.033  # 30 Hz for canvas (half the control rate)
        self.canvas_objects = {}  # Cache canvas objects for efficient updates
        self.last_render_state = {}  # Track what was last rendered to avoid unnecessary updates
        
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
        
        # Recording/playback state - SERVO-BASED MARKOV CHAIN SYSTEM!
        self.recording = False
        self.playing_back = False
        self.recorded_movements = {}  # emotion_name -> list of servo positions for exact playback
        self.markov_chains = {}      # emotion_name -> servo position transition chains
        self.playback_start_time = 0
        self.current_playback = []
        self.record_start_time = 0
        
        # Dataset management - NEW!
        self.available_datasets = {}  # emotion_name -> list of dataset info
        self.active_datasets = {}     # emotion_name -> selected dataset filename
        self.dataset_info = {}        # filename -> dataset metadata
        
        # Time-based recording state (captures stillness!) - WITH MEMORY MANAGEMENT
        self.recording_timer = None
        self.record_interval = 0.025  # 40 Hz - much higher resolution for easing capture
        self.recorded_positions = []  # Current recording session positions
        
        # MEMORY MANAGEMENT for recording system - ULTRA SHORT SEGMENTS FOR STABILITY!
        self.max_recording_points = 800   # 20 seconds at 40Hz (prevent crashes!)
        self.recording_buffer_cleanup_interval = 50   # Clean up every 50 points (more frequent)
        self.recording_point_counter = 0
        
        # Markov chain generation state
        self.generating = False
        self.generation_start_time = 0
        self.current_markov_state = None  # Current state in generation
        self.generation_timer = None
        self.generation_speed = 0.03  # 33Hz for smooth but responsive movement
        self.generation_smoothing = True  # Enable position smoothing
        self.last_generated_pos = (0.5, 0.5)  # For smooth interpolation
        self.target_generated_pos = (0.5, 0.5)  # Target position for easing
        self.generation_easing_factor = 0.3  # Smooth easing between positions
        
        # Keyboard control tracking for live Markov generation (separate from datasets)
        self.live_keyboard_states = []  # Store recent keyboard movements without polluting datasets
        self.live_keyboard_limit = 200  # Keep only recent movements for live generation
        
        self.setup_ui()
        self.start_control_loop()
        
        # DISABLED: Don't load old movements automatically to prevent memory bloat
        # self.load_saved_movements()
        print("⚠️ Automatic loading of saved movements DISABLED to prevent memory bloat")
        
        # DISABLED: Don't refresh datasets automatically to prevent errors during startup
        # self.refresh_datasets()
        print("⚠️ Automatic dataset refresh DISABLED - use 'Refresh' button manually if needed")
        
        # MEMORY MANAGEMENT: Aggressively clean up any existing data on startup
        print("🧹 Performing aggressive startup memory cleanup...")
        total_points_before = sum(len(data) for data in self.recorded_movements.values())
        
        # CLEAR ALL old recording data to ensure fresh start
        self.recorded_movements.clear()
        
        # Clear any other data structures that might hold movement data
        self.recorded_positions.clear()
        if hasattr(self, 'markov_chains'):
            self.markov_chains.clear()
        
        print(f"🧹 Startup cleanup complete: Cleared {total_points_before} old recording points")
        print(f"💾 Memory reset: Fresh session with max {self.max_recording_points} points per segment (20s each)")
        
        print("🎯 Clean Emotional Hand Control initialized")
        print("🎮 Direct wave-based cursor→servo control ready")
        print("😊 5 emotional states available for testing")
        print(f"📐 FIXED canvas dimensions: 480x200 (no more resizing)")
        print(f"🎯 Condensed control area: 25%-75% of canvas width for precise movement")
        print("⚡ ULTRA-SHORT 20s recordings prevent crashes - record multiple segments per emotion!")
        
        # Initialize the cleaner dataset display
        self.update_emotion_dataset_display()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling in the interface."""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def on_text_field_focus_in(self, event=None):
        """Handle text field getting focus - disable keyboard finger control temporarily."""
        self.text_field_has_focus = True
        print("📝 Text field focused - keyboard finger control temporarily disabled")
    
    def on_text_field_focus_out(self, event=None):
        """Handle text field losing focus - re-enable keyboard finger control."""
        self.text_field_has_focus = False
        print("🎯 Text field unfocused - keyboard finger control re-enabled")
    
    def on_text_field_enter(self, event=None):
        """Handle Enter key in text field - clear focus and return to hand control."""
        self.focus_hand_control()
        print("✅ Text entry confirmed - focus returned to hand control")
    
    def on_text_field_escape(self, event=None):
        """Handle Escape key in text field - clear focus and return to hand control."""
        self.focus_hand_control()
        print("❌ Text entry cancelled - focus returned to hand control")
    
    def focus_hand_control(self):
        """Set focus back to main window for keyboard finger control."""
        self.root.focus_set()
        self.text_field_has_focus = False
        print("🎯 Focus returned to hand control - keyboard finger control active")
    
    def setup_ui(self):
        """Create clean, focused UI."""
        
        # === CONNECTION FRAME ===
        conn_frame = ttk.LabelFrame(self.scrollable_frame, text="🔌 Connection")
        conn_frame.pack(fill=tk.X, padx=15, pady=8)  # Better padding
        
        # Create inner frame for better button alignment
        conn_inner = ttk.Frame(conn_frame)
        conn_inner.pack(fill=tk.X, padx=10, pady=8)
        
        self.connect_btn = ttk.Button(conn_inner, text="Connect to Hand Controller", 
                                     command=self.toggle_connection, width=25)  # Fixed width
        self.connect_btn.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(conn_inner, text="❌ Disconnected", width=25)
        self.status_label.pack(side=tk.LEFT, padx=(15, 0))  # More space
        
        # === EMOTIONAL STATE CONTROL ===
        emotion_frame = ttk.LabelFrame(self.scrollable_frame, text="😊 Emotional State Control")
        emotion_frame.pack(fill=tk.X, padx=15, pady=8)  # Better padding
        
        # Create inner frame for better layout
        emotion_inner = ttk.Frame(emotion_frame)
        emotion_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Current state display
        self.current_state_label = ttk.Label(emotion_inner, text=f"Current: {self.current_emotional_state}", 
                                           font=('Arial', 12, 'bold'))
        self.current_state_label.pack(pady=(0, 10))  # Bottom padding
        
        # Emotion buttons - improved layout
        emotion_buttons_frame = ttk.Frame(emotion_inner)
        emotion_buttons_frame.pack(pady=(0, 10))
        
        for emotion_name in self.emotional_states.keys():
            btn = ttk.Button(emotion_buttons_frame, text=emotion_name.title(), width=12,  # Slightly wider
                           command=lambda e=emotion_name: self.switch_emotional_state(e))
            btn.pack(side=tk.LEFT, padx=8)  # More spacing
        
        # Movement recording/playback control - BETTER FORMATTED!
        record_frame = ttk.Frame(emotion_inner)
        record_frame.pack(pady=(0, 10))
        
        # === SIDE-BY-SIDE LAYOUT: RECORD CONTROLS + CANVAS ===
        # Left side - Record buttons
        record_buttons_frame = ttk.Frame(record_frame)
        record_buttons_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Record button - ULTRA SHORT RECORDINGS FOR STABILITY!
        self.record_btn = ttk.Button(record_buttons_frame, text="🎬 Record Movement (20s)", 
                                   command=self.start_recording, width=24)  # Consistent width
        self.record_btn.pack(pady=5)
        
        # Playback button
        self.playback_btn = ttk.Button(record_buttons_frame, text="▶️ Play Back", 
                                     command=self.start_playback, width=24)
        self.playback_btn.pack(pady=5)
        
        # GENERATIVE playback button
        self.generate_btn = ttk.Button(record_buttons_frame, text="🧠 Generate (Markov)", 
                                     command=self.start_markov_generation, width=24)
        self.generate_btn.pack(pady=5)
        
        # MANUAL SAVE BUTTON - NEW!
        self.save_btn = ttk.Button(record_buttons_frame, text="💾 Save Recording", 
                                 command=self.manual_save_recording, width=24)
        self.save_btn.pack(pady=5)
        
        # Right side - Canvas right next to buttons!
        canvas_side_frame = ttk.Frame(record_frame)
        canvas_side_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas with smaller dimensions to fit side-by-side
        self.canvas_width = 300  # Reduced from 400
        self.canvas_height = 200  # Reduced from 400
        
        canvas_label = ttk.Label(canvas_side_frame, text="🎯 Visual Feedback", font=("Arial", 10, "bold"))
        canvas_label.pack(pady=(0, 5))
        
        self.canvas = tk.Canvas(canvas_side_frame, width=self.canvas_width, height=self.canvas_height, 
                               bg='black', highlightthickness=1, highlightcolor='white')
        self.canvas.pack()
        
        # Canvas bindings
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Button-1>', self.on_mouse_click)
        
        # Status frame for better organization
        status_frame = ttk.Frame(emotion_inner)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Status - BETTER FORMATTING - Encourage multiple short recordings!
        self.record_status = ttk.Label(status_frame, text="Ready to record 20s segments (Spacebar)", 
                                     foreground="gray", width=45)
        self.record_status.pack(side=tk.LEFT)
        
        # Markov status display 
        self.markov_status = ttk.Label(status_frame, text="No chains built", 
                                     foreground="gray", font=("Arial", 8), width=25)
        self.markov_status.pack(side=tk.RIGHT)
        
        # === DATASET MANAGEMENT - CLEANER INTERFACE! ===
        dataset_frame = ttk.LabelFrame(emotion_inner, text="📊 Movement Datasets")
        dataset_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Dataset info and controls
        dataset_inner = ttk.Frame(dataset_frame)
        dataset_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # CLEAR emotion-specific dataset info
        emotion_dataset_frame = ttk.Frame(dataset_inner)
        emotion_dataset_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Current emotion indicator with dataset count
        self.current_emotion_dataset_label = ttk.Label(emotion_dataset_frame, 
                                                      text=f"📁 {self.current_emotional_state.title()}: No datasets", 
                                                      font=("Arial", 11, "bold"), foreground="#8B5CF6")
        self.current_emotion_dataset_label.pack(anchor=tk.W)
        
        # Dataset selector frame - CLEARER LABELS
        self.selector_frame = ttk.Frame(dataset_inner)
        self.selector_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.dataset_selector_label = ttk.Label(self.selector_frame, text=f"💾 Active Dataset for {self.current_emotional_state.title()}:", 
                 font=("Arial", 10))
        self.dataset_selector_label.pack(anchor=tk.W, pady=(0, 3))
        
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(self.selector_frame, textvariable=self.dataset_var, 
                                            width=50, state="readonly", font=("Arial", 9))
        self.dataset_dropdown.pack(fill=tk.X, pady=(0, 5))
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self.on_dataset_selected)
        
        # Quick status
        self.dataset_status_label = ttk.Label(self.selector_frame, text="No dataset selected", 
                                            font=("Arial", 8), foreground="gray")
        self.dataset_status_label.pack(anchor=tk.W)
        
        # Dataset management buttons - BETTER ORGANIZED
        dataset_btn_frame = ttk.Frame(dataset_inner) 
        dataset_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Left side - dataset actions
        left_btn_frame = ttk.Frame(dataset_btn_frame)
        left_btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(left_btn_frame, text="🔄 Refresh Lists", 
                  command=self.refresh_datasets, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(left_btn_frame, text="� Show Details", 
                  command=self.show_dataset_details, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        # Right side - management actions  
        right_btn_frame = ttk.Frame(dataset_btn_frame)
        right_btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(right_btn_frame, text="�️ Delete Dataset", 
                  command=self.delete_dataset, width=15).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Button(right_btn_frame, text="🧹 Clean Memory", 
                  command=self.manual_memory_cleanup, width=15).pack(side=tk.LEFT, padx=(5, 0))
        
        # Dataset naming option (for new recordings)
        naming_frame = ttk.Frame(dataset_inner)
        naming_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(naming_frame, text="Next Recording Name:").pack(side=tk.LEFT, padx=(0, 5))
        self.dataset_name_var = tk.StringVar()
        self.dataset_name_entry = ttk.Entry(naming_frame, textvariable=self.dataset_name_var, 
                                           width=25, font=("Arial", 9))
        self.dataset_name_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Bind events to manage focus properly for keyboard finger control
        self.dataset_name_entry.bind("<FocusIn>", self.on_text_field_focus_in)
        self.dataset_name_entry.bind("<FocusOut>", self.on_text_field_focus_out)
        self.dataset_name_entry.bind("<Return>", self.on_text_field_enter)
        self.dataset_name_entry.bind("<Escape>", self.on_text_field_escape)
        
        ttk.Button(naming_frame, text="🎲 Auto", command=self.auto_generate_name, 
                  width=8).pack(side=tk.LEFT)
        
        # Clear focus button for immediate finger control access
        ttk.Button(naming_frame, text="🎯 Focus Hand Control", command=self.focus_hand_control, 
                  width=18).pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress bar frame - BETTER INTEGRATED!
        self.progress_frame = ttk.Frame(emotion_inner)
        self.progress_frame.pack(pady=8, fill=tk.X)  # Better spacing
        
        # Center the progress bar elements
        progress_center_frame = ttk.Frame(self.progress_frame)
        progress_center_frame.pack(expand=True)
        
        # Recording progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_center_frame, variable=self.progress_var, 
                                          maximum=100, length=500, mode='determinate')  # Wider bar
        self.progress_bar.pack(side=tk.LEFT, padx=10)
        
        # Progress label showing time and percentage
        self.progress_label = ttk.Label(progress_center_frame, text="", width=18, font=("Arial", 10))
        self.progress_label.pack(side=tk.LEFT, padx=10)
        
        # Initially hide progress elements
        self.progress_frame.pack_forget()
        
        # Recording state
        self.recording = False
        self.playing_back = False
        self.recorded_movements = {}  # emotion_name -> list of movements
        self.playback_start_time = 0
        self.current_playback = []
        
        # === CONTROL MODES SETUP ===
        self.freeze_behavior = tk.BooleanVar(value=False)
        self.manual_override_mode = tk.BooleanVar(value=False)
        self.show_path = tk.BooleanVar(value=True)
    
    def on_dataset_selected(self, event=None):
        """Handle dataset selection from dropdown."""
        selected = self.dataset_var.get()
        print(f"📊 Selected dataset: {selected}")
        # TODO: Implement dataset activation
    
    def delete_dataset(self):
        """Placeholder for dataset deletion."""
        print("🗑️ Delete dataset functionality not yet implemented")
        pass
    
    def show_dataset_details(self):
        """Placeholder for showing dataset details."""
        print("📋 Show dataset details functionality not yet implemented") 
        pass
    
    def manual_memory_cleanup(self):
        """Manual memory cleanup function."""
        print("🧹 Performing manual memory cleanup...")
        total_before = sum(len(data) for data in self.recorded_movements.values())
        self.recorded_movements.clear()
        print(f"🧹 Cleaned up {total_before} movement points from memory")
    
    def auto_generate_name(self):
        """Auto-generate a name for the next recording."""
        emotion = self.current_emotional_state
        import datetime
        timestamp = datetime.datetime.now().strftime("%H%M")
        name = f"{emotion}_{timestamp}"
        self.dataset_name_var.set(name)
        print(f"🎲 Auto-generated name: {name}")
        
        # === CONTROL MODES - Better formatted ===
        mode_frame = ttk.LabelFrame(self.scrollable_frame, text="🎛️ Control Modes")
        mode_frame.pack(fill=tk.X, padx=15, pady=8)
        
        # Create inner frame for better layout
        mode_inner = ttk.Frame(mode_frame)
        mode_inner.pack(fill=tk.X, padx=10, pady=8)
        
        reverse_cb = ttk.Checkbutton(mode_inner, text="🔄 Reverse Vertical", 
                                   variable=self.reverse_vertical)
        reverse_cb.pack(side=tk.LEFT, padx=10)
        
        # Person detection simulation toggle
        person_cb = ttk.Checkbutton(mode_inner, text="👤 Person Detected (Sim)", 
                                  variable=self.person_detected,
                                  command=self.on_person_detection_toggle)
        person_cb.pack(side=tk.LEFT, padx=20)
        
        reset_btn = ttk.Button(mode_inner, text="🎯 Reset to Center", 
                             command=self.reset_to_center, width=20)
        reset_btn.pack(side=tk.RIGHT, padx=10)
        
        # === FREEZE BEHAVIOR CONTROLS ===
        freeze_frame = ttk.LabelFrame(self.scrollable_frame, text="❄️ Freeze Behavior")
        freeze_frame.pack(fill=tk.X, padx=15, pady=8)
        
        # Create inner frame for better spacing
        freeze_inner = ttk.Frame(freeze_frame)
        freeze_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Min Freeze Duration
        ttk.Label(freeze_inner, text="Min Freeze Duration:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        min_freeze_scale = ttk.Scale(freeze_inner, from_=1.0, to=10.0, variable=self.min_freeze_duration, orient=tk.HORIZONTAL, length=400)
        min_freeze_scale.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        min_freeze_label = ttk.Label(freeze_inner, text="2.0s", width=8)
        min_freeze_label.grid(row=0, column=2, padx=5, pady=5)
        self.min_freeze_duration.trace_add("write", lambda *args: min_freeze_label.config(text=f"{self.min_freeze_duration.get():.1f}s"))
        
        # Max Freeze Duration
        ttk.Label(freeze_inner, text="Max Freeze Duration:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        max_freeze_scale = ttk.Scale(freeze_inner, from_=2.0, to=15.0, variable=self.max_freeze_duration, orient=tk.HORIZONTAL, length=400)
        max_freeze_scale.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        max_freeze_label = ttk.Label(freeze_inner, text="6.0s", width=8)
        max_freeze_label.grid(row=1, column=2, padx=5, pady=5)
        self.max_freeze_duration.trace_add("write", lambda *args: max_freeze_label.config(text=f"{self.max_freeze_duration.get():.1f}s"))
        
        freeze_inner.columnconfigure(1, weight=1)
        
        # === WAVE CONTROL PARAMETERS - Better formatted! ===
        wave_frame = ttk.LabelFrame(self.scrollable_frame, text="🌊 Wave Control Parameters")
        wave_frame.pack(fill=tk.X, padx=15, pady=8)
        
        # Create inner frame for better spacing
        wave_inner = ttk.Frame(wave_frame)
        wave_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Cursor Sensitivity  
        ttk.Label(wave_inner, text="Cursor Sensitivity:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        sensitivity_scale = ttk.Scale(wave_inner, from_=0.5, to=10.0, variable=self.cursor_sensitivity, orient=tk.HORIZONTAL, length=400)
        sensitivity_scale.grid(row=0, column=1, sticky=tk.EW, padx=10, pady=5)
        sensitivity_label = ttk.Label(wave_inner, text="3.0", width=8)
        sensitivity_label.grid(row=0, column=2, padx=5, pady=5)
        self.cursor_sensitivity.trace_add("write", lambda *args: sensitivity_label.config(text=f"{self.cursor_sensitivity.get():.1f}"))
        
        # Wave Strength
        ttk.Label(wave_inner, text="Wave Strength:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        wave_scale = ttk.Scale(wave_inner, from_=0.0, to=5.0, variable=self.wave_strength, orient=tk.HORIZONTAL, length=400)
        wave_scale.grid(row=1, column=1, sticky=tk.EW, padx=10, pady=5)
        wave_label = ttk.Label(wave_inner, text="2.0", width=8)
        wave_label.grid(row=1, column=2, padx=5, pady=5)
        self.wave_strength.trace_add("write", lambda *args: wave_label.config(text=f"{self.wave_strength.get():.1f}"))
        
        # Gravity Width
        ttk.Label(wave_inner, text="Gravity Width:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        gravity_scale = ttk.Scale(wave_inner, from_=0.1, to=1.0, variable=self.gravity_width, orient=tk.HORIZONTAL, length=400)
        gravity_scale.grid(row=2, column=1, sticky=tk.EW, padx=10, pady=5)
        gravity_label = ttk.Label(wave_inner, text="0.4", width=8)
        gravity_label.grid(row=2, column=2, padx=5, pady=5)
        self.gravity_width.trace_add("write", lambda *args: gravity_label.config(text=f"{self.gravity_width.get():.1f}"))
        
        # Default Position
        ttk.Label(wave_inner, text="Default Position:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        default_scale = ttk.Scale(wave_inner, from_=0, to=180, variable=self.default_position, orient=tk.HORIZONTAL, length=400)
        default_scale.grid(row=3, column=1, sticky=tk.EW, padx=10, pady=5)
        default_label = ttk.Label(wave_inner, text="90", width=8)
        default_label.grid(row=3, column=2, padx=5, pady=5)
        self.default_position.trace_add("write", lambda *args: default_label.config(text=f"{self.default_position.get():.0f}"))
        
        # Servo Range - NEW CONTROL!
        ttk.Label(wave_inner, text="Servo Range (±):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        servo_range_scale = ttk.Scale(wave_inner, from_=10.0, to=90.0, variable=self.servo_range, orient=tk.HORIZONTAL, length=400)
        servo_range_scale.grid(row=4, column=1, sticky=tk.EW, padx=10, pady=5)
        servo_range_label = ttk.Label(wave_inner, text="45°", width=8)
        servo_range_label.grid(row=4, column=2, padx=5, pady=5)
        self.servo_range.trace_add("write", lambda *args: servo_range_label.config(text=f"{self.servo_range.get():.0f}°"))
        
            
        wave_inner.columnconfigure(1, weight=1)
        
        # Add helpful info about servo range
        servo_info_label = ttk.Label(wave_inner, text="(Controls how far servos move from default position)", 
                                   font=("Arial", 8), foreground="gray")
        servo_info_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0, 5))
    
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
        
        # Update CLEARER dataset display for new emotion
        self.update_emotion_dataset_display()
        
        # Also make sure we have the latest dataset list
        if len(self.available_datasets) == 0:
            print("🔄 No datasets loaded, refreshing...")
            self.refresh_datasets()
    
    def update_emotion_dataset_display(self):
        """Update the emotion-specific dataset display with clear information."""
        emotion = self.current_emotional_state
        
        # Update the emotion-specific label
        if hasattr(self, 'current_emotion_dataset_label'):
            available_count = len(self.available_datasets.get(emotion, []))
            if available_count > 0:
                self.current_emotion_dataset_label.config(
                    text=f"📁 {emotion.title()}: {available_count} dataset(s) available"
                )
            else:
                self.current_emotion_dataset_label.config(
                    text=f"📁 {emotion.title()}: No datasets (record some first!)"
                )
        
        # Update the dropdown label to be specific to current emotion
        if hasattr(self, 'dataset_selector_label'):
            self.dataset_selector_label.config(text=f"💾 Active Dataset for {emotion.title()}:")
        
        # Update dropdown contents
        self.refresh_emotion_datasets()
        
        # Update status
        if hasattr(self, 'dataset_status_label'):
            active_dataset = self.active_datasets.get(emotion, "")
            if active_dataset:
                self.dataset_status_label.config(
                    text=f"✅ Using: {active_dataset}",
                    foreground="green"
                )
            else:
                self.dataset_status_label.config(
                    text="⚠️ No dataset selected for generation",
                    foreground="orange"
                )
    
    def refresh_emotion_datasets(self):
        """Refresh the dataset dropdown for the current emotion only."""
        emotion = self.current_emotional_state
        emotion_datasets = self.available_datasets.get(emotion, [])
        
        # Clear and populate dropdown
        self.dataset_dropdown['values'] = []
        dataset_options = []
        
        if emotion_datasets:
            for dataset_info in emotion_datasets:
                filename = dataset_info.get('filename', 'Unknown')
                points = dataset_info.get('points', 0)
                name = dataset_info.get('name', filename.replace('.json', ''))
                display_name = f"{name} ({points} points)"
                dataset_options.append(display_name)
        
        if not dataset_options:
            dataset_options = ["No datasets available - record some movements first!"]
        
        self.dataset_dropdown['values'] = dataset_options
        
        # Set current selection
        active_dataset = self.active_datasets.get(emotion, "")
        if active_dataset and dataset_options:
            # Try to find and select the active dataset
            for option in dataset_options:
                if active_dataset in option:
                    self.dataset_var.set(option)
                    break
            else:
                self.dataset_var.set(dataset_options[0])
        elif dataset_options:
            self.dataset_var.set(dataset_options[0])
    
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
        
        # Record movement if recording - SERVO-BASED RECORDING!
        if self.recording and not self.is_frozen:
            current_time = time.time()
            relative_time = current_time - self.record_start_time
            
            # MEMORY MANAGEMENT: Limit recording buffer size to prevent performance degradation
            self.recording_point_counter += 1
            
            # Only store if we haven't exceeded the maximum points
            if len(self.recorded_movements.get(self.current_emotional_state, [])) < self.max_recording_points:
                # Store servo positions directly - unified data for cursor AND keyboard inputs!
                movement_point = {
                    'time': current_time,
                    'relative_time': relative_time,
                    'servo_positions': self.finger_positions.copy()  # NEW: Direct servo recording
                }
                if self.current_emotional_state not in self.recorded_movements:
                    self.recorded_movements[self.current_emotional_state] = []
                self.recorded_movements[self.current_emotional_state].append(movement_point)
            
            # PERIODIC CLEANUP: Clean up old data periodically to prevent memory bloat
            if self.recording_point_counter % self.recording_buffer_cleanup_interval == 0:
                self._cleanup_recording_buffer()
                
            # EMERGENCY BRAKE: Stop recording if we hit memory limits
            if len(self.recorded_movements.get(self.current_emotional_state, [])) >= self.max_recording_points:
                print(f"⚠️ Recording buffer full ({self.max_recording_points} points) - stopping recording to prevent lag")
                self.stop_recording()
                return
            
            # Note: Time-based recording happens in record_position_sample() called by timer
        
        # IMPORTANT: Don't interfere with generative movement!
        elif self.generating:
            # Ignore mouse input during generation to prevent interference
            return
    
    def on_mouse_click(self, event):
        """Handle mouse click in canvas."""
        self.on_mouse_move(event)  # Update position
    
    def on_spacebar_press(self, event):
        """Handle spacebar press for recording toggle - SAFE RECORDING HOTKEY!"""
        # Don't interfere with text input
        if self.text_field_has_focus:
            print("📝 Spacebar ignored - text field has focus")
            return
        
        # Don't interfere with generation
        if self.generating:
            print("🧠 Spacebar ignored - Markov generation in progress")
            return
        
        # Don't interfere with playback
        if self.playing_back:
            print("▶️ Spacebar ignored - playback in progress")
            return
            
        # Safe to toggle recording
        try:
            if self.recording:
                self.stop_recording()
                print("⏹️ Spacebar: Stopped recording")
            else:
                self.start_recording()
                print("🎬 Spacebar: Started recording")
        except Exception as e:
            print(f"❌ Error handling spacebar: {e}")
            # Don't let errors break the interface
    
    def on_key_press(self, event):
        """SIMPLIFIED keyboard press for finger control - NO TIMERS, NO ACCUMULATION!"""
        key = event.keysym.lower()
        
        # Don't interfere with text input
        if self.text_field_has_focus:
            return
        
        # Skip spacebar (handled separately)
        if key == 'space':
            return
            
        # Check if this is a finger control key
        if key in self.key_mappings:
            finger_index, direction = self.key_mappings[key]
            
            # Add to pressed keys set
            self.pressed_keys.add(key)
            
            # Lock this finger to keyboard control IMMEDIATELY
            if not self.finger_locks[finger_index]:
                self.finger_locks[finger_index] = True
                self.finger_lock_targets[finger_index] = self.finger_positions[finger_index]
                print(f"🔒 Finger {finger_index+1} locked to keyboard control at {self.finger_positions[finger_index]:.1f}°")
            
            # Apply movement IMMEDIATELY (no delays!)
            self.apply_keyboard_movement(finger_index, direction)
    
    def apply_keyboard_movement(self, finger_index, direction):
        """Apply keyboard movement immediately with no delay - respects reverse vertical setting."""
        # Respect reverse vertical setting for keyboard controls too!
        effective_direction = direction
        if self.reverse_vertical.get():
            # When reversed, flip the direction mapping
            effective_direction = 'down' if direction == 'up' else 'up'
        
        # Calculate new target position with SMALLER steps for smoother control
        if effective_direction == 'up':
            new_target = min(180.0, self.finger_lock_targets[finger_index] + self.keyboard_step_size)
        else:  # down
            new_target = max(0.0, self.finger_lock_targets[finger_index] - self.keyboard_step_size)
        
        # Set new target IMMEDIATELY
        if new_target != self.finger_lock_targets[finger_index]:
            old_target = self.finger_lock_targets[finger_index]
            self.finger_lock_targets[finger_index] = new_target
            # Apply immediately to positions for instant response!
            self.finger_positions[finger_index] = new_target
            
            # Record live keyboard state for potential Markov generation (separate from datasets)
            if not self.recording:  # Only record keyboard movements when not recording datasets
                keyboard_state = {
                    'timestamp': time.time(),
                    'servo_positions': self.finger_positions.copy(),  # NEW: Unified servo-based recording
                    'source': 'keyboard'
                }
                self.live_keyboard_states.append(keyboard_state)
                
                # Keep only recent states
                if len(self.live_keyboard_states) > self.live_keyboard_limit:
                    self.live_keyboard_states = self.live_keyboard_states[-self.live_keyboard_limit:]
            
            reverse_indicator = " [REVERSED]" if self.reverse_vertical.get() else ""
            if hasattr(self, 'keyboard_move_count'):
                self.keyboard_move_count += 1
            else:
                self.keyboard_move_count = 1
            if self.keyboard_move_count < 10:  # Only log first few moves
                print(f"⚡ Finger {finger_index+1} {direction}→{effective_direction}: {old_target:.1f}° → {new_target:.1f}° (INSTANT){reverse_indicator}")
    
    def process_continuous_keyboard_input(self):
        """Process continuous keyboard input in the main control loop - NO TIMERS!"""
        # Apply movement for all currently pressed keys
        for key in self.pressed_keys:
            if key in self.key_mappings:
                finger_index, direction = self.key_mappings[key]
                if self.finger_locks[finger_index]:  # Only if finger is still locked
                    self.apply_keyboard_movement(finger_index, direction)
    
    def on_key_release(self, event):
        """SIMPLIFIED keyboard release for finger control - NO TIMERS!"""
        key = event.keysym.lower()
        
        # Don't interfere with text input
        if self.text_field_has_focus:
            return
        
        # Remove from pressed keys
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        
        # Check if this finger is no longer controlled by any keys
        if key in self.key_mappings:
            finger_index, _ = self.key_mappings[key]
            
            # Check if any other keys are controlling this finger
            finger_still_controlled = False
            for pressed_key in self.pressed_keys:
                if pressed_key in self.key_mappings:
                    other_finger, _ = self.key_mappings[pressed_key]
                    if other_finger == finger_index:
                        finger_still_controlled = True
                        break
            
            # If no keys controlling this finger, release it back to cursor control IMMEDIATELY
            if not finger_still_controlled and self.finger_locks[finger_index]:
                self.finger_locks[finger_index] = False
                print(f"🔓 Finger {finger_index+1} released to cursor control (INSTANT)")
    
    def release_finger_to_cursor(self, finger_index):
        """Release a finger from keyboard control back to cursor control with smooth transition."""
        if not self.finger_locks[finger_index]:
            return
            
        # Calculate what the cursor target would be for this finger
        cursor_target = self.calculate_cursor_target_for_finger(finger_index)
        
        # Start smooth transition from current keyboard position to cursor position
        self.finger_locks[finger_index] = False
        self.finger_transition_starts[finger_index] = self.finger_lock_targets[finger_index]
        self.finger_lock_targets[finger_index] = cursor_target  # Temporary target for transition
        self.finger_transition_times[finger_index] = time.time()
        self.finger_transitioning[finger_index] = True
        
        print(f"🔓 Finger {finger_index+1} released to cursor control: {self.finger_transition_starts[finger_index]:.1f}° → {cursor_target:.1f}°")
    
    def calculate_cursor_target_for_finger(self, finger_index):
        """Calculate what the cursor target should be for a specific finger."""
        wave_strength = self.wave_strength.get()
        gravity_width = self.gravity_width.get()
        default_pos = self.default_position.get()
        sensitivity = self.cursor_sensitivity.get()
        servo_range = self.servo_range.get()  # Use adjustable servo range
        
        # TIGHTENED MAPPING - same condensed area as visual (25%-75% of screen)
        condensed_start = 0.25  # 25% from left (matches visualization)
        condensed_width = 0.5   # 50% of screen width (matches visualization)
        finger_x = condensed_start + ((finger_index + 0.5) / self.num_fingers) * condensed_width
        
        # Calculate influence of cursor on this finger
        distance = abs(self.mouse_x - finger_x)
        if distance < gravity_width:
            influence = 1.0 - (distance / gravity_width)
            
            # Calculate vertical influence
            y_offset = (self.mouse_y - 0.5) * sensitivity * wave_strength * influence
            if self.reverse_vertical.get():
                y_offset = -y_offset
            
            target = default_pos + (y_offset * servo_range)  # Use adjustable range instead of fixed 45.0
            return max(0, min(180, target))
        else:
            # Return to default position
            return default_pos
    
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
            
            # CONTINUOUS KEYBOARD INPUT - NO TIMERS!
            self.process_continuous_keyboard_input()
            
            # Update playback if active
            if self.playing_back:
                self.update_playback()
            
            # Update generative playback if active - MARKOV CHAIN!
            if self.generating:
                self.update_markov_generation()
        
        # Update canvas visualization (always show current state)
        self.update_canvas()
        
        # Only calculate cursor targets if NOT generating from Markov chain
        if not self.generating:
            # Calculate finger targets from cursor position (even during freeze - maintains position)
            self.calculate_finger_targets()
            
            # Direct control - immediate response!
            self.finger_positions = self.finger_targets.copy()
        # When generating, finger_positions are set by update_markov_generation()
        
        if hasattr(self, 'direct_count'):
            self.direct_count += 1
        else:
            self.direct_count = 1
        
        if self.direct_count < 5 or self.direct_count % 30 == 0:
            freeze_status = " [FROZEN]" if self.is_frozen else ""
            generation_status = " [GENERATING]" if self.generating else " [DIRECT]"
            print(f"🎯 Control {self.direct_count}: positions={[f'{p:.1f}' for p in self.finger_positions]}{generation_status}{freeze_status}")
        
        # Send to hand controller
        self.send_to_hand_controller()
        
        # Schedule next update
        self.root.after(16, self.control_loop)  # ~60 FPS
    
    def calculate_finger_targets(self):
        """Calculate servo targets from cursor position OR keyboard control - INSTANT RESPONSE SYSTEM!"""
        # First calculate cursor-based targets for all fingers
        cursor_targets = self.calculate_cursor_targets()
        
        # Apply keyboard control OR cursor control (no delays, no transitions!)
        for i in range(self.num_fingers):
            if self.finger_locks[i]:
                # Finger is keyboard controlled - use keyboard target directly (already applied in apply_keyboard_movement)
                self.finger_targets[i] = self.finger_lock_targets[i]
            else:
                # Finger is cursor controlled - use cursor target directly
                self.finger_targets[i] = cursor_targets[i]
    
    def calculate_cursor_targets(self):
        """Calculate cursor-based targets for all fingers - separated for clean logic."""
        wave_strength = self.wave_strength.get()
        gravity_width = self.gravity_width.get()
        default_pos = self.default_position.get()
        sensitivity = self.cursor_sensitivity.get()
        servo_range = self.servo_range.get()  # Use adjustable servo range
        
        cursor_targets = []
        
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
                
                target = default_pos + (y_offset * servo_range)  # Use adjustable range instead of fixed 45.0
                cursor_targets.append(max(0, min(180, target)))
            else:
                # Return to default position
                cursor_targets.append(default_pos)
        
        return cursor_targets
    
    def update_canvas(self):
        """OPTIMIZED canvas visualization - only update when necessary and at reduced framerate."""
        current_time = time.time()
        
        # PERFORMANCE: Limit canvas updates to 30 Hz instead of 60 Hz
        if current_time - self.last_canvas_update < self.canvas_update_interval:
            return
        self.last_canvas_update = current_time
        
        # Use stored canvas dimensions for perfect consistency
        canvas_width = self.canvas_width
        canvas_height = self.canvas_height
        
        cursor_x = self.mouse_x * canvas_width
        cursor_y = self.mouse_y * canvas_height
        
        # PERFORMANCE: Check if anything actually changed before updating
        current_state = {
            'cursor_pos': (round(cursor_x), round(cursor_y)),  # Round to avoid micro-updates
            'finger_positions': tuple(round(p, 1) for p in self.finger_positions),  # Round to 0.1°
            'finger_locks': tuple(self.finger_locks),
            'servo_range': round(self.servo_range.get()),
            'reverse_vertical': self.reverse_vertical.get(),
            'pressed_keys': tuple(sorted(self.pressed_keys)),
            'text_focus': self.text_field_has_focus,
            'frozen': self.is_frozen,
            'recording': self.recording,
            'playing': self.playing_back,
            'generating': self.generating
        }
        
        # Only do expensive full redraw if state actually changed
        if current_state != self.last_render_state:
            self._full_canvas_redraw(canvas_width, canvas_height, cursor_x, cursor_y)
            self.last_render_state = current_state.copy()
        else:
            # Just update cursor position for smooth movement
            if 'cursor' in self.canvas_objects:
                self.canvas.coords(self.canvas_objects['cursor'], 
                                 cursor_x-8, cursor_y-8, cursor_x+8, cursor_y+8)
    
    def _full_canvas_redraw(self, canvas_width, canvas_height, cursor_x, cursor_y):
        """Perform full canvas redraw - only called when state changes."""
        # Clear canvas only when doing full redraw
        self.canvas.delete("all")
        self.canvas_objects.clear()
        
        # Draw main cursor - cache the object
        self.canvas_objects['cursor'] = self.canvas.create_oval(
            cursor_x-8, cursor_y-8, cursor_x+8, cursor_y+8, 
            fill="red", outline="white", width=2)
        
        # Pre-calculate common values to avoid repeated calculations
        servo_range = self.servo_range.get()
        base_bar_height = 120  # INCREASED from 60 to 120 for taller bars
        bar_height = max(40, min(int((servo_range / 45.0) * base_bar_height), 200))  # Taller min/max
        bar_width = 25
        
        condensed_area_start = canvas_width * 0.25
        condensed_area_width = canvas_width * 0.5
        finger_y_base = canvas_height - 20
        finger_y_top = finger_y_base - bar_height
        
        # Draw finger indicators with minimal object creation
        for i in range(self.num_fingers):
            finger_x = condensed_area_start + ((i + 0.5) / self.num_fingers) * condensed_area_width
            
            # Colors based on control mode (pre-calculated)
            if self.finger_locks[i]:
                bar_color, outline_color, position_color = "orange", "yellow", "gold"
            else:
                bar_color, outline_color, position_color = "gray30", "white", "lime"
            
            # Main bar outline
            self.canvas.create_rectangle(
                finger_x-bar_width//2, finger_y_top, 
                finger_x+bar_width//2, finger_y_base,
                fill=bar_color, outline=outline_color, width=2)
            
            # Position indicator (optimized calculation)
            if self.reverse_vertical.get():
                pos_ratio = (180.0 - self.finger_positions[i]) / 180.0
            else:
                pos_ratio = self.finger_positions[i] / 180.0
            pos_height = int(pos_ratio * bar_height)
            
            self.canvas.create_rectangle(
                finger_x-bar_width//2, finger_y_base-pos_height, 
                finger_x+bar_width//2, finger_y_base,
                fill=position_color, outline="yellow", width=1)
            
            # Target indicator (optimized calculation)
            if self.reverse_vertical.get():
                target_ratio = (180.0 - self.finger_targets[i]) / 180.0
            else:
                target_ratio = self.finger_targets[i] / 180.0
            target_height = int(target_ratio * bar_height)
            target_y = finger_y_base - target_height
            
            target_color = "white" if self.finger_locks[i] else "red"
            self.canvas.create_line(
                finger_x-bar_width//2-3, target_y, 
                finger_x+bar_width//2+3, target_y,
                fill=target_color, width=2)
            
            # Simplified finger labels (less text objects)
            finger_label = f"F{i+1}🔒" if self.finger_locks[i] else f"F{i+1}"
            self.canvas.create_text(finger_x, finger_y_base+10, text=finger_label, 
                                  fill="white", font=("Arial", 8, "bold"))
        
        # Essential text only - avoid expensive string formatting
        self._draw_essential_text(canvas_width, canvas_height)
    
    def _draw_essential_text(self, canvas_width, canvas_height):
        """Draw only essential text information to avoid performance issues."""
        # Mode indicator (simplified)
        reverse_status = " [REV]" if self.reverse_vertical.get() else ""
        mode_text = f"Hybrid Control{reverse_status}"
        self.canvas.create_text(10, 10, text=mode_text, fill="white", anchor="nw", 
                              font=("Arial", 10, "bold"))
        
        # Current emotion (simplified)
        emotion_text = f"Emotion: {self.current_emotional_state.replace('_', ' ').title()}"
        self.canvas.create_text(10, 25, text=emotion_text, fill="yellow", anchor="nw",
                              font=("Arial", 9))
        
        # Status indicators (only when relevant - reduced text objects)
        status_y = 40
        if self.text_field_has_focus:
            self.canvas.create_text(10, status_y, text="📝 TEXT INPUT", fill="red", anchor="nw",
                                  font=("Arial", 9, "bold"))
        elif self.pressed_keys:
            keys_text = f"Keys: {','.join(sorted(self.pressed_keys)).upper()}"
            self.canvas.create_text(10, status_y, text=keys_text, fill="orange", anchor="nw",
                                  font=("Arial", 9))
        elif self.is_frozen:
            self.canvas.create_text(10, status_y, text="❄️ FROZEN", fill="cyan", anchor="nw",
                                  font=("Arial", 9, "bold"))
        elif self.recording:
            self.canvas.create_text(10, status_y, text="🎬 RECORDING", fill="red", anchor="nw",
                                  font=("Arial", 9, "bold"))
        elif self.playing_back:
            self.canvas.create_text(10, status_y, text="▶️ PLAYBACK", fill="green", anchor="nw",
                                  font=("Arial", 9))
        elif self.generating:
            self.canvas.create_text(10, status_y, text="🧠 GENERATING", fill="purple", anchor="nw",
                                  font=("Arial", 9))
        
        # Servo range (bottom right, simplified)
        servo_range = self.servo_range.get()
        range_text = f"Range: ±{servo_range:.0f}°"
        self.canvas.create_text(canvas_width-10, canvas_height-10, text=range_text, 
                              fill="lightgreen", anchor="se", font=("Arial", 9))
    
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
                # MAP SERVO RANGE: Convert from 0-180° to Arduino's actual range (40-130°)
                # Arduino constrains to 40-130° (90° range), so we need to map our servo_range to that
                arduino_min = 40
                arduino_max = 130
                arduino_center = (arduino_min + arduino_max) / 2  # 85°
                arduino_range = arduino_max - arduino_min  # 90°
                
                positions = []
                for i in range(self.num_fingers):
                    # Convert from our 0-180° system to Arduino's 40-130° system
                    # Center our position around 90° and map to Arduino center (85°)
                    offset_from_center = self.finger_positions[i] - 90.0
                    # Scale to Arduino's available range
                    arduino_offset = (offset_from_center / 90.0) * (arduino_range / 2.0)
                    arduino_position = arduino_center + arduino_offset
                    # Clamp to Arduino's actual limits
                    arduino_position = max(arduino_min, min(arduino_max, arduino_position))
                    positions.append(int(arduino_position))
                
                # Debug output to see what we're actually sending
                if hasattr(self, 'send_count'):
                    self.send_count += 1
                else:
                    self.send_count = 1
                
                if self.send_count < 5 or self.send_count % 20 == 0:
                    print(f"📤 Sending {self.send_count}: arduino_positions={positions} from finger_positions={[f'{p:.1f}' for p in self.finger_positions]} (mapped to 40-130°)")
                
                # Use the WORKING method from the working version
                self.hand_controller.set_hand_positions(positions)
                
                self.last_sent_positions = self.finger_positions.copy()
                self.last_send_time = current_time
                self.last_any_send_time = current_time
                
            except Exception as e:
                print(f"❌ Error sending to hand controller: {e}")
                import traceback
                traceback.print_exc()
    
    def _cleanup_recording_buffer(self):
        """Clean up recording buffer to prevent memory bloat and performance degradation."""
        for emotion_name in list(self.recorded_movements.keys()):
            if emotion_name in self.recorded_movements:
                current_buffer = self.recorded_movements[emotion_name]
                buffer_length = len(current_buffer)
                
                # If buffer is getting large, keep only the most recent data
                if buffer_length > self.max_recording_points * 0.8:  # Start cleanup at 80% capacity
                    # Keep most recent 60% of data
                    keep_count = int(self.max_recording_points * 0.6)
                    self.recorded_movements[emotion_name] = current_buffer[-keep_count:]
                    cleaned_count = buffer_length - keep_count
                    print(f"🧹 Cleaned {cleaned_count} old recording points for {emotion_name} (kept {keep_count})")
    
    def manual_memory_cleanup(self):
        """Manual memory cleanup triggered by user button press."""
        total_points_before = sum(len(data) for data in self.recorded_movements.values())
        
        # Force aggressive cleanup of all emotions
        for emotion_name in list(self.recorded_movements.keys()):
            if emotion_name in self.recorded_movements:
                current_buffer = self.recorded_movements[emotion_name]
                buffer_length = len(current_buffer)
                
                if buffer_length > 0:
                    # Keep only the most recent 30% of data for aggressive cleanup
                    keep_count = max(10, int(buffer_length * 0.3))  # Keep at least 10 points
                    self.recorded_movements[emotion_name] = current_buffer[-keep_count:]
                    cleaned_count = buffer_length - keep_count
                    print(f"🧹 Manual cleanup: {emotion_name} {buffer_length} → {keep_count} points")
        
        total_points_after = sum(len(data) for data in self.recorded_movements.values())
        total_cleaned = total_points_before - total_points_after
        
        # Update UI status
        if hasattr(self, 'record_status'):
            self.record_status.config(text=f"Cleaned {total_cleaned} points from memory", foreground="green")
            self.root.after(3000, lambda: self.record_status.config(text="Ready to record (Spacebar)", foreground="gray"))
        
        print(f"🧹 Manual cleanup complete: {total_points_before} → {total_points_after} points (freed {total_cleaned})")
    
    def start_recording(self):
        """Start time-based recording that captures both movement AND stillness - WITH MEMORY MANAGEMENT."""
        # CRITICAL: Stop any other active operations first
        if self.playing_back:
            print("🛑 Stopping playback to start recording")
            self.stop_playback()
            
        if self.generating:
            print("🛑 Stopping Markov generation to start recording")
            self.stop_markov_generation()
            
        if self.recording:
            # Stop recording
            self.stop_recording()
            return
            
        # MEMORY MANAGEMENT: Clear old recording data to prevent accumulation
        if self.current_emotional_state in self.recorded_movements:
            old_count = len(self.recorded_movements[self.current_emotional_state])
            print(f"🧹 Clearing {old_count} old recording points for {self.current_emotional_state}")
        
        # Start recording with fresh buffer
        self.recording = True
        self.record_start_time = time.time()
        self.recorded_movements[self.current_emotional_state] = []  # Fresh start
        self.recorded_positions = []  # Reset current session
        self.recording_point_counter = 0  # Reset counter
        
        # UPDATE ALL UI BUTTONS
        self.record_btn.config(text="⏹️ Stop Recording")
        self.playback_btn.config(text="▶️ Play Back")        # Reset playback button
        self.generate_btn.config(text="🧠 Generate (Markov)") # Reset generate button
        self.record_status.config(text=f"Recording {self.current_emotional_state}... (Spacebar to stop)", foreground="red")
        self.markov_status.config(text="Capturing positions...", foreground="orange")
        
        # Show and reset progress bar - FIXED: Show correct 20s duration!
        self.progress_frame.pack(pady=8, fill=tk.X)  # Updated padding
        self.progress_var.set(0)
        self.progress_label.config(text="0:00 / 0:20 (0%)")  # FIXED: 20 seconds, not 2 minutes!
        
        print(f"🎬 Started TIME-BASED recording for {self.current_emotional_state}")
        print(f"⏰ Sampling at {1/self.record_interval:.0f} Hz (captures easing motions!)")
        print(f"💾 Memory limit: {self.max_recording_points} points (~{self.max_recording_points/40:.0f}s)")
        
        # Start time-based sampling timer
        self.start_recording_timer()
        
        # Start progress update timer
        self.update_progress()
        
        # Auto-stop after 20 seconds to prevent crashes! - FIXED: Matches display
        self.root.after(20000, self.auto_stop_recording)
    
    def start_recording_timer(self):
        """Start the time-based recording timer that samples positions continuously."""
        if self.recording:
            self.record_position_sample()
            # Schedule next sample
            interval_ms = int(self.record_interval * 1000)
            self.recording_timer = self.root.after(interval_ms, self.start_recording_timer)
        
        # Also start progress updates
        self.update_progress()
    
    def update_progress(self):
        """Update the recording progress bar and time display."""
        if not self.recording:
            return
            
        # Calculate elapsed time and progress
        elapsed = time.time() - self.record_start_time
        total_duration = 20.0  # FIXED: 20 seconds, not 2 minutes!
        progress_percent = min(100.0, (elapsed / total_duration) * 100)
        
        # Update progress bar
        self.progress_var.set(progress_percent)
        
        # Format time display
        elapsed_minutes = int(elapsed // 60)
        elapsed_seconds = int(elapsed % 60)
        total_minutes = int(total_duration // 60)
        total_seconds = int(total_duration % 60)
        
        time_text = f"{elapsed_minutes}:{elapsed_seconds:02d} / {total_minutes}:{total_seconds:02d} ({progress_percent:.0f}%)"
        self.progress_label.config(text=time_text)
        
        # CRITICAL FIX: Auto-stop recording at 20 seconds!
        if elapsed >= 20.0:
            print(f"🔄 Auto-stopping recording at {elapsed:.1f}s")
            self.stop_recording()
            return
        
        # Schedule next update (every 100ms for smooth progress)
        if self.recording:
            self.root.after(100, self.update_progress)
    
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
            'finger_positions': self.finger_positions.copy(),
            'finger_locks': self.finger_locks.copy(),      # NEW: Capture keyboard locks
            'pressed_keys': list(self.pressed_keys),       # NEW: Capture active keys
            'keyboard_targets': self.finger_lock_targets.copy()  # NEW: Capture keyboard targets
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
            
        # Hide progress bar
        self.progress_frame.pack_forget()
            
        duration = time.time() - self.record_start_time
        sample_count = len(self.recorded_positions)
        
        self.record_btn.config(text="🎬 Record Movement (20s)")
        self.record_status.config(text=f"Recorded {sample_count} samples in {duration:.1f}s", foreground="green")
        
        print(f"🎬 Stopped recording. Captured {sample_count} position samples in {duration:.1f} seconds")
        print(f"📊 Sample rate: {sample_count/duration:.1f} Hz")
        
        # Build Markov chain from recorded positions
        self.build_markov_chain()
        
        # Save to file
        self.save_recording()
        
        # FIXED: Refresh dataset dropdown to show newly generated Markov chain
        self.refresh_datasets()
        print("🔄 Dataset dropdown refreshed - new recording should appear!")
        print("💡 TIP: Record multiple 20s segments for this emotion to build richer datasets!")
    
    def build_markov_chain(self):
        """Build servo-based Markov chain from recorded movements - UNIFIED SYSTEM!"""
        emotion = self.current_emotional_state
        if emotion not in self.recorded_movements or len(self.recorded_movements[emotion]) < 2:
            print(f"⚠️ Not enough samples to build Markov chain for {emotion}")
            return
            
        movements = self.recorded_movements[emotion]
        print(f"🔗 Building servo-based Markov chain from {len(movements)} recorded movements...")
        
        # Initialize servo transition matrix
        servo_transitions = {}
        
        # Servo discretization - FINE resolution for ultra-smooth movement!
        discretization_step = 5  # 5 degrees for smooth transitions (was 10, but that was too chunky)
        def discretize_servo_state(servo_positions):
            return tuple(round(pos / discretization_step) * discretization_step for pos in servo_positions)
        
        # Build servo state transitions
        for i in range(len(movements) - 1):
            current_movement = movements[i]
            next_movement = movements[i + 1]
            
            # Get current and next servo states (discretized)
            current_state = discretize_servo_state(current_movement['servo_positions'])
            next_state = discretize_servo_state(next_movement['servo_positions'])
            
            # Build transition probability matrix
            current_key = str(current_state)  # Use string as key for JSON compatibility
            next_key = str(next_state)
            
            if current_key not in servo_transitions:
                servo_transitions[current_key] = {}
            if next_key not in servo_transitions[current_key]:
                servo_transitions[current_key][next_key] = 0
            servo_transitions[current_key][next_key] += 1
        
        # Convert counts to probabilities
        for state in servo_transitions:
            total = sum(servo_transitions[state].values())
            if total > 0:
                for next_state in servo_transitions[state]:
                    servo_transitions[state][next_state] /= total
        
        # Store the servo-based Markov chain with larger discretization
        self.markov_chains[emotion] = {
            'servo_transitions': servo_transitions,
            'discretization': discretization_step,  # degrees per step (now 10 for bigger movements)
            'total_samples': len(movements),
            'unique_states': len(servo_transitions)
        }
        
        print(f"✅ Servo Markov chain built for {emotion}:")
        print(f"   📊 {len(movements)} movements → {len(servo_transitions)} unique servo states")
        print(f"   🎯 Discretization: 5° steps for smooth generation")
        
        # Update status
        if hasattr(self, 'markov_status'):
            self.markov_status.config(
                text=f"{emotion}: {len(servo_transitions)} servo states",
                foreground="green"
            )
    
    def save_recording(self):
        """Save servo-based recorded movements AND Markov chain to file with optional custom name."""
        emotion = self.current_emotional_state
        if emotion not in self.recorded_movements or len(self.recorded_movements[emotion]) < 2:
            print("❌ No servo data to save")
            return
            
        os.makedirs("movement_recordings", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Use custom name if provided
        custom_name = self.dataset_name_var.get().strip()
        if custom_name:
            # Clean up the name for filename
            safe_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            filename = f"movement_recordings/{emotion}_{timestamp}_{safe_name}.json"
            self.dataset_name_var.set("")  # Clear for next time
        else:
            filename = f"movement_recordings/{emotion}_{timestamp}.json"
        
        # Get servo-based movements and Markov chain
        movements = self.recorded_movements[emotion]
        markov_chain = self.markov_chains.get(emotion, {})
        
        # Convert to JSON-safe format
        def convert_numpy_types(obj):
            """Recursively convert numpy data types to native Python types."""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Build servo-based dataset
        data = {
            'emotion': emotion,
            'timestamp': timestamp,
            'custom_name': custom_name if custom_name else None,
            'format_version': '2.0_servo_based',  # NEW: Mark as servo-based format
            'movement_count': len(movements),
            'duration': movements[-1]['time'] - movements[0]['time'] if len(movements) > 1 else 0,
            'sample_rate': len(movements) / (movements[-1]['time'] - movements[0]['time']) if len(movements) > 1 else 0,
            'servo_movements': convert_numpy_types(movements),  # NEW: Servo positions over time
            'markov_chain': convert_numpy_types(markov_chain),  # NEW: Servo-based Markov chain
            'discretization': markov_chain.get('discretization', 5)  # Servo discretization info
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            display_name = custom_name if custom_name else f"Auto-{timestamp}"
            print(f"💾 Saved servo-based recording '{display_name}' to {filename}")
            servo_states = markov_chain.get('unique_states', 0)
            print(f"🎯 Servo chain: {servo_states} unique states from {len(movements)} samples")
            
            # Update dataset display after saving
            self.refresh_datasets()
            
        except Exception as e:
            print(f"❌ ERROR saving servo recording: {e}")
            print(f"📍 Debug: movements type = {type(movements)}")
            if movements:
                print(f"📍 Debug: first movement = {movements[0]}")
            traceback.print_exc()
    
    def manual_save_recording(self):
        """Manual save button - saves current servo-based recordings and Markov chains."""
        emotion = self.current_emotional_state
        if emotion not in self.recorded_movements or len(self.recorded_movements[emotion]) < 2:
            print("❌ No servo data to save for current emotional state")
            self.record_status.config(text="No data to save", foreground="red")
            return
        
        # Check if there's a Markov chain too
        has_markov = emotion in self.markov_chains
        
        movements = self.recorded_movements[emotion]
        print(f"💾 Manual save requested for '{emotion}'")
        print(f"🎯 Recording: {len(movements)} servo samples")
        print(f"🧠 Markov chain: {'Yes' if has_markov else 'No'}")
        
        # Use the existing save function
        self.save_recording()
        
        # Update UI
        save_msg = f"Saved {len(movements)} servo samples"
        if has_markov:
            save_msg += " + Markov chain"
        self.record_status.config(text=save_msg, foreground="green")
    
    def refresh_datasets(self):
        """Scan for available datasets and update display."""
        print("🔄 Refreshing dataset list...")
        self.available_datasets = {}
        self.dataset_info = {}
        
        if not os.path.exists("movement_recordings"):
            self.update_dataset_display()
            return
            
        # Scan all JSON files in movement_recordings
        for filename in os.listdir("movement_recordings"):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join("movement_recordings", filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                emotion = data.get('emotion', 'unknown')
                if emotion not in self.available_datasets:
                    self.available_datasets[emotion] = []
                    
                # Create display info
                timestamp = data.get('timestamp', 'unknown')
                sample_count = data.get('position_count', 0)
                duration = data.get('duration', 0)
                # Extract unique states from both old and new chain formats
                unique_states = data.get('markov_chain', {}).get('unique_states', 0)  # Old format
                if unique_states == 0:  # Try new format
                    chain = data.get('markov_chain', {})
                    unique_states = chain.get('unique_cursor_states', 0) + chain.get('unique_finger_states', 0) + chain.get('unique_combined_states', 0)
                
                # Extract custom name from filename if it exists
                name_part = filename.replace(f"{emotion}_", "").replace(".json", "")
                
                # Check if this is a timestamped filename with custom name
                custom_name = data.get('custom_name')
                if custom_name:
                    display_name = custom_name.replace("_", " ").title()
                elif len(name_part) == 15 and name_part.isdigit():  # Standard timestamp only
                    display_name = f"Auto-{timestamp}"
                else:
                    # Parse custom name from filename structure: emotion_timestamp_customname.json
                    parts = name_part.split("_", 1)  # Split on first underscore after emotion
                    if len(parts) > 1 and len(parts[0]) == 15 and parts[0].isdigit():
                        # Format: timestamp_customname
                        custom_part = parts[1].replace("_", " ").title()
                        display_name = custom_part if custom_part else f"Recording-{timestamp}"
                    else:
                        display_name = name_part.replace("_", " ").title()
                
                dataset_info = {
                    'filename': filename,
                    'filepath': filepath,
                    'display_name': display_name,
                    'timestamp': timestamp,
                    'sample_count': sample_count,
                    'duration': duration,
                    'unique_states': unique_states,
                    'data': data
                }
                
                self.available_datasets[emotion].append(dataset_info)
                self.dataset_info[filename] = dataset_info
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        
        # Sort datasets by timestamp (newest first) - handle both string and numeric timestamps
        for emotion in self.available_datasets:
            try:
                self.available_datasets[emotion].sort(key=lambda x: str(x['timestamp']), reverse=True)
            except Exception as e:
                print(f"⚠️ Error sorting datasets for {emotion}: {e}")
                # Fallback: sort by filename instead
                self.available_datasets[emotion].sort(key=lambda x: x['filename'], reverse=True)
            
        self.update_dataset_display()
        print(f"✅ Loaded datasets for {len(self.available_datasets)} emotions")
    
    def update_dataset_display(self):
        """Update the dataset display for current emotion."""
        emotion = self.current_emotional_state
        datasets = self.available_datasets.get(emotion, [])
        
        # Update emotion dataset label
        if not datasets:
            self.current_emotion_dataset_label.config(
                text=f"📁 {emotion.title()}: No datasets (record some first!)", 
                foreground="#8B5CF6"
            )
            self.dataset_dropdown['values'] = []
            self.dataset_var.set("")
            self.dataset_status_label.config(text="No dataset selected for generation", foreground="gray")
        else:
            total_samples = sum(d.get('sample_count', 0) for d in datasets)
            total_states = sum(d.get('unique_states', 0) for d in datasets)
            self.current_emotion_dataset_label.config(
                text=f"📁 {emotion.title()}: {len(datasets)} dataset(s), {total_samples:,} samples", 
                foreground="darkgreen"
            )
            
            # Update dropdown
            display_options = []
            for i, dataset in enumerate(datasets):
                samples = dataset.get('sample_count', 0)
                states = dataset.get('unique_states', 0)
                duration = dataset.get('duration', 0)
                name = dataset.get('display_name', f'Dataset {i+1}')
                
                display_text = f"{name} ({samples:,} samples, {states} states, {duration:.0f}s)"
                display_options.append(display_text)
            
            self.dataset_dropdown['values'] = display_options
            
            # Select the most recent if none selected
            if not self.dataset_var.get() and display_options:
                self.dataset_var.set(display_options[0])
                self.active_datasets[emotion] = datasets[0]['filename']
    
    def on_dataset_selected(self, event=None):
        """Handle dataset selection from dropdown."""
        if not self.dataset_var.get():
            return
            
        emotion = self.current_emotional_state
        datasets = self.available_datasets.get(emotion, [])
        
        # Find selected dataset
        selected_index = self.dataset_dropdown.current()
        if 0 <= selected_index < len(datasets):
            selected_dataset = datasets[selected_index]
            self.active_datasets[emotion] = selected_dataset['filename']
            
            # Load the Markov chain from selected dataset
            try:
                data = selected_dataset['data']
                markov_chain = data.get('markov_chain', {})
                if markov_chain:
                    self.markov_chains[emotion] = markov_chain
                    
                    # Handle both old and new chain formats for display
                    if 'unique_cursor_states' in markov_chain:  # New enhanced format
                        cursor_states = markov_chain.get('unique_cursor_states', 0)
                        finger_states = markov_chain.get('unique_finger_states', 0) 
                        combined_states = markov_chain.get('unique_combined_states', 0)
                        self.markov_status.config(
                            text=f"Cursor: {cursor_states} | Fingers: {finger_states} | Combined: {combined_states} states", 
                            foreground="green"
                        )
                        print(f"🔗 Loaded enhanced Markov chain: {cursor_states} cursor, {finger_states} finger, {combined_states} combined states")
                    else:  # Old format compatibility
                        unique_states = markov_chain.get('unique_states', 0)
                        avg_transitions = sum(len(t) for t in markov_chain.get('transitions', {}).values()) / len(markov_chain.get('transitions', {})) if markov_chain.get('transitions') else 0
                        self.markov_status.config(text=f"{unique_states} states, {avg_transitions:.1f} avg transitions", foreground="green")
                        print(f"🔗 Loaded legacy Markov chain: {unique_states} states")
                        self.markov_status.config(text=f"{unique_states} states, {avg_transitions:.1f} avg transitions", foreground="green")
                        print(f"🔗 Loaded legacy Markov chain: {unique_states} states")
                    print(f"✅ Loaded Markov chain from {selected_dataset['display_name']}")
                else:
                    print(f"⚠️ No Markov chain in {selected_dataset['display_name']}")
            except Exception as e:
                print(f"❌ Error loading dataset: {e}")
    
    def delete_dataset(self):
        """Delete the currently selected dataset."""
        if not self.dataset_var.get():
            print("⚠️ No dataset selected for deletion")
            return
            
        emotion = self.current_emotional_state
        datasets = self.available_datasets.get(emotion, [])
        selected_index = self.dataset_dropdown.current()
        
        if 0 <= selected_index < len(datasets):
            dataset = datasets[selected_index]
            filename = dataset['filename']
            
            # Confirm deletion
            import tkinter.messagebox as msgbox
            result = msgbox.askyesno("Delete Dataset", 
                                   f"Delete dataset '{dataset['display_name']}'?\n\n"
                                   f"Samples: {dataset['sample_count']:,}\n"
                                   f"States: {dataset['unique_states']}\n"
                                   f"Duration: {dataset['duration']:.1f}s\n\n"
                                   f"This cannot be undone!")
            
            if result:
                try:
                    os.remove(dataset['filepath'])
                    print(f"🗑️ Deleted dataset: {dataset['display_name']}")
                    self.refresh_datasets()
                except Exception as e:
                    print(f"❌ Error deleting dataset: {e}")
    
    def show_dataset_details(self):
        """Show detailed information about selected dataset."""
        if not self.dataset_var.get():
            print("⚠️ No dataset selected")
            return
            
        emotion = self.current_emotional_state
        datasets = self.available_datasets.get(emotion, [])
        selected_index = self.dataset_dropdown.current()
        
        if 0 <= selected_index < len(datasets):
            dataset = datasets[selected_index]
            data = dataset['data']
            
            # Create details window
            details_window = tk.Toplevel(self.root)
            details_window.title(f"Dataset Details: {dataset['display_name']}")
            details_window.geometry("500x400")
            details_window.configure(bg=self.colors['bg_main'])
            
            # Details text
            details_text = tk.Text(details_window, wrap=tk.WORD, bg="white", fg="black",
                                 font=("Consolas", 10), height=20, width=60)
            details_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            # Format details
            markov_chain = data.get('markov_chain', {})
            details = f"""Dataset: {dataset['display_name']}
Emotion: {dataset['timestamp']}
File: {dataset['filename']}

📊 RECORDING STATISTICS:
• Sample Count: {dataset['sample_count']:,}
• Duration: {dataset['duration']:.1f} seconds  
• Sample Rate: {data.get('sample_rate', 0):.1f} Hz
• Movement Count: {data.get('movement_count', 0):,}

🔗 MARKOV CHAIN:
• Unique States: {dataset['unique_states']:,}
• Grid Resolution: {markov_chain.get('grid_size', 'unknown')}x{markov_chain.get('grid_size', 'unknown')}
• Coverage: {100 * dataset['unique_states'] / (markov_chain.get('grid_size', 80)**2):.2f}% of possible positions
• Total Transitions: {sum(len(t) for t in markov_chain.get('transitions', {}).values()):,}

📅 METADATA:
• Created: {dataset['timestamp']}
• File Size: {os.path.getsize(dataset['filepath']) / 1024:.1f} KB
"""
            
            details_text.insert(tk.END, details)
            details_text.config(state=tk.DISABLED)
    
    def auto_generate_name(self):
        """Auto-generate a creative name for the next recording."""
        import random
        
        # Creative name components based on emotion
        emotion_words = {
            'energized_engaged': ['Dynamic', 'Vibrant', 'Electric', 'Powerful', 'Intense'],
            'alert_curious': ['Exploratory', 'Inquisitive', 'Sharp', 'Quick', 'Bright'],
            'calm_observant': ['Serene', 'Peaceful', 'Steady', 'Balanced', 'Zen'],
            'quiet_detached': ['Subtle', 'Minimal', 'Gentle', 'Soft', 'Quiet'],
            'withdrawn_distant': ['Introspective', 'Deep', 'Contemplative', 'Solitary', 'Distant']
        }
        
        descriptors = ['Flow', 'Dance', 'Rhythm', 'Pattern', 'Expression', 'Movement', 'Gesture']
        
        emotion = self.current_emotional_state
        emotion_pool = emotion_words.get(emotion, ['Custom', 'Unique', 'Special'])
        
        name = f"{random.choice(emotion_pool)} {random.choice(descriptors)}"
        self.dataset_name_var.set(name)
        print(f"🎲 Generated name: {name}")
    
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
        """Save recorded movements AND Markov chain to file with optional custom name."""
        if self.current_emotional_state not in self.recorded_movements:
            return
            
        os.makedirs("movement_recordings", exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Use custom name if provided
        custom_name = self.dataset_name_var.get().strip()
        if custom_name:
            # Clean up the name for filename
            safe_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            filename = f"movement_recordings/{self.current_emotional_state}_{timestamp}_{safe_name}.json"
            self.dataset_name_var.set("")  # Clear for next time
        else:
            filename = f"movement_recordings/{self.current_emotional_state}_{timestamp}.json"
        
        movements = self.recorded_movements[self.current_emotional_state]
        positions = self.recorded_positions
        markov_chain = self.markov_chains.get(self.current_emotional_state, {})
        
        data = {
            'emotion': self.current_emotional_state,
            'timestamp': timestamp,
            'custom_name': custom_name if custom_name else None,
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
            
        display_name = custom_name if custom_name else f"Auto-{timestamp}"
        print(f"💾 Saved recording '{display_name}' to {filename}")
        print(f"📈 Enhanced chain - Cursor: {markov_chain.get('unique_cursor_states', 0)}, Fingers: {markov_chain.get('unique_finger_states', 0)}, Combined: {markov_chain.get('unique_combined_states', 0)} states from {len(positions)} samples")
        
        # Update dataset display after saving
        self.refresh_datasets()
    
    def start_playback(self):
        """Start playing back recorded movements for current emotional state."""
        # CRITICAL: Stop any other active operations first
        if self.generating:
            print("🛑 Stopping Markov generation to start playback")
            self.stop_markov_generation()
        
        if self.recording:
            print("🛑 Stopping recording to start playback")
            self.stop_recording()
        
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
        
        # UPDATE UI STATE
        self.playback_btn.config(text="⏹️ Stop Playback")
        self.record_btn.config(text="🎬 Record Movement (20s)")  # Reset record button
        self.generate_btn.config(text="🧠 Generate (Markov)")     # Reset generate button
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
                
                # ENHANCED: Restore complete finger state including keyboard data!
                if 'finger_positions' in movement and movement['finger_positions']:
                    self.finger_positions = movement['finger_positions'].copy()
                    self.finger_targets = movement['finger_positions'].copy()
                    
                    # NEW: Restore keyboard locks and pressed keys for visual feedback
                    if 'finger_locks' in movement:
                        self.finger_locks = movement['finger_locks'].copy()
                        
                    if 'pressed_keys' in movement:
                        self.pressed_keys = movement['pressed_keys'].copy()
                        
                    if 'keyboard_targets' in movement:
                        self.keyboard_targets = movement['keyboard_targets'].copy()
                    
                    print(f"🎬 Restoring complete state: fingers={self.finger_positions}, locks={self.finger_locks}, keys={self.pressed_keys}")
            else:
                break
        
        # Check if playback finished
        if self.current_playback:
            total_duration = self.current_playback[-1]['time'] - self.current_playback[0]['time']
            if playback_elapsed >= total_duration:
                self.stop_playback()
    
    def parse_markov_state_key(self, key_str):
        """Parse a string key back to tuple for generation (handles both old tuple and new string formats)."""
        if isinstance(key_str, tuple):
            return key_str  # Already a tuple (backwards compatibility)
        
        # Parse string representation of tuple: "(68, 57)" -> (68, 57)
        try:
            # Remove parentheses and split by comma
            clean_str = key_str.strip("()")
            parts = [part.strip() for part in clean_str.split(",")]
            
            # Handle different tuple formats
            if len(parts) == 2:
                # Simple (x, y) tuple
                return (int(parts[0]), int(parts[1]))
            elif len(parts) == 4:
                # Finger state tuple (f1, f2, f3, f4)
                return tuple(int(part) for part in parts)
            else:
                # Try to parse as generic tuple
                return tuple(int(part) for part in parts)
        except (ValueError, IndexError) as e:
            print(f"⚠️ Failed to parse Markov state key '{key_str}': {e}")
            # Return a fallback state
            return (40, 40)  # Center-ish position
    
    def start_markov_generation(self):
        """Start Markov chain generation for current emotional state."""
        # CRITICAL: Stop any other active operations first  
        if self.playing_back:
            print("🛑 Stopping playback to start Markov generation")
            self.stop_playback()
        
        if self.recording:
            print("🛑 Stopping recording to start Markov generation")
            self.stop_recording()
            
        if self.generating:
            self.stop_markov_generation()
            return
            
        if self.current_emotional_state not in self.markov_chains:
            print(f"❌ No Markov chain available for {self.current_emotional_state}")
            self.markov_status.config(text="No chain for this emotion", foreground="red")
            return
            
        chain = self.markov_chains[self.current_emotional_state]
        
        # Try different transition formats (enhanced, cursor, or legacy)
        transitions = None  
        transition_type = "unknown"
        
        if 'cursor_transitions' in chain:
            transitions = chain['cursor_transitions']
            transition_type = "cursor"
        elif 'transitions' in chain:
            transitions = chain['transitions']  
            transition_type = "legacy"
        else:
            print(f"❌ No valid transitions found in Markov chain for {self.current_emotional_state}")
            self.markov_status.config(text="Invalid chain format", foreground="red")
            return
            
        if not transitions:
            print(f"❌ Empty Markov chain for {self.current_emotional_state}")
            return
            
        # Start generation
        self.generating = True
        self.generation_start_time = time.time()
        
        # Pick a random starting state from available states  
        start_state_key = random.choice(list(transitions.keys()))
        start_state = self.parse_markov_state_key(start_state_key)
        self.current_markov_state = start_state_key  # Store the key for transitions
        
        # Convert grid state back to mouse position
        grid_size = chain.get('grid_size', 80)  # Default to ultra high resolution
        grid_x, grid_y = start_state[:2]  # Take first two elements (x, y)
        # BUGFIX: Ensure grid coordinates are numbers (convert from strings if needed)
        grid_x = float(grid_x) if isinstance(grid_x, str) else grid_x
        grid_y = float(grid_y) if isinstance(grid_y, str) else grid_y
        self.mouse_x = (grid_x + 0.5) / grid_size  # Center of grid cell
        self.mouse_y = (grid_y + 0.5) / grid_size
        
        self.generate_btn.config(text="⏹️ Stop Generation")
        self.markov_status.config(text=f"Generating {transition_type}...", foreground="purple")
        
        print(f"🎨 Started Markov generation for {self.current_emotional_state}")
        print(f"🎯 Using {transition_type} transitions with {len(transitions)} states")
        print(f"🎯 Starting from key '{start_state_key}' -> state {start_state} -> position ({self.mouse_x:.3f}, {self.mouse_y:.3f})")
        
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
        
        # Get transitions (try different formats)
        transitions = None
        if 'cursor_transitions' in chain:
            transitions = chain['cursor_transitions']
        elif 'transitions' in chain:
            transitions = chain['transitions']
        else:
            print("❌ No transitions found in chain")
            return
        
        if self.current_markov_state not in transitions:
            # Dead end - pick a new random state
            self.current_markov_state = random.choice(list(transitions.keys()))
            print(f"🔄 Dead end reached, jumping to key '{self.current_markov_state}'")
            return
            
        # Get possible next states and their probabilities
        next_states = transitions[self.current_markov_state]
        
        # Choose next state based on probabilities
        state_keys = list(next_states.keys())
        probabilities = list(next_states.values())
        
        # Weighted random choice
        next_state_key = random.choices(state_keys, weights=probabilities)[0]
        
        # Update current state
        self.current_markov_state = next_state_key
        
        # Parse the state key back to coordinates
        next_state = self.parse_markov_state_key(next_state_key)
        
        # Convert grid state to mouse position
        grid_size = chain.get('grid_size', 80)  # Default to ultra high resolution
        grid_x, grid_y = next_state[:2]  # Take first two elements (x, y)
        # BUGFIX: Ensure grid coordinates are numbers (convert from strings if needed)
        grid_x = float(grid_x) if isinstance(grid_x, str) else grid_x
        grid_y = float(grid_y) if isinstance(grid_y, str) else grid_y
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
        if hasattr(self, 'generation_timer') and self.generation_timer:
            self.root.after_cancel(self.generation_timer)
            self.generation_timer = None
        
        # CRITICAL: Reset button text properly
        self.generate_btn.config(text="🧠 Generate (Markov)")
        
        # Clear generation state
        self.current_markov_state = None
        
        # Update status
        emotion = self.current_emotional_state
        if emotion in self.markov_chains:
            chain = self.markov_chains[emotion]
            unique_states = chain.get('unique_states', 0)
            self.markov_status.config(text=f"Chain: {unique_states} states", foreground="blue")
        else:
            self.markov_status.config(text="Generation stopped", foreground="gray")
        
        print("🛑 Markov generation stopped")
        
        # Calculate duration only if start time exists    
        if hasattr(self, 'generation_start_time'):
            duration = time.time() - self.generation_start_time
            print(f"🎨 Stopped Markov generation after {duration:.1f} seconds")
        else:
            print(f"🎨 Stopped Markov generation")
        
        self.generate_btn.config(text="🧠 Generate (Markov)")
        self.markov_status.config(text="Generation stopped", foreground="gray")
    
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


def main():
    """Main function to start the interface."""
    print("� Starting Clean Emotional Hand Control...")
    
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
