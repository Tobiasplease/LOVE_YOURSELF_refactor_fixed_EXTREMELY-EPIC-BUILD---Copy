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
import datetime
from typing import Optional
from serial.tools import list_ports  # For COM port auto-detection
try:
    from PIL import Image, ImageTk  # For better image support
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[TIP] Install Pillow (pip install Pillow) for enhanced image support!")

# Import hand controller from local module
try:
    from hand_expression import HandExpressionController
    HAND_CONTROLLER_AVAILABLE = True
    print("[OK] Hand controller available")
except ImportError as e:
    print(f"[WARNING] Hand controller not available - simulation mode: {e}")
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
        self.root.title("Hand Control")
        
        # FIXED WINDOW SIZE - No more resizing! (Made wider to prevent cutoff)
        # Responsive window — fills available screen, resizable
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(820, screen_w - 80)
        win_h = min(screen_h - 100, 900)
        x = (screen_w - win_w) // 2
        y = max(0, (screen_h - win_h) // 2 - 30)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.resizable(True, True)
        self.root.minsize(600, 500)
        # F11 toggles fullscreen
        self.root.bind('<F11>', lambda e: self.root.attributes('-fullscreen',
                       not self.root.attributes('-fullscreen')))
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        # Soft Pastel Pink Color Scheme 🌸✨
        self.colors = {
            'bg_main': '#FEF7F7',        # Very light rose - softest main background
            'bg_frame': '#F8E8E8',       # Pale rose - subtle frame backgrounds  
            'bg_accent': '#F0D0D0',      # Soft dusty rose - gentle accent areas
            'text_main': '#5D4E75',      # Muted purple - easy on the eyes
            'text_accent': '#8B7B8B',    # Soft mauve - gentle accent text
            'button_bg': '#E8C5E8',      # Very pale lavender pink - soft buttons
            'button_hover': '#D8B5D8',   # Slightly deeper lavender - gentle hover
            'button_active': '#C8A5C8',  # Muted rose - subtle active state
            'success': '#C8E6C9',        # Soft mint green - gentle success
            'warning': '#FFE0B2',        # Pale peach - soft warnings  
            'error': '#FFCDD2',          # Light coral pink - gentle errors
            'canvas_bg': '#FFFEF8',      # Warm white - clean canvas
            'widget_fg': '#6B5B73'       # Soft purple - widget text
        }
        
        # Apply main background color
        self.root.configure(bg=self.colors['bg_main'])
        
        # Custom fonts setup (before style configuration) 🔤
        self.setup_custom_fonts()
        
        # Try to set a cute window icon (optional)
        try:
            # You can replace this with your own icon file!
            if os.path.exists("icon.ico"):
                self.root.iconbitmap("icon.ico")
        except:
            pass  # No icon is fine
        
        # Configure cute pastel ttk style 🌸✨
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a modern base theme
        
        # Configure ttk widget styles with our soft pastel theme
        self.style.configure('TFrame', background=self.colors['bg_main'])
        
        # Beautiful LabelFrame styling - no more boring grey boxes! 🌸
        self.style.configure('TLabelFrame', 
                           background=self.colors['bg_frame'],  # Soft pink background
                           bordercolor='#F472B6',  # Vibrant pink border  
                           relief='raised', 
                           borderwidth=2,
                           lightcolor='#FFE4E6',  # Light pink highlight
                           darkcolor='#DDA0DD')  # Soft purple shadow
        
        # Style the label text of frames
        self.style.configure('TLabelFrame.Label', 
                           background=self.colors['bg_frame'], 
                           foreground='#BE185D',  # Deep pink text
                           font=self.fonts['button'])  # Use our custom font
        
        self.style.configure('TLabel', background=self.colors['bg_main'], 
                           foreground=self.colors['text_main'])
        
        # Soft pastel buttons with subtle hover effects
        self.style.configure('TButton', background=self.colors['button_bg'], 
                           foreground=self.colors['widget_fg'], borderwidth=1, focuscolor='none',
                           relief='raised')
        self.style.map('TButton',
                      background=[('active', self.colors['button_hover']),
                                ('pressed', self.colors['button_active'])])
        
        self.style.configure('TCombobox', fieldbackground=self.colors['bg_frame'],
                           bordercolor=self.colors['bg_accent'], arrowcolor=self.colors['text_accent'])
        
        self.style.configure('TScale', background=self.colors['bg_main'],
                           troughcolor=self.colors['bg_frame'], bordercolor=self.colors['bg_accent'])
        
        self.style.configure('TCheckbutton', background=self.colors['bg_main'],
                           foreground=self.colors['text_main'], focuscolor='none')
        
        # Special style for emotion buttons to make them extra cute
        self.style.configure('Emotion.TButton', background=self.colors['button_bg'],
                           foreground=self.colors['text_main'], font=self.fonts['button'],
                           borderwidth=2, relief='raised')
        self.style.map('Emotion.TButton',
                      background=[('active', self.colors['button_hover']),
                                ('pressed', self.colors['button_active'])])
        
        # FIXED SCROLLABLE INTERFACE - No more auto-resizing
        # Create main canvas with scrollbar for scrollable content
        self.main_canvas = tk.Canvas(self.root, bg=self.colors['bg_main'], highlightthickness=0)
        self.main_canvas.configure(takefocus=True)  # Allow canvas to receive keyboard focus
        
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg=self.colors['bg_main'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar with better centering
        self.main_canvas.pack(side="left", fill="both", expand=True, padx=(20, 0))
        
        # Focus the canvas when clicked to ensure keyboard events work
        self.main_canvas.bind("<Button-1>", lambda e: self.main_canvas.focus_set())
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas (cross-platform)
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind("<Button-4>", lambda e: self.main_canvas.yview_scroll(-3, "units"))
        self.main_canvas.bind("<Button-5>", lambda e: self.main_canvas.yview_scroll(3, "units"))
        
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
        
        # COM Port Auto-Detection 🔌
        self.selected_port = tk.StringVar(value="Auto")
        self.available_ports = []
        
        # Image assets (optional) 🖼️
        self.images = {}  # Store loaded images
        self.load_custom_images()
        self.apply_background_image()  # Apply background after images are loaded
        
        # Direct control state - clean and simple!
        self.num_fingers = 4
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_targets = [90.0] * self.num_fingers
        
        # SIMPLIFIED KEYBOARD FINGER CONTROL - NO TIMERS, NO ACCUMULATION!
        self.finger_locks = [False] * self.num_fingers      # Which fingers are keyboard-locked
        self.finger_lock_targets = [90.0] * self.num_fingers # Target positions for locked fingers
        self.pressed_keys = set()                           # Currently pressed keys
        
        # NEW: ARM SERVO CONTROL - Same pattern as fingers
        self.num_arm_servos = 2
        self.arm_positions = [90.0] * self.num_arm_servos  # shoulder, elbow
        self.arm_targets = [90.0] * self.num_arm_servos
        self.arm_names = ['Shoulder', 'Elbow']
        self.finger_names = ['Index', 'Middle', 'Ring', 'Pinky']
        self.arm_locks = [False] * self.num_arm_servos       # Which arm servos are keyboard-locked
        self.arm_lock_targets = [90.0] * self.num_arm_servos  # Target positions for locked arm servos
        
        # ULTRA SIMPLE keyboard control parameters - NO TIMERS!
        self.keyboard_step_size = 2.0        # SMALLER steps for smoother control
        
        # SIMPLIFIED Key mappings for individual finger control
        self.key_mappings = {
            'w': (0, 'up'),    's': (0, 'down'),    # Index finger (F1)
            'e': (1, 'up'),    'd': (1, 'down'),    # Middle finger (F2)
            'r': (2, 'up'),    'f': (2, 'down'),    # Ring finger (F3)
            't': (3, 'up'),    'g': (3, 'down')     # Pinky finger (F4)
        }
        
        print(f"🎯 Initialized {self.num_fingers} finger positions: {self.finger_positions}")
        print(f"🎯 Initialized {self.num_fingers} finger targets: {self.finger_targets}")
        print(f"🦾 Initialized {self.num_arm_servos} arm positions: {self.arm_positions}")
        print(f"🦾 Initialized {self.num_arm_servos} arm targets: {self.arm_targets}")
        
        # Wave control parameters - the good stuff!
        self.cursor_sensitivity = tk.DoubleVar(value=3.0)
        
        # Wave control parameters - KEEP - working values for smooth wave-based control
        self.wave_strength = tk.DoubleVar(value=2.0)
        self.gravity_width = tk.DoubleVar(value=0.4)
        self.default_position = tk.DoubleVar(value=90.0)
        self.servo_range = tk.DoubleVar(value=60.0)  # ±60 degrees from default for better control precision
        
        # Control toggles - simplified
        self.reverse_vertical = tk.BooleanVar(value=True)  # Default to reversed (better usability)
        self.preserve_original_timing = tk.BooleanVar(value=False)  # Toggle for raw vs smooth Markov generation
        
        # Person detection simulation for testing integration behavior
        self.person_detected = tk.BooleanVar(value=False)
        self.last_detection_time = 0
        self.detection_cooldown = 60.0  # 60 seconds between detections
        self.is_frozen = False
        self.is_thawing = False
        self.thaw_start_time = 0
        self.thaw_duration = 2.0

        # Layer system
        self.recorded_layers = []  # List of layer dicts
        self.current_layer_data = []  # Frames for layer being recorded
        self.layer_recorded_servos = {'fingers': set(), 'arm': set()}  # Which servos active in current recording
        self.layer_playback_active = False
        self.layer_playback_start = 0
        self.layer_max_duration = 0  # Longest layer duration (loop length)
        
        # Mouse tracking (KEEP - this works)
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        
        # Animation state (KEEP - this works)
        self.running = False
        self.last_time = time.time()
        self.last_send_time = 0
        self.send_interval = 0.005  # 200 Hz for much smoother servo movement
        self.position_threshold = 1.0
        
        # PERFORMANCE OPTIMIZATION - Canvas rendering state
        self.last_canvas_update = 0
        self.canvas_update_interval = 0.033  # 30 Hz for canvas (much lower than servo rate)
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
        self.emotion_var = tk.StringVar(value=self.current_emotional_state)  # For UI binding
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
        self.dataset_directory = os.path.join(os.path.dirname(__file__), 'datasets')  # Base directory for datasets
        
        # Time-based recording state (captures stillness!) - WITH MEMORY MANAGEMENT
        self.recording_timer = None
        self.record_interval = 0.025  # 40 Hz - much higher resolution for easing capture
        self.recorded_positions = []  # Current recording session positions
        
        # MEMORY MANAGEMENT for recording system - OPTIMIZED FOR STABILITY
        self.max_recording_points = 800   # 20 seconds at 40Hz - stable duration with rich data
        self.recording_buffer_cleanup_interval = 50   # Clean up every 50 points
        self.recording_point_counter = 0
        
        # Markov chain generation state
        self.generating = False
        self.generation_start_time = 0
        self.current_markov_state = None  # Current state in generation
        self.prev_markov_state = None     # Previous state for second-order chains
        self.generation_timer = None
        self.generation_speed = 0.03  # 33Hz for smooth but responsive movement
        self.generation_smoothing = True  # Enable position smoothing
        self.last_generated_pos = (0.5, 0.5)  # For smooth interpolation
        self.target_generated_pos = (0.5, 0.5)  # Target position for easing
        self.generation_easing_factor = 0.3  # Smooth easing between positions
        self.previous_time = time.time()  # For timing calculations during recording
        
        # Keyboard control tracking for live Markov generation (separate from datasets)
        self.live_keyboard_states = []  # Store recent keyboard movements without polluting datasets
        self.live_keyboard_limit = 200  # Keep only recent movements for live generation
        
        # === ENHANCED DATASET CYCLING SYSTEM ===
        self.dataset_cycling_enabled = tk.BooleanVar(value=True)  # Auto-enabled for machine.py integration
        self.dataset_cycle_interval_min = 10.0  # Minimum 10 seconds between switches
        self.dataset_cycle_interval_max = 60.0  # Maximum 60 seconds between switches
        self.dataset_cycle_interval = self._get_random_cycle_interval()  # Start with random interval
        self.last_dataset_switch_time = 0
        self.current_dataset_index = 0  # Index in available datasets for current emotion
        
        # Dataset transition variables for smooth transitions
        self.transitioning_to_dataset = False
        self.dataset_transitioning = False
        self.transition_start_positions = None
        self.transition_target_positions = None
        self.transition_start_time = 0
        self.transition_duration = 1.5  # 1.5 seconds smooth ease
        
        # === AUTO EMOTION CYCLING SYSTEM ===
        self.emotion_cycling_enabled = tk.BooleanVar(value=False)
        self.emotion_cycle_interval = 180.0  # 3 minutes between emotion switches
        self.last_emotion_switch_time = 0
        self.current_emotion_index = 0  # Index in emotional_states list
        self.emotion_transition_duration = 10.0  # 10 seconds for smooth emotion transitions
        self.emotion_transitioning = False
        self.emotion_transition_start_time = 0
        self.emotion_transition_source = None
        self.emotion_transition_target = None
        
        # === MOOD MONITORING SYSTEM ===
        self.mood_monitoring_enabled = True  # Auto-read mood from machine.py
        self.mood_file_path = "hand_controller_state.json"
        self.last_mood_check_time = 0
        self.mood_check_interval = 2.0  # Check mood every 2 seconds
        self.last_mood_state = None  # Track last mood to avoid unnecessary switches
        print("🎭 Mood monitoring enabled - will auto-switch emotional states based on machine.py mood")
        
        self.setup_ui()
        self.start_control_loop()
        
        # STARTUP DATASET LOADING: Load all datasets automatically for autonomous operation
        print("🔄 Loading datasets for autonomous operation...")
        self.show_startup_loading()
        self.load_all_datasets_on_startup()
        
        # AUTO-CONNECT TO ARDUINO: Automatically connect to hand controller
        print("🔌 Auto-connecting to Arduino...")
        self.auto_connect_arduino()
        
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
        print("🎯 Rich 20s recordings with full data complexity for quality datasets!")
        
        # Perform startup dataset loading
        # self.perform_startup_loading()  # Method doesn't exist - commented out
        
        # Initialize the cleaner dataset display (will be updated after startup loading)
        # Don't update immediately - wait for startup loading to complete
        if hasattr(self, '_startup_loading_complete') and self._startup_loading_complete:
            self.update_emotion_dataset_display()
        
        # Schedule final dataset display update after UI is fully ready
        self.root.after(100, self._finalize_startup_display)
    
    def _finalize_startup_display(self):
        """Finalize the dataset display after startup loading is complete."""
        print("🔄 Finalizing dataset display after startup...")
        # Force update the current emotion display
        self.update_emotion_dataset_display()
        # Also update the dropdown to show proper data
        self.refresh_emotion_datasets()
        # Update UI status for current emotion
        if hasattr(self, 'update_markov_status_for_emotion'):
            self.update_markov_status_for_emotion(self.current_emotional_state)
        print("✅ Startup dataset display finalized")
        self._startup_loading_complete = True
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling (cross-platform)."""
        if event.delta:
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif event.num == 4:
            self.main_canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.main_canvas.yview_scroll(3, "units")
    
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
    
    def _open_wave_params_popup(self):
        """Open a popup window for wave control parameters."""
        if hasattr(self, '_wave_popup') and self._wave_popup and self._wave_popup.winfo_exists():
            self._wave_popup.lift()
            return
        popup = tk.Toplevel(self.root)
        popup.title("Wave Parameters")
        popup.geometry("400x280")
        popup.resizable(False, False)
        self._wave_popup = popup

        params = [
            ("Sensitivity", self.cursor_sensitivity, 0.5, 10.0, ".1f"),
            ("Wave Strength", self.wave_strength, 0.0, 5.0, ".1f"),
            ("Gravity Width", self.gravity_width, 0.1, 1.0, ".2f"),
            ("Default Position", self.default_position, 0, 180, ".0f"),
            ("Servo Range (+/-)", self.servo_range, 10, 90, ".0f"),
        ]
        for row, (name, var, lo, hi, fmt) in enumerate(params):
            ttk.Label(popup, text=f"{name}:").grid(row=row, column=0, sticky=tk.W, padx=8, pady=4)
            ttk.Scale(popup, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL,
                     length=200).grid(row=row, column=1, padx=5, pady=4, sticky=tk.EW)
            lbl = ttk.Label(popup, text="", width=6, font=("Courier", 10))
            lbl.grid(row=row, column=2, padx=5)
            var.trace_add("write", lambda *a, v=var, l=lbl, f=fmt: l.config(text=f"{v.get():{f}}"))
            lbl.config(text=f"{var.get():{fmt}}")
        popup.columnconfigure(1, weight=1)
        ttk.Checkbutton(popup, text="Reverse Vertical", variable=self.reverse_vertical).grid(
            row=len(params), column=0, columnspan=2, sticky=tk.W, padx=8, pady=6)

    def setup_ui(self):
        """Consolidated UI: compact toolbar, persistent timeline, fill workspace."""
        MONO = ("Courier", 10)
        MONO_SM = ("Courier", 9)
        all_servo_names = self.finger_names + self.arm_names
        C = self.colors

        # === ROW 1: CONNECTION (compact) ===
        conn = ttk.Frame(self.scrollable_frame)
        conn.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Label(conn, text="Port:").pack(side=tk.LEFT, padx=(0, 4))
        self.selected_port = tk.StringVar(value="Auto")
        self.port_combo = ttk.Combobox(conn, textvariable=self.selected_port, width=14, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(conn, text="Refresh", command=self.refresh_serial_ports, width=8, takefocus=False).pack(side=tk.LEFT, padx=2)
        self.connect_btn = ttk.Button(conn, text="Connect", command=self.toggle_connection, width=10, takefocus=False)
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        self.status_label = ttk.Label(conn, text="Disconnected", width=20, font=MONO_SM)
        self.status_label.pack(side=tk.LEFT, padx=6)
        self.refresh_serial_ports()

        # === ROW 2: EMOTION BUTTONS ===
        emo_frame = ttk.Frame(self.scrollable_frame)
        emo_frame.pack(fill=tk.X, padx=8, pady=2)
        emojis = {'energized_engaged': 'E', 'alert_curious': 'A', 'calm_observant': 'C',
                  'quiet_detached': 'Q', 'withdrawn_distant': 'W'}
        for ename in self.emotional_states:
            lbl = ename.replace('_', ' ').title()
            ttk.Button(emo_frame, text=lbl, width=14, takefocus=False,
                       command=lambda e=ename: self.switch_emotional_state(e)).pack(side=tk.LEFT, padx=2)
        self.current_state_label = ttk.Label(emo_frame, text=self.current_emotional_state,
                                             font=MONO_SM, width=20)
        self.current_state_label.pack(side=tk.RIGHT, padx=4)

        # === ROW 3: TOOLBAR — Record / Play / Generate / Save + Mode toggle ===
        toolbar = ttk.Frame(self.scrollable_frame)
        toolbar.pack(fill=tk.X, padx=8, pady=2)

        self.record_btn = ttk.Button(toolbar, text="REC (20s)", command=self.start_recording, width=10, takefocus=False)
        self.record_btn.pack(side=tk.LEFT, padx=2)
        self.playback_btn = ttk.Button(toolbar, text="Play", command=self._play_layers_direct, width=7, takefocus=False)
        self.playback_btn.pack(side=tk.LEFT, padx=2)
        self.generate_btn = ttk.Button(toolbar, text="Markov", command=self.start_markov_generation, width=8, takefocus=False)
        self.generate_btn.pack(side=tk.LEFT, padx=2)
        self.save_btn = ttk.Button(toolbar, text="Save", command=self.manual_save_recording, width=6, takefocus=False)
        self.save_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # Control mode
        self.control_mode = tk.StringVar(value="cursor")
        ttk.Radiobutton(toolbar, text="Wave", variable=self.control_mode,
                        value="cursor", command=self._on_mode_change).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(toolbar, text="Single", variable=self.control_mode,
                        value="manual", command=self._on_mode_change).pack(side=tk.LEFT, padx=3)

        # Servo selector
        self.servo_selector_frame = ttk.Frame(toolbar)
        self.servo_selector_frame.pack(side=tk.LEFT, padx=(6, 0))
        self.selected_servo_idx = tk.IntVar(value=0)
        self.servo_combo = ttk.Combobox(self.servo_selector_frame, width=10, state="readonly",
                                        values=all_servo_names)
        self.servo_combo.current(0)
        self.servo_combo.pack(side=tk.LEFT)
        self.servo_combo.bind("<<ComboboxSelected>>", self._on_servo_select)

        # Wave params button
        ttk.Button(toolbar, text="Params...", command=self._open_wave_params_popup,
                  width=8, takefocus=False).pack(side=tk.LEFT, padx=4)

        # Status (right side, fixed width)
        self.markov_status = ttk.Label(toolbar, text="No chains", font=MONO_SM, width=28, anchor=tk.E)
        self.markov_status.pack(side=tk.RIGHT, padx=2)
        self.record_status = ttk.Label(toolbar, text="Ready", font=MONO_SM, width=16, anchor=tk.E)
        self.record_status.pack(side=tk.RIGHT, padx=2)

        # Progress bar (hidden until recording)
        self.progress_var = tk.DoubleVar()
        self.progress_container = ttk.Frame(toolbar)
        self.progress_container.pack(side=tk.RIGHT, padx=4)
        self.progress_frame = ttk.Frame(self.progress_container)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                            length=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=2)
        self.progress_label = ttk.Label(self.progress_frame, text="", width=10, font=("Arial", 8))
        self.progress_label.pack(side=tk.LEFT)

        # === ROW 4: TIMELINE ===
        self.timeline_canvas = tk.Canvas(self.scrollable_frame, height=28, bg='#1a1a2e',
                                         highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, padx=8, pady=(4, 2))

        # === ROW 5: WORKSPACE (fills remaining space) ===
        workspace_outer = tk.Frame(self.scrollable_frame, bg=C['canvas_bg'], relief='sunken', bd=1)
        workspace_outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)

        # -- Wave workspace --
        self.canvas_width = 600
        self.canvas_height = 280
        self.wave_workspace = tk.Frame(workspace_outer, bg=C['canvas_bg'])

        self.canvas = tk.Canvas(self.wave_workspace, height=self.canvas_height,
                               bg=C['canvas_bg'], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Button-1>', self.on_mouse_click)

        # -- Single servo workspace --
        self.single_servo_frame = tk.Frame(workspace_outer, bg=C['canvas_bg'])

        single_inner = tk.Frame(self.single_servo_frame, bg=C['canvas_bg'])
        single_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Large bar (left, tall)
        large_bar_frame = tk.Frame(single_inner, bg=C['canvas_bg'])
        large_bar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5))

        self.single_servo_name_label = ttk.Label(large_bar_frame, text="Index", font=("Arial", 11, "bold"))
        self.single_servo_name_label.pack(pady=(0, 3))

        self.single_servo_canvas = tk.Canvas(large_bar_frame, width=100, height=350,
                                             bg='#FFFEF8', highlightthickness=1,
                                             highlightcolor='#B8A9D9')
        self.single_servo_canvas.pack(fill=tk.Y, expand=True)
        self.single_servo_canvas.bind('<B1-Motion>', self._on_single_servo_drag)
        self.single_servo_canvas.bind('<Button-1>', self._on_single_servo_drag)

        self.single_servo_label = ttk.Label(large_bar_frame, text="090.0", font=MONO, width=6)
        self.single_servo_label.pack(pady=(3, 0))

        # Small bars (right, all servos)
        small_panel = tk.Frame(single_inner, bg=C['canvas_bg'])
        small_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.small_bar_canvases = []
        self.small_bar_labels = []
        bars_row = tk.Frame(small_panel, bg=C['canvas_bg'])
        bars_row.pack(anchor=tk.W, pady=10)

        for i, name in enumerate(all_servo_names):
            col = tk.Frame(bars_row, bg=C['canvas_bg'])
            col.pack(side=tk.LEFT, padx=6)
            ttk.Label(col, text=name[:4], font=("Arial", 8)).pack()
            bar = tk.Canvas(col, width=28, height=180, bg='#FFFEF8',
                           highlightthickness=1, highlightcolor='#D0D0D0')
            bar.pack(pady=2)
            bar.bind('<Button-1>', lambda e, idx=i: self._select_servo_by_bar(idx))
            val_lbl = ttk.Label(col, text=" 90", font=MONO_SM, width=4)
            val_lbl.pack()
            self.small_bar_canvases.append(bar)
            self.small_bar_labels.append(val_lbl)

        # Show wave mode initially
        self.wave_workspace.pack(fill=tk.BOTH, expand=True)

        # === ROW 6: LAYERS (compact footer) ===
        layer_bar = ttk.Frame(self.scrollable_frame)
        layer_bar.pack(fill=tk.X, padx=8, pady=(2, 1))

        ttk.Label(layer_bar, text="Layers:", font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        self.layer_info_label = ttk.Label(layer_bar, text="0", font=MONO_SM, width=4)
        self.layer_info_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(layer_bar, text="Delete", width=7, takefocus=False,
                  command=self._delete_selected_layer).pack(side=tk.RIGHT, padx=2)
        ttk.Button(layer_bar, text="Clear", width=6, takefocus=False,
                  command=self._clear_all_layers).pack(side=tk.RIGHT, padx=2)

        self.layer_listbox = tk.Listbox(self.scrollable_frame, height=3, font=MONO_SM,
                                        selectmode=tk.SINGLE, bg='#FAFAFA')
        self.layer_listbox.pack(fill=tk.X, padx=8, pady=(0, 2))

        # === ROW 7: DATASET (compact) ===
        ds_bar = ttk.Frame(self.scrollable_frame)
        ds_bar.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(ds_bar, text="Dataset:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.dataset_var = tk.StringVar()
        self.dataset_dropdown = ttk.Combobox(ds_bar, textvariable=self.dataset_var, width=40,
                                             state="readonly", font=("Arial", 8))
        self.dataset_dropdown.pack(side=tk.LEFT, padx=4)
        self.dataset_dropdown.bind("<<ComboboxSelected>>", self.on_dataset_selected)
        ttk.Button(ds_bar, text="Refresh", width=7, takefocus=False,
                  command=self.refresh_datasets).pack(side=tk.LEFT, padx=2)

        # Cycling toggles
        cycle_bar = ttk.Frame(self.scrollable_frame)
        cycle_bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.dataset_cycling_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(cycle_bar, text="Auto-cycle datasets", variable=self.dataset_cycling_enabled,
                        command=self.on_dataset_cycling_toggle).pack(side=tk.LEFT, padx=4)
        self.emotion_cycling_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(cycle_bar, text="Auto-cycle emotions", variable=self.emotion_cycling_enabled,
                        command=self.on_emotion_cycling_toggle).pack(side=tk.LEFT, padx=4)

        # Dummy widgets expected by other methods
        self.dataset_cycle_status = ttk.Label(cycle_bar, text="", font=("Arial", 7))
        self.dataset_cycle_status.pack(side=tk.RIGHT)
        self.emotion_cycle_status = ttk.Label(cycle_bar, text="", font=("Arial", 7))
        self.emotion_cycle_status.pack(side=tk.RIGHT, padx=4)
        self.current_emotion_dataset_label = ttk.Label(ds_bar, text="", font=("Arial", 8))
        self.current_emotion_dataset_label.pack(side=tk.RIGHT)
        self.dataset_selector_label = ttk.Label(ds_bar)
        self.dataset_status_label = ttk.Label(ds_bar)
        self.dataset_name_var = tk.StringVar()
        self.preserve_original_timing = tk.BooleanVar(value=False)
        self.person_detected = tk.BooleanVar(value=False)
        self.reverse_vertical = tk.BooleanVar(value=True)
        self.manual_override_mode = tk.BooleanVar(value=False)
        self.show_path = tk.BooleanVar(value=True)
        self.text_field_has_focus = False

        # Recording state
        self.recording = False
        self.playing_back = False
        self.recorded_movements = {}
        self.playback_start_time = 0
        self.current_playback = []

    def load_custom_images(self):
        """Load custom images if they exist in the directory."""
        image_files = {
            'logo': ['logo.png', 'logo.gif', 'icon.png', 'logo.jpg'],
            'background': ['background.png', 'bg.png', 'background.jpg', 'wallpaper.png'],
            'header': ['header.png', 'banner.png', 'header.jpg'],
            'emotion_icons': ['emotions.png', 'emotions.gif'],
            'decorative': ['flower.png', 'star.png', 'heart.png', 'sparkle.png']
        }
        
        for image_type, filenames in image_files.items():
            for filename in filenames:
                if os.path.exists(filename):
                    try:
                        if PIL_AVAILABLE:
                            # Use Pillow for better image handling
                            img = Image.open(filename)
                            
                            # Different sizes for different purposes
                            if image_type == 'logo':
                                img = img.resize((48, 48), Image.Resampling.LANCZOS)
                            elif image_type == 'header':
                                img = img.resize((600, 80), Image.Resampling.LANCZOS)
                            elif image_type == 'background':
                                # Make background much larger to ensure full coverage
                                img = img.resize((800, 1500), Image.Resampling.LANCZOS)
                            elif image_type == 'decorative':
                                img = img.resize((24, 24), Image.Resampling.LANCZOS)
                            else:
                                img = img.resize((32, 32), Image.Resampling.LANCZOS)
                                
                            self.images[image_type] = ImageTk.PhotoImage(img)
                        else:
                            # Fallback to basic tkinter
                            self.images[image_type] = tk.PhotoImage(file=filename)
                        print(f"[IMAGE] Loaded custom image: {filename} as {image_type}")
                        break
                    except Exception as e:
                        print(f"[WARNING] Could not load image {filename}: {e}")
        
        # Create some cute default images if no custom ones found
        if 'decorative' not in self.images:
            self.create_default_decorative_images()

    def apply_background_image(self):
        """Apply background image to the scrollable area."""
        if hasattr(self, 'images') and 'background' in self.images:
            try:
                # Place the background image in the scrollable frame area
                self.bg_label = tk.Label(self.scrollable_frame, image=self.images['background'])
                # Place the background image to cover the full 750px window
                self.bg_label.place(x=-100, y=0, width=800, height=1500)  # Oversized and offset to guarantee full coverage
                # Send it to the back so all other widgets appear on top
                self.bg_label.lower()
                
                # Make the scrollable frame completely transparent
                self.scrollable_frame.configure(bg='')  # Completely transparent
                
                print("[OK] Background image applied with full coverage!")
                self.has_background = True
                
            except Exception as e:
                print(f"[ERROR] Failed to apply background image: {e}")
        else:
            print("[INFO] No background image found")
            self.has_background = False
            
    def make_frames_transparent(self):
        """Make all frames transparent when background image is present."""
        if hasattr(self, 'has_background') and self.has_background:
            # Update the color scheme for transparency
            self.bg_outer_frame = '#B8A9D9'      # Keep purple border
            self.bg_header_frame = '#F8F5FF'     # Very light lavender (almost transparent)
            self.bg_content_frame = '#FDFDFF'    # Almost white (almost transparent)
            
            # Find all frames and update their colors
            try:
                self.update_all_frame_colors()
                print("[OK] Made frames transparent for background image!")
            except Exception as e:
                print(f"[WARNING] Could not update frame transparency: {e}")
    
    def update_all_frame_colors(self):
        """Update all existing frames with new color scheme."""
        # This will recursively update all frames in the interface
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                # Update main frame colors
                if widget.cget('bg') in ['#E8C5E8', '#FDFCFF']:  # If it's one of our custom frames
                    widget.configure(bg=self.bg_content_frame)
                    
                # Update child frames
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame):
                        if child.cget('bg') in ['#E8C5E8', '#FDFCFF']:
                            child.configure(bg=self.bg_content_frame)
            
    def update_frame_transparency(self):
        """Update frame backgrounds to be semi-transparent when background image is present."""
        if hasattr(self, 'has_background') and self.has_background:
            # For frames with background image - use lighter, more transparent-looking versions
            self.bg_outer_frame = '#B8A9D9'    # Keep purple border solid but make it thinner
            self.bg_header_frame = '#F4EEFF'   # Very light lavender for headers 
            self.bg_content_frame = '#FEFEFF'   # Almost white for content areas
        else:
            # Original solid colors when no background
            self.bg_outer_frame = '#B8A9D9'    # Solid purple border
            self.bg_header_frame = '#E8C5E8'   # Solid header
            self.bg_content_frame = '#FDFCFF'  # Solid content
            
    def create_styled_frame(self, parent, frame_type='outer'):
        """Create a styled frame with appropriate background based on image presence."""
        if not hasattr(self, 'has_background'):
            self.has_background = False
            self.update_frame_transparency()
            
        if frame_type == 'outer':
            return tk.Frame(parent, bg=self.bg_outer_frame, relief='raised', bd=1 if self.has_background else 2)
        elif frame_type == 'header':
            return tk.Frame(parent, bg=self.bg_header_frame, relief='flat', bd=0)
        elif frame_type == 'content':
            return tk.Frame(parent, bg=self.bg_content_frame, relief='flat', bd=0)
        else:
            return tk.Frame(parent, bg=self.bg_outer_frame, relief='raised', bd=2)

    def create_default_decorative_images(self):
        """Create cute default decorative images using code."""
        try:
            if PIL_AVAILABLE:
                # Create a cute heart shape
                heart_img = Image.new('RGBA', (24, 24), (0, 0, 0, 0))
                heart_pixels = [
                    "      ████████      ",
                    "    ██████████████  ",
                    "  ████████████████  ",
                    "  ████████████████  ",
                    "    ██████████████  ",
                    "      ████████████  ",
                    "        ████████    ",
                    "          ████      "
                ]
                # This is just a demo - you can get much fancier!
                self.images['heart'] = ImageTk.PhotoImage(heart_img)
                print("💖 Created cute default heart image!")
        except Exception as e:
            print(f"⚠️ Could not create default images: {e}")
    
    def setup_custom_fonts(self):
        """Setup beautiful custom fonts for the interface."""
        # Try different cute font families (fallback to safe defaults)
        self.fonts = {
            'title': self.get_best_font(['Segoe UI', 'Comic Sans MS', 'Arial'], 16, 'bold'),
            'subtitle': self.get_best_font(['Segoe UI', 'Calibri', 'Arial'], 10, 'normal'),
            'button': self.get_best_font(['Segoe UI', 'Tahoma', 'Arial'], 9, 'bold'),
            'label': self.get_best_font(['Segoe UI', 'Calibri', 'Arial'], 9, 'normal'),
            'small': self.get_best_font(['Segoe UI', 'Arial'], 8, 'normal'),
            'canvas': self.get_best_font(['Consolas', 'Courier New', 'Arial'], 9, 'bold')
        }
        print(f"🔤 Using fonts: {[f[0] for f in self.fonts.values()]}")
    
    def get_best_font(self, font_list, size, weight):
        """Get the best available font from a preference list."""
        import tkinter.font as tkfont
        
        for font_name in font_list:
            try:
                # Test if the font exists by creating it
                test_font = tkfont.Font(family=font_name, size=size, weight=weight)
                return (font_name, size, weight)
            except:
                continue
        
        # Fallback to Arial if nothing else works
        return ('Arial', size, weight)

    def create_tooltip(self, widget, text):
        """Create a cute tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            tooltip.configure(bg=self.colors['bg_accent'])
            
            label = tk.Label(tooltip, text=text, background=self.colors['bg_accent'],
                           foreground=self.colors['text_main'], font=('Arial', 8),
                           relief='solid', borderwidth=1, padx=8, pady=4)
            label.pack()
            
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

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
    
    def on_dataset_cycling_toggle(self):
        """Handle dataset cycling toggle."""
        if self.dataset_cycling_enabled.get():
            print("🎲 Dataset cycling enabled - will switch datasets every 20 seconds")
            self.dataset_cycle_status.config(text="Active", foreground="green")
            self.last_dataset_switch_time = time.time()
        else:
            print("🛑 Dataset cycling disabled")
            self.dataset_cycle_status.config(text="Inactive", foreground="gray")
    
    def update_cycling_behavior(self):
        """Update dataset and emotion cycling behavior."""
        current_time = time.time()
        
        # Handle dataset cycling (random intervals between 10s-1min)
        if (self.dataset_cycling_enabled.get() and 
            current_time - self.last_dataset_switch_time >= self.dataset_cycle_interval):
            self.cycle_to_next_dataset()
            self.last_dataset_switch_time = current_time
            # Set new random interval for next cycle
            self.dataset_cycle_interval = self._get_random_cycle_interval()
        
        # Handle emotion cycling (every 3 minutes) 
        if (self.emotion_cycling_enabled.get() and 
            current_time - self.last_emotion_switch_time >= self.emotion_cycle_interval):
            self.cycle_to_next_emotion()
            self.last_emotion_switch_time = current_time
        
        # Update status displays with countdown timers
        self.update_cycling_status_displays(current_time)
    
    def cycle_to_next_dataset(self):
        """Cycle to the next dataset within the current emotion with smooth transition."""
        emotion = self.current_emotional_state
        # Use the actual available datasets from our refresh system, not the filesystem check
        available_datasets = self.available_datasets.get(emotion, [])
        
        if len(available_datasets) > 1:
            # Get current dataset display name and find next one
            current_dataset = self.dataset_var.get()
            print(f"🎲 Cycling from current dataset: {current_dataset}")
            
            # Create list of display names to match dropdown
            display_names = []
            for dataset_info in available_datasets:
                timestamp = dataset_info.get('timestamp', 'unknown')
                sample_count = dataset_info.get('sample_count', 0)
                duration = dataset_info.get('duration', 0)
                custom_name = dataset_info.get('custom_name', '')
                
                # Format timestamp to be more readable (same logic as update_dataset_display)
                try:
                    if len(timestamp) >= 8:
                        date_part = timestamp[:8]
                        time_part = timestamp[9:15] if len(timestamp) > 9 else "000000"
                        formatted_date = f"{date_part[4:6]}/{date_part[6:8]}"
                        formatted_time = f"{time_part[:2]}:{time_part[2:4]}"
                        readable_time = f"{formatted_date} {formatted_time}"
                    else:
                        readable_time = timestamp
                except:
                    readable_time = timestamp
                
                if custom_name:
                    display_name = f"{custom_name} ({readable_time}) - {sample_count:,} samples, {duration:.1f}s"
                else:
                    display_name = f"Recording {readable_time} - {sample_count:,} samples, {duration:.1f}s"
                
                display_names.append(display_name)
            
            try:
                current_index = display_names.index(current_dataset)
                # RANDOM SELECTION instead of sequential cycling
                available_indices = [i for i in range(len(display_names)) if i != current_index]
                if available_indices:
                    next_index = random.choice(available_indices)
                else:
                    next_index = current_index  # Only one dataset available
            except ValueError:
                # Current dataset not in list, pick random starting point
                next_index = random.randint(0, len(display_names) - 1)
                print(f"⚠️ Current dataset not found in list, randomly selecting dataset")
            
            next_dataset = display_names[next_index]
            next_dataset_info = available_datasets[next_index]
            
            print(f"🎲 Random dataset switch: {emotion} → {next_dataset}")
            print(f"📁 Using file: {next_dataset_info['filename']}")
            
            # Start smooth transition to new dataset
            self.start_dataset_transition(next_dataset, next_dataset_info, emotion)
        else:
            print(f"🎲 Dataset cycling: Only {len(available_datasets)} dataset(s) available for {emotion}, no cycling needed")
    
    def cycle_to_next_emotion(self):
        """Cycle to the next emotion with smooth transition."""
        emotion_list = list(self.emotional_states.keys())
        
        if len(emotion_list) > 1:
            # Calculate next emotion
            self.current_emotion_index = (self.current_emotion_index + 1) % len(emotion_list)
            next_emotion = emotion_list[self.current_emotion_index]
            
            print(f"🎭 Emotion cycling: {self.current_emotional_state} → {next_emotion}")
            
            # Start smooth transition
            self.start_emotion_transition(next_emotion)
    
    def get_available_datasets_for_emotion(self, emotion):
        """Get list of available datasets for a specific emotion."""
        try:
            emotion_dir = os.path.join(self.dataset_directory, emotion)
            if os.path.exists(emotion_dir):
                datasets = [d for d in os.listdir(emotion_dir) 
                           if os.path.isdir(os.path.join(emotion_dir, d))]
                return sorted(datasets) if datasets else []
            return []
        except Exception as e:
            print(f"❌ Error getting datasets for {emotion}: {e}")
            return []
    
    def start_emotion_transition(self, target_emotion):
        """Start smooth transition to target emotion."""
        if target_emotion not in self.emotional_states:
            print(f"❌ Unknown emotion: {target_emotion}")
            return
        
        self.emotion_transitioning = True
        self.emotion_transition_target = target_emotion
        self.emotion_transition_start_time = time.time()
        
        # Get available datasets for target emotion - FIXED to use actual loaded datasets
        available_datasets = self.available_datasets.get(target_emotion, [])
        if available_datasets:
            # Use the same display name format as the dropdown
            dataset_info = available_datasets[0]  # Get first dataset info
            timestamp = dataset_info.get('timestamp', 'unknown')
            sample_count = dataset_info.get('sample_count', 0)
            duration = dataset_info.get('duration', 0)
            custom_name = dataset_info.get('custom_name', '')
            
            # Format timestamp to match dropdown display
            try:
                if len(timestamp) >= 8:
                    date_part = timestamp[:8]
                    time_part = timestamp[9:15] if len(timestamp) > 9 else "000000"
                    formatted_date = f"{date_part[4:6]}/{date_part[6:8]}"
                    formatted_time = f"{time_part[:2]}:{time_part[2:4]}"
                    readable_time = f"{formatted_date} {formatted_time}"
                else:
                    readable_time = timestamp
            except:
                readable_time = timestamp
            
            if custom_name:
                target_dataset = f"{custom_name} ({readable_time}) - {sample_count:,} samples, {duration:.1f}s"
            else:
                target_dataset = f"Recording {readable_time} - {sample_count:,} samples, {duration:.1f}s"
                
            self.dataset_var.set(target_dataset)
            self.active_datasets[target_emotion] = dataset_info['filename']
            
            # CRITICAL: Load the Markov chain for the target emotion immediately
            if self.load_markov_chain_from_dataset(dataset_info, target_emotion):
                print(f"🔗 Loaded Markov chain for emotion transition to {target_emotion}")
            else:
                print(f"⚠️ Failed to load Markov chain for {target_emotion}")
                
            print(f"🎭 Emotion transition: Selected dataset {target_dataset} for {target_emotion}")
        else:
            print(f"⚠️ No datasets available for {target_emotion} - emotion will switch but no dataset selected")
        
        print(f"🌊 Starting smooth transition to {target_emotion}")
    
    def update_emotion_transition(self):
        """Update smooth emotion transition progress."""
        if not self.emotion_transitioning:
            return
        
        current_time = time.time()
        elapsed = current_time - self.emotion_transition_start_time
        
        if elapsed >= self.emotion_transition_duration:
            # Complete transition
            self.current_emotional_state = self.emotion_transition_target
            self.emotion_var.set(self.current_emotional_state)
            self.emotion_transitioning = False
            print(f"✅ Emotion transition complete: {self.current_emotional_state}")
            
            # Update UI and restart Markov if needed
            self.update_emotion_display()
            if self.generating:
                self.restart_markov_with_new_dataset()
        else:
            # Smooth interpolation between emotions
            progress = elapsed / self.emotion_transition_duration
            self.interpolate_emotional_state(progress)
    
    def interpolate_emotional_state(self, progress):
        """Smoothly interpolate between current and target emotional states."""
        if not self.emotion_transitioning:
            return
        
        # Use easing function for smooth transition
        eased_progress = self.ease_in_out_cubic(progress)
        
        current_state = self.emotional_states[self.current_emotional_state]
        target_state = self.emotional_states[self.emotion_transition_target]
        
        # Interpolate each parameter
        interpolated_state = {}
        for key in current_state:
            if isinstance(current_state[key], (int, float)):
                current_val = current_state[key]
                target_val = target_state[key]
                interpolated_val = current_val + (target_val - current_val) * eased_progress
                interpolated_state[key] = interpolated_val
            else:
                # For non-numeric values, switch at halfway point
                interpolated_state[key] = target_state[key] if eased_progress > 0.5 else current_state[key]
        
        # Apply interpolated state to hand expression
        if hasattr(self, 'hand_controller'):
            self.hand_controller.apply_emotional_state(interpolated_state)
    
    def update_emotion_display(self):
        """Update the emotion display after transition."""
        self.emotion_var.set(self.current_emotional_state)
        self.update_emotion_dataset_display()
        print(f"🎭 Emotion display updated: {self.current_emotional_state}")
    
    def start_dataset_transition(self, target_dataset, target_dataset_info, emotion):
        """Start simple smooth transition to new dataset."""
        # Store where we're starting from (fingers + arm combined)
        self.transition_start_positions = self.finger_positions.copy() + self.arm_positions.copy()
        self.dataset_transition_start_time = time.time()
        self.dataset_transition_duration = 1.5
        self.dataset_transitioning = True

        self.dataset_transition_target = {
            'display_name': target_dataset,
            'dataset_info': target_dataset_info,
            'emotion': emotion
        }

        # Get first position from new dataset as target
        target_data = target_dataset_info['data']
        recordings = target_data.get('recordings', [])
        if recordings:
            first_recording = recordings[0]
            # Build target from available servo keys, fallback to 90
            target = []
            for i in range(self.num_fingers + self.num_arm_servos):
                target.append(first_recording.get(f'servo{i+1}', 90))
            self.transition_target_positions = target
        else:
            self.transition_target_positions = self.finger_positions.copy() + self.arm_positions.copy()
        
        print(f"🌊 Starting smooth transition to {target_dataset} (1.5s ease)")
        print(f"🎯 Transition: {self.transition_start_positions} → {self.transition_target_positions}")
        
        # DON'T load new chain yet - keep current generation running
    
    def update_dataset_transition(self):
        """Update smooth dataset transition with position interpolation."""
        if not self.dataset_transitioning:
            return
        
        current_time = time.time()
        elapsed = current_time - self.dataset_transition_start_time
        progress = elapsed / self.dataset_transition_duration
        
        if progress >= 1.0:
            # Complete transition - NOW load the new chain
            target = self.dataset_transition_target
            
            print(f"✅ Transition complete - loading new chain")
            if self.load_markov_chain_from_dataset(target['dataset_info'], target['emotion']):
                print(f"✅ New dataset loaded: {target['display_name']}")
                self.dataset_var.set(target['display_name'])
                self.active_datasets[target['emotion']] = target['dataset_info']['filename']
            else:
                print(f"❌ Failed to load dataset: {target['display_name']}")
            
            self.dataset_transitioning = False
            print("🏁 Dataset transition finished")
        else:
            # Interpolate finger positions during transition
            # Use smooth easing (ease-in-out)
            t = progress
            smooth_t = t * t * (3.0 - 2.0 * t)  # Smooth step function
            
            # Override servo positions with interpolated values (fingers + arm)
            total_servos = self.num_fingers + self.num_arm_servos
            for i in range(min(total_servos, len(self.transition_start_positions), len(self.transition_target_positions))):
                start_pos = self.transition_start_positions[i]
                target_pos = self.transition_target_positions[i]
                interpolated_pos = start_pos + (target_pos - start_pos) * smooth_t
                if i < self.num_fingers:
                    self.finger_positions[i] = interpolated_pos
                else:
                    self.arm_positions[i - self.num_fingers] = interpolated_pos
            
            print(f"🔄 Transition {progress:.1%}: positions={[f'{pos:.0f}' for pos in self.finger_positions]}")
            
            # Send interpolated positions to servos
            self.send_to_hand_controller()
    
    def apply_three_phase_transition(self, overall_progress):
        """Apply direct smooth ease to closest position in new dataset."""
        # Initialize the closest target position if we haven't yet
        if not hasattr(self, 'transition_target_positions'):
            self.transition_target_positions = self.find_closest_position_in_new_dataset()
            print(f"🎯 Found closest position in new dataset: {self.transition_target_positions}")
        
        # Direct smooth ease from start to target
        eased_progress = self.ease_in_out_cubic(overall_progress)
        
        # Interpolate from starting positions to closest target position (fingers + arm)
        total_servos = self.num_fingers + self.num_arm_servos
        for i in range(min(total_servos, len(self.transition_start_positions), len(self.transition_target_positions))):
            start_pos = self.transition_start_positions[i]
            target_pos = self.transition_target_positions[i]
            interp = start_pos + (target_pos - start_pos) * eased_progress
            if i < self.num_fingers:
                self.finger_positions[i] = interp
            else:
                self.arm_positions[i - self.num_fingers] = interp
        
        if overall_progress < 0.1:  # Only print at start
            print(f"� Direct ease transition: {overall_progress:.1%} complete")
    
    def find_closest_position_in_new_dataset(self):
        """Find the position in the new dataset closest to current finger positions."""
        current_pos = self.transition_start_positions
        
        # Get the new dataset's Markov chain
        if hasattr(self, 'current_emotional_state') and self.current_emotional_state in self.markov_chains:
            chain = self.markov_chains[self.current_emotional_state]
            if chain and isinstance(chain, dict):
                # Handle different chain formats
                transitions = None
                if 'servo_transitions' in chain:
                    transitions = chain['servo_transitions']
                elif 'cursor_transitions' in chain:
                    transitions = chain['cursor_transitions']
                elif 'transitions' in chain:
                    transitions = chain['transitions']
                
                if transitions and isinstance(transitions, dict):
                    closest_distance = float('inf')
                    closest_position = current_pos.copy()

                    for state_key in transitions.keys():
                        try:
                            parsed = self._parse_single_state(state_key)
                            if parsed and len(parsed) >= 2:
                                positions = [float(v) for v in parsed]
                                n = min(len(positions), len(current_pos))
                                distance = sum((positions[i] - current_pos[i]) ** 2 for i in range(n)) ** 0.5
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_position = positions
                        except (ValueError, IndexError):
                            continue
                    
                    print(f"📏 Closest position distance: {closest_distance:.1f}° from current position")
                    return closest_position
        
        # Fallback - stay at current position if no valid dataset found
        print("⚠️ No valid dataset found - staying at current position")
        return current_pos.copy()
    
    def interpolate_dataset_behavior(self, progress):
        """Smoothly interpolate between old and new dataset behaviors during transition."""
        # Temporarily disabled - transitions not working properly
        return
        
        # During transition, modify generation speed and easing to blend behaviors
        # This creates a smooth transition feel without abrupt behavior changes
        
        # Adjust generation speed during transition (slow down slightly for smoothness)
        base_speed = getattr(self, 'generation_speed', 1.0)
        transition_speed = base_speed * (0.7 + 0.3 * progress)  # Gradually return to normal speed
        self.generation_speed = max(0.1, min(2.0, transition_speed))
        
        # Apply additional smoothing to easing during transition
        if hasattr(self, 'easing_factor'):
            base_easing = 0.05  # Default easing
            transition_easing = base_easing * (0.5 + 0.5 * progress)  # Smoother during transition
            self.easing_factor = max(0.01, min(0.2, transition_easing))
    
    def ease_in_out_cubic(self, t):
        """Smooth easing function for transitions."""
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2
    
    def initialize_markov_state_for_new_dataset(self):
        """Initialize valid Markov states for the new dataset so generation can continue."""
        if not hasattr(self, 'current_emotional_state') or self.current_emotional_state not in self.markov_chains:
            return
            
        chain = self.markov_chains[self.current_emotional_state]
        if not chain:
            return
            
        # Get transitions from the new dataset
        transitions = None
        if 'servo_transitions' in chain:
            transitions = chain['servo_transitions']
        elif 'cursor_transitions' in chain:
            transitions = chain['cursor_transitions']
        elif 'transitions' in chain:
            transitions = chain['transitions']
            
        if transitions and len(transitions) > 0:
            # Pick any valid starting state from the new dataset
            import random
            start_state_key = random.choice(list(transitions.keys()))
            self.current_markov_state = start_state_key
            
            # For second-order chains, set a valid previous state too
            if chain.get('second_order_enabled', False):
                self.prev_markov_state = start_state_key  # Simple approach
                
            print(f"🎯 Initialized Markov state for new dataset: {start_state_key}")
        else:
            print("⚠️ No valid transitions found in new dataset")

    def update_markov_state_from_current_position(self):
        """Update Markov state to match current finger positions without restarting."""
        if not hasattr(self, 'current_emotional_state') or self.current_emotional_state not in self.markov_chains:
            return
            
        chain = self.markov_chains[self.current_emotional_state]
        if not chain:
            return
            
        # Find the state in the new dataset that best matches our current position
        current_pos = self.finger_positions
        best_match_key = None
        closest_distance = float('inf')
        
        # Get transitions from the new dataset
        transitions = None
        if 'servo_transitions' in chain:
            transitions = chain['servo_transitions']
        elif 'cursor_transitions' in chain:
            transitions = chain['cursor_transitions']
        elif 'transitions' in chain:
            transitions = chain['transitions']
            
        if transitions:
            for state_key in transitions.keys():
                try:
                    parsed = self._parse_single_state(state_key)
                    if parsed and len(parsed) >= 2:
                        positions = [float(v) for v in parsed]
                        n = min(len(positions), len(current_pos))
                        distance = sum((positions[i] - current_pos[i]) ** 2 for i in range(n)) ** 0.5
                        if distance < closest_distance:
                            closest_distance = distance
                            best_match_key = state_key
                except (ValueError, IndexError):
                    continue
        
        if best_match_key:
            # Update Markov state to this matching state
            self.current_markov_state = best_match_key
            
            # For second-order chains, we need to set previous state too
            if chain.get('second_order_enabled', False):
                # Set prev_state to a valid predecessor (or same state)
                self.prev_markov_state = best_match_key  # Simple approach
                
            print(f"🎯 Updated Markov state to match transition position: {best_match_key}")
            print(f"📏 Position match distance: {closest_distance:.1f}°")
        else:
            print("⚠️ Could not find matching state in new dataset")

    def restart_markov_with_new_dataset(self):
        """Restart Markov generation smoothly with the new dataset."""
        # Find a good starting state in the new dataset
        current_positions = [self.finger_positions[0], self.finger_positions[1]]
        closest_distance = float('inf')
        best_state = None
        
        emotion = self.current_emotion
        if emotion in self.markov_chains:
            for state in self.markov_chains[emotion]:
                # Calculate distance from current position
                distance = ((state[0] - current_positions[0])**2 + 
                           (state[1] - current_positions[1])**2)**0.5
                if distance < closest_distance:
                    closest_distance = distance
                    best_state = state
            
            if best_state:
                # Set this as our target for the transition
                self.transition_target_positions = [best_state[0], best_state[1]]
                print(f"🎯 Will ease to closest position in new dataset: {best_state[:2]}")
            else:
                # Fallback to first state
                first_state = next(iter(self.markov_chains[emotion]))
                self.transition_target_positions = [first_state[0], first_state[1]]
                print(f"🎯 Will ease to first state in new dataset: {first_state[:2]}")
    
    def on_emotion_cycling_toggle(self):
        """Handle emotion cycling toggle."""
        if self.emotion_cycling_enabled.get():
            print("🎭 Emotion cycling enabled - will switch emotions every 3 minutes")
            self.emotion_cycle_status.config(text="Active", foreground="green")
            self.last_emotion_switch_time = time.time()
            # Reset to first emotion in sequence
            emotion_list = list(self.emotional_states.keys())
            self.current_emotion_index = emotion_list.index(self.current_emotional_state)
        else:
            print("🛑 Emotion cycling disabled")
            self.emotion_cycle_status.config(text="Inactive", foreground="gray")
            self.emotion_transitioning = False
    
    def on_person_detection_toggle(self):
        """Handle person detection toggle (placeholder)."""
        if self.person_detected.get():
            print("👤 Person detection simulation enabled")
        else:
            print("👤 Person detection simulation disabled")
    
    def reset_to_center(self):
        """Reset all controls to center position."""
        print("🎯 Resetting to center...")
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        for i in range(self.num_fingers):
            self.finger_positions[i] = 90.0
            self.finger_targets[i] = 90.0
        print("✅ Reset complete")
    
    def update_cycling_status_displays(self, current_time):
        """Update the cycling status displays with countdown timers."""
        # Update dataset cycling status
        if self.dataset_cycling_enabled.get():
            time_since_last = current_time - self.last_dataset_switch_time
            time_until_next = max(0, self.dataset_cycle_interval - time_since_last)
            minutes = int(time_until_next // 60)
            seconds = int(time_until_next % 60)
            self.dataset_cycle_status.config(
                text=f"Next: {minutes:02d}:{seconds:02d}", 
                foreground="green"
            )
        
        # Update emotion cycling status
        if self.emotion_cycling_enabled.get():
            time_since_last = current_time - self.last_emotion_switch_time
            time_until_next = max(0, self.emotion_cycle_interval - time_since_last)
            minutes = int(time_until_next // 60)
            seconds = int(time_until_next % 60)
            self.emotion_cycle_status.config(
                text=f"Next: {minutes:02d}:{seconds:02d}", 
                foreground="green"
            )
    
    def _get_random_cycle_interval(self):
        """Generate a random interval between 10-60 seconds for more organic behavior"""
        return random.randint(int(self.dataset_cycle_interval_min), int(self.dataset_cycle_interval_max))
    
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
        # Update UI status for current emotion
        self.update_markov_status_for_emotion(emotion_name)
        
        # Update CLEARER dataset display for new emotion
        self.update_emotion_dataset_display()
        
        # Also make sure we have the latest dataset list
        if len(self.available_datasets) == 0:
            print("🔄 No datasets loaded, refreshing...")
            self.refresh_datasets()

        # Auto-start Markov generation for the new emotional state
        # Stop any current generation first
        if self.generating:
            self.stop_markov_generation()
        
        # Start generation for the new emotional state if possible
        print(f"🤖 Auto-starting Markov generation for {emotion_name}")
        
        # Ensure startup is complete before proceeding
        if not hasattr(self, '_startup_loading_complete') or not self._startup_loading_complete:
            print(f"⏳ Completing startup initialization for {emotion_name}...")
            self._finalize_startup_display()
            self._startup_loading_complete = True
        
        # Force refresh of datasets and UI for current emotion  
        self.refresh_emotion_datasets()
        self.update_markov_status_for_emotion(emotion_name)
        
        # First build Markov chain if not available
        if self.current_emotional_state not in self.markov_chains:
            print(f"🔗 Building Markov chain for {emotion_name}...")
            self.build_markov_chain()
            
        # Check if we now have a Markov chain to use
        if self.current_emotional_state in self.markov_chains:
            print(f"✅ Markov chain available for {emotion_name} - starting generation")
            # Simulate clicking the Generate button to ensure proper UI state
            self.generate_btn.invoke()  # This properly triggers the button
        else:
            print(f"⚠️ Could not build Markov chain for {emotion_name} - no recorded movements available")
    
    def check_mood_file(self):
        """Check mood file from machine.py and auto-switch emotional state if needed."""
        if not self.mood_monitoring_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_mood_check_time < self.mood_check_interval:
            return
            
        self.last_mood_check_time = current_time
        
        try:
            if os.path.exists(self.mood_file_path):
                with open(self.mood_file_path, 'r') as f:
                    mood_data = json.load(f)
                
                emotion_state = mood_data.get('emotion_state', '')
                mood_value = mood_data.get('mood_value', 0.0)
                timestamp = mood_data.get('timestamp', 0)
                
                # Only switch if the state has changed
                if emotion_state != self.last_mood_state and emotion_state in self.emotional_states:
                    print(f"🎭 Mood update: {emotion_state} (mood: {mood_value:.2f})")
                    self.switch_emotional_state(emotion_state)
                    self.last_mood_state = emotion_state
                    
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            # Silently handle file reading errors - machine.py might not be running
            pass
    
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
                sample_count = dataset_info.get('sample_count', 0)  # Use sample_count instead of points
                duration = dataset_info.get('duration', 0)
                custom_name = dataset_info.get('custom_name', '')
                timestamp = dataset_info.get('timestamp', 'unknown')
                
                # Format timestamp to be more readable
                try:
                    if len(timestamp) >= 8:
                        date_part = timestamp[:8]
                        time_part = timestamp[9:15] if len(timestamp) > 9 else "000000"
                        formatted_date = f"{date_part[4:6]}/{date_part[6:8]}"
                        formatted_time = f"{time_part[:2]}:{time_part[2:4]}"
                        readable_time = f"{formatted_date} {formatted_time}"
                    else:
                        readable_time = timestamp
                except:
                    readable_time = timestamp
                
                # Create display name that matches the format expected by other methods
                if custom_name:
                    display_name = f"{custom_name} ({readable_time}) - {sample_count:,} samples, {duration:.1f}s"
                else:
                    display_name = f"Recording {readable_time} - {sample_count:,} samples, {duration:.1f}s"
                    
                dataset_options.append(display_name)
        
        if not dataset_options:
            dataset_options = ["No datasets available - record some movements first!"]
        
        self.dataset_dropdown['values'] = dataset_options
        
        # Set current selection - auto-select first available dataset
        if dataset_options and not dataset_options[0].startswith("No datasets"):
            # Auto-select the first (most recent) dataset
            self.dataset_var.set(dataset_options[0])
            # Also set it as the active dataset for this emotion
            if emotion_datasets:
                self.active_datasets[emotion] = emotion_datasets[0]['filename']
                # Load the Markov chain immediately
                if self.load_markov_chain_from_dataset(emotion_datasets[0], emotion):
                    print(f"🔗 Auto-loaded Markov chain for {emotion} from {emotion_datasets[0]['display_name']}")
        elif dataset_options:
            self.dataset_var.set(dataset_options[0])
    
    def refresh_serial_ports(self):
        """Auto-detect available serial ports (works on Windows, Linux, macOS)."""
        try:
            # Get all available serial ports
            ports = list_ports.comports()
            port_list = [port.device for port in ports]
            
            # Store available ports
            self.available_ports = port_list
            
            # Create dropdown options
            if port_list:
                options = ["Auto"] + port_list
                print(f"🔍 Found {len(port_list)} serial ports: {port_list}")
            else:
                options = ["No ports found"]
                print("⚠️ No serial ports detected")
            
            # Update combobox
            if hasattr(self, 'port_combo'):
                self.port_combo['values'] = options
                
                # Keep current selection if still valid, otherwise default to Auto
                current = self.selected_port.get()
                if current in options:
                    self.port_combo.set(current)
                else:
                    self.port_combo.set("Auto" if port_list else "No ports found")
            
        except Exception as e:
            print(f"❌ Error detecting serial ports: {e}")
            if hasattr(self, 'port_combo'):
                self.port_combo['values'] = ["Error detecting ports"]
                self.port_combo.set("Error detecting ports")
    
    def get_selected_port(self):
        """Get the currently selected port, with auto-detection logic."""
        selected = self.selected_port.get()
        
        if selected == "Auto":
            # Auto-detect: prefer udev symlink, otherwise first available port
            if self.available_ports:
                for preferred in ["/dev/arduino_lefthand", "/dev/arduino_hand"]:
                    if preferred in self.available_ports:
                        return preferred
                return self.available_ports[0]
            else:
                return "/dev/arduino_lefthand"  # Fallback
        elif selected in ["No ports found", "Error detecting ports"]:
            return "/dev/arduino_lefthand"  # Fallback
        else:
            return selected

    def toggle_connection(self):
        """Toggle hand controller connection with auto-detected port."""
        if not HAND_CONTROLLER_AVAILABLE:
            self.status_label.config(text="❌ Controller unavailable")
            print("⚠️ Hand controller not available - simulation mode")
            return
        
        if not self.connected:
            try:
                # Get the selected port (with auto-detection)
                port = self.get_selected_port()
                print(f"🔌 Attempting connection to {port}...")
                
                # Initialize HandExpressionController with selected port
                self.hand_controller = HandExpressionController(
                    port=port,
                    baudrate=9600,
                    clean_output=True
                )
                
                # Check if connection was successful
                if self.hand_controller.serial_connection:
                    # Enable manual override to ensure our commands are processed
                    self.hand_controller.enable_manual_override()
                    self.connected = True
                    self.status_label.config(text=f"✅ Connected ({port})")
                    self.connect_btn.config(text="Disconnect")
                    print(f"✅ Connected to hand controller on {port}")
                    print("🎮 Manual override enabled - ready for cursor control")
                else:
                    self.status_label.config(text="❌ Connection failed")
                    print(f"❌ Failed to connect to {port}")
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
    
    def auto_connect_arduino(self):
        """Automatically detect and connect to Arduino."""
        if not HAND_CONTROLLER_AVAILABLE:
            print("⚠️ Hand controller not available - running in simulation mode")
            return
            
        if self.connected:
            print("✅ Already connected to Arduino")
            return
            
        print("🔍 Scanning for Arduino ports...")
        
        # Try to find Arduino automatically
        arduino_ports = []
        try:
            ports = list_ports.comports()
            for port in ports:
                # Look for common Arduino identifiers
                if any(keyword in (port.description or "").lower() for keyword in 
                       ['arduino', 'ch340', 'cp210', 'ftdi', 'serial']):
                    arduino_ports.append(port.device)
                    print(f"🎯 Found potential Arduino: {port.device} - {port.description}")
        except Exception as e:
            print(f"⚠️ Error scanning ports: {e}")
            
        # Try connecting to found ports
        for port in arduino_ports:
            try:
                print(f"🔌 Trying to connect to {port}...")
                
                self.hand_controller = HandExpressionController(
                    port=port,
                    baudrate=9600,
                    clean_output=True
                )
                
                if self.hand_controller.serial_connection:
                    self.hand_controller.enable_manual_override()
                    self.connected = True
                    self.status_label.config(text=f"✅ Auto-connected ({port})")
                    self.connect_btn.config(text="Disconnect")
                    print(f"✅ Auto-connected to Arduino on {port}")
                    return
                else:
                    if self.hand_controller:
                        if hasattr(self.hand_controller, 'cleanup'):
                            self.hand_controller.cleanup()
                        self.hand_controller = None
                        
            except Exception as e:
                print(f"❌ Failed to connect to {port}: {e}")
                if self.hand_controller:
                    if hasattr(self.hand_controller, 'cleanup'):
                        self.hand_controller.cleanup()
                    self.hand_controller = None
                    
        print("❌ No Arduino found - running in simulation mode")
        self.status_label.config(text="❌ No Arduino detected")
    
    def on_canvas_configure(self, event):
        """Prevent canvas from being resized and maintain fixed dimensions."""
        # Force canvas to maintain exact dimensions
        if event.width != self.canvas_width or event.height != self.canvas_height:
            self.canvas.config(width=self.canvas_width, height=self.canvas_height)
            print(f"🔒 Canvas size locked: {self.canvas_width}x{self.canvas_height}")
    
    def on_mouse_move(self, event):
        """Handle mouse movement in canvas - ABSOLUTELY FIXED coordinates."""
        # Use actual canvas dimensions for responsive layout
        canvas_width = self.canvas.winfo_width() or self.canvas_width
        canvas_height = self.canvas.winfo_height() or self.canvas_height
        
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
            
            # Calculate time difference from previous sample for second-order Markov
            dt = current_time - self.previous_time if hasattr(self, 'previous_time') else 0.0
            self.previous_time = current_time
            
            # MEMORY MANAGEMENT: Limit recording buffer size to prevent performance degradation
            self.recording_point_counter += 1
            
            # Only store if we haven't exceeded the maximum points
            if len(self.recorded_movements.get(self.current_emotional_state, [])) < self.max_recording_points:
                # Store complete movement data - fingers + arm servos combined
                movement_point = {
                    'time': current_time,
                    'relative_time': relative_time,
                    'dt': dt,
                    'x': self.mouse_x,
                    'y': self.mouse_y,
                    'finger_positions': self.finger_positions.copy(),
                    'arm_positions': self.arm_positions.copy(),
                    'servo_positions': self.finger_positions.copy() + self.arm_positions.copy(),
                    'movement_phase': 'MEDIUM',
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
            return "break"  # Prevent event propagation
        
        # Don't interfere with generation
        if self.generating:
            print("🧠 Spacebar ignored - Markov generation in progress")
            return "break"  # Prevent event propagation
        
        # Don't interfere with playback
        if self.playing_back:
            print("▶️ Spacebar ignored - playback in progress")
            return "break"  # Prevent event propagation
            
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
        
        return "break"  # Always prevent spacebar from propagating to other widgets
    
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
                    'servo_positions': self.finger_positions.copy(),  # RESTORED: Full data complexity
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
        # Safe defaults if freeze UI was removed
        if not hasattr(self, 'min_freeze_duration'):
            self.min_freeze_duration = type('', (), {'get': lambda self: 2.0})()
            self.max_freeze_duration = type('', (), {'get': lambda self: 6.0})()
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
        self.arm_positions = [90.0] * self.num_arm_servos
        self.arm_targets = [90.0] * self.num_arm_servos

        # Update arm sliders to match
        if hasattr(self, 'arm_slider_vars'):
            for sv in self.arm_slider_vars:
                sv.set(90.0)

        # End any freeze state
        if self.is_frozen:
            self.end_freeze()

        print("Reset to center position")

    # === CONTROL MODE METHODS ===

    def _on_mode_change(self):
        """Toggle between cursor wave and single servo mode."""
        if self.control_mode.get() == "manual":
            self.wave_workspace.pack_forget()
            self.single_servo_frame.pack(fill=tk.X)
            self._update_single_servo_bar()
            self._update_small_bars()
        else:
            self.single_servo_frame.pack_forget()
            self.wave_workspace.pack(fill=tk.X)

    def _on_servo_select(self, event=None):
        """Handle servo selection from combobox."""
        idx = self.servo_combo.current()
        self.selected_servo_idx.set(idx)
        self._update_single_servo_bar()

    def _select_servo_by_bar(self, idx):
        """Select a servo by clicking its small bar."""
        self.servo_combo.current(idx)
        self.selected_servo_idx.set(idx)
        if self.control_mode.get() != "manual":
            self.control_mode.set("manual")
            self._on_mode_change()
        self._update_single_servo_bar()

    def _on_single_servo_drag(self, event):
        """Handle drag on the single servo large bar."""
        if self.control_mode.get() != "manual":
            return
        h = self.single_servo_canvas.winfo_height()
        margin = 15
        bar_h = h - 2 * margin
        ratio = max(0.0, min(1.0, (h - margin - event.y) / bar_h)) if bar_h > 0 else 0.5
        angle = ratio * 180.0
        idx = self.selected_servo_idx.get()
        if idx < self.num_fingers:
            self.finger_positions[idx] = angle
        else:
            self.arm_positions[idx - self.num_fingers] = angle
        self._update_single_servo_bar()
        self.send_to_hand_controller()

    def _update_single_servo_bar(self):
        """Redraw the large single servo bar with degree markers."""
        if not hasattr(self, 'single_servo_canvas'):
            return
        c = self.single_servo_canvas
        c.delete("all")
        w = c.winfo_width() or 100
        h = c.winfo_height() or 350
        idx = self.selected_servo_idx.get()
        if idx < self.num_fingers:
            angle = self.finger_positions[idx]
        else:
            angle = self.arm_positions[idx - self.num_fingers]

        margin = 15
        bar_x0, bar_x1 = margin, w - margin
        bar_y0, bar_y1 = margin, h - margin
        bar_h = bar_y1 - bar_y0

        # Background
        c.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, fill='#F0F0F0', outline='#AAA', width=1)

        # Fill from bottom
        fill_h = (angle / 180.0) * bar_h
        c.create_rectangle(bar_x0, bar_y1 - fill_h, bar_x1, bar_y1, fill='#7CB3D0', outline='')

        # Degree markers and gridlines
        for deg in range(0, 181, 15):
            y = bar_y1 - (deg / 180.0) * bar_h
            is_major = deg % 45 == 0
            if is_major:
                c.create_line(bar_x0 - 4, y, bar_x1, y, fill='#999', dash=(2, 2))
                c.create_text(bar_x0 - 6, y, text=str(deg), font=("Courier", 8),
                             anchor='e', fill='#666')
            else:
                c.create_line(bar_x0, y, bar_x0 + 4, y, fill='#CCC')

        # Current position line
        pos_y = bar_y1 - (angle / 180.0) * bar_h
        c.create_line(bar_x0 - 2, pos_y, bar_x1 + 2, pos_y, fill='red', width=2)

        # Name and value
        all_names = self.finger_names + self.arm_names
        name = all_names[idx] if idx < len(all_names) else f"Servo {idx}"
        if hasattr(self, 'single_servo_name_label'):
            self.single_servo_name_label.config(text=name)
        self.single_servo_label.config(text=f"{angle:6.1f}")

    def _update_small_bars(self):
        """Redraw the small servo bar indicators with fixed-width value labels."""
        if not hasattr(self, 'small_bar_canvases'):
            return
        all_positions = list(self.finger_positions) + list(self.arm_positions)
        selected = self.selected_servo_idx.get()
        for i, c in enumerate(self.small_bar_canvases):
            c.delete("all")
            w = c.winfo_width() or 28
            h = c.winfo_height() or 180
            if i < len(all_positions):
                angle = all_positions[i]
                fill_h = (angle / 180.0) * h
                # Highlight selected servo
                color = '#5B9BD5' if i == selected else '#B8D4E3'
                border = '#2563EB' if i == selected else '#D0D0D0'
                c.create_rectangle(1, h - fill_h, w - 1, h, fill=color, outline='')
                c.create_rectangle(1, 1, w - 1, h - 1, fill='', outline=border, width=2 if i == selected else 1)
                # 90-degree center line
                mid_y = h * 0.5
                c.create_line(0, mid_y, w, mid_y, fill='#AAA', dash=(1, 2))
                if i < len(self.small_bar_labels):
                    self.small_bar_labels[i].config(text=f"{angle:3.0f}")

    # === LAYER SYSTEM METHODS ===

    def _start_layer_recording(self):
        """Begin recording a new layer. Called from start_recording."""
        self.current_layer_data = []
        self.layer_recorded_servos = {'fingers': set(), 'arm': set()}
        # Start playing back existing layers while recording new one
        if self.recorded_layers:
            self.layer_playback_active = True
            self.layer_playback_start = time.time()

    def _record_layer_frame(self):
        """Capture one frame for the current layer. Called from recording timer."""
        current_time = time.time()
        elapsed = current_time - self.record_start_time

        # Build per-servo frame with None for unrecorded servos
        finger_pos = [None] * self.num_fingers
        arm_pos = [None] * self.num_arm_servos

        if self.control_mode.get() == "cursor":
            # Wave mode records all fingers
            for i in range(self.num_fingers):
                finger_pos[i] = self.finger_positions[i]
                self.layer_recorded_servos['fingers'].add(i)
        else:
            # Single servo mode records only selected servo
            idx = self.selected_servo_idx.get()
            if idx < self.num_fingers:
                finger_pos[idx] = self.finger_positions[idx]
                self.layer_recorded_servos['fingers'].add(idx)
            else:
                arm_idx = idx - self.num_fingers
                arm_pos[arm_idx] = self.arm_positions[arm_idx]
                self.layer_recorded_servos['arm'].add(arm_idx)

        frame = {
            'time': elapsed,
            'finger_positions': finger_pos,
            'arm_positions': arm_pos,
            'control_mode': self.control_mode.get(),
        }
        self.current_layer_data.append(frame)

    def _finish_layer_recording(self):
        """Finalize the current layer and add to layer list."""
        if not self.current_layer_data:
            return
        duration = self.current_layer_data[-1]['time'] if self.current_layer_data else 0
        layer = {
            'name': f"Layer {len(self.recorded_layers) + 1}",
            'duration': duration,
            'data': self.current_layer_data,
            'timestamp': time.strftime("%H:%M:%S"),
            'control_mode': self.control_mode.get(),
            'recorded_servos': {
                'fingers': self.layer_recorded_servos['fingers'].copy(),
                'arm': self.layer_recorded_servos['arm'].copy(),
            },
        }
        self.recorded_layers.append(layer)
        # Update loop duration to longest layer
        self.layer_max_duration = max(l['duration'] for l in self.recorded_layers)
        self.layer_playback_active = False
        self._update_layer_list()
        self._update_timeline()
        print(f"Layer {len(self.recorded_layers)} saved: {len(self.current_layer_data)} frames, "
              f"{duration:.1f}s, mode={layer['control_mode']}, "
              f"servos: fingers={layer['recorded_servos']['fingers']}, arm={layer['recorded_servos']['arm']}")

    def _apply_layers_at_time(self, t):
        """Apply all recorded layers at time t using last-writer-wins per servo."""
        if not self.recorded_layers:
            return
        # Build ownership map: later layers override earlier ones per servo
        finger_owner = [None] * self.num_fingers
        arm_owner = [None] * self.num_arm_servos
        for layer_idx, layer in enumerate(self.recorded_layers):
            for fi in layer['recorded_servos']['fingers']:
                if fi < self.num_fingers:
                    finger_owner[fi] = layer_idx
            for ai in layer['recorded_servos']['arm']:
                if ai < self.num_arm_servos:
                    arm_owner[ai] = layer_idx

        # Interpolate and apply from owning layer only
        for layer_idx, layer in enumerate(self.recorded_layers):
            data = layer['data']
            if not data:
                continue
            # Normalize time to loop within layer duration
            layer_dur = layer['duration']
            if layer_dur <= 0:
                continue
            norm_t = t % layer_dur

            # Find surrounding frames for interpolation
            prev_frame = data[0]
            next_frame = data[-1]
            for j in range(len(data)):
                if data[j]['time'] >= norm_t:
                    next_frame = data[j]
                    prev_frame = data[max(0, j - 1)]
                    break

            # Interpolation factor
            dt = next_frame['time'] - prev_frame['time']
            frac = (norm_t - prev_frame['time']) / dt if dt > 0 else 0.0
            frac = max(0.0, min(1.0, frac))

            # Apply finger positions from owning layer
            for fi in range(self.num_fingers):
                if finger_owner[fi] == layer_idx:
                    prev_val = prev_frame['finger_positions'][fi]
                    next_val = next_frame['finger_positions'][fi]
                    if prev_val is not None and next_val is not None:
                        self.finger_positions[fi] = prev_val + (next_val - prev_val) * frac

            # Apply arm positions from owning layer
            for ai in range(self.num_arm_servos):
                if arm_owner[ai] == layer_idx:
                    prev_val = prev_frame['arm_positions'][ai]
                    next_val = next_frame['arm_positions'][ai]
                    if prev_val is not None and next_val is not None:
                        self.arm_positions[ai] = prev_val + (next_val - prev_val) * frac

    def _merge_layers_and_build_markov(self):
        """Merge all layers into recorded_movements and build Markov chain."""
        if not self.recorded_layers:
            print("No layers to merge")
            return
        emotion = self.current_emotional_state
        # Sample the merged layers at 40Hz
        duration = self.layer_max_duration
        if duration <= 0:
            return
        sample_rate = 40.0
        num_samples = int(duration * sample_rate)
        merged = []
        prev_time = 0
        for s in range(num_samples):
            t = s / sample_rate
            self._apply_layers_at_time(t)
            dt = t - prev_time
            prev_time = t
            merged.append({
                'time': t,
                'dt': dt,
                'servo_positions': list(self.finger_positions) + list(self.arm_positions),
                'finger_positions': list(self.finger_positions),
                'arm_positions': list(self.arm_positions),
                'movement_phase': 'MEDIUM',
            })
        self.recorded_movements[emotion] = merged
        print(f"Merged {len(self.recorded_layers)} layers into {len(merged)} samples for {emotion}")
        self.build_markov_chain()

    def _update_layer_list(self):
        """Refresh the layer listbox."""
        if not hasattr(self, 'layer_listbox'):
            return
        self.layer_listbox.delete(0, tk.END)
        for i, layer in enumerate(self.recorded_layers):
            servos = list(layer['recorded_servos']['fingers']) + \
                     [f"A{a}" for a in layer['recorded_servos']['arm']]
            servo_str = ",".join(str(s) for s in servos)
            text = f"{i+1}. {layer['name']} ({layer['duration']:.1f}s, {layer['control_mode']}, [{servo_str}])"
            self.layer_listbox.insert(tk.END, text)
        total = len(self.recorded_layers)
        self.layer_info_label.config(
            text=f"{total} layer{'s' if total != 1 else ''}" if total > 0 else "No layers recorded")

    def _update_timeline(self):
        """Draw the timeline canvas with layer bars and playhead."""
        if not hasattr(self, 'timeline_canvas'):
            return
        c = self.timeline_canvas
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height() or 30

        if not self.recorded_layers:
            c.create_rectangle(0, 0, w, h, fill='#1a1a2e', outline='')
            c.create_text(w // 2, h // 2, text="No layers", fill='#555', font=("Arial", 8))
            return

        max_dur = self.layer_max_duration
        if max_dur <= 0:
            return

        n = len(self.recorded_layers)
        layer_h = max(4, h // max(n, 1))
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

        for i, layer in enumerate(self.recorded_layers):
            y = i * layer_h
            lw = (layer['duration'] / max_dur) * w
            color = colors[i % len(colors)]
            c.create_rectangle(0, y, lw, y + layer_h - 1, fill=color, outline='')
            c.create_text(4, y + layer_h // 2, text=layer['name'], anchor='w',
                         fill='white', font=("Arial", 7))

        # Playhead
        if self.layer_playback_active and max_dur > 0:
            elapsed = (time.time() - self.layer_playback_start) % max_dur
            px = (elapsed / max_dur) * w
            c.create_line(px, 0, px, h, fill='red', width=2)

        # Duration label
        c.create_text(w - 3, h - 3, text=f"{max_dur:.1f}s", anchor='se',
                     fill='#888', font=("Arial", 7))

    def _delete_selected_layer(self):
        """Delete the selected layer from the list."""
        sel = self.layer_listbox.curselection()
        if sel:
            idx = sel[0]
            removed = self.recorded_layers.pop(idx)
            print(f"Deleted {removed['name']}")
            if self.recorded_layers:
                self.layer_max_duration = max(l['duration'] for l in self.recorded_layers)
            else:
                self.layer_max_duration = 0
            self._update_layer_list()
            self._update_timeline()

    def _clear_all_layers(self):
        """Clear all recorded layers."""
        self.recorded_layers.clear()
        self.layer_max_duration = 0
        self.layer_playback_active = False
        self._update_layer_list()
        self._update_timeline()
        print("All layers cleared")

    def _play_layers_direct(self):
        """Toggle direct layer playback (non-Markov — replays recorded movement exactly)."""
        if self.layer_playback_active:
            self.layer_playback_active = False
            self.playback_btn.config(text="Play")
            print("Layer playback stopped")
            return
        if not self.recorded_layers:
            print("No layers to play")
            return
        # Stop other modes
        if self.generating:
            self.stop_markov_generation()
        if self.recording:
            self.stop_recording()
        self.layer_playback_active = True
        self.layer_playback_start = time.time()
        self.playback_btn.config(text="Stop Play")
        print(f"Playing {len(self.recorded_layers)} layers ({self.layer_max_duration:.1f}s loop)")
    
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
        
        # Direct layer playback (during recording overdub OR standalone play)
        if self.layer_playback_active and self.recorded_layers:
            elapsed = time.time() - self.layer_playback_start
            self._apply_layers_at_time(elapsed)
            self.send_to_hand_controller()

        # Update canvas visualization (always show current state)
        self.update_canvas()

        # Update small servo bars and timeline (throttled to ~10Hz to avoid flicker)
        if not hasattr(self, '_ui_update_counter'):
            self._ui_update_counter = 0
        self._ui_update_counter += 1
        if self._ui_update_counter % 3 == 0:
            self._update_small_bars()
            self._update_timeline()
            if self.control_mode.get() == "manual":
                self._update_single_servo_bar()

        # Only calculate cursor targets if NOT in any automated playback mode
        if not self.generating and not self.playing_back and not (self.layer_playback_active and not self.recording):
            if self.control_mode.get() == "cursor":
                # Wave control - calculate finger targets from cursor position
                self.calculate_finger_targets()
                self.finger_positions = self.finger_targets.copy()
            # In manual mode, finger_positions are set by _on_single_servo_drag
        # When generating, finger_positions are set by update_markov_generation()
        # When playing back, finger_positions are set by update_playback()
        
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
        
        # Update cycling behavior (dataset and emotion cycling)
        self.update_cycling_behavior()
        self.update_emotion_transition()
        
        # Check for mood updates from machine.py
        self.check_mood_file()
        
        # Periodic timer cleanup (every 5 minutes to prevent accumulation)
        if hasattr(self, 'last_timer_cleanup'):
            if time.time() - self.last_timer_cleanup > 300:  # 5 minutes
                self.cleanup_orphaned_timers()
                self.last_timer_cleanup = time.time()
        else:
            self.last_timer_cleanup = time.time()
        
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
        
        # Use actual canvas dimensions for responsive layout
        canvas_width = self.canvas.winfo_width() or self.canvas_width
        canvas_height = self.canvas.winfo_height() or self.canvas_height
        
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
            
            # MEMORY CLEANUP: Limit history to prevent memory leaks
            if len(self.canvas_objects) > 100:  # Arbitrary limit
                oldest_keys = list(self.canvas_objects.keys())[:50]
                for key in oldest_keys:
                    if key in self.canvas_objects:
                        del self.canvas_objects[key]
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
        
        # Draw ALL servo indicators (fingers + arm)
        all_positions = list(self.finger_positions) + list(self.arm_positions)
        all_names = self.finger_names + self.arm_names
        total_servos = len(all_positions)

        for i in range(total_servos):
            servo_x = condensed_area_start + ((i + 0.5) / total_servos) * condensed_area_width

            is_finger = i < self.num_fingers
            is_locked = self.finger_locks[i] if is_finger and i < len(self.finger_locks) else False

            # Colors: fingers green, arm blue, locked pink
            if is_locked:
                bar_color = self.colors['bg_accent']
                outline_color = self.colors['button_bg']
                position_color = self.colors['button_hover']
            elif is_finger:
                bar_color = self.colors['bg_frame']
                outline_color = self.colors['text_main']
                position_color = self.colors['success']
            else:
                bar_color = '#E8F0FE'
                outline_color = '#5B7DB1'
                position_color = '#7CB3D0'

            # Bar outline
            self.canvas.create_rectangle(
                servo_x - bar_width // 2, finger_y_top,
                servo_x + bar_width // 2, finger_y_base,
                fill=bar_color, outline=outline_color, width=2)

            # Position fill
            angle = all_positions[i]
            if self.reverse_vertical.get() and is_finger:
                pos_ratio = (180.0 - angle) / 180.0
            else:
                pos_ratio = angle / 180.0
            pos_height = int(pos_ratio * bar_height)

            self.canvas.create_rectangle(
                servo_x - bar_width // 2, finger_y_base - pos_height,
                servo_x + bar_width // 2, finger_y_base,
                fill=position_color, outline=self.colors['text_accent'], width=1)

            # Target line (fingers only — arm servos don't have wave targets)
            if is_finger and i < len(self.finger_targets):
                if self.reverse_vertical.get():
                    target_ratio = (180.0 - self.finger_targets[i]) / 180.0
                else:
                    target_ratio = self.finger_targets[i] / 180.0
                target_y = finger_y_base - int(target_ratio * bar_height)
                self.canvas.create_line(
                    servo_x - bar_width // 2 - 3, target_y,
                    servo_x + bar_width // 2 + 3, target_y,
                    fill="red", width=2)

            # Label
            label = all_names[i][:3] if i < len(all_names) else f"S{i}"
            self.canvas.create_text(servo_x, finger_y_base + 10, text=label,
                                    fill=self.colors['text_main'], font=("Arial", 8, "bold"))
        
        # Essential text only - avoid expensive string formatting
        self._draw_essential_text(canvas_width, canvas_height)
    
    def _draw_essential_text(self, canvas_width, canvas_height):
        """Draw only essential text information to avoid performance issues."""
        # Mode + emotion header
        mode_label = "Wave" if self.control_mode.get() == "cursor" else "Single Servo"
        rev = " [REV]" if self.reverse_vertical.get() else ""
        mode_text = f"{mode_label}{rev}"
        self.canvas.create_text(10, 10, text=mode_text, fill=self.colors['text_main'], anchor="nw",
                              font=self.fonts['canvas'])

        emotion_text = f"{self.current_emotional_state.replace('_', ' ').title()}"
        self.canvas.create_text(10, 25, text=emotion_text, fill=self.colors['text_accent'], anchor="nw",
                              font=self.fonts['small'])

        # Show all servo angles top-right
        all_pos = list(self.finger_positions) + list(self.arm_positions)
        all_names = self.finger_names + self.arm_names
        angle_text = "  ".join(f"{all_names[i][:3]}:{all_pos[i]:.0f}" for i in range(len(all_pos)))
        self.canvas.create_text(canvas_width - 5, 10, text=angle_text,
                              fill='#888', anchor="ne", font=("Arial", 7))
        
        # Status indicators (only when relevant - reduced text objects)
        status_y = 40
        if self.text_field_has_focus:
            self.canvas.create_text(10, status_y, text="📝 TEXT INPUT", fill=self.colors['error'], anchor="nw",
                                  font=("Arial", 9, "bold"))
        elif self.pressed_keys:
            keys_text = f"Keys: {','.join(sorted(self.pressed_keys)).upper()}"
            self.canvas.create_text(10, status_y, text=keys_text, fill=self.colors['warning'], anchor="nw",
                                  font=("Arial", 9))
        elif self.is_frozen:
            self.canvas.create_text(10, status_y, text="❄️ FROZEN", fill=self.colors['text_accent'], anchor="nw",
                                  font=("Arial", 9, "bold"))
        elif self.recording:
            self.canvas.create_text(10, status_y, text="🎬 RECORDING", fill=self.colors['error'], anchor="nw",
                                  font=("Arial", 9, "bold"))
        elif self.playing_back:
            self.canvas.create_text(10, status_y, text="▶️ PLAYBACK", fill=self.colors['success'], anchor="nw",
                                  font=("Arial", 9))
        elif self.generating:
            self.canvas.create_text(10, status_y, text="🧠 GENERATING", fill=self.colors['text_accent'], anchor="nw",
                                  font=("Arial", 9))
        
        # Servo range (bottom right, simplified)
        servo_range = self.servo_range.get()
        range_text = f"Range: ±{servo_range:.0f}°"
        self.canvas.create_text(canvas_width-10, canvas_height-10, text=range_text, 
                              fill=self.colors['text_main'], anchor="se", font=("Arial", 9))
    
    def send_to_hand_controller(self):
        """Send positions to hand controller - FIXED INTERFACE."""
        if not self.connected or not self.hand_controller:
            return
        
        current_time = time.time()
        if current_time - self.last_send_time < self.send_interval:
            return
        
        # Check if positions changed significantly (fingers OR arm)
        changed = False
        # Check finger positions (with safety check for array length)
        last_finger_positions = getattr(self, 'last_sent_finger_positions', [90]*self.num_fingers)
        if len(last_finger_positions) != self.num_fingers:
            last_finger_positions = [90]*self.num_fingers  # Reset if size mismatch
        
        for i in range(self.num_fingers):
            if abs(self.finger_positions[i] - last_finger_positions[i]) > self.position_threshold:
                changed = True
                break
        # Check arm positions
        if not changed:
            for i in range(self.num_arm_servos):
                if abs(self.arm_positions[i] - getattr(self, 'last_sent_arm_positions', [90]*self.num_arm_servos)[i]) > self.position_threshold:
                    changed = True
                    break
        
        if changed or current_time - getattr(self, 'last_any_send_time', 0) > 1.0:
            try:
                # MAP SERVO RANGE: Use Arduino's full 0-180° range (matches updated Arduino code)
                # Arduino now accepts 0-180° (full range), map based on actual servo_range used
                arduino_min = 0     # Arduino now supports full range
                arduino_max = 180   # Arduino now supports full range
                arduino_center = (arduino_min + arduino_max) / 2  # 90°
                arduino_range = arduino_max - arduino_min  # 180°
                
                # Get the current servo range setting to determine actual range used
                current_servo_range = self.servo_range.get()
                default_pos = self.default_position.get()
                
                positions = []
                # Process finger positions (5 fingers)
                for i in range(self.num_fingers):
                    # Calculate offset from default position
                    offset_from_default = self.finger_positions[i] - default_pos
                    
                    # Scale based on actual servo_range setting
                    # This makes the servo_range slider actually control the movement range
                    normalized_offset = offset_from_default / current_servo_range
                    
                    # Map to Arduino's available range (now full 0-180°)
                    arduino_offset = normalized_offset * (arduino_range / 2.0)
                    arduino_position = arduino_center + arduino_offset
                    
                    # Clamp to Arduino's actual limits (now 0-180°)
                    arduino_position = max(arduino_min, min(arduino_max, arduino_position))
                    positions.append(int(arduino_position))
                
                # Process arm positions (3 arm servos) - same mapping as fingers
                for i in range(self.num_arm_servos):
                    # Calculate offset from default position
                    offset_from_default = self.arm_positions[i] - default_pos
                    
                    # Scale based on actual servo_range setting
                    normalized_offset = offset_from_default / current_servo_range
                    
                    # Map to Arduino's available range (now full 0-180°)
                    arduino_offset = normalized_offset * (arduino_range / 2.0)
                    arduino_position = arduino_center + arduino_offset
                    
                    # Clamp to Arduino's actual limits (now 0-180°)
                    arduino_position = max(arduino_min, min(arduino_max, arduino_position))
                    positions.append(int(arduino_position))
                
                # Debug output to see what we're actually sending
                if hasattr(self, 'send_count'):
                    self.send_count += 1
                else:
                    self.send_count = 1
                
                if self.send_count < 5 or self.send_count % 20 == 0:
                    print(f"📤 Sending {self.send_count}: arduino_positions={positions[:5]} (fingers) + {positions[5:]} (arm) from finger_positions={[f'{p:.1f}' for p in self.finger_positions]} arm_positions={[f'{p:.1f}' for p in self.arm_positions]} (mapped to 0-180°)")
                
                # Send all 8 servo positions (5 fingers + 3 arm servos)
                self.hand_controller.set_hand_positions(positions)
                
                # Track both finger and arm positions separately
                self.last_sent_finger_positions = self.finger_positions.copy()
                self.last_sent_arm_positions = self.arm_positions.copy()
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
                
                # Standard cleanup for 20s recordings - start at 80% capacity
                if buffer_length > self.max_recording_points * 0.8:  # Start cleanup at 80% capacity
                    # Keep most recent 60% of data
                    keep_count = int(self.max_recording_points * 0.6)
                    self.recorded_movements[emotion_name] = current_buffer[-keep_count:]
                    cleaned_count = buffer_length - keep_count
                    print(f"🧹 Cleaned {cleaned_count} old recording points for {emotion_name} (kept {keep_count})")
    
    def clear_all_data(self):
        """Clear ALL data - memory, files, and dropdowns. Start completely fresh."""
        # Ask for confirmation since this is destructive
        import tkinter.messagebox as msgbox
        
        result = msgbox.askyesno("Clear All Data", 
                               "This will permanently delete:\n\n"
                               "• All recorded movements in memory\n"
                               "• All saved dataset files\n" 
                               "• All Markov chains\n\n"
                               "You'll start completely fresh.\n\n"
                               "Continue?")
        
        if not result:
            return
            
        # 1. Clear all in-memory data
        self.recorded_movements.clear()
        self.recorded_positions.clear()
        self.markov_chains.clear()
        
        # 2. Clear dataset tracking
        self.available_datasets.clear()
        self.dataset_info.clear()
        self.active_datasets.clear()
        
        # 3. Delete all saved files in movement_recordings directory
        import os
        import glob
        
        if os.path.exists("movement_recordings"):
            files_deleted = 0
            for filepath in glob.glob("movement_recordings/*.json"):
                try:
                    os.remove(filepath)
                    files_deleted += 1
                    print(f"🗑️ Deleted: {filepath}")
                except Exception as e:
                    print(f"⚠️ Could not delete {filepath}: {e}")
            
            print(f"🗑️ Deleted {files_deleted} dataset files")
        
        # 4. Refresh the dropdown to show empty state
        self.refresh_datasets()
        
        # 5. Update UI status
        if hasattr(self, 'record_status'):
            self.record_status.config(text="All data cleared - ready to record!", foreground="green")
            self.root.after(3000, lambda: self.record_status.config(text="Ready to record 20s segments (Spacebar)", foreground="gray"))
        
        print("🗑️ ALL DATA CLEARED - Starting completely fresh!")
        print("💡 TIP: Record new movements for each emotion to build fresh, compatible datasets")
    
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
        self.previous_time = self.record_start_time
        self.recorded_movements[self.current_emotional_state] = []
        self.recorded_positions = []
        self.recording_point_counter = 0

        # Initialize layer recording
        self._start_layer_recording()
        
        # UPDATE ALL UI BUTTONS
        self.record_btn.config(text="⏹️ Stop Recording")
        self.playback_btn.config(text="▶️ Play Back")        # Reset playback button
        self.generate_btn.config(text="🧠 Generate (Markov)") # Reset generate button
        self.record_status.config(text="🎬 RECORDING... (Spacebar to stop)", foreground="red")
        self.markov_status.config(text="Capturing positions...", foreground="orange")
        
        # Show progress bar inside the fixed container (no UI shift!)
        self.progress_frame.pack(expand=True, fill=tk.BOTH)  # Fill the fixed container
        self.progress_var.set(0)
        recording_duration = self.max_recording_points / 40  # 40Hz sampling rate
        self.progress_label.config(text=f"0:00 / {int(recording_duration//60):01d}:{int(recording_duration%60):02d} (0%)")  # Dynamic duration display
        
        print(f"🎬 Started TIME-BASED recording for {self.current_emotional_state}")
        print(f"⏰ Sampling at {1/self.record_interval:.0f} Hz (captures easing motions!)")
        print(f"💾 Memory limit: {self.max_recording_points} points (~{self.max_recording_points/40:.0f}s)")
        
        # Start time-based sampling timer
        self.start_recording_timer()
        
        # Start progress update timer
        self.update_progress()
        
        # Auto-stop after 20 seconds for stable data collection with full complexity!
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
        total_duration = 20.0  # 20 seconds for stable rich datasets!
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
        
        # STABLE: Auto-stop recording at 20 seconds!
        if elapsed >= 60.0:
            print(f"🔄 Auto-stopping recording at {elapsed:.1f}s")
            self.stop_recording()
            return
        
        # Schedule next update (every 100ms for smooth progress)
        if self.recording:
            self.root.after(100, self.update_progress)
    
    def record_position_sample(self):
        """Record a single position sample - captures both movement and stillness with timing."""
        if not self.recording or self.is_frozen:
            return
            
        current_time = time.time()
        relative_time = current_time - self.record_start_time
        
        # Calculate time difference from previous sample for second-order Markov timing
        dt = current_time - self.previous_time if hasattr(self, 'previous_time') else 0.0
        self.previous_time = current_time
        
        # Create position state for Markov chain with ULTRA HIGH RESOLUTION
        # Use 80x80 grid for capturing subtle easing motions (6400 possible states!)
        grid_size = 80  # Ultra high resolution for easing capture - RESTORED for data quality
        grid_x = int(self.mouse_x * grid_size)
        grid_y = int(self.mouse_y * grid_size)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        
        # Enhanced velocity and movement phase detection for natural hesitation
        velocity_x = 0.0
        velocity_y = 0.0
        movement_phase = "STILL"
        
        if len(self.recorded_positions) > 0:
            prev_pos = self.recorded_positions[-1]
            dt_for_velocity = current_time - prev_pos['time']
            if dt_for_velocity > 0:
                velocity_x = (self.mouse_x - prev_pos['x']) / dt_for_velocity
                velocity_y = (self.mouse_y - prev_pos['y']) / dt_for_velocity
                speed = math.sqrt(velocity_x**2 + velocity_y**2)
                
                # Classify movement phase for better hesitation capture
                if speed < 0.1:
                    movement_phase = "STILL"
                elif speed < 0.5:
                    movement_phase = "SLOW"
                elif speed < 1.5:
                    movement_phase = "MEDIUM"
                elif speed < 3.0:
                    movement_phase = "FAST"
                else:
                    movement_phase = "SUDDEN"
        
        # Create comprehensive position state - FULL DATA COMPLEXITY RESTORED
        position_state = {
            'time': current_time,
            'relative_time': relative_time,
            'dt': dt,  # Time difference for second-order Markov
            'x': self.mouse_x,
            'y': self.mouse_y,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_state': (grid_x, grid_y),  # This becomes our Markov state
            'velocity_x': velocity_x,  # Capture movement velocity for easing
            'velocity_y': velocity_y,
            'speed': math.sqrt(velocity_x**2 + velocity_y**2),  # Overall speed
            'movement_phase': movement_phase,  # Movement classification for hesitation
            'finger_positions': self.finger_positions.copy(),
            'arm_positions': self.arm_positions.copy(),
            'finger_locks': self.finger_locks.copy(),
            'pressed_keys': list(self.pressed_keys),
            'keyboard_targets': self.finger_lock_targets.copy(),
            'servo_positions': self.finger_positions.copy() + self.arm_positions.copy()
        }
        
        self.recorded_positions.append(position_state)

        # Also capture layer frame (per-servo tracking)
        self._record_layer_frame()
        
        # Debug first few samples - RESTORED full debug output
        if len(self.recorded_positions) <= 5:
            print(f"📍 Sample {len(self.recorded_positions)}: ({self.mouse_x:.3f}, {self.mouse_y:.3f}) -> grid ({grid_x}, {grid_y}) [80x80 grid] speed: {math.sqrt(velocity_x**2 + velocity_y**2):.3f} dt: {dt:.3f}s")
    
    def auto_stop_recording(self):
        """Auto-stop recording after 20 seconds."""
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
            
        # Hide progress bar (inside fixed container - no UI shift!)
        self.progress_frame.pack_forget()
            
        duration = time.time() - self.record_start_time
        sample_count = len(self.recorded_positions)
        
        self.record_btn.config(text="🎬 Record Movement (20s)")
        self.record_status.config(text=f"✅ Recorded {sample_count} samples ({duration:.1f}s)", foreground="green")
        
        print(f"Stopped recording. Captured {sample_count} position samples in {duration:.1f} seconds")
        print(f"Sample rate: {sample_count/duration:.1f} Hz")

        # Finalize layer and auto-build Markov chain from merged layers
        self._finish_layer_recording()
        self._merge_layers_and_build_markov()

        # Build Markov chain from recorded positions
        self.build_markov_chain()
        
        # Save to file
        self.save_recording()
        
        # FIXED: Refresh dataset dropdown to show newly generated Markov chain
        self.refresh_datasets()
        print("🔄 Dataset dropdown refreshed - new recording should appear!")
        print("💡 TIP: Record multiple 20s segments for this emotion to build richer datasets!")
    
    def build_markov_chain(self):
        """Build second-order Markov chain with timing from servo movements."""
        emotion = self.current_emotional_state
        if emotion not in self.recorded_movements or len(self.recorded_movements[emotion]) < 10:
            print(f"⚠️ Not enough samples to build Markov chain for {emotion}")
            return
            
        movements = self.recorded_movements[emotion]
        print(f"🔗 Building SECOND-ORDER servo Markov chain with timing from {len(movements)} movements...")
        
        # Simple discretization that preserves movement nuance
        discretization_step = 2.0  # Only 2° steps - much finer than before
        
        def simple_discretize(servo_positions, movement_phase="MEDIUM"):
            """Enhanced discretization that includes movement phase for hesitation capture."""
            # Discretize servo positions
            discretized_servos = tuple(int(round(pos / discretization_step) * discretization_step) for pos in servo_positions)
            # Include movement phase for velocity-aware states
            return (*discretized_servos, movement_phase)
        
        # Build first-order transitions as fallback (keep existing logic)
        servo_transitions = {}
        
        # Build second-order transitions with timing
        servo_second_order = {}
        
        # Process movements with sliding window of length three for second-order
        for i in range(len(movements) - 2):
            prev_movement = movements[i]
            curr_movement = movements[i + 1]
            next_movement = movements[i + 2]
            
            # Discretize servo positions with movement phase context for all three states
            prev_state = simple_discretize(prev_movement['servo_positions'], 
                                         prev_movement.get('movement_phase', 'MEDIUM'))
            curr_state = simple_discretize(curr_movement['servo_positions'], 
                                         curr_movement.get('movement_phase', 'MEDIUM'))
            next_state = simple_discretize(next_movement['servo_positions'], 
                                         next_movement.get('movement_phase', 'MEDIUM'))
            
            # Calculate time difference between i+1 and i+2
            dt = next_movement['time'] - curr_movement['time']
            
            # Detect transition types for enhanced timing
            transition_type = f"{curr_movement.get('movement_phase', 'MEDIUM')}→{next_movement.get('movement_phase', 'MEDIUM')}"
            
            # Convert states to strings for JSON compatibility
            prev_key = str(prev_state)
            curr_key = str(curr_state)
            next_key = str(next_state)
            
            # Build first-order transitions (for fallback)
            if curr_key not in servo_transitions:
                servo_transitions[curr_key] = {}
            if next_key not in servo_transitions[curr_key]:
                servo_transitions[curr_key][next_key] = 0
            servo_transitions[curr_key][next_key] += 1
            
            # Build second-order transitions with timing
            # Key format: "prev_state|curr_state"
            second_order_key = f"{prev_key}|{curr_key}"
            
            if second_order_key not in servo_second_order:
                servo_second_order[second_order_key] = {}
            
            if next_key not in servo_second_order[second_order_key]:
                servo_second_order[second_order_key][next_key] = {
                    'count': 0,
                    'dts': [],
                    'transition_types': []  # Track movement phase transitions
                }
            
            # Increment count and append timing with transition context
            servo_second_order[second_order_key][next_key]['count'] += 1
            servo_second_order[second_order_key][next_key]['dts'].append(dt)
            servo_second_order[second_order_key][next_key]['transition_types'].append(transition_type)
        
        # Convert first-order counts to probabilities
        for state in servo_transitions:
            total = sum(servo_transitions[state].values())
            if total > 0:
                for next_state in servo_transitions[state]:
                    servo_transitions[state][next_state] /= total
        
        # Convert second-order counts to probabilities and calculate enhanced timing distributions
        for second_order_key in servo_second_order:
            total_count = sum(servo_second_order[second_order_key][next_state]['count'] 
                            for next_state in servo_second_order[second_order_key])
            
            if total_count > 0:
                for next_state in servo_second_order[second_order_key]:
                    transition_data = servo_second_order[second_order_key][next_state]
                    count = transition_data['count']
                    dts = transition_data['dts']
                    transition_types = transition_data.get('transition_types', [])
                    
                    # Calculate probability
                    prob = count / total_count
                    
                    if dts:
                        # Calculate full timing distribution for natural hesitation
                        import statistics
                        sorted_dts = sorted(dts)
                        
                        avg_dt = sum(dts) / len(dts)
                        median_dt = statistics.median(dts)
                        min_dt = min(dts)
                        max_dt = max(dts)
                        
                        # Percentiles for capturing hesitation patterns
                        p25_dt = sorted_dts[len(sorted_dts) // 4] if len(sorted_dts) > 3 else avg_dt
                        p75_dt = sorted_dts[3 * len(sorted_dts) // 4] if len(sorted_dts) > 3 else avg_dt
                        p90_dt = sorted_dts[9 * len(sorted_dts) // 10] if len(sorted_dts) > 9 else max_dt
                        
                        # Analyze transition types for hesitation detection
                        stillness_transitions = [t for t in transition_types if 'STILL→' in t]
                        sudden_transitions = [t for t in transition_types if '→SUDDEN' in t]
                        
                        # Clamp values to sensible limits
                        avg_dt = max(0.01, min(0.3, avg_dt))
                        p90_dt = max(0.01, min(0.5, p90_dt))  # Allow longer pauses for hesitation
                        
                    else:
                        # Fallback values
                        avg_dt = median_dt = min_dt = max_dt = p25_dt = p75_dt = p90_dt = 0.03
                        stillness_transitions = sudden_transitions = []
                    
                    # Store enhanced transition data with full timing distribution
                    servo_second_order[second_order_key][next_state] = {
                        'prob': prob,
                        'timing_distribution': {
                            'avg_dt': avg_dt,
                            'median_dt': median_dt,
                            'min_dt': min_dt,
                            'max_dt': max_dt,
                            'p25_dt': p25_dt,
                            'p75_dt': p75_dt, 
                            'p90_dt': p90_dt,  # Key for hesitation patterns
                            'sample_count': len(dts),
                            'raw_samples': dts[:5] if dts else []  # Keep some raw samples
                        },
                        'transition_context': {
                            'has_stillness': len(stillness_transitions) > 0,
                            'has_sudden': len(sudden_transitions) > 0,
                            'stillness_ratio': len(stillness_transitions) / len(transition_types) if transition_types else 0,
                            'common_transitions': list(set(transition_types))[:3]  # Most common patterns
                        },
                        'count': count
                    }
        
        # Store both first and second-order Markov chains
        self.markov_chains[emotion] = {
            'servo_transitions': servo_transitions,
            'servo_second_order': servo_second_order,  # NEW: Second-order transitions with timing
            'discretization': discretization_step,
            'total_samples': len(movements),
            'unique_states': len(servo_transitions),
            'unique_second_order_keys': len(servo_second_order),
            'simple_version': True,
            'second_order_enabled': True,  # Flag for second-order support
            'sample_rate': 40
        }
        
        print(f"✅ SECOND-ORDER Servo Markov chain built for {emotion}:")
        print(f"   📊 {len(movements)} movements → {len(servo_transitions)} first-order states")
        print(f"   🔗 {len(servo_second_order)} second-order transitions (prev|curr → next)")
        print(f"   🎯 Fine discretization: {discretization_step}° steps (preserves flow)")
        print(f"   ⏱️ Timing-aware: realistic dwell-time distribution per transition")
        
        # Update status
        if hasattr(self, 'markov_status'):
            self.markov_status.config(
                text=f"{emotion}: {len(servo_transitions)} states, {len(servo_second_order)} 2nd-order",
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
        save_msg = f"💾 Saved {len(movements)} samples"
        if has_markov:
            save_msg += " + chain"
        self.record_status.config(text=save_msg, foreground="green")
    
    def refresh_datasets(self):
        """Scan for available datasets and update display - ONLY compatible current format."""
        print("🔄 Refreshing dataset list (current format only)...")
        self.available_datasets = {}
        self.dataset_info = {}
        
        if not os.path.exists("movement_recordings"):
            print("📁 No movement_recordings directory found")
            self.update_dataset_display()
            return
            
        # Scan only for current format JSON files 
        compatible_files = 0
        incompatible_files = 0
        
        for filename in os.listdir("movement_recordings"):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join("movement_recordings", filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                # Check if this is the current servo-based format
                format_version = data.get('format_version', 'unknown')
                markov_chain = data.get('markov_chain', {})
                
                # Accept current servo format OR older files with servo data OR recent files without format_version
                is_current_format = (format_version == '2.0_servo_based' and 
                                  'servo_transitions' in markov_chain and
                                  'servo_movements' in data)
                
                is_compatible_old_format = (format_version == 'unknown' and
                                          'servo_transitions' in markov_chain and
                                          'movements' in data)
                
                # Also accept files with movement_count but no format_version (recent saves)
                is_recent_format = (format_version == 'unknown' and
                                  'movement_count' in data and
                                  'movements' in data)
                
                if is_current_format or is_compatible_old_format or is_recent_format:
                    emotion = data.get('emotion', 'unknown')
                    if emotion not in self.available_datasets:
                        self.available_datasets[emotion] = []
                        
                    # Extract data for display - handle all formats
                    timestamp = data.get('timestamp', 'unknown')
                    if is_current_format:
                        sample_count = data.get('movement_count', 0)  # New format
                    elif is_recent_format:
                        sample_count = data.get('movement_count', 0)  # Recent format without version
                        print(f"📊 Loading recent format file: {filename} ({sample_count} samples)")
                    else:
                        sample_count = data.get('movement_count', len(data.get('movements', [])))  # Old format
                        print(f"⚠️ Loading compatible old format file: {filename}")
                        
                    duration = data.get('duration', 0)
                    unique_states = markov_chain.get('unique_states', 0)
                    custom_name = data.get('custom_name', '')
                    
                    # Create clear display name
                    if custom_name:
                        display_name = custom_name
                    else:
                        display_name = f"Recording-{timestamp}"
                    
                    dataset_info = {
                        'filename': filename,
                        'filepath': filepath,
                        'emotion': emotion,
                        'timestamp': timestamp,
                        'sample_count': sample_count,
                        'duration': duration,
                        'unique_states': unique_states,
                        'display_name': display_name,
                        'custom_name': custom_name,
                        'data': data
                    }
                    
                    self.available_datasets[emotion].append(dataset_info)
                    self.dataset_info[filename] = dataset_info
                    compatible_files += 1
                    
                else:
                    # Skip incompatible old format files
                    incompatible_files += 1
                    print(f"⚠️ Skipping incompatible file: {filename} (format: {format_version})")
                    
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                incompatible_files += 1
        
        # Sort datasets by timestamp (newest first)
        for emotion in self.available_datasets:
            try:
                self.available_datasets[emotion].sort(key=lambda x: str(x['timestamp']), reverse=True)
            except Exception as e:
                print(f"⚠️ Error sorting datasets for {emotion}: {e}")
                self.available_datasets[emotion].sort(key=lambda x: x['filename'], reverse=True)
            
        self.update_dataset_display()
        
        # Clear status message
        if compatible_files > 0:
            print(f"✅ Loaded {compatible_files} compatible datasets for {len(self.available_datasets)} emotions")
        else:
            print("📁 No compatible datasets found - record some fresh data!")
            
        if incompatible_files > 0:
            print(f"⚠️ Skipped {incompatible_files} incompatible old format files")
            print("💡 Use 'Clear All' to remove old incompatible files and start fresh")
    
    def update_dataset_display(self):
        """Update the dataset display for current emotion with clear, explicit information."""
        emotion = self.current_emotional_state
        datasets = self.available_datasets.get(emotion, [])
        
        # Update emotion dataset label with more explicit information
        if not datasets:
            self.current_emotion_dataset_label.config(
                text=f"📁 {emotion.replace('_', ' ').title()}: No datasets - record some movements first!", 
                foreground="#8B5CF6"
            )
            self.dataset_dropdown['values'] = []
            self.dataset_var.set("")
            self.dataset_status_label.config(text=f"No {emotion.replace('_', ' ')} datasets available", foreground="gray")
        else:
            total_samples = sum(d.get('sample_count', 0) for d in datasets)
            total_states = sum(d.get('unique_states', 0) for d in datasets)
            
            # More explicit label showing exactly what we have
            self.current_emotion_dataset_label.config(
                text=f"📁 {emotion.replace('_', ' ').title()}: {len(datasets)} recording(s) with {total_samples:,} servo samples", 
                foreground="darkgreen"
            )
            
            # Create clearer dropdown options with explicit timestamps and sample counts
            display_options = []
            for i, dataset in enumerate(datasets):
                timestamp = dataset.get('timestamp', 'unknown')
                sample_count = dataset.get('sample_count', 0)
                duration = dataset.get('duration', 0)
                
                # Format timestamp to be more readable
                try:
                    if len(timestamp) >= 8:  # Format: YYYYMMDD_HHMMSS
                        date_part = timestamp[:8]  # YYYYMMDD
                        time_part = timestamp[9:15] if len(timestamp) > 9 else "000000"  # HHMMSS
                        
                        # Convert to readable format
                        formatted_date = f"{date_part[4:6]}/{date_part[6:8]}"  # MM/DD
                        formatted_time = f"{time_part[:2]}:{time_part[2:4]}"   # HH:MM
                        readable_time = f"{formatted_date} {formatted_time}"
                    else:
                        readable_time = timestamp
                except:
                    readable_time = timestamp
                
                # Create explicit dropdown entry
                custom_name = dataset.get('custom_name', '')
                if custom_name:
                    display_name = f"{custom_name} ({readable_time}) - {sample_count:,} samples, {duration:.1f}s"
                else:
                    display_name = f"Recording {readable_time} - {sample_count:,} samples, {duration:.1f}s"
                
                display_options.append(display_name)
            
            self.dataset_dropdown['values'] = display_options
            
        # Auto-select dataset - either pre-loaded during startup or most recent
        active_filename = self.active_datasets.get(emotion, '')
        if active_filename and display_options:
            # Find the display option that corresponds to the active dataset
            for i, dataset in enumerate(datasets):
                if dataset['filename'] == active_filename:
                    self.dataset_var.set(display_options[i])
                    break
            else:
                # Active dataset not found, select most recent
                self.dataset_var.set(display_options[0])
                self.active_datasets[emotion] = datasets[0]['filename']
        elif not self.dataset_var.get() and display_options:
            # No active dataset, select most recent
            self.dataset_var.set(display_options[0])
            self.active_datasets[emotion] = datasets[0]['filename']        # Always show which emotion we're currently working with
        emotion_display = emotion.replace('_', ' ').title()
        if hasattr(self, 'dataset_status_label'):
            if datasets:
                selected_dataset = self.dataset_var.get()
                active_filename = self.active_datasets.get(emotion, '')
                
                if selected_dataset:
                    # Check if this dataset has a pre-loaded Markov chain
                    has_preloaded_chain = emotion in self.markov_chains
                    chain_status = " [Chain Ready]" if has_preloaded_chain else ""
                    
                    self.dataset_status_label.config(
                        text=f"✅ {emotion_display}: {selected_dataset.split(' - ')[0]}{chain_status}", 
                        foreground="darkgreen"
                    )
                else:
                    self.dataset_status_label.config(
                        text=f"⚠️ Select a {emotion_display} dataset for generation", 
                        foreground="orange"
                    )
            else:
                self.dataset_status_label.config(
                    text=f"❌ No {emotion_display} datasets - record some first!", 
                    foreground="red"
                )
    
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
    
    def load_markov_chain_from_dataset(self, dataset_info, emotion):
        """Load Markov chain from a specific dataset info object."""
        try:
            data = dataset_info['data']
            markov_chain = data.get('markov_chain', {})
            if markov_chain:
                self.markov_chains[emotion] = markov_chain
                
                # Reset Markov state variables when loading new chain to prevent generation breaking
                # Initialize with a random starting state from the new chain
                servo_transitions = markov_chain.get('servo_transitions', {})
                if servo_transitions:
                    import random
                    starting_state = random.choice(list(servo_transitions.keys()))
                    self.current_markov_state = starting_state
                    self.prev_markov_state = None
                    print(f"🔄 Reset Markov state variables - starting with: {starting_state}")
                else:
                    self.current_markov_state = None
                    self.prev_markov_state = None
                    print(f"🔄 Reset Markov state variables - no transitions found")
                
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
                    avg_transitions = sum(len(t) for t in markov_chain.get('servo_transitions', {}).values()) / len(markov_chain.get('servo_transitions', {})) if markov_chain.get('servo_transitions') else 0
                    self.markov_status.config(text=f"{unique_states} states, {avg_transitions:.1f} avg transitions", foreground="green")
                    print(f"🔗 Loaded legacy Markov chain: {unique_states} states")
                    
                print(f"✅ Loaded Markov chain from {dataset_info['display_name']} for {emotion}")
                return True
            else:
                print(f"⚠️ No Markov chain in {dataset_info['display_name']}")
                return False
        except Exception as e:
            print(f"❌ Error loading Markov chain from dataset: {e}")
            return False
    

    def update_markov_status_for_emotion(self, emotion):
        """Update the UI markov_status to reflect available chains for an emotion."""
        if emotion in self.markov_chains:
            chain = self.markov_chains[emotion]
            if isinstance(chain, dict):
                unique_states = chain.get('unique_states', 0)
                if chain.get('second_order_enabled', False):
                    second_order_keys = chain.get('unique_second_order_keys', 0)
                    self.markov_status.config(text=f"{unique_states} states, {second_order_keys} 2nd-order keys", foreground="green")
                else:
                    avg_transitions = 0
                    if 'servo_transitions' in chain:
                        transitions = chain['servo_transitions']
                        if transitions:
                            avg_transitions = sum(len(t) for t in transitions.values()) / len(transitions)
                    self.markov_status.config(text=f"{unique_states} states, {avg_transitions:.1f} avg transitions", foreground="green")
                print(f"🎯 UI status updated for {emotion}: chain available")
            else:
                self.markov_status.config(text="Chain format unknown", foreground="orange")
        else:
            self.markov_status.config(text="No chains built", foreground="gray")

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
        self.record_status.config(text="▶️ PLAYING BACK...", foreground="blue")
        
        print(f"▶️ Started playback of {len(self.current_playback)} movements for {self.current_emotional_state}")
    
    def stop_playback(self):
        """Stop playback."""
        self.playing_back = False
        self.playback_btn.config(text="▶️ Play Back")
        self.record_status.config(text="Ready to record 20s segments (Spacebar)", foreground="gray")
        print("⏹️ Playback stopped")
    
    def update_playback(self):
        """Update servo positions during playback - GOLDEN MASTER style: simple and direct."""
        if not self.playing_back or not self.current_playback:
            return
            
        current_time = time.time()
        playback_elapsed = current_time - self.playback_start_time
        
        # GOLDEN MASTER APPROACH: Find exact recorded movement at current time (no interpolation)
        target_movement = None
        
        # Simple linear search like golden master (works perfectly for their cursor system)
        for movement in self.current_playback:
            movement_time = movement['time'] - self.current_playback[0]['time']  # Relative time
            
            # Find the movement that should be playing right now
            if movement_time <= playback_elapsed:
                target_movement = movement
            else:
                break  # Past this point in timeline
        
        # Apply the movement directly (no interpolation - just like golden master)
        if target_movement:
            # SERVO-ONLY PLAYBACK: Directly set servo positions 
            if 'servo_positions' in target_movement and target_movement['servo_positions']:
                self.finger_positions = target_movement['servo_positions'].copy()
            elif 'finger_positions' in target_movement and target_movement['finger_positions']:
                # Fallback for older recordings
                self.finger_positions = target_movement['finger_positions'].copy()
            
            # Send to servos immediately (golden master style - direct, no delays)
            self.send_to_hand_controller()
        
        # Check if playback finished (exactly like golden master)
        if self.current_playback:
            total_duration = self.current_playback[-1]['time'] - self.current_playback[0]['time']
            if playback_elapsed >= total_duration:
                print(f"✅ Playback completed after {total_duration:.1f}s ({len(self.current_playback)} movements)")
                self.stop_playback()
    
    def parse_markov_state_key(self, key_str):
        """Parse a string key back to tuple for generation (handles first-order, second-order, and legacy formats)."""
        if isinstance(key_str, tuple):
            return key_str  # Already a tuple (backwards compatibility)
        
        # Check if this is a second-order key (contains |)
        if '|' in str(key_str):
            try:
                # Split the second-order key: "prev_state|curr_state"
                parts = str(key_str).split('|')
                if len(parts) == 2:
                    prev_state = self._parse_single_state(parts[0])
                    curr_state = self._parse_single_state(parts[1])
                    return [prev_state, curr_state]  # Return list of two tuples
                else:
                    print(f"⚠️ Invalid second-order key format: {key_str}")
                    return (90, 90, 90, 90)  # Fallback
            except Exception as e:
                print(f"⚠️ Error parsing second-order key '{key_str}': {e}")
                return (90, 90, 90, 90)  # Fallback
        else:
            # Regular first-order key
            return self._parse_single_state(key_str)
    
    def _parse_single_state(self, state_str):
        """Parse a single state string back to tuple, handling movement phase enhancement."""
        try:
            # Clean up the string - remove extra characters and fix malformed keys
            clean_str = str(state_str).strip()
            
            # Remove extra parentheses and fix malformed strings
            clean_str = clean_str.replace('))', ')')  # Fix double closing parens
            clean_str = clean_str.replace('..0', '.0')  # Fix truncated decimals
            
            # Remove parentheses and split by comma
            clean_str = clean_str.strip("()")
            parts = [part.strip().strip("'\"") for part in clean_str.split(",")]  # Also strip quotes
            
            # Handle different tuple formats
            if len(parts) == 2:
                # Simple (x, y) tuple
                return (int(float(parts[0])), int(float(parts[1])))
            else:
                # Generic N-servo tuple, possibly with movement_phase as last element
                # Try to parse all as numbers; if last element fails, it's the phase — skip it
                numeric_parts = []
                for part in parts:
                    try:
                        numeric_parts.append(int(float(part)))
                    except ValueError:
                        break  # Hit movement_phase string, stop
                return tuple(numeric_parts) if numeric_parts else (90,) * self.num_fingers
        except (ValueError, IndexError) as e:
            print(f"Failed to parse single state '{state_str}': {e}")
            return (90,) * self.num_fingers
    
    def _apply_markov_state_to_servos(self, state):
        """Apply a parsed Markov state tuple directly to finger + arm positions.
        State format: (f0, f1, ..., fN, arm0, arm1, ..., armM) with optional movement_phase stripped.
        """
        total_servos = self.num_fingers + self.num_arm_servos
        for i in range(min(len(state), total_servos)):
            target_pos = max(0.0, min(180.0, float(state[i])))
            if i < self.num_fingers:
                self.finger_positions[i] = target_pos
            else:
                arm_idx = i - self.num_fingers
                self.arm_positions[arm_idx] = target_pos
                # Update slider to reflect generated position
                if hasattr(self, 'arm_slider_vars') and arm_idx < len(self.arm_slider_vars):
                    self.arm_slider_vars[arm_idx].set(target_pos)

    def _apply_markov_state_with_easing(self, state, easing_factor):
        """Apply a parsed Markov state with interpolation easing to finger + arm positions."""
        total_servos = self.num_fingers + self.num_arm_servos
        for i in range(min(len(state), total_servos)):
            target_pos = max(0.0, min(180.0, float(state[i])))
            if i < self.num_fingers:
                self.finger_positions[i] = self.finger_positions[i] * (1 - easing_factor) + target_pos * easing_factor
            else:
                arm_idx = i - self.num_fingers
                self.arm_positions[arm_idx] = self.arm_positions[arm_idx] * (1 - easing_factor) + target_pos * easing_factor
                if hasattr(self, 'arm_slider_vars') and arm_idx < len(self.arm_slider_vars):
                    self.arm_slider_vars[arm_idx].set(self.arm_positions[arm_idx])

    def start_markov_generation(self):
        """Start Markov chain generation for current emotional state with second-order support."""
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
        
        # Check for second-order support
        if chain.get('second_order_enabled', False) and 'servo_second_order' in chain:
            # Use second-order chain
            servo_second_order = chain['servo_second_order']
            
            if not servo_second_order:
                print(f"❌ Empty second-order Markov chain for {self.current_emotional_state}")
                return
            
            print(f"🎨 Starting SECOND-ORDER Markov generation for {self.current_emotional_state}")
            print(f"🔗 Using {len(servo_second_order)} second-order transitions with timing")
            
            # Pick a random starting pair of states from second-order keys
            start_key = random.choice(list(servo_second_order.keys()))
            
            # Split the key to get prev_state and curr_state
            prev_state_key, curr_state_key = start_key.split('|')
            
            # Parse states
            prev_state = self._parse_single_state(prev_state_key)
            curr_state = self._parse_single_state(curr_state_key)
            
            # Set initial states
            self.prev_markov_state = prev_state_key
            self.current_markov_state = curr_state_key
            
            # Set initial servo positions from current state
            if hasattr(self, 'markov_restart_from_position'):
                self.finger_positions = self.markov_restart_from_position.copy()
                delattr(self, 'markov_restart_from_position')
                print(f"🎯 Starting from transition end position: {self.finger_positions}")
            else:
                self._apply_markov_state_to_servos(curr_state)
            
            # Get initial timing from the starting transition
            if start_key in servo_second_order:
                # Pick first available next state to get timing
                next_states = servo_second_order[start_key]
                if next_states:
                    first_next_state = list(next_states.keys())[0]
                    avg_dt = next_states[first_next_state].get('avg_dt', 0.03)
                    self.generation_speed = max(0.01, min(0.2, avg_dt))  # Clamp timing
                else:
                    self.generation_speed = 0.03  # Default
            else:
                self.generation_speed = 0.03  # Default
            
            print(f"🎯 Starting from: prev='{self.prev_markov_state}' curr='{self.current_markov_state}'")
            print(f"⏱️ Initial timing: {self.generation_speed:.3f}s")
            
            transition_type = "second-order"
            
        else:
            # Fallback to first-order chain
            transitions = None
            transition_type = "first-order"
            
            if 'servo_transitions' in chain:
                transitions = chain['servo_transitions']
            elif 'cursor_transitions' in chain:
                transitions = chain['cursor_transitions']
            elif 'transitions' in chain:
                transitions = chain['transitions']
            else:
                print(f"❌ No valid transitions found in Markov chain for {self.current_emotional_state}")
                self.markov_status.config(text="Invalid chain format", foreground="red")
                return
                
            if not transitions:
                print(f"❌ Empty Markov chain for {self.current_emotional_state}")
                return
            
            # Standard first-order initialization
            start_state_key = random.choice(list(transitions.keys()))
            start_state = self.parse_markov_state_key(start_state_key)
            self.current_markov_state = start_state_key
            self.prev_markov_state = None  # Not used in first-order
            
            # Handle different state types
            if 'servo_transitions' in chain:
                if hasattr(self, 'markov_restart_from_position'):
                    self.finger_positions = self.markov_restart_from_position.copy()
                    delattr(self, 'markov_restart_from_position')
                else:
                    self._apply_markov_state_to_servos(start_state)
            else:
                # Cursor-based fallback
                grid_size = chain.get('grid_size', 80)
                grid_x, grid_y = start_state[:2]
                grid_x = float(grid_x) if isinstance(grid_x, str) else grid_x
                grid_y = float(grid_y) if isinstance(grid_y, str) else grid_y
                self.mouse_x = (grid_x + 0.5) / grid_size
                self.mouse_y = (grid_y + 0.5) / grid_size
            
            self.generation_speed = 0.03  # Default timing for first-order
            
            print(f"🎯 Fallback to first-order generation with {len(transitions)} states")
        
        # Start generation
        self.generating = True
        self.generation_start_time = time.time()
        
        self.generate_btn.config(text="⏹️ Stop Generation")
        self.markov_status.config(text=f"Generating {transition_type}...", foreground="purple")
        
        print(f"🎨 Started {transition_type} Markov generation for {self.current_emotional_state}")
        
        # Start generation timer
        self.start_generation_timer()
    
    def start_generation_timer(self):
        """Start the generation timer for Markov chain steps - ROBUST INFINITE LOOP."""
        if self.generating:
            try:
                self.step_markov_generation()
            except Exception as e:
                print(f"❌ Error in generation step: {e}")
                print("🔄 Continuing generation despite error...")
            
            # Cancel any existing timer before scheduling new one
            if self.generation_timer:
                self.root.after_cancel(self.generation_timer)
                self.generation_timer = None
            
            # Schedule next step (only if still generating)
            if self.generating:
                interval_ms = int(self.generation_speed * 1000)
                self.generation_timer = self.root.after(interval_ms, self.start_generation_timer)
    
    def step_markov_generation(self):
        """Take one step in Markov generation with second-order support."""
        if not self.generating or self.current_emotional_state not in self.markov_chains:
            return
        
        # Handle dataset transitions (smooth easing)
        if hasattr(self, 'dataset_transitioning') and self.dataset_transitioning:
            print(f"🔄 Transition active - calling update_dataset_transition")
            self.update_dataset_transition()
            return  # Skip normal generation during transition
        
        try:
            chain = self.markov_chains[self.current_emotional_state]
            
            # Check if we should use second-order logic
            if (chain.get('second_order_enabled', False) and 
                'servo_second_order' in chain and 
                self.prev_markov_state is not None):
                
                # SECOND-ORDER GENERATION
                servo_second_order = chain['servo_second_order']
                
                # Build lookup key: "prev_state|curr_state"
                lookup_key = f"{self.prev_markov_state}|{self.current_markov_state}"
                
                if lookup_key in servo_second_order:
                    # Use second-order transitions
                    next_states = servo_second_order[lookup_key]
                    
                    if next_states:
                        # Get state keys and probabilities
                        state_keys = list(next_states.keys())
                        probabilities = [next_states[key]['prob'] for key in state_keys]
                        
                        # Choose next state using probabilities
                        next_state_key = random.choices(state_keys, weights=probabilities)[0]
                        
                        # Get enhanced timing for this transition with context awareness
                        transition_data = next_states[next_state_key]
                        timing_dist = transition_data.get('timing_distribution', {})
                        transition_context = transition_data.get('transition_context', {})
                        
                        # Smart timing selection based on context and movement
                        next_state = self._parse_single_state(next_state_key)
                        current_state = self._parse_single_state(self.current_markov_state) if len(self._parse_single_state(self.current_markov_state)) >= 4 else [90, 90, 90, 90]
                        
                        # Calculate movement magnitude across all servo positions
                        num_servos = min(len(next_state), len(current_state), self.num_fingers + self.num_arm_servos)
                        if num_servos >= 4:
                            servo_changes = [abs(float(next_state[i]) - float(current_state[i])) for i in range(num_servos)]
                            max_change = max(servo_changes)
                            total_change = sum(servo_changes)
                            
                            # Context-aware timing selection
                            if transition_context.get('has_stillness', False) and max_change > 20:
                                # Stillness to movement transition - use longer hesitation timing
                                selected_dt = timing_dist.get('p90_dt', timing_dist.get('avg_dt', 0.03))
                                print(f"⏸️ Stillness→Movement: using p90 timing {selected_dt:.3f}s")
                            elif transition_context.get('has_sudden', False):
                                # Sudden movement - use shorter, snappy timing
                                selected_dt = timing_dist.get('min_dt', timing_dist.get('avg_dt', 0.03))
                                print(f"⚡ Sudden movement: using min timing {selected_dt:.3f}s")
                            elif max_change > 30:
                                # Large movement - use upper percentile for weight/hesitation
                                selected_dt = timing_dist.get('p75_dt', timing_dist.get('avg_dt', 0.03))
                                print(f"🏋️ Large movement: using p75 timing {selected_dt:.3f}s")
                            elif max_change < 5:
                                # Small adjustment - use median for natural flow
                                selected_dt = timing_dist.get('median_dt', timing_dist.get('avg_dt', 0.03))
                            else:
                                # Normal movement - use average
                                selected_dt = timing_dist.get('avg_dt', 0.03)
                            
                            self.generation_speed = max(0.01, min(0.3, selected_dt))  # Allow longer pauses
                        else:
                            # Fallback to average timing
                            self.generation_speed = max(0.01, min(0.2, timing_dist.get('avg_dt', 0.03)))
                        
                        # Update states: shift the window
                        self.prev_markov_state = self.current_markov_state
                        self.current_markov_state = next_state_key
                        
                        # Apply the new state to all servos (fingers + arm)
                        self._apply_markov_state_to_servos(next_state)
                        
                        # Silenced 2nd-order output
                        self.send_to_hand_controller()
                        return
                    else:
                        print(f"⚠️ Empty transitions for second-order key: {lookup_key}")
                else:
                    print(f"🔄 Second-order key not found: {lookup_key}, falling back to first-order")
                
                # If second-order lookup failed, fall back to first-order with current state
                # (continuing below)
            
            # FIRST-ORDER FALLBACK (or primary for first-order chains)
            # Get servo transitions (simple like golden master)
            if 'servo_transitions' not in chain:
                print("❌ No servo transitions found in chain")
                return
                
            transitions = chain['servo_transitions']
            
            # ROBUST: Ensure we have transitions to work with
            if not transitions:
                print("❌ Empty transitions dict - cannot generate")
                return
            
            # Handle dead ends like golden master (simple random jump)
            if self.current_markov_state not in transitions:
                # Pick a random new state (simple, effective)
                available_states = list(transitions.keys())
                if available_states:
                    self.current_markov_state = random.choice(available_states)
                    print(f"🔄 Dead end - jumping to new state: {self.current_markov_state}")
                    # Continue to apply the new state immediately (don't return)
                else:
                    print("❌ No available states in Markov chain")
                    return
            
            # ROBUST: Double-check that our current state has transitions
            if self.current_markov_state not in transitions:
                print(f"❌ Current state {self.current_markov_state} still not in transitions after recovery")
                # Force pick the first available state
                available_states = list(transitions.keys())
                if available_states:
                    self.current_markov_state = available_states[0]
                    print(f"🔄 Force-selecting first available state: {self.current_markov_state}")
                else:
                    print("❌ No states available at all")
                    return
            
            # Get possible next states and their probabilities (exactly like golden master)
            next_states = transitions[self.current_markov_state]
            
            # ROBUST: Ensure next_states is not empty
            if not next_states:
                print(f"⚠️ No transitions from current state {self.current_markov_state}, picking random state")
                available_states = list(transitions.keys())
                if available_states:
                    self.current_markov_state = random.choice(available_states)
                    next_states = transitions[self.current_markov_state]
                else:
                    print("❌ No states available for fallback")
                    return
            
            state_keys = list(next_states.keys())
            probabilities = list(next_states.values())
            
            # PROBABILITY FLATTENING: Reduce dominance of high-probability transitions
            # This prevents getting stuck in "curled finger" probability sinks
            flattening_factor = 0.5  # Much more uniform distribution to prevent sinks (0.5 = balanced, 1.0 = original)
            if len(probabilities) > 1:
                # Flatten probabilities to add more variety
                min_prob = min(probabilities)
                max_prob = max(probabilities)
                if max_prob > min_prob:
                    # Scale probabilities toward uniform distribution
                    flattened_probs = []
                    for prob in probabilities:
                        # Blend between original probability and uniform (1/n)
                        uniform_prob = 1.0 / len(probabilities)
                        flattened_prob = prob * flattening_factor + uniform_prob * (1 - flattening_factor)
                        flattened_probs.append(flattened_prob)
                    
                    # Normalize to sum to 1
                    prob_sum = sum(flattened_probs)
                    probabilities = [p / prob_sum for p in flattened_probs]
            
            # ROBUST: Ensure we have valid probabilities
            if not state_keys or not probabilities:
                print(f"⚠️ Invalid state/probability data, using random fallback")
                available_states = list(transitions.keys())
                if available_states:
                    self.current_markov_state = random.choice(available_states)
                    return  # Will try again next cycle
                else:
                    print("❌ Cannot recover - no valid states")
                    return
            
            # Weighted random choice (same as golden master) WITH ENHANCED DIVERSITY INJECTION
            try:
                diversity_jump = False
                
                # STUCK STATE DETECTION: Track how long we've been in the same state
                if not hasattr(self, '_state_repetition_count'):
                    self._state_repetition_count = 0
                    self._last_state = None
                
                if self.current_markov_state == self._last_state:
                    self._state_repetition_count += 1
                else:
                    self._state_repetition_count = 0
                    self._last_state = self.current_markov_state
                
                # ADAPTIVE DIVERSITY: Higher chance if stuck in same state
                base_diversity_chance = 0.02  # 2% baseline chance
                if self._state_repetition_count > 5:
                    # Stuck for 5+ cycles - increase diversity chance dramatically
                    stuck_bonus = min(0.3, self._state_repetition_count * 0.05)  # Up to 30% extra
                    diversity_chance = base_diversity_chance + stuck_bonus
                    if random.random() < diversity_chance:
                        diversity_jump = True
                        print(f"🔓 Breaking stuck state after {self._state_repetition_count} repetitions (chance: {diversity_chance:.1%})")
                elif random.random() < base_diversity_chance:
                    diversity_jump = True
                
                if diversity_jump:
                    # GENTLE DIVERSITY JUMP: Select a nearby state instead of random distant one
                    available_states = list(transitions.keys())
                    
                    # DISTANCE-BASED SELECTION: Prefer states that aren't too far from current
                    current_positions = self.parse_markov_state_key(self.current_markov_state) if self.current_markov_state else self.finger_positions
                    
                    # Calculate distances and filter out extremely distant states
                    candidate_states = []
                    for state_key in available_states:
                        candidate_pos = self.parse_markov_state_key(state_key)
                        if candidate_pos and len(candidate_pos) >= 2:
                            # Calculate average distance across servos
                            total_distance = sum(abs(candidate_pos[i] - current_positions[i]) 
                                               for i in range(min(len(candidate_pos), len(current_positions))))
                            avg_distance = total_distance / min(len(candidate_pos), len(current_positions))
                            
                            # Accept states with reasonable distance (not too extreme)
                            if avg_distance <= 60.0:  # Max 60° average change per servo
                                candidate_states.append((state_key, avg_distance))
                    
                    if candidate_states:
                        # Prefer closer states using weighted selection (closer = higher weight)
                        weights = [1.0 / (1.0 + dist * 0.1) for _, dist in candidate_states]  # Inverse distance weighting
                        selected_state, distance = random.choices(candidate_states, weights=weights)[0]
                        next_state_key = selected_state
                        print(f"🌊 Gentle diversity drift (avg Δ={distance:.1f}°): {next_state_key}")
                        
                        # Reset repetition counter since we're moving to a new state
                        self._state_repetition_count = 0
                    else:
                        # Fallback to normal selection if no reasonable candidates
                        next_state_key = random.choices(state_keys, weights=probabilities)[0]
                        print("🎯 No suitable diversity candidates, using normal selection")
                else:
                    # Normal weighted choice
                    next_state_key = random.choices(state_keys, weights=probabilities)[0]
                
                # Update states for next iteration
                self.prev_markov_state = self.current_markov_state  # For potential second-order use
                self.current_markov_state = next_state_key
                
                # Store diversity jump flag for gentler easing
                self._is_diversity_jump = diversity_jump
            except (ValueError, IndexError) as e:
                print(f"⚠️ Error in weighted choice: {e}, using uniform random")
                next_state_key = random.choice(state_keys)
                self.prev_markov_state = self.current_markov_state
                self.current_markov_state = next_state_key
            
            # Parse the state key back to servo positions
            next_state = self.parse_markov_state_key(next_state_key)
            
            # ROBUST: Ensure we got valid servo positions
            if not next_state or len(next_state) < 2:
                print(f"⚠️ Invalid parsed state {next_state}, using fallback")
                # Use current finger positions as fallback
                next_state = self.finger_positions.copy()
            
            # Apply to servo positions with adaptive easing
            if len(next_state) >= 2:
                # ADAPTIVE EASING: Use data-driven easing from recorded movement characteristics
                if hasattr(self, '_is_diversity_jump') and self._is_diversity_jump:
                    easing_factor = 0.08  # Gentle easing for diversity jumps
                elif hasattr(self, '_intelligent_easing_factor'):
                    easing_factor = self._intelligent_easing_factor
                else:
                    easing_factor = self.generation_easing_factor  # 0.3 default

                if not self.transitioning_to_dataset:
                    self._apply_markov_state_with_easing(next_state, easing_factor)

                self.send_to_hand_controller()
            else:
                print(f"Invalid servo state format: {next_state}, continuing anyway")
            
            # Clean up diversity jump flag for next step
            if hasattr(self, '_is_diversity_jump'):
                self._is_diversity_jump = False
            
            # Handle simple dataset transition
            if hasattr(self, 'transitioning_to_dataset') and self.transitioning_to_dataset:
                self.update_simple_transition()
                
        except Exception as e:
            print(f"❌ Error in Markov generation step: {e}")
            print("🔄 Attempting to recover by selecting random state...")
            try:
                # Emergency recovery - pick any available state
                transitions = self.markov_chains[self.current_emotional_state]['servo_transitions']
                available_states = list(transitions.keys())
                if available_states:
                    self.prev_markov_state = self.current_markov_state
                    self.current_markov_state = random.choice(available_states)
                    print(f"✅ Recovery successful, new state: {self.current_markov_state}")
                else:
                    print("❌ Recovery failed - no available states")
            except Exception as recovery_error:
                print(f"❌ Recovery also failed: {recovery_error}")
                # Continue anyway - don't stop generation
    
    def update_simple_transition(self):
        """Update simple smooth transition to new dataset."""
        if not hasattr(self, 'transitioning_to_dataset') or not self.transitioning_to_dataset:
            return
            
        current_time = time.time()
        elapsed = current_time - self.transition_start_time
        progress = min(1.0, elapsed / self.transition_duration)
        
        # Smooth easing curve (ease-in-out)
        eased_progress = 0.5 * (1 - math.cos(progress * math.pi))
        
        # Interpolate to target positions
        if hasattr(self, 'transition_target_positions'):
            for i in range(min(len(self.finger_positions), len(self.transition_target_positions))):
                start_pos = self.transition_start_positions[i]
                target_pos = self.transition_target_positions[i]
                self.finger_positions[i] = start_pos + (target_pos - start_pos) * eased_progress
            
            # Send updated positions to servos
            self.send_to_hand_controller()
            
            # Check if transition is complete
            if progress >= 1.0:
                self.transitioning_to_dataset = False
                print(f"✅ Dataset transition complete - now using new Markov chain")
                
                # Clear transition data
                if hasattr(self, 'transition_start_positions'):
                    delattr(self, 'transition_start_positions')
                if hasattr(self, 'transition_target_positions'):
                    delattr(self, 'transition_target_positions')

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
        
        # Clear generation state (including second-order state)
        self.current_markov_state = None
        self.prev_markov_state = None  # Clear previous state for second-order
        
        # Update status
        emotion = self.current_emotional_state
        if emotion in self.markov_chains:
            chain = self.markov_chains[emotion]
            unique_states = chain.get('unique_states', 0)
            if chain.get('second_order_enabled', False):
                second_order_keys = chain.get('unique_second_order_keys', 0)
                self.markov_status.config(text=f"Chain: {unique_states} states, {second_order_keys} 2nd-order", foreground="blue")
            else:
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
    
    def cleanup_all_timers(self):
        """Clean up all active timers to prevent UI freezing."""
        timers_to_cancel = []
        
        # Collect all timer references
        if hasattr(self, 'recording_timer') and self.recording_timer:
            timers_to_cancel.append(('recording_timer', self.recording_timer))
            
        if hasattr(self, 'generation_timer') and self.generation_timer:
            timers_to_cancel.append(('generation_timer', self.generation_timer))
            
        # Cancel all timers
        for timer_name, timer_id in timers_to_cancel:
            try:
                self.root.after_cancel(timer_id)
                setattr(self, timer_name, None)
                print(f"✅ Cancelled {timer_name}")
            except Exception as e:
                print(f"⚠️ Error cancelling {timer_name}: {e}")
        
        print(f"🧹 Timer cleanup complete - cancelled {len(timers_to_cancel)} timers")
    
    def cleanup_orphaned_timers(self):
        """Periodic cleanup of orphaned timers to prevent UI freezing."""
        # Reset timer references if they're not actively needed
        if not self.recording and self.recording_timer:
            try:
                self.root.after_cancel(self.recording_timer)
                self.recording_timer = None
                print("🧹 Cleaned up orphaned recording timer")
            except:
                pass
                
        if not self.generating and self.generation_timer:
            try:
                self.root.after_cancel(self.generation_timer)
                self.generation_timer = None
                print("🧹 Cleaned up orphaned generation timer")
            except:
                pass


    def show_startup_loading(self):
        """Show a loading progress bar during startup dataset loading."""
        # Create a simple loading window
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Loading Datasets")
        self.loading_window.geometry("400x150")
        self.loading_window.configure(bg=self.colors['bg_main'])
        self.loading_window.resizable(False, False)
        
        # Center the loading window
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()
        
        # Loading content
        loading_frame = tk.Frame(self.loading_window, bg=self.colors['bg_main'])
        loading_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(loading_frame, text="🚀 Initializing Hand Control System", 
                              bg=self.colors['bg_main'], fg=self.colors['text_main'],
                              font=self.fonts['subtitle'])
        title_label.pack(pady=(0, 10))
        
        # Progress bar
        self.startup_progress_var = tk.DoubleVar()
        self.startup_progress_bar = ttk.Progressbar(loading_frame, 
                                                   variable=self.startup_progress_var,
                                                   length=350, mode='determinate')
        self.startup_progress_bar.pack(pady=(0, 10))
        
        # Status label
        self.startup_status_label = tk.Label(loading_frame, text="Scanning for datasets...", 
                                            bg=self.colors['bg_main'], fg=self.colors['text_accent'],
                                            font=("Arial", 9))
        self.startup_status_label.pack()
        
        # Force update to show the window
        self.loading_window.update()
    
    def update_startup_progress(self, progress, status_text):
        """Update the startup loading progress."""
        if hasattr(self, 'startup_progress_var'):
            self.startup_progress_var.set(progress)
            self.startup_status_label.config(text=status_text)
            self.loading_window.update()
    
    def hide_startup_loading(self):
        """Hide the startup loading window."""
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()
            delattr(self, 'loading_window')
    
    def load_all_datasets_on_startup(self):
        """Load all datasets on startup with progress indication for autonomous operation."""
        try:
            self.update_startup_progress(10, "Checking movement_recordings directory...")
            
            if not os.path.exists("movement_recordings"):
                self.update_startup_progress(100, "No datasets found - ready for recording")
                self.root.after(1000, self.hide_startup_loading)  # Hide after 1 second
                print("📁 No movement_recordings directory - system ready for fresh recordings")
                return
            
            # Get list of all dataset files
            self.update_startup_progress(20, "Scanning dataset files...")
            all_files = [f for f in os.listdir("movement_recordings") if f.endswith('.json')]
            total_files = len(all_files)
            
            if total_files == 0:
                self.update_startup_progress(100, "No datasets found - ready for recording")
                self.root.after(1000, self.hide_startup_loading)
                print("📁 No dataset files found - system ready for fresh recordings")
                return
            
            self.update_startup_progress(30, f"Loading {total_files} dataset files...")
            
            # Initialize dataset storage
            self.available_datasets = {}
            self.dataset_info = {}
            
            compatible_files = 0
            incompatible_files = 0
            
            # Process each file with progress updates
            for i, filename in enumerate(all_files):
                file_progress = 30 + int((i / total_files) * 60)  # 30% to 90%
                self.update_startup_progress(file_progress, f"Loading {filename}...")
                
                filepath = os.path.join("movement_recordings", filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Check format compatibility (same logic as refresh_datasets)
                    format_version = data.get('format_version', 'unknown')
                    markov_chain = data.get('markov_chain', {})
                    
                    is_current_format = (format_version == '2.0_servo_based' and 
                                      'servo_transitions' in markov_chain and
                                      'servo_movements' in data)
                    
                    is_compatible_old_format = (format_version == 'unknown' and
                                              'servo_transitions' in markov_chain and
                                              'movements' in data)
                    
                    is_recent_format = (format_version == 'unknown' and
                                      'movement_count' in data and
                                      'movements' in data)
                    
                    if is_current_format or is_compatible_old_format or is_recent_format:
                        emotion = data.get('emotion', 'unknown')
                        if emotion not in self.available_datasets:
                            self.available_datasets[emotion] = []
                        
                        # Extract dataset info (same logic as refresh_datasets)
                        timestamp = data.get('timestamp', 'unknown')
                        if is_current_format:
                            sample_count = data.get('movement_count', 0)
                        elif is_recent_format:
                            sample_count = data.get('movement_count', 0)
                        else:
                            sample_count = data.get('movement_count', len(data.get('movements', [])))
                        
                        duration = data.get('duration', 0)
                        unique_states = markov_chain.get('unique_states', 0)
                        custom_name = data.get('custom_name', '')
                        
                        if custom_name:
                            display_name = custom_name
                        else:
                            display_name = f"Recording-{timestamp}"
                        
                        dataset_info = {
                            'filename': filename,
                            'filepath': filepath,
                            'emotion': emotion,
                            'timestamp': timestamp,
                            'sample_count': sample_count,
                            'duration': duration,
                            'unique_states': unique_states,
                            'display_name': display_name,
                            'custom_name': custom_name,
                            'data': data
                        }
                        
                        self.available_datasets[emotion].append(dataset_info)
                        self.dataset_info[filename] = dataset_info
                        compatible_files += 1
                    else:
                        incompatible_files += 1
                        print(f"⚠️ Skipping incompatible file: {filename}")
                        
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
                    incompatible_files += 1
            
            # Sort datasets by timestamp (newest first)
            self.update_startup_progress(90, "Sorting datasets...")
            for emotion in self.available_datasets:
                try:
                    self.available_datasets[emotion].sort(key=lambda x: str(x['timestamp']), reverse=True)
                except Exception as e:
                    print(f"⚠️ Error sorting datasets for {emotion}: {e}")
                    self.available_datasets[emotion].sort(key=lambda x: x['filename'], reverse=True)
            
            # Update dataset displays for all emotions
            self.update_startup_progress(95, "Updating interface...")
            self.update_dataset_display()
            
            # Auto-select datasets for each emotion if available
            for emotion in self.emotional_states.keys():
                if emotion in self.available_datasets and self.available_datasets[emotion]:
                    # Auto-select the most recent dataset for each emotion
                    first_dataset = self.available_datasets[emotion][0]
                    self.active_datasets[emotion] = first_dataset['filename']
                    
                    # Pre-load the Markov chain for immediate use
                    try:
                        data = first_dataset['data']
                        markov_chain = data.get('markov_chain', {})
                        if markov_chain:
                            self.markov_chains[emotion] = markov_chain
                            print(f"🔗 Pre-loaded Markov chain for {emotion}: {markov_chain.get('unique_states', 0)} states")
                    except Exception as e:
                        print(f"⚠️ Error pre-loading chain for {emotion}: {e}")
            
            # Final status
            self.update_startup_progress(100, f"Ready! Loaded {compatible_files} datasets for {len(self.available_datasets)} emotions")
            
            # Auto-hide loading window after 2 seconds
            self.root.after(2000, self.hide_startup_loading)
            
            print(f"✅ STARTUP COMPLETE: Loaded {compatible_files} compatible datasets for {len(self.available_datasets)} emotions")
            if incompatible_files > 0:
                print(f"⚠️ Skipped {incompatible_files} incompatible files")
            
            print(f"🚀 System ready for autonomous operation with pre-loaded datasets!")
            
        except Exception as e:
            print(f"❌ Error during startup dataset loading: {e}")
            self.update_startup_progress(100, "Error loading datasets - manual refresh may be needed")
            self.root.after(3000, self.hide_startup_loading)


def main():
    """Main function to start the interface."""
    print("[START] Starting Clean Emotional Hand Control...")
    
    # Create and run the interface
    interface = CleanCursorInterface()
    
    def on_closing():
        """Handle application closing - cleanup timers and resources."""
        print("🧹 Application closing - cleaning up...")
        interface.cleanup_all_timers()
        interface.root.destroy()
    
    # Set up proper window close handler
    interface.root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        # Start the tkinter main loop
        interface.root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        interface.cleanup_all_timers()
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        traceback.print_exc()
        interface.cleanup_all_timers()
    finally:
        # Cleanup
        if hasattr(interface, 'hand_controller') and interface.hand_controller:
            try:
                interface.hand_controller.cleanup()
            except:
                pass
        print("[INFO] Clean shutdown complete")


if __name__ == "__main__":
    main()
