#!/usr/bin/env python3
"""
Conscious Cursor Interface
=========================

Pure consciousness-driven hand control with training mode capability.
AI emotional puppeteering as the primary control system!

Features:
- Pure consciousness mode: AI drives hand expressions directly
- Training mode: Record manual movements to teach new behaviors
- Enhanced liveliness: More dynamic and responsive cursor movement
- Real-time parameter control with immediate effect
- Movement pattern analysis and learning
- Behavioral templates for different emotional states

Author: Emotional Puppeteering System
"""

import tkinter as tk
from tkinter import ttk
import time
import math
import threading
import pyautogui
from typing import Optional

# Import our consciousness cursor system from new location
try:
    from conscious_cursor import ConsciousCursor, ConsciousnessState
    CONSCIOUSNESS_AVAILABLE = True
    print("🚀 ConsciousCursor loaded from servo_control!")
except ImportError:
    print("⚠️ ConsciousCursor not available - manual mode only")
    CONSCIOUSNESS_AVAILABLE = False

# Import movement learning system
try:
    from movement_learning import MovementLearning
    MOVEMENT_LEARNING_AVAILABLE = True
    print("🧬 Movement Learning System loaded!")
except ImportError:
    print("⚠️ Movement learning not available")
    MOVEMENT_LEARNING_AVAILABLE = False

# Import consciousness bridge for live data
try:
    from consciousness_bridge import start_consciousness_bridge, get_live_consciousness_data, is_consciousness_data_fresh
    CONSCIOUSNESS_BRIDGE_AVAILABLE = True
except ImportError:
    print("⚠️ Consciousness bridge not available - using simulated data")
    CONSCIOUSNESS_BRIDGE_AVAILABLE = False

# Import hand controller - Direct import since we're in the same directory
try:
    from hand_expression import HandExpressionController
    HAND_CONTROLLER_AVAILABLE = True
except ImportError:
    print("⚠️ Hand controller not available - simulation mode")
    HAND_CONTROLLER_AVAILABLE = False


class ConsciousCursorInterface:
    """Enhanced physics-based hand control with consciousness cursor option."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Pure Consciousness Controller - Enhanced AI Mode")
        self.root.geometry("650x450")  # Slightly taller for emotional controls
        
        # Create main canvas with scrollbar for scrollable content
        self.main_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas)
        
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
        
        # Hand controller
        self.hand_controller: Optional[HandExpressionController] = None
        self.connected = False
        
        # Training mode variables
        self.training_mode = tk.BooleanVar(value=False)
        self.recording = False
        self.recorded_movements = []
        self.current_recording_emotion = "neutral"
        self.movement_templates = {}  # Store learned behaviors
        
        # Movement Learning System - THE REVOLUTIONARY PART!
        if MOVEMENT_LEARNING_AVAILABLE:
            self.movement_learner = MovementLearning()
            self.movement_learner.load_profiles()  # Load any existing learned profiles
            print("🧬 Movement Learning System initialized - ready to learn your body language!")
        else:
            self.movement_learner = None
        
        # Control mode - PURE CONSCIOUSNESS MODE ONLY
        self.consciousness_mode = tk.BooleanVar(value=True)  # Always consciousness mode
        self.consciousness_cursor = None
        self.consciousness_state = None
        
        if CONSCIOUSNESS_AVAILABLE:
            self.consciousness_cursor = ConsciousCursor(canvas_width=563, canvas_height=304)
            self.consciousness_state = ConsciousnessState()
            
            # Start live consciousness data bridge
            if CONSCIOUSNESS_BRIDGE_AVAILABLE:
                start_consciousness_bridge()
                print("🔗 Live consciousness data bridge activated!")
            else:
                print("🎭 Using simulated consciousness data")
        
        # Physics simulation state
        self.num_fingers = 4  # 4 servos available (pins 8,9,10,11)
        self.finger_positions = [90.0] * self.num_fingers  # Current positions
        self.finger_velocities = [0.0] * self.num_fingers  # Current velocities
        self.finger_targets = [90.0] * self.num_fingers    # Target positions from cursor
        
        # Physics parameters - WORKING VALUES
        self.spring_force = tk.DoubleVar(value=500.0)     # Much stronger spring response
        self.damping = tk.DoubleVar(value=0.1)            # Lower damping for snappier response
        self.max_velocity = tk.DoubleVar(value=1000.0)    # Higher velocity limit
        self.cursor_sensitivity = tk.DoubleVar(value=3.0) # Higher sensitivity
        
        # Wave control parameters - WORKING VALUES
        self.wave_strength = tk.DoubleVar(value=2.0)      # Strong wave influence
        self.gravity_width = tk.DoubleVar(value=0.4)      # Medium gravity field width
        self.default_position = tk.DoubleVar(value=90.0)  # Center default position
        
        # Control toggles - LOCKED OVERRIDE MODE TO PREVENT MACHINE.PY TAKEOVER
        self.physics_mode = tk.BooleanVar(value=True)     # Start with physics enabled
        self.reverse_vertical = tk.BooleanVar(value=False) # Normal vertical by default
        self.override_mode = tk.BooleanVar(value=True)    # START WITH OVERRIDE ENABLED
        self.override_locked = True  # PREVENT AUTOMATIC DISABLING BY MACHINE.PY
        
        # Mouse tracking (for manual mode)
        self.mouse_x = 0.5  # Normalized cursor position (0-1)
        self.mouse_y = 0.5
        
        # Animation state
        self.running = False
        self.last_time = time.time()
        self.last_send_time = 0  # Rate limiting for Arduino communication
        # Communication settings - ENHANCED FOR RESPONSIVENESS
        self.send_interval = 0.008  # 125 Hz for ultra-responsive startle reactions (was 0.016/60Hz)
        self.position_threshold = 1.0  # 1-degree sensitivity for instant response (was 3.0)
        
        # Arduino keep-alive system to prevent 5-second timeout fallback
        self.keepalive_timer = None
        self.last_positions = [70, 70, 70, 70]  # Track last sent positions for heartbeat
        self.keepalive_interval = 2.0  # Send heartbeat every 2 seconds (Arduino timeout is 5s) - FASTER for stillness
        
        self.setup_ui()
        
        # AUTO-INITIALIZE: Conscious cursor system is now the PRIMARY DEFAULT
        if CONSCIOUSNESS_AVAILABLE:
            self.on_consciousness_mode_toggle()  # Enable consciousness control
            print("🚀 CONSCIOUS CURSOR SYSTEM ACTIVATED - This is now the primary hand control system!")
            print("📡 Old static hand expressions are now backup-only for emergency fallback")
        else:
            print("⚠️ Consciousness system unavailable - using basic fallback mode")
            
        self.start_physics_loop()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling in the interface."""
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def setup_ui(self):
        """Create the user interface."""
        # === CONNECTION FRAME ===
        conn_frame = ttk.Frame(self.scrollable_frame)
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="🔌 Connect to Hand Controller", 
                                     command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(conn_frame, text="❌ Disconnected", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # === TRAINING MODE FRAME ===
        training_frame = ttk.LabelFrame(self.scrollable_frame, text="🎓 Training Mode - Record Your Movements")
        training_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Training controls row 1
        training_row1 = tk.Frame(training_frame)
        training_row1.pack(fill=tk.X, padx=5, pady=2)
        
        self.training_cb = ttk.Checkbutton(training_row1, text="� Training Mode (Manual Control)", 
                                          variable=self.training_mode,
                                          command=self.on_training_mode_toggle)
        self.training_cb.pack(side=tk.LEFT)
        
        # Emotion selector for training
        ttk.Label(training_row1, text="Emotion:").pack(side=tk.LEFT, padx=(20, 5))
        self.emotion_var = tk.StringVar(value="neutral")
        self.emotion_combo = ttk.Combobox(training_row1, textvariable=self.emotion_var, 
                                         values=["neutral", "happy", "sad", "angry", "surprised", "focused", "excited"],
                                         width=10)
        self.emotion_combo.pack(side=tk.LEFT)
        
        # Training controls row 2
        training_row2 = tk.Frame(training_frame)
        training_row2.pack(fill=tk.X, padx=5, pady=2)
        
        self.record_btn = ttk.Button(training_row2, text="🔴 Start Recording", 
                                    command=self.toggle_recording, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT)
        
        self.save_template_btn = ttk.Button(training_row2, text="💾 Save Template", 
                                           command=self.save_movement_template, state=tk.DISABLED)
        self.save_template_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Training status with clearer workflow guidance
        self.training_status = ttk.Label(training_frame, text="📝 WORKFLOW: 1) Enable Training 2) Record movements 3) Save Template 4) Apply learned emotion", foreground="gray")
        self.training_status.pack(anchor=tk.W, padx=5, pady=2)
        
        # Add current recordings info
        self.recordings_info = ttk.Label(training_frame, text="📁 No recordings yet", foreground="gray", font=('Arial', 8))
        self.recordings_info.pack(anchor=tk.W, padx=5, pady=1)
        
        # Add learned emotion controls with clearer instructions
        emotions_frame = ttk.LabelFrame(training_frame, text="🎭 Step 4: Apply Learned Emotions to AI Cursor", padding="5")
        emotions_frame.pack(fill="x", pady=5)
        
        # Instructions
        instructions_label = ttk.Label(emotions_frame, text="After saving templates, select and apply emotions to change AI cursor behavior:", 
                                     foreground="blue", font=('Arial', 8))
        instructions_label.pack(anchor="w", pady=2)
        
        emotions_row = ttk.Frame(emotions_frame)
        emotions_row.pack(fill="x", pady=2)
        
        self.learned_emotions_var = tk.StringVar()
        self.learned_emotions_combo = ttk.Combobox(emotions_row, textvariable=self.learned_emotions_var, 
                                                  values=[], state="readonly", width=20)
        self.learned_emotions_combo.pack(side="left", padx=5)
        
        apply_emotion_btn = ttk.Button(emotions_row, text="🎭 Apply to AI Cursor", 
                                     command=self.apply_learned_emotion)
        apply_emotion_btn.pack(side="left", padx=5)
        
        refresh_emotions_btn = ttk.Button(emotions_row, text="🔄 Refresh List", 
                                        command=self.refresh_learned_emotions)
        refresh_emotions_btn.pack(side="left", padx=5)
        
        # Add management controls
        management_row = ttk.Frame(emotions_frame)
        management_row.pack(fill="x", pady=2)
        
        delete_emotion_btn = ttk.Button(management_row, text="🗑️ Delete Selected", 
                                      command=self.delete_learned_emotion)
        delete_emotion_btn.pack(side="left", padx=5)
        
        clear_all_btn = ttk.Button(management_row, text="🧹 Clear All", 
                                 command=self.clear_all_emotions)
        clear_all_btn.pack(side="left", padx=5)
        
        # Add movement playback controls
        analysis_frame = ttk.LabelFrame(training_frame, text="📊 Movement Analysis & Visualization", padding="5")
        analysis_frame.pack(fill="x", pady=5)
        
        # Analysis display area
        self.analysis_text = tk.Text(analysis_frame, height=8, width=80, font=('Courier', 9))
        self.analysis_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Analysis controls
        analysis_controls = ttk.Frame(analysis_frame)
        analysis_controls.pack(fill="x", pady=5)
        
        self.analyze_btn = ttk.Button(analysis_controls, text="🔍 Analyze Last Recording", 
                                    command=self.analyze_last_recording, state=tk.DISABLED)
        self.analyze_btn.pack(side="left", padx=5)
        
        self.clear_analysis_btn = ttk.Button(analysis_controls, text="🗑️ Clear Analysis", 
                                           command=self.clear_analysis_display)
        self.clear_analysis_btn.pack(side="left", padx=5)
        
        # Real-time recording feedback
        self.recording_feedback = ttk.Label(analysis_frame, text="📈 Recording feedback will appear here...", 
                                          foreground="gray", font=('Arial', 9))
        self.recording_feedback.pack(pady=2)
        
        # Initialize learned emotions and recordings info
        self.refresh_learned_emotions()
        self.update_recordings_info()
        
        # === HOTKEY BINDINGS ===
        self.root.bind('<KeyPress-r>', self.hotkey_toggle_recording)
        self.root.bind('<KeyPress-R>', self.hotkey_toggle_recording)
        self.root.bind('<KeyPress-t>', self.hotkey_toggle_recording)  # T for Training
        self.root.bind('<KeyPress-T>', self.hotkey_toggle_recording)
        self.root.focus_set()  # Make sure the window can receive key events
        
        # Show hotkey info
        hotkey_info = ttk.Label(training_frame, text="🎯 Hotkeys: R or T = Start/Stop Recording", 
                               foreground="blue", font=("Arial", 9))
        hotkey_info.pack(pady=2)
        
        # === CONSCIOUSNESS CONTROL FRAME ===
        if CONSCIOUSNESS_AVAILABLE:
            consciousness_frame = ttk.LabelFrame(self.scrollable_frame, text="🧠 Pure Consciousness Control")
            consciousness_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Status display - Always active in pure consciousness mode
            self.consciousness_status = ttk.Label(consciousness_frame, text="🚀 AI Consciousness Mode ACTIVE", 
                                            foreground="green")
            self.consciousness_status.pack(anchor=tk.W, padx=5, pady=2)
            
            # Consciousness data display
            self.consciousness_display = ttk.Label(consciousness_frame, text="No consciousness data", 
                                                  foreground="gray")
            self.consciousness_display.pack(anchor=tk.W, padx=5, pady=2)
        
        # === HAND CONTROL AREA ===
        control_frame = ttk.LabelFrame(self.scrollable_frame, text="🎯 Hand Control Area")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(control_frame, bg="black", height=304, width=563)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        
        # === CONTROL MODES ===
        mode_frame = ttk.LabelFrame(self.scrollable_frame, text="🎛️ Control Modes")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Physics toggle
        physics_cb = ttk.Checkbutton(mode_frame, text="⚡ Physics Mode", 
                                   variable=self.physics_mode)
        physics_cb.pack(side=tk.LEFT, padx=5)
        
        # Vertical reverse toggle
        reverse_cb = ttk.Checkbutton(mode_frame, text="🔄 Reverse Vertical", 
                                   variable=self.reverse_vertical)
        reverse_cb.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_btn = ttk.Button(mode_frame, text="🎯 Reset to Default", 
                             command=self.reset_to_default)
        reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # Startle test button
        startle_btn = ttk.Button(mode_frame, text="⚡ Test Startle", 
                               command=self.test_startle_response)
        startle_btn.pack(side=tk.RIGHT, padx=5)
        
        # === PHYSICS PARAMETERS ===
        params_frame = ttk.LabelFrame(self.scrollable_frame, text="⚙️ Physics Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Spring Force
        ttk.Label(params_frame, text="Spring Force:").grid(row=0, column=0, sticky=tk.W, padx=5)
        spring_scale = ttk.Scale(params_frame, from_=100, to=1000, variable=self.spring_force, orient=tk.HORIZONTAL)
        spring_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        spring_label = ttk.Label(params_frame, text="500")
        spring_label.grid(row=0, column=2, padx=5)
        self.spring_force.trace_add("write", lambda *args: spring_label.config(text=f"{self.spring_force.get():.0f}"))
        
        # Damping
        ttk.Label(params_frame, text="Damping:").grid(row=1, column=0, sticky=tk.W, padx=5)
        damping_scale = ttk.Scale(params_frame, from_=0.01, to=1.0, variable=self.damping, orient=tk.HORIZONTAL)
        damping_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        damping_label = ttk.Label(params_frame, text="0.1")
        damping_label.grid(row=1, column=2, padx=5)
        self.damping.trace_add("write", lambda *args: damping_label.config(text=f"{self.damping.get():.2f}"))
        
        # Max Velocity
        ttk.Label(params_frame, text="Max Velocity:").grid(row=2, column=0, sticky=tk.W, padx=5)
        velocity_scale = ttk.Scale(params_frame, from_=100, to=2000, variable=self.max_velocity, orient=tk.HORIZONTAL)
        velocity_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        velocity_label = ttk.Label(params_frame, text="1000")
        velocity_label.grid(row=2, column=2, padx=5)
        self.max_velocity.trace_add("write", lambda *args: velocity_label.config(text=f"{self.max_velocity.get():.0f}"))
        
        # Cursor Sensitivity
        ttk.Label(params_frame, text="Cursor Sensitivity:").grid(row=3, column=0, sticky=tk.W, padx=5)
        sensitivity_scale = ttk.Scale(params_frame, from_=0.5, to=10.0, variable=self.cursor_sensitivity, orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=3, column=1, sticky=tk.EW, padx=5)
        sensitivity_label = ttk.Label(params_frame, text="3.0")
        sensitivity_label.grid(row=3, column=2, padx=5)
        self.cursor_sensitivity.trace_add("write", lambda *args: sensitivity_label.config(text=f"{self.cursor_sensitivity.get():.1f}"))
        
        params_frame.columnconfigure(1, weight=1)
        
        # === WAVE CONTROL PARAMETERS ===
        wave_frame = ttk.LabelFrame(self.scrollable_frame, text="🌊 Wave Control Parameters")
        wave_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Wave Strength
        ttk.Label(wave_frame, text="Wave Strength:").grid(row=0, column=0, sticky=tk.W, padx=5)
        wave_scale = ttk.Scale(wave_frame, from_=0.0, to=5.0, variable=self.wave_strength, orient=tk.HORIZONTAL)
        wave_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        wave_label = ttk.Label(wave_frame, text="2.0")
        wave_label.grid(row=0, column=2, padx=5)
        self.wave_strength.trace_add("write", lambda *args: wave_label.config(text=f"{self.wave_strength.get():.1f}"))
        
        # Gravity Width
        ttk.Label(wave_frame, text="Gravity Width:").grid(row=1, column=0, sticky=tk.W, padx=5)
        gravity_scale = ttk.Scale(wave_frame, from_=0.1, to=1.0, variable=self.gravity_width, orient=tk.HORIZONTAL)
        gravity_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        gravity_label = ttk.Label(wave_frame, text="0.4")
        gravity_label.grid(row=1, column=2, padx=5)
        self.gravity_width.trace_add("write", lambda *args: gravity_label.config(text=f"{self.gravity_width.get():.1f}"))
        
        # Default Position
        ttk.Label(wave_frame, text="Default Position:").grid(row=2, column=0, sticky=tk.W, padx=5)
        default_scale = ttk.Scale(wave_frame, from_=0, to=180, variable=self.default_position, orient=tk.HORIZONTAL)
        default_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        default_label = ttk.Label(wave_frame, text="90")
        default_label.grid(row=2, column=2, padx=5)
        self.default_position.trace_add("write", lambda *args: default_label.config(text=f"{self.default_position.get():.0f}"))
        
        wave_frame.columnconfigure(1, weight=1)
        
        # === CONSCIOUSNESS PARAMETERS ===
        if CONSCIOUSNESS_AVAILABLE:
            self.create_consciousness_parameter_controls()
        
        # === CONTROL BUTTONS ===
        control_buttons_frame = ttk.Frame(self.scrollable_frame)
        control_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Emergency stop
        emergency_btn = ttk.Button(control_buttons_frame, text="🛑 EMERGENCY STOP", 
                                 command=self.emergency_stop)
        emergency_btn.pack(side=tk.LEFT, padx=5)
        
        # Test startle
        startle_btn = ttk.Button(control_buttons_frame, text="😲 Test Startle", 
                               command=self.test_startle)
        startle_btn.pack(side=tk.LEFT, padx=5)
        
        # Simulate consciousness data (for testing)
        if CONSCIOUSNESS_AVAILABLE:
            simulate_btn = ttk.Button(control_buttons_frame, text="🧪 Simulate Emotions", 
                                    command=self.simulate_consciousness_data)
            simulate_btn.pack(side=tk.RIGHT, padx=5)
    
    def on_consciousness_mode_toggle(self):
        """Handle switching between manual and consciousness cursor control."""
        if not CONSCIOUSNESS_AVAILABLE:
            self.consciousness_mode.set(False)
            return
        
        if self.consciousness_mode.get():
            # Switch to consciousness mode - AI consciousness controls cursor AND disable machine.py automatic updates
            self.consciousness_status.config(text="🧠 AI consciousness driving cursor", foreground="green")
            if self.consciousness_cursor:
                self.consciousness_cursor.reset_to_center()
            
            # CRITICAL: Enable both manual override AND interface override for full control
            if self.connected and self.hand_controller:
                self.hand_controller.enable_manual_override()
                self.override_mode.set(True)  # Also enable interface override
                self.on_override_toggle()  # Update the UI
                print("🎭 Switched to consciousness cursor mode - AI consciousness now controls movement!")
                print("🚫 Disabled automatic consciousness updates from machine.py to prevent conflicts")
                print("🎮 Enabled interface override - consciousness cursor now drives hand movement")
            else:
                print("🎭 Switched to consciousness cursor mode - AI consciousness control ready (hand not connected)")
        else:
            # Switch to manual mode - mouse controls cursor AND re-enable machine.py automatic updates
            self.consciousness_status.config(text="👤 Manual cursor control active", foreground="blue")
            
            # CRITICAL: Disable both overrides to fully restore machine.py control
            if self.connected and self.hand_controller:
                self.hand_controller.disable_manual_override()
                self.override_mode.set(False)  # Also disable interface override
                self.on_override_toggle()  # Update the UI
                print("👤 Switched to manual cursor mode - mouse controls movement")
                print("✅ Re-enabled automatic consciousness updates from machine.py")
                print("🤖 Interface override disabled - machine.py now controls hand")
            else:
                print("👤 Switched to manual cursor mode - mouse controls movement")
    
    def on_override_toggle(self):
        """Handle manual override toggle - PROTECTED FROM AUTOMATIC DISABLING."""
        # OVERRIDE LOCK: Prevent machine.py from automatically disabling override
        if hasattr(self, 'override_locked') and self.override_locked and not self.override_mode.get():
            print("🔒 Override lock active - preventing automatic disable by machine.py")
            self.override_mode.set(True)  # Force back to enabled
            return
        
        if self.override_mode.get():
            self.override_label.config(text="Manual Mode: ON", foreground="red")
            self.start_keepalive_heartbeat()  # Start Arduino heartbeat to prevent timeout
            print("🎮 Manual override enabled - interface controls hand directly")
            print("💓 Arduino keep-alive heartbeat started to prevent 5s timeout fallback")
            print(f"🔧 DEBUG: override_mode={self.override_mode.get()}, connected={self.connected}, hand_controller exists={self.hand_controller is not None}")
        else:
            self.override_label.config(text="Manual Mode: OFF", foreground="blue")
            self.stop_keepalive_heartbeat()  # Stop Arduino heartbeat
            print("🤖 Automatic mode - main system controls hand")
            print("💓 Arduino keep-alive heartbeat stopped - allowing fallback behavior")
            print(f"🔧 DEBUG: override_mode={self.override_mode.get()}, heartbeat stopped")
    
    def simulate_consciousness_data(self):
        """Simulate consciousness data for testing consciousness mode."""
        if not self.consciousness_state:
            return
        
        # Cycle through different emotional states
        import random
        emotions = [
            {"name": "Happy & Curious", "mood": 0.8, "novelty": 0.9, "boredom": 0.1},
            {"name": "Sad & Contemplative", "mood": -0.6, "novelty": 0.2, "boredom": 0.7},
            {"name": "Excited Discovery", "mood": 0.9, "novelty": 1.0, "boredom": 0.0},
            {"name": "Bored & Restless", "mood": -0.2, "novelty": 0.1, "boredom": 0.9},
            {"name": "Focused Attention", "mood": 0.3, "novelty": 0.6, "boredom": 0.2},
        ]
        
        emotion = random.choice(emotions)
        self.consciousness_state.mood = emotion["mood"]
        self.consciousness_state.novelty = emotion["novelty"]
        self.consciousness_state.boredom = emotion["boredom"]
        self.consciousness_state.person_present = random.choice([True, False])
        self.consciousness_state.face_confidence = random.uniform(0.3, 0.9) if self.consciousness_state.person_present else 0.0
        
        print(f"🧪 Simulating emotion: {emotion['name']}")
    
    def update_consciousness_display(self):
        """Update the consciousness data display."""
        if not self.consciousness_state or not CONSCIOUSNESS_AVAILABLE:
            return
        
        # Get emotional state description
        desc = self.consciousness_cursor.get_emotional_state_description(self.consciousness_state)
        pos = self.consciousness_cursor.get_position()
        speed = self.consciousness_cursor.get_movement_speed()
        
        # Check if using live data
        data_source = "🔴 LIVE" if (CONSCIOUSNESS_BRIDGE_AVAILABLE and is_consciousness_data_fresh()) else "🟡 SIM"
        
        display_text = f"{data_source} Pos({pos[0]:.2f}, {pos[1]:.2f}) Speed({speed:.3f}) | {desc}"
        self.consciousness_display.config(text=display_text)

    def on_training_mode_toggle(self):
        """Toggle training mode on/off."""
        if self.training_mode.get():
            # Enable training mode
            self.training_status.config(text="🎯 Step 1: TRAINING ACTIVE! Select emotion and move cursor to record patterns", foreground="green")
            self.record_btn.config(state=tk.NORMAL)
            print("🎓 Training mode activated - manual cursor control enabled for recording")
            
            # Reset any recording
            self.recording = False
            self.recorded_movements = []
            
        else:
            # Disable training mode
            self.training_status.config(text="📝 WORKFLOW: 1) Enable Training 2) Record movements 3) Save Template 4) Apply learned emotion", foreground="gray")
            self.record_btn.config(state=tk.DISABLED, text="🔴 Start Recording")
            self.save_template_btn.config(state=tk.DISABLED)
            self.recording = False
            print("🎓 Training mode deactivated")

    def toggle_recording(self):
        """Start/stop recording cursor movements."""
        if not self.training_mode.get():
            return
            
        if not self.recording:
            # Start recording
            self.recording = True
            self.recorded_movements = []
            self.recording_start_time = time.time()  # For timing analysis
            self.last_movement_time = self.recording_start_time
            self.current_recording_emotion = self.emotion_var.get()
            self.record_btn.config(text="⏹️ Stop Recording")
            self.save_template_btn.config(state=tk.DISABLED)
            self.training_status.config(text=f"🔴 Step 2: RECORDING {self.current_recording_emotion.upper()} movements...", foreground="red")
            print(f"🔴 Started recording movements for emotion: {self.current_recording_emotion}")
            
            # Start real-time feedback
            self.update_recording_feedback()
        else:
            # Stop recording
            self.recording = False
            self.record_btn.config(text="🔴 Start Recording")
            if len(self.recorded_movements) > 10:  # Only allow saving if we have enough data
                self.save_template_btn.config(state=tk.NORMAL)
                self.analyze_btn.config(state=tk.NORMAL)  # Enable analysis
                self.training_status.config(text=f"✅ Step 3: Recorded {len(self.recorded_movements)} movements for {self.current_recording_emotion} - Ready to Save Template!", foreground="blue")
                print(f"⏹️ Stopped recording. Captured {len(self.recorded_movements)} movement points")
            else:
                self.training_status.config(text="❌ Recording too short - need more movement data", foreground="red")
                print("❌ Recording too short - need at least 10 movement points")

    def save_movement_template(self):
        """Save recorded movements as a behavioral template using the learning system."""
        if not self.recorded_movements or len(self.recorded_movements) < 10:
            print("❌ No valid recording to save")
            return
        
        # Use the revolutionary learning system!
        if self.movement_learner:
            success = self.movement_learner.learn_from_recording(
                self.current_recording_emotion, 
                self.recorded_movements
            )
            
            if success:
                # Apply the learned parameters immediately for preview!
                if self.consciousness_cursor:
                    self.movement_learner.apply_learned_parameters(
                        self.consciousness_cursor, 
                        self.current_recording_emotion
                    )
                
                self.training_status.config(
                    text=f"🧬 LEARNED {self.current_recording_emotion} movement DNA! Applied to AI cursor.", 
                    foreground="green"
                )
                print(f"🚀 REVOLUTIONARY: AI has learned your '{self.current_recording_emotion}' body language!")
                print(f"🎭 The consciousness cursor now expresses {self.current_recording_emotion} using YOUR movement signature!")
                
                # Show available learned emotions
                learned_emotions = self.movement_learner.get_available_emotions()
                print(f"📚 Available learned emotions: {', '.join(learned_emotions)}")
                
                # Refresh the dropdown to show the new emotion
                self.refresh_learned_emotions()
                self.update_recordings_info()
                
                # Also save raw movements for analysis and visualization
                self.save_movement_data(self.current_recording_emotion, self.recorded_movements)
                
                # Enable analysis button
                self.analyze_btn.config(state=tk.NORMAL)
                
            else:
                self.training_status.config(text="❌ Learning failed - need more movement data", foreground="red")
        else:
            # Fallback to old template system
            template = self.analyze_movement_pattern(self.recorded_movements)
            template['emotion'] = self.current_recording_emotion
            template['sample_count'] = len(self.recorded_movements)
            self.movement_templates[self.current_recording_emotion] = template
            
            self.training_status.config(text=f"💾 Saved template for {self.current_recording_emotion}!", foreground="green")
            print(f"💾 Saved movement template for {self.current_recording_emotion}")
        
        self.save_template_btn.config(state=tk.DISABLED)

    def refresh_learned_emotions(self):
        """Refresh the list of available learned emotions."""
        if self.movement_learner:
            emotions = self.movement_learner.get_available_emotions()
            self.learned_emotions_combo.config(values=emotions)
            if emotions:
                self.learned_emotions_combo.set(emotions[0])
                print(f"🧬 Available learned emotions: {', '.join(emotions)}")
                # Update the interface to show the emotions are available
                self.learned_emotions_combo.config(state="readonly")
            else:
                print("🧬 No learned emotions yet - record some movements to teach the AI!")
                self.learned_emotions_combo.config(state="disabled")
        else:
            print("🧬 Movement learning system not available")
            self.learned_emotions_combo.config(state="disabled")
    
    def delete_learned_emotion(self):
        """Delete a selected learned emotion."""
        if not self.movement_learner:
            print("❌ Movement learning system not available")
            return
        
        emotion = self.learned_emotions_var.get()
        if not emotion:
            print("❌ No emotion selected to delete")
            return
        
        # Confirm deletion
        import tkinter.messagebox as msgbox
        if msgbox.askyesno("Confirm Deletion", f"Delete the learned emotion '{emotion}'?\nThis cannot be undone."):
            success = self.movement_learner.delete_emotion(emotion)
            if success:
                print(f"🗑️ Deleted learned emotion: {emotion}")
                self.refresh_learned_emotions()
                self.update_recordings_info()
            else:
                print(f"❌ Failed to delete emotion: {emotion}")

    def clear_all_emotions(self):
        """Clear all learned emotions."""
        if not self.movement_learner:
            print("❌ Movement learning system not available")
            return
        
        # Confirm deletion
        import tkinter.messagebox as msgbox
        if msgbox.askyesno("Confirm Clear All", "Delete ALL learned emotions?\nThis cannot be undone."):
            success = self.movement_learner.clear_all_profiles()
            if success:
                print("🧹 Cleared all learned emotions")
                self.refresh_learned_emotions()
                self.update_recordings_info()
            else:
                print("❌ Failed to clear emotions")

    def update_recordings_info(self):
        """Update the recordings information display."""
        try:
            import os
            recordings_dir = "movement_recordings"
            learned_count = len(self.movement_learner.get_available_emotions()) if self.movement_learner else 0
            
            # Count raw recording files
            raw_count = 0
            if os.path.exists(recordings_dir):
                raw_count = len([f for f in os.listdir(recordings_dir) if f.endswith('.json')])
            
            info_text = f"📁 {learned_count} learned emotions, {raw_count} raw recordings"
            self.recordings_info.config(text=info_text)
            
        except Exception as e:
            self.recordings_info.config(text="📁 Recording info unavailable")
            print(f"⚠️ Error updating recordings info: {e}")

    def apply_learned_emotion(self):
        """Apply a learned emotional signature to the consciousness cursor."""
        if not self.movement_learner or not self.consciousness_cursor:
            print("❌ Movement learner or consciousness cursor not available")
            return
        
        emotion = self.learned_emotions_var.get()
        if not emotion:
            print("❌ No emotion selected")
            return
        
        success = self.movement_learner.apply_learned_parameters(self.consciousness_cursor, emotion)
        if success:
            print(f"🎭 Applied {emotion} movement signature to consciousness cursor!")
            print(f"🚀 The AI is now expressing {emotion} using YOUR learned movement patterns!")
            self.training_status.config(
                text=f"🎭 AI expressing {emotion} using your learned patterns", 
                foreground="purple"
            )
        else:
            print(f"❌ Failed to apply {emotion} - emotion not learned yet")
            self.training_status.config(text="❌ Emotion not found in learned patterns", foreground="red")

    def hotkey_toggle_recording(self, event):
        """Toggle recording with hotkey (R or T)"""
        # Only respond to hotkeys if training mode is enabled
        if not self.training_mode.get():
            print("🎯 Training mode disabled - enable it first to use recording hotkeys")
            return
        
        if self.recording:
            # Stop recording by triggering the button
            self.toggle_recording()
            print("🎯 HOTKEY: Stopped recording")
        else:
            # Start recording (but only if emotion is selected)
            emotion = self.emotion_var.get()
            if not emotion or emotion == "Select emotion...":
                print("🎯 HOTKEY: Select an emotion first!")
                # Flash the emotion dropdown to draw attention
                original_bg = self.emotion_combo.cget("background")
                self.emotion_combo.config(background="yellow")
                self.root.after(500, lambda: self.emotion_combo.config(background=original_bg))
                return
            
            self.toggle_recording()
            print(f"🎯 HOTKEY: Started recording for {emotion}")
        
        return "break"  # Prevent the event from propagating

    def save_movement_data(self, emotion, movements):
        """Save movement data with rich timing and analysis information"""
        import json
        import os
        
        # Create recordings directory if it doesn't exist
        recordings_dir = "movement_recordings"
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)
        
        # Calculate movement characteristics for storage
        analysis = self.analyze_movement_characteristics(movements)
        
        # Save comprehensive movement data
        filename = f"{recordings_dir}/{emotion}_detailed.json"
        movement_data = {
            'emotion': emotion,
            'timestamp': time.time(),
            'movements': movements,
            'count': len(movements),
            'analysis': analysis,
            'duration': movements[-1]['timestamp'] - movements[0]['timestamp'] if len(movements) > 1 else 0
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(movement_data, f, indent=2)
            print(f"💾 Saved detailed movement data for {emotion} ({len(movements)} points)")
        except Exception as e:
            print(f"❌ Error saving movement data: {e}")

    def analyze_movement_characteristics(self, movements):
        """Analyze movements to extract meaningful characteristics for cursor behavior"""
        if len(movements) < 2:
            return {
                'avg_speed': 0, 'speed_variance': 0, 'max_speed': 0, 'min_speed': 0,
                'pause_ratio': 0, 'long_pauses_count': 0, 'direction_changes': 0,
                'burst_movements': 0, 'micro_movements': 0, 'total_distance': 0,
                'movement_efficiency': 0, 'avg_time_delta': 0.1
            }
        
        # Convert to simple coordinates for analysis
        positions = [(m['x'], m['y']) for m in movements]
        
        # Handle time_deltas with robust fallback for missing data
        time_deltas = []
        for i in range(1, len(movements)):
            if 'time_delta' in movements[i] and movements[i]['time_delta'] > 0:
                time_deltas.append(movements[i]['time_delta'])
            elif 'time' in movements[i] and 'time' in movements[i-1]:
                time_diff = movements[i]['time'] - movements[i-1]['time']
                time_deltas.append(max(time_diff, 0.001))  # Minimum 1ms
            elif 'timestamp' in movements[i] and 'timestamp' in movements[i-1]:
                time_diff = movements[i]['timestamp'] - movements[i-1]['timestamp']
                time_deltas.append(max(time_diff, 0.001))  # Minimum 1ms
            else:
                # Fallback: estimate based on typical mouse movement frequency
                time_deltas.append(0.016)  # ~60 FPS default
        
        # Calculate distances between points
        distances = []
        for i in range(len(positions) - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            distances.append(math.sqrt(dx*dx + dy*dy))
        
        # Speed analysis (pixels per second, normalized coordinates)
        speeds = []
        for i, distance in enumerate(distances):
            if i < len(time_deltas) and time_deltas[i] > 0:
                speed = distance / time_deltas[i]
                speeds.append(speed)
        
        if not speeds:
            speeds = [0]
        
        avg_speed = sum(speeds) / len(speeds)
        speed_variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
        
        # Pause detection (time deltas > threshold)
        long_pauses = [t for t in time_deltas if t > 0.1]  # Pauses > 100ms
        pause_ratio = len(long_pauses) / len(time_deltas) if time_deltas else 0
        
        # Direction changes
        direction_changes = 0
        if len(positions) > 2:
            for i in range(len(positions) - 2):
                dx1 = positions[i+1][0] - positions[i][0]
                dy1 = positions[i+1][1] - positions[i][1]
                dx2 = positions[i+2][0] - positions[i+1][0]
                dy2 = positions[i+2][1] - positions[i+1][1]
                
                # Calculate angle change
                angle1 = math.atan2(dy1, dx1)
                angle2 = math.atan2(dy2, dx2)
                angle_diff = abs(angle2 - angle1)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                if angle_diff > math.pi / 4:  # More than 45 degrees
                    direction_changes += 1
        
        # Burst detection (sudden speed increases)
        burst_movements = 0
        if len(speeds) > 1:
            for i in range(1, len(speeds)):
                if speeds[i] > speeds[i-1] * 2:  # Speed doubled
                    burst_movements += 1
        
        # Micro-jitter (small rapid movements)
        micro_movements = sum(1 for d in distances if 0 < d < 5)  # Very small movements
        
        return {
            'avg_speed': avg_speed,
            'speed_variance': speed_variance,
            'max_speed': max(speeds) if speeds else 0,
            'min_speed': min(speeds) if speeds else 0,
            'pause_ratio': pause_ratio,
            'long_pauses_count': len(long_pauses),
            'direction_changes': direction_changes,
            'burst_movements': burst_movements,
            'micro_movements': micro_movements,
            'total_distance': sum(distances),
            'movement_efficiency': sum(distances) / len(positions) if positions else 0,
            'avg_time_delta': sum(time_deltas) / len(time_deltas) if time_deltas else 0
        }

    def update_recording_feedback(self):
        """Update real-time recording feedback"""
        if hasattr(self, 'recording') and self.recording and hasattr(self, 'recorded_movements'):
            count = len(self.recorded_movements)
            duration = time.time() - self.recording_start_time if hasattr(self, 'recording_start_time') else 0
            
            # Basic real-time stats
            if count > 1:
                recent_movements = self.recorded_movements[-min(10, count):]  # Last 10 movements
                recent_speeds = []
                for i in range(1, len(recent_movements)):
                    dx = recent_movements[i]['x'] - recent_movements[i-1]['x']
                    dy = recent_movements[i]['y'] - recent_movements[i-1]['y']
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Handle missing time_delta gracefully
                    if 'time_delta' in recent_movements[i]:
                        time_delta = recent_movements[i]['time_delta']
                    else:
                        # Fallback: calculate from timestamps
                        if 'time' in recent_movements[i] and 'time' in recent_movements[i-1]:
                            time_delta = recent_movements[i]['time'] - recent_movements[i-1]['time']
                        else:
                            time_delta = 0.1  # Default fallback
                    
                    if time_delta > 0:
                        recent_speeds.append(distance / time_delta)
                
                avg_recent_speed = sum(recent_speeds) / len(recent_speeds) if recent_speeds else 0
                
                feedback_text = f"📊 Recording: {count} points | {duration:.1f}s | Speed: {avg_recent_speed:.1f}px/s"
            else:
                feedback_text = f"📊 Recording: {count} points | {duration:.1f}s | Move cursor to begin..."
            
            self.recording_feedback.config(text=feedback_text)
            
            # Schedule next update
            self.root.after(100, self.update_recording_feedback)  # Update every 100ms

    def analyze_last_recording(self):
        """Analyze and display the last recorded movements"""
        # Try to get recording data from multiple sources
        movements_to_analyze = None
        emotion = 'unknown'
        
        # First try current recording
        if hasattr(self, 'recorded_movements') and self.recorded_movements:
            movements_to_analyze = self.recorded_movements
            emotion = getattr(self, 'current_recording_emotion', 'neutral')
            print(f"🔍 Analyzing current recording: {len(movements_to_analyze)} movements")
        else:
            # Try to load from saved file
            try:
                import json
                import os
                recordings_dir = "movement_recordings"
                if os.path.exists(recordings_dir):
                    # Get the most recent recording file
                    files = [f for f in os.listdir(recordings_dir) if f.endswith('.json')]
                    if files:
                        # Use the most recent file
                        latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)))
                        with open(os.path.join(recordings_dir, latest_file), 'r') as f:
                            data = json.load(f)
                            movements_to_analyze = data.get('movements', [])
                            emotion = data.get('emotion', 'unknown')
                            print(f"🔍 Loaded recording from {latest_file}: {len(movements_to_analyze)} movements")
            except Exception as e:
                print(f"❌ Error loading recording file: {e}")
        
        if not movements_to_analyze:
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, "No recording data to analyze.\nRecord some movements first, then try again.\n")
            return
        
        # Clear previous analysis
        self.analysis_text.delete(1.0, tk.END)
        
        # Perform comprehensive analysis
        analysis = self.analyze_movement_characteristics(movements_to_analyze)
        
        print(f"🔍 Analysis complete: {len(movements_to_analyze)} movements for {emotion}")
        print(f"📊 Analysis keys: {list(analysis.keys())}")
        
        # Update current recording for future use
        if not hasattr(self, 'recorded_movements') or not self.recorded_movements:
            self.recorded_movements = movements_to_analyze
            self.current_recording_emotion = emotion
        
        # Create detailed analysis report
        report = f"""
🧬 MOVEMENT ANALYSIS REPORT for '{emotion.upper()}'
{'='*60}

📊 BASIC STATS:
   • Total Points: {len(self.recorded_movements)}
   • Duration: {analysis.get('avg_time_delta', 0) * len(self.recorded_movements):.2f} seconds
   • Total Distance: {analysis.get('total_distance', 0):.1f} pixels

🏃 SPEED CHARACTERISTICS:
   • Average Speed: {analysis.get('avg_speed', 0):.1f} px/s
   • Max Speed: {analysis.get('max_speed', 0):.1f} px/s
   • Min Speed: {analysis.get('min_speed', 0):.1f} px/s
   • Speed Variance: {analysis.get('speed_variance', 0):.1f}

⏸️ TIMING PATTERNS:
   • Pause Ratio: {analysis.get('pause_ratio', 0):.1%}
   • Long Pauses: {analysis.get('long_pauses_count', 0)} (>100ms)
   • Avg Time Between Points: {analysis.get('avg_time_delta', 0)*1000:.1f}ms

🎯 MOVEMENT PATTERNS:
   • Direction Changes: {analysis.get('direction_changes', 0)}
   • Burst Movements: {analysis.get('burst_movements', 0)}
   • Micro Movements: {analysis.get('micro_movements', 0)} (<5px)
   • Movement Efficiency: {analysis.get('movement_efficiency', 0):.2f}

🧠 CONSCIOUSNESS CURSOR MAPPING:
   • Base Speed → {analysis.get('avg_speed', 0) / 100:.3f}
   • Chaos Level → {analysis.get('speed_variance', 0) / 1000:.3f}
   • Pause Probability → {analysis.get('pause_ratio', 0):.3f}
   • Burst Chance → {min(analysis.get('burst_movements', 0) / 10, 1.0):.3f}
   • Jitter Level → {min(analysis.get('micro_movements', 0) / 20, 1.0):.3f}

🎭 EMOTIONAL SIGNATURE:
   Movement Style: {'AGGRESSIVE' if analysis.get('avg_speed', 0) > 200 else
                   'ENERGETIC' if analysis.get('avg_speed', 0) > 100 else
                   'CALM' if analysis.get('avg_speed', 0) > 50 else 'GENTLE'}
   
   Rhythm: {'ERRATIC' if analysis.get('speed_variance', 0) > 5000 else
           'VARIED' if analysis.get('speed_variance', 0) > 1000 else 'STEADY'}
   
   Focus: {'SCATTERED' if analysis.get('direction_changes', 0) > len(self.recorded_movements) * 0.3 else
          'WANDERING' if analysis.get('direction_changes', 0) > len(self.recorded_movements) * 0.1 else 'FOCUSED'}

{'='*60}
Ready to apply these characteristics to the consciousness cursor! 🚀
        """
        
        self.analysis_text.insert(tk.END, report)
        
        print(f"🔍 Analysis complete for {emotion} - {len(self.recorded_movements)} movements analyzed")

    def clear_analysis_display(self):
        """Clear the analysis display"""
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Analysis cleared. Record movements and click 'Analyze Last Recording' to see detailed analysis.\n")

    def analyze_movement_pattern(self, movements):
        """Analyze recorded movements to extract behavioral parameters."""
        if len(movements) < 2:
            return {}
            
        # Calculate movement statistics
        speeds = []
        direction_changes = 0
        pauses = 0
        total_distance = 0
        
        prev_x, prev_y = movements[0]['x'], movements[0]['y']
        prev_direction = None
        
        for i in range(1, len(movements)):
            curr_x, curr_y = movements[i]['x'], movements[i]['y']
            
            # Calculate speed (distance per time unit)
            distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
            time_delta = movements[i]['time'] - movements[i-1]['time']
            speed = distance / max(time_delta, 0.001)  # Avoid division by zero
            
            speeds.append(speed)
            total_distance += distance
            
            # Detect direction changes
            if distance > 0.01:  # Only count significant movements
                direction = math.atan2(curr_y - prev_y, curr_x - prev_x)
                if prev_direction is not None:
                    angle_diff = abs(direction - prev_direction)
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    if angle_diff > math.pi / 4:  # 45 degree threshold
                        direction_changes += 1
                prev_direction = direction
            else:
                pauses += 1
                
            prev_x, prev_y = curr_x, curr_y
        
        # Calculate template parameters
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        pause_ratio = pauses / len(movements)
        change_rate = direction_changes / len(movements)
        
        return {
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'direction_changes': direction_changes,
            'change_rate': change_rate,
            'pause_ratio': pause_ratio,
            'total_distance': total_distance,
            'duration': movements[-1]['time'] - movements[0]['time'] if len(movements) > 1 else 0
        }
    
    def on_mouse_move(self, event):
        """Handle mouse movement over the canvas - training mode and manual control."""
        canvas_width = 563  # Fixed width - no dynamic sizing
        canvas_height = 304  # Fixed height - no dynamic sizing
        
        if canvas_width > 1 and canvas_height > 1:
            x = event.x / canvas_width
            y = event.y / canvas_height
            
            # Keep within bounds
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            
            # Record movement if in training mode
            if self.training_mode.get() and self.recording:
                current_time = time.time()
                # Calculate time delta from previous movement
                if not hasattr(self, 'last_recording_time'):
                    self.last_recording_time = current_time
                    time_delta = 0.0
                else:
                    time_delta = current_time - self.last_recording_time
                    self.last_recording_time = current_time
                
                movement_point = {
                    'x': x,
                    'y': y,
                    'time': current_time,
                    'timestamp': current_time,  # For compatibility
                    'time_delta': time_delta,   # CRITICAL: Add this for analysis
                    'emotion': self.current_recording_emotion
                }
                self.recorded_movements.append(movement_point)
                
                # Update recording status
                if len(self.recorded_movements) % 50 == 0:  # Update every 50 points
                    self.training_status.config(text=f"🔴 RECORDING {self.current_recording_emotion.upper()} - {len(self.recorded_movements)} points captured", foreground="red")
            
            # Update mouse position for manual control (when not in pure consciousness mode)
            if self.training_mode.get():  # Training mode uses manual control
                self.mouse_x = x
                self.mouse_y = y
    
    def on_mouse_click(self, event):
        """Handle mouse clicks."""
        print(f"🖱️ Click at ({event.x}, {event.y})")
        if self.consciousness_mode.get() and self.consciousness_cursor:
            # In consciousness mode, clicking resets cursor to center
            self.consciousness_cursor.reset_to_center()
    
    def get_current_cursor_position(self):
        """Get current cursor position based on active mode."""
        if self.training_mode.get():
            # Training mode: Use manual mouse control
            cursor_x, cursor_y = (self.mouse_x, self.mouse_y)
        elif self.consciousness_mode.get() and self.consciousness_cursor:
            # Consciousness mode: Use AI-driven cursor
            cursor_x, cursor_y = self.consciousness_cursor.get_position()
        else:
            # Fallback: Manual control
            cursor_x, cursor_y = (self.mouse_x, self.mouse_y)
        
        # Track cursor position for startle tests
        self.last_cursor_x = cursor_x
        self.last_cursor_y = cursor_y
        
        # Record movements if recording is active
        if hasattr(self, 'recording') and self.recording:
            current_time = time.time()
            time_delta = current_time - self.last_movement_time
            
            # Record with timing information
            movement_data = {
                'x': cursor_x,
                'y': cursor_y,
                'timestamp': current_time,
                'time_delta': time_delta
            }
            self.recorded_movements.append(movement_data)
            self.last_movement_time = current_time
        
        return (cursor_x, cursor_y)
    
    def toggle_connection(self):
        """Toggle connection to hand controller."""
        if not HAND_CONTROLLER_AVAILABLE:
            print("⚠️ Hand controller not available - simulation mode")
            return
        
        if not self.connected:
            try:
                # Attempt connection - Direct import since we're in the same directory
                from hand_expression import HandExpressionController
                self.hand_controller = HandExpressionController(
                    port="COM3",  # Adjust as needed
                    baudrate=9600,
                    clean_output=True
                )
                self.connected = True
                self.connect_btn.config(text="🔌 Disconnect")
                self.status_label.config(text="✅ Connected", foreground="green")
                print("✅ Connected to hand controller")
                
                # CRITICAL: Start heartbeat if override mode is already enabled
                if self.override_mode.get():
                    print("🔧 Starting heartbeat after connection (override mode already active)")
                    self.start_keepalive_heartbeat()
                
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                self.status_label.config(text=f"❌ Connection failed: {e}", foreground="red")
        else:
            # Disconnect
            if self.hand_controller:
                self.hand_controller.cleanup()
                self.hand_controller = None
            self.connected = False
            self.connect_btn.config(text="🔌 Connect to Hand Controller")
            self.status_label.config(text="❌ Disconnected", foreground="red")
            print("🔌 Disconnected from hand controller")
    
    def emergency_stop(self):
        """Emergency stop - reset everything to default."""
        print("🛑 EMERGENCY STOP!")
        self.reset_to_default()
        if self.consciousness_cursor:
            self.consciousness_cursor.reset_to_center()
        self.consciousness_mode.set(False)
        self.on_consciousness_mode_toggle()
    
    def test_startle(self):
        """Test startle reaction."""
        if self.consciousness_state:
            self.consciousness_state.startle_triggered = True
            self.consciousness_state.startle_time = time.time()
            print("😲 Startle triggered!")
            
            # Reset startle after a moment
            def reset_startle():
                time.sleep(0.5)
                if self.consciousness_state:
                    self.consciousness_state.startle_triggered = False
            
            threading.Thread(target=reset_startle, daemon=True).start()
    
    def reset_to_default(self):
        """Reset all finger positions to default."""
        default_pos = self.default_position.get()
        self.finger_positions = [default_pos] * self.num_fingers
        self.finger_velocities = [0.0] * self.num_fingers
        self.finger_targets = [default_pos] * self.num_fingers
        print(f"🎯 Reset to default position: {default_pos}")
    
    def test_startle_response(self):
        """Test immediate startle response - bypasses physics for instant reaction."""
        # Get current cursor position
        cursor_x = getattr(self, 'last_cursor_x', 0.5)
        cursor_y = getattr(self, 'last_cursor_y', 0.5)
        
        # Trigger maximum startle (novelty = 1.0)
        self.apply_startle_override(1.0, cursor_x, cursor_y)
        print("⚡ STARTLE TEST - Immediate response triggered!")
        
        # Visual feedback
        if hasattr(self, 'consciousness_display'):
            original_text = self.consciousness_display.cget("text")
            self.consciousness_display.config(text="⚡ STARTLE TRIGGERED!", foreground="red")
            # Reset display after 1 second
            self.root.after(1000, lambda: self.consciousness_display.config(
                text=original_text, foreground="gray"))

    def start_keepalive_heartbeat(self):
        """Start the Arduino keep-alive heartbeat to prevent 5-second timeout fallback."""
        print(f"🔧 DEBUG: start_keepalive_heartbeat called - connected={self.connected}, hand_controller exists={self.hand_controller is not None}")
        
        if self.keepalive_timer:
            self.root.after_cancel(self.keepalive_timer)
        
        def send_heartbeat():
            print(f"🔧 DEBUG: heartbeat function called - override_mode={self.override_mode.get()}")
            if self.connected and self.hand_controller and self.override_mode.get():
                # Send current positions as heartbeat to keep Arduino in consciousnessMode
                # Use dedicated heartbeat method that bypasses ALL throttling
                try:
                    self.hand_controller._send_heartbeat_command({
                        f"finger{i}": int(self.last_positions[i]) for i in range(4)
                    })
                    print(f"💓 Arduino heartbeat sent: {[int(p) for p in self.last_positions]} (bypasses throttling)")
                except Exception as e:
                    print(f"⚠️ Heartbeat failed: {e}")
            else:
                print(f"🔧 DEBUG: heartbeat skipped - connected={self.connected}, hand_controller exists={self.hand_controller is not None}, override_mode={self.override_mode.get()}")
            
            # Schedule next heartbeat
            if self.override_mode.get():  # Only continue if still in override mode
                self.keepalive_timer = self.root.after(int(self.keepalive_interval * 1000), send_heartbeat)
            else:
                print("🔧 DEBUG: heartbeat stopped - override mode disabled")
        
        # Start the heartbeat cycle
        self.keepalive_timer = self.root.after(int(self.keepalive_interval * 1000), send_heartbeat)
        print(f"💓 Arduino keep-alive heartbeat started ({self.keepalive_interval}s interval)")

    def stop_keepalive_heartbeat(self):
        """Stop the Arduino keep-alive heartbeat."""
        if self.keepalive_timer:
            self.root.after_cancel(self.keepalive_timer)
            self.keepalive_timer = None
            print("💓 Arduino keep-alive heartbeat stopped")
    
    def start_physics_loop(self):
        """Start the physics simulation loop."""
        self.running = True
        self.physics_loop()
    
    def physics_loop(self):
        """Main physics simulation loop."""
        if not self.running:
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Update consciousness cursor if in consciousness mode (and not in training mode)
        if self.consciousness_mode.get() and not self.training_mode.get() and self.consciousness_cursor and self.consciousness_state:
            # Get live consciousness data if available
            if CONSCIOUSNESS_BRIDGE_AVAILABLE and is_consciousness_data_fresh():
                live_data = get_live_consciousness_data()
                # Update consciousness state with live data
                self.consciousness_state.mood = live_data['mood']
                self.consciousness_state.novelty = live_data['novelty']
                self.consciousness_state.boredom = live_data['boredom']
                self.consciousness_state.person_present = live_data['person_present']
                self.consciousness_state.face_confidence = live_data['face_confidence']
                self.consciousness_state.breathing_phase = live_data['breathing_phase']
                self.consciousness_state.gaze_pan = live_data['gaze_pan']
                self.consciousness_state.gaze_tilt = live_data['gaze_tilt']
            
            # Update cursor with consciousness data
            self.consciousness_cursor.update(self.consciousness_state, dt)
            self.update_consciousness_display()
        
        # Get current cursor position
        cursor_x, cursor_y = self.get_current_cursor_position()
        
        # Physics simulation with startle override
        if self.physics_mode.get():
            # Check for startle override first
            if self.consciousness_mode.get() and hasattr(self, 'consciousness_cursor'):
                consciousness = self.consciousness_bridge.get_current_consciousness() if hasattr(self, 'consciousness_bridge') else None
                if consciousness and consciousness.novelty > 0.7:  # High novelty = startle
                    # IMMEDIATE startle response - bypass physics entirely
                    self.apply_startle_override(consciousness.novelty, cursor_x, cursor_y)
                else:
                    # Normal physics simulation
                    self.update_physics(cursor_x, cursor_y, dt)
            else:
                # Normal physics simulation
                self.update_physics(cursor_x, cursor_y, dt)
        else:
            # Direct control mode
            self.update_direct(cursor_x, cursor_y)
        
        # Send to controller if connected and override mode is on
        if self.connected and self.hand_controller and self.override_mode.get():
            self.send_to_controller()
        
        # Update visualization
        self.update_canvas(cursor_x, cursor_y)
        
        # Schedule next update
        self.root.after(16, self.physics_loop)  # ~60 FPS
    
    def update_physics(self, cursor_x, cursor_y, dt):
        """Update physics simulation (wave-based finger control)."""
        # This is your working physics code - unchanged!
        canvas_width = 563  # Fixed width - no dynamic sizing
        canvas_height = 304  # Fixed height - no dynamic sizing
        
        # Apply vertical reverse if enabled (affects finger calculation, not cursor)
        if self.reverse_vertical.get():
            effective_cursor_y = 1.0 - cursor_y
        else:
            effective_cursor_y = cursor_y
        
        # Calculate wave-based targets for each finger
        for i in range(self.num_fingers):
            finger_x = (i + 0.5) / self.num_fingers
            
            # Distance from cursor to finger position
            dx = cursor_x - finger_x
            dy = effective_cursor_y - 0.5
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Gravitational wave influence
            wave_influence = math.exp(-distance / self.gravity_width.get()) * self.wave_strength.get()
            
            # Calculate target position
            base_position = self.default_position.get()
            cursor_influence = (effective_cursor_y - 0.5) * 180 * self.cursor_sensitivity.get()
            wave_effect = wave_influence * cursor_influence
            
            target_position = base_position + wave_effect
            target_position = max(0, min(180, target_position))
            
            self.finger_targets[i] = target_position
            
            # Spring-damper physics
            position_error = target_position - self.finger_positions[i]
            spring_force = position_error * self.spring_force.get()
            damping_force = -self.finger_velocities[i] * self.damping.get()
            
            total_force = spring_force + damping_force
            acceleration = total_force / 10.0  # Mass factor
            
            # Update velocity with limits
            self.finger_velocities[i] += acceleration * dt
            max_vel = self.max_velocity.get()
            self.finger_velocities[i] = max(-max_vel, min(max_vel, self.finger_velocities[i]))
            
            # Update position
            self.finger_positions[i] += self.finger_velocities[i] * dt
            self.finger_positions[i] = max(0, min(180, self.finger_positions[i]))
    
    def update_direct(self, cursor_x, cursor_y):
        """Update direct control mode - FIXED FOR TANDEM MOVEMENT."""
        # Use the SAME wave-based calculation as physics mode for tandem movement
        canvas_width = 563  # Fixed width - no dynamic sizing
        canvas_height = 304  # Fixed height - no dynamic sizing
        
        # Apply vertical reverse if enabled (affects finger calculation, not cursor)
        if self.reverse_vertical.get():
            effective_cursor_y = 1.0 - cursor_y
        else:
            effective_cursor_y = cursor_y
        
        effective_cursor_x = cursor_x
        
        # CRITICAL: Use the SAME wave calculation as physics mode
        cursor_canvas_x = effective_cursor_x * canvas_width
        cursor_canvas_y = effective_cursor_y * canvas_height
        
        # Calculate finger positions using wave pattern (same as physics mode)
        for i in range(self.num_fingers):
            finger_x = (i + 0.5) / self.num_fingers * canvas_width
            
            # Distance from cursor to finger
            distance = abs(cursor_canvas_x - finger_x)
            gravity_width_pixels = self.gravity_width.get() * canvas_width
            
            if distance < gravity_width_pixels:
                # Within gravity field - apply wave influence
                wave_influence = (1.0 - distance / gravity_width_pixels) * self.wave_strength.get()
                cursor_influence = (effective_cursor_y - 0.5) * 180 * wave_influence
                self.finger_targets[i] = self.default_position.get() + cursor_influence
            else:
                # Outside gravity field - return to default
                self.finger_targets[i] = self.default_position.get()
            
            # Clamp to valid range
            self.finger_targets[i] = max(0, min(180, self.finger_targets[i]))
            
            # DIRECT MODE: Set position immediately (no physics smoothing)
            self.finger_positions[i] = self.finger_targets[i]
            self.finger_velocities[i] = 0.0
    
    def apply_startle_override(self, novelty_level, cursor_x, cursor_y):
        """
        IMMEDIATE startle response - bypasses physics entirely for instant reaction.
        This creates the fast, visible startle you wanted to see!
        """
        # Calculate startle intensity (0.7-1.0 novelty maps to 0-1 startle)
        startle_intensity = min(1.0, (novelty_level - 0.7) / 0.3)
        
        # Base position for calculation
        base_position = self.default_position.get()
        
        # Apply vertical reverse if enabled
        if self.reverse_vertical.get():
            effective_cursor_y = 1.0 - cursor_y
        else:
            effective_cursor_y = cursor_y
        
        # Create dramatic startle pattern - more extreme than normal physics
        for i in range(self.num_fingers):
            finger_x = (i + 0.5) / self.num_fingers
            
            # Distance from cursor
            dx = cursor_x - finger_x
            dy = effective_cursor_y - 0.5
            distance = math.sqrt(dx*dx + dy*dy)
            
            # IMMEDIATE high-intensity response (no spring damping)
            startle_wave = math.exp(-distance / self.gravity_width.get()) * startle_intensity
            
            # Exaggerated cursor influence for startle
            cursor_influence = (effective_cursor_y - 0.5) * 180 * self.cursor_sensitivity.get() * 1.5
            startle_effect = startle_wave * cursor_influence * 1.2  # 20% more dramatic
            
            # DIRECT position setting - no physics smoothing
            target_position = base_position + startle_effect
            target_position = max(0, min(180, target_position))
            
            # Immediately set position (bypass physics entirely)
            self.finger_positions[i] = target_position
            self.finger_targets[i] = target_position
            # Reset velocity to make future physics transitions smoother
            self.finger_velocities[i] = 0.0
    
    def send_to_controller(self):
        """Send finger positions to hand controller with rate limiting."""
        current_time = time.time()
        if current_time - self.last_send_time < self.send_interval:
            return
        
        # Check if positions have changed significantly
        position_changed = False
        for i, pos in enumerate(self.finger_positions):
            if not hasattr(self, '_last_sent_positions'):
                self._last_sent_positions = [0] * self.num_fingers
            
            if abs(pos - self._last_sent_positions[i]) > 3.0:  # 3 degree threshold
                position_changed = True
                break
        
        if not position_changed:
            return
        
        try:
            self.hand_controller.set_hand_positions(self.finger_positions)
            self._last_sent_positions = self.finger_positions.copy()
            self.last_positions = self.finger_positions.copy()  # Track for heartbeat system
            self.last_send_time = current_time
        except Exception as e:
            print(f"❌ Send error: {e}")
    
    def update_canvas(self, cursor_x, cursor_y):
        """Update the visual canvas."""
        self.canvas.delete("all")
        
        # Use FIXED canvas dimensions to prevent window resizing
        canvas_width = 563  # Fixed width - no dynamic sizing
        canvas_height = 304  # Fixed height - no dynamic sizing
        
        # Draw cursor position
        cursor_canvas_x = cursor_x * canvas_width
        cursor_canvas_y = cursor_y * canvas_height
        
        # Different cursor styles for different modes
        if self.consciousness_mode.get():
            # Pulsing consciousness cursor
            pulse = math.sin(time.time() * 3) * 0.3 + 0.7
            size = 15 * pulse
            color = "#ff6b6b"  # Consciousness red
            self.canvas.create_oval(cursor_canvas_x - size/2, cursor_canvas_y - size/2,
                                  cursor_canvas_x + size/2, cursor_canvas_y + size/2,
                                  fill=color, outline="white", width=2)
            self.canvas.create_text(cursor_canvas_x, cursor_canvas_y - 25, text="🧠",
                                  fill="white", font=("Arial", 12))
        else:
            # Standard manual cursor
            self.canvas.create_line(cursor_canvas_x - 10, cursor_canvas_y,
                                  cursor_canvas_x + 10, cursor_canvas_y,
                                  fill="yellow", width=3)
            self.canvas.create_line(cursor_canvas_x, cursor_canvas_y - 10,
                                  cursor_canvas_x, cursor_canvas_y + 10,
                                  fill="yellow", width=3)
        
        # Draw finger positions
        for i in range(self.num_fingers):
            finger_x = (i + 0.5) / self.num_fingers * canvas_width
            finger_height = (self.finger_positions[i] / 180.0) * canvas_height
            
            # Color based on movement
            velocity_magnitude = abs(self.finger_velocities[i])
            if velocity_magnitude > 50:
                color = "#ff4444"  # Fast = red
            elif velocity_magnitude > 10:
                color = "#ffaa44"  # Medium = orange
            else:
                color = "#44ff44"  # Slow = green
            
            # Draw finger bar
            self.canvas.create_rectangle(finger_x - 15, canvas_height - finger_height,
                                       finger_x + 15, canvas_height,
                                       fill=color, outline="white")
            
            # Label
            self.canvas.create_text(finger_x, canvas_height - finger_height - 10,
                                  text=f"F{i+1}\n{self.finger_positions[i]:.0f}°",
                                  fill="white", font=("Arial", 8))
        
        # Debug info
        debug_text = f"Mode: {'🧠 Conscious' if self.consciousness_mode.get() else '👤 Manual'}"
        if self.consciousness_cursor and self.consciousness_mode.get():
            speed = self.consciousness_cursor.get_movement_speed()
            debug_text += f" | Speed: {speed:.3f}"
        
        self.canvas.create_text(10, 10, text=debug_text, anchor=tk.NW, fill="white", font=("Arial", 10))
    
    def create_consciousness_parameter_controls(self):
        """Create real-time parameter control sliders"""
        if not self.consciousness_cursor:
            return
        
        consciousness_frame = ttk.LabelFrame(self.scrollable_frame, text="🚀 CONSCIOUSNESS PARAMETER CONTROL")
        consciousness_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create notebook for organized parameter categories
        notebook = ttk.Notebook(consciousness_frame)
        notebook.pack(fill=tk.X, padx=5, pady=5)
        
        # === MOVEMENT PARAMETERS TAB ===
        movement_tab = ttk.Frame(notebook)
        notebook.add(movement_tab, text="🎯 Movement")
        
        self.create_param_slider(movement_tab, "Base Speed", "base_speed", 0.1, 2.0, 0.1, 0)
        self.create_param_slider(movement_tab, "Dampening", "dampening", 0.8, 0.99, 0.01, 1)
        self.create_param_slider(movement_tab, "Emotional Influence", "emotional_influence", 0.0, 2.0, 0.1, 2)
        self.create_param_slider(movement_tab, "Novelty Speed Multi", "novelty_speed_multiplier", 1.0, 5.0, 0.1, 3)
        
        # === BEHAVIORAL MODES TAB ===
        behavior_tab = ttk.Frame(notebook)
        notebook.add(behavior_tab, text="🎭 Behaviors")
        
        self.create_param_slider(behavior_tab, "Vibration Intensity", "vibration_intensity", 0.0, 0.2, 0.01, 0)
        self.create_param_slider(behavior_tab, "Pulsation Rate", "pulsation_rate", 0.5, 5.0, 0.1, 1)
        self.create_param_slider(behavior_tab, "Transition Interval", "behavior_transition_interval", 1.0, 10.0, 0.5, 2)
        self.create_param_slider(behavior_tab, "Direction Persistence", "direction_persistence", 0.0, 1.0, 0.1, 3)
        
        # === FACE TRACKING TAB ===
        face_tab = ttk.Frame(notebook)
        notebook.add(face_tab, text="👁️ Face Tracking")
        
        self.create_param_slider(face_tab, "Face Tracking Strength", "face_tracking_strength", 0.0, 2.0, 0.1, 0)
        self.create_param_slider(face_tab, "Gaze Following Strength", "gaze_following_strength", 0.0, 1.0, 0.1, 1)
        self.create_param_slider(face_tab, "Object Attention Strength", "object_attention_strength", 0.0, 1.0, 0.1, 2)
        
        # === NOISE & CHAOS TAB ===
        noise_tab = ttk.Frame(notebook)
        notebook.add(noise_tab, text="🌪️ Noise & Chaos")
        
        self.create_param_slider(noise_tab, "Base Noise Level", "base_noise_level", 0.0, 0.1, 0.005, 0)
        self.create_param_slider(noise_tab, "Chaos Multiplier", "chaos_multiplier", 0.0, 0.5, 0.05, 1)
        self.create_param_slider(noise_tab, "Micro Jitter", "micro_jitter", 0.0, 0.05, 0.005, 2)
        
        print("🚀 Consciousness parameter controls created!")
    
    def create_param_slider(self, parent, label, attr_name, min_val, max_val, resolution, row):
        """Create a parameter slider that updates consciousness cursor in real-time"""
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Get current value from consciousness cursor
        current_value = getattr(self.consciousness_cursor, attr_name, (min_val + max_val) / 2)
        var = tk.DoubleVar(value=current_value)
        
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, 
                          orient=tk.HORIZONTAL, length=200)
        slider.grid(row=row, column=1, padx=5, pady=2)
        
        value_label = ttk.Label(parent, text=f"{current_value:.3f}")
        value_label.grid(row=row, column=2, padx=5, pady=2)
        
        # Update consciousness cursor parameter in real-time
        def update_parameter(*args):
            new_value = var.get()
            setattr(self.consciousness_cursor, attr_name, new_value)
            value_label.config(text=f"{new_value:.3f}")
            
            # Show feedback for dramatic changes
            if attr_name in ["vibration_intensity", "pulsation_rate", "base_speed"]:
                if new_value > 0.8 * max_val:
                    print(f"🚀 HIGH {attr_name.upper()}: {new_value:.2f}")
        
        var.trace('w', update_parameter)
        
        # Store reference for external access
        setattr(self, f"var_{attr_name}", var)
    
    def on_close(self):
        """Handle window close event."""
        self.running = False
        
        # CRITICAL: Restore automatic consciousness control when closing
        if self.hand_controller and self.consciousness_mode.get():
            print("🔄 Restoring automatic consciousness control to machine.py...")
            self.hand_controller.disable_manual_override()
            print("✅ Automatic consciousness updates re-enabled")
        
        if self.hand_controller:
            self.hand_controller.cleanup()
        self.root.quit()
    
    def on_close(self):
        """Clean up when interface is closed."""
        self.stop_keepalive_heartbeat()
        self.running = False
        self.root.destroy()

    def run(self):
        """Start the interface."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        print("🎮 Starting Conscious Cursor Hand Controller Interface...")
        print("Toggle between manual mouse control and AI consciousness puppeteering!")
        print("Use the checkboxes to switch between modes and adjust physics parameters.")
        self.root.mainloop()


if __name__ == "__main__":
    interface = ConsciousCursorInterface()
    interface.run()
