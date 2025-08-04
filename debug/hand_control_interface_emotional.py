#!/usr/bin/env python3
"""
Emotional Hand Control Interface
===============================

Enhanced version of the physics-based hand controller with consciousness-driven cursor option.
Toggle between manual mouse control and AI emotional puppeteering!

Features:
- Manual mode: Your working mouse-controlled physics system
- Emotional mode: AI consciousness drives the cursor automatically
- Smooth transitions between modes
- All original functionality preserved
- Real-time emotional state visualization

Author: Emotional Puppeteering System
"""

import tkinter as tk
from tkinter import ttk
import time
import math
import threading
from typing import Optional

# Import our consciousness cursor system
try:
    from consciousness_cursor_advanced import AdvancedConsciousnessCursor as ConsciousnessCursor, ConsciousnessState
    CONSCIOUSNESS_AVAILABLE = True
    USE_ADVANCED_CURSOR = True
    print("🚀 Advanced consciousness cursor loaded!")
except ImportError:
    print("⚠️ Advanced cursor not available, falling back to basic")
    try:
        from consciousness_cursor import ConsciousnessCursor, ConsciousnessState
        CONSCIOUSNESS_AVAILABLE = True
        USE_ADVANCED_CURSOR = False
        print("🧠 Basic consciousness cursor loaded")
    except ImportError:
        print("⚠️ No consciousness cursor available - manual mode only")
        CONSCIOUSNESS_AVAILABLE = False
        USE_ADVANCED_CURSOR = False

# Import consciousness bridge for live data
try:
    from consciousness_bridge import start_consciousness_bridge, get_live_consciousness_data, is_consciousness_data_fresh
    CONSCIOUSNESS_BRIDGE_AVAILABLE = True
except ImportError:
    print("⚠️ Consciousness bridge not available - using simulated data")
    CONSCIOUSNESS_BRIDGE_AVAILABLE = False

# Import hand controller
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from servo_control.hand_expression import HandExpressionController
    HAND_CONTROLLER_AVAILABLE = True
except ImportError:
    print("⚠️ Hand controller not available - simulation mode")
    HAND_CONTROLLER_AVAILABLE = False


class EmotionalHandInterface:
    """Enhanced physics-based hand control with emotional cursor option."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Emotional Hand Controller - AI Consciousness Mode")
        self.root.geometry("650x450")  # Slightly taller for emotional controls
        self.root.resizable(False, False)  # Lock window dimensions to prevent resizing
        
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
        
        # Control mode - START WITH EMOTIONAL MODE ENABLED BY DEFAULT
        self.emotional_mode = tk.BooleanVar(value=True)  # Default to AI emotional mode
        self.consciousness_cursor = None
        self.consciousness_state = None
        
        if CONSCIOUSNESS_AVAILABLE:
            self.consciousness_cursor = ConsciousnessCursor(canvas_width=563, canvas_height=304)
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
        
        self.setup_ui()
        
        # AUTO-INITIALIZE: Revolutionary system is now the PRIMARY DEFAULT
        if CONSCIOUSNESS_AVAILABLE:
            self.on_emotional_mode_toggle()  # Enable revolutionary consciousness control
            print("🚀 REVOLUTIONARY SYSTEM ACTIVATED - This is now the primary hand control system!")
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
        
        # Override mode toggle - CRITICAL FOR MANUAL CONTROL
        self.override_cb = ttk.Checkbutton(conn_frame, text="🎮 Override Automatic Control", 
                                          variable=self.override_mode,
                                          command=self.on_override_toggle)
        self.override_cb.pack(side=tk.RIGHT)
        
        self.override_label = ttk.Label(conn_frame, text="Manual Mode: OFF", foreground="blue")
        self.override_label.pack(side=tk.RIGHT, padx=(0, 10))
        
        # === EMOTIONAL CONTROL FRAME ===
        if CONSCIOUSNESS_AVAILABLE:
            emotional_frame = ttk.LabelFrame(self.scrollable_frame, text="🧠 Consciousness Control")
            emotional_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Emotional mode toggle
            self.emotional_cb = ttk.Checkbutton(emotional_frame, 
                                              text="🎭 Emotional Cursor Mode (AI drives cursor)",
                                              variable=self.emotional_mode,
                                              command=self.on_emotional_mode_toggle)
            self.emotional_cb.pack(anchor=tk.W, padx=5, pady=2)
            
            # Status display
            self.emotional_status = ttk.Label(emotional_frame, text="Manual cursor control active", 
                                            foreground="blue")
            self.emotional_status.pack(anchor=tk.W, padx=5, pady=2)
            
            # Consciousness data display
            self.consciousness_display = ttk.Label(emotional_frame, text="No consciousness data", 
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
        
        # === ADVANCED CONSCIOUSNESS PARAMETERS ===
        if USE_ADVANCED_CURSOR:
            self.create_advanced_parameter_controls()
        
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
    
    def on_emotional_mode_toggle(self):
        """Handle switching between manual and emotional cursor control."""
        if not CONSCIOUSNESS_AVAILABLE:
            self.emotional_mode.set(False)
            return
        
        if self.emotional_mode.get():
            # Switch to emotional mode - AI consciousness controls cursor AND disable machine.py automatic updates
            self.emotional_status.config(text="🧠 AI consciousness driving cursor", foreground="green")
            if self.consciousness_cursor:
                self.consciousness_cursor.reset_to_center()
            
            # CRITICAL: Enable both manual override AND interface override for full control
            if self.connected and self.hand_controller:
                self.hand_controller.enable_manual_override()
                self.override_mode.set(True)  # Also enable interface override
                self.on_override_toggle()  # Update the UI
                print("🎭 Switched to emotional cursor mode - AI consciousness now controls movement!")
                print("🚫 Disabled automatic consciousness updates from machine.py to prevent conflicts")
                print("🎮 Enabled interface override - emotional cursor now drives hand movement")
            else:
                print("🎭 Switched to emotional cursor mode - AI consciousness control ready (hand not connected)")
        else:
            # Switch to manual mode - mouse controls cursor AND re-enable machine.py automatic updates
            self.emotional_status.config(text="👤 Manual cursor control active", foreground="blue")
            
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
            print("🎮 Manual override enabled - interface controls hand directly")
        else:
            self.override_label.config(text="Manual Mode: OFF", foreground="blue")
            print("🤖 Automatic mode - main system controls hand")
    
    def simulate_consciousness_data(self):
        """Simulate consciousness data for testing emotional mode."""
        if not self.consciousness_state:
            return
        
        # Emotional states matching the actual mood system from mood.py and captioner/prompts.py
        import random
        emotions = [
            {
                "name": "Energized & Deeply Engaged",  # mood > 0.5
                "mood": 0.75, "novelty": 0.8, "boredom": 0.1,
                "spring_force": 600.0, "damping": 0.1, "sensitivity": 3.0,
                "wave_strength": 2.5, "gravity_width": 0.4
            },
            {
                "name": "Alert & Curious",  # mood > 0.1
                "mood": 0.3, "novelty": 0.7, "boredom": 0.2,
                "spring_force": 450.0, "damping": 0.15, "sensitivity": 2.2,
                "wave_strength": 1.8, "gravity_width": 0.5
            },
            {
                "name": "Calm & Observant",  # mood > -0.1 (neutral)
                "mood": 0.0, "novelty": 0.4, "boredom": 0.3,
                "spring_force": 300.0, "damping": 0.25, "sensitivity": 1.5,
                "wave_strength": 1.0, "gravity_width": 0.7
            },
            {
                "name": "Quiet & Detached",  # mood > -0.5
                "mood": -0.3, "novelty": 0.2, "boredom": 0.6,
                "spring_force": 180.0, "damping": 0.35, "sensitivity": 1.0,
                "wave_strength": 0.6, "gravity_width": 0.8
            },
            {
                "name": "Withdrawn & Distant",  # mood <= -0.5
                "mood": -0.7, "novelty": 0.1, "boredom": 0.8,
                "spring_force": 100.0, "damping": 0.45, "sensitivity": 0.7,
                "wave_strength": 0.3, "gravity_width": 0.9
            }
        ]
        
        emotion = random.choice(emotions)
        
        # Apply emotional state
        self.consciousness_state.mood = emotion["mood"]
        self.consciousness_state.novelty = emotion["novelty"]
        self.consciousness_state.boredom = emotion["boredom"]
        self.consciousness_state.person_present = random.choice([True, False])
        self.consciousness_state.face_confidence = random.uniform(0.3, 0.9) if self.consciousness_state.person_present else 0.0
        
        # Apply emotion-specific parameters
        self.apply_emotion_parameters(emotion)
        
        print(f"🎭 Simulating emotion: {emotion['name']} with custom parameters:")
        print(f"   Spring: {emotion['spring_force']}, Damping: {emotion['damping']}")
        print(f"   Sensitivity: {emotion['sensitivity']}, Wave: {emotion['wave_strength']}, Gravity: {emotion['gravity_width']}")
    
    def apply_emotion_parameters(self, emotion):
        """Apply emotion-specific parameter configurations."""
        self.spring_force.set(emotion["spring_force"])
        self.damping.set(emotion["damping"])
        self.cursor_sensitivity.set(emotion["sensitivity"])
        self.wave_strength.set(emotion["wave_strength"])
        self.gravity_width.set(emotion["gravity_width"])
    
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
    
    def on_mouse_move(self, event):
        """Handle mouse movement over the canvas (manual mode only)."""
        if self.emotional_mode.get():
            return  # Ignore mouse in emotional mode
        
        canvas_width = 563  # Fixed width - no dynamic sizing
        canvas_height = 304  # Fixed height - no dynamic sizing
        
        if canvas_width > 1 and canvas_height > 1:
            self.mouse_x = event.x / canvas_width
            self.mouse_y = event.y / canvas_height
            
            # Keep within bounds
            self.mouse_x = max(0.0, min(1.0, self.mouse_x))
            self.mouse_y = max(0.0, min(1.0, self.mouse_y))
    
    def on_mouse_click(self, event):
        """Handle mouse clicks."""
        print(f"🖱️ Click at ({event.x}, {event.y})")
        if self.emotional_mode.get() and self.consciousness_cursor:
            # In emotional mode, clicking resets cursor to center
            self.consciousness_cursor.reset_to_center()
    
    def get_current_cursor_position(self):
        """Get current cursor position based on active mode."""
        if self.emotional_mode.get() and self.consciousness_cursor:
            cursor_x, cursor_y = self.consciousness_cursor.get_position()
        else:
            cursor_x, cursor_y = (self.mouse_x, self.mouse_y)
        
        # Track cursor position for startle tests
        self.last_cursor_x = cursor_x
        self.last_cursor_y = cursor_y
        
        return (cursor_x, cursor_y)
    
    def toggle_connection(self):
        """Toggle connection to hand controller."""
        if not HAND_CONTROLLER_AVAILABLE:
            print("⚠️ Hand controller not available - simulation mode")
            return
        
        if not self.connected:
            try:
                # Attempt connection
                from servo_control.hand_expression import HandExpressionController
                self.hand_controller = HandExpressionController(
                    port="COM3",  # Adjust as needed
                    baudrate=9600,
                    clean_output=True
                )
                self.connected = True
                self.connect_btn.config(text="🔌 Disconnect")
                self.status_label.config(text="✅ Connected", foreground="green")
                print("✅ Connected to hand controller")
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
        self.emotional_mode.set(False)
        self.on_emotional_mode_toggle()
    
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
        
        # Update consciousness cursor if in emotional mode
        if self.emotional_mode.get() and self.consciousness_cursor and self.consciousness_state:
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
            if self.emotional_mode.get() and hasattr(self, 'consciousness_cursor'):
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
        if self.emotional_mode.get():
            # Pulsing emotional cursor
            pulse = math.sin(time.time() * 3) * 0.3 + 0.7
            size = 15 * pulse
            color = "#ff6b6b"  # Emotional red
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
        debug_text = f"Mode: {'🧠 Emotional' if self.emotional_mode.get() else '👤 Manual'}"
        if self.consciousness_cursor and self.emotional_mode.get():
            speed = self.consciousness_cursor.get_movement_speed()
            debug_text += f" | Speed: {speed:.3f}"
        
        self.canvas.create_text(10, 10, text=debug_text, anchor=tk.NW, fill="white", font=("Arial", 10))
    
    def create_advanced_parameter_controls(self):
        """Create revolutionary real-time parameter control sliders"""
        if not self.consciousness_cursor:
            return
        
        advanced_frame = ttk.LabelFrame(self.scrollable_frame, text="🚀 REVOLUTIONARY PARAMETER CONTROL")
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create notebook for organized parameter categories
        notebook = ttk.Notebook(advanced_frame)
        notebook.pack(fill=tk.X, padx=5, pady=5)
        
        # === MOVEMENT PARAMETERS TAB ===
        movement_tab = ttk.Frame(notebook)
        notebook.add(movement_tab, text="🎯 Movement")
        
        self.create_param_slider(movement_tab, "Base Speed", "base_speed", 0.001, 0.1, 0.001, 0)
        self.create_param_slider(movement_tab, "Momentum Decay", "momentum_decay", 0.8, 0.99, 0.01, 1)
        self.create_param_slider(movement_tab, "Mood Influence", "mood_influence", 0.0, 1.0, 0.1, 2)
        self.create_param_slider(movement_tab, "Novelty Speed Multi", "novelty_speed_multiplier", 1.0, 10.0, 0.5, 3)
        
        # === BEHAVIORAL MODES TAB ===
        behavior_tab = ttk.Frame(notebook)
        notebook.add(behavior_tab, text="🎭 Behaviors")
        
        self.create_param_slider(behavior_tab, "Vibration Intensity", "vibration_intensity", 0.0, 1.0, 0.1, 0)
        self.create_param_slider(behavior_tab, "Pulsation Strength", "pulsation_strength", 0.0, 1.0, 0.1, 1)
        self.create_param_slider(behavior_tab, "Linger Tendency", "linger_tendency", 0.0, 1.0, 0.1, 2)
        self.create_param_slider(behavior_tab, "Exploration Boldness", "exploration_boldness", 0.0, 2.0, 0.1, 3)
        
        # === FACE TRACKING TAB ===
        face_tab = ttk.Frame(notebook)
        notebook.add(face_tab, text="👁️ Face Tracking")
        
        self.create_param_slider(face_tab, "Face Attraction", "face_attraction_strength", 0.0, 2.0, 0.1, 0)
        self.create_param_slider(face_tab, "Face Orbit Radius", "face_orbit_radius", 0.05, 0.5, 0.05, 1)
        self.create_param_slider(face_tab, "Gaze Following", "gaze_following_strength", 0.0, 1.0, 0.1, 2)
        self.create_param_slider(face_tab, "Object Attention", "object_attention_strength", 0.0, 1.0, 0.1, 3)
        
        # === DIRECTION & BOUNCING TAB ===
        direction_tab = ttk.Frame(notebook)
        notebook.add(direction_tab, text="🎢 Direction")
        
        self.create_param_slider(direction_tab, "Direction Change Freq", "direction_change_frequency", 0.1, 3.0, 0.1, 0)
        self.create_param_slider(direction_tab, "Wall Bounce Strength", "wall_bounce_strength", 0.0, 2.0, 0.1, 1)
        self.create_param_slider(direction_tab, "Direction Persistence", "direction_persistence", 0.0, 1.0, 0.1, 2)
        
        # === TEMPORAL BEHAVIORS TAB ===
        temporal_tab = ttk.Frame(notebook)
        notebook.add(temporal_tab, text="⏱️ Temporal")
        
        self.create_param_slider(temporal_tab, "Pause Probability", "pause_probability", 0.0, 0.5, 0.01, 0)
        self.create_param_slider(temporal_tab, "Burst Movement", "burst_movement_chance", 0.0, 0.2, 0.01, 1)
        self.create_param_slider(temporal_tab, "Rhythm Sync", "rhythm_sync_strength", 0.0, 2.0, 0.1, 2)
        
        # === NOISE & CHAOS TAB ===
        noise_tab = ttk.Frame(notebook)
        notebook.add(noise_tab, text="🌪️ Noise & Chaos")
        
        self.create_param_slider(noise_tab, "Micro Noise", "micro_noise_amplitude", 0.0, 0.05, 0.001, 0)
        self.create_param_slider(noise_tab, "Macro Noise", "macro_noise_amplitude", 0.0, 0.2, 0.01, 1)
        self.create_param_slider(noise_tab, "Chaos Threshold", "chaos_threshold", 0.3, 1.0, 0.1, 2)
        
        print("🚀 Revolutionary parameter controls created!")
    
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
            if attr_name in ["vibration_intensity", "pulsation_strength", "wall_bounce_strength"]:
                if new_value > 0.8:
                    print(f"🚀 HIGH {attr_name.upper()}: {new_value:.2f}")
            elif attr_name in ["chaos_threshold"] and new_value < 0.5:
                print(f"🌪️ CHAOS MODE ACTIVE: threshold {new_value:.2f}")
        
        var.trace('w', update_parameter)
        
        # Store reference for external access
        setattr(self, f"var_{attr_name}", var)
    
    def on_close(self):
        """Handle window close event."""
        self.running = False
        
        # CRITICAL: Restore automatic consciousness control when closing
        if self.hand_controller and self.emotional_mode.get():
            print("🔄 Restoring automatic consciousness control to machine.py...")
            self.hand_controller.disable_manual_override()
            print("✅ Automatic consciousness updates re-enabled")
        
        if self.hand_controller:
            self.hand_controller.cleanup()
        self.root.quit()
    
    def run(self):
        """Start the interface."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        print("🎮 Starting Emotional Physics-Based Hand Controller Interface...")
        print("Toggle between manual mouse control and AI emotional puppeteering!")
        print("Use the checkboxes to switch between modes and adjust physics parameters.")
        self.root.mainloop()


if __name__ == "__main__":
    interface = EmotionalHandInterface()
    interface.run()
