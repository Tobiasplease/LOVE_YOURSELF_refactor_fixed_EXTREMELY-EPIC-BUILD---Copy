#!/usr/bin/env python3
"""
Essential Emotional Movement System - FULL FUNCTIONALITY
=======================================================

Clean, focused emotional hand control system with ALL essential controls preserved.
This combines the streamlined interface with the crucial functionality you need.

Key Features:
✅ Manual mouse control for recording movements
✅ Physics toggle (essential for responsiveness)  
✅ All wave/gravity/physics parameters
✅ Direction controls and baseline adjustments
✅ Learning system with preset emotions from main system
✅ Larger movement canvas for better control
✅ Real-time hand control that actually works

Author: Essential Emotional AI Team
"""

import tkinter as tk
from tkinter import ttk
import time
import math
import json
import os
import random
from typing import Optional, Dict, List

# Import our core systems
try:
    from essential_conscious_cursor import ConsciousCursor, ConsciousnessState
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    print("❌ ConsciousCursor not available - check essential_conscious_cursor.py")
    CONSCIOUSNESS_AVAILABLE = False

try:
    from movement_learning import MovementLearning
    LEARNING_AVAILABLE = True
except ImportError:
    print("❌ Movement Learning not available - check movement_learning.py")
    LEARNING_AVAILABLE = False

try:
    from hand_expression import HandExpressionController
    HAND_AVAILABLE = True
except ImportError:
    print("❌ Hand controller not available")
    HAND_AVAILABLE = False


class EmotionalMovementSystem:
    """Complete emotional hand control system with all essential features."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎭 Essential Emotional Movement System - Full Control 🎭")
        self.root.geometry("1000x800")  # Larger for better control
        
        # Color scheme
        self.bg_color = "#2E3440"
        self.accent_color = "#5E81AC"
        self.success_color = "#A3BE8C"
        self.warning_color = "#EBCB8B"
        self.error_color = "#BF616A"
        
        self.root.configure(bg=self.bg_color)
        
        # Core systems
        self.consciousness_cursor = None
        self.consciousness_state = None
        self.movement_learner = None
        self.hand_controller = None
        self.connected = False
        
        # Initialize core systems
        if CONSCIOUSNESS_AVAILABLE:
            self.consciousness_cursor = ConsciousCursor(canvas_width=800, canvas_height=500)  # Larger canvas
            self.consciousness_state = ConsciousnessState()
            print("🧠 Consciousness cursor system loaded")
        
        if LEARNING_AVAILABLE:
            self.movement_learner = MovementLearning()
            self.movement_learner.load_profiles()
            print("🎓 Movement learning system loaded")
        
        # Movement recording
        self.recording = False
        self.current_emotion = ""
        self.recorded_movements = []
        self.recording_start_time = 0
        
        # Physics state for hand control
        self.num_fingers = 4
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_velocities = [0.0] * self.num_fingers
        
        # Physics parameters - ALL ESSENTIAL CONTROLS PRESERVED
        self.spring_force = tk.DoubleVar(value=500.0)
        self.damping = tk.DoubleVar(value=0.1)
        self.max_velocity = tk.DoubleVar(value=1000.0)
        self.cursor_sensitivity = tk.DoubleVar(value=3.0)
        self.wave_strength = tk.DoubleVar(value=2.0)
        self.gravity_width = tk.DoubleVar(value=0.4)
        self.default_position = tk.DoubleVar(value=90.0)
        
        # Control modes - ESSENTIAL TOGGLES PRESERVED
        self.physics_mode = tk.BooleanVar(value=False)  # Start with direct mode for responsiveness
        self.reverse_vertical = tk.BooleanVar(value=False)
        self.manual_mode = tk.BooleanVar(value=True)  # Manual mouse control for recording
        
        # Mouse tracking (for manual mode)
        self.mouse_x = 0.5  # Normalized cursor position (0-1)
        self.mouse_y = 0.5
        
        # Animation state
        self.running = False
        self.last_time = time.time()
        
        # Current emotional state
        self.current_applied_emotion = "neutral"
        
        # Preset emotions from main system (mood.py style)
        self.preset_emotions = [
            "neutral", "energized", "alert", "calm", "quiet", "withdrawn",
            "restless", "curious", "observant", "watchful", "detached", 
            "engaged", "focused", "contemplative", "excited", "content"
        ]
        
        self.setup_ui()
        self.start_main_loop()
    
    def setup_ui(self):
        """Create complete UI with all essential controls."""
        # Create main paned window for better layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        left_frame = tk.Frame(main_paned, bg=self.bg_color, width=400)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for canvas
        right_frame = tk.Frame(main_paned, bg=self.bg_color)
        main_paned.add(right_frame, weight=2)
        
        # === LEFT PANEL CONTROLS ===
        self.setup_connection_controls(left_frame)
        self.setup_learning_controls(left_frame)
        self.setup_control_modes(left_frame)
        self.setup_physics_parameters(left_frame)
        
        # === RIGHT PANEL - MOVEMENT CANVAS ===
        self.setup_movement_canvas(right_frame)
    
    def setup_connection_controls(self, parent):
        """Connection controls."""
        conn_frame = tk.LabelFrame(parent, text="🔌 Hand Connection", font=("Arial", 11, "bold"),
                                  bg=self.bg_color, fg="white")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        conn_row = tk.Frame(conn_frame, bg=self.bg_color)
        conn_row.pack(fill=tk.X, padx=10, pady=10)
        
        self.connect_btn = tk.Button(conn_row, text="🔌 Connect Hand Controller", 
                                   command=self.toggle_connection, 
                                   bg=self.accent_color, fg="white", font=("Arial", 9))
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(conn_row, text="❌ Disconnected", 
                                   bg=self.bg_color, fg=self.error_color, font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=20)
    
    def setup_learning_controls(self, parent):
        """Learning and emotion controls."""
        learn_frame = tk.LabelFrame(parent, text="🎓 Emotion Learning & Expression", font=("Arial", 11, "bold"),
                                   bg=self.bg_color, fg="white")
        learn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Emotion selection
        emotion_row = tk.Frame(learn_frame, bg=self.bg_color)
        emotion_row.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(emotion_row, text="Emotion:", bg=self.bg_color, fg="white", 
                font=("Arial", 9)).pack(side=tk.LEFT)
        
        self.emotion_var = tk.StringVar(value="neutral")
        self.emotion_combo = ttk.Combobox(emotion_row, textvariable=self.emotion_var, 
                                         values=self.preset_emotions, font=("Arial", 9), width=15)
        self.emotion_combo.pack(side=tk.LEFT, padx=10)
        
        # Learning controls
        learn_row = tk.Frame(learn_frame, bg=self.bg_color)
        learn_row.pack(fill=tk.X, padx=10, pady=5)
        
        self.learn_btn = tk.Button(learn_row, text="🔴 Start Learning", 
                                 command=self.toggle_learning, 
                                 bg=self.success_color, fg="black", font=("Arial", 9, "bold"))
        self.learn_btn.pack(side=tk.LEFT, padx=5)
        
        # Expression controls
        express_row = tk.Frame(learn_frame, bg=self.bg_color)
        express_row.pack(fill=tk.X, padx=10, pady=5)
        
        # Learned emotions selector
        tk.Label(express_row, text="Express:", bg=self.bg_color, fg="white", 
                font=("Arial", 9)).pack(side=tk.LEFT)
        
        self.learned_emotion_var = tk.StringVar(value="neutral")
        self.learned_emotion_combo = ttk.Combobox(express_row, textvariable=self.learned_emotion_var, 
                                                 state="readonly", font=("Arial", 9), width=15)
        self.learned_emotion_combo.pack(side=tk.LEFT, padx=5)
        
        self.express_btn = tk.Button(express_row, text="🎭 Express Emotion", 
                                   command=self.apply_emotion, 
                                   bg=self.accent_color, fg="white", font=("Arial", 9, "bold"))
        self.express_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        self.learn_status = tk.Label(learn_frame, text="Select emotion and click 'Start Learning' to record", 
                                   bg=self.bg_color, fg=self.accent_color, font=("Arial", 9))
        self.learn_status.pack(pady=5)
        
        # Current state display
        self.current_emotion_label = tk.Label(learn_frame, text="Current: neutral behavior", 
                                            bg=self.bg_color, fg=self.success_color, 
                                            font=("Arial", 10, "bold"))
        self.current_emotion_label.pack(pady=5)
        
        # Initialize
        self.refresh_learned_emotions()
    
    def setup_control_modes(self, parent):
        """Essential control mode toggles."""
        mode_frame = tk.LabelFrame(parent, text="🎛️ Control Modes", font=("Arial", 11, "bold"),
                                  bg=self.bg_color, fg="white")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Physics toggle - ESSENTIAL
        physics_row = tk.Frame(mode_frame, bg=self.bg_color)
        physics_row.pack(fill=tk.X, padx=10, pady=5)
        
        physics_cb = tk.Checkbutton(physics_row, text="⚡ Physics Mode (unchecked = direct/responsive)", 
                                   variable=self.physics_mode, bg=self.bg_color, fg="white",
                                   selectcolor=self.bg_color, font=("Arial", 9))
        physics_cb.pack(side=tk.LEFT)
        
        # Manual control toggle  
        manual_row = tk.Frame(mode_frame, bg=self.bg_color)
        manual_row.pack(fill=tk.X, padx=10, pady=5)
        
        manual_cb = tk.Checkbutton(manual_row, text="🖱️ Manual Mouse Control (for recording)", 
                                  variable=self.manual_mode, bg=self.bg_color, fg="white",
                                  selectcolor=self.bg_color, font=("Arial", 9))
        manual_cb.pack(side=tk.LEFT)
        
        # Vertical reverse toggle - ESSENTIAL
        reverse_row = tk.Frame(mode_frame, bg=self.bg_color)
        reverse_row.pack(fill=tk.X, padx=10, pady=5)
        
        reverse_cb = tk.Checkbutton(reverse_row, text="🔄 Reverse Vertical Direction", 
                                   variable=self.reverse_vertical, bg=self.bg_color, fg="white",
                                   selectcolor=self.bg_color, font=("Arial", 9))
        reverse_cb.pack(side=tk.LEFT)
        
        # Reset button
        reset_row = tk.Frame(mode_frame, bg=self.bg_color)
        reset_row.pack(fill=tk.X, padx=10, pady=5)
        
        reset_btn = tk.Button(reset_row, text="🎯 Reset to Neutral", 
                             command=self.reset_to_neutral,
                             bg=self.warning_color, fg="black", font=("Arial", 9))
        reset_btn.pack(side=tk.LEFT)
    
    def setup_physics_parameters(self, parent):
        """Essential physics parameter controls."""
        # Physics parameters
        physics_frame = tk.LabelFrame(parent, text="⚙️ Physics Parameters", font=("Arial", 11, "bold"),
                                     bg=self.bg_color, fg="white")
        physics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Spring Force
        self.create_parameter_control(physics_frame, "Spring Force:", self.spring_force, 100, 1000, 0)
        
        # Damping  
        self.create_parameter_control(physics_frame, "Damping:", self.damping, 0.01, 1.0, 1)
        
        # Cursor Sensitivity
        self.create_parameter_control(physics_frame, "Cursor Sensitivity:", self.cursor_sensitivity, 0.5, 10.0, 2)
        
        # Wave controls
        wave_frame = tk.LabelFrame(parent, text="🌊 Wave Controls", font=("Arial", 11, "bold"),
                                  bg=self.bg_color, fg="white")
        wave_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Wave Strength
        self.create_parameter_control(wave_frame, "Wave Strength:", self.wave_strength, 0.0, 5.0, 3)
        
        # Gravity Width  
        self.create_parameter_control(wave_frame, "Gravity Width:", self.gravity_width, 0.1, 1.0, 4)
        
        # Default Position (baseline)
        self.create_parameter_control(wave_frame, "Baseline Position:", self.default_position, 0, 180, 5)
    
    def create_parameter_control(self, parent, label_text, variable, min_val, max_val, row):
        """Create a parameter control row."""
        param_frame = tk.Frame(parent, bg=self.bg_color)
        param_frame.pack(fill=tk.X, padx=10, pady=2)
        
        label = tk.Label(param_frame, text=label_text, bg=self.bg_color, fg="white", 
                        font=("Arial", 8), width=15, anchor="w")
        label.pack(side=tk.LEFT)
        
        scale = tk.Scale(param_frame, from_=min_val, to=max_val, variable=variable, 
                        orient=tk.HORIZONTAL, bg=self.bg_color, fg="white",
                        highlightthickness=0, length=150, font=("Arial", 7))
        scale.pack(side=tk.LEFT, padx=5)
        
        value_label = tk.Label(param_frame, text=f"{variable.get():.2f}", 
                              bg=self.bg_color, fg=self.accent_color, font=("Arial", 8), width=6)
        value_label.pack(side=tk.LEFT, padx=5)
        
        # Update value label when scale changes
        def update_label(*args):
            try:
                value_label.config(text=f"{variable.get():.2f}")
            except:
                pass
        variable.trace_add("write", update_label)
    
    def setup_movement_canvas(self, parent):
        """Large movement canvas for recording."""
        canvas_frame = tk.LabelFrame(parent, text="🎯 Movement Recording Area", font=("Arial", 12, "bold"),
                                    bg=self.bg_color, fg="white")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Instructions
        instructions = tk.Label(canvas_frame, 
                              text="💡 Move mouse here to control hand and record emotions",
                              bg=self.bg_color, fg=self.accent_color, font=("Arial", 10))
        instructions.pack(pady=5)
        
        # Large canvas for better control
        self.canvas = tk.Canvas(canvas_frame, bg="black", height=500, width=800)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        
        # Status display
        status_frame = tk.Frame(canvas_frame, bg=self.bg_color)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.recording_status = tk.Label(status_frame, text="Mode: Manual Control", 
                                       bg=self.bg_color, fg=self.success_color, font=("Arial", 10, "bold"))
        self.recording_status.pack()
    
    def toggle_connection(self):
        """Connect/disconnect hand controller."""
        if not HAND_AVAILABLE:
            self.status_label.config(text="❌ Hand controller not available", fg=self.error_color)
            return
        
        if not self.connected:
            try:
                self.hand_controller = HandExpressionController()
                # Check if serial connection was successful
                if self.hand_controller.serial_connection:
                    self.connected = True
                    self.connect_btn.config(text="🔌 Disconnect")
                    self.status_label.config(text="✅ Connected", fg=self.success_color)
                    self.hand_controller.enable_manual_override()
                    print("🎮 Hand controller connected and enabled")
                else:
                    self.status_label.config(text="❌ Connection failed", fg=self.error_color)
            except Exception as e:
                self.status_label.config(text=f"❌ Error: {str(e)[:20]}...", fg=self.error_color)
                print(f"❌ Connection error: {e}")
        else:
            if self.hand_controller:
                if hasattr(self.hand_controller, 'disconnect'):
                    self.hand_controller.disconnect()
                elif hasattr(self.hand_controller, 'serial_connection') and self.hand_controller.serial_connection:
                    self.hand_controller.serial_connection.close()
            self.connected = False
            self.connect_btn.config(text="🔌 Connect Hand Controller")
            self.status_label.config(text="❌ Disconnected", fg=self.error_color)
            print("🔌 Hand controller disconnected")
    
    def toggle_learning(self):
        """Start/stop learning an emotion."""
        if not self.recording:
            # Start learning
            emotion = self.emotion_var.get().strip().lower()
            if not emotion:
                self.learn_status.config(text="❌ Please select an emotion", fg=self.error_color)
                return
            
            self.current_emotion = emotion
            self.recording = True
            self.recorded_movements = []
            self.recording_start_time = time.time()
            
            self.learn_btn.config(text="⏹️ Stop Learning", bg=self.warning_color)
            self.learn_status.config(text=f"🔴 RECORDING '{emotion.upper()}' - Move mouse to express this emotion!", 
                                   fg=self.warning_color)
            self.recording_status.config(text=f"🔴 RECORDING {emotion.upper()}", fg=self.warning_color)
            print(f"🎓 Started learning emotion: {emotion}")
            
        else:
            # Stop learning
            self.recording = False
            self.learn_btn.config(text="🔴 Start Learning", bg=self.success_color)
            self.recording_status.config(text="Mode: Manual Control", fg=self.success_color)
            
            if len(self.recorded_movements) >= 30:  # Reasonable minimum
                # Learn from the movements
                if self.movement_learner:
                    success = self.movement_learner.learn_from_recording(self.current_emotion, self.recorded_movements)
                    if success:
                        self.learn_status.config(text=f"✅ Learned '{self.current_emotion}' from {len(self.recorded_movements)} movements!", 
                                               fg=self.success_color)
                        self.refresh_learned_emotions()
                        # Auto-apply the learned emotion
                        self.learned_emotion_var.set(self.current_emotion)
                        self.apply_emotion()
                    else:
                        self.learn_status.config(text="❌ Learning failed - try more varied movements", fg=self.error_color)
                else:
                    self.learn_status.config(text="❌ Learning system not available", fg=self.error_color)
            else:
                needed = 30 - len(self.recorded_movements)
                self.learn_status.config(text=f"❌ Need {needed} more movement points (minimum 30)", fg=self.error_color)
            
            print(f"⏹️ Stopped learning - captured {len(self.recorded_movements)} movements")
    
    def apply_emotion(self):
        """Apply selected emotion to the AI cursor."""
        if not self.consciousness_cursor or not self.movement_learner:
            print("❌ Core systems not available")
            return
        
        emotion = self.learned_emotion_var.get()
        if emotion == "neutral":
            # Reset to neutral
            self.consciousness_state.mood = 0.0
            self.consciousness_state.novelty = 0.0
            self.consciousness_state.boredom = 0.0
            self.current_applied_emotion = "neutral"
            self.current_emotion_label.config(text="Current: neutral behavior")
            print("🎯 Reset to neutral behavior")
            return
        
        # Apply learned emotion
        success = self.movement_learner.apply_learned_parameters(self.consciousness_cursor, emotion)
        if success:
            self.current_applied_emotion = emotion
            self.current_emotion_label.config(text=f"Current: expressing '{emotion}'")
            print(f"🎭 Now expressing: {emotion}")
            
            # Set consciousness state to match emotion
            self.set_consciousness_for_emotion(emotion)
        else:
            self.learn_status.config(text=f"❌ Emotion '{emotion}' not learned yet", fg=self.error_color)
    
    def set_consciousness_for_emotion(self, emotion: str):
        """Set consciousness state values to match the emotion."""
        if not self.consciousness_state:
            return
        
        # Emotion mappings based on main system mood states
        emotion_mappings = {
            "energized": {"mood": 0.8, "novelty": 0.7, "boredom": 0.1},
            "alert": {"mood": 0.6, "novelty": 0.5, "boredom": 0.2},
            "calm": {"mood": 0.3, "novelty": 0.2, "boredom": 0.1},
            "quiet": {"mood": 0.1, "novelty": 0.1, "boredom": 0.3},
            "withdrawn": {"mood": -0.2, "novelty": 0.1, "boredom": 0.7},
            "restless": {"mood": 0.2, "novelty": 0.8, "boredom": 0.8},
            "curious": {"mood": 0.5, "novelty": 0.9, "boredom": 0.1},
            "observant": {"mood": 0.4, "novelty": 0.4, "boredom": 0.2},
            "watchful": {"mood": 0.2, "novelty": 0.3, "boredom": 0.3},
            "detached": {"mood": -0.1, "novelty": 0.1, "boredom": 0.5},
            "engaged": {"mood": 0.7, "novelty": 0.6, "boredom": 0.1},
            "focused": {"mood": 0.5, "novelty": 0.3, "boredom": 0.1},
            "contemplative": {"mood": 0.3, "novelty": 0.2, "boredom": 0.2},
            "excited": {"mood": 0.9, "novelty": 1.0, "boredom": 0.0},
            "content": {"mood": 0.6, "novelty": 0.3, "boredom": 0.1}
        }
        
        if emotion in emotion_mappings:
            mapping = emotion_mappings[emotion]
            self.consciousness_state.mood = mapping["mood"]
            self.consciousness_state.novelty = mapping["novelty"]
            self.consciousness_state.boredom = mapping["boredom"]
            print(f"🧠 Set consciousness for {emotion}: mood={mapping['mood']}, novelty={mapping['novelty']}, boredom={mapping['boredom']}")
    
    def refresh_learned_emotions(self):
        """Update the list of learned emotions."""
        emotions = ["neutral"]  # Always include neutral
        
        if self.movement_learner:
            learned = self.movement_learner.get_available_emotions()
            emotions.extend(learned)
        
        self.learned_emotion_combo.config(values=emotions)
        if emotions:
            self.learned_emotion_combo.set(emotions[0])
    
    def reset_to_neutral(self):
        """Reset everything to neutral state."""
        print("🎯 Resetting to neutral state...")
        
        # Reset consciousness cursor
        if self.consciousness_cursor:
            self.consciousness_cursor.reset_to_center()
        
        # Reset consciousness state
        if self.consciousness_state:
            self.consciousness_state.mood = 0.0
            self.consciousness_state.novelty = 0.0
            self.consciousness_state.boredom = 0.0
        
        # Reset interface
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        self.current_applied_emotion = "neutral"
        self.current_emotion_label.config(text="Current: neutral behavior")
        
        print("✅ Reset complete")
    
    def on_mouse_move(self, event):
        """Handle mouse movement for recording and hand control."""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Normalize coordinates (with safety check)
        if canvas_width > 0 and canvas_height > 0:
            x = event.x / canvas_width
            y = event.y / canvas_height
        else:
            x, y = 0.5, 0.5
        
        # Clamp to [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        # Update mouse position
        self.mouse_x = x
        self.mouse_y = y
        
        # Record if learning
        if self.recording:
            movement_point = {
                'x': x,
                'y': y,
                'time': time.time(),
                'timestamp': time.time()
            }
            self.recorded_movements.append(movement_point)
            
            # Update progress
            progress = len(self.recorded_movements)
            if progress % 10 == 0:  # Update every 10 points
                self.learn_status.config(text=f"🔴 RECORDING '{self.current_emotion.upper()}' - {progress} points captured")
    
    def on_mouse_click(self, event):
        """Handle mouse clicks."""
        pass  # Could add click-based controls later
    
    def start_main_loop(self):
        """Start the main animation loop."""
        self.running = True
        self.main_loop()
    
    def main_loop(self):
        """Main animation and physics loop."""
        if not self.running:
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Get cursor position
        if self.manual_mode.get():
            # Manual mouse control
            cursor_x, cursor_y = self.mouse_x, self.mouse_y
        else:
            # AI consciousness cursor control
            if self.consciousness_cursor and self.consciousness_state:
                self.consciousness_cursor.update(self.consciousness_state, dt)
                cursor_x, cursor_y = self.consciousness_cursor.get_position()
            else:
                cursor_x, cursor_y = 0.5, 0.5
        
        # Update hand physics
        self.update_hand_physics(cursor_x, cursor_y, dt)
        
        # Send to hand controller
        if self.connected and self.hand_controller:
            self.send_to_hand()
        
        # Update visualization
        self.update_visualization(cursor_x, cursor_y)
        
        # Schedule next update
        self.root.after(16, self.main_loop)  # ~60 FPS
    
    def update_hand_physics(self, cursor_x, cursor_y, dt):
        """Update hand physics using either direct or physics-based control."""
        # Safety checks
        if dt <= 0 or dt > 1.0:  # Prevent invalid dt values
            dt = 0.016  # Default to ~60fps
        
        # Apply vertical reverse if enabled
        if self.reverse_vertical.get():
            effective_cursor_y = 1.0 - cursor_y
        else:
            effective_cursor_y = cursor_y
        
        # Safety clamps
        cursor_x = max(0.0, min(1.0, cursor_x))
        effective_cursor_y = max(0.0, min(1.0, effective_cursor_y))
        
        try:
            if self.physics_mode.get():
                # Physics-based wave control
                for i in range(self.num_fingers):
                    finger_x = (i + 0.5) / max(1, self.num_fingers)  # Prevent division by zero
                    
                    # Distance from cursor to finger position
                    dx = cursor_x - finger_x
                    dy = effective_cursor_y - 0.5
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Gravitational wave influence (with safety checks)
                    gravity_width = max(0.01, self.gravity_width.get())  # Prevent division by zero
                    wave_influence = math.exp(-distance / gravity_width) * self.wave_strength.get()
                    
                    # Calculate target position
                    base_position = self.default_position.get()
                    cursor_influence = (effective_cursor_y - 0.5) * 180 * self.cursor_sensitivity.get()
                    wave_effect = wave_influence * cursor_influence
                    
                    target_position = base_position + wave_effect
                    target_position = max(0, min(180, target_position))
                    
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
            else:
                # Direct responsive control - ESSENTIAL FOR RECORDING
                for i in range(self.num_fingers):
                    finger_x = (i + 0.5) / max(1, self.num_fingers)  # Prevent division by zero
                    
                    # Distance from cursor to finger position
                    dx = cursor_x - finger_x
                    dy = effective_cursor_y - 0.5
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Direct wave influence (no physics delay, with safety checks)
                    gravity_width = max(0.01, self.gravity_width.get())  # Prevent division by zero
                    wave_influence = math.exp(-distance / gravity_width) * self.wave_strength.get()
                    
                    # Direct position calculation
                    base_position = self.default_position.get()
                    cursor_influence = (effective_cursor_y - 0.5) * 180 * self.cursor_sensitivity.get()
                    wave_effect = wave_influence * cursor_influence
                    
                    target_position = base_position + wave_effect
                    self.finger_positions[i] = max(0, min(180, target_position))
                    
                    # Reset velocities in direct mode
                    self.finger_velocities[i] = 0.0
                    
        except Exception as e:
            print(f"❌ Error in hand physics update: {e}")
            # Reset to safe state
            for i in range(self.num_fingers):
                self.finger_positions[i] = 90.0
                self.finger_velocities[i] = 0.0
    
    def send_to_hand(self):
        """Send finger positions to hand controller."""
        if self.hand_controller:
            try:
                # Convert to integers (Arduino expects int servo positions)
                positions = [int(pos) for pos in self.finger_positions]
                self.hand_controller.set_hand_positions(positions)
            except Exception as e:
                print(f"❌ Error sending to hand: {e}")
    
    def update_visualization(self, cursor_x, cursor_y):
        """Update the visual representation."""
        self.canvas.delete("all")
        
        # Canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Draw cursor position
        cx = cursor_x * width
        cy = cursor_y * height
        
        # Cursor circle (larger for better visibility)
        r = 15
        color = "red" if self.recording else "cyan"
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=color, outline="white", width=3)
        
        # Draw finger positions as bars at the bottom
        bar_height = 120
        bar_width = max(1, width / max(1, self.num_fingers))  # Prevent division by zero
        
        for i, pos in enumerate(self.finger_positions):
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            
            # Position as fraction (0-180 degrees -> 0-1)
            pos_fraction = pos / 180.0
            bar_top = height - (pos_fraction * bar_height)
            
            # Color based on position
            intensity = int(255 * pos_fraction)
            color = f"#{intensity:02x}{intensity//2:02x}{255-intensity:02x}"
            
            self.canvas.create_rectangle(x1, bar_top, x2, height, fill=color, outline="white", width=2)
            
            # Position label
            self.canvas.create_text((x1 + x2) / 2, height - 10, text=f"{pos:.0f}°", 
                                  fill="white", font=("Arial", 10, "bold"))
        
        # Status text
        mode = "PHYSICS" if self.physics_mode.get() else "DIRECT"
        control = "MOUSE" if self.manual_mode.get() else "AI"
        status = f"Mode: {mode} | Control: {control} | Emotion: {self.current_applied_emotion}"
        self.canvas.create_text(10, 10, text=status, fill="white", font=("Arial", 11, "bold"), anchor="nw")
        
        # Recording indicator
        if self.recording:
            self.canvas.create_oval(width-50, 10, width-20, 40, fill="red", outline="white", width=2)
            self.canvas.create_text(width-35, 50, text=f"{len(self.recorded_movements)}", 
                                  fill="red", font=("Arial", 10, "bold"))
        
        # Draw gravity field visualization
        if self.manual_mode.get() and width > 0 and self.num_fingers > 0:
            # Show gravity field for better understanding
            for i in range(self.num_fingers):
                finger_x = (i + 0.5) / self.num_fingers * width
                radius = self.gravity_width.get() * 200  # Visual representation
                
                self.canvas.create_oval(finger_x - radius, height/2 - radius,
                                      finger_x + radius, height/2 + radius,
                                      outline="yellow", width=1, stipple="gray25")
    
    def run(self):
        """Run the application."""
        print("🎭 Essential Emotional Movement System - FULL FUNCTIONALITY started!")
        print("✨ All essential controls preserved - ready for training and expression!")
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionalMovementSystem()
    app.run()
