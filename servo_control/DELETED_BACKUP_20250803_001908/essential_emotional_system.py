#!/usr/bin/env python3
"""
Essential Emotional Movement System
===================================

This is the CORE of what we've been building - pure emotional hand control!

**What this system does:**
1. Record your movements while you express emotions
2. Learn the unique patterns of each emotion 
3. AI cursor expresses those learned emotions autonomously
4. Real-time transitions between emotional states
5. Direct hand control through physics simulation

**How to use:**
1. Click "Learn New Emotion" 
2. Move mouse while expressing that emotion (happy, sad, excited, etc.)
3. Click "Stop Learning"
4. Watch the AI express YOUR emotional movement patterns!

This is the clean, focused version that preserves the revolutionary core!
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
    """Clean, focused emotional hand control system."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎭 Essential Emotional Movement System 🎭")
        self.root.geometry("800x600")
        
        # Simple color scheme
        self.bg_color = "#2E3440"
        self.accent_color = "#5E81AC"
        self.success_color = "#A3BE8C"
        self.warning_color = "#EBCB8B"
        
        self.root.configure(bg=self.bg_color)
        
        # Core systems
        self.consciousness_cursor = None
        self.consciousness_state = None
        self.movement_learner = None
        self.hand_controller = None
        self.connected = False
        
        # Initialize core systems
        if CONSCIOUSNESS_AVAILABLE:
            self.consciousness_cursor = ConsciousCursor(canvas_width=600, canvas_height=400)
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
        
        # Physics parameters - WORKING VALUES from testing
        self.spring_force = 500.0
        self.damping = 0.1
        self.max_velocity = 1000.0
        self.cursor_sensitivity = 3.0
        self.wave_strength = 2.0
        self.gravity_width = 0.4
        self.default_position = 90.0
        
        # Animation state
        self.running = False
        self.last_time = time.time()
        
        # Current emotional state
        self.current_applied_emotion = "neutral"
        
        self.setup_ui()
        self.start_main_loop()
    
    def setup_ui(self):
        """Create clean, focused UI."""
        # Title
        title_frame = tk.Frame(self.root, bg=self.bg_color)
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = tk.Label(title_frame, text="🎭 Essential Emotional Movement System", 
                              font=("Arial", 16, "bold"), bg=self.bg_color, fg="white")
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Learn emotions → AI expresses them through hand movement", 
                                 font=("Arial", 10), bg=self.bg_color, fg=self.accent_color)
        subtitle_label.pack()
        
        # Connection panel
        conn_frame = tk.LabelFrame(self.root, text="Hand Connection", font=("Arial", 12, "bold"),
                                  bg=self.bg_color, fg="white")
        conn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        conn_controls = tk.Frame(conn_frame, bg=self.bg_color)
        conn_controls.pack(fill=tk.X, padx=10, pady=10)
        
        self.connect_btn = tk.Button(conn_controls, text="🔌 Connect Hand Controller", 
                                   command=self.toggle_connection, 
                                   bg=self.accent_color, fg="white", font=("Arial", 10))
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(conn_controls, text="❌ Disconnected", 
                                   bg=self.bg_color, fg=self.warning_color, font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Learning panel - THE CORE FUNCTIONALITY
        learn_frame = tk.LabelFrame(self.root, text="🎓 Emotion Learning", font=("Arial", 12, "bold"),
                                   bg=self.bg_color, fg="white")
        learn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Emotion input and recording
        input_frame = tk.Frame(learn_frame, bg=self.bg_color)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(input_frame, text="Emotion to learn:", bg=self.bg_color, fg="white", 
                font=("Arial", 10)).pack(side=tk.LEFT)
        
        self.emotion_entry = tk.Entry(input_frame, font=("Arial", 10), width=20)
        self.emotion_entry.pack(side=tk.LEFT, padx=10)
        
        self.learn_btn = tk.Button(input_frame, text="🔴 Start Learning", 
                                 command=self.toggle_learning, 
                                 bg=self.success_color, fg="black", font=("Arial", 10, "bold"))
        self.learn_btn.pack(side=tk.LEFT, padx=10)
        
        # Progress display
        self.learn_status = tk.Label(learn_frame, text="Enter emotion name and click 'Start Learning'", 
                                   bg=self.bg_color, fg=self.accent_color, font=("Arial", 10))
        self.learn_status.pack(pady=5)
        
        # Expression panel - APPLY LEARNED EMOTIONS
        express_frame = tk.LabelFrame(self.root, text="🎭 Emotional Expression", font=("Arial", 12, "bold"),
                                     bg=self.bg_color, fg="white")
        express_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Learned emotions selector
        emotion_frame = tk.Frame(express_frame, bg=self.bg_color)
        emotion_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(emotion_frame, text="Express emotion:", bg=self.bg_color, fg="white", 
                font=("Arial", 10)).pack(side=tk.LEFT)
        
        self.emotion_var = tk.StringVar(value="neutral")
        self.emotion_combo = ttk.Combobox(emotion_frame, textvariable=self.emotion_var, 
                                         state="readonly", font=("Arial", 10), width=15)
        self.emotion_combo.pack(side=tk.LEFT, padx=10)
        
        self.express_btn = tk.Button(emotion_frame, text="🎭 Express This Emotion", 
                                   command=self.apply_emotion, 
                                   bg=self.accent_color, fg="white", font=("Arial", 10, "bold"))
        self.express_btn.pack(side=tk.LEFT, padx=10)
        
        # Current state display
        self.current_emotion_label = tk.Label(express_frame, text="Current: neutral behavior", 
                                            bg=self.bg_color, fg=self.success_color, 
                                            font=("Arial", 11, "bold"))
        self.current_emotion_label.pack(pady=5)
        
        # Movement canvas - VISUAL FEEDBACK
        canvas_frame = tk.LabelFrame(self.root, text="🎯 Movement Area", font=("Arial", 12, "bold"),
                                    bg=self.bg_color, fg="white")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="black", height=400, width=600)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Instructions
        instructions = tk.Label(self.root, 
                              text="💡 Move your mouse in the black area to record emotions, then watch AI express them!",
                              bg=self.bg_color, fg=self.accent_color, font=("Arial", 9))
        instructions.pack(pady=5)
        
        # Initialize
        self.refresh_learned_emotions()
        
    def toggle_connection(self):
        """Connect/disconnect hand controller."""
        if not HAND_AVAILABLE:
            self.status_label.config(text="❌ Hand controller not available", fg=self.warning_color)
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
                    self.status_label.config(text="❌ Connection failed", fg=self.warning_color)
            except Exception as e:
                self.status_label.config(text=f"❌ Error: {str(e)[:20]}...", fg=self.warning_color)
                print(f"❌ Connection error: {e}")
        else:
            if self.hand_controller:
                if hasattr(self.hand_controller, 'disconnect'):
                    self.hand_controller.disconnect()
                elif hasattr(self.hand_controller, 'serial_connection') and self.hand_controller.serial_connection:
                    self.hand_controller.serial_connection.close()
            self.connected = False
            self.connect_btn.config(text="🔌 Connect Hand Controller")
            self.status_label.config(text="❌ Disconnected", fg=self.warning_color)
            print("🔌 Hand controller disconnected")
    
    def toggle_learning(self):
        """Start/stop learning an emotion."""
        if not self.recording:
            # Start learning
            emotion = self.emotion_entry.get().strip().lower()
            if not emotion:
                self.learn_status.config(text="❌ Please enter an emotion name", fg=self.warning_color)
                return
            
            self.current_emotion = emotion
            self.recording = True
            self.recorded_movements = []
            self.recording_start_time = time.time()
            
            self.learn_btn.config(text="⏹️ Stop Learning", bg=self.warning_color)
            self.learn_status.config(text=f"🔴 LEARNING '{emotion.upper()}' - Move your mouse to express this emotion!", 
                                   fg=self.warning_color)
            print(f"🎓 Started learning emotion: {emotion}")
            
        else:
            # Stop learning
            self.recording = False
            self.learn_btn.config(text="🔴 Start Learning", bg=self.success_color)
            
            if len(self.recorded_movements) >= 50:  # Minimum points for good learning
                # Learn from the movements
                if self.movement_learner:
                    success = self.movement_learner.learn_from_recording(self.current_emotion, self.recorded_movements)
                    if success:
                        self.learn_status.config(text=f"✅ Learned '{self.current_emotion}' from {len(self.recorded_movements)} movements!", 
                                               fg=self.success_color)
                        self.refresh_learned_emotions()
                        # Auto-apply the learned emotion
                        self.emotion_var.set(self.current_emotion)
                        self.apply_emotion()
                    else:
                        self.learn_status.config(text="❌ Learning failed - try more varied movements", fg=self.warning_color)
                else:
                    self.learn_status.config(text="❌ Learning system not available", fg=self.warning_color)
            else:
                needed = 50 - len(self.recorded_movements)
                self.learn_status.config(text=f"❌ Need {needed} more movement points (minimum 50)", fg=self.warning_color)
            
            print(f"⏹️ Stopped learning - captured {len(self.recorded_movements)} movements")
    
    def apply_emotion(self):
        """Apply selected emotion to the AI cursor."""
        if not self.consciousness_cursor or not self.movement_learner:
            print("❌ Core systems not available")
            return
        
        emotion = self.emotion_var.get()
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
            self.learn_status.config(text=f"❌ Emotion '{emotion}' not learned yet", fg=self.warning_color)
    
    def set_consciousness_for_emotion(self, emotion: str):
        """Set consciousness state values to match the emotion."""
        if not self.consciousness_state:
            return
        
        # Basic emotional mappings - you can expand this
        emotion_mappings = {
            "happy": {"mood": 0.8, "novelty": 0.6, "boredom": 0.1},
            "sad": {"mood": -0.7, "novelty": 0.2, "boredom": 0.6},
            "excited": {"mood": 0.9, "novelty": 1.0, "boredom": 0.0},
            "angry": {"mood": -0.5, "novelty": 0.8, "boredom": 0.2},
            "calm": {"mood": 0.3, "novelty": 0.1, "boredom": 0.0},
            "focused": {"mood": 0.2, "novelty": 0.5, "boredom": 0.1},
            "bored": {"mood": -0.2, "novelty": 0.1, "boredom": 0.9},
            "surprised": {"mood": 0.5, "novelty": 1.0, "boredom": 0.0}
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
        
        self.emotion_combo.config(values=emotions)
        if emotions:
            self.emotion_combo.set(emotions[0])
    
    def on_mouse_move(self, event):
        """Handle mouse movement for recording and cursor control."""
        # Normalize coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        x = event.x / canvas_width
        y = event.y / canvas_height
        
        # Clamp to [0, 1]
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
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
                self.learn_status.config(text=f"🔴 LEARNING '{self.current_emotion.upper()}' - {progress} movement points recorded")
        
        # Update consciousness cursor position (for manual mode)
        if self.consciousness_cursor and not self.recording:
            # In expression mode, let consciousness cursor drive itself
            pass
    
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
        
        # Update consciousness cursor
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
        """Update hand physics using the proven wave-based system."""
        # Apply vertical reverse (affects finger calculation, not cursor)
        effective_cursor_y = cursor_y
        
        # Calculate wave-based targets for each finger
        for i in range(self.num_fingers):
            finger_x = (i + 0.5) / self.num_fingers
            
            # Distance from cursor to finger position
            dx = cursor_x - finger_x
            dy = effective_cursor_y - 0.5
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Gravitational wave influence
            wave_influence = math.exp(-distance / self.gravity_width) * self.wave_strength
            
            # Calculate target position
            base_position = self.default_position
            cursor_influence = (effective_cursor_y - 0.5) * 180 * self.cursor_sensitivity
            wave_effect = wave_influence * cursor_influence
            
            target_position = base_position + wave_effect
            target_position = max(0, min(180, target_position))
            
            # Spring-damper physics
            position_error = target_position - self.finger_positions[i]
            spring_force = position_error * self.spring_force
            damping_force = -self.finger_velocities[i] * self.damping
            
            total_force = spring_force + damping_force
            acceleration = total_force / 10.0  # Mass factor
            
            # Update velocity with limits
            self.finger_velocities[i] += acceleration * dt
            self.finger_velocities[i] = max(-self.max_velocity, min(self.max_velocity, self.finger_velocities[i]))
            
            # Update position
            self.finger_positions[i] += self.finger_velocities[i] * dt
            self.finger_positions[i] = max(0, min(180, self.finger_positions[i]))
    
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
        
        # Cursor circle
        r = 10
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="cyan", outline="white", width=2)
        
        # Draw finger positions as bars at the bottom
        bar_height = 100
        bar_width = width / self.num_fingers
        
        for i, pos in enumerate(self.finger_positions):
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            
            # Position as fraction (0-180 degrees -> 0-1)
            pos_fraction = pos / 180.0
            bar_top = height - (pos_fraction * bar_height)
            
            # Color based on position
            intensity = int(255 * pos_fraction)
            color = f"#{intensity:02x}{intensity//2:02x}{255-intensity:02x}"
            
            self.canvas.create_rectangle(x1, bar_top, x2, height, fill=color, outline="white")
            
            # Position label
            self.canvas.create_text((x1 + x2) / 2, height - 10, text=f"{pos:.0f}°", 
                                  fill="white", font=("Arial", 8))
        
        # Status text
        status = f"Mode: {self.current_applied_emotion} | Recording: {'ON' if self.recording else 'OFF'}"
        self.canvas.create_text(10, 10, text=status, fill="white", font=("Arial", 10), anchor="nw")
        
        # Recording indicator
        if self.recording:
            self.canvas.create_oval(width-30, 10, width-10, 30, fill="red", outline="white")
            self.canvas.create_text(width-20, 35, text=f"{len(self.recorded_movements)}", 
                                  fill="red", font=("Arial", 8))
    
    def run(self):
        """Run the application."""
        print("🎭 Essential Emotional Movement System started!")
        print("✨ This is the core of something amazing - pure emotional hand expression!")
        self.root.mainloop()


if __name__ == "__main__":
    app = EmotionalMovementSystem()
    app.run()
