#!/usr/bin/env python3
"""
Conscious Cursor Interface - CLEAN MARKOV ONLY
==============================================

SIMPLE working system with ONLY high-resolution Markov chains.
NO vector synthesis, NO complex algorithms - just pure recorded data playback.

Core Features:
- Direct cursor→4 servo control (WORKING)
- Record your movements with full timing data
- Simple Markov chain generation from YOUR recorded patterns
- 5 emotional states with clean switching
- Real playback of your movement language

Author: Clean Markov Chain System
"""

import tkinter as tk
from tkinter import ttk
import time
import math
import random
import json
import os
from typing import Optional, Dict, List, Tuple

# Import hand controller - Direct import since we're in the same directory
try:
    from hand_expression import HandExpressionController
    HAND_CONTROLLER_AVAILABLE = True
except ImportError:
    print("⚠️ Hand controller not available - simulation mode")
    HAND_CONTROLLER_AVAILABLE = False


class SimpleMarkovChain:
    """SIMPLE Markov chain that learns from recorded movement data."""
    
    def __init__(self, emotion_name: str):
        self.emotion_name = emotion_name
        self.transition_table = {}  # (current_state) -> [(next_state, probability), ...]
        self.states = []  # List of (x, y, dt) tuples
        self.current_state_index = 0
        
    def learn_from_recording(self, movements: List[Dict]) -> bool:
        """Learn transition patterns from recorded movements."""
        if len(movements) < 10:
            print(f"❌ Not enough data to learn from ({len(movements)} points)")
            return False
        
        print(f"🧠 Learning Markov transitions for {self.emotion_name} from {len(movements)} points...")
        
        # Convert movements to simple states
        self.states = []
        for i, movement in enumerate(movements):
            x = round(movement['x'], 3)  # Round to prevent state explosion
            y = round(movement['y'], 3)
            dt = movement.get('dt', 0.016)  # Default 60fps timing
            self.states.append((x, y, dt))
        
        # Build transition table
        self.transition_table = {}
        
        for i in range(len(self.states) - 1):
            current_state = self.states[i]
            next_state = self.states[i + 1]
            
            if current_state not in self.transition_table:
                self.transition_table[current_state] = []
            
            self.transition_table[current_state].append(next_state)
        
        # Convert to probabilities (for now, just store all transitions)
        print(f"✅ Learned {len(self.transition_table)} transition states")
        return True
    
    def generate_next_position(self, current_x: float, current_y: float) -> Tuple[float, float]:
        """Generate next position based on learned patterns."""
        if not self.states or not self.transition_table:
            return current_x, current_y
        
        # Find closest state to current position
        current_state = self._find_closest_state(current_x, current_y)
        
        if current_state in self.transition_table:
            # Pick a random transition from this state
            transitions = self.transition_table[current_state]
            if transitions:
                next_state = random.choice(transitions)
                return next_state[0], next_state[1]  # Return x, y
        
        # Fallback: continue in current direction with small random variation
        dx = random.uniform(-0.01, 0.01)
        dy = random.uniform(-0.01, 0.01)
        return current_x + dx, current_y + dy
    
    def _find_closest_state(self, x: float, y: float) -> Tuple[float, float, float]:
        """Find the closest learned state to current position."""
        if not self.states:
            return (x, y, 0.016)
        
        min_distance = float('inf')
        closest_state = self.states[0]
        
        for state in self.states:
            state_x, state_y, state_dt = state
            distance = (x - state_x) ** 2 + (y - state_y) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_state = state
        
        return closest_state


class CleanCursorInterface:
    """Clean cursor interface with ONLY Markov chain functionality."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 Clean Markov Cursor Control")
        self.root.geometry("700x600")
        
        # Hand controller
        self.hand_controller: Optional[HandExpressionController] = None
        self.connected = False
        
        # Movement recording
        self.recording = False
        self.recorded_movements = []
        self.recording_start_time = 0
        self.last_movement_time = 0
        
        # Markov chain generation
        self.generating = False
        self.markov_chains = {}  # emotion -> SimpleMarkovChain
        self.current_chain = None
        self.generation_start_time = 0
        
        # Emotional states
        self.emotions = ["neutral", "happy", "sad", "excited", "focused"]
        self.current_emotion = "neutral"
        
        # Physics simulation state
        self.num_fingers = 4
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_velocities = [0.0] * self.num_fingers
        self.finger_targets = [90.0] * self.num_fingers
        
        # Physics parameters
        self.spring_force = 500.0
        self.damping = 0.1
        self.max_velocity = 1000.0
        self.cursor_sensitivity = 3.0
        
        # Mouse tracking
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        
        # Animation state
        self.running = False
        self.last_time = time.time()
        self.last_send_time = 0
        self.send_interval = 0.016  # 60 Hz
        
        self.setup_ui()
        self.start_physics_loop()
        
        # Load any existing chains
        self.load_markov_chains()
        
    def setup_ui(self):
        """Create the simplified user interface."""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === CONNECTION FRAME ===
        conn_frame = tk.LabelFrame(main_frame, text="🔌 Connection", padx=5, pady=5)
        conn_frame.pack(fill=tk.X, pady=5)
        
        self.connect_btn = tk.Button(conn_frame, text="Connect to Hand Controller", 
                                   command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(conn_frame, text="❌ Disconnected", fg="red")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # === EMOTION SELECTION ===
        emotion_frame = tk.LabelFrame(main_frame, text="🎭 Emotion Selection", padx=5, pady=5)
        emotion_frame.pack(fill=tk.X, pady=5)
        
        self.emotion_buttons = {}
        button_row = tk.Frame(emotion_frame)
        button_row.pack()
        
        for emotion in self.emotions:
            btn = tk.Button(button_row, text=emotion.title(), width=10,
                          command=lambda e=emotion: self.set_emotion(e))
            btn.pack(side=tk.LEFT, padx=2)
            self.emotion_buttons[emotion] = btn
        
        self.current_emotion_label = tk.Label(emotion_frame, text=f"Current: {self.current_emotion}", 
                                            font=("Arial", 10, "bold"))
        self.current_emotion_label.pack(pady=5)
        
        # === RECORDING CONTROLS ===
        record_frame = tk.LabelFrame(main_frame, text="🔴 Recording", padx=5, pady=5)
        record_frame.pack(fill=tk.X, pady=5)
        
        self.record_btn = tk.Button(record_frame, text="🔴 Start Recording", 
                                  command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.record_status = tk.Label(record_frame, text="Ready to record", fg="gray")
        self.record_status.pack(side=tk.LEFT, padx=10)
        
        # === MARKOV GENERATION ===
        markov_frame = tk.LabelFrame(main_frame, text="🧠 Markov Generation", padx=5, pady=5)
        markov_frame.pack(fill=tk.X, pady=5)
        
        self.generate_btn = tk.Button(markov_frame, text="🧠 Start Markov Chain", 
                                    command=self.toggle_markov_generation)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        self.markov_status = tk.Label(markov_frame, text="No chains learned yet", fg="gray")
        self.markov_status.pack(side=tk.LEFT, padx=10)
        
        # Clear data button
        self.clear_btn = tk.Button(markov_frame, text="🗑️ Clear Data", 
                                 command=self.clear_learned_data)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)
        
        # === HAND CONTROL AREA ===
        control_frame = tk.LabelFrame(main_frame, text="🎯 Hand Control Area", padx=5, pady=5)
        control_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas = tk.Canvas(control_frame, bg="black", height=300, width=600)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        
        # === PHYSICS PARAMETERS ===
        physics_frame = tk.LabelFrame(main_frame, text="⚙️ Physics Parameters", padx=5, pady=5)
        physics_frame.pack(fill=tk.X, pady=5)
        
        # Spring Force
        spring_row = tk.Frame(physics_frame)
        spring_row.pack(fill=tk.X, pady=2)
        tk.Label(spring_row, text="Spring Force:", width=15).pack(side=tk.LEFT)
        self.spring_scale = tk.Scale(spring_row, from_=100, to=1000, orient=tk.HORIZONTAL,
                                   command=self.update_spring_force)
        self.spring_scale.set(self.spring_force)
        self.spring_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Damping
        damping_row = tk.Frame(physics_frame)
        damping_row.pack(fill=tk.X, pady=2)
        tk.Label(damping_row, text="Damping:", width=15).pack(side=tk.LEFT)
        self.damping_scale = tk.Scale(damping_row, from_=0.01, to=1.0, resolution=0.01, 
                                    orient=tk.HORIZONTAL, command=self.update_damping)
        self.damping_scale.set(self.damping)
        self.damping_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Update emotion button colors
        self.update_emotion_buttons()
        
    def set_emotion(self, emotion: str):
        """Set the current emotion."""
        self.current_emotion = emotion
        self.current_emotion_label.config(text=f"Current: {emotion}")
        self.update_emotion_buttons()
        
        # Update Markov status
        if emotion in self.markov_chains:
            chain = self.markov_chains[emotion]
            self.markov_status.config(text=f"{emotion}: {len(chain.states)} learned states", fg="green")
        else:
            self.markov_status.config(text=f"{emotion}: No data recorded yet", fg="orange")
        
        print(f"🎭 Switched to emotion: {emotion}")
        
    def update_emotion_buttons(self):
        """Update emotion button appearance."""
        for emotion, btn in self.emotion_buttons.items():
            if emotion == self.current_emotion:
                btn.config(bg="lightblue", relief="sunken")
            else:
                btn.config(bg="lightgray", relief="raised")
    
    def toggle_connection(self):
        """Toggle connection to hand controller."""
        if not HAND_CONTROLLER_AVAILABLE:
            self.status_label.config(text="❌ Hand controller not available", fg="red")
            return
            
        if not self.connected:
            try:
                self.hand_controller = HandExpressionController()
                self.connected = True
                self.connect_btn.config(text="Disconnect")
                self.status_label.config(text="✅ Connected", fg="green")
                print("✅ Connected to hand controller")
            except Exception as e:
                self.status_label.config(text=f"❌ Connection failed: {e}", fg="red")
                print(f"❌ Connection failed: {e}")
        else:
            self.connected = False
            self.hand_controller = None
            self.connect_btn.config(text="Connect to Hand Controller")
            self.status_label.config(text="❌ Disconnected", fg="red")
            print("❌ Disconnected from hand controller")
    
    def toggle_recording(self):
        """Toggle movement recording."""
        if not self.recording:
            # Start recording
            self.recording = True
            self.recorded_movements = []
            self.recording_start_time = time.time()
            self.last_movement_time = self.recording_start_time
            
            self.record_btn.config(text="⏹️ Stop Recording")
            self.record_status.config(text=f"Recording {self.current_emotion}...", fg="red")
            
            print(f"🔴 Started recording for emotion: {self.current_emotion}")
        else:
            # Stop recording
            self.recording = False
            self.record_btn.config(text="🔴 Start Recording")
            
            if len(self.recorded_movements) >= 10:
                # Learn from the recording
                self.learn_markov_chain()
                self.record_status.config(text=f"Learned {len(self.recorded_movements)} movements", fg="green")
                print(f"✅ Recorded {len(self.recorded_movements)} movements for {self.current_emotion}")
            else:
                self.record_status.config(text="Recording too short - need more movements", fg="orange")
                print(f"❌ Recording too short: {len(self.recorded_movements)} movements")
    
    def learn_markov_chain(self):
        """Learn Markov chain from recorded movements."""
        if not self.recorded_movements:
            return
            
        # Create or update Markov chain for current emotion
        if self.current_emotion not in self.markov_chains:
            self.markov_chains[self.current_emotion] = SimpleMarkovChain(self.current_emotion)
        
        chain = self.markov_chains[self.current_emotion]
        success = chain.learn_from_recording(self.recorded_movements)
        
        if success:
            self.save_markov_chains()
            self.markov_status.config(text=f"{self.current_emotion}: {len(chain.states)} learned states", fg="green")
            print(f"🧠 Learned Markov chain for {self.current_emotion}")
        else:
            self.markov_status.config(text="Learning failed - not enough data", fg="red")
    
    def toggle_markov_generation(self):
        """Toggle Markov chain generation."""
        if not self.generating:
            # Start generation
            if self.current_emotion not in self.markov_chains:
                self.markov_status.config(text="No data for this emotion - record first!", fg="orange")
                return
            
            self.generating = True
            self.current_chain = self.markov_chains[self.current_emotion]
            self.generation_start_time = time.time()
            
            self.generate_btn.config(text="⏹️ Stop Generation")
            self.markov_status.config(text=f"Generating {self.current_emotion} movement...", fg="purple")
            
            print(f"🧠 Started Markov generation for {self.current_emotion}")
        else:
            # Stop generation
            self.generating = False
            self.current_chain = None
            
            self.generate_btn.config(text="🧠 Start Markov Chain")
            chain = self.markov_chains.get(self.current_emotion)
            if chain:
                self.markov_status.config(text=f"{self.current_emotion}: {len(chain.states)} learned states", fg="green")
            else:
                self.markov_status.config(text="No chains learned yet", fg="gray")
            
            print("⏹️ Stopped Markov generation")
    
    def clear_learned_data(self):
        """Clear all learned Markov chains."""
        self.markov_chains = {}
        self.generating = False
        self.current_chain = None
        
        self.generate_btn.config(text="🧠 Start Markov Chain")
        self.markov_status.config(text="No chains learned yet", fg="gray")
        
        # Clear saved files
        try:
            if os.path.exists("markov_chains.json"):
                os.remove("markov_chains.json")
        except Exception as e:
            print(f"Error clearing saved data: {e}")
        
        print("🗑️ Cleared all learned data")
    
    def save_markov_chains(self):
        """Save learned Markov chains to file."""
        try:
            save_data = {}
            for emotion, chain in self.markov_chains.items():
                save_data[emotion] = {
                    'states': chain.states,
                    'transition_table': {str(k): v for k, v in chain.transition_table.items()}
                }
            
            with open("markov_chains.json", "w") as f:
                json.dump(save_data, f)
            
            print("💾 Saved Markov chains to file")
        except Exception as e:
            print(f"Error saving Markov chains: {e}")
    
    def load_markov_chains(self):
        """Load saved Markov chains from file."""
        try:
            if not os.path.exists("markov_chains.json"):
                return
                
            with open("markov_chains.json", "r") as f:
                save_data = json.load(f)
            
            for emotion, data in save_data.items():
                chain = SimpleMarkovChain(emotion)
                chain.states = [tuple(state) for state in data['states']]
                
                # Rebuild transition table
                chain.transition_table = {}
                for k_str, v in data['transition_table'].items():
                    k = eval(k_str)  # Convert string back to tuple
                    chain.transition_table[k] = [tuple(state) for state in v]
                
                self.markov_chains[emotion] = chain
            
            print(f"📁 Loaded {len(self.markov_chains)} Markov chains from file")
            
            # Update status for current emotion
            if self.current_emotion in self.markov_chains:
                chain = self.markov_chains[self.current_emotion]
                self.markov_status.config(text=f"{self.current_emotion}: {len(chain.states)} learned states", fg="green")
                
        except Exception as e:
            print(f"Error loading Markov chains: {e}")
    
    def on_mouse_move(self, event):
        """Handle mouse movement on canvas."""
        if not self.generating:  # Only respond to manual movement when not generating
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 0 and canvas_height > 0:
                self.mouse_x = event.x / canvas_width
                self.mouse_y = event.y / canvas_height
                
                # Clamp to [0, 1]
                self.mouse_x = max(0, min(1, self.mouse_x))
                self.mouse_y = max(0, min(1, self.mouse_y))
                
                # Record movement if recording
                if self.recording:
                    current_time = time.time()
                    dt = current_time - self.last_movement_time
                    
                    movement = {
                        'x': self.mouse_x,
                        'y': self.mouse_y,
                        'time': current_time - self.recording_start_time,
                        'dt': dt
                    }
                    
                    self.recorded_movements.append(movement)
                    self.last_movement_time = current_time
    
    def on_mouse_click(self, event):
        """Handle mouse click on canvas."""
        # Update position on click too
        self.on_mouse_move(event)
    
    def update_spring_force(self, value):
        """Update spring force parameter."""
        self.spring_force = float(value)
    
    def update_damping(self, value):
        """Update damping parameter."""
        self.damping = float(value)
    
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
        
        # Update Markov generation if active
        if self.generating and self.current_chain:
            self.update_markov_generation()
        
        # Update physics simulation
        self.update_physics(dt)
        
        # Send to servos if connected
        if self.connected and self.hand_controller:
            self.send_to_servos()
        
        # Update display
        self.update_display()
        
        # Schedule next update
        self.root.after(16, self.physics_loop)  # ~60 FPS
    
    def update_markov_generation(self):
        """Update cursor position using Markov chain generation."""
        if not self.current_chain:
            return
            
        # Generate next position
        new_x, new_y = self.current_chain.generate_next_position(self.mouse_x, self.mouse_y)
        
        # Clamp to canvas bounds
        self.mouse_x = max(0, min(1, new_x))
        self.mouse_y = max(0, min(1, new_y))
    
    def update_physics(self, dt):
        """Update physics simulation."""
        # Convert cursor position to finger targets
        canvas_width = 563
        canvas_height = 304
        
        # Map cursor to finger positions with wave-like influence
        for i in range(self.num_fingers):
            # Calculate horizontal position for this finger
            finger_ratio = i / max(1, self.num_fingers - 1) if self.num_fingers > 1 else 0.5
            
            # Calculate distance from cursor to this finger's horizontal position
            distance = abs(self.mouse_x - finger_ratio)
            
            # Apply wave-like influence (closer = more influence)
            gravity_width = 0.4
            influence = max(0, 1 - distance / gravity_width)
            
            # Calculate target position
            wave_strength = 2.0
            cursor_influence = (self.mouse_y - 0.5) * wave_strength * influence
            default_position = 90.0
            
            target = default_position + cursor_influence * self.cursor_sensitivity * 30
            
            # Clamp to servo limits
            self.finger_targets[i] = max(0, min(180, target))
        
        # Apply physics simulation
        for i in range(self.num_fingers):
            # Spring force towards target
            spring_force = (self.finger_targets[i] - self.finger_positions[i]) * self.spring_force * dt
            
            # Damping force
            damping_force = -self.finger_velocities[i] * self.damping
            
            # Update velocity
            self.finger_velocities[i] += spring_force + damping_force
            
            # Limit velocity
            self.finger_velocities[i] = max(-self.max_velocity, min(self.max_velocity, self.finger_velocities[i]))
            
            # Update position
            self.finger_positions[i] += self.finger_velocities[i] * dt
            
            # Clamp position
            self.finger_positions[i] = max(0, min(180, self.finger_positions[i]))
    
    def send_to_servos(self):
        """Send positions to servo controller."""
        current_time = time.time()
        if current_time - self.last_send_time < self.send_interval:
            return
            
        try:
            positions = [int(pos) for pos in self.finger_positions]
            self.hand_controller.send_positions(positions)
            self.last_send_time = current_time
        except Exception as e:
            print(f"Error sending to servos: {e}")
    
    def update_display(self):
        """Update the canvas display."""
        self.canvas.delete("all")
        
        # Draw cursor position
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            x = self.mouse_x * canvas_width
            y = self.mouse_y * canvas_height
            
            # Cursor dot
            radius = 8
            if self.generating:
                color = "purple"  # Purple when generating
            elif self.recording:
                color = "red"     # Red when recording
            else:
                color = "white"   # White for manual control
                
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, 
                                  fill=color, outline="white", width=2)
            
            # Draw finger influence zones
            for i in range(self.num_fingers):
                finger_x = (i / max(1, self.num_fingers - 1)) * canvas_width if self.num_fingers > 1 else canvas_width / 2
                finger_y = (1 - self.finger_positions[i] / 180) * canvas_height
                
                # Finger position indicator
                self.canvas.create_oval(finger_x-6, finger_y-6, finger_x+6, finger_y+6, 
                                      fill="orange", outline="yellow", width=1)
                
                # Finger number
                self.canvas.create_text(finger_x, finger_y-15, text=str(i+1), 
                                      fill="yellow", font=("Arial", 8))


def main():
    """Main function to run the application."""
    print("🎯 Starting Clean Markov Cursor Interface...")
    app = CleanCursorInterface()
    
    try:
        app.root.mainloop()
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        print("👋 Application closed")


if __name__ == "__main__":
    main()
