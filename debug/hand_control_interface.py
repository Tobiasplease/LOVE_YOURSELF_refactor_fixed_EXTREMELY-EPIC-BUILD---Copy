#!/usr/bin/env python3
"""
Physics-Based Hand Servo Control Interface

Interactive GUI for controlling hand servos through cursor physics simulation.
Allows real-time parameter tuning and movement pattern recording.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
import json
import math
from typing import Optional, Dict, List

# Import the hand controller
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.hand_expression import HandExpressionController
from config.config import HAND_SERIAL_PORT, BAUD_RATE, CLEAN_CAPTION_OUTPUT


class PhysicsHandInterface:
    """Interactive physics-based hand control interface."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Physics Hand Controller")
        self.root.geometry("800x600")
        
        # Hand controller
        self.hand_controller: Optional[HandExpressionController] = None
        self.connected = False
        
        # Physics simulation state
        self.num_fingers = 5
        self.finger_positions = [90.0] * self.num_fingers  # Current positions
        self.finger_velocities = [0.0] * self.num_fingers  # Current velocities
        self.finger_targets = [90.0] * self.num_fingers    # Target positions from cursor
        
        # Physics parameters (adjustable)
        self.spring_force = tk.DoubleVar(value=25.0)
        self.damping = tk.DoubleVar(value=0.8)
        self.max_velocity = tk.DoubleVar(value=30.0)
        self.cursor_sensitivity = tk.DoubleVar(value=0.5)
        
        # Recording system
        self.recording = False
        self.recorded_movements = []
        self.presets = {}
        
        # Animation state
        self.running = False
        self.last_time = time.time()
        
        self.setup_ui()
        self.start_physics_loop()
    
    def setup_ui(self):
        """Create the user interface."""
        # Connection frame
        conn_frame = ttk.Frame(self.root)
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect to Hand Controller", 
                                     command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Control canvas
        control_frame = ttk.LabelFrame(self.root, text="Hand Control Area")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(control_frame, bg="black", height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        
        # Physics parameters
        params_frame = ttk.LabelFrame(self.root, text="Physics Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Spring force
        ttk.Label(params_frame, text="Spring Force:").grid(row=0, column=0, sticky=tk.W)
        spring_scale = ttk.Scale(params_frame, from_=1.0, to=100.0, 
                                variable=self.spring_force, orient=tk.HORIZONTAL)
        spring_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.spring_force).grid(row=0, column=2)
        
        # Damping
        ttk.Label(params_frame, text="Damping:").grid(row=1, column=0, sticky=tk.W)
        damping_scale = ttk.Scale(params_frame, from_=0.1, to=2.0, 
                                 variable=self.damping, orient=tk.HORIZONTAL)
        damping_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.damping).grid(row=1, column=2)
        
        # Max velocity
        ttk.Label(params_frame, text="Max Velocity:").grid(row=2, column=0, sticky=tk.W)
        velocity_scale = ttk.Scale(params_frame, from_=5.0, to=100.0, 
                                  variable=self.max_velocity, orient=tk.HORIZONTAL)
        velocity_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.max_velocity).grid(row=2, column=2)
        
        # Cursor sensitivity
        ttk.Label(params_frame, text="Cursor Sensitivity:").grid(row=3, column=0, sticky=tk.W)
        sens_scale = ttk.Scale(params_frame, from_=0.1, to=2.0, 
                              variable=self.cursor_sensitivity, orient=tk.HORIZONTAL)
        sens_scale.grid(row=3, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.cursor_sensitivity).grid(row=3, column=2)
        
        params_frame.columnconfigure(1, weight=1)
        
        # Recording controls
        record_frame = ttk.LabelFrame(self.root, text="Recording & Presets")
        record_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.record_btn = ttk.Button(record_frame, text="Start Recording", 
                                    command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT)
        
        ttk.Button(record_frame, text="Save Preset", 
                  command=self.save_preset).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(record_frame, text="Load Preset", 
                  command=self.load_preset).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(record_frame, text="Reset to Center", 
                  command=self.reset_to_center).pack(side=tk.LEFT, padx=5)
    
    def toggle_connection(self):
        """Connect or disconnect from hand controller."""
        if not self.connected:
            try:
                self.hand_controller = HandExpressionController(
                    port=HAND_SERIAL_PORT, 
                    baudrate=BAUD_RATE, 
                    clean_output=True
                )
                self.connected = True
                self.connect_btn.config(text="Disconnect")
                self.status_label.config(text="Connected", foreground="green")
                print("✅ Connected to hand controller")
            except Exception as e:
                messagebox.showerror("Connection Error", f"Failed to connect: {e}")
        else:
            if self.hand_controller:
                self.hand_controller.cleanup()
                self.hand_controller = None
            self.connected = False
            self.connect_btn.config(text="Connect to Hand Controller")
            self.status_label.config(text="Disconnected", foreground="red")
            print("❌ Disconnected from hand controller")
    
    def on_mouse_move(self, event):
        """Handle mouse movement to control finger targets."""
        if not self.connected:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Convert mouse position to finger targets
        x_norm = event.x / canvas_width
        y_norm = 1.0 - (event.y / canvas_height)  # Invert Y
        
        # Map to servo ranges with sensitivity
        sensitivity = self.cursor_sensitivity.get()
        
        # Different mapping strategies for fingers
        for i in range(self.num_fingers):
            if i < 2:  # First two fingers follow X more
                target = 90 + (x_norm - 0.5) * 180 * sensitivity
            elif i < 4:  # Middle fingers follow Y more  
                target = 90 + (y_norm - 0.5) * 180 * sensitivity
            else:  # Last finger follows combination
                target = 90 + ((x_norm + y_norm) / 2 - 0.5) * 180 * sensitivity
            
            self.finger_targets[i] = max(0, min(180, target))
        
        # Record if recording
        if self.recording:
            self.recorded_movements.append({
                'time': time.time(),
                'targets': self.finger_targets.copy(),
                'mouse_pos': (x_norm, y_norm)
            })
    
    def on_mouse_click(self, event):
        """Handle mouse clicks for special actions."""
        # Could add click-based gestures here
        pass
    
    def start_physics_loop(self):
        """Start the physics simulation loop."""
        self.running = True
        self.physics_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.physics_thread.start()
        
        # Start UI update loop
        self.update_ui()
    
    def physics_loop(self):
        """Main physics simulation loop."""
        while self.running:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Limit timestep to prevent instability
            dt = min(dt, 0.05)
            
            # Update physics for each finger
            for i in range(self.num_fingers):
                # Spring force towards target
                target = self.finger_targets[i]
                position = self.finger_positions[i]
                velocity = self.finger_velocities[i]
                
                force = (target - position) * self.spring_force.get()
                
                # Apply force to velocity
                velocity += force * dt
                
                # Apply damping
                velocity *= (1.0 - self.damping.get() * dt)
                
                # Limit velocity
                max_vel = self.max_velocity.get()
                velocity = max(-max_vel, min(max_vel, velocity))
                
                # Update position
                position += velocity * dt
                position = max(0, min(180, position))
                
                # Store updated values
                self.finger_positions[i] = position
                self.finger_velocities[i] = velocity
            
            # Send to hardware if connected
            if self.connected and self.hand_controller:
                try:
                    # Update the hand controller's internal state
                    for i in range(self.num_fingers):
                        if i < len(self.hand_controller.finger_positions):
                            self.hand_controller.finger_positions[i] = self.finger_positions[i]
                    
                    # Send positions to servos
                    self.hand_controller._send_positions()
                except Exception as e:
                    print(f"Hardware update error: {e}")
            
            time.sleep(0.016)  # ~60 FPS
    
    def update_ui(self):
        """Update the visual interface."""
        if not self.running:
            return
            
        # Clear canvas
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Draw finger positions as bars
            bar_width = canvas_width // self.num_fingers
            
            for i in range(self.num_fingers):
                x = i * bar_width
                position_norm = self.finger_positions[i] / 180.0
                target_norm = self.finger_targets[i] / 180.0
                
                # Current position bar (blue)
                pos_height = position_norm * canvas_height
                self.canvas.create_rectangle(
                    x, canvas_height - pos_height, x + bar_width - 2, canvas_height,
                    fill="lightblue", outline="blue"
                )
                
                # Target position line (red)
                target_y = canvas_height - (target_norm * canvas_height)
                self.canvas.create_line(
                    x, target_y, x + bar_width - 2, target_y,
                    fill="red", width=2
                )
                
                # Finger label
                self.canvas.create_text(
                    x + bar_width // 2, canvas_height - 10,
                    text=f"F{i+1}", fill="white"
                )
        
        # Schedule next update
        self.root.after(50, self.update_ui)  # 20 FPS for UI
    
    def toggle_recording(self):
        """Start or stop recording movements."""
        if not self.recording:
            self.recording = True
            self.recorded_movements = []
            self.record_btn.config(text="Stop Recording")
            print("🔴 Recording started")
        else:
            self.recording = False
            self.record_btn.config(text="Start Recording")
            print(f"⏹️ Recording stopped - {len(self.recorded_movements)} samples")
    
    def save_preset(self):
        """Save current parameters and recorded movements as a preset."""
        name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
        if name:
            preset = {
                'spring_force': self.spring_force.get(),
                'damping': self.damping.get(),
                'max_velocity': self.max_velocity.get(),
                'cursor_sensitivity': self.cursor_sensitivity.get(),
                'movements': self.recorded_movements.copy()
            }
            
            # Save to file
            try:
                with open(f"hand_preset_{name}.json", 'w') as f:
                    json.dump(preset, f, indent=2)
                messagebox.showinfo("Success", f"Preset '{name}' saved!")
                print(f"💾 Preset '{name}' saved with {len(self.recorded_movements)} movements")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def load_preset(self):
        """Load a saved preset."""
        # This would open a file dialog in a full implementation
        # For now, just implement basic loading
        pass
    
    def reset_to_center(self):
        """Reset all fingers to center position."""
        for i in range(self.num_fingers):
            self.finger_targets[i] = 90.0
            self.finger_positions[i] = 90.0
            self.finger_velocities[i] = 0.0
        print("🔄 Reset to center position")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.connected and self.hand_controller:
            self.hand_controller.cleanup()
    
    def run(self):
        """Run the interface."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.cleanup()
    
    def on_closing(self):
        """Handle window closing."""
        self.cleanup()
        self.root.destroy()


if __name__ == "__main__":
    print("🎮 Starting Physics-Based Hand Controller Interface...")
    print("Move your mouse over the control area to control finger positions!")
    print("Use the sliders to adjust physics parameters in real-time.")
    
    try:
        interface = PhysicsHandInterface()
        interface.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
