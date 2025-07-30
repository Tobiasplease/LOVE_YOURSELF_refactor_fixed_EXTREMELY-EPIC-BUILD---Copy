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
        
        # Physics simulation state - CORRECTED TO 4 SERVOS
        self.num_fingers = 4  # Fixed: Only 4 servos available (pins 8,9,10,11)
        self.finger_positions = [90.0] * self.num_fingers  # Current positions
        self.finger_velocities = [0.0] * self.num_fingers  # Current velocities
        self.finger_targets = [90.0] * self.num_fingers    # Target positions from cursor
        
        # Physics parameters (optimized for MAXIMUM responsiveness)
        self.spring_force = tk.DoubleVar(value=200.0)    # Very high for instant response
        self.damping = tk.DoubleVar(value=0.05)          # Minimal damping
        self.max_velocity = tk.DoubleVar(value=500.0)    # Very high max velocity
        self.cursor_sensitivity = tk.DoubleVar(value=2.0) # High sensitivity
        
        # Wave control parameters
        self.wave_strength = tk.DoubleVar(value=1.0)     # Overall wave amplitude
        self.gravity_width = tk.DoubleVar(value=0.3)     # Width of gravitational influence (0.1-1.0)
        self.default_position = tk.DoubleVar(value=90.0) # Default finger position (10-170)
        self.reverse_vertical = tk.BooleanVar(value=False) # Reverse vertical control
        
        # Physics mode toggle (False = Direct, True = Physics)
        self.physics_mode = tk.BooleanVar(value=False)   # Start in direct mode
        
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
        # Connection and mode controls
        conn_frame = ttk.Frame(self.root)
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect to Hand Controller", 
                                     command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Override mode toggle
        self.override_mode = tk.BooleanVar(value=False)
        override_cb = ttk.Checkbutton(conn_frame, text="Override Automatic Control", 
                                     variable=self.override_mode,
                                     command=self.on_override_toggle)
        override_cb.pack(side=tk.RIGHT)
        
        self.override_label = ttk.Label(conn_frame, text="Manual Mode: OFF", foreground="blue")
        self.override_label.pack(side=tk.RIGHT, padx=(0, 10))
        
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
        sens_scale = ttk.Scale(params_frame, from_=0.1, to=3.0, 
                              variable=self.cursor_sensitivity, orient=tk.HORIZONTAL)
        sens_scale.grid(row=3, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.cursor_sensitivity).grid(row=3, column=2)
        
        # Wave strength
        ttk.Label(params_frame, text="Wave Strength:").grid(row=4, column=0, sticky=tk.W)
        wave_scale = ttk.Scale(params_frame, from_=0.1, to=3.0, 
                              variable=self.wave_strength, orient=tk.HORIZONTAL)
        wave_scale.grid(row=4, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.wave_strength).grid(row=4, column=2)
        
        # Gravity field width
        ttk.Label(params_frame, text="Gravity Width:").grid(row=5, column=0, sticky=tk.W)
        width_scale = ttk.Scale(params_frame, from_=0.1, to=1.0, 
                               variable=self.gravity_width, orient=tk.HORIZONTAL)
        width_scale.grid(row=5, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.gravity_width).grid(row=5, column=2)
        
        # Default position
        ttk.Label(params_frame, text="Default Position:").grid(row=6, column=0, sticky=tk.W)
        default_scale = ttk.Scale(params_frame, from_=10.0, to=170.0, 
                                 variable=self.default_position, orient=tk.HORIZONTAL)
        default_scale.grid(row=6, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.default_position).grid(row=6, column=2)
        
        # Reverse vertical toggle
        reverse_cb = ttk.Checkbutton(params_frame, text="Reverse Vertical Control", 
                                    variable=self.reverse_vertical)
        reverse_cb.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=2)
        
        # PHYSICS MODE TOGGLE - FIXED!
        physics_cb = ttk.Checkbutton(params_frame, text="🎛️ Physics Mode (Spring Simulation)", 
                                    variable=self.physics_mode)
        physics_cb.grid(row=8, column=0, columnspan=3, sticky=tk.W, pady=5)
        
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
        
        ttk.Button(record_frame, text="Reset to Default", 
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
    
    def on_override_toggle(self):
        """Handle override mode toggle."""
        if self.override_mode.get():
            self.override_label.config(text="Manual Mode: ON", foreground="red")
            if self.hand_controller:
                self.hand_controller.enable_manual_override()
                print("🔧 Manual override enabled - automatic consciousness control paused")
        else:
            self.override_label.config(text="Manual Mode: OFF", foreground="blue")
            if self.hand_controller:
                self.hand_controller.disable_manual_override()
                print("🤖 Manual override disabled - consciousness control resumed")
    
    def on_mouse_move(self, event):
        """Handle mouse movement to control finger targets with wave-like gravitational field."""
        if not self.connected:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Convert mouse position to normalized coordinates
        x_norm = event.x / canvas_width
        y_norm = 1.0 - (event.y / canvas_height)  # Invert Y so up = 1.0, down = 0.0
        
        # Get control parameters
        sensitivity = self.cursor_sensitivity.get()
        wave_strength = self.wave_strength.get()
        gravity_width = self.gravity_width.get()
        default_pos = self.default_position.get()
        reverse_vertical = self.reverse_vertical.get()
        
        # Apply vertical reversal at the coordinate level to flip interface logic
        if reverse_vertical:
            y_norm = 1.0 - y_norm  # Flip the Y coordinate completely
        
        # WAVE-BASED CONTROL - CURSOR AS ELEVATION CONTROLLER:
        # The cursor Y position sets the "elevation" at that X position
        # Fingers rise/fall based on their proximity to the cursor X position
        
        # Define finger positions along X axis (0.0 to 1.0)
        finger_x_positions = [0.2, 0.4, 0.6, 0.8]  # Index, Middle, Ring, Pinky
        
        # Calculate cursor elevation effect (Y position determines the "height" of the wave)
        # Now the reversal is handled at the coordinate level above
        cursor_elevation = (y_norm - 0.5) * 80 * sensitivity
        
        for i in range(self.num_fingers):
            # Calculate distance from cursor X to this finger's X position
            distance = abs(x_norm - finger_x_positions[i])
            
            # Start with user-defined default position instead of fixed center
            target = default_pos
            
            # Calculate wave influence based on distance from cursor
            if distance <= gravity_width:
                # Normalized distance within the gravity field (0.0 = directly under cursor, 1.0 = edge)
                norm_distance = distance / gravity_width
                
                # Use a smoother falloff function for more natural wave
                # Cosine falloff creates a nice bell curve
                influence = math.cos(norm_distance * math.pi / 2) ** 2  # Smooth 0-1 falloff
                
                # Apply cursor elevation scaled by influence
                # This creates the wave effect - fingers closer to cursor get more elevation
                wave_effect = cursor_elevation * influence * wave_strength
                target += wave_effect
            
            # Clamp to servo range
            self.finger_targets[i] = max(10, min(170, target))
        
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
                if not self.physics_mode.get():  # Direct mode when physics_mode is False
                    # DIRECT MODE: Instant response, no physics lag!
                    self.finger_positions[i] = self.finger_targets[i]
                    self.finger_velocities[i] = 0.0  # Reset velocity
                else:
                    # PHYSICS MODE: Spring-damper simulation
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
                    position = max(10, min(170, position))  # Safer servo range
                    
                    # Store updated values
                    self.finger_positions[i] = position
                    self.finger_velocities[i] = velocity
            
            # Send to hardware if connected AND in override mode
            if self.connected and self.hand_controller and self.override_mode.get():
                try:
                    # THROTTLE COMMANDS - only send if positions changed significantly
                    current_time = time.time()
                    
                    # Initialize previous positions and last send time
                    if not hasattr(self, '_last_sent_positions'):
                        self._last_sent_positions = [90, 90, 90, 90]
                        self._last_send_time = 0
                    
                    # Check if positions changed enough to warrant sending
                    position_changed = False
                    min_change_threshold = 3  # Only send if any finger moved >3 degrees
                    min_time_interval = 0.05  # Send at most every 50ms (20 Hz max)
                    
                    for i in range(self.num_fingers):
                        if abs(self.finger_positions[i] - self._last_sent_positions[i]) > min_change_threshold:
                            position_changed = True
                            break
                    
                    # Send if positions changed significantly OR enough time has passed
                    if position_changed or (current_time - self._last_send_time > min_time_interval):
                        positions_int = [int(round(pos)) for pos in self.finger_positions]
                        
                        # Determine mode for debug output
                        mode = "PHYSICS" if self.physics_mode.get() else "DIRECT"
                        print(f"🤖 [{mode}] SENT Hand positions: {positions_int}")
                        
                        # Actually send the command
                        self.hand_controller.set_hand_positions(positions_int)
                        
                        # Update tracking variables
                        self._last_sent_positions = positions_int.copy()
                        self._last_send_time = current_time
                        
                except Exception as e:
                    print(f"❌ Failed to send hand positions: {e}")
            
            # Sleep to limit update rate
            time.sleep(0.001)  # 1000 FPS physics loop
    
    def update_ui(self):
        """Update UI elements."""
        if self.running:
            # Draw current finger positions on canvas
            self.draw_finger_positions()
            self.root.after(16, self.update_ui)  # ~60 FPS UI updates
    
    def draw_finger_positions(self):
        """Draw finger positions on the canvas."""
        self.canvas.delete("finger")
        
        if not self.connected:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Draw finger positions as vertical bars
        finger_x_positions = [0.2, 0.4, 0.6, 0.8]
        
        for i, x_pos in enumerate(finger_x_positions):
            x_pixel = x_pos * canvas_width
            
            # Convert servo position (10-170) to normalized height (0-1)
            servo_pos = self.finger_positions[i]
            height_norm = (servo_pos - 10) / 160.0
            y_pixel = canvas_height * (1.0 - height_norm)  # Invert for display
            
            # Draw finger position as a line
            self.canvas.create_line(
                x_pixel, canvas_height, x_pixel, y_pixel,
                fill="cyan", width=8, tags="finger"
            )
            
            # Draw finger label
            self.canvas.create_text(
                x_pixel, canvas_height - 15,
                text=f"F{i+1}\n{int(servo_pos)}°",
                fill="white", font=("Arial", 8), tags="finger"
            )
    
    def toggle_recording(self):
        """Toggle movement recording."""
        if not self.recording:
            self.recording = True
            self.recorded_movements = []
            self.record_btn.config(text="Stop Recording", style="Accent.TButton")
            print("🔴 Recording started")
        else:
            self.recording = False
            self.record_btn.config(text="Start Recording", style="TButton")
            print(f"⏸️ Recording stopped - captured {len(self.recorded_movements)} movements")
    
    def save_preset(self):
        """Save current parameters and recorded movements as a preset."""
        name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
        if name:
            preset = {
                'spring_force': self.spring_force.get(),
                'damping': self.damping.get(),
                'max_velocity': self.max_velocity.get(),
                'cursor_sensitivity': self.cursor_sensitivity.get(),
                'wave_strength': self.wave_strength.get(),
                'gravity_width': self.gravity_width.get(),
                'default_position': self.default_position.get(),
                'reverse_vertical': self.reverse_vertical.get(),
                'physics_mode': self.physics_mode.get(),
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
        """Reset all fingers to default position."""
        default_pos = self.default_position.get()
        for i in range(self.num_fingers):
            self.finger_targets[i] = default_pos
            self.finger_positions[i] = default_pos
            self.finger_velocities[i] = 0.0
        print(f"🔄 Reset to default position: {default_pos:.1f}")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.connected and self.hand_controller:
            self.hand_controller.cleanup()
    
    def on_closing(self):
        """Handle window closing."""
        self.cleanup()
        self.root.destroy()
    
    def run(self):
        """Run the interface."""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.cleanup()


if __name__ == "__main__":
    print("🎮 Starting Physics-Based Hand Controller Interface...")
    print("Move your mouse over the control area to control finger positions!")
    print("Use the sliders to adjust physics parameters in real-time.")
    
    interface = PhysicsHandInterface()
    interface.run()
