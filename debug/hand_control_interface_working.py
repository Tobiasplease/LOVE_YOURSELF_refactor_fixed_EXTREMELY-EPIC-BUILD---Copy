#!/usr/bin/env python3
"""
Physics-Based Hand Servo Control Interface - WORKING VERSION

Interactive GUI for controlling hand servos through cursor physics simulation.
Wave-based gravitational field system with real-time parameter tuning.
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import math
from typing import Optional

# Import the hand controller
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servo_control.hand_expression import HandExpressionController
from config.config import HAND_SERIAL_PORT, BAUD_RATE


class PhysicsHandInterface:
    """Interactive physics-based hand control interface."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎮 Physics-Based Hand Controller - WORKING VERSION")
        self.root.geometry("800x700")
        
        # Hand controller
        self.hand_controller: Optional[HandExpressionController] = None
        self.connected = False
        
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
        
        # Control toggles - WORKING SETUP
        self.physics_mode = tk.BooleanVar(value=True)     # Start with physics enabled
        self.reverse_vertical = tk.BooleanVar(value=False) # Normal vertical by default
        self.override_mode = tk.BooleanVar(value=False)   # Manual override mode
        
        # Mouse tracking
        self.mouse_x = 0.5  # Normalized cursor position (0-1)
        self.mouse_y = 0.5
        
        # Animation state
        self.running = False
        self.last_time = time.time()
        self.last_send_time = 0  # Rate limiting for Arduino communication
        self.send_interval = 0.016  # 60 Hz send rate for smooth movement (was 0.05/20Hz)
        
        self.setup_ui()
        self.start_physics_loop()
    
    def setup_ui(self):
        """Create the user interface."""
        # === CONNECTION FRAME ===
        conn_frame = ttk.Frame(self.root)
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
        
        # === CONTROL CANVAS ===
        control_frame = ttk.LabelFrame(self.root, text="🎯 Hand Control Area")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(control_frame, bg="black", height=300)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        
        # === MODE TOGGLES ===
        mode_frame = ttk.LabelFrame(self.root, text="🎛️ Control Modes")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Physics mode toggle
        physics_cb = ttk.Checkbutton(mode_frame, text="🌊 Physics Mode (Wave Simulation)", 
                                    variable=self.physics_mode,
                                    command=self.on_physics_toggle)
        physics_cb.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Vertical reverse toggle
        reverse_cb = ttk.Checkbutton(mode_frame, text="🔄 Reverse Vertical Control", 
                                    variable=self.reverse_vertical)
        reverse_cb.pack(side=tk.LEFT, padx=10, pady=5)
        
        # === PHYSICS PARAMETERS ===
        params_frame = ttk.LabelFrame(self.root, text="⚙️ Physics Parameters")
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Configure grid weights
        for i in range(3):
            params_frame.columnconfigure(i, weight=1)
        
        # Spring Force
        ttk.Label(params_frame, text="Spring Force:").grid(row=0, column=0, sticky=tk.W, padx=5)
        spring_scale = ttk.Scale(params_frame, from_=10.0, to=300.0, 
                                variable=self.spring_force, orient=tk.HORIZONTAL)
        spring_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.spring_force).grid(row=0, column=2, padx=5)
        
        # Damping
        ttk.Label(params_frame, text="Damping:").grid(row=1, column=0, sticky=tk.W, padx=5)
        damping_scale = ttk.Scale(params_frame, from_=0.1, to=2.0, 
                                 variable=self.damping, orient=tk.HORIZONTAL)
        damping_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.damping).grid(row=1, column=2, padx=5)
        
        # Max Velocity
        ttk.Label(params_frame, text="Max Velocity:").grid(row=2, column=0, sticky=tk.W, padx=5)
        velocity_scale = ttk.Scale(params_frame, from_=50.0, to=500.0, 
                                  variable=self.max_velocity, orient=tk.HORIZONTAL)
        velocity_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.max_velocity).grid(row=2, column=2, padx=5)
        
        # Cursor Sensitivity
        ttk.Label(params_frame, text="Cursor Sensitivity:").grid(row=3, column=0, sticky=tk.W, padx=5)
        sensitivity_scale = ttk.Scale(params_frame, from_=0.5, to=5.0, 
                                     variable=self.cursor_sensitivity, orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=3, column=1, sticky=tk.EW, padx=5)
        ttk.Label(params_frame, textvariable=self.cursor_sensitivity).grid(row=3, column=2, padx=5)
        
        # === WAVE PARAMETERS ===
        wave_frame = ttk.LabelFrame(self.root, text="🌊 Wave Control Parameters")
        wave_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Configure grid weights
        for i in range(3):
            wave_frame.columnconfigure(i, weight=1)
        
        # Wave Strength
        ttk.Label(wave_frame, text="Wave Strength:").grid(row=0, column=0, sticky=tk.W, padx=5)
        wave_scale = ttk.Scale(wave_frame, from_=0.1, to=5.0, 
                              variable=self.wave_strength, orient=tk.HORIZONTAL)
        wave_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Label(wave_frame, textvariable=self.wave_strength).grid(row=0, column=2, padx=5)
        
        # Gravity Width
        ttk.Label(wave_frame, text="Gravity Width:").grid(row=1, column=0, sticky=tk.W, padx=5)
        gravity_scale = ttk.Scale(wave_frame, from_=0.1, to=1.0, 
                                 variable=self.gravity_width, orient=tk.HORIZONTAL)
        gravity_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Label(wave_frame, textvariable=self.gravity_width).grid(row=1, column=2, padx=5)
        
        # Default Position
        ttk.Label(wave_frame, text="Default Position:").grid(row=2, column=0, sticky=tk.W, padx=5)
        default_scale = ttk.Scale(wave_frame, from_=10.0, to=170.0, 
                                 variable=self.default_position, orient=tk.HORIZONTAL)
        default_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        ttk.Label(wave_frame, textvariable=self.default_position).grid(row=2, column=2, padx=5)
        
        # === CONTROLS ===
        control_buttons_frame = ttk.Frame(self.root)
        control_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(control_buttons_frame, text="🎯 Reset to Center", 
                  command=self.reset_to_center).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_buttons_frame, text="🛑 Emergency Stop", 
                  command=self.emergency_stop).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_buttons_frame, text="🧪 Test Connection", 
                  command=self.test_connection).pack(side=tk.LEFT, padx=5)
    
    def toggle_connection(self):
        """Connect or disconnect from hand controller."""
        if not self.connected:
            try:
                print("🔌 Attempting to connect to hand controller...")
                self.hand_controller = HandExpressionController(
                    port=HAND_SERIAL_PORT, 
                    baudrate=BAUD_RATE, 
                    clean_output=True
                )
                self.connected = True
                self.connect_btn.config(text="🔌 Disconnect")
                self.status_label.config(text="✅ Connected", foreground="green")
                print("✅ Connected to hand controller successfully!")
            except Exception as e:
                print(f"❌ Failed to connect: {e}")
                tk.messagebox.showerror("Connection Error", f"Failed to connect: {e}")
        else:
            try:
                if self.hand_controller:
                    self.hand_controller.cleanup()
                self.hand_controller = None
                self.connected = False
                self.connect_btn.config(text="🔌 Connect to Hand Controller")
                self.status_label.config(text="❌ Disconnected", foreground="red")
                print("🔌 Disconnected from hand controller")
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")
    
    def on_mouse_move(self, event):
        """Handle mouse movement in the canvas."""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Normalize mouse position (0-1)
        old_x, old_y = self.mouse_x, self.mouse_y
        self.mouse_x = max(0, min(1, event.x / canvas_width))
        self.mouse_y = max(0, min(1, event.y / canvas_height))
        
        # Apply vertical reversal if enabled
        if self.reverse_vertical.get():
            self.mouse_y = 1.0 - self.mouse_y
        
        # DEBUG: Print mouse movement to see if it's working
        if abs(self.mouse_x - old_x) > 0.01 or abs(self.mouse_y - old_y) > 0.01:
            print(f"🖱️ Mouse: ({self.mouse_x:.3f}, {self.mouse_y:.3f}) Canvas: {canvas_width}x{canvas_height}")
        
        # Update visual feedback
        self.update_canvas_visuals()
    
    def on_mouse_click(self, event):
        """Handle mouse click in the canvas."""
        print(f"🎯 Click at ({self.mouse_x:.2f}, {self.mouse_y:.2f})")
    
    def on_physics_toggle(self):
        """Handle physics mode toggle."""
        if self.physics_mode.get():
            print("🌊 Physics mode ENABLED - Wave simulation active")
        else:
            print("🎯 Physics mode DISABLED - Direct control active")
    
    def on_override_toggle(self):
        """Handle override mode toggle - CRITICAL FOR MANUAL CONTROL."""
        if self.override_mode.get():
            print("🎮 Manual override ENABLED - Disabling automatic hand control")
            self.override_label.config(text="Manual Mode: ON", foreground="red")
            # Enable manual override in hand controller
            if self.connected and self.hand_controller:
                try:
                    self.hand_controller.enable_manual_override()
                    print("✅ Manual override enabled in hand controller")
                except AttributeError:
                    print("⚠️ Hand controller doesn't support manual override method")
        else:
            print("🎮 Manual override DISABLED - Re-enabling automatic hand control")
            self.override_label.config(text="Manual Mode: OFF", foreground="blue")
            # Disable manual override in hand controller
            if self.connected and self.hand_controller:
                try:
                    self.hand_controller.disable_manual_override()
                    print("✅ Manual override disabled in hand controller")
                except AttributeError:
                    print("⚠️ Hand controller doesn't support manual override method")
    
    def update_canvas_visuals(self):
        """Update the visual feedback on the canvas."""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Draw cursor position
        cursor_x = self.mouse_x * canvas_width
        cursor_y = self.mouse_y * canvas_height
        
        # Draw large cursor crosshair
        self.canvas.create_line(cursor_x - 20, cursor_y, cursor_x + 20, cursor_y, 
                               fill="yellow", width=3)
        self.canvas.create_line(cursor_x, cursor_y - 20, cursor_x, cursor_y + 20, 
                               fill="yellow", width=3)
        
        # Draw gravity field visualization
        gravity_width = self.gravity_width.get()
        wave_strength = self.wave_strength.get()
        
        for i in range(self.num_fingers):
            finger_x = (i + 0.5) * (canvas_width / self.num_fingers)
            
            # Calculate distance from cursor to finger
            distance = abs(cursor_x - finger_x) / canvas_width
            
            # Calculate wave influence using cosine-squared falloff
            if distance < gravity_width:
                influence = math.cos(distance * math.pi / (2 * gravity_width)) ** 2
                influence *= wave_strength
            else:
                influence = 0
            
            # Draw finger position line - THICK LINES
            finger_pos = self.finger_positions[i]
            line_height = (finger_pos / 180.0) * canvas_height
            
            # Color based on influence
            if influence > 0.1:
                color = "red"
                width = 6
            else:
                color = "white"
                width = 4
            
            self.canvas.create_line(finger_x, canvas_height, finger_x, canvas_height - line_height,
                                   fill=color, width=width)
            
            # Draw finger label
            self.canvas.create_text(finger_x, canvas_height - 10, 
                                   text=f"F{i+1}", fill="cyan", font=("Arial", 10, "bold"))
    
    def calculate_wave_targets(self):
        """Calculate target positions based on wave physics."""
        canvas_width = 800  # Assume standard width
        cursor_x = self.mouse_x * canvas_width
        
        gravity_width = self.gravity_width.get()
        wave_strength = self.wave_strength.get()
        default_pos = self.default_position.get()
        sensitivity = self.cursor_sensitivity.get()
        
        for i in range(self.num_fingers):
            finger_x = (i + 0.5) * (canvas_width / self.num_fingers)
            
            # Calculate distance from cursor
            distance = abs(cursor_x - finger_x) / canvas_width
            
            # Calculate wave influence
            if distance < gravity_width:
                influence = math.cos(distance * math.pi / (2 * gravity_width)) ** 2
                influence *= wave_strength * sensitivity
                
                # Calculate target based on cursor Y position and influence
                wave_target = default_pos + (self.mouse_y - 0.5) * 160 * influence
                self.finger_targets[i] = max(10, min(170, wave_target))
            else:
                # Return to default position
                self.finger_targets[i] = default_pos
    
    def update_physics(self, dt):
        """Update physics simulation."""
        if self.physics_mode.get():
            # Physics mode: spring-damper system
            self.calculate_wave_targets()
            
            spring_k = self.spring_force.get()
            damping_c = self.damping.get()
            max_vel = self.max_velocity.get()
            
            for i in range(self.num_fingers):
                # Spring force towards target
                spring_force = spring_k * (self.finger_targets[i] - self.finger_positions[i])
                
                # Damping force
                damping_force = -damping_c * self.finger_velocities[i]
                
                # Total force
                total_force = spring_force + damping_force
                
                # Update velocity (assume unit mass)
                self.finger_velocities[i] += total_force * dt
                
                # Limit velocity
                self.finger_velocities[i] = max(-max_vel, min(max_vel, self.finger_velocities[i]))
                
                # Update position
                self.finger_positions[i] += self.finger_velocities[i] * dt
                
                # Clamp position
                self.finger_positions[i] = max(10, min(170, self.finger_positions[i]))
        else:
            # Direct mode: much more responsive, less physics
            self.calculate_wave_targets()
            for i in range(self.num_fingers):
                # Move directly towards target with fast interpolation
                diff = self.finger_targets[i] - self.finger_positions[i]
                # Use 90% interpolation for very fast response
                self.finger_positions[i] += diff * 0.9
                self.finger_velocities[i] = diff / dt if dt > 0 else 0
    
    def send_to_controller(self):
        """Send current positions to hand controller - OPTIMIZED FOR BUTTERY SMOOTHNESS!"""
        current_time = time.time()
        
        # BUTTERY SMOOTH: 50Hz rate (20ms intervals) - fast enough for smooth motion
        min_send_interval = 0.02  # 50 Hz = 20ms intervals (was 50ms)
        if current_time - self.last_send_time < min_send_interval:
            return
        
        if self.connected and self.hand_controller:
            try:
                positions = [int(pos) for pos in self.finger_positions]
                
                # SMOOTH: Only 1-degree threshold for ultra-responsive movement
                if not hasattr(self, 'last_sent_positions'):
                    self.last_sent_positions = [0, 0, 0, 0]  # Initialize
                
                # Check if any finger moved more than 1 degree (was 3)
                significant_change = False
                for i in range(len(positions)):
                    if abs(positions[i] - self.last_sent_positions[i]) > 1:
                        significant_change = True
                        break
                
                # Send on any small movement for smoothness
                if significant_change:
                    # Minimal debug output to avoid console spam
                    if int(current_time * 0.5) % 20 == 0:  # Print every 40 seconds
                        print(f"🎯 Smooth update: {positions}")
                    
                    # Use the correct method name
                    if hasattr(self.hand_controller, 'set_hand_positions'):
                        self.hand_controller.set_hand_positions(positions)
                    elif hasattr(self.hand_controller, 'set_positions'):
                        self.hand_controller.set_positions(positions)
                    else:
                        print("❌ Hand controller has no position setting method!")
                    
                    # Update tracking variables
                    self.last_sent_positions = positions.copy()
                    self.last_send_time = current_time
                
            except Exception as e:
                print(f"❌ Error sending to controller: {e}")
                import traceback
                traceback.print_exc()
    
    def physics_loop(self):
        """Main physics update loop."""
        while self.running:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Limit dt to prevent huge jumps
            dt = min(dt, 0.033)  # Max 30ms jumps
            
            # Update physics
            self.update_physics(dt)
            
            # Send to controller (with rate limiting)
            self.send_to_controller()
            
            # Update visuals
            try:
                self.root.after_idle(self.update_canvas_visuals)
            except:
                break
            
            # Sleep for smooth updates - 100 FPS physics, 60 Hz Arduino
            time.sleep(0.01)  # 100 FPS physics updates
    
    def start_physics_loop(self):
        """Start the physics update loop."""
        self.running = True
        self.physics_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.physics_thread.start()
    
    def reset_to_center(self):
        """Reset all fingers to center position."""
        for i in range(self.num_fingers):
            self.finger_positions[i] = 90.0
            self.finger_velocities[i] = 0.0
            self.finger_targets[i] = 90.0
        print("🎯 Reset all fingers to center position")
    
    def emergency_stop(self):
        """Emergency stop - reset everything."""
        self.reset_to_center()
        if self.connected and self.hand_controller:
            try:
                self.hand_controller.set_hand_positions([90, 90, 90, 90])
                print("🛑 Emergency stop - sent center positions to controller")
            except Exception as e:
                print(f"❌ Emergency stop failed: {e}")
    
    def test_connection(self):
        """Test the connection by sending a simple command."""
        if not self.connected or not self.hand_controller:
            print("❌ Not connected to hand controller")
            return
        
        print("🧪 Testing connection...")
        try:
            # Try to send a simple test position
            test_positions = [45, 90, 135, 90]  # Varied positions for testing
            
            if hasattr(self.hand_controller, 'set_hand_positions'):
                self.hand_controller.set_hand_positions(test_positions)
                print(f"✅ Test successful - sent positions: {test_positions}")
            else:
                print("❌ Hand controller missing set_hand_positions method")
                print(f"Available methods: {[method for method in dir(self.hand_controller) if not method.startswith('_')]}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run the interface."""
        print("🎮 Starting Physics-Based Hand Controller Interface...")
        print("Move your mouse over the control area to control finger positions!")
        print("Use the toggles and sliders to adjust behavior in real-time.")
        print("Toggle Physics Mode to switch between wave simulation and direct control.")
        
        try:
            self.root.mainloop()
        finally:
            self.running = False
            if self.connected:
                self.toggle_connection()


def main():
    """Main entry point."""
    interface = PhysicsHandInterface()
    interface.run()


if __name__ == "__main__":
    main()
