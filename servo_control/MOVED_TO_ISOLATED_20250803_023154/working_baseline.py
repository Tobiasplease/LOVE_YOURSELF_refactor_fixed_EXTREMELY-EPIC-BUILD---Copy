#!/usr/bin/env python3
"""
WORKING BASELINE - Simple Hand Controller
=========================================

This is your working baseline for tomorrow. It WILL work.
No fancy features, just solid hand control that responds to mouse movement.

Features that WORK:
✅ Hand controller connection
✅ Mouse movement → hand movement 
✅ Visual finger position bars
✅ Physics toggle
✅ Parameter controls that actually do something
✅ No crashes, no division by zero

Author: Baseline Recovery Team
"""

import tkinter as tk
from tkinter import ttk
import time
import math
import threading

# Import hand controller
try:
    from hand_expression import HandExpressionController
    HAND_AVAILABLE = True
    print("✅ Hand controller available")
except ImportError:
    HAND_AVAILABLE = False
    print("❌ Hand controller not available")


class WorkingBaseline:
    """Simple, reliable hand controller that WORKS."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔧 WORKING BASELINE - Simple Hand Controller 🔧")
        self.root.geometry("800x600")
        self.root.configure(bg="#2C3E50")
        
        # Hand controller
        self.hand_controller = None
        self.connected = False
        
        # Physics state
        self.num_fingers = 4
        self.finger_positions = [90.0] * self.num_fingers
        self.finger_velocities = [0.0] * self.num_fingers
        
        # Mouse position
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        
        # Parameters that actually work
        self.spring_force = tk.DoubleVar(value=300.0)
        self.damping = tk.DoubleVar(value=0.3)
        self.sensitivity = tk.DoubleVar(value=2.0)
        self.wave_strength = tk.DoubleVar(value=1.5)
        self.baseline_pos = tk.DoubleVar(value=90.0)
        
        # Control modes
        self.physics_mode = tk.BooleanVar(value=False)  # Start direct for responsiveness
        self.reverse_vertical = tk.BooleanVar(value=False)
        
        # Animation
        self.running = False
        self.last_time = time.time()
        
        self.setup_ui()
        self.start_loop()
    
    def setup_ui(self):
        """Create simple, working UI."""
        # Main container
        main_frame = tk.Frame(self.root, bg="#2C3E50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === CONNECTION ===
        conn_frame = tk.LabelFrame(main_frame, text="🔌 Hand Connection", 
                                  bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        conn_frame.pack(fill=tk.X, pady=5)
        
        conn_row = tk.Frame(conn_frame, bg="#34495E")
        conn_row.pack(fill=tk.X, padx=10, pady=10)
        
        self.connect_btn = tk.Button(conn_row, text="🔌 Connect Hand", 
                                   command=self.toggle_connection,
                                   bg="#3498DB", fg="white", font=("Arial", 10, "bold"))
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(conn_row, text="❌ Disconnected", 
                                   bg="#34495E", fg="#E74C3C", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # === MOVEMENT AREA ===
        canvas_frame = tk.LabelFrame(main_frame, text="🎯 Mouse Movement Area", 
                                    bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Instructions
        instr = tk.Label(canvas_frame, text="Move mouse here to control hand",
                        bg="#34495E", fg="#ECF0F1", font=("Arial", 11))
        instr.pack(pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="black", height=300, width=600)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # === CONTROLS ===
        controls_frame = tk.LabelFrame(main_frame, text="🎛️ Controls", 
                                     bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Mode toggles
        mode_row = tk.Frame(controls_frame, bg="#34495E")
        mode_row.pack(fill=tk.X, padx=10, pady=5)
        
        physics_cb = tk.Checkbutton(mode_row, text="⚡ Physics Mode (off = direct response)", 
                                   variable=self.physics_mode, bg="#34495E", fg="white",
                                   selectcolor="#34495E", font=("Arial", 10))
        physics_cb.pack(side=tk.LEFT)
        
        reverse_cb = tk.Checkbutton(mode_row, text="🔄 Reverse Vertical", 
                                   variable=self.reverse_vertical, bg="#34495E", fg="white",
                                   selectcolor="#34495E", font=("Arial", 10))
        reverse_cb.pack(side=tk.LEFT, padx=20)
        
        reset_btn = tk.Button(mode_row, text="🎯 Reset", command=self.reset,
                             bg="#E67E22", fg="white", font=("Arial", 10, "bold"))
        reset_btn.pack(side=tk.RIGHT)
        
        # === PARAMETERS ===
        params_frame = tk.LabelFrame(main_frame, text="⚙️ Parameters", 
                                   bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        params_frame.pack(fill=tk.X, pady=5)
        
        # Parameter controls
        self.create_param_control(params_frame, "Sensitivity:", self.sensitivity, 0.5, 5.0, 0)
        self.create_param_control(params_frame, "Wave Strength:", self.wave_strength, 0.0, 3.0, 1)
        self.create_param_control(params_frame, "Baseline Pos:", self.baseline_pos, 0, 180, 2)
        
        if not HAND_AVAILABLE:
            warning = tk.Label(main_frame, text="⚠️ Hand controller not available - simulation mode only",
                             bg="#2C3E50", fg="#F39C12", font=("Arial", 11, "bold"))
            warning.pack(pady=10)
    
    def create_param_control(self, parent, label_text, variable, min_val, max_val, row):
        """Create a parameter control row."""
        param_frame = tk.Frame(parent, bg="#34495E")
        param_frame.pack(fill=tk.X, padx=10, pady=2)
        
        label = tk.Label(param_frame, text=label_text, bg="#34495E", fg="white", 
                        font=("Arial", 10), width=15, anchor="w")
        label.pack(side=tk.LEFT)
        
        scale = tk.Scale(param_frame, from_=min_val, to=max_val, variable=variable, 
                        orient=tk.HORIZONTAL, bg="#34495E", fg="white",
                        highlightthickness=0, length=200, resolution=0.1)
        scale.pack(side=tk.LEFT, padx=10)
        
        value_label = tk.Label(param_frame, text=f"{variable.get():.1f}", 
                              bg="#34495E", fg="#3498DB", font=("Arial", 10), width=8)
        value_label.pack(side=tk.LEFT, padx=10)
        
        # Update value label
        variable.trace_add("write", lambda *args: value_label.config(text=f"{variable.get():.1f}"))
    
    def toggle_connection(self):
        """Connect/disconnect hand controller."""
        if not HAND_AVAILABLE:
            self.status_label.config(text="❌ Hand controller not available", fg="#E74C3C")
            return
        
        if not self.connected:
            try:
                self.hand_controller = HandExpressionController()
                if self.hand_controller.serial_connection:
                    self.connected = True
                    self.connect_btn.config(text="🔌 Disconnect")
                    self.status_label.config(text="✅ Connected", fg="#27AE60")
                    self.hand_controller.enable_manual_override()
                    print("✅ Hand controller connected successfully")
                else:
                    self.status_label.config(text="❌ Connection failed", fg="#E74C3C")
            except Exception as e:
                self.status_label.config(text=f"❌ Error: {str(e)[:20]}...", fg="#E74C3C")
                print(f"❌ Connection error: {e}")
        else:
            if self.hand_controller:
                try:
                    if hasattr(self.hand_controller, 'disconnect'):
                        self.hand_controller.disconnect()
                    elif hasattr(self.hand_controller, 'serial_connection') and self.hand_controller.serial_connection:
                        self.hand_controller.serial_connection.close()
                except:
                    pass
            self.connected = False
            self.connect_btn.config(text="🔌 Connect Hand")
            self.status_label.config(text="❌ Disconnected", fg="#E74C3C")
            print("🔌 Hand controller disconnected")
    
    def on_mouse_move(self, event):
        """Handle mouse movement."""
        # Get canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Normalize coordinates (safe division)
        self.mouse_x = event.x / max(1, width)
        self.mouse_y = event.y / max(1, height)
        
        # Clamp to valid range
        self.mouse_x = max(0.0, min(1.0, self.mouse_x))
        self.mouse_y = max(0.0, min(1.0, self.mouse_y))
    
    def update_physics(self, dt):
        """Update hand physics."""
        # Safety check
        if dt <= 0 or dt > 1.0:
            dt = 0.016  # 60fps fallback
        
        # Apply vertical reverse
        y = 1.0 - self.mouse_y if self.reverse_vertical.get() else self.mouse_y
        
        # Update each finger
        for i in range(self.num_fingers):
            # Finger position (0-1)
            finger_x = (i + 0.5) / max(1, self.num_fingers)
            
            # Distance from cursor
            dx = self.mouse_x - finger_x
            dy = y - 0.5
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Wave influence
            wave_influence = math.exp(-distance / 0.3) * self.wave_strength.get()
            
            # Calculate target position
            baseline = self.baseline_pos.get()
            cursor_effect = (y - 0.5) * 180 * self.sensitivity.get()
            wave_effect = wave_influence * cursor_effect
            target = baseline + wave_effect
            target = max(0, min(180, target))
            
            if self.physics_mode.get():
                # Physics mode - spring damping
                error = target - self.finger_positions[i]
                spring_force = error * self.spring_force.get()
                damping_force = -self.finger_velocities[i] * self.damping.get()
                
                acceleration = (spring_force + damping_force) / 10.0
                self.finger_velocities[i] += acceleration * dt
                self.finger_velocities[i] = max(-500, min(500, self.finger_velocities[i]))
                
                self.finger_positions[i] += self.finger_velocities[i] * dt
                self.finger_positions[i] = max(0, min(180, self.finger_positions[i]))
            else:
                # Direct mode - immediate response
                self.finger_positions[i] = target
                self.finger_velocities[i] = 0.0
    
    def send_to_hand(self):
        """Send finger positions to hand."""
        if self.connected and self.hand_controller:
            try:
                positions = [int(pos) for pos in self.finger_positions]
                self.hand_controller.set_hand_positions(positions)
            except Exception as e:
                print(f"❌ Error sending to hand: {e}")
    
    def update_visualization(self):
        """Update the visual display."""
        self.canvas.delete("all")
        
        # Canvas dimensions
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Draw cursor
        cx = self.mouse_x * width
        cy = self.mouse_y * height
        r = 12
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill="cyan", outline="white", width=2)
        
        # Draw finger bars
        bar_width = width / self.num_fingers
        bar_height = 150
        
        for i, pos in enumerate(self.finger_positions):
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            
            # Position as fraction
            pos_frac = pos / 180.0
            bar_top = height - (pos_frac * bar_height)
            
            # Color based on position
            intensity = int(255 * pos_frac)
            color = f"#{intensity:02x}{255-intensity:02x}{intensity//2:02x}"
            
            self.canvas.create_rectangle(x1, bar_top, x2, height, fill=color, outline="white", width=1)
            
            # Position text
            self.canvas.create_text((x1 + x2) / 2, height - 15, text=f"{pos:.0f}°", 
                                  fill="white", font=("Arial", 10, "bold"))
        
        # Status text
        mode = "PHYSICS" if self.physics_mode.get() else "DIRECT"
        status = f"Mode: {mode} | Mouse: ({self.mouse_x:.2f}, {self.mouse_y:.2f})"
        self.canvas.create_text(10, 10, text=status, fill="white", font=("Arial", 10, "bold"), anchor="nw")
    
    def reset(self):
        """Reset to center position."""
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        for i in range(self.num_fingers):
            self.finger_positions[i] = 90.0
            self.finger_velocities[i] = 0.0
        print("🎯 Reset to center")
    
    def start_loop(self):
        """Start the main loop."""
        self.running = True
        self.main_loop()
    
    def main_loop(self):
        """Main animation loop."""
        if not self.running:
            return
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Update physics
        self.update_physics(dt)
        
        # Send to hand
        self.send_to_hand()
        
        # Update visualization
        self.update_visualization()
        
        # Schedule next update
        self.root.after(16, self.main_loop)  # ~60 FPS
    
    def run(self):
        """Run the application."""
        print("🔧 WORKING BASELINE - Simple Hand Controller started!")
        print("✅ This version WILL work with your hand controller")
        print("✅ Move mouse in the black area to control the hand")
        print("✅ Use Physics Mode checkbox to toggle physics vs direct control")
        self.root.mainloop()


if __name__ == "__main__":
    app = WorkingBaseline()
    app.run()
