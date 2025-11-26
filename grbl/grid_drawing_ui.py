#!/usr/bin/env python3
"""
GRBL Grid Drawing UI with Warp Transform Toggle

Simple GUI for demonstrating warp transform effects on grid drawing.
Perfect for showing the mathematician coordinate transformations in real-time.
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import serial
from serial.tools import list_ports

# Import warp transform
try:
    from warp_transform import inverse, scale_x, scale_y, translate

    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False


class GridDrawingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GRBL Grid Drawing - Warp Transform Demo")
        self.root.geometry("600x700")

        # State variables
        self.ser = None
        self.is_connected = False
        self.is_drawing = False

        # Configuration variables
        self.use_warp_transform = tk.BooleanVar(value=False)
        self.use_corrected_warp = tk.BooleanVar(value=True)
        self.grid_size = tk.IntVar(value=40)
        self.grid_spacing = tk.IntVar(value=10)
        self.feed_rate = tk.IntVar(value=3000)
        self.pen_up_cmd = tk.StringVar(value="M3 S30")
        self.pen_down_cmd = tk.StringVar(value="M3 S50")

        self.create_widgets()
        self.update_status()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="GRBL Grid Drawing with Warp Transform", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Connection section
        conn_frame = ttk.LabelFrame(main_frame, text="Connection", padding="10")
        conn_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.connect_btn = ttk.Button(conn_frame, text="Connect to GRBL", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=0, padx=(0, 10))

        self.status_label = ttk.Label(conn_frame, text="Not connected", foreground="red")
        self.status_label.grid(row=0, column=1)

        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Warp transform toggle (prominent)
        warp_frame = ttk.Frame(config_frame)
        warp_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        warp_check = ttk.Checkbutton(warp_frame, text="Enable Original Warp Transform", variable=self.use_warp_transform, command=self.on_warp_toggle)
        warp_check.grid(row=0, column=0)

        corrected_warp_check = ttk.Checkbutton(
            warp_frame, text="Enable Corrected Warp Transform", variable=self.use_corrected_warp, command=self.on_warp_toggle
        )
        corrected_warp_check.grid(row=1, column=0)

        self.warp_status = ttk.Label(warp_frame, text="", font=("Arial", 10, "italic"))
        self.warp_status.grid(row=0, column=1, rowspan=2, padx=(20, 0))

        # Grid settings
        ttk.Label(config_frame, text="Grid Size (mm):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.grid_size, width=10).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(config_frame, text="Grid Spacing (mm):").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.grid_spacing, width=10).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(config_frame, text="Feed Rate:").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.feed_rate, width=10).grid(row=3, column=1, sticky=tk.W)

        # Pen commands
        ttk.Label(config_frame, text="Pen Up Command:").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.pen_up_cmd, width=15).grid(row=4, column=1, sticky=tk.W)

        ttk.Label(config_frame, text="Pen Down Command:").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(config_frame, textvariable=self.pen_down_cmd, width=15).grid(row=5, column=1, sticky=tk.W)

        # Drawing controls
        control_frame = ttk.LabelFrame(main_frame, text="Drawing Controls", padding="10")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.unlock_btn = ttk.Button(control_frame, text="Unlock ($X)", command=self.unlock_alarm)
        self.unlock_btn.grid(row=0, column=0, padx=(0, 10))

        self.home_btn = ttk.Button(control_frame, text="Home & Setup", command=self.home_machine)
        self.home_btn.grid(row=0, column=1, padx=(0, 10))

        self.draw_btn = ttk.Button(control_frame, text="Draw Grid", command=self.draw_grid)
        self.draw_btn.grid(row=0, column=2, padx=(0, 10))

        self.stop_btn = ttk.Button(control_frame, text="Emergency Stop", command=self.emergency_stop)
        self.stop_btn.grid(row=0, column=3)

        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.log_text = scrolledtext.ScrolledText(log_frame, width=70, height=15, font=("Courier", 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=(5, 0))

        # Configure grid weights for resizing
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def log(self, message, tag=None):
        """Add message to log with optional color coding"""
        self.log_text.insert(tk.END, message + "\n")
        if tag:
            # Configure tags for different message types
            if tag == "error":
                self.log_text.tag_config("error", foreground="red")
            elif tag == "success":
                self.log_text.tag_config("success", foreground="green")
            elif tag == "warp":
                self.log_text.tag_config("warp", foreground="blue")
            elif tag == "info":
                self.log_text.tag_config("info", foreground="gray")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """Clear the log output"""
        self.log_text.delete(1.0, tk.END)

    def on_warp_toggle(self):
        """Handle warp transform toggle with mutual exclusion"""
        original_enabled = self.use_warp_transform.get()
        corrected_enabled = self.use_corrected_warp.get()

        if (original_enabled or corrected_enabled) and not WARP_AVAILABLE:
            self.warp_status.config(text="❌ Warp transform not available!", foreground="red")
            messagebox.showwarning("Warning", "Warp transform module not available!")
            self.use_warp_transform.set(False)
            self.use_corrected_warp.set(False)
        else:
            # Ensure only one is enabled at a time
            if original_enabled and corrected_enabled:
                # If both were enabled, prefer corrected and disable original
                self.use_warp_transform.set(False)
                status = "✅ Corrected Enabled"
            elif corrected_enabled:
                status = "✅ Corrected Enabled"
            elif original_enabled:
                status = "✅ Original Enabled"
            else:
                status = "⭕ Disabled"

            color = "green" if (original_enabled or corrected_enabled) else "orange"
            self.warp_status.config(text=status, foreground=color)

    def update_status(self):
        """Update connection status and UI state"""
        if self.is_connected:
            self.status_label.config(text="✅ Connected", foreground="green")
            self.connect_btn.config(text="Disconnect")
        else:
            self.status_label.config(text="❌ Not connected", foreground="red")
            self.connect_btn.config(text="Connect to GRBL")

        # Update button states
        connected = self.is_connected
        self.unlock_btn.config(state="normal" if connected else "disabled")
        self.home_btn.config(state="normal" if connected else "disabled")
        self.draw_btn.config(state="normal" if (connected and not self.is_drawing) else "disabled")
        self.stop_btn.config(state="normal" if connected else "disabled")

        # Update warp status
        self.on_warp_toggle()

    def find_grbl_port(self):
        """Find GRBL port automatically"""
        ports = list(list_ports.comports())
        if not ports:
            raise RuntimeError("No serial ports found")

        for port in ports:
            try:
                self.log(f"Testing {port.device}...", "info")
                ser = serial.Serial(port.device, 115200, timeout=0.5)
                time.sleep(2.0)
                ser.reset_input_buffer()
                ser.write(b"?")
                ser.flush()
                line = ser.readline().decode(errors="ignore").strip()
                if line.startswith("<") or "Grbl" in line:
                    self.log(f"✅ Found GRBL at {port.device}: {line}", "success")
                    return ser
                ser.close()
            except Exception as e:
                self.log(f"❌ {port.device} failed: {e}", "error")

        raise RuntimeError("No GRBL port found")

    def toggle_connection(self):
        """Connect or disconnect from GRBL"""
        if self.is_connected:
            # Disconnect
            if self.ser:
                self.ser.close()
            self.ser = None
            self.is_connected = False
            self.log("Disconnected from GRBL", "info")
        else:
            # Connect
            try:
                self.ser = self.find_grbl_port()
                self.is_connected = True
                self.log("Successfully connected to GRBL!", "success")
            except Exception as e:
                messagebox.showerror("Connection Error", f"Failed to connect to GRBL:\n{e}")
                self.log(f"Connection failed: {e}", "error")

        self.update_status()

    def corrected_warp_transform_line(self, gcode_line):
        """Apply corrected warp transform that outputs proper GRBL coordinates"""
        import re

        # Andreas's A4 corner coordinates (proven safe)
        ANDREAS_A4_CORNERS = [
            (66, -2),  # bottom_left  → A4 (0, 0)
            (111, -1),  # bottom_right → A4 (210, 0)
            (-2, 67),  # top_left     → A4 (0, 297)
            (24, 67),  # top_right    → A4 (210, 297)
        ]

        def bilinear_interpolation(x, y, corner_values):
            """Convert from A4 coordinates to Andreas's GRBL coordinates"""
            # Normalize coordinates to [0,1] range for A4
            u = max(0, min(1, x / 210.0))  # A4 width
            v = max(0, min(1, y / 297.0))  # A4 height

            # Get corner values
            bottom_left, bottom_right, top_left, top_right = corner_values

            # Bilinear interpolation
            bottom = ((1 - u) * bottom_left[0] + u * bottom_right[0], (1 - u) * bottom_left[1] + u * bottom_right[1])
            top = ((1 - u) * top_left[0] + u * top_right[0], (1 - u) * top_left[1] + u * top_right[1])

            result_x = (1 - v) * bottom[0] + v * top[0]
            result_y = (1 - v) * bottom[1] + v * top[1]

            return result_x, result_y
        # Den här ska in direkt efter bilinear interpolation i ui-filen
        # Den mappar (x,y) from 40x40-rutan in i en skev rektangel som blir en kvadratish när man skriver ut den

        def map_to_quad(x, y, x_max=40, y_max=40):
            # Normalisera till [0, 1]
            u = x / x_max
            v = y / y_max

            # Den skeva rektangelns hörn i 40x40-världen, uppskattade från rutnätet så de kan behöver putsas
            Dx, Dy = 40, 0   # vänster längst från robot
            Ax, Ay = 8, 18   # vänster närmast robot
            Bx, By = 0, 40   # höger närmast robot
            Cx, Cy = 33, 20  # höger längst från robot

            # Bilinjär interpolation
            X = (1 - u) * (1 - v) * Ax + u * (1 - v) * Dx + (1 - u)* v * Bx + u * v * Cx
            Y = (1 - u) * (1 - v) * Ay + u * (1 - v) * Dy + (1 - u) * v * By + u * v * Cy
            return X, Y



        # Extract coordinates from G-code line
        x_match = re.search(r"X([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)
        y_match = re.search(r"Y([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)

        if x_match and y_match:
            # Get original coordinates
            original_x = float(x_match.group(1))
            original_y = float(y_match.group(1))

            try:
                # Den här raden ska ersätta alla andra manipuleringar
                grbl_x, grbl_y = map_to_quad(original_x, original_y, 40, 40)
                # Step 1: Apply inverse kinematics to get robot parameters
                # p = (original_x, original_y)
                # p = inverse(original_x, original_y)
                # p = inverse(theta, l)
                # p = scale_x(p, 80)
                # p = scale_y(p, 7)
                # p = translate(p[0], p[1], 0, -52)
                # Step 3: Convert from theoretical coordinates to GRBL coordinates
                # Clamp to A4 range for safety
                # safe_x = max(0, min(210, p[0]))
                # safe_y = max(0, min(297, p[1]))
                # grbl_x, grbl_y = bilinear_interpolation(safe_x, safe_y, ANDREAS_A4_CORNERS)
                # grbl_x, grbl_y = p
                # Update G-code line with corrected coordinates
                gcode_line = re.sub(r"X[-+]?\d*\.?\d+", f"X{grbl_x:.4f}", gcode_line, flags=re.IGNORECASE)
                gcode_line = re.sub(r"Y[-+]?\d*\.?\d+", f"Y{grbl_y:.4f}", gcode_line, flags=re.IGNORECASE)

            except Exception as e:
                # If warp transform fails, return original line
                self.log(f"⚠️  Warp transform failed: {e}", "error")
                return gcode_line

        return gcode_line

    def send_cmd(self, cmd, wait_ok=True, timeout=5.0):
        """Send command to GRBL with optional warp transform"""
        if not self.ser:
            raise RuntimeError("Not connected to GRBL")

        original_cmd = cmd

        # Apply warp transform if enabled
        if WARP_AVAILABLE and cmd.startswith(("G0", "G1", "G00", "G01")) and ("X" in cmd or "Y" in cmd):

            if self.use_corrected_warp.get():
                # Use corrected warp transform
                cmd = self.corrected_warp_transform_line(cmd)
                self.log(f"🔄 CORRECTED {original_cmd} -> {cmd}", "warp")
            elif self.use_warp_transform.get():
                # Use original warp transform
                from warp_transform import warp_transform_line

                cmd = warp_transform_line(cmd)
                self.log(f"🔄 ORIGINAL {original_cmd} -> {cmd}", "warp")
            else:
                # No warp transform
                self.log(f"📤 {cmd}")
        else:
            self.log(f"📤 {cmd}")

        self.ser.write((cmd + "\n").encode())
        self.ser.flush()

        if wait_ok:
            start = time.time()
            while time.time() - start < timeout:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    if line.lower() == "ok":
                        return
                    elif line.lower().startswith("error"):
                        raise RuntimeError(f"GRBL error: {line}")

    def unlock_alarm(self):
        """Send $X to unlock GRBL from alarm state"""
        try:
            self.log("🔓 Sending alarm unlock ($X)...", "info")
            self.send_cmd("$X", wait_ok=True, timeout=2.0)
            self.log("✅ Alarm unlocked!", "success")

            # Check status after unlock
            time.sleep(0.5)
            status = self.get_status()
            if status:
                self.log(f"Status after unlock: {status}", "info")

        except Exception as e:
            self.log(f"❌ Unlock failed: {e}", "error")
            messagebox.showerror("Unlock Error", f"Failed to unlock alarm:\n{e}")

    def home_machine(self):
        """Home the machine and set up coordinate system"""

        def home_thread():
            try:
                self.log("🏠 Starting homing sequence...", "info")

                # Check if in alarm state and unlock if needed
                status = self.get_status()
                if "Alarm" in status:
                    self.log("⚠️ Machine in alarm state, unlocking...", "info")
                    self.send_cmd("$X")
                    time.sleep(0.5)

                # Pen up before homing
                self.send_cmd(self.pen_up_cmd.get())
                time.sleep(0.5)

                # Home
                self.log("Homing machine ($H)...")
                self.send_cmd("$H", wait_ok=False)

                # Wait for homing to complete
                start = time.time()
                while time.time() - start < 120:
                    status_line = self.get_status()
                    if "Idle" in status_line:
                        break
                    elif "Alarm" in status_line:
                        raise RuntimeError(f"Homing failed: {status_line}")
                    time.sleep(0.2)

                # Set up coordinate system
                self.send_cmd("G54")  # Use G54 coordinate system
                self.send_cmd("G10 L20 P1 X0 Y0 Z0")  # Set current position as origin
                self.send_cmd(f"G0 X0 Y0 F{self.feed_rate.get()}")  # Go to origin

                self.log("✅ Homing and setup complete!", "success")

            except Exception as e:
                self.log(f"❌ Homing failed: {e}", "error")
                messagebox.showerror("Homing Error", f"Homing failed:\n{e}")

        threading.Thread(target=home_thread, daemon=True).start()

    def get_status(self):
        """Get GRBL status"""
        if not self.ser:
            return ""
        self.ser.write(b"?")
        self.ser.flush()
        return self.ser.readline().decode(errors="ignore").strip()

    def draw_grid(self):
        """Draw the grid with current settings"""

        def draw_thread():
            try:
                self.is_drawing = True
                self.update_status()

                grid_size = self.grid_size.get()
                spacing = self.grid_spacing.get()

                self.log(f"🎨 Starting grid drawing: {grid_size}x{grid_size}mm, spacing={spacing}mm")
                self.log(f"Warp transform: {'ENABLED' if self.use_warp_transform.get() else 'DISABLED'}")

                # Pen up to start
                self.send_cmd(self.pen_up_cmd.get())

                # Draw vertical lines
                self.log("Drawing vertical lines...")
                for x in range(0, grid_size + 1, spacing):
                    # Move to start of line
                    self.send_cmd(f"G0 X{x} Y0")
                    # Pen down and draw
                    self.send_cmd(self.pen_down_cmd.get())
                    self.send_cmd(f"G0 X{x} Y{grid_size}")
                    # Pen up
                    self.send_cmd(self.pen_up_cmd.get())

                # Draw horizontal lines
                self.log("Drawing horizontal lines...")
                for y in range(0, grid_size + 1, spacing):
                    # Move to start of line
                    self.send_cmd(f"G0 X0 Y{y}")
                    # Pen down and draw
                    self.send_cmd(self.pen_down_cmd.get())
                    self.send_cmd(f"G0 X{grid_size} Y{y}")
                    # Pen up
                    self.send_cmd(self.pen_up_cmd.get())

                # Return to origin
                self.send_cmd("G0 X0 Y0")

                self.log("✅ Grid drawing complete!", "success")

            except Exception as e:
                self.log(f"❌ Drawing failed: {e}", "error")
                messagebox.showerror("Drawing Error", f"Drawing failed:\n{e}")
            finally:
                self.is_drawing = False
                self.update_status()

        threading.Thread(target=draw_thread, daemon=True).start()

    def emergency_stop(self):
        """Emergency stop - send reset to GRBL"""
        if self.ser:
            try:
                self.ser.write(b"\x18")  # Ctrl-X (reset)
                self.ser.flush()
                self.log("🛑 EMERGENCY STOP sent!", "error")
                # Pen up after reset
                time.sleep(1)
                self.send_cmd(self.pen_up_cmd.get())
            except Exception as e:
                self.log(f"❌ Emergency stop failed: {e}", "error")


def main():
    root = tk.Tk()
    app = GridDrawingUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
