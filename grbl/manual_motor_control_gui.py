#!/usr/bin/env python3
"""
GRBL Manual Motor Control - GUI Version
Simple Tkinter interface for individual motor control with live position feedback
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grbl.grbl_utils import (
    ensure_homed,
    find_grbl_port,
    get_status,
    send_cmd,
    setup_basic_grbl,
    wait_until_idle,
)

# Global for clean shutdown
SER = None


def parse_position(status_line: str) -> Tuple[float, float, float]:
    """Parse X,Y,Z position from GRBL status line"""
    try:
        if "WPos:" in status_line:
            pos_part = status_line.split("WPos:")[1].split("|")[0]
        elif "MPos:" in status_line:
            pos_part = status_line.split("MPos:")[1].split("|")[0]
        else:
            return 0.0, 0.0, 0.0

        coords = pos_part.split(",")
        x = float(coords[0]) if len(coords) > 0 else 0.0
        y = float(coords[1]) if len(coords) > 1 else 0.0
        z = float(coords[2]) if len(coords) > 2 else 0.0
        return x, y, z
    except (IndexError, ValueError):
        return 0.0, 0.0, 0.0


def get_machine_state(status_line: str) -> str:
    """Extract machine state from status line"""
    try:
        if status_line.startswith("<"):
            return status_line.split(",")[0][1:]  # Remove < and get first part
        return "Unknown"
    except (ValueError, IndexError):
        return "Unknown"


class GRBLControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GRBL Manual Motor Control")
        self.root.geometry("500x400")

        self.selected_motor = tk.StringVar(value="X")
        self.step_size = tk.DoubleVar(value=1.0)
        self.x_pos = tk.DoubleVar(value=0.0)
        self.y_pos = tk.DoubleVar(value=0.0)
        self.machine_state = tk.StringVar(value="Unknown")
        self.bounds_status = tk.StringVar(value="OK")

        self.bounds = (0.0, 40.0, 0.0, 40.0)  # 40x40mm work area
        self.running = False

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Position display
        pos_frame = ttk.LabelFrame(main_frame, text="Current Position", padding="10")
        pos_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(pos_frame, text="X:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(pos_frame, textvariable=self.x_pos).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(pos_frame, text="mm").grid(row=0, column=2, sticky=tk.W)

        ttk.Label(pos_frame, text="Y:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(pos_frame, textvariable=self.y_pos).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(pos_frame, text="mm").grid(row=1, column=2, sticky=tk.W)

        ttk.Label(pos_frame, text="Status:").grid(row=0, column=3, sticky=tk.W, padx=(20, 0))
        ttk.Label(pos_frame, textvariable=self.machine_state).grid(row=0, column=4, sticky=tk.W)

        ttk.Label(pos_frame, text="Bounds:").grid(row=1, column=3, sticky=tk.W, padx=(20, 0))
        ttk.Label(pos_frame, textvariable=self.bounds_status).grid(row=1, column=4, sticky=tk.W)

        # Motor selection
        motor_frame = ttk.LabelFrame(main_frame, text="Motor Selection", padding="10")
        motor_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Radiobutton(motor_frame, text="X Motor", variable=self.selected_motor, value="X").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(motor_frame, text="Y Motor", variable=self.selected_motor, value="Y").grid(row=0, column=1, sticky=tk.W)

        # Step size
        step_frame = ttk.LabelFrame(main_frame, text="Step Size", padding="10")
        step_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Radiobutton(step_frame, text="0.1 mm", variable=self.step_size, value=0.1).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(step_frame, text="1.0 mm", variable=self.step_size, value=1.0).grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(step_frame, text="10.0 mm", variable=self.step_size, value=10.0).grid(row=0, column=2, sticky=tk.W)

        # Movement controls
        move_frame = ttk.LabelFrame(main_frame, text="Movement Controls", padding="10")
        move_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(move_frame, text="← Move -", command=self.move_negative).grid(row=0, column=0, padx=5)
        ttk.Button(move_frame, text="Move + →", command=self.move_positive).grid(row=0, column=1, padx=5)

        # Navigation controls
        nav_frame = ttk.LabelFrame(main_frame, text="Navigation", padding="10")
        nav_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(nav_frame, text="Set Origin (0,0)", command=self.set_origin).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Go Home", command=self.go_home).grid(row=0, column=1, padx=5)
        ttk.Button(nav_frame, text="Go Center", command=self.go_center).grid(row=0, column=2, padx=5)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=2, pady=10)

        self.connect_btn = ttk.Button(control_frame, text="Connect", command=self.connect_grbl)
        self.connect_btn.grid(row=0, column=0, padx=5)

        self.disconnect_btn = ttk.Button(control_frame, text="Disconnect", command=self.disconnect_grbl, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=1, padx=5)

        # Status bar
        self.status_text = tk.StringVar(value="Ready to connect")
        status_bar = ttk.Label(main_frame, textvariable=self.status_text, relief=tk.SUNKEN)
        status_bar.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

    def is_within_bounds(self, x: float, y: float) -> bool:
        """Check if position is within safe bounds"""
        x_min, x_max, y_min, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def update_position(self):
        """Update position display from GRBL"""
        if not SER or not self.running:
            return

        try:
            status = get_status(SER)
            x_pos, y_pos, _ = parse_position(status)
            machine_state = get_machine_state(status)

            self.x_pos.set(f"{x_pos:.2f}")
            self.y_pos.set(f"{y_pos:.2f}")
            self.machine_state.set(machine_state)

            in_bounds = self.is_within_bounds(x_pos, y_pos)
            self.bounds_status.set("OK" if in_bounds else "WARN")

        except Exception as e:
            self.status_text.set(f"Position update error: {e}")

        if self.running:
            self.root.after(500, self.update_position)  # Update every 500ms

    def connect_grbl(self):
        """Connect to GRBL controller"""
        global SER

        def connect_thread():
            global SER
            try:
                self.root.after(0, lambda: self.status_text.set("Connecting to GRBL..."))
                self.root.after(0, lambda: self.connect_btn.config(state=tk.DISABLED))

                try:
                    from config.config import GRBL_CNC_PORT

                    self.root.after(0, lambda: self.status_text.set(f"Connecting to {GRBL_CNC_PORT}..."))
                    SER = find_grbl_port(preferred_port=GRBL_CNC_PORT)
                except ImportError:
                    self.root.after(0, lambda: self.status_text.set("Scanning for GRBL ports..."))
                    SER = find_grbl_port()

                if not SER:
                    raise RuntimeError("No GRBL device found")

                self.root.after(0, lambda: self.status_text.set(f"Connected to {SER.port}, homing..."))
                ensure_homed(SER)
                setup_basic_grbl(SER, use_absolute_positioning=True)

                # Set relative positioning mode
                send_cmd(SER, "G91")

                self.running = True
                self.root.after(0, lambda: self.status_text.set(f"Connected and ready on {SER.port}"))
                self.root.after(0, lambda: self.disconnect_btn.config(state=tk.NORMAL))

                # Start position updates
                self.root.after(100, self.update_position)

            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                print(f"DEBUG: {error_msg}")  # Also print to console for debugging
                self.root.after(0, lambda: self.status_text.set(error_msg))
                self.root.after(0, lambda: self.connect_btn.config(state=tk.NORMAL))
                if SER:
                    try:
                        SER.close()
                        SER = None
                    except:
                        pass

        threading.Thread(target=connect_thread, daemon=True).start()

    def disconnect_grbl(self):
        """Disconnect from GRBL controller"""
        global SER

        self.running = False
        if SER:
            try:
                send_cmd(SER, "G90")  # Return to absolute positioning
                SER.close()
                SER = None
            except:
                pass

        self.status_text.set("Disconnected")
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)

    def move_negative(self):
        """Move selected motor in negative direction"""
        if not SER or not self.running:
            return

        motor = self.selected_motor.get()
        step = self.step_size.get()

        try:
            # Check bounds before moving
            current_x = float(self.x_pos.get())
            current_y = float(self.y_pos.get())

            if motor == "X":
                new_x = current_x - step
                if new_x >= self.bounds[0]:  # Check X min bound
                    send_cmd(SER, f"G0 X-{step}")
                else:
                    self.status_text.set("Movement would exceed X minimum bound")
                    return
            else:  # Y motor
                new_y = current_y - step
                if new_y >= self.bounds[2]:  # Check Y min bound
                    send_cmd(SER, f"G0 Y-{step}")
                else:
                    self.status_text.set("Movement would exceed Y minimum bound")
                    return

            self.status_text.set(f"Moved {motor} -{step}mm")
        except Exception as e:
            self.status_text.set(f"Movement error: {e}")

    def move_positive(self):
        """Move selected motor in positive direction"""
        if not SER or not self.running:
            return

        motor = self.selected_motor.get()
        step = self.step_size.get()

        try:
            # Check bounds before moving
            current_x = float(self.x_pos.get())
            current_y = float(self.y_pos.get())

            if motor == "X":
                new_x = current_x + step
                if new_x <= self.bounds[1]:  # Check X max bound
                    send_cmd(SER, f"G0 X{step}")
                else:
                    self.status_text.set("Movement would exceed X maximum bound")
                    return
            else:  # Y motor
                new_y = current_y + step
                if new_y <= self.bounds[3]:  # Check Y max bound
                    send_cmd(SER, f"G0 Y{step}")
                else:
                    self.status_text.set("Movement would exceed Y maximum bound")
                    return

            self.status_text.set(f"Moved {motor} +{step}mm")
        except Exception as e:
            self.status_text.set(f"Movement error: {e}")

    def set_origin(self):
        """Set current position as origin (0,0)"""
        if not SER or not self.running:
            return

        try:
            send_cmd(SER, "G90")  # Absolute mode
            send_cmd(SER, "G10 L20 P1 X0 Y0 Z0")  # Set work coordinate
            send_cmd(SER, "G91")  # Back to relative
            self.status_text.set("Origin set at current position")
        except Exception as e:
            self.status_text.set(f"Set origin error: {e}")

    def go_home(self):
        """Go to origin (0,0)"""
        if not SER or not self.running:
            return

        try:
            send_cmd(SER, "G90")  # Absolute mode
            send_cmd(SER, "G0 X0 Y0")
            send_cmd(SER, "G91")  # Back to relative
            wait_until_idle(SER, 10)
            self.status_text.set("Moved to home position")
        except Exception as e:
            self.status_text.set(f"Go home error: {e}")

    def go_center(self):
        """Go to center of work area"""
        if not SER or not self.running:
            return

        try:
            center_x = (self.bounds[0] + self.bounds[1]) / 2
            center_y = (self.bounds[2] + self.bounds[3]) / 2

            send_cmd(SER, "G90")  # Absolute mode
            send_cmd(SER, f"G0 X{center_x} Y{center_y}")
            send_cmd(SER, "G91")  # Back to relative
            wait_until_idle(SER, 10)
            self.status_text.set(f"Moved to center ({center_x}, {center_y})")
        except Exception as e:
            self.status_text.set(f"Go center error: {e}")


def main():
    root = tk.Tk()
    app = GRBLControlGUI(root)

    def on_closing():
        app.disconnect_grbl()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
