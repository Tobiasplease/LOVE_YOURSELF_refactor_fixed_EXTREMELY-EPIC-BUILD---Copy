#!/usr/bin/env python3

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uarm_control.uarm_controller import UarmController
from uarm_control.motion_manager import MotionManager

try:
    from config.config import (
        UARM_MOTION_STORAGE,
        UARM_MOVEMENT_NAMES,
        UARM_PORT
    )
except ImportError:
    UARM_MOTION_STORAGE = "movement_recordings/uarm"
    UARM_MOVEMENT_NAMES = {1: "pickup", 2: "place", 3: "gesture"}
    UARM_PORT = "/dev/arduino_uarm"


class UarmRecordingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("uArm Swift Pro - Teach & Play Recording")
        self.root.geometry("700x600")

        self.controller = None
        self.motion_manager = None
        self.recording_in_progress = False

        self.setup_ui()
        self.initialize_system()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="uArm Swift Pro - Teach & Play", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Connection Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        self.status_label = ttk.Label(status_frame, text="Initializing...", foreground="orange")
        self.status_label.grid(row=0, column=0, sticky=tk.W)

        self.connect_button = ttk.Button(status_frame, text="Reconnect", command=self.connect_uarm)
        self.connect_button.grid(row=0, column=1, padx=(10, 0))

        # Motion slots frame
        motions_frame = ttk.LabelFrame(main_frame, text="Motion Slots", padding="10")
        motions_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # Headers
        ttk.Label(motions_frame, text="Slot", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5)
        ttk.Label(motions_frame, text="Name", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5)
        ttk.Label(motions_frame, text="Status", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5)
        ttk.Label(motions_frame, text="Actions", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=5)

        # Motion slot rows
        self.slot_widgets = {}
        for slot in [1, 2, 3]:
            row = slot

            # Slot number
            ttk.Label(motions_frame, text=f"Slot {slot}").grid(row=row, column=0, padx=5, pady=2)

            # Name entry
            name_var = tk.StringVar(value=UARM_MOVEMENT_NAMES.get(slot, f"motion_{slot}"))
            name_entry = ttk.Entry(motions_frame, textvariable=name_var, width=12)
            name_entry.grid(row=row, column=1, padx=5, pady=2)

            # Status label
            status_label = ttk.Label(motions_frame, text="Not Recorded", foreground="red")
            status_label.grid(row=row, column=2, padx=5, pady=2)

            # Action buttons frame
            actions_frame = ttk.Frame(motions_frame)
            actions_frame.grid(row=row, column=3, padx=5, pady=2)

            record_btn = ttk.Button(actions_frame, text="Record",
                                  command=lambda s=slot: self.record_motion(s), width=8)
            record_btn.grid(row=0, column=0, padx=2)

            play_btn = ttk.Button(actions_frame, text="Play",
                                command=lambda s=slot: self.play_motion(s), width=8)
            play_btn.grid(row=0, column=1, padx=2)

            delete_btn = ttk.Button(actions_frame, text="Delete",
                                  command=lambda s=slot: self.delete_motion(s), width=8)
            delete_btn.grid(row=0, column=2, padx=2)

            self.slot_widgets[slot] = {
                'name_var': name_var,
                'status_label': status_label,
                'record_btn': record_btn,
                'play_btn': play_btn,
                'delete_btn': delete_btn
            }

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=3, pady=10)

        ttk.Button(control_frame, text="Go Home", command=self.home_uarm).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Set Home Here", command=self.set_home_here).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Suction ON", command=lambda: self.set_suction(True)).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Suction OFF", command=lambda: self.set_suction(False)).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Refresh Status", command=self.update_status).grid(row=0, column=4, padx=5)

        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, width=70)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def log(self, message):
        """Add message to log with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        def update_log():
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)

        if threading.current_thread() != threading.main_thread():
            self.root.after(0, update_log)
        else:
            update_log()

    def initialize_system(self):
        """Initialize uArm controller and motion manager"""
        def init():
            try:
                self.log("Initializing uArm system...")

                self.controller = UarmController(port=UARM_PORT, connect_on_init=False)
                self.motion_manager = MotionManager(
                    storage_path=UARM_MOTION_STORAGE,
                    controller=self.controller
                )

                self.log("System initialized. Attempting connection...")
                self.connect_uarm()

            except Exception as e:
                self.log(f"Initialization failed: {e}")
                self.root.after(0, lambda: self.status_label.config(
                    text="Initialization Failed", foreground="red"))

        threading.Thread(target=init, daemon=True).start()

    def connect_uarm(self):
        """Connect to uArm hardware"""
        def connect():
            try:
                self.log("Connecting to uArm...")

                if self.controller.connect():
                    self.log("✅ uArm connected successfully!")
                    self.root.after(0, lambda: self.status_label.config(
                        text="Connected", foreground="green"))
                    self.root.after(0, self.update_status)
                else:
                    error = self.controller.last_error or "Unknown error"
                    self.log(f"❌ Connection failed: {error}")
                    self.root.after(0, lambda: self.status_label.config(
                        text="Disconnected", foreground="red"))

            except Exception as e:
                self.log(f"Connection error: {e}")
                self.root.after(0, lambda: self.status_label.config(
                    text="Connection Error", foreground="red"))

        threading.Thread(target=connect, daemon=True).start()

    def update_status(self):
        """Update the status of all motion slots"""
        if not self.motion_manager:
            return

        try:
            for slot in [1, 2, 3]:
                is_recorded = self.motion_manager.is_motion_recorded(slot)
                status_text = "Recorded" if is_recorded else "Not Recorded"
                color = "green" if is_recorded else "red"

                self.slot_widgets[slot]['status_label'].config(
                    text=status_text, foreground=color)

                # Update name from saved data
                motion_info = self.motion_manager.get_motion_info(slot)
                if motion_info:
                    self.slot_widgets[slot]['name_var'].set(motion_info['name'])

        except Exception as e:
            self.log(f"Status update failed: {e}")

    def record_motion(self, slot):
        """Record a new motion for the specified slot"""
        if self.recording_in_progress:
            messagebox.showwarning("Recording in Progress", "Please wait for current recording to complete.")
            return

        if not self.controller or not self.controller.is_connected():
            messagebox.showerror("Not Connected", "uArm is not connected. Please connect first.")
            return

        def record():
            self.recording_in_progress = True
            try:
                name = self.slot_widgets[slot]['name_var'].get().strip()
                if not name:
                    name = f"motion_{slot}"

                self.log(f"Starting recording for Slot {slot}: {name}")

                # Show recording dialog
                self.root.after(0, lambda: self.show_recording_dialog(slot, name))

            except Exception as e:
                self.log(f"Recording setup failed: {e}")
                self.recording_in_progress = False

        threading.Thread(target=record, daemon=True).start()

    def show_recording_dialog(self, slot, name):
        """Show modal dialog for recording process"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Recording Motion {slot}: {name}")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))

        # Instructions
        instructions = """
RECORDING INSTRUCTIONS:

1. Click 'Start Recording' to release the motors
2. The uArm will become "limp" and can be moved manually
3. Move the uArm through your desired motion at the desired speed
4. Press the BUTTON on the uArm base to toggle suction ON/OFF
5. The system captures all positions, timing, and suction states
6. When finished, the motors re-enable and motion is saved

During playback, the exact motion and suction timing will be reproduced.
        """.strip()

        ttk.Label(dialog, text=instructions, justify=tk.LEFT).pack(pady=20, padx=20)

        # Status label
        status_label = ttk.Label(dialog, text="Ready to record...", font=("Arial", 10, "bold"))
        status_label.pack(pady=10)

        # Buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=20)

        start_btn = ttk.Button(buttons_frame, text="Start Recording",
                              command=lambda: self.start_recording_process(slot, name, dialog, status_label))
        start_btn.pack(side=tk.LEFT, padx=10)

        cancel_btn = ttk.Button(buttons_frame, text="Cancel", command=lambda: self.cancel_recording(dialog))
        cancel_btn.pack(side=tk.LEFT, padx=10)

    def start_recording_process(self, slot, name, dialog, status_label):
        """Start the actual recording process"""
        def record():
            try:
                # Update status
                self.root.after(0, lambda: status_label.config(text="Recording in progress..."))

                success = self.motion_manager.record_motion(slot, name)

                if success:
                    self.log(f"✅ Motion {slot} recorded successfully!")
                    self.root.after(0, lambda: messagebox.showinfo("Success", f"Motion '{name}' recorded successfully!"))
                    self.root.after(0, self.update_status)
                else:
                    self.log(f"❌ Recording failed for motion {slot}")
                    self.root.after(0, lambda: messagebox.showerror("Error", "Recording failed. Check the log for details."))

            except Exception as e:
                self.log(f"Recording error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Recording error: {e}"))

            finally:
                self.recording_in_progress = False
                self.root.after(0, lambda: dialog.destroy())

        threading.Thread(target=record, daemon=True).start()

    def cancel_recording(self, dialog):
        """Cancel recording process"""
        self.recording_in_progress = False
        dialog.destroy()
        self.log("Recording cancelled by user")

    def play_motion(self, slot):
        """Play back a recorded motion"""
        if not self.motion_manager.is_motion_recorded(slot):
            messagebox.showwarning("Not Recorded", f"Motion slot {slot} has no recording.")
            return

        if not self.controller or not self.controller.is_connected():
            messagebox.showerror("Not Connected", "uArm is not connected.")
            return

        def play():
            try:
                name = self.slot_widgets[slot]['name_var'].get()
                self.log(f"Playing motion {slot}: {name}")

                success = self.motion_manager.play_motion(slot)

                if success:
                    self.log(f"✅ Motion {slot} played successfully!")
                else:
                    self.log(f"❌ Motion playback failed for slot {slot}")

            except Exception as e:
                self.log(f"Playback error: {e}")

        threading.Thread(target=play, daemon=True).start()

    def delete_motion(self, slot):
        """Delete a recorded motion"""
        if not self.motion_manager.is_motion_recorded(slot):
            messagebox.showinfo("Nothing to Delete", f"Motion slot {slot} has no recording.")
            return

        name = self.slot_widgets[slot]['name_var'].get()
        if messagebox.askyesno("Confirm Delete", f"Delete recorded motion '{name}' from slot {slot}?"):
            try:
                success = self.motion_manager.delete_motion(slot)
                if success:
                    self.log(f"✅ Motion {slot} deleted successfully!")
                    self.update_status()
                else:
                    self.log(f"❌ Failed to delete motion {slot}")
            except Exception as e:
                self.log(f"Delete error: {e}")

    def home_uarm(self):
        """Move uArm to home position"""
        if not self.controller or not self.controller.is_connected():
            messagebox.showerror("Not Connected", "uArm is not connected.")
            return

        def home():
            try:
                home_pos = self.controller.get_home_position()
                self.log(f"Moving to home position: ({home_pos['x']}, {home_pos['y']}, {home_pos['z']})")
                success = self.controller.home()

                if success:
                    self.log("✅ Homing completed!")
                else:
                    self.log("❌ Homing failed!")

            except Exception as e:
                self.log(f"Homing error: {e}")

        threading.Thread(target=home, daemon=True).start()

    def set_home_here(self):
        """Set the current position as the home position"""
        if not self.controller or not self.controller.is_connected():
            messagebox.showerror("Not Connected", "uArm is not connected.")
            return

        confirm = messagebox.askyesno("Set Home Position",
                                    "Set the current uArm position as the new home position?\n\n"
                                    "This will be the position the arm returns to when 'Go Home' is pressed.")
        if not confirm:
            return

        def set_home():
            try:
                # Get current position first
                current_pos = self.controller.get_position()
                if current_pos and len(current_pos) >= 3:
                    self.log(f"Current position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")

                success = self.controller.save_current_as_home()

                if success:
                    new_home = self.controller.get_home_position()
                    self.log(f"✅ Home position set to: ({new_home['x']}, {new_home['y']}, {new_home['z']})")
                    messagebox.showinfo("Success",
                                      f"Home position updated!\n"
                                      f"New home: ({new_home['x']:.1f}, {new_home['y']:.1f}, {new_home['z']:.1f})")
                else:
                    self.log("❌ Failed to set home position!")
                    messagebox.showerror("Error", "Could not set home position.")

            except Exception as e:
                self.log(f"Set home error: {e}")
                messagebox.showerror("Error", f"Error setting home position: {e}")

        threading.Thread(target=set_home, daemon=True).start()

    def set_suction(self, on: bool):
        """Manually control suction cup"""
        if not self.controller or not self.controller.is_connected():
            messagebox.showerror("Not Connected", "uArm is not connected.")
            return

        def control_suction():
            try:
                success = self.controller.set_pump(on)
                action = "activated" if on else "deactivated"

                if success:
                    self.log(f"✅ Suction {action} manually")
                else:
                    self.log(f"❌ Failed to {action.lower()} suction")

            except Exception as e:
                self.log(f"Suction control error: {e}")

        threading.Thread(target=control_suction, daemon=True).start()

    def on_closing(self):
        """Handle window closing"""
        if self.controller:
            try:
                self.controller.disconnect()
            except:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = UarmRecordingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()