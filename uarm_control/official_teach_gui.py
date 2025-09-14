#!/usr/bin/env python3
"""
Official uArm Teach & Play GUI
==============================

Pure official uArm SDK implementation using the Teach class.
Records manual movements with accurate timing and smooth playback.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import time
import threading
from uarm.wrapper.swift_api import SwiftAPI
from uarm.swift.teach import Teach


class OfficialTeachGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("uArm Official Teach & Play")
        self.root.geometry("600x500")

        self.swift = None
        self.teach_systems = {}  # Store teach objects for each slot
        self.connected = False

        # Motion slots - using official uArm SDK Teach format
        self.motion_slots = {
            1: {"name": "pickup", "file": "movement_recordings/uarm/pickup.txt"},
            2: {"name": "place", "file": "movement_recordings/uarm/place.txt"},
            3: {"name": "gesture", "file": "movement_recordings/uarm/gesture.txt"}
        }

        # Ensure recording directory exists
        os.makedirs("movement_recordings/uarm", exist_ok=True)

        self.setup_ui()
        self.update_status()

    def setup_ui(self):
        """Setup the user interface"""

        # Connection frame
        conn_frame = ttk.Frame(self.root)
        conn_frame.pack(pady=10, padx=10, fill="x")

        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect)
        self.connect_btn.pack(side="left", padx=5)

        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", command=self.disconnect, state="disabled")
        self.disconnect_btn.pack(side="left", padx=5)

        self.emergency_stop_btn = ttk.Button(conn_frame, text="EMERGENCY STOP", command=self.emergency_stop)
        self.emergency_stop_btn.pack(side="right", padx=5)

        self.stop_all_btn = ttk.Button(conn_frame, text="Stop All Playback", command=self.stop_all_playback)
        self.stop_all_btn.pack(side="right", padx=5)

        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(pady=5, padx=10, fill="x")

        self.status_label = ttk.Label(status_frame, text="Not connected", foreground="red")
        self.status_label.pack(side="left")

        # Motion slots frame
        slots_frame = ttk.LabelFrame(self.root, text="Motion Slots", padding=10)
        slots_frame.pack(pady=10, padx=10, fill="both", expand=True)

        for slot_id, slot_info in self.motion_slots.items():
            self.create_slot_ui(slots_frame, slot_id, slot_info)

        # Recording settings
        settings_frame = ttk.LabelFrame(self.root, text="Recording Settings", padding=10)
        settings_frame.pack(pady=10, padx=10, fill="x")

        duration_frame = ttk.Frame(settings_frame)
        duration_frame.pack(fill="x")
        ttk.Label(duration_frame, text="Recording Duration (seconds):").pack(side="left")
        self.duration_var = tk.IntVar(value=10)
        duration_spin = ttk.Spinbox(duration_frame, from_=3, to=60, textvariable=self.duration_var, width=8)
        duration_spin.pack(side="left", padx=10)

        # Instructions
        instructions = ttk.LabelFrame(self.root, text="Instructions", padding=10)
        instructions.pack(pady=10, padx=10, fill="x")

        instruction_text = """
Official uArm Teach & Play with Duration Control:
1. Connect to uArm
2. Set recording duration (3-60 seconds)
3. Click "Start Record" - robot goes limp automatically
4. Move robot manually at your desired speed
5. Recording stops automatically after set duration
6. Click "Play Motion" to replay with original timing
        """

        ttk.Label(instructions, text=instruction_text.strip(), justify="left").pack()

    def create_slot_ui(self, parent, slot_id, slot_info):
        """Create UI for a motion slot"""

        slot_frame = ttk.LabelFrame(parent, text=f"Slot {slot_id}: {slot_info['name'].title()}", padding=5)
        slot_frame.pack(fill="x", pady=5)

        # Status
        status_text = "No recording" if not os.path.exists(slot_info['file']) else "Recording available"
        status_label = ttk.Label(slot_frame, text=status_text)
        status_label.pack(side="left", padx=5)

        # Buttons
        btn_frame = ttk.Frame(slot_frame)
        btn_frame.pack(side="right")

        record_btn = ttk.Button(btn_frame, text="Start Record",
                               command=lambda: self.start_recording(slot_id))
        record_btn.pack(side="left", padx=2)

        stop_btn = ttk.Button(btn_frame, text="Stop Record",
                             command=lambda: self.stop_recording(slot_id))
        stop_btn.pack(side="left", padx=2)

        play_btn = ttk.Button(btn_frame, text="Play Motion",
                             command=lambda: self.play_motion(slot_id))
        play_btn.pack(side="left", padx=2)

        delete_btn = ttk.Button(btn_frame, text="Delete",
                               command=lambda: self.delete_recording(slot_id))
        delete_btn.pack(side="left", padx=2)

        # Store references
        setattr(self, f"slot_{slot_id}_status", status_label)
        setattr(self, f"slot_{slot_id}_record", record_btn)
        setattr(self, f"slot_{slot_id}_stop", stop_btn)
        setattr(self, f"slot_{slot_id}_play", play_btn)
        setattr(self, f"slot_{slot_id}_delete", delete_btn)

    def connect(self):
        """Connect to uArm"""
        try:
            self.swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})
            self.swift.waiting_ready(timeout=3)
            self.connected = True

            # Initialize official uArm Teach objects for each slot
            for slot_id, slot_info in self.motion_slots.items():
                teach = Teach(slot_info['file'], self.swift)
                teach.start_standby_mode()  # Enable teach mode
                self.teach_systems[slot_id] = teach

            messagebox.showinfo("Success", "Connected to uArm!")

        except Exception as e:
            messagebox.showerror("Error", f"Connection failed: {e}")
            return

        self.update_status()

    def disconnect(self):
        """Disconnect from uArm"""
        try:
            # Stop any ongoing operations
            for teach in self.teach_systems.values():
                if teach.is_recording():
                    teach.stop_record()
                if teach.is_playing():
                    teach.stop_play()
                teach.stop_standby_mode()

            if self.swift:
                self.swift.disconnect()

            self.connected = False
            self.teach_systems.clear()
            messagebox.showinfo("Success", "Disconnected from uArm")

        except Exception as e:
            messagebox.showerror("Error", f"Disconnect failed: {e}")

        self.update_status()

    def emergency_stop(self):
        """Emergency stop all movement"""
        try:
            if self.swift and self.connected:
                # Stop all teach systems
                for teach in self.teach_systems.values():
                    if teach.is_recording():
                        teach.stop_record()
                    if teach.is_playing():
                        teach.stop_play()

                # Detach servos
                self.swift.set_servo_detach()
                messagebox.showinfo("Emergency Stop", "All movements stopped - robot is now limp")
            else:
                messagebox.showwarning("Warning", "Not connected to uArm")

        except Exception as e:
            messagebox.showerror("Error", f"Emergency stop failed: {e}")

    def stop_all_playback(self):
        """Stop all playback operations smoothly"""
        try:
            if self.swift and self.connected:
                stopped_count = 0
                for slot_id, teach in self.teach_systems.items():
                    if teach.is_playing():
                        teach.stop_play()
                        stopped_count += 1
                        # Update UI status
                        status_label = getattr(self, f"slot_{slot_id}_status")
                        status_label.config(text="Recording available")

                if stopped_count > 0:
                    messagebox.showinfo("Stopped", f"Stopped {stopped_count} playback operations")
                else:
                    messagebox.showinfo("Info", "No playback operations were running")
            else:
                messagebox.showwarning("Warning", "Not connected to uArm")

        except Exception as e:
            messagebox.showerror("Error", f"Stop playback failed: {e}")

    def start_recording(self, slot_id):
        """Start recording motion for slot with automatic duration stop"""
        if not self.connected or slot_id not in self.teach_systems:
            messagebox.showerror("Error", "Not connected to uArm")
            return

        try:
            teach = self.teach_systems[slot_id]
            if not teach.is_recording() and not teach.is_playing():
                duration = self.duration_var.get()

                # Start recording
                teach.start_record(interval=0.02)  # Higher frequency for smoother capture

                # Auto-stop recording after duration
                def auto_stop():
                    try:
                        if teach.is_recording():
                            teach.stop_record()
                            messagebox.showinfo("Recording Complete",
                                f"Recording completed for {self.motion_slots[slot_id]['name']} ({duration}s)")
                            self.update_slot_status(slot_id)
                    except Exception as e:
                        print(f"Auto-stop error: {e}")

                # Schedule auto-stop
                self.root.after(duration * 1000, auto_stop)

                messagebox.showinfo("Recording Started",
                    f"Recording {self.motion_slots[slot_id]['name']} for {duration} seconds\n\n"
                    f"Robot is now limp - move it at your desired speed!\n"
                    f"Will automatically stop in {duration} seconds")
            else:
                messagebox.showwarning("Warning", "Already recording or playing")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")

        self.update_slot_status(slot_id)

    def stop_recording(self, slot_id):
        """Stop recording motion for slot"""
        if not self.connected or slot_id not in self.teach_systems:
            messagebox.showerror("Error", "Not connected to uArm")
            return

        try:
            teach = self.teach_systems[slot_id]
            if teach.is_recording():
                teach.stop_record()
                messagebox.showinfo("Recording", f"Recording stopped and saved for {self.motion_slots[slot_id]['name']}")
            else:
                messagebox.showwarning("Warning", "Not currently recording")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {e}")

        self.update_slot_status(slot_id)

    def play_motion(self, slot_id):
        """Play recorded motion for slot using direct G-code execution"""
        if not self.connected or not self.swift:
            messagebox.showerror("Error", "Not connected to uArm")
            return

        slot_info = self.motion_slots[slot_id]
        if not os.path.exists(slot_info['file']):
            messagebox.showerror("Error", f"No recording found for {slot_info['name']}")
            return

        try:
            # Read the G-code file directly
            with open(slot_info['file'], 'r') as f:
                gcode_lines = f.readlines()

            # Filter out empty lines and parse G-code
            waypoints = []
            for line in gcode_lines:
                line = line.strip()
                if line and line.startswith('G0,'):
                    # Parse G0,x,y,z,calculated_speed,time_interval_ms format
                    parts = line.split(',')
                    if len(parts) >= 5:
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            calculated_speed = float(parts[4])  # Speed calculated by SDK
                            time_interval_ms = int(parts[5]) if len(parts) > 5 else 100
                            waypoints.append((x, y, z, calculated_speed, time_interval_ms))
                        except ValueError:
                            continue

            if not waypoints:
                messagebox.showerror("Error", "No valid waypoints found in recording")
                return

            print(f"DEBUG: Found {len(waypoints)} waypoints")

            # Start playback in separate thread - FAST AND SIMPLE
            def execute_playback():
                try:
                    import time as time_module

                    # Use SDK-calculated speed and timing for perfect organic replication
                    for i, (x, y, z, calculated_speed, interval_ms) in enumerate(waypoints):
                        # Update progress occasionally
                        if i % 20 == 0:
                            progress = (i / len(waypoints)) * 100
                            self.root.after(0, lambda p=progress: getattr(self, f"slot_{slot_id}_status").config(text=f"Playing {p:.1f}%"))

                        # Use the SDK-calculated speed directly (it's already correct!)
                        speed = max(50, min(2000, int(calculated_speed)))  # Direct use with safety bounds
                        self.swift.set_position(x=x, y=y, z=z, speed=speed, wait=False)

                        # Use the actual recorded timing intervals (this is the key!)
                        time_module.sleep(interval_ms / 1000.0)

                    # Playback complete
                    self.root.after(0, lambda: getattr(self, f"slot_{slot_id}_status").config(text="Recording available"))
                    self.root.after(0, lambda: messagebox.showinfo("Complete", f"Playback of {slot_info['name']} completed"))

                except Exception as e:
                    self.root.after(0, lambda: getattr(self, f"slot_{slot_id}_status").config(text="Recording available"))
                    self.root.after(0, lambda e=e: messagebox.showerror("Playback Error", f"Failed during playback: {e}"))

            threading.Thread(target=execute_playback, daemon=True).start()
            messagebox.showinfo("Playback", f"Playing {len(waypoints)} waypoints from {slot_info['name']}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start playback: {e}")

    def _monitor_playback(self, slot_id, motion_name):
        """Monitor playback progress and update UI"""
        teach = self.teach_systems[slot_id]

        while teach.is_playing():
            try:
                progress = teach.get_progress(wait=False)
                if progress and len(progress) >= 2:
                    percentage = progress[1]
                    status_label = getattr(self, f"slot_{slot_id}_status")
                    status_label.config(text=f"Playing {percentage:.1f}%")
                time.sleep(0.1)
            except Exception as e:
                print(f"Progress monitoring error: {e}")
                break

        # Update status when complete
        status_label = getattr(self, f"slot_{slot_id}_status")
        status_label.config(text="Recording available")
        print(f"Playback complete: {motion_name}")

    def delete_recording(self, slot_id):
        """Delete recording for slot"""
        slot_info = self.motion_slots[slot_id]

        if not os.path.exists(slot_info['file']):
            messagebox.showwarning("Warning", f"No recording to delete for {slot_info['name']}")
            return

        if messagebox.askyesno("Confirm Delete", f"Delete recording for {slot_info['name']}?"):
            try:
                os.remove(slot_info['file'])
                messagebox.showinfo("Success", f"Recording deleted for {slot_info['name']}")
                self.update_slot_status(slot_id)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete recording: {e}")

    def update_status(self):
        """Update connection status"""
        if self.connected:
            self.status_label.config(text="Connected to uArm", foreground="green")
            self.connect_btn.config(state="disabled")
            self.disconnect_btn.config(state="normal")
        else:
            self.status_label.config(text="Not connected", foreground="red")
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")

        # Update slot statuses
        for slot_id in self.motion_slots:
            self.update_slot_status(slot_id)

    def update_slot_status(self, slot_id):
        """Update status for a specific slot"""
        slot_info = self.motion_slots[slot_id]
        status_label = getattr(self, f"slot_{slot_id}_status")

        if os.path.exists(slot_info['file']):
            # Check file size to determine if recording is valid
            file_size = os.path.getsize(slot_info['file'])
            if file_size > 0:
                status_label.config(text="Recording available")
            else:
                status_label.config(text="Empty recording")
        else:
            status_label.config(text="No recording")

    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle window closing"""
        if self.connected:
            self.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    app = OfficialTeachGUI()
    app.run()