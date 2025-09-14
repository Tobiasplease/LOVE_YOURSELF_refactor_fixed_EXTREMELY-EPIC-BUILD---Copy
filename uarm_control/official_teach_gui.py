#!/usr/bin/env python3
"""
Official Teach & Play GUI
=========================

Uses the official uArm SDK teach functionality.
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

        # Motion slots
        self.motion_slots = {
            1: {"name": "pickup", "file": "movement_recordings/uarm/pickup_official.txt"},
            2: {"name": "place", "file": "movement_recordings/uarm/place_official.txt"},
            3: {"name": "gesture", "file": "movement_recordings/uarm/gesture_official.txt"}
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

        # Instructions
        instructions = ttk.LabelFrame(self.root, text="Instructions", padding=10)
        instructions.pack(pady=10, padx=10, fill="x")

        instruction_text = """
Official uArm Teach & Play:
1. Connect to uArm
2. Click "Start Record" for a motion slot
3. Robot will go limp - manually move it through the motion
4. Press the Menu button (Key0) on robot base to toggle suction
5. Click "Stop Record" when done
6. Click "Play Motion" to replay the recorded motion
7. Use "Delete" to remove a recording
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

            # Initialize teach systems for each slot
            for slot_id, slot_info in self.motion_slots.items():
                teach = Teach(slot_info['file'], self.swift)
                teach.start_standby_mode()
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

    def start_recording(self, slot_id):
        """Start recording motion for slot"""
        if not self.connected or slot_id not in self.teach_systems:
            messagebox.showerror("Error", "Not connected to uArm")
            return

        try:
            teach = self.teach_systems[slot_id]
            if not teach.is_recording() and not teach.is_playing():
                teach.start_record(interval=0.05)  # 20Hz recording
                messagebox.showinfo("Recording", f"Recording started for {self.motion_slots[slot_id]['name']}\n\nRobot is now limp - move it manually\nPress Menu button on base to toggle suction")
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
        """Play recorded motion for slot"""
        if not self.connected or slot_id not in self.teach_systems:
            messagebox.showerror("Error", "Not connected to uArm")
            return

        slot_info = self.motion_slots[slot_id]
        if not os.path.exists(slot_info['file']):
            messagebox.showerror("Error", f"No recording found for {slot_info['name']}")
            return

        try:
            teach = self.teach_systems[slot_id]
            if not teach.is_recording() and not teach.is_playing():
                teach.start_play(speed=1, times=1)
                messagebox.showinfo("Playback", f"Playing motion: {slot_info['name']}")
            else:
                messagebox.showwarning("Warning", "Already recording or playing")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to play motion: {e}")

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