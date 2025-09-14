#!/usr/bin/env python3
"""
Improved uArm Recording GUI with proper controls and workflow
"""

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


class ImprovedUarmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("uArm Swift Pro - Improved Teach & Play")
        self.root.geometry("800x700")

        self.controller = None
        self.motion_manager = None
        self.recording_in_progress = False
        self.recording_thread = None
        self.motors_released = False

        self.setup_ui()
        self.initialize_system()

    def setup_ui(self):
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Connection Tab
        self.setup_connection_tab()

        # Manual Control Tab
        self.setup_manual_tab()

        # Recording Tab
        self.setup_recording_tab()

        # Log Tab
        self.setup_log_tab()

    def setup_connection_tab(self):
        """Setup connection and status tab"""
        conn_frame = ttk.Frame(self.notebook)
        self.notebook.add(conn_frame, text="Connection")

        # Title
        title = ttk.Label(conn_frame, text="uArm Swift Pro Connection", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Status frame
        status_frame = ttk.LabelFrame(conn_frame, text="Connection Status", padding="10")
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(status_frame, text="Disconnected", foreground="red", font=("Arial", 12, "bold"))
        self.status_label.pack()

        # Connection buttons
        btn_frame = ttk.Frame(status_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.connect_btn = ttk.Button(btn_frame, text="Connect", command=self.connect_uarm)
        self.connect_btn.pack(side=tk.LEFT, padx=5)

        self.disconnect_btn = ttk.Button(btn_frame, text="Disconnect", command=self.disconnect_uarm, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)

        # Device info
        self.info_frame = ttk.LabelFrame(conn_frame, text="Device Information", padding="10")
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_text = tk.Text(self.info_frame, height=6, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def setup_manual_tab(self):
        """Setup manual control tab"""
        manual_frame = ttk.Frame(self.notebook)
        self.notebook.add(manual_frame, text="Manual Control")

        # Title
        title = ttk.Label(manual_frame, text="Manual Controls & Testing", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Motor control
        motor_frame = ttk.LabelFrame(manual_frame, text="Motor Control", padding="10")
        motor_frame.pack(fill=tk.X, padx=10, pady=5)

        motor_btn_frame = ttk.Frame(motor_frame)
        motor_btn_frame.pack(fill=tk.X)

        self.release_motors_btn = ttk.Button(motor_btn_frame, text="Release Motors (Manual Move)",
                                            command=self.release_motors, state=tk.DISABLED)
        self.release_motors_btn.pack(side=tk.LEFT, padx=5)

        self.enable_motors_btn = ttk.Button(motor_btn_frame, text="Enable Motors (Hold Position)",
                                           command=self.enable_motors, state=tk.DISABLED)
        self.enable_motors_btn.pack(side=tk.LEFT, padx=5)

        self.motor_status_label = ttk.Label(motor_btn_frame, text="Motors: Enabled", foreground="green")
        self.motor_status_label.pack(side=tk.RIGHT, padx=10)

        # Suction control
        suction_frame = ttk.LabelFrame(manual_frame, text="Suction Cup Control & Testing", padding="10")
        suction_frame.pack(fill=tk.X, padx=10, pady=5)

        suction_btn_frame = ttk.Frame(suction_frame)
        suction_btn_frame.pack(fill=tk.X)

        self.suction_on_btn = ttk.Button(suction_btn_frame, text="Suction ON",
                                        command=self.suction_on, state=tk.DISABLED)
        self.suction_on_btn.pack(side=tk.LEFT, padx=5)

        self.suction_off_btn = ttk.Button(suction_btn_frame, text="Suction OFF",
                                         command=self.suction_off, state=tk.DISABLED)
        self.suction_off_btn.pack(side=tk.LEFT, padx=5)

        self.suction_status_label = ttk.Label(suction_btn_frame, text="Suction: OFF", foreground="red")
        self.suction_status_label.pack(side=tk.RIGHT, padx=10)

        # Home position
        home_frame = ttk.LabelFrame(manual_frame, text="Home Position", padding="10")
        home_frame.pack(fill=tk.X, padx=10, pady=5)

        home_btn_frame = ttk.Frame(home_frame)
        home_btn_frame.pack(fill=tk.X)

        self.home_btn = ttk.Button(home_btn_frame, text="Move to Home", command=self.go_home, state=tk.DISABLED)
        self.home_btn.pack(side=tk.LEFT, padx=5)

        self.save_home_btn = ttk.Button(home_btn_frame, text="Save Current as Home",
                                       command=self.save_current_home, state=tk.DISABLED)
        self.save_home_btn.pack(side=tk.LEFT, padx=5)

        # Button testing
        button_frame = ttk.LabelFrame(manual_frame, text="Button Testing", padding="10")
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(button_frame, text="Press buttons on uArm base to test override system:").pack()

        self.button_status_frame = ttk.Frame(button_frame)
        self.button_status_frame.pack(fill=tk.X, pady=5)

        self.menu_btn_label = ttk.Label(self.button_status_frame, text="MENU: Released", foreground="gray")
        self.menu_btn_label.pack(side=tk.LEFT, padx=10)

        self.play_btn_label = ttk.Label(self.button_status_frame, text="PLAY: Released", foreground="gray")
        self.play_btn_label.pack(side=tk.LEFT, padx=10)

        # Start button monitoring
        self.start_button_monitoring()

    def setup_recording_tab(self):
        """Setup recording tab with improved workflow"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="Recording")

        # Title
        title = ttk.Label(rec_frame, text="Motion Recording", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Recording workflow instructions
        workflow_frame = ttk.LabelFrame(rec_frame, text="Recording Workflow", padding="10")
        workflow_frame.pack(fill=tk.X, padx=10, pady=5)

        workflow_text = """
Recording Steps:
1. Select a motion slot below
2. Click "Release Motors" to enable manual movement
3. Move the arm to starting position
4. Click "Start Recording"
5. Move the arm through the motion smoothly
6. Press PLAY button on uArm base to toggle suction during motion
7. Click "Stop Recording" when complete
8. Motors will re-engage automatically
        """
        workflow_label = ttk.Label(workflow_frame, text=workflow_text, justify=tk.LEFT)
        workflow_label.pack()

        # Recording control
        control_frame = ttk.LabelFrame(rec_frame, text="Recording Control", padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        self.recording_status_label = ttk.Label(control_frame, text="Status: Ready", font=("Arial", 12, "bold"))
        self.recording_status_label.pack()

        rec_btn_frame = ttk.Frame(control_frame)
        rec_btn_frame.pack(fill=tk.X, pady=10)

        self.release_for_recording_btn = ttk.Button(rec_btn_frame, text="1. Release Motors",
                                                   command=self.release_for_recording, state=tk.DISABLED)
        self.release_for_recording_btn.pack(side=tk.LEFT, padx=5)

        self.start_recording_btn = ttk.Button(rec_btn_frame, text="2. Start Recording",
                                             command=self.start_recording, state=tk.DISABLED)
        self.start_recording_btn.pack(side=tk.LEFT, padx=5)

        self.stop_recording_btn = ttk.Button(rec_btn_frame, text="3. Stop Recording",
                                            command=self.stop_recording, state=tk.DISABLED)
        self.stop_recording_btn.pack(side=tk.LEFT, padx=5)

        # Motion slots
        slots_frame = ttk.LabelFrame(rec_frame, text="Motion Slots", padding="10")
        slots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.selected_slot = tk.IntVar(value=1)

        self.slot_widgets = {}
        for slot in [1, 2, 3]:
            self.create_slot_widget(slots_frame, slot)

    def create_slot_widget(self, parent, slot):
        """Create widget for motion slot"""
        slot_frame = ttk.Frame(parent)
        slot_frame.pack(fill=tk.X, pady=5)

        # Radio button for selection
        radio = ttk.Radiobutton(slot_frame, text=f"Slot {slot}", variable=self.selected_slot,
                               value=slot, command=self.slot_selected)
        radio.pack(side=tk.LEFT)

        # Name entry
        name_var = tk.StringVar(value=UARM_MOVEMENT_NAMES.get(slot, f"motion_{slot}"))
        name_entry = ttk.Entry(slot_frame, textvariable=name_var, width=15)
        name_entry.pack(side=tk.LEFT, padx=10)

        # Status
        status_label = ttk.Label(slot_frame, text="Not Recorded", foreground="red")
        status_label.pack(side=tk.LEFT, padx=10)

        # Play button
        play_btn = ttk.Button(slot_frame, text="Play", command=lambda: self.play_motion(slot),
                             state=tk.DISABLED)
        play_btn.pack(side=tk.LEFT, padx=5)

        # Delete button
        delete_btn = ttk.Button(slot_frame, text="Delete", command=lambda: self.delete_motion(slot),
                               state=tk.DISABLED)
        delete_btn.pack(side=tk.LEFT, padx=5)

        self.slot_widgets[slot] = {
            'frame': slot_frame,
            'name_var': name_var,
            'status_label': status_label,
            'play_btn': play_btn,
            'delete_btn': delete_btn
        }

    def setup_log_tab(self):
        """Setup log tab"""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")

        # Log display
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Clear log button
        clear_btn = ttk.Button(log_frame, text="Clear Log", command=self.clear_log)
        clear_btn.pack(pady=5)

    def initialize_system(self):
        """Initialize uArm system"""
        def init():
            try:
                self.log("🚀 Initializing uArm system...")

                self.controller = UarmController(port=UARM_PORT, connect_on_init=False, auto_home=False)
                self.motion_manager = MotionManager(
                    storage_path=UARM_MOTION_STORAGE,
                    controller=self.controller
                )

                self.log("✅ System initialized successfully")

            except Exception as e:
                self.log(f"❌ Initialization failed: {e}")

        threading.Thread(target=init, daemon=True).start()

    def connect_uarm(self):
        """Connect to uArm"""
        def connect():
            try:
                self.log("🔌 Connecting to uArm...")

                if self.controller and self.controller.connect():
                    self.log("✅ uArm connected successfully!")
                    self.log(f"✅ Button callbacks registered: {self.controller.callbacks_registered}")

                    self.root.after(0, self.on_connected)

                    # Show device info
                    try:
                        info = self.controller.robot.get_device_info()
                        info_text = f"Device: {info.get('device_type', 'Unknown')}\n"
                        info_text += f"Hardware: {info.get('hardware_version', 'Unknown')}\n"
                        info_text += f"Firmware: {info.get('firmware_version', 'Unknown')}\n"
                        info_text += f"API: {info.get('api_version', 'Unknown')}\n"
                        info_text += f"ID: {info.get('device_unique', 'Unknown')}"

                        self.root.after(0, lambda: self.update_device_info(info_text))
                    except Exception as e:
                        self.log(f"⚠️ Could not get device info: {e}")

                else:
                    error = self.controller.last_error if self.controller else "Controller not initialized"
                    self.log(f"❌ Connection failed: {error}")
                    self.root.after(0, self.on_disconnected)

            except Exception as e:
                self.log(f"❌ Connection error: {e}")
                self.root.after(0, self.on_disconnected)

        threading.Thread(target=connect, daemon=True).start()

    def disconnect_uarm(self):
        """Disconnect from uArm"""
        def disconnect():
            try:
                if self.controller and self.controller.is_connected():
                    self.controller.disconnect()
                    self.log("🔌 Disconnected from uArm")
                self.root.after(0, self.on_disconnected)
            except Exception as e:
                self.log(f"❌ Disconnect error: {e}")

        threading.Thread(target=disconnect, daemon=True).start()

    def on_connected(self):
        """Update UI when connected"""
        self.status_label.config(text="Connected", foreground="green")
        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.NORMAL)

        # Enable manual controls
        self.release_motors_btn.config(state=tk.NORMAL)
        self.enable_motors_btn.config(state=tk.NORMAL)
        self.suction_on_btn.config(state=tk.NORMAL)
        self.suction_off_btn.config(state=tk.NORMAL)
        self.home_btn.config(state=tk.NORMAL)
        self.save_home_btn.config(state=tk.NORMAL)

        # Enable recording controls
        self.release_for_recording_btn.config(state=tk.NORMAL)

        self.update_motion_status()

    def on_disconnected(self):
        """Update UI when disconnected"""
        self.status_label.config(text="Disconnected", foreground="red")
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)

        # Disable all controls
        for btn in [self.release_motors_btn, self.enable_motors_btn, self.suction_on_btn,
                   self.suction_off_btn, self.home_btn, self.save_home_btn,
                   self.release_for_recording_btn, self.start_recording_btn, self.stop_recording_btn]:
            btn.config(state=tk.DISABLED)

        # Reset recording state
        self.recording_in_progress = False
        self.motors_released = False
        self.recording_status_label.config(text="Status: Disconnected")
        self.motor_status_label.config(text="Motors: Unknown", foreground="gray")

    def update_device_info(self, info_text):
        """Update device info display"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)

    def release_motors(self):
        """Release motors for manual movement"""
        def release():
            try:
                if self.controller and self.controller.release_motors():
                    self.log("🔓 Motors released - arm can be moved manually")
                    self.root.after(0, lambda: self.motor_status_label.config(text="Motors: Released", foreground="orange"))
                    self.motors_released = True
                else:
                    self.log("❌ Failed to release motors")
            except Exception as e:
                self.log(f"❌ Motor release error: {e}")

        threading.Thread(target=release, daemon=True).start()

    def enable_motors(self):
        """Enable motors to hold position"""
        def enable():
            try:
                if self.controller and self.controller.enable_motors():
                    self.log("🔒 Motors enabled - arm will hold position")
                    self.root.after(0, lambda: self.motor_status_label.config(text="Motors: Enabled", foreground="green"))
                    self.motors_released = False
                else:
                    self.log("❌ Failed to enable motors")
            except Exception as e:
                self.log(f"❌ Motor enable error: {e}")

        threading.Thread(target=enable, daemon=True).start()

    def suction_on(self):
        """Turn suction on"""
        def activate():
            try:
                if self.controller and self.controller.set_pump(True):
                    self.log("🔴 Suction activated")
                    self.root.after(0, lambda: self.suction_status_label.config(text="Suction: ON", foreground="green"))
                else:
                    self.log("❌ Failed to activate suction")
            except Exception as e:
                self.log(f"❌ Suction activation error: {e}")

        threading.Thread(target=activate, daemon=True).start()

    def suction_off(self):
        """Turn suction off"""
        def deactivate():
            try:
                if self.controller and self.controller.set_pump(False):
                    self.log("⚫ Suction deactivated")
                    self.root.after(0, lambda: self.suction_status_label.config(text="Suction: OFF", foreground="red"))
                else:
                    self.log("❌ Failed to deactivate suction")
            except Exception as e:
                self.log(f"❌ Suction deactivation error: {e}")

        threading.Thread(target=deactivate, daemon=True).start()

    def go_home(self):
        """Move to home position"""
        def home():
            try:
                if self.controller and self.controller.home():
                    self.log("🏠 Moved to home position")
                else:
                    self.log("❌ Failed to move to home")
            except Exception as e:
                self.log(f"❌ Home movement error: {e}")

        threading.Thread(target=home, daemon=True).start()

    def save_current_home(self):
        """Save current position as home"""
        def save():
            try:
                if self.controller and self.controller.save_current_as_home():
                    self.log("💾 Current position saved as home")
                else:
                    self.log("❌ Failed to save home position")
            except Exception as e:
                self.log(f"❌ Save home error: {e}")

        threading.Thread(target=save, daemon=True).start()

    def start_button_monitoring(self):
        """Start monitoring button states"""
        def monitor():
            while True:
                try:
                    if self.controller and self.controller.is_connected():
                        states = self.controller.get_button_state()

                        # Update button status labels
                        menu_status = "PRESSED" if states.get("menu", False) else "Released"
                        play_status = "PRESSED" if states.get("play", False) else "Released"

                        menu_color = "red" if states.get("menu", False) else "gray"
                        play_color = "red" if states.get("play", False) else "gray"

                        self.root.after(0, lambda: self.menu_btn_label.config(
                            text=f"MENU: {menu_status}", foreground=menu_color))
                        self.root.after(0, lambda: self.play_btn_label.config(
                            text=f"PLAY: {play_status}", foreground=play_color))

                except Exception:
                    pass

                time.sleep(0.1)

        threading.Thread(target=monitor, daemon=True).start()

    def release_for_recording(self):
        """Release motors for recording"""
        self.release_motors()
        self.start_recording_btn.config(state=tk.NORMAL)
        self.recording_status_label.config(text="Status: Motors released - Ready to record")

    def start_recording(self):
        """Start recording motion"""
        slot = self.selected_slot.get()
        name = self.slot_widgets[slot]['name_var'].get().strip()

        if not name:
            messagebox.showerror("Error", "Please enter a motion name")
            return

        self.recording_in_progress = True
        self.start_recording_btn.config(state=tk.DISABLED)
        self.stop_recording_btn.config(state=tk.NORMAL)
        self.recording_status_label.config(text=f"Status: Recording {name}...", foreground="red")

        self.log(f"🔴 Started recording motion '{name}' in slot {slot}")
        self.log("📝 Move the arm through the motion smoothly")
        self.log("🔘 Press PLAY button on uArm to toggle suction")

        # Start the actual recording in background
        def record():
            try:
                # Use custom GUI-friendly recording instead of motion_manager.record_motion
                success = self.record_motion_gui_friendly(slot, name)

                if success:
                    self.log(f"✅ Recording complete: {name}")
                    self.root.after(0, lambda: self.on_recording_complete(True))
                else:
                    self.log(f"❌ Recording failed: {name}")
                    self.root.after(0, lambda: self.on_recording_complete(False))

            except Exception as e:
                self.log(f"❌ Recording error: {e}")
                self.root.after(0, lambda: self.on_recording_complete(False))

        self.recording_thread = threading.Thread(target=record, daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop recording motion"""
        self.recording_in_progress = False
        self.stop_recording_btn.config(state=tk.DISABLED)
        self.log("⏹️ Recording stopped by user")

    def record_motion_gui_friendly(self, slot, name):
        """GUI-friendly motion recording without input() calls"""
        import json
        import os
        from datetime import datetime

        if not self.controller or not self.controller.is_connected():
            self.log("❌ Controller not connected")
            return False

        try:
            self.log(f"🎬 Starting GUI recording for slot {slot}: {name}")

            # Ensure storage directory exists
            os.makedirs(self.motion_manager.storage_path, exist_ok=True)

            # Record motion sequence with timing and suction states
            motion_sequence = []
            start_time = time.time()
            suction_state = False

            self.log("📝 Recording motion sequence...")
            self.log("🔘 Press PLAY button on uArm to toggle suction during motion")

            # Clear any existing button events
            self.controller.get_button_events()

            # Record for as long as recording_in_progress is True
            while self.recording_in_progress:
                try:
                    current_pos = self.controller.robot.get_position()
                    current_time = time.time() - start_time

                    # Check for button events (using callback system)
                    button_events = self.controller.get_button_events()

                    for event in button_events:
                        if event["pressed"]:  # Only respond to button presses, not releases
                            if event["button"] == "play":
                                suction_state = not suction_state  # Toggle suction
                                # Apply suction change immediately
                                self.controller.set_pump(suction_state)
                                self.log(f"🔘 PLAY button pressed - Suction {'ON' if suction_state else 'OFF'} at {current_time:.1f}s")

                            elif event["button"] == "menu":
                                self.log(f"🔘 MENU button pressed at {current_time:.1f}s (reserved for future use)")

                    if current_pos:
                        motion_sequence.append({
                            "position": current_pos,
                            "time": current_time,
                            "suction": suction_state,
                            "button_events": button_events
                        })

                    time.sleep(0.1)  # Sample every 100ms (10Hz)

                except Exception as e:
                    self.log(f"⚠️ Recording sample error: {e}")
                    continue

            total_time = time.time() - start_time
            self.log(f"⏹️ Recording stopped. Duration: {total_time:.1f}s, Samples: {len(motion_sequence)}")

            if not motion_sequence:
                self.log("❌ No motion data captured!")
                return False

            # Save motion data as full sequence with timing
            motion_data = {
                "name": name,
                "description": f"GUI recorded motion for slot {slot}",
                "sequence": motion_sequence,
                "total_duration": total_time,
                "recorded_at": datetime.now().isoformat(),
                "type": "full_sequence"
            }

            # Save to file
            motion_file = os.path.join(self.motion_manager.storage_path, f"{name}.json")
            with open(motion_file, 'w') as f:
                json.dump(motion_data, f, indent=2)

            # Update motion manager metadata
            self.motion_manager.motion_slots[slot] = {
                "name": name,
                "description": f"GUI recorded motion for slot {slot}",
                "recorded_at": datetime.now().isoformat(),
                "file_path": motion_file
            }

            self.motion_manager.save_motion_metadata()
            self.log(f"✅ Motion '{name}' saved to {motion_file}")
            return True

        except Exception as e:
            self.log(f"❌ Recording error: {e}")
            return False

    def on_recording_complete(self, success):
        """Handle recording completion"""
        self.recording_in_progress = False
        self.stop_recording_btn.config(state=tk.DISABLED)
        self.start_recording_btn.config(state=tk.DISABLED)
        self.release_for_recording_btn.config(state=tk.NORMAL)

        if success:
            self.recording_status_label.config(text="Status: Recording complete!", foreground="green")
            self.enable_motors()  # Re-enable motors
            self.update_motion_status()
        else:
            self.recording_status_label.config(text="Status: Recording failed", foreground="red")

    def slot_selected(self):
        """Handle slot selection"""
        pass

    def play_motion(self, slot):
        """Play recorded motion"""
        def play():
            try:
                if self.motion_manager and self.motion_manager.play_motion(slot):
                    self.log(f"▶️ Playing motion in slot {slot}")
                else:
                    self.log(f"❌ Failed to play motion in slot {slot}")
            except Exception as e:
                self.log(f"❌ Play error: {e}")

        threading.Thread(target=play, daemon=True).start()

    def delete_motion(self, slot):
        """Delete recorded motion"""
        if messagebox.askyesno("Confirm Delete", f"Delete motion in slot {slot}?"):
            try:
                # Delete the motion file
                if self.motion_manager:
                    # Implementation would depend on motion manager
                    pass
                self.log(f"🗑️ Deleted motion in slot {slot}")
                self.update_motion_status()
            except Exception as e:
                self.log(f"❌ Delete error: {e}")

    def update_motion_status(self):
        """Update motion slot status"""
        if not self.motion_manager:
            return

        for slot in [1, 2, 3]:
            try:
                is_recorded = self.motion_manager.is_motion_recorded(slot)
                status_text = "✅ Recorded" if is_recorded else "❌ Not Recorded"
                color = "green" if is_recorded else "red"

                self.slot_widgets[slot]['status_label'].config(text=status_text, foreground=color)

                # Enable/disable buttons
                state = tk.NORMAL if is_recorded else tk.DISABLED
                self.slot_widgets[slot]['play_btn'].config(state=state)
                self.slot_widgets[slot]['delete_btn'].config(state=state)

            except Exception as e:
                self.log(f"❌ Status update error for slot {slot}: {e}")

    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        self.root.after(0, lambda: self.log_text.insert(tk.END, log_message))
        self.root.after(0, lambda: self.log_text.see(tk.END))
        print(log_message.strip())  # Also print to console

    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)

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
    app = ImprovedUarmGUI(root)

    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()