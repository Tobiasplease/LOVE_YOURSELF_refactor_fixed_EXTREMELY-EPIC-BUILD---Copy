#!/usr/bin/env python3
"""
Interactive uArm Teach/Play menu (Official SDK, no Tk)

- Named recordings with a small index so you can select by name.
- In-record pump toggle: press 'm' to toggle suction; 'q' to stop early.
- Clear sequencing: record with standby ON (limp); play with standby OFF + attach; restore standby afterward.

Flow
- Record: standby ON (limp) → start_record(interval) → stop → standby stays ON
- Play:   standby OFF + attach servos → start_play → complete → restore standby
"""

import json
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime

try:
    from uarm.swift.teach import Teach
    from uarm.wrapper.swift_api import SwiftAPI
except Exception as e:
    print("Error: uArm SDK not available. Ensure the official pyuf package is installed.")
    print(f"Details: {e}")
    sys.exit(1)


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


FIXED_PORT = "/dev/arduino_uarm"


class UArmTeachApp:
    def __init__(self):
        self.swift = None
        self.connected = False
        # Fixed device path as requested
        self.port = FIXED_PORT

        self.library_dir = "movement_recordings/uarm"
        self.index_path = os.path.join(self.library_dir, "index.json")
        self.config_path = os.path.join(self.library_dir, "config.json")
        self.record_file = os.path.join(self.library_dir, "teach_motion.txt")
        self.record_duration = 20.0
        self.record_interval = 0.02  # ~50 Hz (stable)
        self.play_file = self.record_file
        self.current_play_name = None
        # Limp mode used during record: 'detach' (via Swift) or 'standby' (Teach standby mode)
        # Default to 'detach' to ensure key1 pump toggles work reliably during recording
        self.limp_mode = "detach"
        self.teach = None
        # Global pump state tracking for key1 button
        self.pump_on = False
        # Home pose (compact/folded) settings
        self.home_x = 110.0
        self.home_y = 0.0
        self.home_z = 120.0
        self.home_speed = 600.0  # mm/min per SDK
        self.home_mode = "xyz"  # 'xyz' or 'polar'
        self.home_stretch = 70.0  # mm reach for polar home
        self.home_rotation = 0.0  # deg base rotation for polar home
        self.use_home_before_record = True
        # Homing before play can add noticeable delay if home_speed is low.
        # Default it OFF for snappy playback; can be toggled in menu 19.
        self.use_home_before_play = False
        self.use_home_after_play = True
        # Note: rely on firmware 'ee,*' pump timing for accuracy
        # Optional slight smoothing during playback (off by default)
        self.smoothing_enabled = False
        self.smoothing_window = 3  # fixed small window for gentle smoothing

        # Load persisted configuration if available
        try:
            self._load_config()
        except Exception:
            pass

    # --- Custom key1 callback for pump control ---
    def _sync_pump_state(self):
        """Sync our pump_on state with actual hardware state"""
        try:
            # Try to get actual pump state from hardware
            pump_state = self.swift.get_pump()
            if pump_state is not None:
                if isinstance(pump_state, (tuple, list)):
                    self.pump_on = bool(pump_state[0])
                else:
                    self.pump_on = bool(pump_state)
        except Exception:
            # If we can't read state, assume pump is off
            self.pump_on = False

    def _custom_key1_callback(self, ret):
        """Custom key1 callback that toggles pump, overriding firmware play function"""
        if ret == '1':  # Button pressed
            try:
                print(f"\nDEBUG: Button pressed, pump_on was: {self.pump_on}")
                # Simple toggle - just flip the state
                self.pump_on = not self.pump_on
                print(f"DEBUG: After toggle, pump_on is: {self.pump_on}")
                self.swift.set_pump(self.pump_on)
                print(f"Hardware button: Pump {'ON' if self.pump_on else 'OFF'}")

                # If we're recording, also add the ee command to the recording
                if self.teach and hasattr(self.teach, '_Teach__is_recording') and self.teach._Teach__is_recording:
                    if hasattr(self.teach, '_Teach__record_list'):
                        self.teach._Teach__record_list.append(f'ee,{1 if self.pump_on else 0}')
                        print("  (Recorded as ee command)")
            except Exception as e:
                print(f"\nHardware button pump toggle failed: {e}")

    # --- Connection management ---
    def connect(self):
        if self.connected:
            print("Already connected.")
            return
        try:
            # Enforce fixed port and provide clear error if missing
            if not os.path.exists(self.port):
                print(f"Error: device not found at {self.port}. Ensure the USB is connected and the udev symlink exists.")
                print("Tip: check 'ls -l /dev/arduino_uarm' and USB cable/extension.")
                return
            print(f"Connecting on fixed port: {self.port}")
            self.swift = SwiftAPI(port=self.port)
            print("Waiting for ready…")
            self.swift.waiting_ready(timeout=10)
            info = self.swift.get_device_info() or {}
            print("Connected. Device info:")
            for k, v in info.items():
                print(f"  {k}: {v}")
            ensure_dir(self.record_file)
            self.teach = Teach(self.record_file, self.swift)
            self.connected = True

            # Override key1 callback to disable firmware play function and enable pump control
            try:
                self.swift.release_key1_callback()  # Remove any existing callbacks
                self.swift.register_key1_callback(callback=self._custom_key1_callback)
                self.swift.set_report_keys(on=True)
                print("Key1 button overridden: now controls pump instead of firmware play function")
            except Exception as e:
                print(f"Warning: Could not override key1 callback: {e}")

            # Sync pump state on connection
            self._sync_pump_state()
            print(f"Pump state synchronized: {'ON' if self.pump_on else 'OFF'}")

            # Expand workspace to allow full range during teaching
            try:
                result = self.swift.set_height_offset(30)
                print(f"Height workspace expanded (+30mm): {result}")
            except Exception as e:
                print(f"Warning: Could not expand height workspace: {e}")

            # Default idle: detach servos so the arm is limp/discrete
            try:
                self.swift.set_servo_detach()
                print("Servos detached (idle limp mode)")
            except Exception:
                pass
        except Exception as e:
            print("Connection failed:", e)
            print(f"Port tried: {self.port}")

    def disconnect(self):
        if not self.connected:
            print("Not connected.")
            return
        try:
            if self.teach:
                try:
                    if self.teach.is_recording():
                        self.teach.stop_record()
                except Exception:
                    pass
                try:
                    if self.teach.is_playing():
                        self.teach.stop_play()
                except Exception:
                    pass
                try:
                    self.teach.stop_standby_mode()
                except Exception:
                    pass
            try:
                if self.swift:
                    self.swift.set_servo_detach()
            except Exception:
                pass
            if self.swift:
                self.swift.disconnect()
        except Exception as e:
            print("Disconnect error:", e)
        finally:
            self.connected = False
            self.swift = None
            self.teach = None

    # --- Recording ---
    def record(self):
        if not self.connected or not self.swift:
            print("Not connected. Choose 1 to connect first.")
            return
        # Prompt for explicit movement name and prepare target file
        name = input("Enter movement name: ").strip()
        if not name:
            print("Recording cancelled (name required).")
            return
        slug = self._slugify(name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_file = os.path.join(self.library_dir, f"{slug}_{ts}.txt")
        ensure_dir(self.record_file)
        # Bind Teach to current file
        self.teach = Teach(self.record_file, self.swift)
        print(f"Recording '{name}' to: {self.record_file}")

        # Re-override key1 callback since Teach() may have registered its own
        try:
            self.swift.release_key1_callback()
            self.swift.register_key1_callback(callback=self._custom_key1_callback)
            self.swift.set_report_keys(on=True)
        except Exception as e:
            print(f"Warning: Could not re-override key1 callback: {e}")

        try:
            # Optionally go to home before recording so playback ends folded
            if self.use_home_before_record:
                self._go_home(reason="before_record")
            if self.limp_mode == "detach":
                print("Detaching servos for manual guidance (limp)…")
                try:
                    self.swift.set_servo_detach()
                except Exception as e:
                    print("Warn: set_servo_detach failed:", e)
            else:
                print("Enabling Teach standby (limp) for manual guidance…")
                try:
                    self.teach.start_standby_mode()
                    # Re-override key1 callback immediately after standby mode starts
                    self.swift.release_key1_callback()
                    self.swift.register_key1_callback(callback=self._custom_key1_callback)
                    self.swift.set_report_keys(on=True)
                    print("Key1 callback re-overridden after standby mode")
                except Exception as e:
                    print("Warn: start_standby_mode failed:", e)

            print(f"Starting record for {self.record_duration:.2f}s at interval {self.record_interval:.3f}s…")
            print("Use base key1 to toggle pump (recorded as ee commands). 'm' toggles live but is NOT recorded.")
            self.teach.start_record(interval=float(self.record_interval))

            # CRITICAL: start_record() calls start_standby_mode() which re-registers Teach callbacks
            # We must override them again immediately after start_record()
            try:
                self.swift.release_key1_callback()
                self.swift.register_key1_callback(callback=self._custom_key1_callback)
                self.swift.set_report_keys(on=True)
                print("Key1 callback final override after start_record()")
            except Exception as e:
                print(f"Warning: Could not final-override key1 callback: {e}")

            # Non-blocking single-key input while recording
            # Prefer reading from controlling TTY to avoid stdin redirection issues
            tty_file = None
            try:
                tty_file = open("/dev/tty", "rb", buffering=0)
                fd = tty_file.fileno()
            except Exception:
                fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                pump_on = False
                t0 = time.monotonic()

                # Optional: also watch hardware pump state changes if SDK exposes a getter
                stop_probe = False

                def _discover_pump_getter(sw):
                    for name in ("get_pump", "get_pump_state", "get_pump_status", "get_suction_cup", "get_valve", "get_air_pump"):
                        g = getattr(sw, name, None)
                        if callable(g):
                            return g
                    return None

                pump_get = _discover_pump_getter(self.swift)

                def pump_probe():
                    nonlocal pump_on
                    last = None
                    # Prime initial
                    try:
                        if pump_get:
                            v = pump_get()
                            last = bool(v[0] if isinstance(v, (tuple, list)) and v else v)
                    except Exception:
                        pass
                    while not stop_probe:
                        if pump_get is None:
                            break
                        try:
                            v = pump_get()
                            cur = bool(v[0] if isinstance(v, (tuple, list)) and v else v)
                            if last is None:
                                last = cur
                            elif cur != last:
                                last = cur
                                pump_on = cur
                                # Rely on firmware Teach to record ee,* via key1; do not inject here
                        except Exception:
                            break
                        time.sleep(0.05)

                probe_thread = None
                if pump_get is not None:
                    try:
                        probe_thread = threading.Thread(target=pump_probe, daemon=True)
                        probe_thread.start()
                        print("(Hardware pump state probe active)")
                    except Exception:
                        probe_thread = None

                last_ui = 0.0
                while time.monotonic() - t0 < self.record_duration:
                    remain = self.record_duration - (time.monotonic() - t0)
                    now = time.monotonic()
                    if now - last_ui >= 0.25:
                        print(f"  Rec… {remain:4.1f}s left  | Pump: {'ON ' if pump_on else 'OFF'}  (m=toggle, q=stop)", end="\r", flush=True)
                        last_ui = now
                    r, _, _ = select.select([fd], [], [], 0.1)
                    if r:
                        try:
                            ch = os.read(fd, 1).decode(errors="ignore")
                        except Exception:
                            ch = ""
                        if ch.lower() == "m":
                            pump_on = not pump_on
                            try:
                                # Official SDK
                                self.swift.set_pump(pump_on)
                                print(f"\nPump {'ON' if pump_on else 'OFF'}")
                            except Exception as e:
                                print("\nPump toggle failed:", e)
                            # Note: 'm' is not recorded; use base key1 to capture ee,*
                        elif ch.lower() == "q":
                            print("\nStop requested by user.")
                            break
                print("\nStopping record…")
            finally:
                stop_probe = True
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
                if tty_file is not None:
                    try:
                        tty_file.close()
                    except Exception:
                        pass
            self.teach.stop_record()
            size = os.path.getsize(self.record_file) if os.path.exists(self.record_file) else 0
            print(f"Saved {size} bytes.")

            # Sync pump state after recording to ensure clean state
            self._sync_pump_state()
            print(f"Post-recording pump state: {'ON' if self.pump_on else 'OFF'}")

            # Pump events are embedded as 'ee,*' inside the Teach file
            # Add to named index
            self._add_movement_index_entry(name=name, file=self.record_file)
            # Make the just-recorded file the active play target
            self.play_file = self.record_file
            self.current_play_name = name
            print(f"Ready to play: {self.current_play_name} -> {self.play_file}")
        except KeyboardInterrupt:
            print("\nInterrupted. Stopping record…")
            try:
                self.teach.stop_record()
            except Exception:
                pass
        except Exception as e:
            print("Record error:", e)
        finally:
            # Default idle: detach servos so the arm is limp/discrete
            try:
                self.swift.set_servo_detach()
            except Exception:
                pass

    # --- Playback ---
    def play(self):
        if not self.connected or not self.swift:
            print("Not connected. Choose 1 to connect first.")
            return
        # Ensure a valid play target; try latest, then prompt selection
        if not self._is_playable(self.play_file):
            print(f"No valid play target. Current: {self.play_file!r}")
            latest = self._pick_most_recent()
            if latest:
                self.play_file = latest
                self.current_play_name = self._resolve_name_from_index(latest) or os.path.basename(latest)
                print(f"Auto-selected latest: {self.current_play_name} -> {self.play_file}")
            if not self._is_playable(self.play_file):
                print("Opening movement list for selection…")
                self.list_recordings()
                if not self._is_playable(self.play_file):
                    print("No playable movement selected.")
                    return
        # Optionally create a smoothed copy of the play file (only one .smooth suffix max)
        play_path = self.play_file
        if self.smoothing_enabled:
            lower = play_path.lower()
            if lower.endswith(".smooth.txt"):
                pass  # already smoothed
            else:
                base_no_smooth = self._strip_smooth_suffix(play_path)
                smoothed = f"{base_no_smooth}.smooth.txt"
                try:
                    self._smooth_teach_file(play_path, smoothed, window=self.smoothing_window)
                    play_path = smoothed
                    print(f"Using smoothed play file: {os.path.basename(smoothed)}")
                except Exception as e:
                    print("Smoothing failed; using original file:", e)
                    play_path = self.play_file
        self.teach = Teach(play_path, self.swift)
        display_name = os.path.basename(play_path)
        self.current_play_name = display_name
        print(f"Playing movement: {display_name}\n  File: {play_path}")

        # Re-override key1 callback since Teach() may have registered its own
        try:
            self.swift.release_key1_callback()
            self.swift.register_key1_callback(callback=self._custom_key1_callback)
            self.swift.set_report_keys(on=True)
        except Exception as e:
            print(f"Warning: Could not re-override key1 callback: {e}")

        try:
            print("Disabling standby and attaching servos for playback…")
            try:
                self.teach.stop_standby_mode()
            except Exception as e:
                print("Warn: stop_standby_mode failed:", e)
            try:
                self.swift.set_servo_attach()
            except Exception as e:
                print("Warn: set_servo_attach failed:", e)

            # Move to home before play if enabled
            if self.use_home_before_play:
                self._go_home(reason="before_play")

            # Small settle to ensure torque engages before stream starts
            time.sleep(0.2)

            print("Starting playback… Press Ctrl+C to abort.")
            self.teach.start_play()

            last_pct = -1
            while self.teach.is_playing():
                try:
                    prog = self.teach.get_progress(wait=False)
                    if prog and len(prog) >= 2:
                        pct = prog[1]
                        if pct != last_pct:
                            print(f"  Progress: {pct:5.1f}%", end="\r", flush=True)
                            last_pct = pct
                except Exception:
                    pass
                time.sleep(0.1)
            print("\nPlayback complete.")
            # Optionally return to home after playback
            if self.use_home_after_play:
                self._go_home(reason="after_play")
        except KeyboardInterrupt:
            print("\nAborting playback…")
            try:
                self.teach.stop_play()
            except Exception:
                pass
        except Exception as e:
            print("Playback error:", e)
        finally:
            # Default idle between plays: detach servos (limp)
            try:
                self.swift.set_servo_detach()
                print("Servos detached (idle limp mode)")
            except Exception:
                pass

    # --- Utilities ---
    def toggle_standby(self):
        if not self.connected or not self.teach:
            print("Not connected.")
            return
        try:
            print("Toggling standby: OFF then ON…")
            try:
                self.teach.stop_standby_mode()
                print("  Standby OFF (servos active)")
            except Exception as e:
                print("  Warn: stop_standby_mode failed:", e)
            time.sleep(0.5)
            try:
                self.teach.start_standby_mode()
                print("  Standby ON (limp)")
            except Exception as e:
                print("  Warn: start_standby_mode failed:", e)
        except Exception as e:
            print("Toggle standby error:", e)

    def attach_servos(self):
        if not self.connected or not self.swift:
            print("Not connected.")
            return
        try:
            self.swift.set_servo_attach()
            print("Servos ATTACHED (active control)")
        except Exception as e:
            print("Attach failed:", e)

    def detach_servos(self):
        if not self.connected or not self.swift:
            print("Not connected.")
            return
        try:
            self.swift.set_servo_detach()
            print("Servos DETACHED (limp)")
        except Exception as e:
            print("Detach failed:", e)

    def toggle_pump_manual(self):
        if not self.connected or not self.swift:
            print("Not connected.")
            return
        try:
            self._sync_pump_state()
            self.pump_on = not self.pump_on
            self.swift.set_pump(self.pump_on)
            print(f"Pump manually toggled: {'ON' if self.pump_on else 'OFF'}")
        except Exception as e:
            print("Manual pump toggle failed:", e)

    def diagnostics(self):
        if not self.connected or not self.swift:
            print("Not connected.")
            return
        try:
            info = self.swift.get_device_info() or {}
            print("Device info:")
            for k, v in info.items():
                print(f"  {k}: {v}")
            print("Home pose:")
            print(f"  Mode={self.home_mode}")
            if self.home_mode == "polar":
                print(f"  Polar: stretch={self.home_stretch:.1f} rot={self.home_rotation:.1f} height={self.home_z:.1f} speed={self.home_speed:.1f}")
            else:
                print(f"  XYZ=({self.home_x:.1f}, {self.home_y:.1f}, {self.home_z:.1f}) speed={self.home_speed:.1f}")
            print(
                f"  Use before record: {self.use_home_before_record} | before play: {self.use_home_before_play} | after play: {self.use_home_after_play}"
            )
            print(f"Smoothing: {'ON' if self.smoothing_enabled else 'OFF'}  (window={self.smoothing_window})")
        except Exception as e:
            print("Diagnostics error:", e)

    def inspect_file(self, path: str):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            print(f"File: {path} | lines: {len(lines)} | size: {os.path.getsize(path)} bytes")
            head = lines[:5]
            tail = lines[-5:] if len(lines) > 5 else []
            if head:
                print("-- first lines --")
                for ln in head:
                    print(ln.rstrip())
            if tail:
                print("-- last lines --")
                for ln in tail:
                    print(ln.rstrip())
        except Exception as e:
            print("Inspect error:", e)

    def list_recordings(self):
        # Prefer named index
        entries = self._load_index()
        if entries:
            entries = sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)
            print("Movements:")
            for i, e in enumerate(entries, 1):
                name = e.get("name", "(unnamed)")
                file = e.get("file", "?")
                t = e.get("created_at", "")
                try:
                    size = os.path.getsize(file)
                except Exception:
                    size = 0
                print(f"  {i:2d}) {name}  [{os.path.basename(file)}]  {t}  ({size} bytes)")
            sel = input("Select movement number to set for playback (Enter to cancel): ").strip()
            if not sel:
                return
            try:
                idx = int(sel)
                if 1 <= idx <= len(entries):
                    chosen = entries[idx - 1]
                    self.play_file = chosen.get("file")
                    self.current_play_name = chosen.get("name")
                    print(f"Selected: {self.current_play_name} -> {self.play_file}")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
            return

        # Fallback to directory scan
        base = self.library_dir
        if not os.path.exists(base):
            print(f"Directory not found: {base}")
            return
        files = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith((".txt", ".teach", ".log")) and f != "index.json"]
        files.sort()
        if not files:
            print(f"No recordings found in {base}")
            return
        print("Available recordings:")
        for i, path in enumerate(files, 1):
            try:
                size = os.path.getsize(path)
            except Exception:
                size = 0
            print(f"  {i:2d}) {os.path.basename(path)}  ({size} bytes)")
        sel = input("Select number to set as play file (or Enter to cancel): ").strip()
        if not sel:
            return
        try:
            idx = int(sel)
            if 1 <= idx <= len(files):
                self.play_file = files[idx - 1]
                self.current_play_name = os.path.basename(self.play_file)
                print(f"Play file set to: {self.play_file}")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

    def show_pump_events(self, path: str):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        # Prefer sidecar if present
        try:
            sidecar = self._pump_sidecar_path(path)
        except Exception:
            sidecar = None
        if sidecar and os.path.exists(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as pf:
                    data = json.load(pf) or {}
                events = data.get("events", []) if isinstance(data, dict) else []
                print(f"Pump sidecar: {sidecar}")
                print(f"Events: {len(events)}")
                if events:
                    first = events[0]
                    last = events[-1]
                    print(f"  First: t={float(first.get('t', 0)):.3f}s on={bool(first.get('on'))}")
                    if last is not first:
                        print(f"  Last:  t={float(last.get('t', 0)):.3f}s on={bool(last.get('on'))}")
                    for e in events[:10]:
                        print(f"    t={float(e.get('t', 0)):.3f}s on={bool(e.get('on'))}")
                    if len(events) > 10:
                        print(f"    ... and {len(events) - 10} more")
                return
            except Exception as e:
                print("Pump sidecar read error:", e)
        # Fallback: scan the play file for pump-related commands (including 'ee,<0|1>' from official Teach)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            matches = []
            ee_count = 0
            keywords = ("M2231", "M2232", "PUMP", "SUCK", "VALVE", "GRIP", "EE,")
            for i, ln in enumerate(lines, 1):
                up = ln.upper()
                if up.startswith("EE,"):
                    ee_count += 1
                    matches.append((i, ln.rstrip()))
                elif any(k in up for k in keywords):
                    matches.append((i, ln.rstrip()))
            if not matches and ee_count == 0:
                print("No pump-related entries found in play file.")
                return
            print(f"Pump-related entries in {path}:")
            if ee_count:
                print(f"  Teach pump toggles (ee,*): {ee_count}")
            for i, text in matches[:30]:
                print(f"  L{i}: {text}")
            if len(matches) > 30:
                print(f"  ... and {len(matches) - 30} more")
        except Exception as e:
            print("Pump event scan error:", e)

    # --- Menu loop ---
    def menu(self):
        while True:
            print()
            print("=== uArm Official Teach (Terminal) ===")
            print(f"Port: {self.port or 'auto-detect'} | Connected: {self.connected}")
            print(f"Record settings: duration {self.record_duration:.1f}s | interval {self.record_interval:.3f}s")
            pf_line = self.play_file
            if self.current_play_name:
                pf_line = f"{self.current_play_name} -> {self.play_file}"
            print(f"Play target:  {pf_line}")
            print("--------------------------------------")
            print("1) Connect")
            print("2) Disconnect")
            print("3) Set port")
            print("4) Record motion")
            print("5) Play motion")
            print("6) Toggle standby (OFF then ON)")
            print("7) Attach servos")
            print("8) Detach servos")
            print("9) Set record settings")
            print("10) Set play file (manual)")
            print("11) Diagnostics")
            print("12) List movements and select play file")
            print("13) Inspect recording file")
            print("14) Inspect play file")
            print("15) Show pump events in play file")
            print(f"16) Toggle record limp mode (now: {self.limp_mode})")
            print("17) Play most recent recording")
            print("18) Set home pose (XYZ, speed)")
            print("19) Toggle home use (before record / before play / after play)")
            print("20) Set home from current (manual limp → capture)")
            print("21) Move to home now")
            print(f"22) Toggle smoothing (now: {'ON' if self.smoothing_enabled else 'OFF'})")
            print("23) Clear all recordings (confirm)")
            print("24) Delete individual recording")
            print("25) Manual pump toggle (if stuck)")
            print("0) Exit")
            try:
                choice = input("> ").strip()
            except EOFError:
                choice = "0"

            if choice == "1":
                self.connect()
            elif choice == "2":
                self.disconnect()
            elif choice == "3":
                print(f"Port is fixed to {FIXED_PORT} and cannot be changed from the menu.")
            elif choice == "4":
                self.record()
            elif choice == "5":
                self.play()
            elif choice == "6":
                self.toggle_standby()
            elif choice == "7":
                self.attach_servos()
            elif choice == "8":
                self.detach_servos()
            elif choice == "9":
                try:
                    d = input(f"Duration seconds [{self.record_duration}]: ").strip()
                    if d:
                        self.record_duration = float(d)
                    i = input(f"Interval seconds [{self.record_interval}]: ").strip()
                    if i:
                        iv = float(i)
                        if iv <= 0:
                            print("Interval must be > 0; keeping previous.")
                        else:
                            self.record_interval = iv
                    try:
                        self._save_config()
                    except Exception:
                        pass
                except ValueError:
                    print("Invalid number. Keeping previous settings.")
            elif choice == "10":
                path = input(f"Play file [{self.play_file}]: ").strip()
                if path:
                    self.play_file = path
                    self.current_play_name = self._resolve_name_from_index(self.play_file) or os.path.basename(self.play_file)
            elif choice == "11":
                self.diagnostics()
            elif choice == "12":
                self.list_recordings()
            elif choice == "13":
                self.inspect_file(self.record_file)
            elif choice == "14":
                self.inspect_file(self.play_file)
            elif choice == "15":
                self.show_pump_events(self.play_file)
            elif choice == "16":
                self.limp_mode = "standby" if self.limp_mode == "detach" else "detach"
                print(f"Record limp mode set to: {self.limp_mode}")
                try:
                    self._save_config()
                except Exception:
                    pass
            elif choice == "17":
                latest = self._pick_most_recent()
                if latest:
                    self.play_file = latest
                    self.current_play_name = self._resolve_name_from_index(latest) or os.path.basename(latest)
                    print(f"Selected latest: {self.current_play_name} -> {self.play_file}")
                    self.play()
                else:
                    print("No recordings found to play.")
            elif choice == "18":
                try:
                    hx = input(f"Home X [{self.home_x}]: ").strip()
                    if hx:
                        self.home_x = float(hx)
                    hy = input(f"Home Y [{self.home_y}]: ").strip()
                    if hy:
                        self.home_y = float(hy)
                    hz = input(f"Home Z [{self.home_z}]: ").strip()
                    if hz:
                        self.home_z = float(hz)
                    hs = input(f"Home speed [{self.home_speed}]: ").strip()
                    if hs:
                        self.home_speed = float(hs)
                    print(f"Home set to XYZ=({self.home_x:.1f}, {self.home_y:.1f}, {self.home_z:.1f}) @ {self.home_speed:.1f}")
                    try:
                        self._save_config()
                    except Exception:
                        pass
                except ValueError:
                    print("Invalid number. Home unchanged.")
            elif choice == "19":
                self.use_home_before_record = not self.use_home_before_record
                self.use_home_before_play = not self.use_home_before_play
                self.use_home_after_play = not self.use_home_after_play
                print(
                    f"Use home before record: {self.use_home_before_record} | before play: {self.use_home_before_play} | after play: {self.use_home_after_play}"
                )
                try:
                    self._save_config()
                except Exception:
                    pass
            elif choice == "20":
                self.set_home_from_current()
            elif choice == "21":
                self._go_home(reason="manual_test")
            elif choice == "22":
                self.smoothing_enabled = not self.smoothing_enabled
                print(f"Smoothing is now {'ON' if self.smoothing_enabled else 'OFF'}.")
                try:
                    self._save_config()
                except Exception:
                    pass
            elif choice == "23":
                self._clear_all_recordings()
            elif choice == "24":
                self.delete_individual_recording()
            elif choice == "25":
                self.toggle_pump_manual()
            elif choice == "0":
                self.disconnect()
                print("Bye.")
                break
            else:
                print("Unknown choice.")

    # --- Index helpers ---
    def _load_index(self):
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            pass
        return []

    def _save_index(self, entries):
        try:
            ensure_dir(self.index_path)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
        except Exception as e:
            print("Warning: failed to write index:", e)

    def _add_movement_index_entry(self, name: str, file: str):
        entries = self._load_index()
        entries.append({"name": name, "file": file, "created_at": datetime.now().isoformat(timespec="seconds")})
        self._save_index(entries)

    def _resolve_name_from_index(self, file: str):
        entries = self._load_index()
        for e in entries:
            try:
                if os.path.abspath(e.get("file", "")) == os.path.abspath(file):
                    return e.get("name")
            except Exception:
                continue
        return None

    def _slugify(self, name: str):
        keep = []
        for ch in name.strip().lower().replace(" ", "_"):
            if ch.isalnum() or ch in ("_", "-"):
                keep.append(ch)
        slug = "".join(keep) or "movement"
        return slug

    # --- Helpers ---
    def _pump_sidecar_path(self, path: str) -> str:
        base, _ = os.path.splitext(path)
        return f"{base}.pump.json"

    def _is_playable(self, path):
        try:
            return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0
        except Exception:
            return False

    def _pick_most_recent(self):
        try:
            if not os.path.exists(self.library_dir):
                return None
            candidates = []
            for fn in os.listdir(self.library_dir):
                if fn == "index.json":
                    continue
                if not fn.lower().endswith((".txt", ".teach", ".log")):
                    continue
                path = os.path.join(self.library_dir, fn)
                try:
                    if os.path.getsize(path) > 0:
                        candidates.append((os.path.getmtime(path), path))
                except Exception:
                    continue
            if not candidates:
                return None
            candidates.sort(reverse=True)
            return candidates[0][1]
        except Exception:
            return None

    # Move to configured home pose safely
    def _go_home(self, reason: str = ""):
        if not self.swift:
            return
        try:
            # Ensure active control
            try:
                if hasattr(self.teach, "stop_standby_mode"):
                    self.teach.stop_standby_mode()
            except Exception:
                pass
            try:
                self.swift.set_servo_attach()
            except Exception:
                pass
            # Some firmware benefits from larger pending queue
            try:
                self.swift.set_property("cmd_pend_size", 20)
            except Exception:
                pass
            # Move to home (staged for safety)
            if self.home_mode == "polar":
                # Stage 1: raise to safe height
                safe_h = max(self.home_z, 130.0)
                print(f"Moving to home ({reason}): POLAR safe height {safe_h:.1f}")
                try:
                    self.swift.set_polar(height=safe_h, speed=max(self.home_speed, 800.0), wait=True, timeout=5)
                except Exception:
                    pass
                # Stage 2: reduce stretch (fold in)
                try:
                    self.swift.set_polar(stretch=self.home_stretch, speed=max(self.home_speed, 800.0), wait=True, timeout=5)
                except Exception:
                    pass
                # Stage 3: set rotation
                try:
                    self.swift.set_polar(rotation=self.home_rotation, speed=max(self.home_speed, 800.0), wait=True, timeout=5)
                except Exception:
                    pass
                # Stage 4: final height
                try:
                    self.swift.set_polar(height=self.home_z, speed=max(self.home_speed, 800.0), wait=True, timeout=5)
                except Exception as e:
                    print("Home polar move warning:", e)
                # Ensure queue empty then fine-correct to exact XYZ if needed
                try:
                    self.swift.flush_cmd(wait_stop=True)
                except Exception:
                    pass
                self._nudge_to_exact(float(self.home_x), float(self.home_y), float(self.home_z))
            else:
                # XYZ staged: up, in, down
                x, y, z = float(self.home_x), float(self.home_y), float(self.home_z)
                spd = max(float(self.home_speed), 800.0)
                safe_z = max(z, 130.0)
                print(f"Moving to home ({reason}): raise to Z={safe_z:.1f}")
                try:
                    self.swift.set_position(None, None, safe_z, speed=spd, wait=True, timeout=5)
                except Exception:
                    pass
                print(f"Moving to home XY=({x:.1f},{y:.1f}) @ Z={safe_z:.1f}")
                try:
                    self.swift.set_position(x, y, safe_z, speed=spd, wait=True, timeout=5)
                except Exception:
                    pass
                print(f"Lowering to Z={z:.1f}")
                try:
                    self.swift.set_position(x, y, z, speed=spd, wait=True, timeout=5)
                except Exception as e:
                    print("Home XYZ move warning:", e)
                # Ensure queue empty then fine-correct to exact XYZ if needed
                try:
                    self.swift.flush_cmd(wait_stop=True)
                except Exception:
                    pass
                self._nudge_to_exact(x, y, z)
        except Exception as e:
            print("Home move warning:", e)

    def _nudge_to_exact(self, x: float, y: float, z: float, tol: float = 0.8):
        """Verify actual position and perform up to two small corrections to hit target XYZ within tol (mm)."""
        if not self.swift:
            return
        try:
            for _ in range(2):
                pos = None
                try:
                    pos = self.swift.get_position()
                except Exception:
                    pos = None
                if not pos or pos == "TIMEOUT" or len(pos) < 3:
                    break
                dx = float(x) - float(pos[0])
                dy = float(y) - float(pos[1])
                dz = float(z) - float(pos[2])
                if abs(dx) <= tol and abs(dy) <= tol and abs(dz) <= tol:
                    break
                # Small corrective move at a moderate accurate speed
                try:
                    self.swift.set_position(x, y, z, speed=300.0, wait=True, timeout=5)
                    try:
                        self.swift.flush_cmd(wait_stop=True)
                    except Exception:
                        pass
                except Exception:
                    break
        except Exception:
            pass

    def set_home_from_current(self):
        if not self.connected or not self.swift:
            print("Not connected. Choose 1 to connect first.")
            return
        print("Entering manual home set mode: detaching servos (limp). Move arm by hand to desired home, then press Enter.")
        try:
            try:
                self.swift.set_servo_detach()
            except Exception:
                pass
            input("When positioned, press Enter to capture (or Ctrl+C to cancel)… ")
            # Try a few reads to get a stable position
            pos = None
            for _ in range(5):
                try:
                    p = self.swift.get_position()
                    if p and p != "TIMEOUT" and len(p) >= 3:
                        pos = p
                        break
                except Exception:
                    time.sleep(0.05)
            if not pos:
                print("Could not read position; home unchanged.")
                return
            self.home_x = float(pos[0])
            self.home_y = float(pos[1])
            self.home_z = float(pos[2])
            print(f"Captured home: XYZ=({self.home_x:.1f}, {self.home_y:.1f}, {self.home_z:.1f})")
            try:
                self._save_config()
                print("Saved to config.")
            except Exception:
                pass
        except KeyboardInterrupt:
            print("\nCanceled.")
        finally:
            # Restore to a controlled state: attach then standby
            try:
                self.swift.set_servo_attach()
            except Exception:
                pass
            try:
                self.teach.start_standby_mode()
            except Exception:
                pass

    # --- Config persistence ---
    def _load_config(self):
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
        if not isinstance(cfg, dict):
            return
        # Always use fixed port regardless of config
        self.port = FIXED_PORT
        self.record_duration = float(cfg.get("record_duration", self.record_duration))
        self.record_interval = float(cfg.get("record_interval", self.record_interval))
        lm = cfg.get("limp_mode", self.limp_mode)
        if lm in ("detach", "standby"):
            self.limp_mode = lm
        self.home_x = float(cfg.get("home_x", self.home_x))
        self.home_y = float(cfg.get("home_y", self.home_y))
        self.home_z = float(cfg.get("home_z", self.home_z))
        self.home_speed = float(cfg.get("home_speed", self.home_speed))
        self.use_home_before_record = bool(cfg.get("use_home_before_record", self.use_home_before_record))
        # Default before_play to False if not set for snappy playback
        self.use_home_before_play = bool(cfg.get("use_home_before_play", self.use_home_before_play))
        self.use_home_after_play = bool(cfg.get("use_home_after_play", self.use_home_after_play))
        self.home_mode = cfg.get("home_mode", self.home_mode)
        if self.home_mode not in ("xyz", "polar"):
            self.home_mode = "xyz"
        self.home_stretch = float(cfg.get("home_stretch", self.home_stretch))
        self.home_rotation = float(cfg.get("home_rotation", self.home_rotation))
        self.smoothing_enabled = bool(cfg.get("smoothing_enabled", self.smoothing_enabled))
        self.smoothing_window = int(cfg.get("smoothing_window", self.smoothing_window))

    def _save_config(self):
        ensure_dir(self.config_path)
        cfg = {
            "port": self.port,
            "record_duration": self.record_duration,
            "record_interval": self.record_interval,
            "limp_mode": self.limp_mode,
            "home_x": self.home_x,
            "home_y": self.home_y,
            "home_z": self.home_z,
            "home_speed": self.home_speed,
            "use_home_before_record": self.use_home_before_record,
            "use_home_before_play": self.use_home_before_play,
            "use_home_after_play": self.use_home_after_play,
            "home_mode": self.home_mode,
            "home_stretch": self.home_stretch,
            "home_rotation": self.home_rotation,
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_window": self.smoothing_window,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    # --- Smoothing helpers ---
    def _strip_smooth_suffix(self, path: str) -> str:
        base, _ = os.path.splitext(path)
        while base.lower().endswith(".smooth"):
            base = base[:-7]
        return base

    def _smooth_teach_file(self, in_path: str, out_path: str, window: int = 3):
        # Apply a very light 3-point smoothing on G0 lines only; keep ee,* untouched
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        g0 = []
        indices = []
        for i, ln in enumerate(lines):
            if ln.startswith("G0,"):
                parts = ln.split(",")
                if len(parts) >= 6:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        a = float(parts[4])
                        sp = float(parts[5])
                        g0.append([x, y, z, a, sp])
                        indices.append(i)
                    except Exception:
                        pass
        if not g0 or len(g0) < 3 or window < 3:
            # Not enough points; just copy
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return

        # 3-point kernel [0.25, 0.5, 0.25]
        kernel = (0.25, 0.5, 0.25)
        smoothed = g0[:]
        for i in range(1, len(g0) - 1):
            p0, p1, p2 = g0[i - 1], g0[i], g0[i + 1]
            sx = kernel[0] * p0[0] + kernel[1] * p1[0] + kernel[2] * p2[0]
            sy = kernel[0] * p0[1] + kernel[1] * p1[1] + kernel[2] * p2[1]
            sz = kernel[0] * p0[2] + kernel[1] * p1[2] + kernel[2] * p2[2]
            # Keep angle and speed from center point to avoid gripper/wrist artifacts
            smoothed[i] = [sx, sy, sz, p1[3], p1[4]]

        # Rebuild lines
        out_lines = lines[:]
        for idx, gi in enumerate(indices):
            x, y, z, a, sp = smoothed[idx]
            out_lines[gi] = f"G0,{x:.3f},{y:.3f},{z:.3f},{a:.3f},{int(sp)}"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))

    # --- Recording management ---
    def delete_individual_recording(self):
        entries = self._load_index()
        if not entries:
            # Fallback to directory scan if no index
            base = self.library_dir
            if not os.path.exists(base):
                print("No recordings directory found.")
                return
            files = [os.path.join(base, f) for f in os.listdir(base) if f.lower().endswith((".txt", ".teach", ".log")) and f != "index.json"]
            if not files:
                print("No recordings found.")
                return
            print("Available recordings:")
            for i, path in enumerate(files, 1):
                try:
                    size = os.path.getsize(path)
                except Exception:
                    size = 0
                print(f"  {i:2d}) {os.path.basename(path)}  ({size} bytes)")
            sel = input("Select number to DELETE (or Enter to cancel): ").strip()
            if not sel:
                return
            try:
                idx = int(sel)
                if 1 <= idx <= len(files):
                    file_path = files[idx - 1]
                    file_name = os.path.basename(file_path)
                    confirm = input(f"Delete '{file_name}'? Type 'YES' to confirm: ").strip()
                    if confirm == "YES":
                        os.remove(file_path)
                        print(f"Deleted: {file_name}")
                        # Also remove any associated .smooth files
                        base_name = os.path.splitext(file_path)[0]
                        smooth_file = f"{base_name}.smooth.txt"
                        if os.path.exists(smooth_file):
                            os.remove(smooth_file)
                            print(f"Also deleted: {os.path.basename(smooth_file)}")
                    else:
                        print("Cancelled.")
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
            return

        # Use named index
        entries = sorted(entries, key=lambda e: e.get("created_at", ""), reverse=True)
        print("Recorded movements:")
        for i, e in enumerate(entries, 1):
            name = e.get("name", "(unnamed)")
            file = e.get("file", "?")
            t = e.get("created_at", "")
            try:
                size = os.path.getsize(file)
            except Exception:
                size = 0
            print(f"  {i:2d}) {name}  [{os.path.basename(file)}]  {t}  ({size} bytes)")

        sel = input("Select movement number to DELETE (Enter to cancel): ").strip()
        if not sel:
            return
        try:
            idx = int(sel)
            if 1 <= idx <= len(entries):
                entry = entries[idx - 1]
                name = entry.get("name", "(unnamed)")
                file_path = entry.get("file")

                confirm = input(f"Delete '{name}'? Type 'YES' to confirm: ").strip()
                if confirm == "YES":
                    # Remove the file
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {os.path.basename(file_path)}")

                    # Remove any associated .smooth files
                    if file_path:
                        base_name = os.path.splitext(file_path)[0]
                        smooth_file = f"{base_name}.smooth.txt"
                        if os.path.exists(smooth_file):
                            os.remove(smooth_file)
                            print(f"Also deleted: {os.path.basename(smooth_file)}")

                    # Remove from index
                    entries.pop(idx - 1)
                    self._save_index(entries)
                    print(f"Removed '{name}' from index.")

                    # Clear play target if it was the deleted file
                    if self.play_file == file_path:
                        self.play_file = self.record_file
                        self.current_play_name = None
                        print("Play target reset.")
                else:
                    print("Cancelled.")
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

    def _clear_all_recordings(self):
        print("This will delete all recordings and index in:", self.library_dir)
        print("Config will be kept.")
        ans = input("Type 'YES' to confirm: ").strip()
        if ans != "YES":
            print("Cancelled.")
            return
        removed = 0
        try:
            for fn in os.listdir(self.library_dir):
                if fn == "config.json":

                    continue
                path = os.path.join(self.library_dir, fn)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        removed += 1
                except Exception:
                    pass
            print(f"Removed {removed} files. Library cleared (config preserved).")
        except FileNotFoundError:
            print("Library directory not found.")


def main():
    app = UArmTeachApp()
    try:
        app.menu()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting…")
        app.disconnect()


if __name__ == "__main__":
    main()
