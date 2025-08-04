"""
Debug version of the movement learning system to identify issues.
This version has extensive logging and simpler logic.
"""

import tkinter as tk
from tkinter import ttk
import time
import json
import os
import math

class SimpleMovementDebugger:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔍 Movement System Debugger")
        self.root.geometry("800x600")
        
        # Simple recording state
        self.recording = False
        self.recorded_movements = []
        self.last_mouse_pos = None
        self.recording_start_time = None
        
        self.setup_ui()
        self.root.focus_set()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready to debug movement recording", 
                                     font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)
        
        # Recording controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(pady=10)
        
        self.record_btn = ttk.Button(controls_frame, text="🔴 Start Recording", 
                                    command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = ttk.Button(controls_frame, text="🔍 Analyze Recording", 
                                     command=self.analyze_recording, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(controls_frame, text="🗑️ Clear", 
                                   command=self.clear_data)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Movement area
        self.canvas = tk.Canvas(main_frame, bg="black", height=300, width=600)
        self.canvas.pack(pady=10)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Output area
        output_frame = ttk.LabelFrame(main_frame, text="Debug Output", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.output_text = tk.Text(output_frame, height=10, wrap=tk.WORD, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Hotkeys
        self.root.bind('<KeyPress-r>', self.hotkey_record)
        self.root.bind('<KeyPress-R>', self.hotkey_record)
        
        self.log("🚀 Movement debugger initialized")
        self.log("Press R to start/stop recording, move mouse over black area")
        
    def log(self, message):
        """Add a timestamped log message"""
        timestamp = time.strftime("%H:%M:%S")
        self.output_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
        print(f"[{timestamp}] {message}")
        
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.recording:
            # Start recording
            self.recording = True
            self.recorded_movements = []
            self.recording_start_time = time.time()
            self.last_mouse_pos = None
            
            self.record_btn.config(text="⏹️ Stop Recording")
            self.status_label.config(text="🔴 RECORDING - Move mouse over black area", foreground="red")
            self.analyze_btn.config(state=tk.DISABLED)
            
            self.log("🔴 Started recording movements")
            
        else:
            # Stop recording
            self.recording = False
            self.record_btn.config(text="🔴 Start Recording")
            self.status_label.config(text=f"✅ Recorded {len(self.recorded_movements)} movements", 
                                   foreground="green")
            self.analyze_btn.config(state=tk.NORMAL)
            
            self.log(f"⏹️ Stopped recording - captured {len(self.recorded_movements)} points")
            
    def hotkey_record(self, event):
        """Handle R key press"""
        self.log("⌨️ R key pressed - toggling recording")
        self.toggle_recording()
        
    def on_mouse_move(self, event):
        """Track mouse movement in canvas"""
        if not self.recording:
            return
            
        current_time = time.time()
        current_pos = (event.x, event.y)
        
        # Calculate time delta
        if self.last_mouse_pos is None:
            time_delta = 0.0
        else:
            time_delta = current_time - (self.recording_start_time + 
                                       sum(m.get('time_delta', 0) for m in self.recorded_movements))
        
        # Record the movement
        movement_data = {
            'x': event.x,
            'y': event.y,
            'time_delta': time_delta,
            'timestamp': current_time
        }
        
        self.recorded_movements.append(movement_data)
        self.last_mouse_pos = current_pos
        
        # Visual feedback - draw a small dot
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, 
                               fill="white", outline="")
        
        # Update status every 10 points
        if len(self.recorded_movements) % 10 == 0:
            self.status_label.config(text=f"🔴 RECORDING - {len(self.recorded_movements)} points")
            
    def analyze_recording(self):
        """Analyze the recorded movements with detailed logging"""
        if not self.recorded_movements:
            self.log("❌ No movements to analyze")
            return
            
        self.log(f"🔍 Starting analysis of {len(self.recorded_movements)} movements...")
        
        # Clear canvas and show analysis
        self.canvas.delete("all")
        
        movements = self.recorded_movements
        self.log(f"📊 Raw data: {len(movements)} points")
        
        if len(movements) < 2:
            self.log("❌ Need at least 2 points for analysis")
            return
            
        # Basic analysis
        positions = [(m['x'], m['y']) for m in movements]
        time_deltas = [m['time_delta'] for m in movements[1:]]
        
        self.log(f"📍 Position range: X({min(p[0] for p in positions)}-{max(p[0] for p in positions)}), "
                f"Y({min(p[1] for p in positions)}-{max(p[1] for p in positions)})")
        
        # Distance calculation
        distances = []
        total_distance = 0
        for i in range(len(positions) - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            distance = math.sqrt(dx*dx + dy*dy)
            distances.append(distance)
            total_distance += distance
            
        self.log(f"📏 Total distance: {total_distance:.1f} pixels")
        self.log(f"📏 Average distance per step: {total_distance/len(distances):.1f} pixels")
        
        # Speed calculation
        speeds = []
        for i, distance in enumerate(distances):
            if time_deltas[i] > 0:
                speed = distance / time_deltas[i]
                speeds.append(speed)
                
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds)
            min_speed = min(speeds)
            
            self.log(f"🏃 Speed analysis:")
            self.log(f"  • Average: {avg_speed:.1f} px/s")
            self.log(f"  • Maximum: {max_speed:.1f} px/s")
            self.log(f"  • Minimum: {min_speed:.1f} px/s")
            
            # Speed variance
            speed_variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
            self.log(f"  • Variance: {speed_variance:.1f}")
            
        # Time analysis
        if time_deltas:
            avg_time = sum(time_deltas) / len(time_deltas)
            total_time = sum(time_deltas)
            
            self.log(f"⏱️ Timing analysis:")
            self.log(f"  • Total time: {total_time:.2f} seconds")
            self.log(f"  • Average time between points: {avg_time*1000:.1f}ms")
            
            # Pause detection
            long_pauses = [t for t in time_deltas if t > 0.1]  # > 100ms
            self.log(f"  • Long pauses (>100ms): {len(long_pauses)}")
            
        # Movement characteristics for cursor mapping
        self.log("🎯 CURSOR MAPPING PARAMETERS:")
        if speeds:
            base_speed = avg_speed / 100  # Scale down
            chaos_level = min(speed_variance / 1000, 1.0)  # Normalize
            pause_prob = len(long_pauses) / len(time_deltas) if time_deltas else 0
            
            self.log(f"  • Base Speed: {base_speed:.3f}")
            self.log(f"  • Chaos Level: {chaos_level:.3f}")
            self.log(f"  • Pause Probability: {pause_prob:.3f}")
            
        # Visual representation
        self.draw_movement_analysis(positions)
        
        # Save analysis
        self.save_analysis_data()
        
        self.log("✅ Analysis complete!")
        
    def draw_movement_analysis(self, positions):
        """Draw the movement path on canvas"""
        if len(positions) < 2:
            return
            
        # Draw the path
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill="cyan", width=1)
            
        # Mark start and end
        if positions:
            start_x, start_y = positions[0]
            end_x, end_y = positions[-1]
            
            self.canvas.create_oval(start_x-5, start_y-5, start_x+5, start_y+5, 
                                   fill="green", outline="white", width=2)
            self.canvas.create_text(start_x, start_y-15, text="START", fill="green", font=('Arial', 8, 'bold'))
            
            self.canvas.create_oval(end_x-5, end_y-5, end_x+5, end_y+5, 
                                   fill="red", outline="white", width=2)
            self.canvas.create_text(end_x, end_y-15, text="END", fill="red", font=('Arial', 8, 'bold'))
            
    def save_analysis_data(self):
        """Save the analysis data to a file"""
        if not self.recorded_movements:
            return
            
        filename = f"debug_movement_{int(time.time())}.json"
        filepath = os.path.join(os.getcwd(), filename)
        
        data = {
            'movements': self.recorded_movements,
            'analysis_timestamp': time.time(),
            'total_points': len(self.recorded_movements)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.log(f"💾 Saved analysis data to: {filename}")
        except Exception as e:
            self.log(f"❌ Failed to save data: {e}")
            
    def clear_data(self):
        """Clear all recorded data"""
        self.recorded_movements = []
        self.canvas.delete("all")
        self.status_label.config(text="Cleared - Ready for new recording", foreground="black")
        self.analyze_btn.config(state=tk.DISABLED)
        self.log("🗑️ Cleared all data")
        
    def run(self):
        """Start the debugger"""
        self.log("🎯 Ready! Press R to record, move mouse over black area")
        self.root.mainloop()

if __name__ == "__main__":
    debugger = SimpleMovementDebugger()
    debugger.run()
