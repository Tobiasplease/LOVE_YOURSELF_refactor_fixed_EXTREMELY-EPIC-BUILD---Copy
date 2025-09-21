#!/usr/bin/env python3
"""
Simple SVG Transform UI
Provides sliders for scale, skew, translate, rotate before G-code conversion
"""

import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import xml.etree.ElementTree as ET
import re
import math

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bcnc.converters.vpype_converter import VpypeConverter


class SVGTransformUI:
    def __init__(self, root):
        self.root = root
        root.title("SVG Transform Tool")
        root.geometry("900x700")

        # State
        self.input_svg = None
        self.output_path = None
        self.preview_enabled = True
        self.svg_paths = []  # Parsed SVG path data

        # Load saved settings
        self.settings_file = REPO_ROOT / "config" / "svg_transform_settings.json"
        self.load_settings()

        self.setup_ui()

    def setup_ui(self):
        # Main layout: controls on left, preview on right
        main_frame = ttk.PanedWindow(self.root, orient='horizontal')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Left panel: controls
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=1)

        # Right panel: preview
        right_frame = ttk.LabelFrame(main_frame, text="Visual Preview", padding=5)
        main_frame.add(right_frame, weight=2)

        # File selection
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(file_frame, text="Select SVG", command=self.select_svg).pack(side='left')
        self.svg_label = ttk.Label(file_frame, text="No file selected", foreground='gray')
        self.svg_label.pack(side='left', padx=(10, 0))

        # Transform controls
        controls_frame = ttk.LabelFrame(left_frame, text="Transform Controls", padding=10)
        controls_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Preview canvas
        self.canvas = tk.Canvas(right_frame, bg='white', width=400, height=400)
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)

        # Scale
        self.add_slider(controls_frame, "Scale X", "scale_x", 0.1, 3.0, 1.5, 0.1)
        self.add_slider(controls_frame, "Scale Y", "scale_y", 0.1, 3.0, 1.5, 0.1)

        # Translation (mm)
        self.add_slider(controls_frame, "Translate X (mm)", "translate_x", -50, 50, 0, 1)
        self.add_slider(controls_frame, "Translate Y (mm)", "translate_y", -50, 50, 0, 1)

        # Rotation
        self.add_slider(controls_frame, "Rotation (degrees)", "rotation", -15, 15, 3, 0.5)

        # Skew
        self.add_slider(controls_frame, "Skew X (degrees)", "skew_x", -10, 10, 0, 0.5)
        self.add_slider(controls_frame, "Skew Y (degrees)", "skew_y", -10, 10, 0, 0.5)

        # Presets
        preset_frame = ttk.Frame(controls_frame)
        preset_frame.pack(fill='x', pady=(10, 0))

        ttk.Button(preset_frame, text="Reset All", command=self.reset_transforms).pack(side='left')
        ttk.Button(preset_frame, text="Save Preset", command=self.save_preset).pack(side='left', padx=(5, 0))
        ttk.Button(preset_frame, text="Load Preset", command=self.load_preset).pack(side='left', padx=(5, 0))

        # Enable toggle
        self.enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Enable transforms",
                       variable=self.enabled_var).pack(pady=(10, 0))

        # Output
        output_frame = ttk.LabelFrame(left_frame, text="Output", padding=10)
        output_frame.pack(fill='x', padx=10, pady=5)

        # Preview command
        self.command_text = tk.Text(output_frame, height=4, wrap='word',
                                   background='#f8f8f8', state='disabled')
        self.command_text.pack(fill='x', pady=(0, 5))

        # Action buttons
        button_frame = ttk.Frame(output_frame)
        button_frame.pack(fill='x')

        ttk.Button(button_frame, text="Preview Command",
                  command=self.update_preview).pack(side='left')
        ttk.Button(button_frame, text="Convert to G-code",
                  command=self.convert_to_gcode).pack(side='left', padx=(5, 0))

        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                              relief='sunken', anchor='w')
        status_bar.pack(fill='x', side='bottom')

        # Initialize preview after a short delay to ensure UI is ready
        self.root.after(100, self.update_preview)
        self.root.after(200, self.update_visual_preview)

    def add_slider(self, parent, label, var_name, min_val, max_val, default, resolution):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)

        # Label with current value
        label_frame = ttk.Frame(frame)
        label_frame.pack(fill='x')

        ttk.Label(label_frame, text=label).pack(side='left')

        # Variable and value label
        var = tk.DoubleVar(value=default)
        setattr(self, f"{var_name}_var", var)

        value_label = ttk.Label(label_frame, text=f"{default}")
        value_label.pack(side='right')

        # Slider
        slider = ttk.Scale(frame, from_=min_val, to=max_val,
                          variable=var, orient='horizontal')
        slider.pack(fill='x', pady=(2, 0))

        # Update value label and preview when changed
        def on_change(*args):
            value_label.config(text=f"{var.get():.2f}")
            if self.preview_enabled:
                self.root.after_idle(self.update_preview)
                self.root.after_idle(self.update_visual_preview)

        var.trace('w', on_change)

    def select_svg(self):
        filetypes = [
            ("SVG files", "*.svg"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select SVG file",
            filetypes=filetypes,
            initialdir="/home/impostor/ComfyUI/output",
            initialfile="impostor-20250920_005332_00001__center_lined.svg"
        )

        if filename:
            self.input_svg = Path(filename)
            self.svg_label.config(text=self.input_svg.name, foreground='black')
            self.parse_svg()
            self.update_preview()
            self.update_visual_preview()

    def build_vpype_command(self, input_path, output_path):
        """Build vpype command with current transform settings"""
        if not self.enabled_var.get():
            # No transforms, just basic conversion
            return ["vpype", "read", str(input_path), "gwrite", str(output_path)]

        cmd = ["vpype", "read", str(input_path)]

        # Apply transforms in order
        scale_x = self.scale_x_var.get()
        scale_y = self.scale_y_var.get()
        if scale_x != 1.0 or scale_y != 1.0:
            cmd.extend(["scale", str(scale_x), str(scale_y)])

        translate_x = self.translate_x_var.get()
        translate_y = self.translate_y_var.get()
        if translate_x != 0 or translate_y != 0:
            cmd.extend(["translate", f"{translate_x}mm", f"{translate_y}mm"])

        rotation = self.rotation_var.get()
        if rotation != 0:
            cmd.extend(["rotate", f"{rotation}deg"])

        skew_x = self.skew_x_var.get()
        skew_y = self.skew_y_var.get()
        if skew_x != 0 or skew_y != 0:
            cmd.extend(["skew", f"{skew_x}deg", f"{skew_y}deg"])

        # Add G-code generation
        cmd.extend(["gwrite", "--profile", "gcode", str(output_path)])

        return cmd

    def update_preview(self):
        """Update the command preview"""
        if not self.input_svg:
            command = "# Select an SVG file first"
        else:
            temp_output = self.input_svg.parent / f"{self.input_svg.stem}_transformed.gcode"
            cmd = self.build_vpype_command(self.input_svg, temp_output)
            command = " ".join(cmd)

        self.command_text.config(state='normal')
        self.command_text.delete(1.0, 'end')
        self.command_text.insert(1.0, command)
        self.command_text.config(state='disabled')

    def convert_to_gcode(self):
        """Convert SVG to G-code with transforms"""
        if not self.input_svg:
            messagebox.showerror("Error", "Please select an SVG file first")
            return

        # Get output path
        output_path = filedialog.asksaveasfilename(
            title="Save G-code as",
            defaultextension=".gcode",
            filetypes=[("G-code files", "*.gcode"), ("All files", "*.*")],
            initialdir=str(self.input_svg.parent),
            initialfile=f"{self.input_svg.stem}_transformed.gcode"
        )

        if not output_path:
            return

        try:
            self.status_var.set("Converting...")
            self.root.update()

            cmd = self.build_vpype_command(self.input_svg, output_path)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            self.status_var.set(f"Converted: {Path(output_path).name}")
            messagebox.showinfo("Success", f"G-code saved to:\n{output_path}")

        except subprocess.CalledProcessError as e:
            error_msg = f"vpype failed: {e.stderr}"
            self.status_var.set("Conversion failed")
            messagebox.showerror("Error", error_msg)
        except Exception as e:
            self.status_var.set("Conversion failed")
            messagebox.showerror("Error", str(e))

    def reset_transforms(self):
        """Reset all transforms to default values"""
        self.scale_x_var.set(1.0)
        self.scale_y_var.set(1.0)
        self.translate_x_var.set(0)
        self.translate_y_var.set(0)
        self.rotation_var.set(0)
        self.skew_x_var.set(0)
        self.skew_y_var.set(0)
        self.update_preview()

    def save_preset(self):
        """Save current settings as preset"""
        settings = {
            'scale_x': self.scale_x_var.get(),
            'scale_y': self.scale_y_var.get(),
            'translate_x': self.translate_x_var.get(),
            'translate_y': self.translate_y_var.get(),
            'rotation': self.rotation_var.get(),
            'skew_x': self.skew_x_var.get(),
            'skew_y': self.skew_y_var.get(),
            'enabled': self.enabled_var.get()
        }

        try:
            os.makedirs(self.settings_file.parent, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            self.status_var.set("Preset saved")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {e}")

    def load_preset(self):
        """Load saved preset"""
        try:
            self.load_settings()
            self.update_preview()
            self.status_var.set("Preset loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {e}")

    def parse_svg(self):
        """Parse SVG file to extract basic geometry for preview"""
        if not self.input_svg or not self.input_svg.exists():
            self.svg_paths = []
            return

        try:
            tree = ET.parse(self.input_svg)
            root = tree.getroot()
            self.svg_paths = []

            print(f"Parsing SVG: {self.input_svg.name}")

            # Count elements
            polyline_count = 0
            path_count = 0
            rect_count = 0
            line_count = 0

            # Extract basic shapes and paths
            for elem in root.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag  # Handle namespaces

                if tag_name == 'rect':
                    rect_count += 1
                    self.parse_rect(elem)
                elif tag_name == 'path':
                    path_count += 1
                    self.parse_path(elem)
                elif tag_name == 'line':
                    line_count += 1
                    self.parse_line(elem)
                elif tag_name == 'polyline':
                    polyline_count += 1
                    self.parse_polyline(elem)

            print(f"Found: {polyline_count} polylines, {path_count} paths, {rect_count} rects, {line_count} lines")
            print(f"Parsed paths: {len(self.svg_paths)}")

        except Exception as e:
            print(f"SVG parse error: {e}")
            self.svg_paths = []

    def parse_rect(self, elem):
        """Convert rectangle to path points"""
        try:
            x = float(elem.get('x', 0))
            y = float(elem.get('y', 0))
            w = float(elem.get('width', 0))
            h = float(elem.get('height', 0))

            points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
            self.svg_paths.append(points)
        except:
            pass

    def parse_path(self, elem):
        """Basic path parsing - just extract moveto/lineto commands"""
        try:
            d = elem.get('d', '')
            points = []
            current_pos = [0, 0]

            # Very basic parser for M and L commands
            commands = re.findall(r'[ML]\s*[-+]?\d*\.?\d+(?:\s*,?\s*[-+]?\d*\.?\d+)*', d)

            for cmd in commands:
                parts = re.findall(r'[-+]?\d*\.?\d+', cmd)
                if cmd.startswith('M') and len(parts) >= 2:
                    current_pos = [float(parts[0]), float(parts[1])]
                    points.append(tuple(current_pos))
                elif cmd.startswith('L') and len(parts) >= 2:
                    current_pos = [float(parts[0]), float(parts[1])]
                    points.append(tuple(current_pos))

            if points:
                self.svg_paths.append(points)
        except:
            pass

    def parse_line(self, elem):
        """Parse line element"""
        try:
            x1 = float(elem.get('x1', 0))
            y1 = float(elem.get('y1', 0))
            x2 = float(elem.get('x2', 0))
            y2 = float(elem.get('y2', 0))

            # Only add lines that have actual coordinates
            if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                self.svg_paths.append([(x1, y1), (x2, y2)])

        except Exception as e:
            print(f"Line parse error: {e}")
            pass

    def parse_polyline(self, elem):
        """Parse polyline element"""
        try:
            points_str = elem.get('points', '')
            if not points_str:
                return

            # Split by spaces and commas, then filter out empty strings
            coords = re.findall(r'[-+]?\d*\.?\d+', points_str)
            points = []
            for i in range(0, len(coords)-1, 2):
                if i+1 < len(coords):
                    points.append((float(coords[i]), float(coords[i+1])))

            if len(points) > 1:  # Need at least 2 points for a line
                self.svg_paths.append(points)

        except Exception as e:
            print(f"Polyline parse error: {e}, points_str: {points_str[:100]}...")

    def apply_transform(self, points):
        """Apply current transform settings to points"""
        if not self.enabled_var.get():
            return points

        transformed = []

        # Get transform values
        scale_x = self.scale_x_var.get()
        scale_y = self.scale_y_var.get()
        translate_x = self.translate_x_var.get()
        translate_y = self.translate_y_var.get()
        rotation = math.radians(self.rotation_var.get())
        skew_x = math.radians(self.skew_x_var.get())
        skew_y = math.radians(self.skew_y_var.get())

        for x, y in points:
            # Apply transformations in order
            # 1. Scale
            x *= scale_x
            y *= scale_y

            # 2. Skew
            x_skewed = x + y * math.tan(skew_x)
            y_skewed = y + x * math.tan(skew_y)

            # 3. Rotate
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)
            x_rot = x_skewed * cos_r - y_skewed * sin_r
            y_rot = x_skewed * sin_r + y_skewed * cos_r

            # 4. Translate
            x_final = x_rot + translate_x
            y_final = y_rot + translate_y

            transformed.append((x_final, y_final))

        return transformed

    def update_visual_preview(self):
        """Update the visual preview canvas"""
        self.canvas.delete("all")

        # Force canvas update and get actual size
        self.canvas.update()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # Debug info
        self.canvas.create_text(10, 10, text=f"Canvas: {canvas_w}x{canvas_h}",
                               anchor="nw", fill="green", font=("Arial", 8))

        if not self.svg_paths:
            self.canvas.create_text(canvas_w//2, canvas_h//2, text="Load an SVG file to preview",
                                   fill="gray", font=("Arial", 12))
            return

        # Debug: show that we have paths
        self.canvas.create_text(10, 30, text=f"Paths loaded: {len(self.svg_paths)}",
                               anchor="nw", fill="green", font=("Arial", 8))

        # Calculate bounds for all paths
        all_points = []
        all_transformed = []

        for path in self.svg_paths:
            all_points.extend(path)
            transformed = self.apply_transform(path)
            all_transformed.extend(transformed)

        if not all_points:
            return

        # Use actual canvas size (already updated above)
        margin = 30

        # Debug: Draw a simple test square first
        self.canvas.create_rectangle(50, 50, 100, 100, outline="red", width=2)
        self.canvas.create_text(10, 50, text="Test square", anchor="nw", fill="red", font=("Arial", 8))

        # Calculate bounds
        orig_bounds = self.get_bounds(all_points)
        trans_bounds = self.get_bounds(all_transformed)

        # Combine bounds to fit both original and transformed
        min_x = min(orig_bounds[0], trans_bounds[0])
        max_x = max(orig_bounds[1], trans_bounds[1])
        min_y = min(orig_bounds[2], trans_bounds[2])
        max_y = max(orig_bounds[3], trans_bounds[3])

        # Calculate scale to fit in canvas
        data_w = max_x - min_x
        data_h = max_y - min_y

        if data_w == 0 or data_h == 0:
            return

        scale = min((canvas_w - 2*margin) / data_w, (canvas_h - 2*margin) / data_h)

        # Draw original paths in light gray
        for path in self.svg_paths:
            canvas_points = []
            for x, y in path:
                canvas_x = (x - min_x) * scale + margin
                # Flip Y coordinate for normal coordinate system display
                canvas_y = canvas_h - ((y - min_y) * scale + margin)
                canvas_points.extend([canvas_x, canvas_y])

            if len(canvas_points) >= 4:
                self.canvas.create_line(canvas_points, fill="#cccccc", width=1, tags="original")

        # Draw transformed paths in blue
        for path in self.svg_paths:
            transformed = self.apply_transform(path)
            canvas_points = []
            for x, y in transformed:
                canvas_x = (x - min_x) * scale + margin
                # Flip Y coordinate for normal coordinate system display
                canvas_y = canvas_h - ((y - min_y) * scale + margin)
                canvas_points.extend([canvas_x, canvas_y])

            if len(canvas_points) >= 4:
                self.canvas.create_line(canvas_points, fill="#2196F3", width=2, tags="transformed")

        # Add legend
        self.canvas.create_line(10, canvas_h-40, 30, canvas_h-40, fill="#cccccc", width=1)
        self.canvas.create_text(35, canvas_h-40, text="Original", anchor="w", fill="#666")

        self.canvas.create_line(10, canvas_h-20, 30, canvas_h-20, fill="#2196F3", width=2)
        self.canvas.create_text(35, canvas_h-20, text="Transformed", anchor="w", fill="#2196F3")

    def get_bounds(self, points):
        """Get bounding box of points: (min_x, max_x, min_y, max_y)"""
        if not points:
            return (0, 0, 0, 0)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), max(xs), min(ys), max(ys))

    def load_settings(self):
        """Load settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)

                # Apply settings if UI is initialized
                if hasattr(self, 'scale_x_var'):
                    self.scale_x_var.set(settings.get('scale_x', 1.5))
                    self.scale_y_var.set(settings.get('scale_y', 1.5))
                    self.translate_x_var.set(settings.get('translate_x', 0))
                    self.translate_y_var.set(settings.get('translate_y', 0))
                    self.rotation_var.set(settings.get('rotation', 3))
                    self.skew_x_var.set(settings.get('skew_x', 0))
                    self.skew_y_var.set(settings.get('skew_y', 0))
                    self.enabled_var.set(settings.get('enabled', True))
        except Exception:
            pass  # Use defaults if loading fails


def main():
    # Check if vpype is available
    if not VpypeConverter.is_available():
        print("Error: vpype and vpype-gcode are required but not found")
        print("Install with: pip install vpype vpype-gcode")
        return

    root = tk.Tk()
    app = SVGTransformUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()