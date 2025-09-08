"""
Drawing Self-Inspection Module
Allows the system to automatically look down at its drawing surface during drawing operations
"""

import json
import os
import time
from typing import Optional, Dict, Tuple

# Path to calibrated angles file
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug", "drawing_inspection_angles.json")


def load_inspection_angles() -> Dict:
    """Load calibrated drawing inspection angles"""
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: No calibrated drawing inspection angles found at {CALIBRATION_FILE}")
        print("Run debug/drawing_inspection_calibrator.py to create them")
        return {}


def get_inspection_position(position_name: str) -> Optional[Tuple[int, int]]:
    """Get specific inspection position angles
    
    Args:
        position_name: Name of saved position (e.g., 'drawing_center', 'drawing_near')
        
    Returns:
        Tuple of (pan_angle, tilt_angle) or None if not found
    """
    angles = load_inspection_angles()
    if position_name in angles:
        pos = angles[position_name]
        return (pos['pan'], pos['tilt'])
    return None


def look_at_drawing(servos, position: str = "drawing_center", duration: float = 2.0) -> bool:
    """Look at specific drawing area
    
    Args:
        servos: ServoController instance
        position: Name of saved inspection position
        duration: How long to look (seconds)
        
    Returns:
        True if successful, False if position not found
    """
    inspection_pos = get_inspection_position(position)
    if not inspection_pos:
        print(f"Drawing inspection position '{position}' not calibrated")
        return False
    
    pan_angle, tilt_angle = inspection_pos
    print(f"Looking at drawing ({position}): PAN={pan_angle}°, TILT={tilt_angle}°")
    
    servos.set_pan(pan_angle)
    servos.set_tilt(tilt_angle)
    
    if duration > 0:
        time.sleep(duration)
    
    return True


def sweep_drawing_area(servos, positions: list = None, pause_duration: float = 1.5) -> bool:
    """Sweep across different areas of the drawing surface
    
    Args:
        servos: ServoController instance
        positions: List of position names to visit (default: common drawing areas)
        pause_duration: Seconds to pause at each position
        
    Returns:
        True if any positions were found and visited
    """
    if positions is None:
        positions = ["drawing_near", "drawing_center", "drawing_far", "drawing_left", "drawing_right"]
    
    angles = load_inspection_angles()
    available_positions = [pos for pos in positions if pos in angles]
    
    if not available_positions:
        print("No calibrated drawing inspection positions available for sweep")
        return False
    
    print(f"Sweeping drawing area: {' -> '.join(available_positions)}")
    
    for position in available_positions:
        pos = angles[position]
        pan_angle, tilt_angle = pos['pan'], pos['tilt']
        print(f"  Inspecting {position}: PAN={pan_angle}°, TILT={tilt_angle}°")
        
        servos.set_pan(pan_angle)
        servos.set_tilt(tilt_angle)
        time.sleep(pause_duration)
    
    return True


def return_to_normal_gaze(servos, transition_time: float = 1.0):
    """Return to normal gaze position (center)
    
    Args:
        servos: ServoController instance
        transition_time: Seconds for smooth transition back
    """
    print("Returning to normal gaze position")
    servos.set_pan(90)
    servos.set_tilt(90)
    
    if transition_time > 0:
        time.sleep(transition_time)


class DrawingInspector:
    """Context manager for drawing self-inspection"""
    
    def __init__(self, servos, position: str = "drawing_center", return_to_center: bool = True):
        self.servos = servos
        self.position = position
        self.return_to_center = return_to_center
        self.original_pan = None
        self.original_tilt = None
        self.success = False
    
    def __enter__(self):
        # Save current position (if we can get it somehow)
        # For now, assume we're starting from center
        self.original_pan = 90
        self.original_tilt = 90
        
        # Look at drawing
        self.success = look_at_drawing(self.servos, self.position, duration=0)
        return self.success
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.return_to_center:
            return_to_normal_gaze(self.servos)


# Usage examples:
# 
# # Simple inspection
# look_at_drawing(servos, "drawing_center", duration=3.0)
# 
# # Context manager usage
# with DrawingInspector(servos, "drawing_center") as inspector:
#     if inspector:
#         print("Now looking at drawing center")
#         # Do drawing-related processing
#         time.sleep(2)
#     # Automatically returns to center when done
# 
# # Full area sweep
# sweep_drawing_area(servos, pause_duration=2.0)