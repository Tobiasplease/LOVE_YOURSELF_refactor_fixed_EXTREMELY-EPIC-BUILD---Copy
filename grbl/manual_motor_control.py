#!/usr/bin/env python3
"""
GRBL Manual Motor Control
Simple script for individual motor control with live position feedback
Perfect for defining work area edges and fine-tuning positions
"""

import os
import sys
import time
from typing import Tuple

try:
    import termios
    import tty
    TTY_AVAILABLE = True
except ImportError:
    TTY_AVAILABLE = False


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


class KeyboardInput:
    """Cross-platform keyboard input handler"""

    def __init__(self):
        self.old_settings = None
        if TTY_AVAILABLE:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
            except:
                pass

    def __enter__(self):
        if TTY_AVAILABLE and self.old_settings:
            try:
                tty.setraw(sys.stdin.fileno())
            except:
                pass
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if TTY_AVAILABLE and self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass

    def get_char(self):
        """Get single character from keyboard"""
        if TTY_AVAILABLE and self.old_settings:
            try:
                return sys.stdin.read(1)
            except:
                return input("Command: ").lower()[:1]
        else:
            return input("Command: ").lower()[:1]


def parse_position(status_line: str) -> Tuple[float, float, float]:
    """Parse X,Y,Z position from GRBL status line"""
    try:
        # Status format: <Idle|MPos:0.000,0.000,0.000|FS:0,0>
        if 'WPos:' in status_line:
            pos_part = status_line.split('WPos:')[1].split('|')[0]
        elif 'MPos:' in status_line:
            pos_part = status_line.split('MPos:')[1].split('|')[0]
        else:
            return 0.0, 0.0, 0.0

        coords = pos_part.split(',')
        x = float(coords[0]) if len(coords) > 0 else 0.0
        y = float(coords[1]) if len(coords) > 1 else 0.0
        z = float(coords[2]) if len(coords) > 2 else 0.0
        return x, y, z
    except (IndexError, ValueError):
        return 0.0, 0.0, 0.0


def get_machine_state(status_line: str) -> str:
    """Extract machine state from status line"""
    try:
        if status_line.startswith('<'):
            return status_line.split(',')[0][1:]  # Remove < and get first part
        return "Unknown"
    except:
        return "Unknown"


def is_within_bounds(x: float, y: float, bounds: Tuple[float, float, float, float]) -> bool:
    """Check if position is within safe bounds"""
    x_min, x_max, y_min, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')


def display_interface(x_pos: float, y_pos: float, selected_motor: str, step_size: float,
                     machine_state: str, bounds: Tuple[float, float, float, float]):
    """Display the control interface"""
    clear_screen()

    _, x_max, _, y_max = bounds
    in_bounds = is_within_bounds(x_pos, y_pos, bounds)
    bounds_status = "OK" if in_bounds else "WARN"

    print("╔══════════════════════════════════════════════════════════╗")
    print("║                GRBL MANUAL MOTOR CONTROL                 ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║                                                          ║")
    print(f"║  Position:   X: {x_pos:6.2f}mm    Y: {y_pos:6.2f}mm              ║")
    print(f"║  Selected:   [{selected_motor}-MOTOR]     Step: {step_size:4.1f}mm          ║")
    print(f"║  Status:     {machine_state:10s}    Bounds: {bounds_status}             ║")
    print(f"║  Work Area:  X(0-{x_max:.0f})  Y(0-{y_max:.0f})                    ║")
    print("║                                                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  CONTROLS:                                               ║")
    print("║                                                          ║")
    print("║  [X] [Y]     Select X or Y motor                         ║")
    print("║  [A] [D]     Move selected motor left/right              ║")
    print("║  [1] [5] [0] Step size: 0.1 / 1.0 / 5.0 / 10.0 mm       ║")
    print("║                                                          ║")
    print("║  [S]         Set current position as origin (0,0)        ║")
    print("║  [H]         Go to home/origin (0,0)                     ║")
    print("║  [C]         Go to center of work area (25,25)           ║")
    print("║  [Q]         Exit                                        ║")
    print("║                                                          ║")

    if not in_bounds:
        print("║  WARNING: Position outside safe bounds!                  ║")
        print("║                                                          ║")

    print("╚══════════════════════════════════════════════════════════╝")

    if not TTY_AVAILABLE:
        print("\nCommand options: x/y/a/d/1/5/0/s/h/c/q")


def run_manual_control():
    """Main manual control loop"""
    global SER

    try:
        # Connect to GRBL
        print("Connecting to GRBL...")
        try:
            from config.config import GRBL_CNC_PORT
            SER = find_grbl_port(preferred_port=GRBL_CNC_PORT)
        except ImportError:
            SER = find_grbl_port()

        # Setup GRBL - ensure it's homed and ready
        print("Ensuring GRBL is homed...")
        ensure_homed(SER)
        setup_basic_grbl(SER, use_absolute_positioning=True)

        # Control parameters
        selected_motor = 'X'
        step_size = 1.0
        bounds = (0.0, 50.0, 0.0, 50.0)  # 50x50mm work area

        # Set relative positioning mode
        send_cmd(SER, "G91")  # Relative positioning for moves

        print("Starting manual control... (Press any key)")
        time.sleep(2)

        with KeyboardInput() as keyboard:
            while True:
                # Get current position
                status = get_status(SER)
                x_pos, y_pos, _ = parse_position(status)  # Ignore Z position
                machine_state = get_machine_state(status)

                # Display interface
                display_interface(x_pos, y_pos, selected_motor, step_size, machine_state, bounds)

                # Get keyboard input
                try:
                    char = keyboard.get_char()
                except:
                    char = 'q'  # Exit on any input error

                if char.lower() in ['q', '\x1b']:  # ESC or Q
                    break
                elif char.lower() == 'x':
                    selected_motor = 'X'
                elif char.lower() == 'y':
                    selected_motor = 'Y'
                elif char == '1':
                    step_size = 0.1
                elif char == '5':
                    step_size = 1.0
                elif char == '0':
                    step_size = 10.0
                elif char in ['a', 'A'] or char == '\x1b[D':  # A key or left arrow
                    # Move selected motor backward
                    if selected_motor == 'X':
                        new_x = x_pos - step_size
                        if new_x >= bounds[0]:  # Check X min bound
                            send_cmd(SER, f"G0 X-{step_size}")
                    else:  # Y motor
                        new_y = y_pos - step_size
                        if new_y >= bounds[2]:  # Check Y min bound
                            send_cmd(SER, f"G0 Y-{step_size}")
                elif char in ['d', 'D'] or char == '\x1b[C':  # D key or right arrow
                    # Move selected motor forward
                    if selected_motor == 'X':
                        new_x = x_pos + step_size
                        if new_x <= bounds[1]:  # Check X max bound
                            send_cmd(SER, f"G0 X{step_size}")
                    else:  # Y motor
                        new_y = y_pos + step_size
                        if new_y <= bounds[3]:  # Check Y max bound
                            send_cmd(SER, f"G0 Y{step_size}")
                elif char.lower() == 's':
                    # Set current position as origin
                    send_cmd(SER, "G90")  # Absolute mode
                    send_cmd(SER, "G10 L20 P1 X0 Y0 Z0")  # Set work coordinate
                    send_cmd(SER, "G91")  # Back to relative
                    time.sleep(0.5)
                elif char.lower() == 'h':
                    # Go to origin
                    send_cmd(SER, "G90")  # Absolute mode
                    send_cmd(SER, "G0 X0 Y0")
                    send_cmd(SER, "G91")  # Back to relative
                    wait_until_idle(SER, 10)
                elif char.lower() == 'c':
                    # Go to center of work area
                    send_cmd(SER, "G90")  # Absolute mode
                    send_cmd(SER, "G0 X25 Y25")  # Center of 50x50 area
                    send_cmd(SER, "G91")  # Back to relative
                    wait_until_idle(SER, 10)

                # Small delay for responsiveness
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if SER:
            try:
                # Return to absolute positioning
                send_cmd(SER, "G90")
                SER.close()
            except:
                pass
        print("\nManual control ended.")


if __name__ == "__main__":
    run_manual_control()
