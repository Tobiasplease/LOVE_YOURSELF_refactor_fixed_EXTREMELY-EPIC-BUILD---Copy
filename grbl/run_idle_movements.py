#!/usr/bin/env python3
"""
Run GRBL Idle Movements
Standalone script to run continuous idle movements on GRBL CNC
Movements happen in far corner away from home position
"""

import argparse
import os
import random
import signal
import sys
import time

import serial

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grbl.grbl_utils import (
    DEFAULT_BAUD,
    DEFAULT_FEED_RATE,
    PEN_UP_CMD,
    ensure_homed,
    get_status,
    parse_state,
    send_cmd,
    setup_basic_grbl,
    wait_until_idle,
)
from grbl.idle_movements import IdleMovementController

# Global for clean shutdown
ser = None
controller = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n[⚠️] Interrupt received, shutting down safely...")

    if ser:
        try:
            # Stop any current movement
            send_cmd(ser, "!", wait_ok=False)  # Feed hold
            time.sleep(0.5)

            # Pen up to safe height
            send_cmd(ser, PEN_UP_CMD, wait_ok=False)
            time.sleep(0.5)

            # Return to safe position (center of idle zone)
            print("[🏠] Returning to safe position...")
            send_cmd(ser, "G1 X30 Y30 F3000")
            wait_until_idle(ser, 10)

            # Final status
            status = get_status(ser)
            print(f"[📍] Final position: {status}")

            ser.close()
            print("[✅] Shutdown complete")
        except Exception as e:
            print(f"[❌] Error during shutdown: {e}")

    sys.exit(0)


def run_idle_movements(emotion_state="calm_observant", duration=None):
    """
    Run continuous idle movements on GRBL

    Args:
        emotion_state: One of energized_engaged, alert_curious, calm_observant,
                      quiet_detached, withdrawn_distant
        duration: Run for specified seconds (None = run forever)
    """
    global ser, controller

    try:
        # Connect to GRBL using fixed udev symlink
        print("[🔌] Connecting to GRBL...")
        try:
            from config.config import GRBL_CNC_PORT
            print(f"[🔌] Using fixed GRBL port: {GRBL_CNC_PORT}")
            ser = serial.Serial(GRBL_CNC_PORT, DEFAULT_BAUD, timeout=1.0)
        except ImportError:
            # Fallback to default CNC port if config not available
            grbl_port = "/dev/arduino_cnc"
            print(f"[🔌] Using default GRBL port: {grbl_port}")
            ser = serial.Serial(grbl_port, DEFAULT_BAUD, timeout=1.0)

        # Setup signal handler for clean shutdown
        signal.signal(signal.SIGINT, signal_handler)

        # Initialize GRBL
        print("[🏠] Initializing GRBL...")
        ensure_homed(ser)
        setup_basic_grbl(ser, feed_rate=DEFAULT_FEED_RATE, use_absolute_positioning=True)

        # Pen up for all idle movements - higher position to avoid paper contact
        print("[✏️] Pen up for idle movements...")
        send_cmd(ser, PEN_UP_CMD)
        wait_until_idle(ser, 5)

        # Move to idle zone center (40x40mm work area)
        print("[📍] Moving to idle zone center (30, 30)...")
        send_cmd(ser, "G0 X30 Y30 F3000")
        wait_until_idle(ser, 15)

        # Initialize movement controller
        controller = IdleMovementController()
        controller.set_emotion_state(emotion_state)
        print(f"[🎭] Emotion state: {emotion_state}")

        # Track last position to allow arc moves
        last_x, last_y = 30.0, 30.0

        # Start idle movements
        try:
            from config.config import PRINT_CLEAN_CAPTIONS
            show_verbose = not PRINT_CLEAN_CAPTIONS
        except ImportError:
            show_verbose = True  # Default to showing verbose if config unavailable
        
        if show_verbose:
            print("[🌊] Starting idle movements... (Press Ctrl+C to stop)")
            print(f"[📊] Movement radius: {controller.radius_min}-{controller.radius_max}mm")
            print(f"[📊] Feed rate: {controller.feed_rate} mm/min")

        start_time = time.time()
        movement_count = 0
        last_status_time = time.time()

        while True:
            # Check if duration expired
            if duration and (time.time() - start_time) > duration:
                print(f"[⏱️] Duration {duration}s reached, stopping...")
                break

            # Check if we should start a new movement cycle
            if controller.should_start_new_cycle():
                controller.start_new_cycle()
                controller.set_emotion_state(emotion_state)  # Refresh pattern for new cycle

            # Generate next position
            x, y = controller.generate_position()

            # Send movement command (may be linear or arc) without waiting for completion
            gcode = controller.get_move_gcode(last_x, last_y, x, y)
            send_cmd(ser, gcode, wait_ok=False)  # Don't wait - let it flow

            # Occasionally add subtle pen height variations (every 8-15 movements)
            if movement_count % random.randint(8, 15) == 0:
                pen_cmd = controller.get_pen_height_command()
                send_cmd(ser, pen_cmd, wait_ok=False)

            # Only wait briefly for command to be queued, not completed
            time.sleep(0.05)

            movement_count += 1
            controller.movements_in_cycle += 1
            last_x, last_y = x, y

            # Print status every 10 seconds
            if time.time() - last_status_time > 10:
                status = get_status(ser)
                cycle_time = (time.time() - controller.cycle_start_time) / 60
                cycle_progress = (cycle_time / (controller.current_cycle_duration / 60)) * 100
                if show_verbose:
                    print(f"[📊] Movements: {movement_count} | Cycle: {cycle_progress:.0f}% ({cycle_time:.1f}m) | Status: {status}")
                last_status_time = time.time()

                # Randomly vary the pattern more frequently for unpredictability
                if random.random() < 0.15:  # 15% chance each movement to refresh pattern
                    controller.set_emotion_state(emotion_state)  # Refresh variations

                # Occasionally drift the time factor for different curve shapes
                if movement_count % 5 == 0:
                    controller.time_offset += random.uniform(-0.5, 0.5)

            # Variable timing - mostly flowing, with occasional contemplative micro-movements
            timing_type = random.choice(
                [
                    "flow",  # 75% chance - continuous flowing commands
                    "brief",  # 15% chance - brief spacing
                    "contemplation",  # 8% chance - micro-movement contemplation
                    "deep_contemplation",  # 2% chance - extended micro-movement sequence
                ]
            )

            if timing_type == "flow":
                time.sleep(random.uniform(0.1, 0.4))  # Continuous flowing
            elif timing_type == "brief":
                time.sleep(random.uniform(0.6, 1.2))  # Brief spacing
            elif timing_type == "contemplation":
                # Generate contemplative micro-movements instead of stopping
                current_x, current_y = x, y  # Use current position
                micro_sequence = controller.generate_contemplation_sequence(current_x, current_y, random.uniform(3.0, 6.0))

                if show_verbose:
                    print(f"[🧘] Contemplation sequence: {len(micro_sequence)} micro-movements")
                for micro_x, micro_y, micro_feed in micro_sequence:
                    micro_gcode = f"G1 X{micro_x:.3f} Y{micro_y:.3f} F{micro_feed}"
                    send_cmd(ser, micro_gcode, wait_ok=False)
                    time.sleep(random.uniform(0.8, 2.0))  # Gentle pauses between micros
                    movement_count += 1

            else:  # deep_contemplation
                # Extended micro-movement sequence
                current_x, current_y = x, y
                micro_sequence = controller.generate_contemplation_sequence(current_x, current_y, random.uniform(8.0, 15.0))

                if show_verbose:
                    print(f"[🧘] Deep contemplation: {len(micro_sequence)} gentle micro-movements")
                for micro_x, micro_y, micro_feed in micro_sequence:
                    micro_gcode = f"G1 X{micro_x:.3f} Y{micro_y:.3f} F{micro_feed}"
                    send_cmd(ser, micro_gcode, wait_ok=False)
                    time.sleep(random.uniform(1.5, 4.0))  # Longer pauses for deep contemplation
                    movement_count += 1

                    # Occasional pen breathing during deep contemplation
                    if random.random() < 0.3:
                        pen_cmd = controller.get_pen_height_command()
                        send_cmd(ser, pen_cmd, wait_ok=False)

    except KeyboardInterrupt:
        # Handled by signal handler
        pass
    except Exception as e:
        print(f"[❌] Error: {e}")
        if ser:
            try:
                send_cmd(ser, PEN_UP_CMD, wait_ok=False)  # Pen up
                send_cmd(ser, "!", wait_ok=False)  # Feed hold
                ser.close()
            except:
                pass
        sys.exit(1)
    finally:
        if ser and ser.is_open:
            try:
                send_cmd(ser, PEN_UP_CMD, wait_ok=False)
            except Exception:
                pass
            ser.close()


def main():
    parser = argparse.ArgumentParser(description="Run idle movements on GRBL CNC")
    parser.add_argument(
        "--emotion",
        choices=["energized_engaged", "alert_curious", "calm_observant", "quiet_detached", "withdrawn_distant"],
        default="calm_observant",
        help="Emotional state for movement pattern (default: calm_observant)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Run for specified seconds (omit for continuous)",
    )

    args = parser.parse_args()

    print("[🎨] GRBL Idle Movement Controller")
    print("=" * 40)

    run_idle_movements(emotion_state=args.emotion, duration=args.duration)


if __name__ == "__main__":
    main()
