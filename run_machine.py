#!/usr/bin/env python3
"""
Auto-restart wrapper for machine.py to handle segfaults and other critical failures.
Provides continuous operation with crash recovery and logging.
"""

import subprocess
import sys
import time
import json
import os
from datetime import datetime
from pathlib import Path

def log_crash(exit_code, stderr_output=None, restart_count=0):
    """Log crash information to event log directory."""
    try:
        log_dir = Path("event_log")
        log_dir.mkdir(exist_ok=True)

        crash_log = {
            "timestamp": datetime.now().isoformat(),
            "exit_code": exit_code,
            "crash_type": "segfault" if exit_code == -11 else "other_crash",
            "restart_count": restart_count,
            "stderr": stderr_output,
            "event_type": "system_crash"
        }

        crash_file = log_dir / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(crash_file, 'w') as f:
            json.dump(crash_log, f, indent=2)

        print(f"[CRASH LOG] Logged to {crash_file}")
    except Exception as e:
        print(f"[ERROR] Failed to log crash: {e}")

def run_machine_with_restart():
    """Run machine.py with automatic restart on crashes."""
    restart_count = 0
    max_restarts = 50  # Prevent infinite restart loops
    restart_delay = 5  # Seconds between restarts

    # Pass through all command line arguments
    cmd_args = ["python", "machine.py"] + sys.argv[1:]

    print(f"[MONITOR] Starting machine.py with auto-restart (max {max_restarts} restarts)")
    print(f"[MONITOR] Command: {' '.join(cmd_args)}")

    while restart_count <= max_restarts:
        try:
            print(f"\n[MONITOR] Starting attempt {restart_count + 1}")

            # Run the main process
            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())

            # Process has terminated
            exit_code = process.wait()
            stderr_output = process.stderr.read()

            if exit_code == 0:
                print("[MONITOR] Clean exit - stopping monitor")
                break
            elif exit_code == -11:  # SIGSEGV (segmentation fault)
                print(f"[MONITOR] SEGMENTATION FAULT detected (exit code: {exit_code})")
                log_crash(exit_code, stderr_output, restart_count)
            elif exit_code < 0:  # Other signals
                print(f"[MONITOR] Process killed by signal {-exit_code}")
                log_crash(exit_code, stderr_output, restart_count)
            else:
                print(f"[MONITOR] Process exited with code {exit_code}")
                log_crash(exit_code, stderr_output, restart_count)

            if stderr_output:
                print(f"[STDERR] {stderr_output}")

            restart_count += 1

            if restart_count <= max_restarts:
                print(f"[MONITOR] Restarting in {restart_delay} seconds... ({restart_count}/{max_restarts})")
                time.sleep(restart_delay)
            else:
                print(f"[MONITOR] Maximum restart limit reached ({max_restarts}). Stopping.")
                break

        except KeyboardInterrupt:
            print("\n[MONITOR] Keyboard interrupt - stopping monitor")
            if 'process' in locals():
                process.terminate()
            break
        except Exception as e:
            print(f"[MONITOR] Unexpected error in monitor: {e}")
            restart_count += 1
            if restart_count <= max_restarts:
                time.sleep(restart_delay)
            else:
                break

if __name__ == "__main__":
    run_machine_with_restart()