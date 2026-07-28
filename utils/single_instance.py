"""One machine per body.

Two machine.py processes writing the same serial ports interleave bytes and
the servos glitch (July 28: a forgotten login autostart ran a hidden copy in
tmux for months). flock on a pidfile: the second instance exits with a clear
message instead of fighting for the hardware. The kernel releases the lock
when the process dies, so crash-restart loops keep working.
"""

import fcntl
import os
import sys

LOCK_PATH = "/tmp/love_yourself_machine.lock"
_lock_file = None  # held for the life of the process


def claim_machine_or_exit():
    global _lock_file
    _lock_file = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lock_file.seek(0)
        holder = _lock_file.read().strip() or "unknown pid"
        print(f"[LOCK] machine.py is already running ({holder}) — exiting.")
        print("[LOCK] Two machines on one body garble the serial ports.")
        print("[LOCK] Find the other: tmux ls; ps aux | grep machine.py")
        sys.exit(1)
    _lock_file.seek(0)
    _lock_file.truncate()
    _lock_file.write(f"pid {os.getpid()}")
    _lock_file.flush()
