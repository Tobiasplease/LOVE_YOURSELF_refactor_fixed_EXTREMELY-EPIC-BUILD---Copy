"""Refuse to start a second machine. Twice now (July's swallowed SIGINT, and
Aug 18: a restart while the old process still held camera+serial) two
machines have raced — the new one starves silently after two log entries
while the old one keeps the body. Pidfile + /proc liveness check; a stale
pidfile (dead pid, or reused by something else) never blocks."""

import os
import sys

PIDFILE = "/tmp/love_yourself_machine.pid"


def refuse_second_machine():
    try:
        if os.path.exists(PIDFILE):
            try:
                pid = int(open(PIDFILE).read().strip())
            except Exception:
                pid = 0
            if pid and pid != os.getpid():
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmd = f.read().replace(b"\x00", b" ").decode(errors="replace")
                except FileNotFoundError:
                    cmd = ""
                if "machine.py" in cmd:
                    print("\n" + "!" * 70)
                    print(f"[MACHINE] ⚠ Another machine.py is ALREADY RUNNING (pid {pid}).")
                    print("[MACHINE] Two machines race for the camera and serial ports — the")
                    print("[MACHINE] new one starves silently. Stop the old one first:")
                    print(f"[MACHINE]   kill {pid}   # then verify: pgrep -f machine.py")
                    print("!" * 70 + "\n")
                    sys.exit(1)
        with open(PIDFILE, "w") as f:
            f.write(str(os.getpid()))
    except SystemExit:
        raise
    except Exception:
        pass  # the guard must never block a legitimate start


# Two sessions built this guard concurrently on Aug 18 under two names, and
# one commit clobbered the other's file — every boot then died on the line-95
# ImportError. Both names stay valid; calling twice is harmless (own pid).
claim_machine_or_exit = refuse_second_machine
