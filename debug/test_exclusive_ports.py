"""Prove the two guards against phantom drivers (July 28).

Background: a forgotten login autostart ran a hidden machine.py in tmux;
two processes wrote the same serial ports and the left arm glitched on
interleaved bytes. Two locks now make that class of failure loud:

1. serial exclusive=True — a second opener of the same port raises
   instead of silently interleaving writes (tested on a pty pair)
2. utils.single_instance — a second machine.py exits immediately with a
   clear message; the lock dies with the process so restart loops work

  python debug/test_exclusive_ports.py
"""

import os
import pty
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serial as pyserial

failures = []

# --- 1: exclusive serial ---------------------------------------------------
master, slave = pty.openpty()
name = os.ttyname(slave)
first = pyserial.Serial(name, 9600, timeout=0.1, exclusive=True)
try:
    pyserial.Serial(name, 9600, timeout=0.1, exclusive=True)
    failures.append("second exclusive open of the same port succeeded")
except pyserial.SerialException:
    pass
first.close()
second = pyserial.Serial(name, 9600, timeout=0.1, exclusive=True)  # freed on close
second.close()
print("serial: second opener refused while held, admitted after close")

# --- 2: machine single-instance lock ---------------------------------------
HOLDER = """
import sys, time
sys.path.insert(0, {root!r})
from utils.single_instance import claim_machine_or_exit
claim_machine_or_exit()
print("claimed", flush=True)
time.sleep(10)
"""
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a = subprocess.Popen([sys.executable, "-c", HOLDER.format(root=root)], stdout=subprocess.PIPE, text=True)
line = a.stdout.readline().strip()
if line != "claimed":
    failures.append(f"first instance failed to claim: {line!r}")
b = subprocess.run([sys.executable, "-c", HOLDER.format(root=root)], capture_output=True, text=True, timeout=15)
if b.returncode != 1 or "already running" not in b.stdout:
    failures.append(f"second instance not refused (rc={b.returncode}, out={b.stdout!r})")
a.kill()
a.wait()
time.sleep(0.2)
c = subprocess.run([sys.executable, "-c", HOLDER.format(root=root).replace("time.sleep(10)", "pass")], capture_output=True, text=True, timeout=15)
if c.returncode != 0 or "claimed" not in c.stdout:
    failures.append(f"lock not released on process death (rc={c.returncode}, out={c.stdout!r})")
print("lock: second machine.py refused with message, lock freed on death")

print("\n" + ("ALL OK" if not failures else "FAILURES:\n  " + "\n  ".join(failures)))
sys.exit(0 if not failures else 1)
