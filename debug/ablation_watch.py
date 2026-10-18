"""Watchdog for the Oct 18 ablation run (docs/diary-memory-design-oct18.md).

Emits ONE line per state change and stays silent otherwise, so it can be driven
by a Monitor without flooding. Covers every way the run can fail, not just the
happy path — a restart loop and a wedged generation look identical to "quiet"
unless something says so:

  RESTART   machine.py pid changed (the supervisor relaunched it)
  DOWN      machine.py gone and no STOP file (crashed out of the loop)
  STOPPED   machine.py gone with a STOP file (someone ended it deliberately)
  WEDGED    no new caption for WEDGE_S (the Aug 5 stderr-pipe failure signature)
  UNWEDGED  captions resumed after a wedge
  LLAMA     llama-server gone (nothing can generate)
  GPU       nvidia-smi stopped answering (the Xid 79 signature — card off the bus)
  PROGRESS  a heartbeat every PROGRESS_EVERY_S with caption count and rate

    python debug/ablation_watch.py            # watches the newest run
    python debug/ablation_watch.py <run_id>
"""

import glob
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO, "event_log")
POLL_S = 30
WEDGE_S = 600
PROGRESS_EVERY_S = 1800


def say(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] {tag} {msg}", flush=True)


def machine_pid():
    try:
        out = subprocess.run(["pgrep", "-f", "machine.py"], capture_output=True, text=True, timeout=10).stdout.split()
        return out[0] if out else None
    except Exception:
        return None


def llama_up():
    try:
        return bool(subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        return True  # don't cry wolf on a pgrep hiccup


def gpu_ok():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def newest_run():
    paths = glob.glob(os.path.join(LOG_DIR, "*-event-log.json"))
    if not paths:
        return None
    return os.path.basename(max(paths, key=os.path.getmtime)).split("-event-log")[0]


def caption_count(run):
    path = os.path.join(LOG_DIR, f"{run}-event-log.json")
    try:
        with open(path, errors="replace") as fh:
            return sum(1 for ln in fh if '"type": "caption"' in ln)
    except OSError:
        return 0


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else newest_run()
    if not run:
        say("DOWN", "no event log found")
        return
    pid = machine_pid()
    caps = caption_count(run)
    started = time.time()
    say("WATCH", f"run {run} pid {pid} captions {caps}")

    last_change = time.time()
    last_progress = time.time()
    wedged = False
    gpu_bad = False
    llama_bad = False

    while True:
        time.sleep(POLL_S)
        now = time.time()

        new_pid = machine_pid()
        if new_pid != pid:
            if new_pid is None:
                stop = os.path.exists(os.path.join(REPO, "STOP"))
                say("STOPPED" if stop else "DOWN", f"machine.py gone (STOP file {'present' if stop else 'absent'}) after {(now - started) / 60:.0f} min")
                if stop:
                    return
            else:
                say("RESTART", f"pid {pid} -> {new_pid}; the stationarity measurement needs ONE run, so this splits the record")
                run = newest_run() or run
                caps = caption_count(run)
            pid = new_pid
            last_change = now

        if not llama_up():
            if not llama_bad:
                say("LLAMA", "llama-server is gone — nothing can generate")
                llama_bad = True
        elif llama_bad:
            say("LLAMA", "llama-server is back")
            llama_bad = False

        if not gpu_ok():
            if not gpu_bad:
                say("GPU", "nvidia-smi stopped answering — the Xid 79 signature, card may be off the bus")
                gpu_bad = True
        elif gpu_bad:
            say("GPU", "nvidia-smi answering again")
            gpu_bad = False

        new_caps = caption_count(run)
        if new_caps != caps:
            if wedged:
                say("UNWEDGED", f"captions resumed at {new_caps}")
                wedged = False
            caps = new_caps
            last_change = now
        elif not wedged and now - last_change > WEDGE_S:
            say("WEDGED", f"no new caption for {(now - last_change) / 60:.0f} min at {caps} (generation may have stalled while /health still answers)")
            wedged = True

        if now - last_progress >= PROGRESS_EVERY_S:
            mins = (now - started) / 60
            say("PROGRESS", f"{caps} captions, {mins / 60:.1f}h elapsed, {caps / max(1.0, mins):.1f} captions/min")
            last_progress = now


if __name__ == "__main__":
    main()
