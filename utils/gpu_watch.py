"""GPU watch — telemetry into the event log, and a loud alarm when the card is
gone.

Sep 8 21:02 and Sep 14 23:27 the 3090 fell off the PCIe bus (kernel: NVRM Xid
79, then Xid 154 "Node Reboot Required"), both within 20s of a drawing's VRAM
handoff. Nothing recovers without a reboot. Sep 8 the machine kept running
inference-dead for 1h51m and nothing in its own log said why; Sep 14 it took
Xorg down and fed a boot loop that read as a GPU fault for the wrong reason.
And "some runs the fans work harder" had no record to check against.

Three things, one daemon thread, stdlib only (dashboard/server.py imports the
readers too):

1. Every GPU_WATCH_TICK_S: scan this boot's kernel journal for Xid lines. Ones
   already there when the watch starts are reported once — a machine started
   after a fault is told the card needs a reboot. New ones get a banner and an
   ERROR event immediately. The journal scan runs BEFORE nvidia-smi on every
   tick, because nvidia-smi can hang on a lost card.
2. Every tick: nvidia-smi. A failed or hung query is "GPU lost" — ERROR on the
   transition, banner repeated every GPU_WATCH_REALERT_S while it persists,
   INFO when it answers again.
3. Every GPU_TELEMETRY_LOG_S: one "telemetry" event — GPU temp/fan/power/
   util/VRAM/pstate/throttle reasons, VRAM per process, CPU Tctl, load
   average, and every fan and coolant sensor hwmon exposes.
"""

import os
import re
import shutil
import subprocess
import threading
import time

_XID_RE = re.compile(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+), (.*)")
FATAL_XIDS = {48, 79, 154}  # double-bit ECC, fallen off the bus, reboot required
_unreaped = []  # (binary, proc) killed but not yet exited — at most one per binary

# nvidia-smi clocks_event_reasons bitmask — the ones that say something about
# power or heat. Idle (0x1) and app/user settings are left out.
_THROTTLE_BITS = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}

_GPU_FIELDS = (
    "temperature.gpu",
    "fan.speed",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "memory.used",
    "memory.free",
    "memory.total",
    "pstate",
    "clocks_event_reasons.active",
)


def _run(cmd, timeout):
    """(returncode, stdout) or (None, '') on timeout/missing binary. Never waits
    on a child it had to kill: a process stuck in the driver can sit in D state,
    and subprocess.run's post-timeout wait would hang this thread with it. While
    a killed one is still stuck, no second copy of that binary is started."""
    _unreaped[:] = [(b, p) for b, p in _unreaped if p.poll() is None]
    if any(b == cmd[0] for b, _ in _unreaped):
        return None, ""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except (FileNotFoundError, OSError):
        return None, ""
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        _unreaped.append((cmd[0], proc))
        return None, ""


def _num(s):
    s = s.strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() else round(f, 1)
    except ValueError:
        return None if s in ("", "[N/A]", "N/A", "[Not Supported]") else s


def read_xid_lines(timeout=10):
    """Every Xid line in this boot's kernel journal, oldest first, as dicts
    {when, pci, xid, text, line}. [] if none or the journal is unreadable."""
    rc, out = _run(
        ["journalctl", "-k", "-b", "0", "--no-pager", "-o", "short-iso", "--grep", "NVRM: Xid"],
        timeout,
    )
    if rc != 0:
        return []
    events = []
    for line in out.splitlines():
        m = _XID_RE.search(line)
        if m:
            events.append({"when": line.split(" ", 1)[0], "pci": m.group(1), "xid": int(m.group(2)), "text": m.group(3).strip(), "line": line})
    return events


def read_gpu(timeout=10):
    """(sample dict, None) or (None, reason) when the card does not answer."""
    rc, out = _run(["nvidia-smi", f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits", "-i", "0"], timeout)
    if rc is None:
        return None, "nvidia-smi did not answer"
    first = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in first.split(",")]
    if rc != 0 or len(parts) != len(_GPU_FIELDS):
        return None, (first or f"nvidia-smi exit {rc}")[:200]
    v = dict(zip(_GPU_FIELDS, parts))
    sample = {
        "gpu_temp_c": _num(v["temperature.gpu"]),
        "gpu_fan_pct": _num(v["fan.speed"]),
        "gpu_power_w": _num(v["power.draw"]),
        "gpu_power_limit_w": _num(v["power.limit"]),
        "gpu_util_pct": _num(v["utilization.gpu"]),
        "vram_used_mib": _num(v["memory.used"]),
        "vram_free_mib": _num(v["memory.free"]),
        "vram_total_mib": _num(v["memory.total"]),
        "pstate": v["pstate"],
    }
    try:
        mask = int(v["clocks_event_reasons.active"], 16)
        sample["throttle"] = [name for bit, name in _THROTTLE_BITS.items() if mask & bit]
    except ValueError:
        pass
    rc, apps = _run(["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"], timeout)
    if rc == 0:
        per = {}
        for row in apps.strip().splitlines():
            name, _, mib = row.rpartition(",")
            path = name.strip()
            label = "comfyui" if "ComfyUI" in path else os.path.basename(path) or path
            per[label] = per.get(label, 0) + (_num(mib) or 0)
        sample["vram_by_process_mib"] = per
    return sample, None


def read_host():
    """CPU temperature, load average, and every fan / coolant reading hwmon has."""
    host = {}
    try:
        host["load_1m"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    base = "/sys/class/hwmon"
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return host
    for entry in entries:
        d = os.path.join(base, entry)
        try:
            with open(os.path.join(d, "name")) as f:
                chip = f.read().strip()
        except OSError:
            continue
        for fname in sorted(os.listdir(d)):
            m = re.fullmatch(r"(temp|fan)(\d+)_input", fname)
            if not m:
                continue
            try:
                with open(os.path.join(d, fname)) as f:
                    raw = int(f.read().strip())
            except (OSError, ValueError):
                continue
            label = ""
            try:
                with open(os.path.join(d, f"{m.group(1)}{m.group(2)}_label")) as f:
                    label = f.read().strip()
            except OSError:
                pass
            if chip in ("k10temp", "coretemp") and label in ("Tctl", "Package id 0"):
                host["cpu_temp_c"] = round(raw / 1000, 1)
            elif m.group(1) == "fan":
                host[f"{chip} {label or fname[:-6]} rpm"] = raw
            elif "coolant" in label.lower():
                host[f"{chip} {label} c"] = round(raw / 1000, 1)
    return host


def _log(kind, data):
    try:
        from event_logging.event_logger import log_json_entry
        from event_logging.log_type import LogType

        log_json_entry(LogType[kind], data)
    except Exception:
        pass


def _banner(lines):
    print("\n" + "!" * 70)
    for line in lines:
        print(f"[GPU] {line}")
    print("!" * 70 + "\n")


class GpuWatch:
    def __init__(self, tick_s, log_every_s, realert_s):
        self.tick_s = tick_s
        self.log_every_s = log_every_s
        self.realert_s = realert_s
        self._seen_xids = set()
        self._lost_since = None
        self._last_lost_alert = 0.0
        self._last_sample_log = 0.0
        self.has_nvidia = shutil.which("nvidia-smi") is not None

    def _check_xids(self, first):
        new = [e for e in read_xid_lines() if e["line"] not in self._seen_xids]
        if not new:
            return
        self._seen_xids.update(e["line"] for e in new)
        codes = ", ".join(f"Xid {e['xid']} ({e['text']})" for e in new)
        when = new[0]["when"]
        earlier = "earlier this boot" if first else "just now"
        fatal = any(e["xid"] in FATAL_XIDS for e in new)
        remedy = "stop the machine (./stop_machine.sh) and reboot the computer" if fatal else "none needed if inference and drawing still work"
        _banner([f"THE KERNEL REPORTED A GPU {'FAULT' if fatal else 'ERROR'} {earlier.upper()} — {when}", codes[:300], f"Remedy: {remedy}."])
        _log(
            "ERROR",
            {
                "message": f"GPU {'fault' if fatal else 'error'} in kernel log ({earlier}): " + ", ".join(f"Xid {e['xid']}" for e in new),
                "component": "gpu_watch",
                "fatal": fatal,
                "first_seen_at_start": first,
                "xid_events": [{k: e[k] for k in ("when", "pci", "xid", "text")} for e in new],
                "remedy": remedy,
            },
        )

    def _check_gpu(self, now):
        sample, reason = read_gpu()
        if sample is None:
            if self._lost_since is None:
                self._lost_since = now
                self._last_lost_alert = now
                _banner([f"nvidia-smi cannot reach the GPU: {reason}", "Inference and drawing are dead until the computer is rebooted."])
                _log("ERROR", {"message": "GPU lost — nvidia-smi cannot reach the card", "component": "gpu_watch", "reason": reason})
            elif now - self._last_lost_alert >= self.realert_s:
                self._last_lost_alert = now
                mins = (now - self._lost_since) / 60
                print(f"[GPU] ⚠ still unreachable after {mins:.0f} min ({reason}) — reboot the computer")
            return
        if self._lost_since is not None:
            _log("INFO", {"message": "GPU answering again", "component": "gpu_watch", "lost_for_s": round(now - self._lost_since)})
            print("[GPU] answering again")
            self._lost_since = None
        if now - self._last_sample_log >= self.log_every_s:
            self._last_sample_log = now
            _log("TELEMETRY", {**sample, **read_host()})

    def run(self):
        first = True
        while True:
            try:
                self._check_xids(first)
                first = False
                if self.has_nvidia:
                    self._check_gpu(time.monotonic())
            except Exception as e:
                print(f"[GPU] watch tick failed: {e}")
            time.sleep(self.tick_s)


def start_gpu_watch():
    try:
        from config.config import GPU_TELEMETRY_LOG_S, GPU_WATCH_REALERT_S, GPU_WATCH_TICK_S
    except Exception:
        GPU_WATCH_TICK_S, GPU_TELEMETRY_LOG_S, GPU_WATCH_REALERT_S = 10.0, 30.0, 600.0
    if float(GPU_WATCH_TICK_S) <= 0:
        return None
    watch = GpuWatch(float(GPU_WATCH_TICK_S), float(GPU_TELEMETRY_LOG_S), float(GPU_WATCH_REALERT_S))
    threading.Thread(target=watch.run, daemon=True, name="gpu-watch").start()
    return watch
