"""Checks utils/gpu_watch.py and the YOLO device events without touching the
running machine or writing to event_log/ (every log call is captured).

    python debug/test_gpu_watch.py

1. live readers on this boot (GPU sample, host sensors, Xid scan)
2. the REAL Sep 14 23:27 Xid lines (journal boot containing them) → one fatal
   ERROR + banner, reported once, "earlier this boot" on the first scan
3. a lost card: ERROR on the transition, re-alert only after REALERT, INFO on
   recovery; telemetry cadence while healthy
4. a hung child: _run returns on timeout, never starts a second copy of the
   same binary, and still runs a different one
5. YOLO start / fallback / retry each write one event with the device
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.gpu_watch as gw

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


logged = []
gw._log = lambda kind, data: logged.append((kind, data))

print("1. live readers")
sample, lost = gw.read_gpu()
check("nvidia-smi answers", sample is not None, lost or f"{sample.get('gpu_temp_c')}°C fan {sample.get('gpu_fan_pct')}% {sample.get('gpu_power_w')} W")
if sample:
    check("sample has the core fields", all(sample.get(k) is not None for k in ("gpu_temp_c", "gpu_power_w", "vram_used_mib", "vram_total_mib")))
    print(f"     vram by process: {sample.get('vram_by_process_mib')}  throttle: {sample.get('throttle')}")
host = gw.read_host()
check("host sensors read", "load_1m" in host, str(host))
live_xids = gw.read_xid_lines()
print(f"     Xid lines this boot: {len(live_xids)}")

print("2. replay of the real Sep 14 23:27 Xid lines")
boot = None
for b in range(-1, -30, -1):
    r = subprocess.run(["journalctl", "-k", "-b", str(b), "--no-pager", "-o", "short-iso", "--grep", "NVRM: Xid"], capture_output=True, text=True)
    if r.returncode == 0 and "2026-09-14T23:27" in r.stdout:
        boot = r.stdout
        break
if boot is None:
    print("     (journal boot with the Sep 14 Xid not found — skipped)")
else:
    real_run = gw._run
    gw._run = lambda cmd, timeout: (0, boot) if cmd[0] == "journalctl" else real_run(cmd, timeout)
    events = gw.read_xid_lines()
    check("parsed Xid 79 and 154", [e["xid"] for e in events if "23:27" in e["when"]] == [79, 154], str([(e["when"], e["xid"]) for e in events]))
    w = gw.GpuWatch(10, 30, 600)
    logged.clear()
    w._check_xids(first=True)
    errs = [d for k, d in logged if k == "ERROR"]
    check("one ERROR for the batch", len(errs) == 1)
    check("fatal, reported as earlier this boot", errs and errs[0]["fatal"] and errs[0]["first_seen_at_start"] and "earlier" in errs[0]["message"])
    logged.clear()
    w._check_xids(first=False)
    check("same lines not reported twice", not logged)
    gw._run = real_run

print("3. lost card, re-alert, recovery, telemetry cadence")
w = gw.GpuWatch(10, 30, 600)
good = ({"gpu_temp_c": 50, "gpu_power_w": 120}, None)
bad = (None, "Unable to determine the device handle for GPU0000:08:00.0: Unknown Error")
real_read_gpu, real_read_host = gw.read_gpu, gw.read_host
gw.read_host = lambda: {"load_1m": 1.0}
seq = {0: good, 10: good, 20: good, 30: good, 40: bad, 50: bad, 640: bad, 650: good}
base = 50000.0  # monotonic clock is seconds of uptime, never near zero
logged.clear()
for t in sorted(seq):
    gw.read_gpu = lambda timeout=10, r=seq[t]: r
    w._check_gpu(base + t)
kinds = [(k, d.get("message", "")) for k, d in logged]
tele = [k for k, _ in kinds if k == "TELEMETRY"]
check("telemetry on the first tick, at +30, and on recovery", len(tele) == 3, str(kinds))
check("one GPU lost ERROR across three failed ticks", sum(1 for k, m in kinds if k == "ERROR" and "GPU lost" in m) == 1)
check("INFO on recovery", any(k == "INFO" and "again" in m for k, m in kinds))
gw.read_gpu, gw.read_host = real_read_gpu, real_read_host

print("4. hung child")
t0 = time.monotonic()
rc, _ = gw._run(["sleep", "30"], timeout=1)
check("timeout returns promptly", rc is None and time.monotonic() - t0 < 3, f"{time.monotonic() - t0:.1f}s")
t0 = time.monotonic()
rc, _ = gw._run(["sleep", "30"], timeout=1)
check("no second copy of a stuck binary", rc is None and time.monotonic() - t0 < 0.5)
rc, out = gw._run(["echo", "ok"], timeout=5)
check("a different binary still runs", rc == 0 and out.strip() == "ok")
for _, p in gw._unreaped:
    p.wait()

print("5. YOLO device events")
import perception.object_detection as od

yolo_logged = []
od.log_json_entry = lambda t, d: yolo_logged.append(d)
real_free = od._free_vram_mib
od._free_vram_mib = lambda: 400
det = od.ObjectDetectionThread()
check("low VRAM at start → cpu event", yolo_logged and yolo_logged[-1]["device"] == "cpu" and yolo_logged[-1]["reason"] == "start", str(yolo_logged[-1:]))
od._free_vram_mib = lambda: 5000
det2 = od.ObjectDetectionThread()
check("room at start → cuda event", yolo_logged[-1]["device"] == "cuda" and yolo_logged[-1]["vram_free_mib"] == 5000)
det2._fall_back_to_cpu(time.time(), RuntimeError("CUDA error: out of memory"))
check("fallback → cpu event with the error", yolo_logged[-1]["reason"] == "cuda_error" and "out of memory" in yolo_logged[-1]["error"])
od._free_vram_mib = real_free

print()
print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
