"""Sample llama-server health over a live run, to catch a slow degradation.

The artist's read (Aug 5): multi-image worked fine for weeks and now wedges
"after a while", never recovering. That is the shape of something ACCUMULATING
— VRAM creep, KV/slot state, a leak in the multimodal path — not a bug that
would fire on call one. Nothing we log today would show it, because the event
log records calls, not the server's condition between them.

Samples every INTERVAL seconds into a CSV: VRAM (total and llama-server's own),
process RSS, server uptime (so restarts are visible as a reset), and /health
latency. When the wedge happens, the run-up is in the file.

    python debug/watch_server_health.py            # -> debug/server_health.csv
    python debug/watch_server_health.py --report   # summarise what was captured
"""

import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "server_health.csv")
INTERVAL = 20.0


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def sample():
    row = {"t": round(time.time(), 1), "iso": time.strftime("%H:%M:%S")}
    total = _sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    row["vram_total_mb"] = int(total) if total.isdigit() else -1

    pid = _sh("pgrep -x llama-server | head -1")
    row["server_pid"] = pid or ""
    if pid:
        own = _sh(f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' '$1=={pid}{{print $2}}'")
        row["vram_server_mb"] = int(own) if own.isdigit() else -1
        rss = _sh(f"ps -o rss= -p {pid}")
        row["rss_mb"] = int(rss) // 1024 if rss.strip().isdigit() else -1
        et = _sh(f"ps -o etimes= -p {pid}")
        row["uptime_s"] = int(et) if et.strip().isdigit() else -1
    else:
        row["vram_server_mb"] = row["rss_mb"] = row["uptime_s"] = -1

    t0 = time.time()
    code = _sh("curl -s -o /dev/null -w '%{http_code}' -m 8 http://localhost:8080/health")
    row["health_ms"] = int((time.time() - t0) * 1000)
    row["health"] = code or "none"
    return row


def report():
    if not os.path.exists(OUT):
        print("no samples yet")
        return
    rows = list(csv.DictReader(open(OUT)))
    if not rows:
        print("no samples yet")
        return
    print(f"{len(rows)} samples, {rows[0]['iso']} -> {rows[-1]['iso']}\n")
    # restarts show up as uptime going backwards
    restarts = [r for a, r in zip(rows, rows[1:]) if int(r["uptime_s"] or -1) >= 0 and int(a["uptime_s"] or -1) > int(r["uptime_s"] or -1)]
    print(f"server restarts detected: {len(restarts)}  at {[r['iso'] for r in restarts][:10]}")
    print(f"\n{'time':10} {'uptime_s':>9} {'vram_srv':>9} {'rss_mb':>8} {'health_ms':>10} {'health':>7}")
    step = max(1, len(rows) // 28)
    for r in rows[::step]:
        print(f"{r['iso']:10} {r['uptime_s']:>9} {r['vram_server_mb']:>9} {r['rss_mb']:>8} {r['health_ms']:>10} {r['health']:>7}")
    # does anything creep WITHIN a single server life?
    print("\nper-server-life trend (VRAM and RSS from first to last sample of each life):")
    life, lives = [], []
    for r in rows:
        if life and int(r["uptime_s"] or -1) < int(life[-1]["uptime_s"] or -1):
            lives.append(life)
            life = []
        life.append(r)
    lives.append(life)
    for i, lf in enumerate(l for l in lives if len(l) > 2):
        a, b = lf[0], lf[-1]
        print(
            f"  life {i+1}: {a['iso']}->{b['iso']} "
            f"vram {a['vram_server_mb']}->{b['vram_server_mb']} MB, "
            f"rss {a['rss_mb']}->{b['rss_mb']} MB, "
            f"health {a['health_ms']}->{b['health_ms']} ms"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--interval", type=float, default=INTERVAL)
    a = ap.parse_args()
    if a.report:
        report()
        sys.exit(0)
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = None
    print(f"[watch] sampling every {a.interval:.0f}s -> {OUT}   (Ctrl-C to stop, --report to read)")
    while True:
        row = sample()
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
        w.writerow(row)
        f.flush()
        time.sleep(a.interval)
