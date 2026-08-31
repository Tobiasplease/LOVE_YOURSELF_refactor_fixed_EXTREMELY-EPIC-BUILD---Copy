"""Clock guard — the RTC on this machine runs ~53 days fast, so boots start
in the future until NTP steps the clock back. If machine.py starts inside
that window, the step lands mid-run and every `now - last_X` interval gate
freezes for ~51 days (observed Aug 12 2026: run 980f6e82 went silent 2
minutes in). Two protections, both stdlib-only so they can run before any
project import writes a timestamp:

1. wait_for_clock_sync(): at boot, if NTP is active but the clock not yet
   synchronized AND the machine has a network route, wait briefly for the
   step to happen BEFORE the run starts. Offline (exhibitions) there is no
   route and no step coming — starts immediately, never blocks.
2. start_clock_jump_watch(): wall clock vs monotonic clock drift watch.
   A jump bigger than the threshold can only be a clock step — banner +
   ERROR event, repeated while the skew persists, with the remedy named
   (restart + debug/sanitize_future_timestamps.py).

Env overrides: CLOCK_SYNC_MAX_WAIT_S (default 45), CLOCK_JUMP_ALERT_S (30).
"""

import os
import subprocess
import threading
import time

SYNC_MAX_WAIT_S = float(os.getenv("CLOCK_SYNC_MAX_WAIT_S", 45))
JUMP_ALERT_S = float(os.getenv("CLOCK_JUMP_ALERT_S", 30))
_WATCH_INTERVAL_S = 5.0
_REALERT_EVERY_S = 60.0


def _timedatectl(prop):
    try:
        out = subprocess.run(
            ["timedatectl", "show", "-p", prop, "--value"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _has_default_route():
    try:
        out = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=5)
        return bool(out.stdout.strip())
    except Exception:
        return False


def wait_for_clock_sync(max_wait_s=SYNC_MAX_WAIT_S):
    """Block until the system clock is NTP-synchronized, bounded. Returns a
    short status string. No-ops instantly when sync is done, NTP is off,
    timedatectl is missing, or there is no network to sync against."""
    if _timedatectl("NTPSynchronized") != "no":
        return "clock ok"
    if _timedatectl("NTP") != "yes":
        return "ntp inactive — trusting the clock as it is"
    if not _has_default_route():
        return "offline — no clock step coming, starting on the local clock"

    print(
        f"[CLOCK] Clock not yet NTP-synchronized — waiting up to {max_wait_s:.0f}s "
        f"so the correction lands BEFORE the run starts (RTC runs ~53 days fast)"
    )
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        time.sleep(2.0)
        if _timedatectl("NTPSynchronized") == "yes":
            print("[CLOCK] Clock synchronized — starting clean")
            return "synchronized after wait"
    print("[CLOCK] ⚠ Gave up waiting for sync — if the clock steps mid-run, " "the jump watch will call it out")
    return "sync wait timed out"


class ClockJumpWatch:
    """Detects wall-clock steps by comparing wall time against the monotonic
    clock, which no NTP step can touch. Clock funcs injectable for tests."""

    def __init__(self, threshold_s=JUMP_ALERT_S, wall=time.time, mono=time.monotonic):
        self.threshold_s = threshold_s
        self._wall = wall
        self._mono = mono
        self._ref_wall = wall()
        self._ref_mono = mono()
        self.total_drift_s = 0.0

    def check(self):
        """Return the wall-clock jump (seconds, signed) since last check, or
        0.0 if the clocks moved in step. Negative = clock went backwards."""
        wall, mono = self._wall(), self._mono()
        drift = (wall - self._ref_wall) - (mono - self._ref_mono)
        self._ref_wall, self._ref_mono = wall, mono
        if abs(drift) < self.threshold_s:
            return 0.0
        self.total_drift_s += drift
        return drift


def _alert(drift, total):
    direction = "BACKWARDS" if drift < 0 else "FORWARDS"
    days = abs(total) / 86400.0
    print("\n" + "!" * 70)
    print(f"[CLOCK] ⚠ THE CLOCK JUMPED {direction} by {abs(drift):.0f}s (total {days:.1f} days).")
    print("[CLOCK] Interval gates are now poisoned — the machine may go silent.")
    print("[CLOCK] Remedy: stop machine.py, run debug/sanitize_future_timestamps.py,")
    print("[CLOCK] then restart on the corrected clock.")
    print("!" * 70 + "\n")
    try:
        from event_logging.event_logger import log_json_entry
        from event_logging.log_type import LogType

        log_json_entry(
            LogType.ERROR,
            {
                "message": f"System clock jumped {direction.lower()} by {drift:.0f}s",
                "component": "clock_guard",
                "jump_seconds": round(drift),
                "total_drift_seconds": round(total),
                "remedy": "restart machine.py, then run debug/sanitize_future_timestamps.py",
            },
        )
    except Exception:
        pass


def start_clock_jump_watch():
    watch = ClockJumpWatch()

    def _run():
        last_alert_mono = 0.0
        while True:
            time.sleep(_WATCH_INTERVAL_S)
            drift = watch.check()
            now_mono = time.monotonic()
            if drift and now_mono - last_alert_mono >= _REALERT_EVERY_S:
                _alert(drift, watch.total_drift_s)
                last_alert_mono = now_mono
            elif watch.total_drift_s and now_mono - last_alert_mono >= _REALERT_EVERY_S:
                # skew persists until restart — keep saying so, once a minute
                print(f"[CLOCK] ⚠ Running on a stepped clock (total drift {watch.total_drift_s:+.0f}s) — restart when you can")
                last_alert_mono = now_mono

    threading.Thread(target=_run, daemon=True, name="clock-jump-watch").start()


def guard_clock():
    """Call once, first thing in machine.py: bounded sync wait, then the
    permanent jump watch."""
    status = wait_for_clock_sync()
    print(f"[CLOCK] {status}")
    start_clock_jump_watch()
