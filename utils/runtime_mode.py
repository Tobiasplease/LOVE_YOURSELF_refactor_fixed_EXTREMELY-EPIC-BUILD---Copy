"""Persisted runtime modes — the filesystem is the bus (prompt_overrides pattern).

One JSON file, event_log/runtime_mode.json, read by every gate that needs it
(machine.py loop, kinetic bus supervisor, drawing trigger) and written by the
dashboard sidecar. mtime-cached so per-loop polling costs one stat(). Missing
file = all modes off. Survives restarts by construction — a machine that
reboots while low_energy is set comes back parked.

Modes:
    low_energy   bool — actuation limited to gaze pan/tilt (lung, left arm,
                 CNC drawing, awakening choreography all parked). Lightbulb,
                 LCD, camera and the whole LLM loop stay live.
    digital_only reserved, not yet implemented (see drawing/drawing.py gate).
"""

import json
import os
import time

try:
    from config.config import MOOD_SNAPSHOT_FOLDER
except Exception:
    MOOD_SNAPSHOT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "event_log")

MODE_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "runtime_mode.json")

_cache = {"mtime": None, "data": {}}


def mode() -> dict:
    try:
        mtime = os.stat(MODE_PATH).st_mtime_ns
    except OSError:
        _cache["mtime"], _cache["data"] = None, {}
        return _cache["data"]
    if mtime != _cache["mtime"]:
        try:
            with open(MODE_PATH, encoding="utf-8") as f:
                data = json.load(f)
            _cache["data"] = data if isinstance(data, dict) else {}
        except Exception:
            _cache["data"] = {}
        _cache["mtime"] = mtime
    return _cache["data"]


def low_energy() -> bool:
    return bool(mode().get("low_energy"))


def set_low_energy(on: bool, changed_by: str = "dashboard") -> dict:
    data = dict(mode())
    data["low_energy"] = bool(on)
    data["changed_at"] = time.time()
    data["changed_by"] = changed_by
    tmp = MODE_PATH + ".tmp"
    os.makedirs(os.path.dirname(MODE_PATH), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, MODE_PATH)
    return data
