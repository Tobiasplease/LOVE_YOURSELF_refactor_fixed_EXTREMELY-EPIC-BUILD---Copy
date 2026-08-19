"""Auto-launch ComfyUI with the machine (Aug 19 — artist: "I have to launch
it separately when running outside of tmux").

At boot: if nothing answers on the ComfyUI port, spawn it detached in its own
session (it SURVIVES machine.py restarts — one warm ComfyUI serves many
machine sessions, and the VRAM handoff around drawings already manages the
GPU between the two). Non-blocking: ComfyUI takes ~30-60s to load and the
first drawing is minutes away; the drawing path's reachability probe remains
the authority at draw time. stdout/stderr → event_log/comfyui.log.
"""

import os
import socket
import subprocess

from config.config import MOOD_SNAPSHOT_FOLDER

COMFYUI_DIR = os.getenv("COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))
COMFYUI_PYTHON = os.getenv("COMFYUI_PYTHON", os.path.join(COMFYUI_DIR, ".venv", "bin", "python"))
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", 8188))
COMFYUI_AUTO_START = os.getenv("COMFYUI_AUTO_START", "true").lower() in ("1", "true", "yes")
COMFYUI_LOG = os.path.join(MOOD_SNAPSHOT_FOLDER, "comfyui.log")


def comfyui_reachable(timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", COMFYUI_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_comfyui_up() -> str:
    """Spawn ComfyUI if the port is silent. Returns a short status string.
    Never raises and never blocks on model loading."""
    if not COMFYUI_AUTO_START:
        return "auto-start disabled"
    if comfyui_reachable():
        return "already running"
    if not os.path.isfile(os.path.join(COMFYUI_DIR, "main.py")):
        return f"not found at {COMFYUI_DIR} (set COMFYUI_DIR)"
    try:
        log = open(COMFYUI_LOG, "a")
        log.write("\n=== ComfyUI auto-launched by machine.py ===\n")
        subprocess.Popen(
            [COMFYUI_PYTHON, "main.py", "--port", str(COMFYUI_PORT)],
            cwd=COMFYUI_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # survives machine.py restarts
        )
        return f"launched (loading in background, log: {COMFYUI_LOG})"
    except Exception as e:
        return f"launch failed: {e}"
