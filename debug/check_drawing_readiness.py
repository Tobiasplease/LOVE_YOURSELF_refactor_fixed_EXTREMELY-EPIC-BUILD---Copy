#!/usr/bin/env python3
"""
Check if drawing system is ready and diagnose why it might not be triggering.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DRAWING_COOLDOWN, DRAWING_INTERVAL, DRAWING_STARTUP_DELAY
from drawing.drawing import DrawingController
from utils.state_manager import state_manager


def check_drawing_readiness():
    print("=" * 80)
    print("DRAWING SYSTEM READINESS CHECK")
    print("=" * 80)

    # Configuration
    print(f"\n📋 Configuration:")
    print(f"  DRAWING_INTERVAL (check frequency): {DRAWING_INTERVAL}s ({DRAWING_INTERVAL/60:.1f} min)")
    print(f"  DRAWING_COOLDOWN (between drawings): {DRAWING_COOLDOWN}s ({DRAWING_COOLDOWN/60:.1f} min)")
    print(f"  DRAWING_STARTUP_DELAY (first drawing): {DRAWING_STARTUP_DELAY}s")

    # Drawing controller state
    drawing = DrawingController()
    now = time.time()

    print(f"\n🎨 DrawingController State:")
    print(f"  last_drawing_time: {drawing.last_drawing_time:.1f} ({(now - drawing.last_drawing_time):.1f}s ago)")
    print(f"  cooldown: {drawing.cooldown}s")
    print(f"  Cooldown ready: {(now - drawing.last_drawing_time) > drawing.cooldown}")

    # State manager flags
    print(f"\n⚙️ State Manager:")
    print(f"  is_generating_drawing: {getattr(state_manager, 'is_generating_drawing', False)}")
    print(f"  is_executing_cnc: {getattr(state_manager, 'is_executing_cnc', False)}")

    # ready_to_draw check
    print(f"\n✅ ready_to_draw(): {drawing.ready_to_draw()}")

    # First drawing timeline
    print(f"\n⏰ Expected First Drawing Timeline:")
    print(f"  1. Startup: t=0s")
    print(f"  2. Wait for startup delay: {DRAWING_STARTUP_DELAY}s")
    print(f"  3. First check happens: t≈{DRAWING_STARTUP_DELAY + 10}s (after startup delay + caption)")
    print(f"  4. Drawing should trigger if ready_to_draw() returns True")
    print(f"  5. If blocked, next check: t≈{DRAWING_INTERVAL}s ({DRAWING_INTERVAL/60:.1f} minutes!)")

    # Warning about timing issue
    print(f"\n⚠️ POTENTIAL ISSUE:")
    print(f"  - Drawing check happens every {DRAWING_INTERVAL}s ({DRAWING_INTERVAL/60:.1f} min)")
    print(f"  - If first check fails (paper, ComfyUI, etc.), won't retry for {DRAWING_INTERVAL/60:.1f} minutes!")
    print(f"  - Check captioner.py:508 - updates last_drawing_check_time BEFORE verifying drawing can start")


if __name__ == "__main__":
    check_drawing_readiness()
