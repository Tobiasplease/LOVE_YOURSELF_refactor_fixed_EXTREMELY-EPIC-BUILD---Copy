#!/usr/bin/env python3
"""
Finished-drawing capture (Sep 10 2026).

The completion ritual homed the gantry, released the gaze, and let the uArm
discard the sheet — so the machine never saw what it made. This is the look
before the discard, and it reuses the pre-draw paper gate's choreography
verbatim: the kinetic get-clear move so neither arm nor the gantry occludes
the sheet, the gaze parked on the table, frames pulled from the shared camera
feed, then the body released.

CAPTURE ONLY, deliberately. Judging what the pen actually made is the critique
removed Aug 5, and it wants a post-processed image — deskewed and cropped to
the sheet — not a raw table view. This step exists to produce that raw material
and to prove the choreography lands in the window before the paper goes.

Never raises. The completion ritual and the uArm discard must not depend on a
photograph succeeding.
"""

import os
import time
from typing import List, Optional

from config import config as _cfg
from event_logging.event_logger import LogType, log_json_entry


def _shoot(camera) -> List[str]:
    """Park the gaze on the table, write N frames, release the gaze."""
    import cv2

    from config.config import PAPER_DETECTION_GAZE_PAN, PAPER_DETECTION_GAZE_TILT
    from safety.paper_detection import grab_table_frame

    # range 0: the paper gate's organic drift is driven by
    # update_paper_search_target(), which only the aruco sweep calls — and a
    # still camera is what a picture of the drawing wants anyway.
    parked = False
    try:
        from vision.gaze import set_paper_search_mode

        set_paper_search_mode(active=True, center_pan=PAPER_DETECTION_GAZE_PAN, center_tilt=PAPER_DETECTION_GAZE_TILT, range_pan=0.0, range_tilt=0.0)
        parked = True
        time.sleep(float(getattr(_cfg, "FINISHED_CAPTURE_SETTLE_S", 4.0)))
    except Exception:
        pass

    img_dir = os.path.join(_cfg.MOOD_SNAPSHOT_FOLDER, "finished_drawings")
    os.makedirs(img_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out: List[str] = []
    try:
        for i in range(max(1, int(getattr(_cfg, "FINISHED_CAPTURE_FRAMES", 2)))):
            if i > 0:
                time.sleep(0.8)
            frame = grab_table_frame(camera)
            if frame is None:
                print(f"[📷] Finished-drawing capture: no frame ({i})")
                continue
            path = os.path.join(img_dir, f"finished_{stamp}_{i}.jpg")
            if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                out.append(path)
    finally:
        if parked:
            # Also drops the gaze lock (set_paper_search_mode shares the drawing-mode
            # flags); the ritual's explicit unlock right after is then a no-op.
            try:
                from vision.gaze import set_paper_search_mode

                set_paper_search_mode(active=False)
            except Exception:
                pass
    return out


def capture_finished_drawing(camera=None, extra_clear_s: float = 0.0) -> Optional[str]:
    """Photograph the finished sheet before the arm takes it away.

    extra_clear_s: seconds another get-clear started outside this function needs
    before the view is actually clear — the completion ritual replays the paper
    take's gantry track itself (grbl/paper_gantry.py), and both halves of the
    body must finish moving before the shutter.

    Returns the path of the last frame written, or None when disabled, when
    there is no camera, or on any failure.
    """
    if not bool(getattr(_cfg, "ENABLE_FINISHED_DRAWING_CAPTURE", True)):
        return None

    started = time.time()
    try:
        if camera is None:
            from utils.state_manager import state_manager as _sm

            camera = getattr(_sm, "camera", None)
        if camera is None:
            print("[📷] Finished-drawing capture skipped — no camera")
            return None

        clear_wait = 0.0
        clear_expected = False
        try:
            from utils import hooks as _kin_hooks

            if _kin_hooks.on_paper_check_start:
                clear_expected = True
                cap_s = float(getattr(_cfg, "FINISHED_CAPTURE_MAX_CLEAR_S", 20.0))
                clear_wait = min(float(_kin_hooks.on_paper_check_start() or 0.0), cap_s)
        except Exception:
            pass
        wait_s = max(clear_wait, float(extra_clear_s or 0.0))
        if wait_s > 0:
            arms = f"{clear_wait:.1f}s arms" if clear_wait else "arms did not move"
            print(f"[📷] Body clearing the view ({wait_s:.1f}s — {arms}, {extra_clear_s:.1f}s gantry) before photographing the drawing…")
            time.sleep(wait_s)
        elif clear_expected:
            # Unlike the gate, an occluded frame is still worth keeping here —
            # nothing is decided on it. The warning says what the picture is worth.
            print("[📷] ⚠️ Body did NOT clear the view — no 'paper' get-clear recording; the capture may be occluded")

        try:
            paths = _shoot(camera)
        finally:
            try:
                from utils import hooks as _kin_hooks

                if _kin_hooks.on_paper_check_done:
                    _kin_hooks.on_paper_check_done()
            except Exception:
                pass

        path = paths[-1] if paths else None
        try:
            from utils.state_manager import state_manager as _sm

            _sm.last_finished_drawing_image = path
            _sm.last_finished_drawing_ts = time.time() if path else 0.0
            prompt = _sm.current_drawing_prompt or getattr(_sm, "last_completed_drawing_prompt", None)
        except Exception:
            prompt = None

        log_json_entry(
            LogType.NEW_DRAWING,
            {
                "action": "finished_drawing_captured",
                "frames": len(paths),
                "images": paths,
                "image": path or "",
                "view_cleared": wait_s > 0 or not clear_expected,
                "clear_s": {"arms": clear_wait, "gantry": float(extra_clear_s or 0.0)},
                "duration": time.time() - started,
                "prompt": prompt,
            },
            print_message=(
                f"[📷] Finished drawing photographed: {os.path.basename(path)} ({time.time() - started:.1f}s)"
                if path
                else "[📷] Finished-drawing capture produced no frames"
            ),
        )
        return path

    except Exception as e:
        print(f"[📷] Finished-drawing capture failed: {e}")
        return None
