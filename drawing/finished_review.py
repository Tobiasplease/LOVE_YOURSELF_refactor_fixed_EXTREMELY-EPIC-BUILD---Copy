#!/usr/bin/env python3
"""
Drawing review (Sep 10 2026) — the machine looks at what it made.

The critique was removed Aug 5 ("not useful and underutilised"), with the terms
of its return written down at the time: it should judge THE PAPER, not the
ComfyUI image, and it should only ever run once. This is that pass. It sees the
sheet crop from the completion ritual (drawing/sheet_crop.py) and the intent in
the machine's own words, and it is asked what actually landed.

Nothing here tells it the drawing is good, bad, or even present. Plenty of these
come out barely legible — a faint scratch where a dense pool of shadow was
meant — and that is precisely the thing worth knowing, so the ask is framed to
let "hardly anything is there" be a true answer rather than a failure to
describe. Which is also why the crop it reads is neither straightened nor
contrast-stretched: correcting the image would tell it the pen did better than
it did.

Never raises, and never stores an empty answer. The old pass lost trust by
saving its own timeouts as the machine's reflection; the completion memory has
always survived without one.
"""

import os
import time
from typing import Optional

from config import config as _cfg
from event_logging.event_logger import LogType, log_json_entry


def _publish(text: str) -> None:
    """Hand the review to the drawing manager so the completion memory records
    it instead of running a second pass over the same drawing."""
    try:
        from utils.state_manager import state_manager as _sm

        captioner = getattr(_sm, "captioner", None)
        drawing = getattr(captioner, "drawing", None) if captioner else None
        if drawing is not None:
            drawing.last_reflection = text
    except Exception:
        pass


def review_finished_drawing(image_path: Optional[str] = None, intent: Optional[str] = None) -> Optional[str]:
    """Look at the finished sheet against what it was meant to be.

    Returns the machine's own words, or None when the pass is disabled, when
    there is no photograph, when no intent was recorded (nothing to judge
    against), or when the model gives nothing back.
    """
    if not bool(getattr(_cfg, "ENABLE_FINISHED_DRAWING_REVIEW", True)):
        return None

    started = time.time()
    try:
        from utils.state_manager import state_manager as _sm

        if not image_path:
            # The crop, not the full table view: it carries ~1024 image tokens
            # of paper instead of ~184 (see drawing/sheet_crop.py).
            image_path = getattr(_sm, "last_finished_drawing_sheet", None) or getattr(_sm, "last_finished_drawing_image", None)
        if not image_path or not os.path.exists(image_path):
            print("[🔍] Drawing review skipped — no photograph of the sheet")
            return None

        if not intent:
            intent = getattr(_sm, "current_drawing_prompt", None) or getattr(_sm, "last_completed_drawing_prompt", None)
        intent = (intent or "").strip()
        if not intent:
            print("[🔍] Drawing review skipped — no intent on record to judge against")
            return None

        from captioner.prompt_registry import P
        from utils.inference import query_model

        system_prompt = P("situation.reflexive") + P("review.frame")
        prompt = P("review.intent-wrap").format(intent=intent) + P("review.elicit")

        text = (
            query_model(
                prompt,
                image=image_path,
                system_prompt=system_prompt,
                timeout=int(getattr(_cfg, "FINISHED_REVIEW_TIMEOUT_S", 90)),
                prompt_type="drawing_review",
            )
            or ""
        ).strip()

        if not text:
            log_json_entry(
                LogType.NEW_DRAWING,
                {"action": "drawing_review_empty", "image": image_path, "duration": time.time() - started},
                print_message="[🔍] Drawing review came back empty — leaving the completion memory without one",
            )
            return None

        _publish(text)
        log_json_entry(
            LogType.NEW_DRAWING,
            {
                "action": "drawing_reviewed",
                "image": image_path,
                "intent": intent,
                "review": text,
                "duration": time.time() - started,
            },
            print_message=f"[🔍] Looked at what it made ({time.time() - started:.1f}s): {text[:160]}",
        )
        return text

    except Exception as e:
        print(f"[🔍] Drawing review failed: {e}")
        return None
