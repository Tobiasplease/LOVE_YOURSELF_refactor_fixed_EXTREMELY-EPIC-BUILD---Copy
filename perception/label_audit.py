# label_audit.py

import threading
import time

import cv2
import numpy as np

from config.config import (
    LABEL_AUDIT_ENABLED,
    LABEL_AUDIT_INTERVAL,
    LABEL_AUDIT_MARGIN,
    LABEL_AUDIT_MIN_HITS,
    LABEL_AUDIT_REAUDIT_HOURS,
    OPEN_VOCAB_MODEL_PATH,
)

_AUDIT_PROMPT = (
    "What is this object? Answer with 1 to 3 candidate names, one per line. "
    "Plain appearance language — material, colour, shape (like 'red foam finger' "
    "or 'white styrofoam head'). No sentences, no punctuation, just the names."
)
_AUDIT_SYSTEM = "You name objects in photos, tersely and literally."


class LabelAuditThread(threading.Thread):
    """The self-correction loop: the machine's richer eye (the VLM) audits its
    faster one (YOLO-World). A wrong label looks healthy from the inside —
    hits accumulate whether or not the name is right — so periodically a
    well-detected term's latest crop is shown to the VLM; if it disagrees, the
    incumbent and the VLM's candidates fight a CLIP head-to-head on the actual
    pixels, and a winning candidate is promoted into the vocabulary (origin
    "audit"). The incumbent is never removed — it keeps its true referent and
    loses only the stolen patch on future passes."""

    def __init__(self, detector):
        super().__init__(daemon=True)
        self.detector = detector
        self.running = True
        self._judge = None  # private YOLO-World for head-to-heads; never touches the live model's compiled vocab

    def run(self):
        print("[LabelAudit] Label audit thread started")
        while self.running:
            time.sleep(15.0)
            try:
                self._maybe_audit()
            except Exception as e:
                print(f"[LabelAudit] Audit error: {e}")

    def stop(self):
        self.running = False

    def _maybe_audit(self):
        if not LABEL_AUDIT_ENABLED:
            return
        from utils.state_manager import state_manager

        if getattr(state_manager, "is_generating_drawing", False):
            return  # never compete with the drawing pipeline for the VLM
        term = self._pick_target()
        if term is None:
            return
        crop_rec = self.detector.get_term_crop(term)
        from perception.spatial_registry import spatial_registry

        if crop_rec is None or time.time() - crop_rec["ts"] > 3600.0:
            spatial_registry.mark_audit(term, "skipped", "no fresh crop")
            return
        self._audit(term, crop_rec["jpg"])

    def _pick_target(self):
        """The well-detected term least recently audited."""
        from perception.spatial_registry import spatial_registry

        now = time.time()
        hits = self.detector.get_term_hit_counts()
        candidates = []
        for term, e in spatial_registry.get_entries().items():
            if hits.get(term, {}).get("count", 0) < LABEL_AUDIT_MIN_HITS:
                continue
            last_audit = e.get("last_audit", 0.0)
            if now - last_audit < LABEL_AUDIT_REAUDIT_HOURS * 3600.0:
                continue
            if now - getattr(self, "_last_audit_time", 0.0) < LABEL_AUDIT_INTERVAL:
                return None
            candidates.append((last_audit, term))
        return sorted(candidates)[0][1] if candidates else None

    def _audit(self, term, jpg):
        from event_logging.event_logger import log_json_entry
        from event_logging.log_type import LogType
        from perception.spatial_registry import spatial_registry
        from utils.inference import query_model

        self._last_audit_time = time.time()
        reply = query_model(
            _AUDIT_PROMPT,
            image=jpg,
            system_prompt=_AUDIT_SYSTEM,
            timeout=90,
            prompt_type="label_audit",
            options={"temperature": 0.3},
        )
        candidates = self._parse_candidates(reply, exclude=term)
        print(f"[LabelAudit] '{term}' → VLM says: {candidates or 'nothing usable'}")
        if not candidates:
            spatial_registry.mark_audit(term, "no_candidates", (reply or "")[:120])
            return

        if any(self._same_name(c, term) for c in candidates):
            spatial_registry.mark_audit(term, "confirmed")
            log_json_entry(LogType.VOCAB_PROMOTION, {"event": "audit", "term": term, "verdict": "confirmed"})
            return

        scores = self._head_to_head(jpg, [term] + candidates)
        incumbent = scores.get(term, 0.0)
        challenger = max(candidates, key=lambda c: scores.get(c, 0.0))
        detail = ", ".join(f"{t}:{scores.get(t, 0.0):.2f}" for t in [term] + candidates)
        if scores.get(challenger, 0.0) > incumbent + LABEL_AUDIT_MARGIN:
            from perception.vocab_promotion import vocab_promoter

            promoted = vocab_promoter.promote_term(challenger, origin="audit", note=f"beat '{term}' on its own crop ({detail})")
            verdict = "relabelled" if promoted else "challenger_already_known"
            spatial_registry.mark_audit(term, verdict, f"-> {challenger} ({detail})")
            log_json_entry(
                LogType.VOCAB_PROMOTION,
                {"event": "audit", "term": term, "verdict": verdict, "challenger": challenger, "scores": detail},
                print_message=f"[LabelAudit] RELABEL: '{term}' patch likely '{challenger}' ({detail})",
            )
        else:
            spatial_registry.mark_audit(term, "held", detail)
            log_json_entry(LogType.VOCAB_PROMOTION, {"event": "audit", "term": term, "verdict": "held", "scores": detail})

    def _parse_candidates(self, reply, exclude):
        if not reply:
            return []
        out = []
        for line in reply.strip().splitlines()[:5]:
            name = " ".join(w for w in line.strip().lower().strip(".-•*:'\"`,").split() if w.isalpha())
            if 2 < len(name) <= 40 and 1 <= len(name.split()) <= 4 and name != exclude and name not in out:
                out.append(name)
        return out[:3]

    @staticmethod
    def _same_name(a, b):
        return f" {a} " in f" {b} " or f" {b} " in f" {a} "

    def _head_to_head(self, jpg, terms):
        """Compile only the contestants, score them on the crop itself."""
        from perception.open_vocab_detector import _cpu_without_hiding_the_gpu

        if self._judge is None:
            from ultralytics import YOLOWorld

            with _cpu_without_hiding_the_gpu():
                self._judge = YOLOWorld(OPEN_VOCAB_MODEL_PATH)
        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        with _cpu_without_hiding_the_gpu():
            self._judge.set_classes(terms)
            results = self._judge.predict(img, conf=0.01, iou=0.5, device="cpu", verbose=False)[0]
        scores = {}
        for box in results.boxes:
            t = terms[int(box.cls)]
            scores[t] = max(scores.get(t, 0.0), float(box.conf))
        return scores
