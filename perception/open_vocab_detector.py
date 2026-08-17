# open_vocab_detector.py

import contextlib
import os
import threading
import time

from config.config import (
    OPEN_VOCAB_CONF_FLOOR,
    OPEN_VOCAB_DRAW_MOVE_TOL,
    OPEN_VOCAB_INTERVAL,
    OPEN_VOCAB_MAX_BOXES,
    OPEN_VOCAB_MAX_WAIT,
    OPEN_VOCAB_MODEL_PATH,
    OPEN_VOCAB_PERSON_OVERLAP_MAX,
    OPEN_VOCAB_SETTLE_VELOCITY,
    OPEN_VOCAB_TERM_FLOORS,
    OPEN_VOCAB_VOCABULARY,
)
from perception.detection_memory import DetectionMemory


@contextlib.contextmanager
def _cpu_without_hiding_the_gpu():
    """Keep this detector on the CPU without blinding the rest of the process.

    Ultralytics implements device="cpu" by setting a PROCESS-GLOBAL
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" (utils/torch_utils.py:185) and
    never putting it back. Every subprocess spawned afterwards inherits it — so
    when the wedge watchdog restarted llama-server mid-run, the new server hit
    "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is
    detected" and loaded the 27B entirely into system RAM. It still answered
    /health, so it looked alive while generating at ~2 tok/s: every call past
    that point timed out, and only a restart that beat the first detection pass
    ever recovered. Aug 9: cost a whole run before the server's own log named it.

    Restoring the variable is enough — torch has already resolved its device by
    the time we return, so the CPU pin still holds for this model.
    """
    prior = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior


class OpenVocabDetectorThread(threading.Thread):
    """Zero-shot object naming (YOLO-World, CPU-only). Runs on a cadence, only
    when the gaze is settled; each result set carries the pan/tilt it was
    captured at. Vocabulary is hot-swappable for promotion events (Phase 2).
    Phase 0 feasibility report: debug/phase0_report/."""

    def __init__(self):
        super().__init__(daemon=True)
        self.model = None  # loaded lazily in run(): vocab compile takes seconds
        self.running = True
        self.shared_frame = None
        self.lock = threading.Lock()
        self._vocabulary = list(OPEN_VOCAB_VOCABULARY)
        self._pending_vocabulary = None
        self._detections = []
        self._detection_time = 0.0
        self._detection_pan = None
        self._detection_tilt = None
        self._last_pass_time = time.time()
        self._prev_pass_terms = set()
        self._term_hits = {}  # term -> {"count": int, "last": ts}; cumulative, feeds ghost detection (Phase 2)
        self._term_crops = {}  # term -> {"jpg": bytes, "ts": ts, "conf": float}; latest settled crop, feeds the label audit

    def set_frame(self, frame):
        with self.lock:
            self.shared_frame = frame.copy()

    def set_vocabulary(self, terms):
        """Hot-swap: takes effect (with recompile) on the next pass."""
        with self.lock:
            self._pending_vocabulary = [t.strip() for t in terms if t.strip()]

    def get_detections(self, max_age=None):
        """Latest pass as [{term, conf, box, pan, tilt, time}]."""
        with self.lock:
            if max_age is not None and time.time() - self._detection_time > max_age:
                return []
            return [dict(d) for d in self._detections]

    def get_term_hit_counts(self):
        with self.lock:
            return {t: dict(s) for t, s in self._term_hits.items()}

    def get_term_crop(self, term):
        """Latest settled crop of a term as {"jpg", "ts", "conf"}, or None."""
        with self.lock:
            c = self._term_crops.get(term)
            return dict(c) if c else None

    def get_detections_for_drawing(self):
        """Boxes still valid on screen, capped at OPEN_VOCAB_MAX_BOXES: best box
        per term, new arrivals ranked before confident regulars. Tolerance must
        sit above tremor jitter or the boxes flash; a real look-away hides them
        (a box drawn after a turn points at nothing)."""
        from vision.gaze import physics_state

        with self.lock:
            dets = [dict(d) for d in self._detections]
            det_time, det_pan, det_tilt = self._detection_time, self._detection_pan, self._detection_tilt
        if not dets or time.time() - det_time > OPEN_VOCAB_INTERVAL * 2:
            return []
        if det_pan is not None and abs(physics_state.pan - det_pan) + abs(physics_state.tilt - det_tilt) > OPEN_VOCAB_DRAW_MOVE_TOL:
            return []
        best_per_term = {}
        for d in dets:
            if d["conf"] > best_per_term.get(d["term"], {"conf": 0})["conf"]:
                best_per_term[d["term"]] = d
        ranked = sorted(best_per_term.values(), key=lambda d: (not d.get("novel", False), -d["conf"]))
        return ranked[:OPEN_VOCAB_MAX_BOXES]

    def _effective_floors(self):
        """Static per-term floors merged with audit discipline (a term that
        lost its VLM audit must clear a higher bar until re-confirmed)."""
        floors = dict(OPEN_VOCAB_TERM_FLOORS)
        try:
            from perception.spatial_registry import spatial_registry

            for t, f in spatial_registry.get_audit_floors().items():
                floors[t] = max(floors.get(t, OPEN_VOCAB_CONF_FLOOR), f)
        except Exception:
            pass
        return floors

    def detect_once(self, frame):
        """One pass on one frame; person-overlapping boxes suppressed. Public so
        debug scripts can run it without the thread."""
        if self.model is None:
            from ultralytics import YOLOWorld

            with _cpu_without_hiding_the_gpu():
                self.model = YOLOWorld(OPEN_VOCAB_MODEL_PATH)
                self.model.set_classes(self._vocabulary)  # CLIP text pass also routes through select_device
            print(f"[OpenVocab] Model loaded, vocabulary compiled ({len(self._vocabulary)} terms)")

        floors = self._effective_floors()
        predict_floor = min([OPEN_VOCAB_CONF_FLOOR, *OPEN_VOCAB_TERM_FLOORS.values()])
        with _cpu_without_hiding_the_gpu():
            results = self.model.predict(frame, conf=predict_floor, iou=0.5, device="cpu", verbose=False)[0]
        person_bbox = DetectionMemory.get_person_bbox()
        detections = []
        for box in results.boxes:
            term = self._vocabulary[int(box.cls)]
            conf = float(box.conf)
            if conf < floors.get(term, OPEN_VOCAB_CONF_FLOOR):
                continue
            xyxy = tuple(int(v) for v in box.xyxy[0])
            if person_bbox and self._overlap_fraction(xyxy, person_bbox) > OPEN_VOCAB_PERSON_OVERLAP_MAX:
                continue  # live humans belong to the person tracker, not the object map
            detections.append({"term": term, "conf": conf, "box": xyxy})
        return self._dedup_by_location(detections)

    @staticmethod
    def _dedup_by_location(detections):
        """One patch of the world gets one name: strongest box wins. Same term:
        nested/overlapping boxes are duplicates. Different terms: only boxes of
        similar extent compete (laptop vs monitor) — a small object in front of
        a big one (lamp before shelf) is two objects, not a duplicate."""
        kept = []
        for det in sorted(detections, key=lambda d: -d["conf"]):
            x1, y1, x2, y2 = det["box"]
            area = max(1, (x2 - x1) * (y2 - y1))
            duplicate = False
            for k in kept:
                kx1, ky1, kx2, ky2 = k["box"]
                inter = max(0, min(x2, kx2) - max(x1, kx1)) * max(0, min(y2, ky2) - max(y1, ky1))
                k_area = max(1, (kx2 - kx1) * (ky2 - ky1))
                same_term = det["term"] == k["term"]
                if same_term and inter / min(area, k_area) > 0.6:
                    duplicate = True
                    break
                if not same_term and inter / (area + k_area - inter) > 0.55:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    @staticmethod
    def _drop_self_patches(frame, detections, pan, tilt):
        """The machine's own arms are not world: detections matching the body
        schema (place + appearance) are dropped before storage, so they never
        reach the overlay, the registry, or the audit.

        Consistency rework (Aug 17): the old top-4-by-confidence budget let the
        arm slip through whenever its patch ranked 5th in a busy pass — the
        rooster label flickered over the arms all night. Now (1) a box
        overlapping a place confirmed self in the last BODY_SELF_REGION_TTL s
        at this pose drops with NO embed (arms don't teleport); (2) the embed
        budget is spent envelope-first — boxes in places the body can occupy
        at this pose are exactly the ones worth checking — then by confidence.

        Proprioceptive drop (same day, the 0.74 rooster-on-the-hand): while
        the CNC is EXECUTING, the arm is certainly over the paper — the same
        evidence the harvest trusts. Any box in the body envelope at this pose
        drops on PLACE ALONE, no appearance vote: a confident wrong label
        (hand+pencil reads as beak to CLIP) beats every floor, and appearance
        is exactly the witness being fooled."""
        from config.config import BODY_SELF_CHECK_BUDGET, BODY_SELF_FILTER_DETECTIONS

        if not BODY_SELF_FILTER_DETECTIONS or not detections:
            return detections
        from perception.body_schema import body_schema

        if body_schema.gallery_size() == 0:
            return detections
        drawing_now = False
        try:
            from utils.state_manager import state_manager

            drawing_now = getattr(state_manager, "current_drawing_phase", None) == "executing"
        except Exception:
            pass
        h, w = frame.shape[0], frame.shape[1]

        def norm_box(d):
            x1, y1, x2, y2 = d["box"]
            return (x1 / w, y1 / h, x2 / w, y2 / h)

        kept = []
        budget = BODY_SELF_CHECK_BUDGET
        ordered = sorted(detections, key=lambda d: (not body_schema.in_pose_envelope(norm_box(d), pan, tilt), -d["conf"]))
        for d in ordered:
            nb = norm_box(d)
            if drawing_now and body_schema.in_pose_envelope(nb, pan, tilt):
                # place alone suffices while the arm is certainly out; no
                # sticky note — proprioceptive drops are free every pass, and
                # seeding regions from them once blanketed the frame
                print(f"[OpenVocab] Dropped self-patch labelled '{d['term']}' (drawing, body envelope)")
                continue
            if body_schema.in_recent_self_region(nb, pan, tilt):
                print(f"[OpenVocab] Dropped self-patch labelled '{d['term']}' (recent self region)")
                continue
            if budget > 0:
                budget -= 1
                x1, y1, x2, y2 = d["box"]
                is_self, sim = body_schema.is_self(frame[max(0, y1) : y2, max(0, x1) : x2], nb, pan, tilt, enrich=True)
                if is_self:
                    print(f"[OpenVocab] Dropped self-patch labelled '{d['term']}' (sim {sim:.2f})")
                    continue
            kept.append(d)
        return kept

    @staticmethod
    def _overlap_fraction(box, person_box):
        x1, y1, x2, y2 = box
        px1, py1, px2, py2 = person_box
        ix = max(0, min(x2, px2) - max(x1, px1))
        iy = max(0, min(y2, py2) - max(y1, py1))
        return (ix * iy) / max(1, (x2 - x1) * (y2 - y1))

    def _is_settled(self):
        from vision.gaze import physics_state

        return abs(physics_state.pan_velocity) + abs(physics_state.tilt_velocity) < OPEN_VOCAB_SETTLE_VELOCITY

    def run(self):
        from vision.gaze import physics_state

        print("[OpenVocab] Open-vocabulary detection thread started (CPU).")
        while self.running:
            cycle_start = time.time()
            with self.lock:
                if self._pending_vocabulary:
                    self._vocabulary = self._pending_vocabulary
                    self._pending_vocabulary = None
                    if self.model is not None:
                        self.model.set_classes(self._vocabulary)
                        print(f"[OpenVocab] Vocabulary recompiled ({len(self._vocabulary)} terms)")
                frame = self.shared_frame.copy() if self.shared_frame is not None else None

            calm = self._is_settled()
            starving = time.time() - self._last_pass_time > OPEN_VOCAB_MAX_WAIT
            if frame is None or (not calm and not starving):
                time.sleep(0.25)
                continue

            pan, tilt = physics_state.pan, physics_state.tilt
            self._last_pass_time = time.time()
            try:
                detections = self.detect_once(frame)
                detections = self._drop_self_patches(frame, detections, pan, tilt)
            except Exception as e:
                print(f"[OpenVocab] Detection error: {e}")
                time.sleep(OPEN_VOCAB_INTERVAL)
                continue

            now = time.time()
            terms_now = {d["term"] for d in detections}
            with self.lock:
                self._detections = [
                    {**d, "pan": pan, "tilt": tilt, "time": now, "settled": calm, "novel": d["term"] not in self._prev_pass_terms} for d in detections
                ]
                self._detection_time = now
                self._detection_pan, self._detection_tilt = pan, tilt
                self._prev_pass_terms = terms_now
                for t in terms_now:
                    stats = self._term_hits.setdefault(t, {"count": 0, "last": None})
                    stats["count"] += 1
                    stats["last"] = now
                if calm:
                    import cv2

                    for d in detections:
                        x1, y1, x2, y2 = d["box"]
                        crop = frame[max(0, y1) : y2, max(0, x1) : x2]
                        if crop.size:
                            ok, jpg = cv2.imencode(".jpg", crop)
                            if ok:
                                self._term_crops[d["term"]] = {"jpg": jpg.tobytes(), "ts": now, "conf": d["conf"]}
                current = list(self._detections)
            try:
                from perception.spatial_registry import spatial_registry

                spatial_registry.update_from_detections(current, frame.shape)
            except Exception as e:
                print(f"[OpenVocab] Registry update failed: {e}")
            try:
                # Discernment: if this settled pass happened while the gaze was
                # deliberately parked on a remembered object, the pass IS the
                # verification — seen refreshes the memory, not-seen decays it
                # (2nd consecutive miss becomes an absence event for the mind,
                # 4th forgets the anchor). Only judged when the camera is
                # actually pointed near the anchor — a clamped or drifting
                # glance must not count as evidence of absence.
                from config.config import SPATIAL_REGISTRY_HFOV, SPATIAL_REGISTRY_VFOV
                from vision.gaze import get_glance_info

                gi = get_glance_info()
                if calm and gi and gi["kind"] == "revisit":
                    anchor = spatial_registry.get_anchor(gi["label"])
                    if anchor and abs(pan - anchor[0]) < SPATIAL_REGISTRY_HFOV / 3 and abs(tilt - anchor[1]) < SPATIAL_REGISTRY_VFOV / 3:
                        outcome = spatial_registry.note_glance_result(gi["label"], gi["label"] in terms_now)
                        if outcome and outcome != "seen":
                            print(f"[OpenVocab] Glance check '{gi['label']}': {outcome}")
            except Exception:
                pass
            try:
                # Keep the person-is-self verdict warm for the main loop (its
                # cached read must never trigger an embed); 5s-cached inside.
                if DetectionMemory.get_person_bbox() is not None:
                    from perception.body_schema import body_schema

                    body_schema.is_self_current_person()
            except Exception:
                pass

            while self.running and (time.time() - cycle_start) < OPEN_VOCAB_INTERVAL:
                time.sleep(0.1)

    def stop(self):
        self.running = False
        with self.lock:
            self.shared_frame = None
