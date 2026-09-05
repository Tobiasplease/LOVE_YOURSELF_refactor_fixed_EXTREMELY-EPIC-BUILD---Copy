# presence_adjudicator.py

import json
import os
import threading
import time

from config.config import (
    ADJUDICATED_PERSON_TTL_S,
    ENTITY_VETO_TTL_S,
    MOOD_SNAPSHOT_FOLDER,
    PRESENCE_ADJUDICATE_MIN_INTERVAL_S,
    PRESENCE_ADJUDICATION_ENABLED,
)

# Structure only, never content (the artist's law): the question presupposes
# nothing about what might be seen — no candidate categories, no place-specific
# nouns. The machine describes freely; the code reads the ONTOLOGY of its own
# words (person-reference vs thing-reference) and nothing else.
_ADJ_SYSTEM = "You describe what you see in photos, tersely and literally."
_ADJ_PROMPT = "Look closely. What is this? One short line, plain words."

# Ontology lexicons — generic English, not studio knowledge. Decisive person
# reference; artificial/effigy markers; words too ambiguous to decide.
_PERSON_WORDS = {
    "man",
    "woman",
    "person",
    "people",
    "human",
    "guy",
    "child",
    "boy",
    "girl",
    "visitor",
    "stranger",
    "someone",
    "somebody",
    "lady",
    "gentleman",
    "worker",
    "artist",
    "kid",
    "adult",
    "teenager",
    "men",
    "women",
    "children",
}
_ARTIFICIAL_WORDS = {
    "mannequin",
    "doll",
    "robot",
    "robotic",
    "sculpture",
    "statue",
    "figurine",
    "foam",
    "styrofoam",
    "cast",
    "carved",
    "toy",
    "model",
    "mechanical",
    "dummy",
    "puppet",
    "prosthetic",
    "artificial",
    "fake",
    "replica",
    "wooden",
    "plastic",
    "cardboard",
    "plaster",
    "clay",
    "papier",
    "effigy",
    "arm",
    "machine",
}
_AMBIGUOUS_WORDS = {"figure", "figures", "silhouette", "body", "shape", "form", "torso"}


def parse_ontology(reply):
    """The machine's free description -> "person" | "thing" | None (can't say).
    Artificial markers beat person words ("a mannequin of a man" is a thing);
    a person word without them is a person; no person word is a thing unless
    the description leans on ambiguous body-shape words — then no verdict."""
    if not reply:
        return None
    words = {w.strip(".,;:!?'\"()").lower() for w in reply.split()}
    if words & _ARTIFICIAL_WORDS:
        return "thing"
    if words & _PERSON_WORDS:
        return "person"
    if words & _AMBIGUOUS_WORDS:
        return None
    return "thing" if words else None


class PresenceAdjudicatorThread(threading.Thread):
    """Adjudicated presence, phase 1 (Aug 18): YOLO proposes, the machine's own
    eye decides. A faceless person-candidate does NOT commit the presence
    belief; it queues one open VLM look at the crop. The reply's ontology
    (the machine's words, parsed — never a category we offered) either commits
    presence ("person") or records the thing in the entity ledger, whose place
    then vetoes candidates without re-asking. Face evidence never comes here:
    faces commit directly, and the veto can never fire against one."""

    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.lock = threading.Lock()
        self.ledger_path = os.path.join(MOOD_SNAPSHOT_FOLDER, "entity_ledger.json")
        self._entities = []  # {desc, verdict, box, pan, tilt, ts}
        self._pending = None  # {jpg, box(norm), pan, tilt, ts}
        self._person_until = 0.0
        self._last_call = 0.0
        self._load()

    # ---- consumed by the captioner every cycle ----
    def gate(self):
        """Verdict for the current faceless person-candidate:
        "person" (commit presence), "thing" (not company), None (pending —
        hold belief, a request has been queued)."""
        if not PRESENCE_ADJUDICATION_ENABLED:
            return "person"  # feature off: behave as before
        now = time.time()
        if now < self._person_until:
            return "person"
        nb = self._current_candidate_box()
        if nb is not None:
            pan, tilt = self._gaze_now()
            with self.lock:
                for e in self._entities:
                    if (
                        e["verdict"] == "thing"
                        and now - e["ts"] < ENTITY_VETO_TTL_S
                        and self._iou(nb, e["box"]) >= 0.5
                        and self._same_gaze(e, pan, tilt)  # Sep 5: a box only means something at the gaze it was seen from
                    ):
                        return "thing"
        self._request()
        return None

    @staticmethod
    def _gaze_now():
        try:
            from vision.gaze import physics_state

            return physics_state.pan, physics_state.tilt
        except Exception:
            return None, None

    @staticmethod
    def _same_gaze(entity, pan, tilt) -> bool:
        ep, et = entity.get("pan"), entity.get("tilt")
        if ep is None or et is None or pan is None or tilt is None:
            return True  # no gaze on record: box-only match, as before
        from config.config import ENTITY_VETO_GAZE_TOL_DEG

        return abs(float(ep) - float(pan)) <= ENTITY_VETO_GAZE_TOL_DEG and abs(float(et) - float(tilt)) <= ENTITY_VETO_GAZE_TOL_DEG

    def notify_presence_dropped(self):
        """Presence belief fell — the adjudicated-person grace ends with it.

        Sep 5 (three false arrivals in one morning: the black bundle on the top
        shelf twice — "a man lying down", "a person in a black shirt lying
        down" — and the mannequin head once): a person verdict that verified
        absence closes within PRESENCE_FALSE_ARRIVAL_WINDOW_S, while the same
        shape is STILL in the candidate box, was furniture. Retract it to a
        thing at that gaze + box so the veto fires next time instead of asking
        the same question of the same shelf. A real visitor who leaves is not
        in the box any more, so they are never retracted."""
        self._person_until = 0.0
        try:
            from config.config import PRESENCE_FALSE_ARRIVAL_WINDOW_S
        except Exception:
            PRESENCE_FALSE_ARRIVAL_WINDOW_S = 240.0
        now = time.time()
        nb = self._current_candidate_box()
        retracted = []
        with self.lock:
            for e in self._entities:
                if (
                    e.get("verdict") == "person"
                    and now - e.get("ts", 0) < PRESENCE_FALSE_ARRIVAL_WINDOW_S
                    and nb is not None
                    and self._iou(nb, e["box"]) >= 0.5
                ):
                    e["verdict"] = "thing"
                    e["retracted"] = True
                    e["desc"] = "(retracted: absence verified within minutes, shape still there) " + e.get("desc", "")
                    e["ts"] = now
                    retracted.append(e)
            if retracted:
                self._save_locked()
        for e in retracted:
            print(f"[Adjudicator] retracted person verdict → thing: {e['desc'][:90]}")
            try:
                from event_logging.event_logger import log_json_entry
                from event_logging.log_type import LogType

                log_json_entry(
                    LogType.DECISION,
                    {
                        "event": "presence_adjudication",
                        "verdict": "retracted",
                        "description": e["desc"][:160],
                        "pan": e.get("pan"),
                        "tilt": e.get("tilt"),
                    },
                    print_message=None,
                )
            except Exception:
                pass

    # ---- internals ----
    def _request(self):
        import cv2

        from perception.detection_memory import DetectionMemory

        now = time.time()
        with self.lock:
            if self._pending is not None:
                return
        crop = DetectionMemory.get_person_crop()
        nb = self._current_candidate_box()
        if crop is None or nb is None or crop.shape[0] < 40 or crop.shape[1] < 20:
            return
        ok, jpg = cv2.imencode(".jpg", crop)
        if not ok:
            return
        pan = tilt = None
        try:
            from vision.gaze import physics_state

            pan, tilt = physics_state.pan, physics_state.tilt
        except Exception:
            pass
        with self.lock:
            self._pending = {"jpg": jpg.tobytes(), "box": list(nb), "pan": pan, "tilt": tilt, "ts": now}

    def run(self):
        print("[Adjudicator] Presence adjudication thread started")
        while self.running:
            time.sleep(2.0)
            try:
                self._maybe_adjudicate()
            except Exception as e:
                print(f"[Adjudicator] Error: {e}")

    def stop(self):
        self.running = False

    def _maybe_adjudicate(self):
        if not PRESENCE_ADJUDICATION_ENABLED:
            return
        with self.lock:
            pending = self._pending
        if pending is None:
            return
        now = time.time()
        if now - self._last_call < PRESENCE_ADJUDICATE_MIN_INTERVAL_S:
            return
        from utils.state_manager import state_manager

        if getattr(state_manager, "is_generating_drawing", False):
            return  # never compete with the drawing pipeline for the model
        self._last_call = now
        from utils.inference import query_model

        reply = query_model(
            _ADJ_PROMPT,
            image=pending["jpg"],
            system_prompt=_ADJ_SYSTEM,
            timeout=90,
            prompt_type="presence_adjudication",
            options={"temperature": 0.3, "num_predict": 40},
        )
        verdict = parse_ontology(reply)
        desc = (reply or "").strip().splitlines()[0][:120] if reply else ""
        print(f"[Adjudicator] '{desc}' -> {verdict or 'no verdict'}")
        with self.lock:
            self._pending = None
            if verdict == "person":
                self._person_until = time.time() + ADJUDICATED_PERSON_TTL_S
            if verdict is not None and desc:
                self._entities.append(
                    {"desc": desc, "verdict": verdict, "box": pending["box"], "pan": pending["pan"], "tilt": pending["tilt"], "ts": time.time()}
                )
                self._entities = self._entities[-100:]
                self._save_locked()
        try:
            from event_logging.event_logger import log_json_entry
            from event_logging.log_type import LogType

            log_json_entry(
                LogType.DECISION,
                {"event": "presence_adjudication", "verdict": verdict or "none", "description": desc},
                print_message=None,
            )
        except Exception:
            pass

    @staticmethod
    def _current_candidate_box():
        from perception.detection_memory import DetectionMemory
        from utils.state_manager import state_manager

        bbox = DetectionMemory.get_person_bbox()
        if bbox is None:
            return None
        try:
            frame = state_manager.get_shared_frame(max_age=5.0)
            if frame is None:
                return None
            h, w = frame.shape[0], frame.shape[1]
            return (bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h)
        except Exception:
            return None

    @staticmethod
    def _iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (area_a + area_b - inter)

    def _load(self):
        try:
            if os.path.exists(self.ledger_path):
                with open(self.ledger_path) as f:
                    self._entities = json.load(f).get("entities", [])
        except Exception as e:
            print(f"[Adjudicator] Could not load ledger: {e}")

    def _save_locked(self):
        try:
            with open(self.ledger_path, "w") as f:
                json.dump({"entities": self._entities}, f, indent=2)
        except Exception as e:
            print(f"[Adjudicator] Could not save ledger: {e}")


presence_adjudicator = PresenceAdjudicatorThread()
