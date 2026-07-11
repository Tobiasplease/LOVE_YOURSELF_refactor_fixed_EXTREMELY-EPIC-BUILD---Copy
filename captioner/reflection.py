"""The Reflect loop — the minutes-to-hours timescale of thought.

North-star principle 3: three loops, three timescales. Captions NOTICE
(seconds). This loop REFLECTS: every ~20 minutes, when the scene is quiet,
the machine steps back from the caption stream and thinks at length about
one subject — the room, the visitor, the drawings, time passing, or itself.

Each reflection sees the thread of previous reflections (across sessions),
is stored in ChromaDB as a first-class memory (SemanticMemory.reflections),
and surfaces back into quiet captions by relevance
(prompts.get_reflection_echo_line).

The reflection uses the main model and happily occupies it for a minute —
quiet stretches are exactly when the machine is supposed to turn inward
(north-star principle 6). It is skipped while a drawing is in progress to
avoid VRAM contention with ComfyUI.

Replaced June 2026: model_wrapper.reason_about_caption (shallow per-caption
reasoning, output discarded each cycle) and the SemanticMemory per-concept
reflection worker.
"""

import threading
import time

from config import config
from config.config import LLM_TIMEOUT_REFLECTION, REFLECTION_LOOP_INTERVAL, REFLECTION_NUM_PREDICT, REFLECTION_TEMPERATURE
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType

_CHECK_EVERY = 30  # seconds between trigger checks
_QUIET_DEFERRAL_MAX = 2  # if the scene stays busy, reflect anyway after interval x this


class ReflectionLoop:
    def __init__(self, agent) -> None:
        self.agent = agent
        self.last_reflection_time = time.time()  # full interval before the first one
        self._subject_idx = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Resume rotation after the last stored subject — resetting to index 0
        # on every restart made "the room" dominate (6 of 11 reflections on a
        # restart-heavy day)
        try:
            from captioner.prompts import REFLECTION_SUBJECTS
            from captioner.semantic_memory import get_semantic_memory
            last = get_semantic_memory().get_recent_reflections(limit=1)
            if last:
                names = [s for s, _ in REFLECTION_SUBJECTS]
                subj = (last[0].get("subject") or "").strip()
                if subj in names:
                    self._subject_idx = (names.index(subj) + 1) % len(names)
        except Exception:
            pass

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="reflection-loop")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=_CHECK_EVERY):
            try:
                if self._should_reflect():
                    self._reflect()
            except Exception as e:
                print(f"[REFLECT] Cycle failed: {e}")
                self.last_reflection_time = time.time() - REFLECTION_LOOP_INTERVAL + 120  # retry in 2 min

    def _should_reflect(self) -> bool:
        # Clean room: the reflection loop is detox blind spot #2 — it consumes
        # stored prose (compressions, prior reflections, journal, drawings,
        # desire) AND writes long-form prose back, both contamination channels.
        # Pause it entirely under detox so the store stops filling with purple.
        from config.config import BASE_VOICE_DETOX
        if BASE_VOICE_DETOX:
            return False
        elapsed = time.time() - self.last_reflection_time
        if elapsed < REFLECTION_LOOP_INTERVAL:
            return False
        if not getattr(self.agent, "first_caption_done", False):
            return False
        try:
            if self.agent._is_currently_drawing():
                return False
        except Exception:
            pass
        # Wait for a quiet scene, but only up to a point — a busy day still
        # deserves reflection eventually
        scene_quiet = getattr(self.agent, "_last_scene_motion", None) is not True
        if not scene_quiet and elapsed < REFLECTION_LOOP_INTERVAL * _QUIET_DEFERRAL_MAX:
            return False
        return True

    def _next_subject(self) -> tuple:
        """Rotate through subjects, skipping ones with no material behind them."""
        from captioner.prompts import REFLECTION_SUBJECTS

        for _ in range(len(REFLECTION_SUBJECTS)):
            subject, question = REFLECTION_SUBJECTS[self._subject_idx]
            self._subject_idx = (self._subject_idx + 1) % len(REFLECTION_SUBJECTS)
            if subject == "the visitor" and not self._has_visitor_material():
                continue
            if subject == "the drawings" and not self._gather_drawings():
                continue
            return subject, question
        return REFLECTION_SUBJECTS[0]

    def _has_visitor_material(self) -> bool:
        try:
            from utils.episodic_log import episodic_log

            if episodic_log.get_last_event("person_arrived"):
                return True
        except Exception:
            pass
        try:
            from captioner.context_compression import context_compressor

            return len(context_compressor.core_facts.get("people", "").strip()) > 5
        except Exception:
            return False

    def _gather_drawings(self) -> str:
        """Framed sentences, executed-only — this used to inject a bare,
        unframed tag list ("chair, cables, desk") that read as scene noise."""
        try:
            from drawing.drawing_memory import get_drawing_memory

            dm = get_drawing_memory()
            parts = []
            last = dm.get_last_drawing_description(executed_only=True)
            if last:
                parts.append(f"The last drawing that actually reached paper: {last}.")
            summaries = dm.get_recent_drawings_summary(max_count=3, completed_only=True)
            if summaries:
                parts.append(f"Your recent drawings: {summaries}.")
            return " ".join(parts)
        except Exception:
            return ""

    def _gather_context(self) -> dict:
        """Rich context, all of it the machine's own past material."""
        from captioner.context_compression import context_compressor
        from captioner.semantic_memory import get_semantic_memory

        data = {}
        try:
            history = context_compressor.get_compression_history(max_entries=6)
            data["today"] = [h["understanding"] for h in history if h.get("understanding")]
        except Exception:
            pass
        try:
            data["reflections"] = get_semantic_memory().get_recent_reflections(limit=3)
        except Exception:
            pass
        try:
            data["journal"] = list(context_compressor.journal)[-3:]
        except Exception:
            pass
        data["drawings"] = self._gather_drawings()
        try:
            data["desire"] = context_compressor.get_current_desire()
            # Desire arc: a freshly spent want enters the reflection as a fact,
            # so the next want forms informed by the act instead of amnesiac.
            if not data["desire"]:
                spent = context_compressor.introspective_state.get("last_spent_desire") or {}
                if spent.get("desire") and time.time() - spent.get("spent", 0) < 48 * 3600:
                    data["desire_spent"] = spent
        except Exception:
            pass
        return data

    def _reflect(self) -> None:
        from captioner.prompts import build_reflection_loop_prompt, get_reflection_system_prompt
        from captioner.semantic_memory import get_semantic_memory
        from utils.inference import query_model

        data = self._gather_context()
        # Nothing lived yet — reflecting on an empty store just invents a
        # past, and a stored invention echoes back into captions forever.
        # Wait until at least some real material exists.
        if not (data.get("today") or data.get("journal") or data.get("reflections")):
            print("[REFLECT] Nothing lived yet — postponing reflection until there is real material.")
            self.last_reflection_time = time.time() - REFLECTION_LOOP_INTERVAL + 300  # retry in 5 min
            return

        subject, question = self._next_subject()
        prompt = build_reflection_loop_prompt(question, data)
        system_prompt = get_reflection_system_prompt()

        print(f"[REFLECT] Stepping back to think about {subject}...")
        response = query_model(
            prompt=prompt,
            model=config.MODEL_NAME,
            system_prompt=system_prompt,
            timeout=LLM_TIMEOUT_REFLECTION,
            options={
                "temperature": REFLECTION_TEMPERATURE,
                "top_p": 0.9,
                "num_predict": REFLECTION_NUM_PREDICT,
            },
            prompt_type="reflection",
        )

        # Storage-gate cleanup (north-star: guards live at storage, not the mouth):
        # markdown emphasis is a format artifact, and num_predict can cut the
        # final sentence mid-thought — trim to the last complete one.
        text = (response or "").strip().strip('"').strip()
        text = text.replace("**", "").replace("##", "")
        if text and text[-1] not in ".!?":
            cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if cut > 80:
                text = text[: cut + 1]
        if len(text) < 80:
            log_json_entry(
                LogType.REFLECTION,
                {"message": "Reflection too short, skipped", "action": "skip_short", "subject": subject, "length": len(text)},
                print_message=f"[REFLECT] Too short ({len(text)} chars), skipping",
            )
            self.last_reflection_time = time.time()
            return

        refl_id = None
        try:
            refl_id = get_semantic_memory().store_reflection_entry(text, subject)
        except Exception as e:
            print(f"[REFLECT] ChromaDB store failed: {e}")

        try:
            from utils.live_log import log_reflection

            log_reflection(subject, text)
        except Exception:
            pass

        self.last_reflection_time = time.time()

        # Identity engine (north-star Reflect → Become): distill this reflection
        # into a few PLAIN ledger takeaways — a self-trait (persona), a belief, a
        # want. This is where development now happens; the prose above is the
        # thinking, these are the product. Replaces the inert compression-thread
        # introspection/self-synthesis (retired June 28).
        try:
            from captioner.context_compression import context_compressor
            context_compressor.distill_reflection(text, subject, model=config.MODEL_NAME)
        except Exception as e:
            print(f"[REFLECT] distill failed: {e}")

        log_json_entry(
            LogType.REFLECTION,
            {"reflection": text, "subject": subject, "reflection_id": refl_id, "stored": refl_id is not None},
            print_message=f"[REFLECT] ({subject}) {text[:140]}...",
        )
