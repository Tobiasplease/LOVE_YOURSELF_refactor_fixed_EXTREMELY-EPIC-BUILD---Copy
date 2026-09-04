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

import re
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
        self._drawings_cache: tuple | None = None
        # Resume rotation after the last stored subject — resetting to index 0
        # on every restart made "the room" dominate (6 of 11 reflections on a
        # restart-heavy day)
        try:
            from captioner.prompts import get_reflection_subjects
            from captioner.semantic_memory import get_semantic_memory

            last = get_semantic_memory().get_recent_reflections(limit=1)
            if last:
                names = [s for s, _ in get_reflection_subjects()]
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
        from captioner.prompts import get_reflection_subjects

        subjects = get_reflection_subjects()
        for _ in range(len(subjects)):
            subject, question = subjects[self._subject_idx]
            self._subject_idx = (self._subject_idx + 1) % len(subjects)
            if subject == "the visitor" and not self._has_visitor_material():
                continue
            if subject == "the drawings" and not self._gather_drawings():
                continue
            return subject, question
        return subjects[0]

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
        unframed tag list ("chair, cables, desk") that read as scene noise.

        Cached for the cycle: _next_subject calls this as a material gate and
        the drawings diet calls it again moments later.
        """
        now = time.time()
        if self._drawings_cache and now - self._drawings_cache[0] < 60:
            return self._drawings_cache[1]
        result = self._build_drawings_line()
        self._drawings_cache = (now, result)
        return result

    def _build_drawings_line(self) -> str:
        try:
            from drawing.drawing_memory import get_drawing_memory

            dm = get_drawing_memory()
            parts = []
            # Vision offline (July 30): an evening of not-being-able-to-draw is
            # identity-pertinent fact — the reflection should know it, dated.
            try:
                from utils.drawing_state import DrawingState

                hours = DrawingState.vision_offline_hours()
                if hours is not None:
                    if hours < 1:
                        parts.append("Drawing is not possible right now — you reached for one and nothing could form.")
                    else:
                        parts.append(
                            f"No drawing has been able to form for over {int(hours)} hour{'s' if int(hours) != 1 else ''} — the picturing part is dark."
                        )
            except Exception:
                pass
            last = dm.get_last_drawing_description(executed_only=True)
            if last:
                parts.append(f"The last drawing that actually reached paper: {last}.")
            summaries = dm.get_recent_drawings_summary(max_count=3, completed_only=True)
            if summaries:
                parts.append(f"Your recent drawings: {summaries}.")
            return " ".join(parts)
        except Exception:
            return ""

    def _gather_spine(self) -> dict:
        """The generic material every gather starts from, and the liveness
        check. What actually REACHES a prompt is chosen per organ below."""
        from captioner.context_compression import context_compressor
        from captioner.semantic_memory import get_semantic_memory

        spine = {}
        try:
            # THE DREAM'S RAW MATERIAL (July 12): the machine's actual thoughts
            # from the last stretch, verbatim. Every prior input to reflection
            # was a summary of a summary — the loop could never notice what
            # actually happened in its own head (e.g. an hour of questions
            # addressed to a visitor that nothing ever answered).
            cutoff = time.time() - 75 * 60
            hour = [e["text"][:220] for e in context_compressor.hour_log if e.get("timestamp", 0) > cutoff]
            spine["hour"] = hour[-80:]
        except Exception:
            pass
        try:
            history = context_compressor.get_compression_history(max_entries=6)
            spine["today"] = [h["understanding"] for h in history if h.get("understanding")]
        except Exception:
            pass
        try:
            spine["journal"] = list(context_compressor.journal)
        except Exception:
            pass
        try:
            spine["lived"] = bool(get_semantic_memory().get_recent_reflections(limit=1))
        except Exception:
            pass
        return spine

    @staticmethod
    def _has_lived(spine: dict) -> bool:
        """Is there any real material at all? Deliberately checks the WHOLE
        spine, not the organ bundle — a thin organ (a day with no visitor)
        must not read as an unlived day and postpone the loop forever."""
        return bool(spine.get("hour") or spine.get("today") or spine.get("journal") or spine.get("lived"))

    def _gather_context(self, subject: str, spine: dict) -> dict:
        """The organ's diet (July 31). Each subject sees a DIFFERENT bundle —
        that is the whole point: the variety has to live in the DATA, not in
        the question. Five lenses over one identical bundle collapsed into one
        thought every time; the lens stays soft, the material behind it is what
        changes. Specialize data and consequence, never the voice.

        Shared spine: the dream (the raw record of recent thought) — no subject
        should be blind to what just went through its own head.
        """
        from captioner.semantic_memory import get_semantic_memory

        data = {"hour": spine.get("hour") or []}
        try:
            # This organ's OWN thread, not the last three of any subject.
            data["reflections"] = get_semantic_memory().get_recent_reflections(limit=3, subject=subject)
        except Exception:
            pass
        try:
            # Reveries ride the spine (Sep 3 evening, re-entry round): no
            # subject is blind to what it has been imagining — rendered by the
            # prompt builder as inventions, never observations. This is the
            # loom: the distiller downstream may keep a thread of it as lore.
            from config.config import LORE_ENABLED

            if LORE_ENABLED:
                from utils.lore_ledger import lore_ledger

                data["reveries"] = lore_ledger.recent_reveries(5)
        except Exception:
            pass

        diet = {
            "the room": self._diet_room,
            "the visitor": self._diet_visitor,
            "the drawings": self._diet_drawings,
            "time passing": self._diet_time,
            "yourself": self._diet_self,
        }.get(subject)
        if diet:
            try:
                diet(data, spine)
            except Exception as e:
                print(f"[REFLECT] Context for '{subject}' came up incomplete: {e}")
        return data

    def _diet_room(self, data: dict, spine: dict) -> None:
        """The place itself: what is in it, and what the compressions made of
        it. Compressions are scene-heavy, so they live here.

        Deliberately NO events ledger — it reads as people-and-happenings, and
        four organs sharing one block is how this collapsed the first time.
        """
        from captioner.semantic_memory import get_semantic_memory

        data["today"] = spine.get("today") or []
        try:
            # The concepts ledger, not core_facts['place'] — that prose is
            # retired from surfacing (see get_core_facts_string).
            inventory = get_semantic_memory().get_place_inventory(max_items=8, min_times_seen=3)
            if inventory:
                data["place_inventory"] = inventory
        except Exception:
            pass

    def _diet_visitor(self, data: dict, spine: dict) -> None:
        """The people: presence spans from the episodic log — how long they
        stayed, how long since — plus what has been learned about them."""
        from captioner.context_compression import context_compressor

        try:
            from utils.episodic_log import get_episodic_log

            log = get_episodic_log()
            pairs = log.get_pairs_in_window("person_arrived", "person_left", window_seconds=72 * 3600)
            spans = []
            for p in pairs[-6:]:
                start = p["start"].get("timestamp", 0)
                minutes = max(1, int(p.get("duration_seconds", 0) / 60))
                if p.get("end"):
                    spans.append(f"someone was here about {minutes} minutes, starting {log.format_ago(time.time() - start)}")
                elif start < float(getattr(self.agent, "true_session_start", 0.0) or 0.0):
                    # Sep 5: an arrival left open by an earlier run is not "never saw
                    # them go" — the machine was off. Say only what is known.
                    spans.append(f"someone came by {log.format_ago(time.time() - start)}")
                else:
                    spans.append(f"someone arrived {log.format_ago(time.time() - start)} and you never saw them go")
            if spans:
                data["visitor_spans"] = spans
        except Exception:
            pass
        try:
            people = (context_compressor.core_facts.get("people") or "").strip()
            if people:
                data["people_note"] = people
        except Exception:
            pass
        try:
            data["events"] = context_compressor.events[-5:]
        except Exception:
            pass

    def _diet_drawings(self, data: dict, spine: dict) -> None:
        """The fat one. "The ones you've made and the ones you've wanted to
        make" is literally the executed sequence plus the desire history — the
        organ used to get a three-line scrap and then wonder why it had nothing
        to say about drawing."""
        from captioner.context_compression import context_compressor

        data["drawings"] = self._gather_drawings()
        try:
            from drawing.drawing_memory import get_drawing_memory

            dm = get_drawing_memory()
            data["executed"] = dm.get_executed_sequence(max_count=8)
            # An LLM call, and a channel that has run purple before — trimmed
            # to its first two sentences and framed as a past look, never as a
            # present reading. Returns "" under two executed drawings.
            arc = (dm.get_artistic_arc() or "").strip().replace("**", "")
            if arc:
                sentences = [s for s in re.split(r"(?<=[.!?])\s+", arc) if s.strip()]
                data["arc"] = " ".join(sentences[:2])[:320]
        except Exception:
            pass
        try:
            # The wants, spent and unspent — the other half of "the ones you've
            # wanted to make". Current desire first, then the trail behind it.
            data["desire"] = context_compressor.get_current_desire()
            if not data["desire"]:
                spent = context_compressor.introspective_state.get("last_spent_desire") or {}
                if spent.get("desire") and time.time() - spent.get("spent", 0) < 48 * 3600:
                    data["desire_spent"] = spent
            history = (context_compressor.get_full_identity() or {}).get("desire_history") or []
            current = (data.get("desire") or "").strip().lower()
            past = [h for h in history[-6:] if (h.get("desire") or "").strip().lower() != current]
            if past:
                data["desire_history"] = past
        except Exception:
            pass

    @staticmethod
    def _add_felt_arc(data: dict) -> None:
        """The day's felt trajectory for the yourself/time organs (Sep 4) —
        the identity engine had distilled thousands of reflections without
        ever reading how a day felt. Up to 8 phrase-bearing reads, evenly
        spread; rendered by the prompt builder as the machine's own words."""
        try:
            from config.config import FELT_ARC_ENABLED

            if not FELT_ARC_ENABLED:
                return
            from captioner.context_compression import context_compressor

            hist = [h for h in (getattr(context_compressor, "felt_history", None) or []) if h.get("felt")]
            if len(hist) < 2:
                return
            step = max(1, len(hist) // 8)
            sampled = hist[::step][-8:]
            data["felt_arc"] = [{"ts": h["timestamp"], "felt": h["felt"]} for h in sampled]
        except Exception:
            pass

    def _diet_time(self, data: dict, spine: dict) -> None:
        """The long clock: the diary as chronology, how long this session has
        run, and how long the durable facts have held."""
        from captioner.context_compression import context_compressor

        data["journal"] = (spine.get("journal") or [])[-6:]
        try:
            data["session"] = context_compressor.get_current_session_info()
        except Exception:
            pass
        self._add_felt_arc(data)
        try:
            from captioner.durable_ledger import get_durable_ledger

            spans = []
            for f in get_durable_ledger().all_facts():
                if f.get("cls") not in ("permanent", "stable"):
                    continue
                days = len(f.get("days") or [])
                if days >= 2:
                    spans.append({"fact": f.get("fact", ""), "days": days, "established": f.get("established", 0)})
            spans.sort(key=lambda s: -s["days"])
            if spans:
                data["ledger_spans"] = spans[:5]
        except Exception:
            pass
        try:
            data["events"] = context_compressor.events[-5:]
        except Exception:
            pass

    def _diet_self(self, data: dict, spine: dict) -> None:
        """Already effectively this organ's diet via distill — made explicit.
        The identity slots stay HERE now rather than riding every subject's
        system prompt. self_notes is this organ's own event ledger, so the
        general one is left to the room-and-people subjects."""
        from captioner.context_compression import context_compressor

        try:
            data["self_notes"] = context_compressor.self_notes[-4:]
        except Exception:
            pass
        self._add_felt_arc(data)
        try:
            data["identity"] = {
                "persona": (context_compressor.core_facts.get("self") or "").strip(),
                "belief": (context_compressor.get_current_belief() or "").strip(),
                "desire": (context_compressor.get_current_desire() or "").strip(),
                "desire_since": context_compressor.introspective_state.get("desire_since", 0.0),
            }
        except Exception:
            pass

    def _reflect(self) -> None:
        from captioner.prompts import build_reflection_loop_prompt, get_reflection_system_prompt
        from captioner.semantic_memory import get_semantic_memory
        from utils.inference import query_model

        # Spine first, then the subject, THEN the subject's own material — the
        # order matters now that the bundle depends on which organ is up. The
        # liveness gate runs before _next_subject so a postponed cycle doesn't
        # burn a rotation slot.
        spine = self._gather_spine()
        # Nothing lived yet — reflecting on an empty store just invents a
        # past, and a stored invention echoes back into captions forever.
        # Wait until at least some real material exists.
        if not self._has_lived(spine):
            print("[REFLECT] Nothing lived yet — postponing reflection until there is real material.")
            self.last_reflection_time = time.time() - REFLECTION_LOOP_INTERVAL + 300  # retry in 5 min
            return

        self._drawings_cache = None  # fresh gather per cycle
        subject, question = self._next_subject()
        data = self._gather_context(subject, spine)
        prompt = build_reflection_loop_prompt(question, data)
        system_prompt = get_reflection_system_prompt(subject)

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
        # Failed queries return their error as TEXT ("[WARNING] llama-server
        # API failed: ... Read timed out.") — 100+ chars ending in a period,
        # which sailed past the too-short gate and entered ChromaDB as a
        # first-class memory three times during the July 30 wedge outage.
        if text.startswith("[WARNING]") or "llama-server API failed" in text:
            log_json_entry(
                LogType.REFLECTION,
                {"message": "Reflection was an API error string, skipped", "action": "skip_error_string", "subject": subject},
                print_message="[REFLECT] Query failed — no reflection this cycle",
            )
            self.last_reflection_time = time.time()
            return
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

            kernel = context_compressor.distill_reflection(text, subject, model=config.MODEL_NAME)
            # Echo kernel (July 30): ride the stored entry so the echo line can
            # surface a re-thinkable clause instead of a bare subject label.
            if kernel and refl_id:
                get_semantic_memory().set_reflection_kernel(refl_id, kernel)
            # KERNEL INTO THE STREAM (Sep 2, artist's call: "it's the
            # interaction between the exterior and the interior that is the
            # whole point"). The reflection's load-bearing sentence enters the
            # visible train of thought as the machine's own turn — it WAS just
            # thought, this is not memory posing as present. The stream then
            # carries interior register alongside room-talk, and the same law
            # that locked in observational genre starts teaching wondering
            # from within the machine's own voice. Naturally dosed (one per
            # reflection, ~20+ min apart); the standard admission gate holds
            # register; length bounds mirror the blink-seed check.
            try:
                if kernel and 20 < len(kernel) < 220 and self.agent._stream_admissible(kernel):
                    self.agent._stream_push(kernel.strip())
                    log_json_entry(
                        LogType.DEBUG,
                        {"message": "Reflection kernel admitted to stream", "action": "kernel_to_stream", "kernel": kernel[:160]},
                        print_message=f"[🪞→] kernel joins the stream: {kernel[:80]}",
                    )
            except Exception:
                pass
        except Exception as e:
            print(f"[REFLECT] distill failed: {e}")

        log_json_entry(
            LogType.REFLECTION,
            {"reflection": text, "subject": subject, "reflection_id": refl_id, "stored": refl_id is not None},
            print_message=f"[REFLECT] ({subject}) {text[:140]}...",
        )
