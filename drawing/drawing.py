from __future__ import annotations


"""drawing.py – final version

Captioner now hands us a *ready-made* drawing prompt that already contains
scene caption + reflection. This version does not query the LLM again,
and passes the prompt directly to ComfyUI.
"""

import base64
import os
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from config.config import (
    CLEAN_LLM_OUTPUT,
    COMFY_CNET_STRENGTH,
    COMFY_FLUX_GUIDANCE,
    COMFY_LATENT_HEIGHT,
    COMFY_LATENT_WIDTH,
    COMFY_LORA_STRENGTH,
    COMFY_STEPS,
    DRAWING_CALL_TIMEOUT,
    DRAWING_COOLDOWN,
    MOOD_SNAPSHOT_FOLDER,
    TRIGGER_PROMPT,
)
from captioner.prompts import SELF_CRITIQUE_PROMPT, SELF_CRITIQUE_SYSTEM_PROMPT
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from utils.inference import is_failed_response, query_model
from utils.llm_log import truncate_for_print
from utils.state_manager import state_manager

from .comfy import create_impostor_controller

if TYPE_CHECKING:
    from captioner.captioner import Captioner


class DrawingController:
    """Decides when to draw and queues ComfyUI jobs."""

    def __init__(self) -> None:
        self.last_drawing_time: float = time.time() - DRAWING_COOLDOWN - 10  # Allow immediate first drawing
        self.cooldown: float = DRAWING_COOLDOWN  # seconds between drawings
        self.last_prompt: Optional[str] = None
        # Why ready_to_draw() last said no. The caller used to print "cooldown"
        # for every refusal, so a stuck GRBL flag read as a cooldown that had
        # already expired ("Blocked: cooldown (0s remaining)").
        self.last_block_reason: str = ""
        self.last_drawing_prompt: str = ""
        self.last_reflection: Optional[str] = None
        # The single self-critique of the latest drawing. Published here so the
        # GRBL completion path can record it WITHOUT running a second critique
        # of the same drawing (Aug 5, artist: "it should definitely only
        # critique it once").
        self.last_critique: Optional[str] = None
        self.quota_manager = None  # No quota system - use timer-based drawing
        # First drawing of a session rides the timer regardless of the want
        # (artist, Aug 17: keep the startup drawing for now, for testing)
        self._startup_drawing_done = False

    # ------------------------------------------------------------------
    # decision helpers
    # ------------------------------------------------------------------
    def ready_to_draw(self) -> bool:
        # Check if cooldown period has passed
        cooldown_ready = time.time() - self.last_drawing_time > self.cooldown

        # Check if currently generating or executing a drawing
        try:
            from utils.drawing_state import DrawingState
            from utils.state_manager import state_manager

            currently_executing = DrawingState.is_drawing()
            currently_generating = getattr(state_manager, "is_generating_drawing", False)

            if currently_generating:
                self.last_block_reason = "ComfyUI generation in progress"
                if not CLEAN_LLM_OUTPUT:
                    print(f"[🎨] Drawing blocked: ComfyUI generation currently in progress")
                return False
            if currently_executing:
                self.last_block_reason = "GRBL execution in progress"
                if not CLEAN_LLM_OUTPUT:
                    print(f"[🎨] Drawing blocked: GRBL execution currently in progress")
                return False
        except Exception as e:
            if not CLEAN_LLM_OUTPUT:
                print(f"[⚠️] Could not check drawing state: {e}")

        if not cooldown_ready:
            self.last_block_reason = f"cooldown ({self.cooldown - (time.time() - self.last_drawing_time):.0f}s remaining)"
        return cooldown_ready

    # Word boundaries are the whole point — the retired _drawing_intentions
    # matcher died on "ink" in "think", "void" in "avoid".
    _DESIRE_DRAW_WORDS = re.compile(
        r"\b(draw|draws|drew|drawn|drawing|sketch|sketching|ink|pen|pencil|paper|page|line|lines|mark|marks|trace|tracing|shape|figure)\b",
        re.IGNORECASE,
    )
    # Persistence threshold: a want must survive at least a few distill cycles
    # (one distill ≈ 8 captions) before the trigger counts it as meant.
    DESIRE_SHADOW_MIN_AGE_S = 600
    # Phase B (Aug 17, artist-approved after 5 shadow days: formula drew 26/26
    # like clockwork, desire would have drawn 7/26 with every refusal
    # explainable): the want decides. "formula" reverts to the timer.
    TRIGGER_MODE = os.getenv("DRAWING_TRIGGER_MODE", "desire")
    # Soft hunger (artist: compromise between forced output and allowed
    # silence): with no formed want, drawing-hunger fires after this long.
    # The monologue always carries the time since the pen last touched paper.
    HUNGER_S = float(os.getenv("DRAWING_HUNGER_S", 7200))

    def desire_shadow_verdict(self) -> dict:
        """North-star step 5, phase A: would the desire slot draw right now?

        Observability only — logged beside the formula's verdict, never acted
        on. The "no" is structural: no drawing-directed want, or one too young
        to have proven itself across distills.
        """
        want, since = "", 0.0
        try:
            from captioner.context_compression import context_compressor

            want = (context_compressor.get_current_desire() or "").strip()
            since = float(context_compressor.introspective_state.get("desire_since", 0.0) or 0.0)
        except Exception:
            pass
        age_s = time.time() - since if since > 0 else 0.0
        drawing_directed = bool(want) and bool(self._DESIRE_DRAW_WORDS.search(want))
        return {
            "desire": want[:200],
            "desire_age_s": round(age_s),
            "drawing_directed": drawing_directed,
            "would_draw": drawing_directed and age_s >= self.DESIRE_SHADOW_MIN_AGE_S,
        }

    def _log_trigger_decision(
        self, *, mode: str, verdict: bool, reason: str, shadow: dict, formula_verdict: bool, mood: float, novelty: float, boredom: float, time_since: float
    ) -> None:
        try:
            log_json_entry(
                LogType.DECISION,
                {
                    "decision": "trigger_decision",
                    "mode": mode,
                    "will_draw": verdict,
                    "reason": reason,
                    "formula_would_draw": formula_verdict,
                    "shadow_would_draw": shadow["would_draw"],
                    "desire": shadow["desire"],
                    "desire_age_s": shadow["desire_age_s"],
                    "drawing_directed": shadow["drawing_directed"],
                    "minutes_since_last": round(time_since / 60),
                    "mood": mood,
                    "novelty": novelty,
                    "boredom": boredom,
                },
                print_message=(
                    f"[🎨 TRIGGER] {mode}: {'DRAW' if verdict else 'wait'} ({reason}) — "
                    f"desire={'∅' if not shadow['desire'] else shadow['desire'][:60]!r}"
                ),
            )
        except Exception as e:
            if not CLEAN_LLM_OUTPUT:
                print(f"[⚠️] Trigger decision logging failed: {e}")

    def should_draw(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        # Clean room: the 5-step drawing pipeline is detox blind spot #3 — it
        # injects ~23 layers of stored prose AND its step system-prompts push
        # metaphor by design, so a drawing can't come out plain yet and would
        # pollute drawing_memory. Skip drawing entirely under detox; the pipeline
        # gets cleaned when its source channels (drawings/desire/reflection) and
        # its system-prompts are reworked. See docs/memory-redesign-plan.md.
        from config.config import BASE_VOICE_DETOX

        if BASE_VOICE_DETOX:
            return False
        try:
            from config.config import DRAWING_MIN_INTERVAL, DRAWING_USE_STATE_MOTIVATION
        except ImportError:
            return self._should_draw_original(mood=mood, novelty=novelty, boredom=boredom, reflection=reflection)

        time_since = time.time() - self.last_drawing_time

        if self.TRIGGER_MODE == "desire":
            # Phase B: the want decides. The floor stays as a hard guardrail;
            # startup and hunger are the two timer exceptions the artist kept.
            if time_since < DRAWING_MIN_INTERVAL:
                return False
            shadow = self.desire_shadow_verdict()
            # The retired formula still runs silently as the comparison shadow
            formula = self._should_draw_state_motivated(mood=mood, novelty=novelty, boredom=boredom, reflection=reflection)
            if not self._startup_drawing_done:
                verdict, reason = True, "startup"
            elif time_since >= self.HUNGER_S:
                verdict, reason = True, "hunger"
            elif shadow["would_draw"]:
                verdict, reason = True, "desire"
            else:
                verdict, reason = False, "no formed want"
            self._log_trigger_decision(
                mode="desire", verdict=verdict, reason=reason, shadow=shadow, formula_verdict=formula,
                mood=mood, novelty=novelty, boredom=boredom, time_since=time_since,
            )
            if verdict:
                self._startup_drawing_done = True
            return verdict

        # Legacy formula mode (DRAWING_TRIGGER_MODE="formula" reverts to this)
        if DRAWING_USE_STATE_MOTIVATION:
            verdict = self._should_draw_state_motivated(mood=mood, novelty=novelty, boredom=boredom, reflection=reflection)
        else:
            verdict = self._should_draw_original(mood=mood, novelty=novelty, boredom=boredom, reflection=reflection)

        # Log only real evaluations, not the 720-900s window where the
        # min-interval short-circuit re-runs every caption cycle.
        if time_since >= DRAWING_MIN_INTERVAL:
            self._log_trigger_decision(
                mode="formula", verdict=verdict, reason="formula", shadow=self.desire_shadow_verdict(), formula_verdict=verdict,
                mood=mood, novelty=novelty, boredom=boredom, time_since=time_since,
            )
        return verdict

    def _should_draw_original(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        """Pure timer-based drawing decision logic for debugging."""
        if not self.ready_to_draw():
            cooldown_remaining = max(0, self.cooldown - (time.time() - self.last_drawing_time))
            if not CLEAN_LLM_OUTPUT:
                print(f"[🎨] Timer drawing check: BLOCKED by cooldown ({cooldown_remaining:.0f}s remaining)")
            return False

        # Pure timer-based: if cooldown passed, always draw
        if not CLEAN_LLM_OUTPUT:
            print(f"[🎨] ✨ TIMER DRAWING TRIGGERED (debug mode - ignoring mood/novelty/boredom)")
        return True

    def _should_draw_state_motivated(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        """Sophisticated state-motivated drawing decision logic."""
        import random
        from config.config import (
            DRAWING_MIN_INTERVAL,
            DRAWING_MAX_INTERVAL,
            DRAWING_BASE_THRESHOLD,
            DRAWING_NOVELTY_WEIGHT,
            DRAWING_BOREDOM_WEIGHT,
            DRAWING_MOOD_WEIGHT,
        )

        current_time = time.time()
        time_since_last = current_time - self.last_drawing_time

        # Absolute minimum interval - safety check
        if time_since_last < DRAWING_MIN_INTERVAL:
            remaining = DRAWING_MIN_INTERVAL - time_since_last
            if not CLEAN_LLM_OUTPUT:
                print(f"[🎨] State drawing check: BLOCKED by minimum interval ({remaining:.0f}s remaining)")
            return False

        # Force drawing if maximum interval exceeded (ensure some activity)
        if time_since_last >= DRAWING_MAX_INTERVAL:
            if not CLEAN_LLM_OUTPUT:
                print(f"[🎨] ✨ STATE DRAWING TRIGGERED: Maximum interval exceeded ({time_since_last:.0f}s)")
            return True

        # Calculate state-based drawing motivation score
        # Normalize inputs to 0-1 range
        normalized_mood = max(0, min(1, (mood + 1) / 2))  # mood is -1 to 1, normalize to 0-1
        normalized_novelty = max(0, min(1, novelty))  # novelty should be 0-1
        normalized_boredom = max(0, min(1, boredom))  # boredom should be 0-1

        # Calculate weighted motivation score
        motivation_score = (
            normalized_novelty * DRAWING_NOVELTY_WEIGHT + normalized_boredom * DRAWING_BOREDOM_WEIGHT + normalized_mood * DRAWING_MOOD_WEIGHT
        )

        # Add time pressure - gradually increase motivation over time
        time_factor = min(1.0, time_since_last / DRAWING_MAX_INTERVAL)
        time_pressure = time_factor * 0.2  # Up to 0.2 additional motivation

        total_motivation = motivation_score + time_pressure

        # Startup bonus - ensure first drawing happens within 5 minutes
        startup_bonus = 0.0
        STARTUP_DRAWING_WINDOW = 300  # 5 minutes
        if time_since_last > STARTUP_DRAWING_WINDOW:
            # This is likely the first drawing (time since last is huge)
            # Add bonus to ensure it triggers soon after startup delay
            startup_bonus = 0.3
            print(f"[🎨] First drawing bonus: +{startup_bonus}")

        total_motivation = total_motivation + startup_bonus

        # Add small random factor for unpredictability (±0.1)
        randomness = (random.random() - 0.5) * 0.2
        final_score = total_motivation + randomness

        # Decision threshold
        will_draw = final_score >= DRAWING_BASE_THRESHOLD

        # Calculate cooldown remaining for display
        cooldown_remaining = max(0, DRAWING_MIN_INTERVAL - time_since_last)
        cooldown_minutes = cooldown_remaining / 60
        cooldown_percent = (time_since_last / DRAWING_MIN_INTERVAL) * 100

        if not CLEAN_LLM_OUTPUT:
            print(f"[🎨] State drawing evaluation:")
            print(f"  ⏱️  Cooldown: {cooldown_remaining:.0f}s remaining ({cooldown_minutes:.1f} min) - {cooldown_percent:.0f}% elapsed")
            print(f"  Time since last: {time_since_last:.0f}s (min: {DRAWING_MIN_INTERVAL}s, max: {DRAWING_MAX_INTERVAL}s)")
            print(f"  Mood: {normalized_mood:.3f}, Novelty: {normalized_novelty:.3f}, Boredom: {normalized_boredom:.3f}")
            bonus_str = f", Startup: +{startup_bonus:.3f}" if startup_bonus > 0 else ""
            print(f"  Base motivation: {motivation_score:.3f}, Time pressure: {time_pressure:.3f}{bonus_str}")
            print(f"  Final score: {final_score:.3f} (threshold: {DRAWING_BASE_THRESHOLD})")
            print(f"  Decision: {'DRAW' if will_draw else 'WAIT'}")

        if will_draw and not CLEAN_LLM_OUTPUT:
            print(f"[🎨] ✨ STATE DRAWING TRIGGERED: Internal motivation reached threshold")

        return will_draw

    def register_drawing(self, prompt: str) -> None:
        from config.config import DRAWING_MAX_INTERVAL

        self.last_drawing_time = time.time()
        self.last_prompt = prompt
        self.last_drawing_prompt = prompt

        # The pen actually drew — this is the one place that may promote the
        # drawing-memory entry to executed (the arc reads executed-only)
        try:
            from drawing.drawing_memory import DrawingMemory, get_drawing_memory

            get_drawing_memory().mark_last_completed()
            # Desire arc: the act discharges the want. Post-GRBL only — an
            # intent that never reached paper spends nothing.
            from captioner.context_compression import context_compressor

            context_compressor.spend_desire(drawing_summary=DrawingMemory._strip_comfy_preamble(prompt or ""))
        except Exception:
            pass

        if not CLEAN_LLM_OUTPUT:
            print(f"\n{'='*60}")
            print(f"🔔 DRAWING COOLDOWN RESET")
            print(f"{'='*60}")
            print(f"Physical drawing completed. Cooldown timer started.")
            print(f"Next drawing possible in: {self.cooldown}s ({self.cooldown/60:.1f} minutes)")
            print(f"Forced drawing at: {DRAWING_MAX_INTERVAL}s ({DRAWING_MAX_INTERVAL/60:.1f} minutes)")
            print(f"{'='*60}\n")

        # Surface a clean line in the live caption monitor + episodic log
        try:
            from utils.live_log import log_drawing_complete

            # Strip the standard ComfyUI preamble for readability
            desc = (prompt or "").strip()
            for prefix in (
                "Black ink line drawing on white paper. ",
                "Black ink line drawing on white paper.",
                "black ink line drawing on white paper. ",
            ):
                if desc.lower().startswith(prefix.lower()):
                    desc = desc[len(prefix) :]
                    break
            if len(desc) > 200:
                desc = desc[:200].rsplit(" ", 1)[0] + "..."
            log_drawing_complete(desc)
            try:
                from utils.episodic_log import episodic_log

                # Truncate further for episodic — just the subject
                short = desc[:60].rsplit(" ", 1)[0] if len(desc) > 60 else desc
                episodic_log.record("drew", f"finished a drawing of {short}")
            except Exception:
                pass
        except Exception:
            pass

        # LCD display is now handled in handle_drawing_flow with the drawing summary

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def handle_drawing_flow(
        self,
        agent: "Captioner",
        drawing_prompt: str,
        latest_image: str,
        *,
        reflection: Optional[str] = None,
    ) -> None:
        """Captioner passes the prompt already built – we just queue it.

        NOTE: The captioner has already evaluated should_draw() and approved.
        Do NOT re-check here — the cooldown was reset after prompt generation,
        so a second check would always fail.
        """
        self.last_reflection = reflection
        novelty = getattr(agent, "novelty_score", 0.0)
        boredom = getattr(agent, "boredom", 0.0)

        try:
            # === EARLY PAPER CHECK (before ComfyUI generation to save resources) ===
            try:
                from config.config import ENABLE_EARLY_PAPER_CHECK, ENABLE_PAPER_DETECTION

                if ENABLE_PAPER_DETECTION and ENABLE_EARLY_PAPER_CHECK:
                    from safety.paper_detection import check_paper_before_drawing

                    camera = state_manager.camera
                    servos = state_manager.servos

                    if camera is not None:
                        print("[📄] Running early paper check before ComfyUI generation...")
                        paper_present = check_paper_before_drawing(camera, servos, None)

                        try:
                            from vision.gaze import set_drawing_mode

                            set_drawing_mode(active=False)
                        except Exception:
                            pass

                        if not paper_present:
                            log_json_entry(
                                LogType.DECISION,
                                {
                                    "decision": "skip_drawing",
                                    "reason": "early_paper_check_failed",
                                    "mood": agent.current_mood,
                                    "novelty": novelty,
                                    "boredom": boredom,
                                },
                                print_message="[📄] Early paper check: NO PAPER - skipping ComfyUI generation",
                            )
                            state_manager.last_no_paper_skip_ts = time.time()
                            try:
                                from drawing.drawing_memory import get_drawing_memory

                                get_drawing_memory().record_failure(
                                    reason="no paper",
                                    prompt=getattr(state_manager, "current_drawing_prompt", None),
                                )
                            except Exception:
                                pass
                            try:
                                from utils.live_log import log_drawing_failed

                                log_drawing_failed("no paper")
                            except Exception:
                                pass
                            return
                        else:
                            print("[📄] Early paper check: PAPER PRESENT - proceeding with ComfyUI")
                    else:
                        print("[📄] Early paper check skipped: no camera available")
            except Exception as e:
                print(f"[📄] Early paper check error (proceeding anyway): {e}")

            # Display, shared state, and the drawing_intent memory all carry the
            # intent the machine actually formed (stream pipeline's
            # _last_drawing_intent). The old LLM summary call here is retired
            # (Aug 17): it re-described the RENDER prompt and stored the
            # paraphrase as a second "drawing_intent" memory, so every drawing
            # was remembered twice in two voices. The memory write itself stays —
            # the live drawing-introspection captions read this type — but now
            # in the machine's own words. One drawing, one memory, one voice.
            try:
                from config.config import INCLUDE_DRAWING_HISTORY

                intent_text = (getattr(agent, "_last_drawing_intent", "") or "").strip() or drawing_prompt
                display_line = intent_text.strip().split("\n")[0]
                if len(display_line) > 200:
                    display_line = display_line[:200].rsplit(" ", 1)[0] + "..."
                state_manager.current_drawing_prompt = display_line
                try:
                    from utils.caption_display import send_caption_to_display

                    send_caption_to_display(f"Drawing: {display_line}")
                except Exception:
                    pass
                if INCLUDE_DRAWING_HISTORY and hasattr(agent, "observe"):
                    agent.observe(f"Drawing intent: {intent_text}", agent.current_mood, latest_image or "", memory_type="drawing_intent")
            except Exception as e:
                print(f"[❌] Failed to surface drawing intent: {e}")

            log_json_entry(
                LogType.DECISION,
                {
                    "decision": "trigger_drawing",
                    "reason": "inspired",
                    "mood": agent.current_mood,
                    "novelty": novelty,
                    "boredom": boredom,
                    "drawing_prompt": drawing_prompt,
                    "reflection": (reflection or "").strip(),
                },
                print_message=f"[🎨] Drawing prompt:\n{drawing_prompt}",
            )
            try:
                print("[🖼️ Drawing Prompt]\n" + drawing_prompt)
            except Exception:
                pass

            if latest_image and os.path.exists(latest_image):
                self._invoke_comfyui_drawing(drawing_prompt, latest_image)
            else:
                log_json_entry(
                    LogType.ERROR,
                    {"message": "Cannot invoke ComfyUI - no valid image available", "component": "drawing", "image_path": latest_image},
                    print_message="[❌] Cannot invoke ComfyUI – no valid image available",
                )

        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Error in drawing flow: {exc}", "component": "drawing"},
                print_message=f"[❌] Error in drawing flow: {exc}",
            )

    # ------------------------------------------------------------------
    # VRAM management
    # ------------------------------------------------------------------
    def _unload_inference_model(self) -> None:
        """Unload llama-server from VRAM to free space for ComfyUI/Flux."""
        from utils.inference import unload_model

        try:
            unload_model()
            print("[VRAM] Inference model unloaded for ComfyUI")
        except Exception as e:
            print(f"[VRAM] Model unload failed: {e}")

    # ------------------------------------------------------------------
    # ComfyUI invocation helper
    # ------------------------------------------------------------------
    def _invoke_comfyui_drawing(self, drawing_prompt: str, latest_image: str) -> None:
        try:
            # Reachability first (July 30): with ComfyUI unplugged, unloading
            # llama-server here bought a pointless reload and the block stayed
            # invisible to the machine. Check before touching VRAM; a blocked
            # attempt is a code-attested event the identity layer should see.
            import requests as _rq

            from utils.drawing_state import DrawingState

            try:
                _rq.get("http://localhost:8188/system_stats", timeout=3)
                DrawingState.mark_vision_online()
            except Exception:
                DrawingState.mark_vision_offline()
                try:
                    from captioner.context_compression import context_compressor

                    context_compressor.note_perception_event("drawing_blocked")
                except Exception:
                    pass
                log_json_entry(
                    LogType.DECISION,
                    {"decision": "drawing_blocked", "reason": "comfyui_unreachable", "drawing_prompt": drawing_prompt[:120]},
                    print_message="[🎨] ComfyUI unreachable — the machine reached for a drawing and nothing could form",
                )
                return

            # Unload the inference model from VRAM so ComfyUI/Flux can allocate
            self._unload_inference_model()

            if os.path.exists(latest_image):
                image_path = latest_image
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"draw_input_{timestamp}.jpg")
                image_data = base64.b64decode(latest_image)
                with open(image_path, "wb") as f:
                    f.write(image_data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"impostor-{timestamp}"
            controller = create_impostor_controller(
                load_image_path=image_path,
                override_prompt=drawing_prompt,
                primitive_string=TRIGGER_PROMPT,
                filename_prefix=filename_prefix,
                latent_width=COMFY_LATENT_WIDTH,
                latent_height=COMFY_LATENT_HEIGHT,
                cnet_strength=COMFY_CNET_STRENGTH,
                flux_guidance=COMFY_FLUX_GUIDANCE,
                steps=COMFY_STEPS,
                lora_strength=COMFY_LORA_STRENGTH,
            )
            if controller.queue_prompt():
                # Track expected output so the monitor only processes this job's image
                try:
                    state_manager.set_expected_output_prefix(filename_prefix)
                except Exception:
                    pass
                state_manager.start_drawing_generation(drawing_prompt)
                log_json_entry(
                    LogType.COMFY_PROMPT,
                    {"message": "ComfyUI drawing queued successfully", "drawing_prompt": drawing_prompt},
                    print_message=f"[🎨] Queued to ComfyUI with prompt:\n{drawing_prompt}",
                )
                # Always echo queued prompt to console as well
                try:
                    print("[🖼️ Queued Prompt]\n" + drawing_prompt)
                except Exception:
                    pass
                # Store an immediate post-queue note for drawing history
                try:
                    from config.config import INCLUDE_DRAWING_HISTORY

                    if INCLUDE_DRAWING_HISTORY:
                        note = drawing_prompt.strip().split("\n")[-1]
                        if len(note) > 160:
                            note = note[:157] + "..."
                        # We don't have the agent here; rely on state manager after image completes
                except Exception:
                    pass
            else:
                log_json_entry(
                    LogType.ERROR,
                    {"message": "Failed to queue ComfyUI drawing", "component": "comfy", "drawing_prompt": drawing_prompt},
                    print_message="[❌] Failed to queue ComfyUI drawing",
                )
        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Error invoking ComfyUI: {exc}", "component": "comfy"},
                print_message=f"[❌] Error invoking ComfyUI: {exc}",
            )
