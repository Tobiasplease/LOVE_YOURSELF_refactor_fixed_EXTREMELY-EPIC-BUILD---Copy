"""
Clean model wrapper - pure API handler only.
All prompt logic moved to prompt_interface.py for centralization.
"""

import re
from typing import Optional

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    MOTIF_MODEL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_REFLECTION,
    TINYLLAMA_NUM_PREDICT,
    TINYLLAMA_TEMPERATURE,
    TINYLLAMA_TIMEOUT,
    TINYLLAMA_TOP_P,
)
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.ollama import query_ollama, truncate_for_print

from .prompt_interface import PromptInterface
from .prompts import NUMBER_GENERATOR_SYSTEM_PROMPT


class MultimodalModel:
    """Simplified model wrapper - pure API handler."""

    def __init__(self, memory_ref: Optional[any] = None) -> None:  # type: ignore
        self.memory_ref = memory_ref
        self.model_name = OLLAMA_MODEL
        self.prompt_interface = PromptInterface(self.model_name)

    def caption_image(self, image_path: str, *, flowing: bool = True, first_time: bool = False) -> str:
        """Generate image caption using centralized prompt interface."""
        # Get prompt and options from centralized interface
        prompt, model_options, system_prompt = self.prompt_interface.build_caption_prompt_with_options(
            self.memory_ref, image_path, flowing=flowing, first_time=first_time
        )

        if prompt is None:
            return "Vision initializing... camera systems coming online..."

        log_json_entry(
            LogType.DEBUG,
            {
                "message": "Caption prompt generated",
                "action": "prompt_hash",
                "prompt_hash": hash(prompt),
                "prompt_preview": prompt[:200],
                "flowing": flowing,
                "first_time": first_time,
            },
            print_message=f"[🐞] Prompt hash: {hash(prompt)}, preview: {truncate_for_print(prompt, 200)}",
        )

        # Mood-aware candidate sampling (selection, not instruction)
        n_candidates = 1
        try:
            mv = getattr(self.memory_ref, "current_mood_vector", (0.0, 0.0, 0.0)) if self.memory_ref else (0.0, 0.0, 0.0)
            arousal = float(mv[1]) if isinstance(mv, (tuple, list)) and len(mv) >= 2 else 0.0
            if abs(arousal) > 0.3:
                n_candidates = 2
        except Exception:
            n_candidates = 1

        if n_candidates == 1:
            result = self._call_ollama(prompt, image_path=image_path, system_prompt=system_prompt, model_options=model_options, prompt_type="caption")
        else:
            import copy, random
            candidates: list[str] = []
            options_list = []
            for _ in range(n_candidates):
                opts = copy.deepcopy(model_options)
                opts["seed"] = random.randint(1, 1000000)
                options_list.append(opts)
            for opts in options_list:
                candidates.append(self._call_ollama(prompt, image_path=image_path, system_prompt=system_prompt, model_options=opts, prompt_type="caption"))

            # Simple selector: prefer lower overlap with last caption when arousal is high; shorter when arousal is low
            last_caption = getattr(self.memory_ref, "last_caption", "") if self.memory_ref else ""
            def _overlap(a: str, b: str) -> float:
                sa = set([w.lower() for w in a.split() if len(w) > 3])
                sb = set([w.lower() for w in b.split() if len(w) > 3])
                if not sa or not sb:
                    return 0.0
                return len(sa & sb) / max(1, len(sa | sb))

            best = candidates[0]
            if abs(arousal) > 0.3:
                # Prefer novelty when arousal is high
                best = min(candidates, key=lambda c: _overlap(c, last_caption) - 0.05 * len(c))
            else:
                # Prefer brevity/stability
                best = min(candidates, key=lambda c: 0.1 * _overlap(c, last_caption) + 0.01 * len(c))

            # Affective stance gating: rare, state-driven selection preference
            try:
                mv = getattr(self.memory_ref, "current_mood_vector", (0.0, 0.0, 0.0)) if self.memory_ref else (0.0, 0.0, 0.0)
                val = float(mv[0]) if isinstance(mv, (tuple, list)) and len(mv) >= 1 else 0.0
                top_aff = []
                if hasattr(self.memory_ref, "get_top_affective_motifs"):
                    top_aff = self.memory_ref.get_top_affective_motifs(1)  # type: ignore
                last_ts = getattr(self.memory_ref, "_last_affect_utter_ts", 0.0)
                import time
                cooldown_ok = (time.time() - last_ts) > 180.0  # 3 min cooldown
                if top_aff and cooldown_ok:
                    m, mval, mfix = top_aff[0]
                    strength = abs(mval) * mfix
                    if strength > 0.35 and abs(val) > 0.4:
                        neg = val < 0
                        pos_tokens = ["glad", "relieved", "open", "clear"]
                        neg_tokens = ["tired", "can’t stand", "can not stand", "sick of", "clutter", "again", "still"]
                        def has_token(s: str, toks: list[str]) -> bool:
                            low = s.lower()
                            return any(t in low for t in toks)
                        if neg:
                            cand = next((c for c in candidates if has_token(c, neg_tokens)), None)
                        else:
                            cand = next((c for c in candidates if has_token(c, pos_tokens)), None)
                        if cand:
                            best = cand
                            try:
                                setattr(self.memory_ref, "_last_affect_utter_ts", time.time())
                            except Exception:
                                pass
            except Exception:
                pass
            result = best

        log_json_entry(
            LogType.DEBUG,
            {
                "message": "Caption response received",
                "action": "response_hash",
                "response_hash": hash(result),
                "response_preview": result[:50],
                "response_length": len(result),
            },
            print_message=f"[🐞] Response hash: {hash(result)}, preview: {truncate_for_print(result, 50)}",
        )
        return result

    def reason_about_caption(
        self, caption: str, *, agent: Optional[any] = None, mood_text: Optional[str] = None, extra: Optional[str] = None  # type: ignore
    ) -> str:  # type: ignore
        """Generate reflection using centralized prompt interface."""
        try:
            prompt, model_options, system_prompt = self.prompt_interface.build_reflection_prompt_with_options(caption, agent=agent, extra=extra)

            log_json_entry(
                LogType.REFLECTION,
                {
                    "message": "Starting reflection",
                    "action": "reflection_start",
                    "timeout": OLLAMA_TIMEOUT_REFLECTION,
                    "caption_preview": caption[:50],
                },
                print_message=f"[🤔] Starting reflection with timeout={OLLAMA_TIMEOUT_REFLECTION}s",
            )

            response = self._call_ollama(
                prompt, system_prompt=system_prompt, model_options=model_options, timeout=OLLAMA_TIMEOUT_REFLECTION, prompt_type="reflection"
            )

            log_json_entry(
                LogType.REFLECTION,
                {
                    "message": "Reflection completed",
                    "action": "reflection_success",
                    "response_length": len(response),
                    "response_preview": response[:100],
                },
                print_message=f"[✅] Reflection completed: {len(response)} chars",
            )
            return response
        except Exception as e:
            print(f"[ERROR] Reflection failed: {e}")
            import traceback

            traceback.print_exc()
            return "[WARNING] Reflection generation failed"

    def generate_drawing_prompt(self, *, extra: Optional[str] = None) -> str:
        """Generate drawing prompt using centralized prompt interface."""
        prompt, model_options, system_prompt = self.prompt_interface.build_drawing_prompt_with_options(self.memory_ref, extra=extra)

        if prompt is None:
            return "[WARNING] No memory available for drawing prompt"

        # Log the exact input we send to the LLM for drawing prompt generation
        try:
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "Drawing LLM input prepared",
                    "action": "llm_input",
                    "prompt_preview": truncate_for_print(prompt, 400),
                    "prompt_length": len(prompt),
                    "options": {k: model_options.get(k) for k in ("temperature", "top_p", "repeat_penalty", "top_k", "num_predict", "seed")},
                },
                print_message=f"[🐞] Drawing LLM input: {truncate_for_print(prompt, 220)}",
            )
        except Exception:
            pass

        return self._call_ollama(prompt, system_prompt=system_prompt, model_options=model_options, prompt_type="drawing")

    def query_tinyllama(self, prompt: str) -> str:
        """Query TinyLlama model for motif scoring and emotional analysis."""
        tinyllama_options = {
            "temperature": TINYLLAMA_TEMPERATURE,
            "top_p": TINYLLAMA_TOP_P,
            "num_predict": TINYLLAMA_NUM_PREDICT,
        }

        try:
            response = query_ollama(
                prompt=prompt,
                model=MOTIF_MODEL,
                timeout=TINYLLAMA_TIMEOUT,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                system_prompt=NUMBER_GENERATOR_SYSTEM_PROMPT,
                options=tinyllama_options,
                prompt_type="motif_scoring",
            )
            return response.strip()
        except Exception:
            return "0.5"

    def _call_ollama(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: int = 90,
        model_options: dict | None = None,
        prompt_type: str = "general",
    ) -> str:
        """Pure API call handler - no prompt logic here."""

        # Use provided options or get defaults
        if model_options is None:
            model_options = self.prompt_interface._get_base_model_options()

        response = query_ollama(
            prompt=prompt,
            model=self.model_name,
            image=image_path,
            timeout=timeout,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options=model_options,
            prompt_type=prompt_type,
        )

        if not response:
            response = ""

        return self._clean_response(response)

    def _clean_response(self, response: str) -> str:
        """Remove unwanted AI-generated prompt leakage from responses."""
        unwanted_patterns = [
            r"\\n\\nFeelings:.*?\\?",
            r"\\n\\nReflection:.*?\\?",
            r"\\n\\nWhat do you feel\\?",
            r"\\n\\nHow does.*?feel\\?",
            r"Feelings: What do you feel\\?",
            r"Reflection: How does.*?\\?",
        ]

        cleaned = response
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        return cleaned.strip()
