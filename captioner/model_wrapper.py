"""
Clean model wrapper - pure API handler only.
All prompt logic moved to prompt_interface.py for centralization.
"""

import re
from typing import Optional

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_REFLECTION,
)
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.ollama import query_ollama, truncate_for_print

from .prompt_interface import PromptInterface

_VQA_STARTS = (
    "the image", "the scene depicts", "the room appears",
    "this image", "this scene", "appears to be", "appears to show",
    "can be seen", "is shown", "depicts", "the workspace is",
)

def _is_plantable_prior(text: str) -> bool:
    """Return True if the caption is safe to plant in voice thread.
    Rejects VQA-register text, Natsumura roleplay bleed, and garbage.
    """
    t = text.strip()
    if not t:
        return False
    # Reject known garbage tokens
    if t.startswith("addCriterion") or t.startswith("[WARNING]") or t.startswith("Vision initializing"):
        return False

    t_lower = t.lower()
    first80 = t_lower[:80]

    # Reject Natsumura roleplay bleed: "You:" prefix, asterisk actions, second-person
    if t_lower.startswith("you:") or t_lower.startswith("you "):
        return False
    if "*you " in t_lower or "*your " in t_lower:
        return False
    if "your " in first80 or "you are" in first80 or "you're " in first80:
        return False

    # Reject VQA analytical patterns (detailed description from observer perspective)
    # These are NOT simple openings but actual analytical language
    vqa_patterns = [
        "it's clear that", "it is clear that", "it's obvious",
        "appears to be", "appears to show", "seems to be",
        "is filled with", "is cluttered", "is disorganized",
        "is covered with", "contains", "features",
        "as i stand", "as i look", "as i observe",
        "looking at", "examining", "observing",
        "can be seen", "is shown", "is depicted",
    ]

    for pattern in vqa_patterns:
        if pattern in t_lower:
            return False

    # Reject VQA image-description openings
    for bad in _VQA_STARTS:
        if t_lower.startswith(bad.lower()):
            return False

    # Reject overly long captions (not inner monologue)
    if len(t) > 150:
        return False

    return True


def _extract_first_sentence(text: str, min_chars: int = 15, max_chars: int = 80) -> str:
    """Extract first complete sentence from text.

    Returns first sentence ending with . ! or ? after min_chars.
    If no sentence boundary found within max_chars, returns empty string.
    """
    text = text.strip()
    if len(text) < min_chars:
        return ""

    # Find first sentence boundary
    for end_idx, char in enumerate(text):
        if char in ".!?" and end_idx >= min_chars:
            return text[:end_idx + 1]

    # No complete sentence found within reasonable length
    return ""


def build_caption_thread(agent, max_captions: int = 3) -> str:
    """Build caption thread from recent captions.

    Returns formatted dashed thread of filtered, truncated prior captions.
    """
    if not hasattr(agent, "recent_captions") or not agent.recent_captions:
        return ""

    valid_captions = []

    # Pull recent captions, filter and truncate
    for caption_entry in list(agent.recent_captions)[-max_captions:]:
        if isinstance(caption_entry, dict):
            caption_text = caption_entry.get("text", "")
        else:
            # Handle tuple format (caption, timestamp, mode)
            caption_text = caption_entry[0] if caption_entry else ""

        if not caption_text:
            continue

        # Filter through safety checks
        if not _is_plantable_prior(caption_text):
            continue

        # Extract first sentence only
        sentence = _extract_first_sentence(caption_text)
        if not sentence:
            continue

        valid_captions.append(sentence)

    # Format as dashed thread
    if not valid_captions:
        return ""

    thread_lines = ["My thoughts:"]
    for caption in valid_captions:
        thread_lines.append(f"— {caption}")
    thread_lines.append("—")

    return "\n".join(thread_lines)


class MultimodalModel:
    """Simplified model wrapper - pure API handler."""

    def __init__(self, memory_ref: Optional[any] = None) -> None:  # type: ignore
        self.memory_ref = memory_ref
        self.model_name = OLLAMA_MODEL
        self.prompt_interface = PromptInterface(self.model_name)

    def caption_image(self, image_path: str, *, flowing: bool = True, first_time: bool = False, drawing_introspection_mode: bool = False, person_present: bool = False) -> tuple:
        """Generate image caption using centralized prompt interface.

        Returns:
            tuple: (caption_text, prompt_mode) where prompt_mode is 'introspective', 'observational', 'relational', etc.
        """
        # Get prompt and options from centralized interface
        prompt, model_options, system_prompt, prompt_mode = self.prompt_interface.build_caption_prompt_with_options(
            self.memory_ref, image_path, flowing=flowing, first_time=first_time, drawing_introspection_mode=drawing_introspection_mode, person_present=person_present
        )

        if prompt is None:
            return ("Vision initializing... camera systems coming online...", "awakening")

        log_json_entry(
            LogType.DEBUG,
            {
                "message": "Caption prompt generated",
                "action": "prompt_hash",
                "prompt_hash": hash(prompt),
                "prompt_preview": prompt[:200],
                "flowing": flowing,
                "first_time": first_time,
                "prompt_mode": prompt_mode,
            },
            print_message=f"[🐞] Prompt hash: {hash(prompt)}, mode: {prompt_mode}, preview: {truncate_for_print(prompt, 200)}",
        )

        # Introspective mode uses text-only Natsumura for LLaVA (no image needed).
        # Qwen is a vision model — route ALL modes through the visual API path.
        is_qwen = "qwen" in self.model_name.lower()
        if prompt_mode == "introspective" and not is_qwen:
            result = self._call_natsumura_introspective(prompt, system_prompt, model_options)
        else:
            result = self._call_ollama(prompt, image_path=image_path, system_prompt=system_prompt, model_options=model_options, prompt_type="caption")

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
        return (result, prompt_mode)

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

    def generate_drawing_prompt(self, *, extra: Optional[str] = None, image_path: Optional[str] = None) -> str:
        """Generate drawing prompt using centralized prompt interface with VISUAL GROUNDING."""
        prompt, model_options, system_prompt = self.prompt_interface.build_drawing_prompt_with_options(
            self.memory_ref, extra=extra, image_path=image_path
        )

        if prompt is None:
            return "[WARNING] No memory available for drawing prompt"

        # Log the exact input we send to the LLM for drawing prompt generation
        try:
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "Visual drawing LLM input prepared",
                    "action": "llm_input",
                    "prompt_preview": truncate_for_print(prompt, 400),
                    "prompt_length": len(prompt),
                    "image_provided": image_path is not None,
                    "image_path": image_path,
                    "options": {k: model_options.get(k) for k in ("temperature", "top_p", "repeat_penalty", "top_k", "num_predict", "seed")},
                },
                print_message=f"[🎨] Visual drawing prompt generation {'WITH IMAGE' if image_path else 'TEXT ONLY'}: {truncate_for_print(prompt, 220)}",
            )
        except Exception:
            pass

        # If using natsumura or multi-step analysis, the prompt IS the final result, don't call LLM again
        try:
            from config.config import DRAWING_ANALYSIS_MODE
            if DRAWING_ANALYSIS_MODE in ("natsumura", "multi_step"):
                return prompt  # These modes already return the final drawing prompt
        except ImportError:
            pass

        # For single-prompt approach, call LLM
        return self._call_ollama(prompt, image_path=image_path, system_prompt=system_prompt, model_options=model_options, prompt_type="drawing")

    def _call_natsumura_introspective(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_options: dict | None = None,
    ) -> str:
        """Call Natsumura model for introspective/narrative captions (no image needed)."""
        try:
            from config.config import COMPRESSION_MODEL
            natsumura_model = COMPRESSION_MODEL
        except ImportError:
            natsumura_model = "natsumura-storytelling-rp:latest"

        # Use provided options or get defaults, but adjust for narrative mode
        if model_options is None:
            model_options = self.prompt_interface._get_base_model_options()

        # Adjust options for narrative introspection - SHORT inner thoughts, not prose
        model_options.update({
            "temperature": 0.9,
            "top_p": 0.7,
            "repeat_penalty": 1.5,
            "num_predict": 60,  # Force brevity - let num_predict handle length
            # Removed aggressive stop sequences - they cause mid-thought truncation
        })

        log_json_entry(
            LogType.DEBUG,
            {
                "message": "Natsumura introspective call",
                "action": "natsumura_caption",
                "model": natsumura_model,
                "prompt_preview": prompt[:150],
            },
            print_message=f"[🌸] Natsumura introspective: {truncate_for_print(prompt, 100)}",
        )

        response = query_ollama(
            prompt=prompt,
            model=natsumura_model,
            image=None,  # No image for introspective mode
            timeout=60,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options=model_options,
            prompt_type="introspective_caption",
        )

        if not response:
            response = ""

        return self._clean_response(response)

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

        if model_options is None:
            model_options = self.prompt_interface._get_base_model_options()

        # For Qwen captions: plant prior caption as owned assistant turn so Qwen
        # continues its voice rather than describing the prior caption as external context.
        # Only plant if the prior caption is genuine first-person inner monologue —
        # VQA register (starts with "The scene/room/image") or garbage poisons the voice.
        prior_turn = None
        if prompt_type == "caption" and "qwen" in self.model_name.lower() and self.memory_ref:
            last = getattr(self.memory_ref, "last_caption", None)
            if last and _is_plantable_prior(last):
                prior_turn = last

        # Debug: print full prompt structure for review
        if prompt_type in ("caption", "reflection", "drawing"):
            print(f"\n{'='*80}\n[PROMPT] {prompt_type.upper()}\n{'='*80}")
            if system_prompt:
                print(f"SYSTEM:\n{system_prompt}\n")
            print(f"USER:\n{prompt}\n")
            if prior_turn:
                print(f"[PLANTED PRIOR] (injected as assistant turn for voice continuity):\n{prior_turn}\n")
            print(f"{'='*80}\n")

        response = query_ollama(
            prompt=prompt,
            model=self.model_name,
            image=image_path,
            timeout=timeout,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options=model_options,
            prompt_type=prompt_type,
            prior_assistant_turn=prior_turn,
        )

        if not response:
            response = ""

        cleaned = self._clean_response(response)

        # Print the caption immediately after the prompt for continuity review
        if prompt_type == "caption" and cleaned:
            print(f"[RESPONSE] CAPTION:\n{cleaned}\n")

        return cleaned

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
