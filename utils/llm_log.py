"""LLM call logging — moved from utils/ollama.py when the Ollama backend
was retired (July 9 2026, Qwen-only via llama-server). One entry per model
call: prompt/response (truncated), endpoint, stream/prefill observability."""

import os
from typing import Any, Dict, Optional

from config.config import CLEAN_LLM_OUTPUT, MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType


def truncate_for_print(text: str | None, max_length: int) -> str:
    """Truncate text for print/log output (LLM_PRINT_FULL_RESPONSE disables)."""
    if not text:
        return ""
    try:
        from config.config import LLM_PRINT_FULL_RESPONSE

        if LLM_PRINT_FULL_RESPONSE:
            return text
    except ImportError:
        pass
    return text[:max_length] + "..." if len(text) > max_length else text


def _get_prompt_emoji(prompt_type: str) -> str:
    """Get appropriate emoji for prompt type."""
    emoji_map = {
        "reflection": "🤔",
        "drawing": "🎨",
        "awakening": "🌅",
        "compression": "🧠",
        "sentiment": "😊",
        "vision": "👁️",
        "caption": "📸",
        "general": "💭",
    }
    return emoji_map.get(prompt_type, "💭")


def log_llm_call(
    prompt: str,
    model: str = "",
    image_path: Optional[str] = None,
    response: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    timeout: Optional[int] = None,
    log_dir: str = "mood_snapshots",
    system_prompt: Optional[str] = None,
    prompt_type: str = "general",
    api_endpoint: Optional[str] = None,
    history_len: int = 0,
    stream_mode: Optional[str] = None,
    num_frames: int = 0,
    prefill_tail: Optional[str] = None,
    duration_s: Optional[float] = None,
    queued_s: Optional[float] = None,
):
    """
    Log an LLM API call for monitoring and debugging.

    Args:
        prompt: The prompt sent to Ollama
        model: The model name (default: "from config")
        image_path: Path to input image if any
        response: The response from Ollama
        success: Whether the API call was successful
        error_message: Error message if call failed
        timeout: Request timeout used
        log_dir: Directory to store the log
        system_prompt: Optional system prompt sent to Ollama

    Returns:
        Path to the log file
    """
    # Guard: empty log_dir would crash makedirs downstream and kill the LLM call
    if not log_dir or not log_dir.strip():
        log_dir = "mood_snapshots"

    # Truncate very long prompts and responses for readability

    truncated_prompt = truncate_for_print(prompt, 500)
    truncated_response = truncate_for_print(response, 1000)
    truncated_system_prompt = truncate_for_print(system_prompt, 500)

    data = {
        "prompt": truncated_prompt,
        "full_prompt_length": len(prompt),
        "system_prompt": truncated_system_prompt,
        "full_system_prompt_length": len(system_prompt) if system_prompt else 0,
        "model": model,
        "image_path": image_path if image_path and os.path.exists(image_path) else None,
        "has_image": image_path is not None and os.path.exists(image_path) if image_path else False,
        "response": truncated_response,
        "full_response_length": len(response) if response else 0,
        "success": success,
        "error_message": error_message,
        "timeout": timeout,
        "api_endpoint": api_endpoint or "http://localhost:11434/api/generate",
    }
    # Stream observability: the logged prompt understates what the model saw
    # whenever a history/prefill rode along — record it here.
    if history_len:
        data["history_len"] = history_len
    if stream_mode:
        data["stream_mode"] = stream_mode
    if num_frames:
        data["num_frames"] = num_frames
    if prefill_tail:
        data["prefill_tail"] = truncate_for_print(prefill_tail, 150)

    type_emoji = _get_prompt_emoji(prompt_type)
    data["prompt_type"] = prompt_type
    # HOW LONG IT TOOK, AND HOW MUCH OF THAT WAS QUEUEING (Aug 3). Nothing
    # recorded call duration, so "why does it occasionally time out" could only
    # be guessed at: one llama-server slot, three threads calling it, and a
    # caption's 60s timeout counts while it waits behind a reflection. These two
    # numbers separate a slow model from a busy one.
    if duration_s is not None:
        data["duration_s"] = round(duration_s, 2)
    if queued_s is not None:
        data["queued_s"] = round(queued_s, 2)

    call_details = [f"[🤖{type_emoji}] {prompt_type.title()} prompt -> {model}"]

    if success:
        call_details.append(f"✅ Success ({len(response)} chars)" if response else "✅ Success")
        if timeout:
            call_details.append(f"⏱️ {timeout}s timeout")
    else:
        call_details.append(f"❌ Failed: {error_message}" if error_message else "❌ Failed")

    if image_path:
        call_details.append("📸 with image")

    if response:
        call_details.append("\nResponse: " + truncate_for_print(response, 1000))

    if CLEAN_LLM_OUTPUT and response:
        # Clean output: prefer caller-owned printing for text types to avoid duplicates
        if prompt_type in ("caption", "reflection", "drawing", "awakening", "introspective_caption"):
            print_message = None  # Suppress here; higher-level modules print cleanly
        else:
            print_message = truncate_for_print(response, 1000)
    else:
        try:
            from config.config import DEBUG_LLM_PROMPTS
        except ImportError:
            DEBUG_LLM_PROMPTS = False
        if DEBUG_LLM_PROMPTS:
            debug_details = [f"\n[🤖{type_emoji}] {prompt_type.title()} PROMPT:\n{'-' * 50}", truncate_for_print(prompt, 1000), "-" * 50]
            debug_details.append(" | ".join(call_details))
            print_message = "\n".join(debug_details)
        else:
            print_message = " | ".join(call_details)

    log_json_entry(LogType.LLM_API_CALL, data, log_dir, print_message=print_message)

    if error_message and not success:
        log_json_entry(
            LogType.ERROR,
            {
                "message": f"LLM API error in {prompt_type} prompt",
                "error": error_message,
                "model": model,
                "prompt_type": prompt_type,
                "timeout": timeout,
            },
            print_message=f"[❌🤖] LLM {prompt_type} error: {error_message}",
        )
