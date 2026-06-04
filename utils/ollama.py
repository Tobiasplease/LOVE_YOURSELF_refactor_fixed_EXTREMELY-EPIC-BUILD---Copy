# from calendar import c
import base64
import json
import os
import time
from typing import Optional, Union

import requests

from config.config import CLEAN_LLM_OUTPUT, DEBUG_OLLAMA_PROMPTS, MOOD_SNAPSHOT_FOLDER, OLLAMA_MODEL, OLLAMA_PRINT_FULL_RESPONSE, OLLAMA_SHOW_PROGRESS
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.progress_bar import ProgressBar


def truncate_for_print(text: str | None, max_length: int) -> str:
    """Format text for print output, respecting OLLAMA_PRINT_FULL_RESPONSE config.

    Args:
        text: The text to format
        max_length: Maximum length before truncation (ignored if OLLAMA_PRINT_FULL_RESPONSE is True)

    Returns:
        Full text if OLLAMA_PRINT_FULL_RESPONSE is True, otherwise truncated with "..." if needed
    """
    if not text:
        return ""

    if OLLAMA_PRINT_FULL_RESPONSE:
        return text

    return text[:max_length] + "..." if len(text) > max_length else text


def _get_prompt_emoji(prompt_type: str) -> str:
    """Get appropriate emoji for prompt type."""
    emoji_map = {
        "reflection": "🤔",
        "drawing": "🎨",
        "awakening": "🌅",
        "compression": "🧠",
        "sentiment": "😊",
        "motif_scoring": "📊",
        "vision": "👁️",
        "caption": "📸",
        "general": "💭",
    }
    return emoji_map.get(prompt_type, "💭")


def log_ollama_call(
    prompt: str,
    model: str = OLLAMA_MODEL,
    image_path: Optional[str] = None,
    response: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    timeout: Optional[int] = None,
    log_dir: str = "mood_snapshots",
    system_prompt: Optional[str] = None,
    prompt_type: str = "general",
):
    """
    Log Ollama API call details for monitoring and debugging.

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
        "api_endpoint": "http://localhost:11434/api/generate",
    }

    type_emoji = _get_prompt_emoji(prompt_type)
    data["prompt_type"] = prompt_type

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
    elif DEBUG_OLLAMA_PROMPTS:
        debug_details = [f"\n[🤖{type_emoji}] {prompt_type.title()} PROMPT:\n{'-' * 50}", truncate_for_print(prompt, 1000), "-" * 50]

        # if system_prompt:
        #     debug_details.extend([f"[🤖⚙️] SYSTEM PROMPT:\n{'-' * 30}", truncate_for_print(system_prompt, 500)])

        debug_details.append(" | ".join(call_details))

        print_message = "\n".join(debug_details)
    else:
        print_message = " | ".join(call_details)

    log_json_entry(LogType.OLLAMA_API_CALL, data, log_dir, print_message=print_message)

    if error_message and not success:
        log_json_entry(
            LogType.ERROR,
            {
                "message": f"Ollama API error in {prompt_type} prompt",
                "error": error_message,
                "model": model,
                "prompt_type": prompt_type,
                "timeout": timeout,
            },
            print_message=f"[❌🤖] Ollama {prompt_type} error: {error_message}",
        )


def _wait_for_drawing_completion() -> None:
    """Wait for ComfyUI drawing generation to complete, but allow captions during CNC execution."""
    from utils.state_manager import state_manager

    # Only block during ComfyUI generation for performance
    # Allow captions during CNC execution so user can see AI's drawing awareness
    if state_manager.is_generating_drawing:
        log_json_entry(
            LogType.INFO,
            {
                "message": "Ollama API call paused - waiting for ComfyUI generation to complete",
                "drawing_prompt": state_manager.current_drawing_prompt,
            },
            print_message="⏸️ Ollama paused - ComfyUI generating",
        )

        while state_manager.is_generating_drawing:
            time.sleep(1.0)

        log_json_entry(
            LogType.INFO, {"message": "ComfyUI generation completed - resuming Ollama calls"}, print_message="▶️ Ollama resumed - generation complete"
        )

        # Free ComfyUI's VRAM (Flux weights) so Ollama can reload
        try:
            resp = requests.post(
                "http://localhost:8188/free",
                json={"unload_models": True, "free_memory": True},
                timeout=10,
            )
            if resp.ok:
                print("[🧹] Freed ComfyUI/Flux models from VRAM")
            else:
                print(f"[⚠️] ComfyUI /free returned {resp.status_code}")
        except Exception as e:
            print(f"[⚠️] Could not free ComfyUI VRAM: {e}")

        # Preload the vision model first so it gets VRAM priority over the text-only model
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": "5m"},
                timeout=120,
            )
            if resp.ok:
                print(f"[🔄] Preloaded {OLLAMA_MODEL} into VRAM (vision priority)")
            else:
                print(f"[⚠️] Failed to preload {OLLAMA_MODEL}: {resp.status_code}")
        except Exception as e:
            print(f"[⚠️] Could not preload {OLLAMA_MODEL}: {e}")

    # Do NOT block during CNC execution - captions should continue during physical drawing


def query_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    image: Optional[Union[str, bytes]] = None,
    timeout: int = 20,
    log_dir: str = MOOD_SNAPSHOT_FOLDER,
    system_prompt: Optional[str] = None,
    strict_evaluation: bool = False,
    options: Optional[dict] = None,
    show_progress: bool = OLLAMA_SHOW_PROGRESS,
    prompt_type: str = "general",
    skip_generation_wait: bool = False,
    prior_assistant_turn: Optional[str] = None,
) -> str:
    """
    Query Ollama API with a prompt and optional image.

    Args:
        prompt: The text prompt to send
        model: The model name (default: "OLLAMA_MODEL")
        image: Either a file path to an image or base64 encoded image bytes
        timeout: Request timeout in seconds
        log_dir: Directory to store logs
        system_prompt: Optional system prompt to set context
        options: Model-specific generation options (temperature, top_p, etc.)
        show_progress: Show animated ASCII progress bar during generation
        prior_assistant_turn: If set and model is Qwen, use /api/chat with this as a
            planted prior assistant turn so Qwen continues its own voice rather than
            describing the prior caption as external context.

    Returns:
        Response text from Ollama
    """
    if not skip_generation_wait:
        _wait_for_drawing_completion()

    use_streaming = show_progress

    # Encode image early so both API paths can use it
    image_path = None
    img_b64 = None
    if image is not None:
        if isinstance(image, str):
            if os.path.exists(image):
                image_path = image
                with open(image, "rb") as img_file:
                    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            else:
                img_b64 = image  # already base64
        elif isinstance(image, bytes):
            img_b64 = base64.b64encode(image).decode("utf-8")

    # Qwen3.5+ requires chat API with think:false to get direct responses.
    # Older Qwen models also benefit from chat API for planted prior turns.
    is_qwen35 = "qwen3" in model.lower()
    use_chat_api = is_qwen35 or (prior_assistant_turn is not None and "qwen" in model.lower())

    if use_chat_api:
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        if prior_assistant_turn and not is_qwen35:
            # Legacy Qwen2.5: plant prior caption as assistant turn for voice continuity
            messages.append({"role": "user", "content": "..."})
            prior_clean = prior_assistant_turn.strip()
            sent_end = min((prior_clean.find(c) for c in ".?!" if prior_clean.find(c) > 8), default=-1)
            prior_anchor = prior_clean[: sent_end + 1] if sent_end > 0 else prior_clean[:80]
            messages.append({"role": "assistant", "content": prior_anchor})

        # Current prompt + image
        current_user: dict = {"role": "user", "content": prompt}
        if img_b64:
            current_user["images"] = [img_b64]
        messages.append(current_user)

        payload: dict = {"model": model, "messages": messages, "stream": use_streaming}
        if options:
            payload["options"] = options
        if is_qwen35:
            payload["think"] = False
        endpoint = "http://localhost:11434/api/chat"

        if prior_assistant_turn:
            log_json_entry(
                LogType.DEBUG,
                {"message": "Using /api/chat with planted prior turn", "prior_preview": prior_assistant_turn[:60]},
                print_message=f"[💬] Qwen chat mode — prior: {prior_assistant_turn[:60]}",
            )
    else:
        payload = {"model": model, "prompt": prompt, "stream": use_streaming}
        if options:
            payload["options"] = options
        elif strict_evaluation:
            payload["options"] = {"temperature": 0.1, "top_p": 0.8, "repeat_penalty": 1.1}
        if system_prompt and system_prompt.strip():
            payload["system"] = system_prompt
        if img_b64:
            payload["images"] = [img_b64]
        else:
            payload["images"] = []
        endpoint = "http://localhost:11434/api/generate"

    try:
        progress_bar = None
        if use_streaming:
            progress_bar = ProgressBar(description="")
            progress_bar.start()

            response = requests.post(endpoint, json=payload, timeout=timeout, stream=True)
            response.raise_for_status()

            response_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if use_chat_api:
                            response_text += chunk.get("message", {}).get("content", "")
                        else:
                            response_text += chunk.get("response", "")
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

            progress_bar.stop(success=True)
        else:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            if use_chat_api:
                response_text = response.json().get("message", {}).get("content", "")
            else:
                response_text = response.json().get("response", "")

        log_ollama_call(
            prompt=prompt,
            model=model,
            image_path=image_path,
            response=response_text,
            success=True,
            timeout=timeout,
            log_dir=log_dir,
            system_prompt=system_prompt,
            prompt_type=prompt_type,
        )

        return response_text

    except Exception as e:
        error_msg = str(e)

        if progress_bar:
            progress_bar.stop(success=False)

        log_ollama_call(
            prompt=prompt,
            model=model,
            image_path=image_path,
            response=None,
            success=False,
            error_message=error_msg,
            timeout=timeout,
            log_dir=log_dir,
            system_prompt=system_prompt,
            prompt_type=prompt_type,
        )

        return f"[WARNING] Ollama API failed: {error_msg}"
