import os

# import time
import base64
import requests
from typing import Optional, Union
from config.config import MOOD_SNAPSHOT_FOLDER, OLLAMA_MODEL
from event_logging.event_logger import log_json_entry, LogType


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
) -> str:
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
    truncated_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
    truncated_response = response[:1000] + "..." if response and len(response) > 1000 else response
    truncated_system_prompt = system_prompt[:500] + "..." if system_prompt and len(system_prompt) > 500 else system_prompt

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

    return log_json_entry(LogType.OLLAMA_API_CALL, data, log_dir)


def query_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    image: Optional[Union[str, bytes]] = None,
    timeout: int = 20,
    log_dir: str = MOOD_SNAPSHOT_FOLDER,
    system_prompt: Optional[str] = None,
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

    Returns:
        Response text from Ollama
    """
    # Prepare the payload
    payload = {"model": model, "prompt": prompt, "stream": False}

    # Add system prompt if provided
    if system_prompt:
        payload["system"] = system_prompt

    # Handle image input
    image_path = None
    if image is not None:
        if isinstance(image, str):
            # Assume it's a file path
            if os.path.exists(image):
                image_path = image
                with open(image, "rb") as img_file:
                    img_bytes = img_file.read()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    payload["images"] = [img_b64]
            else:
                # Assume it's already base64 encoded
                payload["images"] = [image]
        elif isinstance(image, bytes):
            # Raw bytes, encode to base64
            img_b64 = base64.b64encode(image).decode("utf-8")
            payload["images"] = [img_b64]
    else:
        payload["images"] = []

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
        response.raise_for_status()

        response_text = response.json().get("response", "")

        # Log successful call
        log_ollama_call(
            prompt=prompt,
            model=model,
            image_path=image_path,
            response=response_text,
            success=True,
            timeout=timeout,
            log_dir=log_dir,
            system_prompt=system_prompt,
        )

        return response_text

    except Exception as e:
        error_msg = str(e)

        # Log failed call
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
        )

        return f"[⚠️] Ollama API failed: {error_msg}"
