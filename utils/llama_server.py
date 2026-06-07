"""
utils/llama_server.py
---------------------
Drop-in replacement for utils/ollama.py that talks to llama-server directly.
Supports both single-image (regular captioning) and multi-frame video input
via the llama-video super-frame pipeline.

Usage:
    # Single image (same API as query_ollama):
    from utils.llama_server import query_llama_server
    result = query_llama_server(prompt="Describe this.", image="/path/to/frame.jpg")

    # Multi-frame video (new capability):
    from utils.llama_server import query_llama_server_video
    result = query_llama_server_video(
        prompt="What happened in this sequence?",
        frames=[frame1_bytes, frame2_bytes, ...],
        fps=2.0,
        system_prompt="You are a drawing machine..."
    )

    # VRAM management:
    from utils.llama_server import start_server, stop_server
    stop_server()   # before ComfyUI
    start_server()  # after ComfyUI
"""

import base64
import json
import os
import subprocess
import time
from typing import List, Optional, Union

import requests

from config.config import MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import LogType, log_json_entry
from utils.progress_bar import ProgressBar

# Server configuration
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", os.path.expanduser("~/llama.cpp/build/bin/llama-server"))
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", os.path.expanduser("~/models/qwen3.5-9b/Qwen3.5-9B-Q5_K_M.gguf"))
LLAMA_MMPROJ_PATH = os.getenv("LLAMA_MMPROJ_PATH", os.path.expanduser("~/models/qwen3.5-9b/mmproj-F16.gguf"))
LLAMA_CTX_SIZE = int(os.getenv("LLAMA_CTX_SIZE", "65536"))
LLAMA_GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "99"))

SHOW_PROGRESS = os.getenv("LLAMA_SHOW_PROGRESS", "true").lower() == "true"

_server_process = None

# ---------------------------------------------------------------------------
# Logging (reuse existing infrastructure)
# ---------------------------------------------------------------------------

try:
    from utils.ollama import log_ollama_call
except ImportError:
    def log_ollama_call(**kwargs):
        pass


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_server(model_path: str = None, mmproj_path: str = None, ctx_size: int = None) -> bool:
    """Start llama-server as a subprocess. Returns True if started successfully."""
    global _server_process

    if _server_process and _server_process.poll() is None:
        print("[llama-server] Already running")
        return True

    model = model_path or LLAMA_MODEL_PATH
    mmproj = mmproj_path or LLAMA_MMPROJ_PATH
    ctx = ctx_size or LLAMA_CTX_SIZE

    if not model or not os.path.exists(model):
        print(f"[llama-server] Model not found: {model}")
        return False

    cmd = [
        LLAMA_SERVER_BIN,
        "-m", model,
        "--host", "0.0.0.0",
        "--port", "8080",
        "--ctx-size", str(ctx),
        "-ngl", str(LLAMA_GPU_LAYERS),
        "--jinja",
    ]
    if mmproj and os.path.exists(mmproj):
        cmd.extend(["--mmproj", mmproj])

    print(f"[llama-server] Starting: {' '.join(cmd)}")
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for i in range(60):
        try:
            resp = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if resp.ok:
                print(f"[llama-server] Ready after {i + 1}s")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)

    print("[llama-server] Failed to start within 60s")
    return False


def stop_server() -> None:
    """Stop the llama-server process to free VRAM."""
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        print("[llama-server] Stopped (VRAM freed)")
    _server_process = None


def is_server_running() -> bool:
    """Check if llama-server is responding."""
    try:
        resp = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
        return resp.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Drawing completion wait (mirrors utils/ollama._wait_for_drawing_completion)
# ---------------------------------------------------------------------------

def _wait_for_drawing_completion() -> None:
    """Wait for ComfyUI generation to complete before making LLM calls."""
    try:
        from utils.state_manager import state_manager
    except ImportError:
        return

    if not state_manager.is_generating_drawing:
        return

    log_json_entry(
        LogType.INFO,
        {"message": "llama-server call paused - waiting for ComfyUI generation"},
        print_message="llama-server paused - ComfyUI generating",
    )

    while state_manager.is_generating_drawing:
        time.sleep(1.0)

    log_json_entry(
        LogType.INFO,
        {"message": "ComfyUI generation completed - resuming llama-server calls"},
        print_message="llama-server resumed",
    )

    # Free ComfyUI VRAM
    try:
        resp = requests.post(
            "http://localhost:8188/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
        if resp.ok:
            print("[llama-server] Freed ComfyUI VRAM")
    except Exception:
        pass

    # Restart llama-server if it was stopped for VRAM
    if not is_server_running():
        print("[llama-server] Restarting after ComfyUI...")
        start_server()


# ---------------------------------------------------------------------------
# Single-image query (drop-in for query_ollama)
# ---------------------------------------------------------------------------

def query_llama_server(
    prompt: str,
    model: str = "",
    image: Optional[Union[str, bytes]] = None,
    timeout: int = 30,
    log_dir: str = MOOD_SNAPSHOT_FOLDER,
    system_prompt: Optional[str] = None,
    strict_evaluation: bool = False,
    options: Optional[dict] = None,
    show_progress: bool = SHOW_PROGRESS,
    prompt_type: str = "general",
    skip_generation_wait: bool = False,
    prior_assistant_turn: Optional[str] = None,
) -> str:
    """
    Query llama-server with a prompt and optional image.
    API-compatible with query_ollama() for easy migration.
    """
    if not skip_generation_wait:
        _wait_for_drawing_completion()

    # Encode image
    img_b64 = None
    image_path = None
    if image is not None:
        if isinstance(image, str):
            if os.path.exists(image):
                image_path = image
                with open(image, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
            else:
                img_b64 = image
        elif isinstance(image, bytes):
            img_b64 = base64.b64encode(image).decode("utf-8")

    # Build messages (OpenAI chat format)
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    if prior_assistant_turn:
        messages.append({"role": "user", "content": "..."})
        prior_clean = prior_assistant_turn.strip()
        sent_end = min(
            (prior_clean.find(c) for c in ".?!" if prior_clean.find(c) > 8),
            default=-1,
        )
        prior_anchor = prior_clean[: sent_end + 1] if sent_end > 0 else prior_clean[:80]
        messages.append({"role": "assistant", "content": prior_anchor})

    if img_b64:
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": prompt})

    # Build payload
    payload = {
        "messages": messages,
        "stream": show_progress,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    # Map generation options
    if options:
        payload["temperature"] = options.get("temperature", 0.8)
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
        if "num_predict" in options or "max_tokens" in options:
            payload["max_tokens"] = options.get("max_tokens", options.get("num_predict", 60))
        if "repeat_penalty" in options:
            payload["repeat_penalty"] = options["repeat_penalty"]
        if "seed" in options:
            payload["seed"] = options["seed"]
    elif strict_evaluation:
        payload["temperature"] = 0.1
        payload["top_p"] = 0.8

    endpoint = f"{LLAMA_SERVER_URL}/v1/chat/completions"

    try:
        progress_bar = None
        if show_progress:
            progress_bar = ProgressBar(description="")
            progress_bar.start()

            response = requests.post(endpoint, json=payload, timeout=timeout, stream=True)
            response.raise_for_status()

            response_text = ""
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                if line_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(line_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""
                    response_text += content
                except json.JSONDecodeError:
                    continue

            if progress_bar:
                progress_bar.stop(success=True)
        else:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        log_ollama_call(
            prompt=prompt,
            model=model or "llama-server",
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
            model=model or "llama-server",
            image_path=image_path,
            response=None,
            success=False,
            error_message=error_msg,
            timeout=timeout,
            log_dir=log_dir,
            system_prompt=system_prompt,
            prompt_type=prompt_type,
        )

        return f"[WARNING] llama-server API failed: {error_msg}"


# ---------------------------------------------------------------------------
# Multi-frame video query (new capability)
# ---------------------------------------------------------------------------

def _query_multi_image(
    prompt: str,
    frames: List[bytes],
    system_prompt: Optional[str] = None,
    options: Optional[dict] = None,
    timeout: int = 60,
    show_progress: bool = SHOW_PROGRESS,
) -> str:
    """Plain multi-image mode: send each frame as a separate image_url in the content array.
    The model sees them as independent images but can still infer temporal change.
    ~1200 vision tokens for 4-6 frames.
    """
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    # Build user content with interleaved images + final text prompt
    user_content = []
    for frame_bytes in frames:
        img_b64 = base64.b64encode(frame_bytes).decode("utf-8")
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "messages": messages,
        "stream": show_progress,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if options:
        payload["temperature"] = options.get("temperature", 0.8)
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
        if "num_predict" in options or "max_tokens" in options:
            payload["max_tokens"] = options.get("max_tokens", options.get("num_predict", 60))
        if "repeat_penalty" in options:
            payload["repeat_penalty"] = options["repeat_penalty"]

    endpoint = f"{LLAMA_SERVER_URL}/v1/chat/completions"

    progress_bar = None
    if show_progress:
        progress_bar = ProgressBar(description="")
        progress_bar.start()
        response = requests.post(endpoint, json=payload, timeout=timeout, stream=True)
        response.raise_for_status()
        response_text = ""
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                line_str = line_str[6:]
            if line_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or ""
                response_text += content
            except json.JSONDecodeError:
                continue
        progress_bar.stop(success=True)
    else:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return response_text


def _query_superframe(
    prompt: str,
    frames: List[bytes],
    fps: float = 2.0,
    system_prompt: Optional[str] = None,
    options: Optional[dict] = None,
    timeout: int = 60,
) -> str:
    """Super-frame mode: Conv3D paired frames + M-RoPE temporal encoding.
    Genuine temporal perception — the model sees continuous motion.
    ~600 vision tokens for 4-6 frames (more compressed).
    Requires llama-video package.
    """
    from llama_video import Preprocessor, Settings
    from llama_video.client import LlamaServerClient
    from llama_video.types import Frame
    import cv2
    import numpy as np
    import asyncio

    settings = Settings()
    preprocessor = Preprocessor(settings.model)

    temp_frames = []
    for i, frame_bytes in enumerate(frames):
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            # llama_video expects Frame objects with RGB numpy data
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            temp_frames.append(Frame(
                data=rgb,
                index=i,
                timestamp=i / fps,
                width=w,
                height=h,
            ))

    if not temp_frames:
        return "[WARNING] No valid frames to process"

    video_input = preprocessor.process(temp_frames, fps=fps)

    client = LlamaServerClient(settings.server)

    # Prepend system prompt to user prompt since caption_video doesn't support it
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    async def _caption():
        try:
            return await client.caption_video(
                video_input,
                prompt=full_prompt,
                temperature=options.get("temperature", 0.9) if options else 0.9,
            )
        finally:
            await client.close()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_caption())
    finally:
        loop.close()

    return result


def query_llama_server_video(
    prompt: str,
    frames: List[bytes],
    fps: float = 2.0,
    system_prompt: Optional[str] = None,
    options: Optional[dict] = None,
    timeout: int = 60,
    show_progress: bool = SHOW_PROGRESS,
    skip_generation_wait: bool = False,
    mode: str = "",
) -> str:
    """
    Query llama-server with multiple video frames.

    Routes to either:
    - "multi": Plain multi-image (separate image_url entries, ~1200 vision tokens)
    - "superframe": Conv3D super-frames via llama-video (~600 vision tokens, richer temporal)

    Args:
        prompt: Text prompt
        frames: List of JPEG-encoded frame bytes (chronological order)
        fps: Original capture FPS of the frames
        system_prompt: System prompt
        options: Generation options
        timeout: Request timeout
        mode: "multi" or "superframe" (empty = use config.VIDEO_MODE)
    """
    if not skip_generation_wait:
        _wait_for_drawing_completion()

    if not mode:
        from config.config import VIDEO_MODE
        mode = VIDEO_MODE

    if mode == "superframe":
        try:
            return _query_superframe(
                prompt=prompt,
                frames=frames,
                fps=fps,
                system_prompt=system_prompt,
                options=options,
                timeout=timeout,
            )
        except ImportError:
            print("[llama-server] llama-video not installed, falling back to multi-image mode")
            mode = "multi"
        except Exception as e:
            print(f"[llama-server] Super-frame failed ({e}), falling back to multi-image")
            mode = "multi"

    if mode == "multi":
        try:
            return _query_multi_image(
                prompt=prompt,
                frames=frames,
                system_prompt=system_prompt,
                options=options,
                timeout=timeout,
                show_progress=show_progress,
            )
        except Exception as e:
            error_msg = str(e)
            print(f"[llama-server] Multi-image failed: {error_msg}")

    # Final fallback: single last frame
    if frames:
        print("[llama-server] Falling back to single-frame caption")
        return query_llama_server(
            prompt=prompt,
            image=frames[-1],
            system_prompt=system_prompt,
            options=options,
            timeout=timeout,
            show_progress=show_progress,
            skip_generation_wait=True,
        )
    return "[WARNING] No frames provided"
