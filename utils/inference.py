"""
utils/inference.py
------------------
Unified inference wrapper. All LLM call sites import query_model() from here.
Routes to either Ollama or llama-server based on config.INFERENCE_BACKEND.

For video-capable calls, use query_model_video() which sends multiple frames
through the super-frame pipeline when llama-server is active, or falls back
to single-frame via query_model() on Ollama.

Also re-exports start_server / stop_server for VRAM lifecycle management.
"""

from typing import List, Optional, Union

from config import config as _cfg


def query_model(
    prompt: str,
    model: str = "",
    image: Optional[Union[str, bytes]] = None,
    timeout: int = 30,
    log_dir: str = "",
    system_prompt: Optional[str] = None,
    strict_evaluation: bool = False,
    options: Optional[dict] = None,
    show_progress: bool = False,
    prompt_type: str = "general",
    skip_generation_wait: bool = False,
    prior_assistant_turn: Optional[str] = None,
) -> str:
    """
    Query the active inference backend with a prompt and optional image.
    Drop-in replacement for query_ollama() — same signature, routes by config.
    """
    if _cfg.INFERENCE_BACKEND == "llama_server":
        from utils.llama_server import query_llama_server
        return query_llama_server(
            prompt=prompt,
            model=model,
            image=image,
            timeout=timeout,
            log_dir=log_dir,
            system_prompt=system_prompt,
            strict_evaluation=strict_evaluation,
            options=options,
            show_progress=show_progress,
            prompt_type=prompt_type,
            skip_generation_wait=skip_generation_wait,
            prior_assistant_turn=prior_assistant_turn,
        )
    else:
        from utils.ollama import query_ollama
        return query_ollama(
            prompt=prompt,
            model=model,
            image=image,
            timeout=timeout,
            log_dir=log_dir,
            system_prompt=system_prompt,
            strict_evaluation=strict_evaluation,
            options=options,
            show_progress=show_progress,
            prompt_type=prompt_type,
            skip_generation_wait=skip_generation_wait,
            prior_assistant_turn=prior_assistant_turn,
        )


def query_model_video(
    prompt: str,
    frames: List[bytes],
    fps: float = 2.0,
    system_prompt: Optional[str] = None,
    options: Optional[dict] = None,
    timeout: int = 60,
    show_progress: bool = False,
    skip_generation_wait: bool = False,
) -> str:
    """
    Query with multiple video frames. Only works with llama-server backend.
    Falls back to single-frame (last frame) on Ollama.
    """
    if _cfg.INFERENCE_BACKEND == "llama_server":
        from utils.llama_server import query_llama_server_video
        return query_llama_server_video(
            prompt=prompt,
            frames=frames,
            fps=fps,
            system_prompt=system_prompt,
            options=options,
            timeout=timeout,
            show_progress=show_progress,
            skip_generation_wait=skip_generation_wait,
        )
    else:
        # Ollama fallback: use last frame as single image
        from utils.ollama import query_ollama
        last_frame = frames[-1] if frames else None
        return query_ollama(
            prompt=prompt,
            image=last_frame,
            system_prompt=system_prompt,
            options=options,
            timeout=timeout,
            show_progress=show_progress,
            skip_generation_wait=skip_generation_wait,
        )


# --- VRAM lifecycle (used by drawing.py) ---

def unload_model() -> None:
    """Free VRAM before ComfyUI generation."""
    if _cfg.INFERENCE_BACKEND == "llama_server":
        from utils.llama_server import stop_server
        stop_server()
    else:
        import requests
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "", "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass


def reload_model() -> None:
    """Reload model into VRAM after ComfyUI is done."""
    if _cfg.INFERENCE_BACKEND == "llama_server":
        from utils.llama_server import start_server, is_server_running
        if not is_server_running():
            start_server()
    else:
        from config.config import OLLAMA_MODEL
        import requests
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": "5m"},
                timeout=120,
            )
        except Exception:
            pass
