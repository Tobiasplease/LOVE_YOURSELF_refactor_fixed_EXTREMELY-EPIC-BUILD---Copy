"""
utils/inference.py
------------------
Model query interface. Single backend since July 9 2026: the patched
llama-server (Qwen3.5, video super-frames, assistant prefill). The Ollama
backend + the separate text-side model (mistral-nemo) were retired — since
the llama.cpp migration, per-call model selection was ignored anyway (one
loaded model), so compression/reflection/drawing analysis were already
running on the main model in practice.

Also re-exports unload_model / reload_model for the ComfyUI VRAM dance.
"""

from typing import List, Optional, Union

from config import config as _cfg  # kept for MODEL_NAME/log defaults


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
    history: Optional[List[str]] = None,
    react: bool = False,
) -> str:
    """
    Query llama-server with a prompt and optional image.

    history: prior captions as the model's own assistant turns/prefill (the
    stream). `model` is a log label only — llama-server serves the one model
    it loaded (LLAMA_MODEL_PATH).
    """
    # Empty log_dir crashes log_llm_call (os.makedirs('')) — default it here
    # so call sites don't all have to pass it.
    if not log_dir:
        log_dir = _cfg.MOOD_SNAPSHOT_FOLDER

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
        history=history,
        react=react,
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
    history: Optional[List[str]] = None,
    react: bool = False,
) -> str:
    """Query with multiple video frames (super-frame or multi-image mode)."""
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
        history=history,
        react=react,
    )


# --- VRAM lifecycle (used by drawing.py) ---

def unload_model() -> None:
    """Free VRAM before ComfyUI generation."""
    from utils.llama_server import stop_server
    stop_server()


def reload_model() -> None:
    """Reload model into VRAM after ComfyUI is done."""
    from utils.llama_server import is_server_running, start_server
    if not is_server_running():
        start_server()
