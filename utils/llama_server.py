"""
utils/llama_server.py
---------------------
THE inference backend (sole backend since July 9 2026 — utils/ollama.py
retired). Talks to the patched llama-server directly: single-image captioning,
multi-frame video via the llama-video super-frame pipeline, and document-mode
assistant prefill (the stream).

Usage:
    # Single image:
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
import re
import subprocess
import time
from typing import List, Optional, Union

import requests

from config.config import MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import LogType, log_json_entry
from utils.progress_bar import ProgressBar

# Server configuration
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
# DEFAULTS ARE THE 27B HYBRID STACK (Aug 5) — the configuration we actually
# landed on, so a bare `python machine.py` runs it. These used to point at the
# 9B, and the live setup existed only inside run_27b.sh: which launcher you
# reached for decided what you measured. NOTE the binary — the older
# ~/llama.cpp build predates the MTP head and cannot load this model at all
# ("missing tensor blk.64.ssm_conv1d.weight"). The 9B remains one env away
# (see run_9b.sh) for A/B.
LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", os.path.expanduser("~/llama.cpp-27b/build/bin/llama-server"))
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", os.path.expanduser("~/models/qwen3.6-27b-mtp/Qwen3.6-27B-Q4_K_M.gguf"))
LLAMA_MMPROJ_PATH = os.getenv("LLAMA_MMPROJ_PATH", os.path.expanduser("~/models/qwen3.6-27b-mtp/mmproj-F16.gguf"))
LLAMA_CTX_SIZE = int(os.getenv("LLAMA_CTX_SIZE", "16384"))
LLAMA_GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "99"))

SHOW_PROGRESS = os.getenv("LLAMA_SHOW_PROGRESS", "true").lower() == "true"

# Sampler keys forwarded verbatim to llama-server beyond the basic temp/top_p/
# repeat_penalty. DRY (anti-repetition over SEQUENCES, not single tokens) is the
# only thing that stops the model reproducing a whole prior caption from the
# stream — plain repeat_penalty only looks back repeat_last_n=64 tokens by
# default, far short of the stream's length, so a verbatim prior caption sits
# outside its window entirely.
_SAMPLER_PASSTHROUGH = (
    "repeat_last_n",
    "min_p",
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "dry_penalty_last_n",
)


def _forward_sampler_options(payload: dict, options: dict) -> None:
    for k in _SAMPLER_PASSTHROUGH:
        if k in options:
            payload[k] = options[k]


_server_process = None

# Wedge watchdog (July 30): the 27B server can hang mid-generation while
# /health still answers ok — observed 3x in one session, always on
# multi-image calls (suspected mtmd+MTP interaction). is_server_running()
# cannot see this failure class, so the machine sat timing out until an
# operator killed the process. Consecutive read-timeouts are the only
# signal: two in a row force a full stop/start. Connection-refused errors
# don't count (that's the drawing handoff's intentional unload).
import threading as _threading

_timeout_streak = 0
_recovery_lock = _threading.Lock()

# ---------------------------------------------------------------------------
# The inference gate — ONE REQUEST AT A TIME (Aug 3, twice corrected Aug 5)
# ---------------------------------------------------------------------------
# Two wrong turns are recorded here because both were plausible and neither
# survived measurement.
#
# First I assumed one server slot and built an exclusive lock, believing a
# caption could queue behind a reflection and burn its timeout. Queue waits
# measured 0.0s; the wedges were a 64KB stderr pipe nobody drained.
#
# Then the server's own log said "n_slots = 4, n_ctx_slot = 16384" and I read
# that as 4x capacity, so I opened it to three concurrent. Also wrong: slots
# are concurrent SESSIONS, but a single GPU still serialises the compute. The
# log settles it — a 272-token request took 27.9s while a 4810-token one took
# 8.9s. The small one was not slow, it was starved by the requests beside it.
#
# So: one at a time. Every call then runs at its true speed (3-7s) and nothing
# comes near the 60s timeout. Realtime calls (captions, memory) SKIP rather
# than queue, which is free — they are periodic, and the cadence already skips
# constantly. Background work waits its turn. The starvation this caused after
# a drawing is fixed at the source instead: that burst is smaller now the
# critique is gone.
INFERENCE_CONCURRENCY = int(os.getenv("INFERENCE_CONCURRENCY", "1"))
_inference_sem = _threading.BoundedSemaphore(INFERENCE_CONCURRENCY)
_inflight_lock = _threading.Lock()
_inflight = []  # what is running, for logging


def _acquire_inference(prompt_type: str, wait: bool):
    """Take a slot. Returns (acquired, seconds_waited)."""
    t0 = time.time()
    got = _inference_sem.acquire(blocking=wait, timeout=30 if wait else None) if wait else _inference_sem.acquire(blocking=False)
    if got:
        with _inflight_lock:
            _inflight.append(prompt_type)
    return got, time.time() - t0


def _release_inference(prompt_type: str = ""):
    with _inflight_lock:
        if prompt_type and prompt_type in _inflight:
            _inflight.remove(prompt_type)
        elif _inflight:
            _inflight.pop()
    try:
        _inference_sem.release()
    except ValueError:
        pass


def _is_realtime(prompt_type: str) -> bool:
    """Calls that are better skipped than queued."""
    return prompt_type in ("caption", "caption_blind", "memory")


def busy_with() -> Optional[str]:
    with _inflight_lock:
        return ", ".join(_inflight) if _inflight else None


def _note_query_outcome(error_msg: Optional[str] = None) -> None:
    """Call with None on success, the error string on failure."""
    global _timeout_streak
    if error_msg is None:
        _timeout_streak = 0
        return
    if "timed out" not in error_msg.lower():
        return
    _timeout_streak += 1
    if _timeout_streak < 2:
        return
    if not _recovery_lock.acquire(blocking=False):
        return  # another thread is already recovering
    try:
        _timeout_streak = 0
        print("[llama-server] WEDGED — 2 consecutive read-timeouts with health ok; forcing restart")
        stop_server()
        ensure_server_up()
    finally:
        _recovery_lock.release()


# ---------------------------------------------------------------------------
# Logging (reuse existing infrastructure)
# ---------------------------------------------------------------------------

from utils.llm_log import log_llm_call


# ---------------------------------------------------------------------------
# The stream: how prior captions reach the model (docs/continuity-plan.md)
# ---------------------------------------------------------------------------

# Qwen3.5 emits empty <think></think> blocks even with enable_thinking=false —
# and not only leading ones: when it regurgitates a poisoned document it can
# emit several (a ^-anchored sub only removed the first; the survivor entered
# the stream and bred). Strip ALL blocks plus any dangling unclosed tag.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL)


def _stream_mode() -> str:
    from config import config as _c

    return getattr(_c, "STREAM_MODE", "turns")


# Hybrid seam: the log lines arrive stamped ("14:02 — ..."); the prefill must
# be raw voice, or the model continues a timestamp instead of a thought.
_LOG_STAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*[—–-]\s*")


def _hybrid_prefill_chars() -> int:
    from config import config as _c

    return int(getattr(_c, "HYBRID_PREFILL_CHARS", 220))


# Sentence boundary, used to cut the seam at a coherent joint.
_SENTENCE_END_RE = re.compile(r"[.!?…]['\")\]]?\s+")


def _seam_of(entry: str, budget: int) -> str:
    """The tail of `entry` to hand back as continuation prefill.

    CUT AT A SENTENCE BOUNDARY (Aug 1). A raw char-slice starts mid-word, and
    a model handed a partial assistant turn that begins mid-word does not
    continue it — it RE-TYPES the passage properly from an earlier point and
    appends a few words. Measured: the model reproduced a whole prior entry
    and added "It's good.", the exact-prefix stripper could not match, and the
    re-typed text was stored as a mid-word slice ("rocessing over there at his
    screens..."), which then became the next seam. That is the "very truncated"
    feed: each entry a shifted window of the same passage.

    Starting at the last sentence boundary keeps the seam a coherent unit. If
    the previous thought was itself cut off mid-sentence, the seam is that
    partial sentence — so genuine mid-clause continuation still happens, it is
    just never mid-WORD.
    """
    entry = (entry or "").strip()
    if not entry:
        return ""
    marks = [m.end() for m in _SENTENCE_END_RE.finditer(entry)]
    seam = entry[marks[-1] :] if marks else entry
    if not seam.strip():  # entry ended exactly on a boundary — keep its final sentence
        seam = entry[marks[-2] :] if len(marks) > 1 else entry
    if len(seam) > budget:
        seam = seam[-budget:]
        cut = seam.find(" ")
        if cut > 0:
            seam = seam[cut + 1 :]
    return seam.strip()


def _trim_to_sentence(text: str) -> str:
    """Cut to the last complete sentence (keeps text intact if none found)."""
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"), text.rfind("…"))
    return text[: cut + 1] if cut > 20 else text


def _document_prefill(history: Optional[List[str]]) -> str:
    """Join the stream into one flowing monologue text for assistant prefill.

    Older entries are trimmed to complete sentences — token-budget cuts left
    mid-sentence breaks INSIDE the document, which the model then repaired in
    assistant register ("(Note: the final line was cut off mid-sentence. I
    will continue from where I left off.)", July 9). The NEWEST entry stays
    raw: a mid-sentence tail is the good kind of unfinished — the model picks
    it up and completes it (the original document-mode magic).

    Ends with a single space (not a paragraph break) so the model continues
    the same flow instead of opening a fresh, list-shaped item.
    """
    parts = [p for p in ((h or "").strip() for h in history or []) if p]
    if not parts:
        return ""
    parts = [_trim_to_sentence(p) for p in parts[:-1]] + [parts[-1]]
    return " ".join(p for p in parts if p) + " "


def _append_stream_and_user(messages: list, history: Optional[List[str]], user_message: dict, react: bool = False) -> str:
    """Append the stream + current user message per STREAM_MODE.

    "document": user message first, then the monologue-so-far as ONE trailing
    assistant message — llama-server continues it (assistant prefill; the
    payload must set enable_thinking=false or the server rejects the request).
    The model's next tokens continue its own thought.

    react=True (salience hot — arrival, eye contact, scene motion): the shape
    flips. The monologue (truncated to the last two thoughts) comes FIRST and
    the event+frames come LAST, so the model answers the moment instead of
    continuing past it. Attention capture truncates rehearsal; the reaction is
    stored into the stream, so the interruption becomes part of the document.

    "world" (July 26, the inversion): the stream rides as ONE assistant
    message of timestamped log lines (formatted upstream by the captioner),
    and the user message — frames + the world's turn — always comes LAST.
    Generation begins right after the present, never after the machine's own
    prose: every call answers the world instead of extending an essay. The
    react ordering made the default; react needs no special case here,
    salience only changes how much interior material rides in the user prompt.

    "hybrid" (Aug 1): world ORDERING with a document SEAM. The older log rides
    as an assistant message, the world's turn (frames + present) comes next —
    so perception stays last and grounding is preserved — and then the tail of
    the machine's own latest thought is appended as a SHORT assistant prefill,
    so generation begins inside its own voice mid-turn instead of composing a
    fresh caption. Motivation (measured, Aug 1 world run): perception-last
    maximised grounding AND the image-captioning prior — 72-76% of entries
    opened "The ___", 69% carried a semicolon, length pinned at 37-61 words.
    You cannot open with "The" when you are handed your own unfinished clause.
    Poison exposure stays bounded: the prefill is ONE short tail, never the
    whole document (the Aug 1 deadlock came from prefilling a poisoned
    document 187 times). react drops the prefill entirely — an interruption
    should be answered, not continued.

    "turns": legacy turn-pairs, then the user message.

    Returns the prefill text ("" in world/turns/react modes) for seam cleaning + logging.
    """
    if _stream_mode() == "hybrid":
        lines = [p for p in ((h or "").strip() for h in history or []) if p]
        prefill = ""
        if lines and not react:
            tail = _LOG_STAMP_RE.sub("", lines.pop())  # newest entry leaves the log, becomes the seam
            seam = _seam_of(tail, _hybrid_prefill_chars())
            if seam:
                prefill = seam + " "
        if lines:
            messages.append({"role": "assistant", "content": "\n".join(lines)})
        messages.append(user_message)
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        return prefill
    if _stream_mode() == "world":
        lines = [p for p in ((h or "").strip() for h in history or []) if p]
        if lines:
            messages.append({"role": "assistant", "content": "\n".join(lines)})
        messages.append(user_message)
        return ""
    if _stream_mode() == "document":
        if react:
            recent = [p for p in ((h or "").strip() for h in history or []) if p][-2:]
            if recent:
                messages.append({"role": "assistant", "content": " ".join(recent)})
            messages.append(user_message)
            return ""
        prefill = _document_prefill(history)
        messages.append(user_message)
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        return prefill
    if history:
        for past in history:
            past = (past or "").strip()
            if past:
                messages.append({"role": "user", "content": "..."})
                messages.append({"role": "assistant", "content": past})
    messages.append(user_message)
    return ""


def _clean_continuation(text: str, prefill: str = "") -> str:
    """Strip ALL think blocks (and a dangling unclosed one), then any verbatim
    re-typing of the prefill seam (the model occasionally re-says the tail it
    was continuing)."""
    text = _THINK_RE.sub("", text or "")
    text = _THINK_OPEN_RE.sub("", text).strip()
    if prefill:
        tail = prefill.rstrip()
        for n in range(min(len(tail), 120), 11, -1):
            if text.startswith(tail[-n:]):
                text = text[n:].lstrip()
                break
        else:
            # RE-TYPING (Aug 1): the model doesn't always resume at the seam —
            # it reproduces the passage from an earlier point and appends its
            # new words at the END ("...And him working...  It's good."). The
            # prefix test above can't see that, so the whole re-typed passage
            # was stored as a caption, arriving as a mid-word slice and then
            # becoming the next seam — the shifted-window feed that read as
            # "very truncated". If the seam's own ending appears anywhere in
            # the output, everything up to and including it is the model
            # catching up, and only what follows is new.
            key = tail[-40:]
            if len(key) >= 12:
                idx = text.find(key)
                if idx >= 0:
                    text = text[idx + len(key) :].lstrip()
    return text


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
        "-m",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--ctx-size",
        str(ctx),
        "-ngl",
        str(LLAMA_GPU_LAYERS),
        "--jinja",
    ]
    if mmproj and os.path.exists(mmproj):
        cmd.extend(["--mmproj", mmproj])
    # Extra launch flags via env (July 28) — the Qwen3.6-27B experiment needs
    # MTP speculative decoding ("--spec-type draft-mtp --spec-draft-n-max 2
    # -fa on") without hardcoding model-specific flags here. See run_27b.sh.
    # -fa on by default with the 27B stack (run_27b.sh set it explicitly before
    # the defaults moved). MTP speculative decoding stays OFF: it is a variable
    # the 9B never had and the voice question is a distribution question —
    # LLAMA_MTP=1 ./run_27b.sh re-enables it for the ~1.7x decode speed.
    extra = os.getenv("LLAMA_EXTRA_ARGS", "-fa on").split()
    if extra:
        cmd.extend(extra)

    print(f"[llama-server] Starting: {' '.join(cmd)}")
    # STDERR GOES TO A FILE, NEVER A PIPE (Aug 5). It was subprocess.PIPE and
    # nothing in this codebase ever read it — a latent deadlock with a fuse:
    # the pipe buffer is 64KB, llama-server logs every request, and once it
    # fills, the writing thread BLOCKS in write(). Generation stops; /health
    # keeps answering because that is a different thread. That is exactly the
    # failure the wedge watchdog was written for ("hangs mid-generation while
    # /health still answers ok"), and exactly the shape the artist described:
    # works for a while, then hangs forever, restart fixes it. Time-to-wedge is
    # however long it takes to emit 64KB of logs — which is why multi-image
    # calls, being the chattiest, appeared to be the culprit.
    # A file has no such limit, and it also gives us the server's own account
    # of what happened, which we have wanted several times this week.
    log_path = os.path.join(MOOD_SNAPSHOT_FOLDER, "llama_server.log")
    try:
        os.makedirs(MOOD_SNAPSHOT_FOLDER, exist_ok=True)
        if os.path.exists(log_path) and os.path.getsize(log_path) > 50 * 1024 * 1024:
            os.replace(log_path, log_path + ".1")  # one generation of rotation is plenty
        _server_log = open(log_path, "ab", buffering=0)
    except Exception:
        _server_log = subprocess.DEVNULL  # never let logging stop the machine
    _server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=_server_log,
    )

    # Wait for the server to be ready. A big model on -ngl loads for many
    # seconds — longer on a contended GPU — and the poll is silent, so a slow
    # load looks exactly like a freeze. Print a heartbeat, bail fast if the
    # subprocess actually died (port in use / OOM), and allow up to 150s.
    for i in range(150):
        try:
            resp = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if resp.ok:
                print(f"[llama-server] Ready after {i + 1}s")
                return True
        except requests.ConnectionError:
            pass
        if _server_process and _server_process.poll() is not None:
            print("[llama-server] process exited during load — check stderr (port in use? VRAM OOM?)")
            return False
        if i and i % 10 == 0:
            print(f"[llama-server] still loading model... ({i}s)")
        time.sleep(1)

    print("[llama-server] Failed to become ready within 150s")
    return False


def stop_server() -> None:
    """Stop llama-server to free VRAM — by process pattern, not just our own
    handle. After a machine.py restart the running server is an adopted
    orphan (started by a previous process; no handle), and terminating
    nothing silently left ~10GB on the GPU: ComfyUI OOM'd on every drawing
    until the 5-minute timeout (July 9)."""
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
    _server_process = None

    # Orphan sweep: anything still answering the port is holding the VRAM
    # this call exists to free
    if is_server_running():
        try:
            subprocess.run(["pkill", "-f", LLAMA_SERVER_BIN], timeout=10)
            for _ in range(20):
                if not is_server_running():
                    break
                time.sleep(0.5)
        except Exception as e:
            print(f"[llama-server] Orphan kill failed: {e}")

    if is_server_running():
        print("[llama-server] WARNING: a server is still responding after stop — VRAM not freed")
    else:
        print("[llama-server] Stopped (VRAM freed)")


def is_server_running() -> bool:
    """Check if llama-server is responding."""
    try:
        resp = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
        return resp.ok
    except Exception:
        return False


def _free_comfyui_vram() -> None:
    """Ask ComfyUI to release its models/VRAM. Harmless (caught) if ComfyUI
    isn't running. Flux can linger on the GPU after a generation and starve the
    -ngl load, so this must run before restarting llama-server."""
    try:
        requests.post(
            "http://localhost:8188/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
    except Exception:
        pass


def ensure_server_up() -> bool:
    """Bring llama-server back up robustly: free ComfyUI VRAM first, then start,
    retrying once. The bare start_server() failed 'during GRBL execution' — a
    query hit a down server while Flux still held the VRAM, so -ngl couldn't
    allocate and the 60s health check timed out."""
    if is_server_running():
        return True
    _free_comfyui_vram()
    for attempt in range(2):
        if start_server():
            return True
        _free_comfyui_vram()
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Drawing completion wait
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

    # Restart llama-server if it was stopped for VRAM (frees ComfyUI VRAM first).
    if not is_server_running():
        print("[llama-server] Restarting after ComfyUI...")
        ensure_server_up()


# ---------------------------------------------------------------------------
# Single-image query
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
    history: Optional[List[str]] = None,
    react: bool = False,
) -> str:
    """
    Query llama-server with a prompt and optional image.

    history: prior outputs of the same voice, included as the model's own
    assistant turns — each call then CONTINUES a visible stream of thought
    instead of restarting one (CoT-style continuity). Text only; past
    images are never re-sent.
    """
    if not skip_generation_wait:
        _wait_for_drawing_completion()

    # Auto-restart if llama-server has crashed. ensure_server_up frees ComfyUI
    # VRAM first + retries — a bare start_server() failed during the GRBL draw
    # phase when Flux still held the GPU.
    if not is_server_running():
        print("[llama-server] Server not responding — attempting restart...")
        if not ensure_server_up():
            return f"[WARNING] llama-server unavailable and restart failed"

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

    # prior_assistant_turn: a single anchor thought (turns mode only — in
    # document mode the prefill IS the prior thought, so the anchor is skipped)
    if prior_assistant_turn and _stream_mode() != "document":
        messages.append({"role": "user", "content": "..."})
        prior_clean = prior_assistant_turn.strip()
        sent_end = min(
            (prior_clean.find(c) for c in ".?!" if prior_clean.find(c) > 8),
            default=-1,
        )
        prior_anchor = prior_clean[: sent_end + 1] if sent_end > 0 else prior_clean[:80]
        messages.append({"role": "assistant", "content": prior_anchor})

    if img_b64:
        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    else:
        user_message = {"role": "user", "content": prompt}

    # In document mode, prior_assistant_turn doubles as the stream when no
    # history was passed (memory-mode calls) — continuity either way.
    effective_history = history or ([prior_assistant_turn] if prior_assistant_turn else None)
    prefill = _append_stream_and_user(messages, effective_history, user_message, react=react)

    # Build payload
    payload = {
        "messages": messages,
        "stream": show_progress,
        "cache_prompt": True,  # the stream is a stable prefix — reuse the KV cache
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
        _forward_sampler_options(payload, options)
    elif strict_evaluation:
        payload["temperature"] = 0.1
        payload["top_p"] = 0.8

    endpoint = f"{LLAMA_SERVER_URL}/v1/chat/completions"

    # THE GATE. Realtime calls skip rather than queue: a caption waiting behind
    # a reflection burns its own timeout and then looks like a hung server.
    _rt = _is_realtime(prompt_type)
    _got, _queued = _acquire_inference(prompt_type, wait=not _rt)
    if not _got:
        print(f"[llama-server] {prompt_type} skipped — busy with {busy_with()}")
        return ""
    _t0 = time.time()

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

        response_text = _clean_continuation(response_text, prefill)

        log_llm_call(
            duration_s=time.time() - _t0,
            queued_s=_queued,
            prompt=prompt,
            model=model or "llama-server",
            image_path=image_path,
            response=response_text,
            success=True,
            timeout=timeout,
            log_dir=log_dir,
            system_prompt=system_prompt,
            prompt_type=prompt_type,
            api_endpoint=endpoint,
            history_len=len(effective_history or []),
            stream_mode=(_stream_mode() + ("-react" if react else "")) if effective_history else None,
            prefill_tail=prefill[-150:] if prefill else None,
        )

        _note_query_outcome(None)
        return response_text

    except Exception as e:
        error_msg = str(e)
        _note_query_outcome(error_msg)
        if progress_bar:
            progress_bar.stop(success=False)

        log_llm_call(
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
            api_endpoint=endpoint,
            history_len=len(effective_history or []),
            stream_mode=(_stream_mode() + ("-react" if react else "")) if effective_history else None,
        )

        return f"[WARNING] llama-server API failed: {error_msg}"

    finally:
        # The gate must open on every path — a leaked slot would silence the
        # machine permanently, which is worse than anything it protects against.
        _release_inference(prompt_type)


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
    history: Optional[List[str]] = None,
    react: bool = False,
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

    # The stream: prior captions, per STREAM_MODE (document prefill or turn-pairs)
    prefill = _append_stream_and_user(messages, history, {"role": "user", "content": user_content}, react=react)

    payload = {
        "messages": messages,
        "stream": show_progress,
        "cache_prompt": True,
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
        _forward_sampler_options(payload, options)

    _vt0 = time.time()
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

    response_text = _clean_continuation(response_text, prefill)

    log_llm_call(
        prompt=prompt,
        model="llama-server",
        response=response_text,
        success=True,
        timeout=timeout,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=system_prompt,
        prompt_type="caption",
        api_endpoint=endpoint,
        history_len=len(history or []),
        stream_mode=(_stream_mode() + ("-react" if react else "")) if history else None,
        num_frames=len(frames),
        prefill_tail=prefill[-150:] if prefill else None,
        duration_s=time.time() - _vt0,
    )

    _note_query_outcome(None)
    return response_text


def _query_superframe(
    prompt: str,
    frames: List[bytes],
    fps: float = 2.0,
    system_prompt: Optional[str] = None,
    options: Optional[dict] = None,
    timeout: int = 60,
    show_progress: bool = SHOW_PROGRESS,
    history: Optional[List[str]] = None,
    react: bool = False,
) -> str:
    """Super-frame mode: Conv3D paired frames + M-RoPE temporal encoding.
    Genuine temporal perception — the model sees continuous motion.
    ~600 vision tokens for 4-6 frames (more compressed).
    Requires llama-video package.

    Bypasses llama_video's LlamaServerClient.caption_video() to:
    - Add chat_template_kwargs.enable_thinking=false (prevents CoT dump)
    - Support system prompt as a proper message role
    - Use our streaming infrastructure
    """
    from llama_video import Preprocessor, Settings
    from llama_video.client import LlamaServerClient
    from llama_video.types import Frame
    import cv2
    import numpy as np
    import io
    from PIL import Image

    settings = Settings()
    preprocessor = Preprocessor(settings.model)

    temp_frames = []
    for i, frame_bytes in enumerate(frames):
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            temp_frames.append(
                Frame(
                    data=rgb,
                    index=i,
                    timestamp=i / fps,
                    width=w,
                    height=h,
                )
            )

    if not temp_frames:
        return "[WARNING] No valid frames to process"

    video_input = preprocessor.process(temp_frames, fps=fps)

    # Build the request ourselves instead of using caption_video()
    # Use _build_video_message for the image content, then add our own payload keys
    client = LlamaServerClient(settings.server)
    user_message = client._build_video_message(video_input, prompt)

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    # The stream: prior captions, per STREAM_MODE (document prefill or turn-pairs)
    prefill = _append_stream_and_user(messages, history, user_message, react=react)

    payload = {
        "messages": messages,
        "stream": show_progress,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {
            "fps": video_input.fps,
            "is_video": True,
            "grid_thw": list(video_input.grid_thw),
            "temporal_positions": video_input.temporal_positions,
        },
    }

    if options:
        payload["temperature"] = options.get("temperature", 0.9)
        if "top_p" in options:
            payload["top_p"] = options["top_p"]
        if "num_predict" in options or "max_tokens" in options:
            payload["max_tokens"] = options.get("max_tokens", options.get("num_predict", 80))
        if "repeat_penalty" in options:
            payload["repeat_penalty"] = options["repeat_penalty"]
        _forward_sampler_options(payload, options)
    else:
        payload["temperature"] = 0.9

    _vt0 = time.time()
    endpoint = f"{LLAMA_SERVER_URL}/v1/chat/completions"

    print(f"[SUPERFRAME] {video_input.num_source_frames} frames, grid_thw={video_input.grid_thw}, {len(video_input.super_frames)} super-frames")

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

    response_text = _clean_continuation(response_text, prefill)

    log_llm_call(
        prompt=prompt,
        model="llama-server",
        response=response_text,
        success=True,
        timeout=timeout,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=system_prompt,
        prompt_type="caption",
        api_endpoint=endpoint,
        history_len=len(history or []),
        stream_mode=(_stream_mode() + ("-react" if react else "")) if history else None,
        num_frames=len(frames),
        prefill_tail=prefill[-150:] if prefill else None,
        duration_s=time.time() - _vt0,
    )

    return response_text


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
    history: Optional[List[str]] = None,
    react: bool = False,
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
    # Same gate as the single-image path. Video calls are the expensive
    # ones, so a caption arriving mid-video is exactly the case worth
    # skipping rather than queueing.
    # Video calls are always captions (both internals log prompt_type="caption"),
    # and captions are the realtime path: skip rather than queue.
    _got, _queued = _acquire_inference("caption", wait=False)
    if not _got:
        print(f"[llama-server] caption (video) skipped — busy with {busy_with()}")
        return ""
    try:
        if not skip_generation_wait:
            _wait_for_drawing_completion()

        # Auto-restart if llama-server has crashed (frees ComfyUI VRAM first + retries)
        if not is_server_running():
            print("[llama-server] Server not responding — attempting restart...")
            if ensure_server_up():
                print("[llama-server] Restarted successfully")
            else:
                print("[llama-server] Restart failed — falling back to single frame")
                if frames:
                    return query_llama_server(
                        prompt=prompt,
                        # without prompt_type these land in the log as "general" — a
                        # real caption invisible to every per-type measurement (Aug 2)
                        prompt_type="caption",
                        image=frames[-1],
                        system_prompt=system_prompt,
                        options=options,
                        timeout=timeout,
                        show_progress=show_progress,
                        skip_generation_wait=True,
                    )
                return "[WARNING] llama-server unavailable"

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
                    history=history,
                    react=react,
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
                    history=history,
                    react=react,
                )
            except Exception as e:
                error_msg = str(e)
                _note_query_outcome(error_msg)
                # Log it. This used to be swallowed — only the successful
                # single-frame fallback reached the log, so multi-image showed a
                # 100% success rate while the console wedged every few minutes.
                # An invisible failure is one nobody can measure or fix.
                log_llm_call(
                    prompt=prompt,
                    model="llama-server",
                    response=None,
                    success=False,
                    error_message=error_msg,
                    timeout=timeout,
                    log_dir=MOOD_SNAPSHOT_FOLDER,
                    system_prompt=system_prompt,
                    prompt_type="caption_multiimage_failed",
                    num_frames=len(frames),
                )
                print(f"[llama-server] Multi-image failed: {error_msg}")

        # Final fallback: single last frame
        if frames:
            print("[llama-server] Falling back to single-frame caption")
            return query_llama_server(
                prompt=prompt,
                prompt_type="caption",
                image=frames[-1],
                system_prompt=system_prompt,
                options=options,
                timeout=timeout,
                show_progress=show_progress,
                skip_generation_wait=True,
            )
        return "[WARNING] No frames provided"
    finally:
        _release_inference("caption")
