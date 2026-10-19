"""A/B the render formatter's MEDIUM clause on a real past intent.

Sep 19: DSV turns smooth tone and clean contour into coherent strokes, and
hatch texture into sub-millimetre debris (Aug 12 eval: clean_lines 0% tiny
fragments, median stroke 5% of diagonal; Sep 18 live render: 44% tiny, 1.1%).
The formatter's own clause orders "dense cross-hatching", which is the one
input the engine cannot parse — so the fix belongs upstream, in the prompt,
not in the vectoriser. This measures that.

Everything is held constant except the medium sentence: same intent, same
seed, same ControlNet frame, same LoRA/guidance/steps. Runs in three stages so
no single call is long, and because the GPU has to change hands between them:

    python debug/test_render_variants.py prompts    # needs llama-server up
    python debug/test_render_variants.py renders    # stops llama-server, uses ComfyUI
    python debug/test_render_variants.py measure    # centerlines + montage

machine.py must be stopped: stage 2 takes llama-server down for VRAM, the same
handoff the drawing pipeline does. Settle pauses sit on both sides of it
(LLAMA_HANDOFF_SETTLE_S) — both Xid 79 crashes happened at that moment.
"""

import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE = "/tmp/claude-1000/render_variants.json"
FRAME = "event_log/7386244c-images/mood_1789735747_t1024.jpg"  # the frame the Sep 18 14:50 intent looked at
SEED = 326367778914853  # fixed: the prompt is the only variable
RUN_LOG = "event_log/7386244c-event-log.json"
INTENT_AT = "2026-09-18T14:50"

_COMMON_HEAD = "You format a drawing machine's intention into a prompt for an image generator. " "One plain paragraph: what is in the picture, where it sits, how it is treated — " "in the machine's own words wherever they hold. "
_COMMON_TAIL = (
    "Colour words never enter the prompt; give a colour's intensity as darkness instead. "
    "Be concrete: things, placement, scale, light and dark. "
    "Translate metaphor into what can be seen; every sentence puts something visible "
    "on the page; describe what is there, not what is absent or how to read it. "
    "Add nothing of your own; never mention plotters, tracing, vectors, or machines. "
    "50-120 words, no commentary."
)

VARIANTS = {
    # live text, verbatim (captioner/prompts.py render_system)
    "A_live": (
        "It is black ink line art drawn "
        "with a fine-tipped pen: strokes are thin and distinct, all tone is hatched "
        "line — the deepest shadow is dense cross-hatching with white paper breathing "
        "through, never a solid filled mass or a fat marker stroke. "
    ),
    # draft 1: contour first, tone smooth, hatching gone
    "B_smooth": (
        "It is a drawing in black on white paper, made with a fine-tipped pen: "
        "one thin, distinct stroke per edge, never a bundle of repeated strokes "
        "searching for the same line. Where the drawing darkens it darkens as smooth "
        "continuous shading — never ruled lines, never cross-hatching, never a solid "
        "filled mass. Complexity belongs to what is in the picture and how it is "
        "arranged, not to the number of marks. "
    ),
    # draft 2: pure contour, no tone at all
    "C_contour": (
        "It is a line drawing in black on white paper, made with a fine-tipped pen: "
        "one thin, distinct stroke per edge, and no shading anywhere — light and dark "
        "are told by what is drawn and what is left as bare paper. No hatching, no "
        "filled areas, no bundles of repeated strokes. Complexity belongs to what is "
        "in the picture and how it is arranged, not to the number of marks. "
    ),
}


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(d):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(d, open(STATE, "w"), indent=1)


def _events(path):
    raw = open(path).read()
    dec, i = json.JSONDecoder(), 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \n\r\t":
            i += 1
        if i >= len(raw):
            return
        try:
            o, i = dec.raw_decode(raw, i)
        except Exception:
            return
        yield o


def find_intent():
    """The full intent, from the drawing_render call's own prompt (the
    stream_drawing_analysis event truncates it to 300 chars)."""
    best = None
    for o in _events(RUN_LOG):
        if o.get("prompt_type") == "drawing_render" and str(o.get("iso_timestamp", "")).startswith(INTENT_AT):
            m = re.search(r"own words:\s*(.*?)\s*Give this intention", str(o.get("prompt", "")), re.S)
            if m:
                best = m.group(1).strip()
    return best


def settle(label):
    try:
        from config.config import LLAMA_HANDOFF_SETTLE_S as s
    except Exception:
        s = 10.0
    if float(s) > 0:
        print(f"[settle] {float(s):.0f}s {label}")
        time.sleep(float(s))


# ---------------------------------------------------------------- stage 1
def stage_prompts():
    from config.config import DRAWING_CALL_TIMEOUT, MOOD_SNAPSHOT_FOLDER
    from utils.inference import query_model

    intent = find_intent()
    if not intent:
        sys.exit("could not recover the intent from the run log")
    print(f"intent ({len(intent)} chars): {intent[:220]}…\n")
    st = {"intent": intent, "prompts": {}}
    user = f"The intention, in the machine's own words:\n\n{intent}\n\n" "Give this intention its visible form — write the description."
    for name, clause in VARIANTS.items():
        out = query_model(
            prompt=user,
            image=None,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=_COMMON_HEAD + clause + _COMMON_TAIL,
            prompt_type="drawing_render",
            timeout=DRAWING_CALL_TIMEOUT,
            options={"temperature": 0.5, "num_predict": 220, "top_p": 0.9, "repeat_penalty": 1.2},
        )
        out = (out or "").strip()
        st["prompts"][name] = out
        hatch = len(re.findall(r"hatch|cross-hatch|shading|tone|dense", out, re.I))
        print(f"--- {name}  ({len(out.split())} words, {hatch} tone/hatch mentions)\n{out}\n")
    save_state(st)
    print(f"saved -> {STATE}")


# ---------------------------------------------------------------- stage 2
def stage_renders():
    st = load_state()
    if not st.get("prompts"):
        sys.exit("run the prompts stage first")
    from config.config import COMFY_CNET_STRENGTH, COMFY_FLUX_GUIDANCE, COMFY_LATENT_HEIGHT, COMFY_LATENT_WIDTH, COMFY_LORA_STRENGTH, COMFY_OUTPUT_FOLDER, COMFY_STEPS, TRIGGER_PROMPT
    from drawing.comfy import create_impostor_controller

    print("freeing VRAM: stopping llama-server (machine.py must be stopped)")
    try:
        from utils.inference import unload_model

        unload_model()
    except Exception as e:
        print(f"  unload failed: {e}")
    settle("after the model unload, before ComfyUI takes the card")

    st.setdefault("renders", {})
    for name, prompt in st["prompts"].items():
        prefix = f"variant-{name}-{int(time.time())}"
        before = set(glob.glob(os.path.join(COMFY_OUTPUT_FOLDER, "*.png")))
        ctl = create_impostor_controller(
            load_image_path=os.path.abspath(FRAME),
            override_prompt=prompt,
            primitive_string=TRIGGER_PROMPT,
            filename_prefix=prefix,
            latent_width=COMFY_LATENT_WIDTH,
            latent_height=COMFY_LATENT_HEIGHT,
            cnet_strength=COMFY_CNET_STRENGTH,
            flux_guidance=COMFY_FLUX_GUIDANCE,
            steps=COMFY_STEPS,
            lora_strength=COMFY_LORA_STRENGTH,
            noise_seed=SEED,
        )
        if not ctl.queue_prompt():
            print(f"  {name}: queue FAILED")
            continue
        print(f"  {name}: queued as {prefix}, waiting…")
        got, deadline = None, time.time() + 300
        while time.time() < deadline:
            new = [p for p in set(glob.glob(os.path.join(COMFY_OUTPUT_FOLDER, "*.png"))) - before if prefix in p]
            if new and os.path.getsize(new[0]) > 100000:
                time.sleep(1.0)
                got = new[0]
                break
            time.sleep(2.0)
        print(f"  {name}: {'-> ' + os.path.basename(got) if got else 'TIMED OUT'}")
        if got:
            st["renders"][name] = got
        save_state(st)
    settle("after ComfyUI, before anything else wants the card")
    print(f"saved -> {STATE}")


# ---------------------------------------------------------------- stage 3
def stage_measure():
    import cv2
    import numpy as np

    st = load_state()
    if not st.get("renders"):
        sys.exit("run the renders stage first")
    from bcnc import raster_to_centerline_svg

    def polys(path):
        out = []
        txt = open(path).read()
        for pts in re.findall(r'points="([^"]+)"', txt):
            p = [[float(v) for v in q.split(",")] for q in pts.replace("\n", " ").split()]
            if len(p) >= 2:
                out.append(np.array(p, float))
        return out

    rows, table = [], []
    for name in VARIANTS:
        png = st["renders"].get(name)
        if not png or not os.path.exists(png):
            continue
        svg = f"/tmp/claude-1000/variant_{name}.svg"
        t0 = time.time()
        raster_to_centerline_svg(png, svg)  # the LIVE engine (CENTERLINE_ENGINE)
        ps = polys(svg)
        im = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        ink = (im < 170).astype(np.uint8)
        n_comp, _ = cv2.connectedComponents(ink)
        if ps:
            allp = np.vstack(ps)
            diag = np.linalg.norm(allp.max(0) - allp.min(0))
            L = np.array([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in ps]) / diag * 100
            table.append((name, len(ps), float(np.median(L)), 100 * float((L < 1).mean()), ink.mean() * 100, n_comp, time.time() - t0))
        # preview: render + centerline side by side
        prev = cv2.resize(cv2.imread(png), (420, 420))
        cl = np.full((420, 420, 3), 255, np.uint8)
        if ps:
            lo, hi = allp.min(0), allp.max(0)
            s = 396 / max(max(hi - lo), 1e-6)
            for p in ps:
                cv2.polylines(cl, [((p - lo) * s + 12).astype(np.int32)], False, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(prev, name, (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
        cv2.putText(cl, f"{name} centerlined", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
        rows.append(np.hstack([prev, cl]))
    print(f"{'variant':12} {'strokes':>8} {'median %diag':>13} {'tiny <1%':>9} {'ink %':>7} {'components':>11} {'centerline s':>13}")
    for r in table:
        print(f"{r[0]:12} {r[1]:8} {r[2]:13.2f} {r[3]:8.0f}% {r[4]:7.2f} {r[5]:11} {r[6]:13.0f}")
    if rows:
        out = "/tmp/claude-1000/render_variants.png"
        cv2.imwrite(out, np.vstack(rows))
        print(f"\n-> {out}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage.startswith("prompt"):
        stage_prompts()
    elif stage.startswith("render"):
        stage_renders()
    elif stage.startswith("measure"):
        stage_measure()
    else:
        print(__doc__)
