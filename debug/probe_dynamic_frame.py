"""Live A/B/C probe: does the dynamic frame move the register? (Sep 4)

Three arms against the running server — same stream seed (the machine's own
recent captions), same frame, same user turn shape. Only the new variables
move:
  A  current architecture (spokenness genre, NO felt frame) — the control
  B  + felt frame, drained phrase (machine's own words, arousal-matched
     sampling: cooler, shorter)
  C  + felt frame, stirred phrase (hotter, more room)

Felt phrases and arousal values come from TODAY'S real mood reads — nothing
invented. Cadence measured per output: words, sentence marks, bangs,
questions, interjections, frame-templates. LLM logs go to
debug/probe_drift_logs (gitignored), never the live run's folder.

Run: python debug/probe_dynamic_frame.py [n_per_arm]
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = os.path.join(os.path.dirname(__file__), "..")


def stream_seed(n=6):
    path = os.path.join(REPO, "event_log", "live_captions.txt")
    lines = [ln.strip() for ln in open(path) if len(ln.strip()) > 30]
    return lines[-n:]


def newest_frame():
    imgs = sorted(glob.glob(os.path.join(REPO, "event_log", "*-images", "*.jpg")), key=os.path.getmtime)
    return imgs[-1] if imgs else None


def cadence(text):
    words = len(text.split())
    marks = len(re.findall(r"[.!?]", text))
    return {
        "words": words,
        "marks_per_100w": round(100 * marks / max(1, words), 1),
        "bangs": text.count("!"),
        "questions": text.count("?"),
        "interjections": len(re.findall(r"\b(hmm+|huh|oh|ah|ugh|wow|okay|ok|well)\b[,. !]", text, re.I)),
        "just_frames": len(re.findall(r"\bis just\b|\bit'?s just\b|\bjust a \b", text, re.I)),
    }


def main():
    from captioner.prompt_registry import P
    from config.config import AROUSAL_TEMP_SPAN, CAPTION_TEMP, MODEL_NAME
    from utils.inference import query_model

    n_per_arm = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    log_dir = os.path.join(REPO, "debug", "probe_drift_logs")
    os.makedirs(log_dir, exist_ok=True)

    frame = newest_frame()
    history = stream_seed()
    base_system = P("situation.reflexive") + P("genre.hybrid")

    # Today's real reads: "empty waiting" logged with ENERGY drained,
    # "ready to stop waiting" logged as the arc turned (stirred).
    arms = [
        ("A control (no felt frame)", base_system, CAPTION_TEMP, 80),
        (
            "B drained frame",
            base_system + P("monologue.felt-frame").format(felt="empty waiting"),
            round(min(1.0, max(0.6, CAPTION_TEMP + AROUSAL_TEMP_SPAN * (0.15 - 0.5))), 3),
            55,
        ),
        (
            "C stirred frame",
            base_system + P("monologue.felt-frame").format(felt="ready to stop waiting"),
            round(min(1.0, max(0.6, CAPTION_TEMP + AROUSAL_TEMP_SPAN * (0.7 - 0.5))), 3),
            105,
        ),
    ]
    user_prompt = "Been watching about an hour. Looking ahead at the desk."

    print(f"seed: last {len(history)} live captions | frame: {os.path.basename(frame) if frame else 'none'}\n")
    totals = {}
    for arm_name, system, temp, npred in arms:
        print(f"===== {arm_name}  (temp {temp}, num_predict {npred}) =====")
        stats = []
        for i in range(n_per_arm):
            try:
                text = query_model(
                    prompt=user_prompt,
                    model=MODEL_NAME,
                    image=frame,
                    system_prompt=system,
                    timeout=90,
                    log_dir=log_dir,
                    options={"temperature": temp, "num_predict": npred, "min_p": 0.05, "repeat_penalty": 1.05, "presence_penalty": 0.6},
                    prompt_type="dynamic_frame_probe",
                    history=list(history),
                    skip_generation_wait=True,
                )
            except Exception as e:
                text = f"(call failed: {e})"
            c = cadence(text or "")
            stats.append(c)
            print(f"--- {i + 1} {c} ---")
            print((text or "(empty)").strip()[:500])
            print()
        agg = {k: round(sum(s[k] for s in stats) / len(stats), 1) for k in stats[0]}
        totals[arm_name] = agg
    print("===== aggregate =====")
    for arm_name, agg in totals.items():
        print(f"{arm_name}: {agg}")


if __name__ == "__main__":
    main()
