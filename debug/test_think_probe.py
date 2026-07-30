"""The think-channel probe (docs/next-session-brief.md, open question #2).

Qwen3.x with enable_thinking:false puts polished conclusions in the output and
the messy deliberative register inside the suppressed think block. This probe
replays real caption calls from a live 27B session with enable_thinking:TRUE
and reads the think channel. Verdict wanted:
  - scene-deliberation ("wait — no, that's the same guy… should I…")
    → design think-as-monologue (harvest think text as the caption)
  - task-meta ("the user wants a caption…")
    → the idea dies here

Deviations from the live call, both forced by the probe's purpose:
  - NO assistant prefill: thinking forbids prefill (server rejects it). The
    stream rides as one assistant message BEFORE the user turn — the same
    no-prefill ordering (react/world shape) think-as-monologue would use.
  - max_tokens raised from the live 80: the point is READING the think
    channel; truncating it mid-thought defeats the probe.
Everything else is faithful: real system prompt, real user prompt, live
sampler settings, a real frame from within seconds of the original call.

Usage (27B server must be up on --url):
  python debug/test_think_probe.py --log event_log/58f99bba-event-log.json \
      --n 4 --out debug/think_probe_results.md
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def load_events(path):
    evs = []
    for line in open(path):
        line = line.strip().rstrip(",")
        if line.startswith("{"):
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    evs.append(o)
            except json.JSONDecodeError:
                pass
    return evs


def session_images(log_path):
    folder = log_path.replace("-event-log.json", "-images")
    pairs = []
    for p in glob.glob(os.path.join(folder, "mood_*.jpg")):
        try:
            pairs.append((int(os.path.basename(p)[5:-4]), p))
        except ValueError:
            pass
    return sorted(pairs)


def pick_calls(evs, imgs, n):
    """Unique successful caption calls (dedupe by response), each with a
    session frame within ±6s, spread across the session."""
    seen = set()
    candidates = []
    for e in evs:
        if e.get("type") != "llm_api_call" or e.get("prompt_type") != "caption":
            continue
        if not (e.get("success") and e.get("response") and e.get("system_prompt")):
            continue
        key = e["response"][:60]
        if key in seen:
            continue
        seen.add(key)
        ts = e["timestamp"]
        if e.get("image_path") and os.path.exists(e["image_path"]):
            img = e["image_path"]
        else:
            near = min(imgs, key=lambda ip: abs(ip[0] - ts), default=None)
            if not near or abs(near[0] - ts) > 6:
                continue
            img = near[1]
        candidates.append((e, img))
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


def build_payload(call, img_path, history):
    import base64

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    messages = [{"role": "system", "content": call["system_prompt"]}]
    stream_text = " ".join(h.strip() for h in history if h and h.strip())
    if stream_text:
        messages.append({"role": "assistant", "content": stream_text})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": call["prompt"]},
            ],
        }
    )
    return {
        "messages": messages,
        "stream": False,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "temperature": 0.7,
        "top_p": 0.85,
        "repeat_penalty": 1.15,
        "dry_multiplier": 0.85,
        "dry_base": 1.75,
        "dry_allowed_length": 3,
        "dry_penalty_last_n": 128,
        "max_tokens": 900,
        "seed": random.randint(1, 1000000),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="event_log/58f99bba-event-log.json")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="debug/think_probe_results.md")
    args = ap.parse_args()

    evs = load_events(args.log)
    imgs = session_images(args.log)
    captions = [(e["timestamp"], e["caption"]) for e in evs if e.get("type") == "caption" and e.get("caption")]
    sampled = pick_calls(evs, imgs, args.n)
    print(f"{len(captions)} captions in log, probing {len(sampled)} calls")

    rows = []
    for i, (call, img) in enumerate(sampled, 1):
        ts = call["timestamp"]
        k = call.get("history_len") or len(captions)
        history = [c for t, c in captions if t < ts][-k:]
        payload = build_payload(call, img, history)
        t0 = time.time()
        r = requests.post(f"{args.url}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        dt = time.time() - t0
        content = (msg.get("content") or "").strip()
        think = (msg.get("reasoning_content") or "").strip()
        if not think:
            m = THINK_RE.search(content)
            if m:
                think = m.group(1).strip()
                content = THINK_RE.sub("", content).strip()
        rows.append(
            {
                "iso": call.get("iso_timestamp", "?"),
                "prompt": call["prompt"],
                "img": os.path.basename(img),
                "history": history,
                "live_answer": call["response"],
                "think": think or "(EMPTY — no think channel returned)",
                "answer": content,
                "secs": round(dt, 1),
            }
        )
        print(f"\n[{i}/{len(sampled)}] {call.get('iso_timestamp', '?')[11:]} ({dt:.1f}s)")
        print(f"  THINK: {(think or '(empty)')[:300]}")
        print(f"  ANSWER: {content[:160]}")

    with open(args.out, "w") as f:
        f.write(f"# Think-channel probe — {os.path.basename(args.log)}, enable_thinking:true\n\n")
        f.write("The question: is the suppressed think register scene-deliberation or task-meta?\n")
        f.write("Shape: no prefill (thinking forbids it) — stream as one assistant msg before the user turn.\n\n")
        for r in rows:
            f.write(f"## {r['iso']}  ({r['secs']}s, frame {r['img']})\n\n")
            f.write(f"**User prompt:**\n```\n{r['prompt']}\n```\n\n")
            if r["history"]:
                f.write(f"**Stream ({len(r['history'])} entries):** {' '.join(r['history'])[:400]}\n\n")
            f.write(f"**THINK CHANNEL:**\n\n> {r['think'].replace(chr(10), chr(10) + '> ')}\n\n")
            f.write(f"**Final answer (thinking on):** {r['answer']}\n\n")
            f.write(f"**Live answer (thinking off, July 28):** {r['live_answer']}\n\n---\n\n")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
