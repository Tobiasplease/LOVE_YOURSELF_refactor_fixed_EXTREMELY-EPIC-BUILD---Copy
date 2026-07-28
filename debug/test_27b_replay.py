"""Isolated register A/B: replay the live run's caption calls against a second
model on another port. Same system prompt, same user prompt, same stream
reconstruction, same image, same sampling, same message-assembly code path
(document prefill via utils.llama_server) — only the weights differ.

QUALITY test only: the second server runs partial CPU offload while the live
9B keeps the GPU, so per-call latency here means nothing. The latency bench
happens later with the GPU to itself.

Usage:
  python debug/test_27b_replay.py --log event_log/<run>-event-log.json \
      --url http://127.0.0.1:8081 --n 8 --out debug/replay_27b.md
"""

import argparse
import json
import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="debug/replay_27b.md")
    args = ap.parse_args()

    evs = load_events(args.log)
    captions = [(e["timestamp"], e.get("caption", "")) for e in evs if e.get("type") == "caption" and e.get("caption")]
    calls = [
        e
        for e in evs
        if e.get("type") == "llm_api_call"
        and e.get("prompt_type") == "caption"
        and e.get("success")
        and e.get("response")
        and e.get("has_image")
        and e.get("image_path")
        and os.path.exists(e.get("image_path", ""))
        and (e.get("history_len") or 0) >= 2
    ]
    if len(calls) < args.n:
        print(f"only {len(calls)} replayable calls found; using all")
    step = max(1, len(calls) // args.n)
    sampled = calls[::step][: args.n]
    print(f"{len(captions)} captions, {len(calls)} replayable calls, sampling {len(sampled)}")

    # Same assembly path as the live system, pointed at the second server.
    import utils.llama_server as ls

    ls.LLAMA_SERVER_URL = args.url

    from captioner.captioner import Captioner

    # Live gen_options copied from captioner._process_frame (quiet, not bored)
    def gen_options():
        return {
            "temperature": 0.7,
            "top_p": 0.85,
            "repeat_penalty": 1.15,
            "dry_multiplier": 0.85,
            "dry_base": 1.75,
            "dry_allowed_length": 3,
            "dry_penalty_last_n": 128,
            "num_predict": 80,
            "num_ctx": 4096,
            "seed": random.randint(1, 1000000),
        }

    def gate_verdict(text, history, prompt_text):
        cap = object.__new__(Captioner)
        cap._stream = deque(history, maxlen=8)
        try:
            return cap._caption_reject_reason(cap._strip_list_shape(text), prompt_text) or "PASS"
        except Exception as e:
            return f"gate-error: {e}"

    rows = []
    for i, call in enumerate(sampled, 1):
        ts = call["timestamp"]
        k = call.get("history_len") or 4
        history = [c for t, c in captions if t < ts][-k:]
        t0 = time.time()
        resp = ls.query_llama_server(
            prompt=call["prompt"],
            image=call["image_path"],
            system_prompt=call["system_prompt"],
            options=gen_options(),
            history=history,
            timeout=300,
            show_progress=False,
            skip_generation_wait=True,
        )
        dt = time.time() - t0
        prompt_text = (call.get("system_prompt") or "") + "\n" + (call.get("prompt") or "")
        row = {
            "iso": call.get("iso_timestamp", "?"),
            "user_prompt": call.get("prompt", ""),
            "live": call["response"],
            "candidate": (resp or "").strip(),
            "live_gate": gate_verdict(call["response"], history, prompt_text),
            "cand_gate": gate_verdict(resp or "", history, prompt_text),
            "secs": round(dt, 1),
        }
        rows.append(row)
        print(f"[{i}/{len(sampled)}] {row['iso'][11:]}  ({row['secs']}s, offloaded — latency not meaningful)")
        print(f"  9B : {row['live'][:160]}")
        print(f"  27B: {row['candidate'][:160]}")

    live_pass = sum(1 for r in rows if r["live_gate"] == "PASS")
    cand_pass = sum(1 for r in rows if r["cand_gate"] == "PASS")
    with open(args.out, "w") as f:
        f.write(f"# Replay A/B — {os.path.basename(args.log)} vs candidate model\n\n")
        f.write(f"Gate pass rate: live {live_pass}/{len(rows)} · candidate {cand_pass}/{len(rows)}\n\n")
        for r in rows:
            f.write(f"## {r['iso']}\n\n**User prompt:**\n```\n{r['user_prompt']}\n```\n\n")
            f.write(f"**9B (live):** {r['live']}\n\n_gate: {r['live_gate']}_\n\n")
            f.write(f"**27B:** {r['candidate']}\n\n_gate: {r['cand_gate']}_\n\n---\n\n")
    print(f"\ngate pass: live {live_pass}/{len(rows)}, candidate {cand_pass}/{len(rows)}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
