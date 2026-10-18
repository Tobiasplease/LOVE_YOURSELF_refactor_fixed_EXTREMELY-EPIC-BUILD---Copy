"""Phase 0 of the diary layer (docs/diary-memory-design-oct18.md): the metrics
that say whether a change moved the monologue, measured on real runs.

The failure these catch is stationarity — the monologue converging to a fixed
point and staying there. On the 92-hour run of Oct 9-13 the word-frequency
cosine between the first and last 6-hour window was 0.992: four days moved the
output distribution by under 1%. Nothing in the repo measured that, which is why
it ran for months unnoticed.

    python debug/diary_metrics.py                  # longest run on disk
    python debug/diary_metrics.py --run f5cbdb1c
    python debug/diary_metrics.py --all-months     # the Jul-Oct trend
    python debug/diary_metrics.py --since 2026-10-01

Reads event_log/*-event-log.json directly (caption + llm_api_call events) so it
works on history predating the caption spool.
"""

import argparse
import datetime
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO, "event_log")

WINDOW_H = 6
TOKEN_BUDGET = 60000

# Any line telling the machine how long it has been waiting. The Sep 10-14 work
# converted the numerals to words, so the word forms have to count too.
DURATION_RE = re.compile(
    r"\b(\d+\s*(second|minute|hour|day|week|month)s?"
    r"|(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|a few|half)\s+\w{0,8}\s*(second|minute|hour|day|week|month)s?"
    r"|ago|since|last time|been off|offline|in this room)\b",
    re.I,
)
# The template NEW ABOUT ME collapsed into: 53% of October self-facts.
TEMPLATE_RE = re.compile(r"\b(perceive|rather than)\b", re.I)

WORD_RE = re.compile(r"[a-z']+")


def words(text):
    return WORD_RE.findall((text or "").lower())


def cosine(a, b):
    keys = set(a) | set(b)
    num = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def entropy(counter):
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counter.values() if v)


def unique_per_budget(captions, budget=TOKEN_BUDGET):
    """Vocabulary at a FIXED token budget — raw unique-word counts are
    meaningless here because caption length and rate both changed."""
    toks = []
    for cap in captions:
        toks += words(cap)
        if len(toks) >= budget:
            break
    toks = toks[:budget]
    return len(set(toks)), len(toks)


def iter_events(paths, wanted_types):
    """Caption and llm_api_call events. Most logs are JSONL; the stub runs are
    pretty-printed, so fall back to a brace-depth accumulator for those."""
    for path in paths:
        try:
            handle = open(path, errors="replace")
        except OSError:
            continue
        with handle:
            buf, depth, started = [], 0, False
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        event = json.loads(stripped)
                    except ValueError:
                        pass
                    else:
                        if event.get("type") in wanted_types:
                            yield event
                        continue
                if not started:
                    if stripped.startswith("{"):
                        started, buf = True, [line]
                        depth = line.count("{") - line.count("}")
                    continue
                buf.append(line)
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    started = False
                    try:
                        event = json.loads("".join(buf))
                    except ValueError:
                        buf = []
                        continue
                    buf = []
                    if event.get("type") in wanted_types:
                        yield event


def load(since=None, run=None):
    paths = sorted(glob.glob(os.path.join(LOG_DIR, "*-event-log.json")))
    if run:
        paths = [p for p in paths if os.path.basename(p).startswith(run)]
        if not paths:
            sys.exit(f"no event log for run {run}")
    caps, calls = [], []
    for event in iter_events(paths, {"caption", "llm_api_call"}):
        ts = event.get("iso_timestamp") or ""
        if since and ts[:10] < since:
            continue
        if event.get("type") == "caption":
            text = (event.get("caption") or "").strip()
            if text:
                caps.append((ts, event.get("run_id", ""), text))
        elif event.get("prompt_type") in ("caption", "caption_reroute"):
            calls.append(
                (
                    ts,
                    (event.get("system_prompt") or "") + "\n" + (event.get("prompt") or ""),
                    int(event.get("full_prompt_length") or 0) + int(event.get("full_system_prompt_length") or 0),
                )
            )
    caps.sort()
    calls.sort()
    return caps, calls


def frame_metrics(calls, label):
    """Duration pressure: what share of what the machine reads before each
    thought is telling it how long it has been waiting. 2.93 lines/call at 100%
    of calls in October, out of 6.9 lines total."""
    if not calls:
        return
    lines_total = dur_lines = with_dur = 0
    prompt_chars = 0
    for _ts, text, plen in calls:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines_total += len(lines)
        hits = sum(1 for ln in lines if DURATION_RE.search(ln))
        dur_lines += hits
        with_dur += 1 if hits else 0
        prompt_chars += plen
    n = len(calls)
    print(f"\n[frame] {label}  ({n} caption calls)")
    print(f"  lines per call            {lines_total / n:6.1f}")
    print(f"  duration lines per call   {dur_lines / n:6.2f}   target <= 0.30")
    print(f"  calls carrying duration   {100 * with_dur / n:5.0f}%   target < 20%")
    print(f"  prompt chars per call     {prompt_chars / n:6.0f}   must not grow")


def self_material_metrics(since=None, run=None):
    """Template share of the machine's self-knowledge. The NEW ABOUT ME quota
    produced one sentence frame -- 'I perceive X rather than Y' -- in 53% of
    October's self-facts. Reads the compression slots; after the diary lands,
    point this at diary.jsonl instead."""
    paths = sorted(glob.glob(os.path.join(LOG_DIR, "*-event-log.json")))
    if run:
        paths = [p for p in paths if os.path.basename(p).startswith(run)]
    facts, felt = [], Counter()
    for event in iter_events(paths, {"llm_api_call"}):
        if event.get("prompt_type") != "compression":
            continue
        ts = event.get("iso_timestamp") or ""
        if since and ts[:10] < since:
            continue
        for line in (event.get("response") or "").splitlines():
            match = re.match(r"\s*([A-Z][A-Z ]+):\s*(.*)", line)
            if not match:
                continue
            slot, value = match.group(1).strip(), match.group(2).strip()
            if not value:
                continue
            if slot == "NEW ABOUT ME" and value.lower().strip(" .") not in ("none", "nothing"):
                facts.append(value)
            elif slot == "FELT":
                felt[value.lower()] += 1
    if facts:
        templated = sum(1 for f in facts if TEMPLATE_RE.search(f))
        print(f"\n[self] {len(facts)} self-facts")
        print(f"  template share            {100 * templated / len(facts):5.0f}%   target < 15%")
    if felt:
        print(f"  FELT distinct / entropy   {len(felt)} / {entropy(felt):.2f}   (10.18 in Aug, 6.42 in Oct)")


def stationarity(caps, run_id):
    """The headline. Word-frequency cosine between 6-hour windows of one run:
    adjacent windows, and every window against the first. A developing
    monologue drifts; this one does not."""
    rows = []
    for ts, rid, text in caps:
        if rid != run_id:
            continue
        try:
            rows.append((datetime.datetime.fromisoformat(ts), text))
        except ValueError:
            continue
    if len(rows) < 200:
        print(f"\n[stationarity] run {run_id}: only {len(rows)} captions, skipping")
        return
    rows.sort()
    t0 = rows[0][0]
    bins = defaultdict(list)
    for stamp, text in rows:
        bins[int((stamp - t0).total_seconds() // 3600 // WINDOW_H)].append(text)
    span_h = (rows[-1][0] - t0).total_seconds() / 3600
    print(f"\n[stationarity] run {run_id}: {len(rows)} captions over {span_h:.1f}h")
    print(f"  {'window':>12} {'caps':>6} {'uniq/1k':>8} {'cos prev':>9} {'cos first':>10}")
    first = prev = None
    for b in sorted(bins):
        bag = Counter(w for c in bins[b] for w in words(c))
        total = sum(bag.values())
        if total < 2000:
            continue
        if first is None:
            first = bag
        cp = f"{cosine(bag, prev):9.3f}" if prev else " " * 9
        print(f"  {b * WINDOW_H:5d}-{b * WINDOW_H + WINDOW_H:<6d} {len(bins[b]):6d} {1000 * len(bag) / total:8.1f} {cp} {cosine(bag, first):10.3f}")
        prev = bag
    if first is not None and prev is not None and prev is not first:
        verdict = "STATIONARY — the run is not developing" if cosine(prev, first) > 0.95 else "drifting"
        print(f"  first vs last: {cosine(prev, first):.3f}   target < 0.95   [{verdict}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run id prefix; default = run with most captions")
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--all-months", action="store_true", help="per-month lexical range trend")
    args = ap.parse_args()

    caps, calls = load(since=args.since, run=args.run)
    if not caps:
        sys.exit("no captions found")
    print(f"{len(caps)} captions  {caps[0][0][:16]} -> {caps[-1][0][:16]}")

    if args.all_months:
        by_month = defaultdict(list)
        for ts, _rid, text in caps:
            by_month[ts[:7]].append(text)
        print(f"\n[lexical range] unique words per {TOKEN_BUDGET // 1000}k caption tokens")
        for month in sorted(by_month):
            uniq, total = unique_per_budget(by_month[month])
            short = "  (under budget)" if total < TOKEN_BUDGET else ""
            print(f"  {month}  {len(by_month[month]):7d} caps   {uniq:6d}{short}")
    else:
        uniq, total = unique_per_budget([c for _ts, _r, c in caps])
        print(f"\n[lexical range] {uniq} unique words per {total} tokens   target > 3000")

    run_id = args.run
    if not run_id:
        run_id = Counter(r for _ts, r, _c in caps).most_common(1)[0][0]
    else:
        matching = [r for _ts, r, _c in caps if r.startswith(run_id)]
        run_id = matching[0] if matching else run_id
    stationarity(caps, run_id)

    frame_metrics(calls, args.run or args.since or "all history")
    self_material_metrics(since=args.since, run=args.run)


if __name__ == "__main__":
    main()
