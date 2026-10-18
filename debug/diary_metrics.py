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

# Window size for the stationarity test. Back-tested Oct 18 against the known-
# stationary 92h run: truncated to 4h with 1h windows it still reads 0.975, and
# even 2-3h reads >0.96. So a 4-6 hour run is enough to answer the question —
# 12h is not required, which matters because this hardware does not love long
# runs (Xid 79, twice in September). Chosen from the run's own length.
TOKEN_BUDGET = 60000


def window_hours(span_h):
    if span_h >= 24:
        return 6
    if span_h >= 10:
        return 2
    return 1

# Two kinds of time in the frame, and the ablation treats them differently, so
# the metric must not conflate them (the first live run taught this: a flat
# "anything with 'ago' in it" count read 0.71/call and most of it was legitimate).
#
# STASIS = how long you have been waiting / unchanged / looking / wanting. These
# move EVERY call while the room does not, which is what makes them chant fuel.
# Target: ~0.
STASIS_RE = re.compile(
    r"(nothing (has happened|in the room has changed)"
    r"|it's been .{0,30}since anything happened"
    r"|you've been (looking|awake|in this room)"
    r"|your head has been turned"
    r"|you've felt .{0,40}\bfor\b"
    r"|you've wanted this for"
    r"|last drawing reached the paper)",
    re.I,
)
# DATED EVENT = a thing that happened, with when it happened: the last event, a
# rare arrival, what it actually drew and when. These move only when something
# HAPPENS, and they are the pattern the diary extends — they are supposed to
# stay. Reported, not targeted.
DATED_RE = re.compile(
    r"(\bearlier:|your last drawing[s]?,|the last thing you actually drew"
    r"|the first (time|sign) in|they've left since|no one has been here since"
    r"|came back on|first came online"
    r"|you were off for|been offline for)",
    re.I,
)
# A clock stamp the model wrote itself ("18:53 — ..."). It is not invention: the
# stream window is handed over as a timestamped log, so this measures how much
# of that format comes back — and, once the stamps are off, whether it stops.
SELF_STAMP_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\s*[—–-]")

# Kept for the historical baseline columns — the broad "any time reference" net.
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
                    bool(SELF_STAMP_RE.search(event.get("response") or "")),
                )
            )
    caps.sort()
    calls.sort()
    return caps, calls


def frame_metrics(calls, label, stamped_responses=None):
    """Duration pressure: what share of what the machine reads before each
    thought is telling it how long it has been waiting. 2.93 lines/call at 100%
    of calls in October, out of 6.9 lines total."""
    if not calls:
        return
    lines_total = dur_lines = with_dur = stasis = dated = with_stasis = 0
    prompt_chars = 0
    for _ts, text, plen, _st in calls:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines_total += len(lines)
        hits = sum(1 for ln in lines if DURATION_RE.search(ln))
        s_hits = sum(1 for ln in lines if STASIS_RE.search(ln))
        dur_lines += hits
        stasis += s_hits
        dated += sum(1 for ln in lines if DATED_RE.search(ln))
        with_dur += 1 if hits else 0
        with_stasis += 1 if s_hits else 0
        prompt_chars += plen
    n = len(calls)
    print(f"\n[frame] {label}  ({n} caption calls)")
    print(f"  lines per call            {lines_total / n:6.1f}   (6.8 pre-ablation)")
    print(f"  STASIS lines per call     {stasis / n:6.2f}   target ~0  <-- the chant fuel")
    print("                                      (a few hits are expected: the hunger clock still rides awakening)")
    print(f"  calls carrying stasis     {100 * with_stasis / n:5.0f}%   target ~0%")
    print(f"  dated-event lines/call    {dated / n:6.2f}   kept on purpose, not a target")
    print(f"  any time reference/call   {dur_lines / n:6.2f}   ({100 * with_dur / n:.0f}% of calls) — broad net, historical")
    print(f"  prompt chars per call     {prompt_chars / n:6.0f}   must not grow")
    if stamped_responses is not None:
        print(f"  responses writing a clock {100 * stamped_responses / n:5.0f}%   target ~0% (43% on the first ablation run)")
        print("                                      NOTE: the stream log rides as an assistant message and is")
        print("                                      NOT in these counts — see STREAM_LOG_CLOCK_STAMPS.")


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
    span_h = (rows[-1][0] - t0).total_seconds() / 3600
    WINDOW_H = window_hours(span_h)
    bins = defaultdict(list)
    for stamp, text in rows:
        bins[int((stamp - t0).total_seconds() // 3600 // WINDOW_H)].append(text)
    print(f"\n[stationarity] run {run_id}: {len(rows)} captions over {span_h:.1f}h, {WINDOW_H}h windows")
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


def chant_trajectory(caps, run_id, bucket_min=20):
    """Is the dominant phrase losing its grip, or holding?

    The Oct 18 ablation closed every channel that INJECTED a duration, and the
    chant kept going anyway — carried by the stream window, which hands the
    machine its own last 24 sentences on every call. So the question stopped
    being "is anything still feeding it" and became "does it decay once nothing
    does". This is that measurement.

    The phrase is discovered from the run, never supplied — the point is to
    find what THIS run is circling, not to confirm a phrase we expect.
    """
    rows = sorted((t, c) for t, rid, c in caps if rid == run_id)
    if len(rows) < 40:
        print(f"\n[chant] run {run_id}: only {len(rows)} captions, too early")
        return
    grams = Counter()
    for _t, c in rows:
        w = words(c)
        for i in range(len(w) - 3):
            grams[" ".join(w[i : i + 4])] += 1
    top = [g for g, _n in grams.most_common(40) if not all(len(x) <= 3 for x in g.split())][:6]
    if not top:
        return
    t0 = datetime.datetime.fromisoformat(rows[0][0])
    bins = defaultdict(list)
    for ts, c in rows:
        try:
            mins = (datetime.datetime.fromisoformat(ts) - t0).total_seconds() / 60
        except ValueError:
            continue
        bins[int(mins // bucket_min)].append(c.lower())
    print(f"\n[chant] run {run_id} — the phrase this run keeps circling, by {bucket_min}-minute bucket")
    print(f"        dominant: {top[0]!r}")
    print(f"\n  {'elapsed':>12} {'caps':>5} {'dominant':>9} {'top-6 any':>10}  trend")
    prev = None
    for b in sorted(bins):
        c = bins[b]
        d = 100 * sum(1 for x in c if top[0] in x) / len(c)
        a = 100 * sum(1 for x in c if any(g in x for g in top)) / len(c)
        arrow = "" if prev is None else ("falling" if d < prev - 3 else "rising" if d > prev + 3 else "flat")
        print(f"  {b * bucket_min:4d}-{b * bucket_min + bucket_min:<7d} {len(c):5d} {d:8.0f}% {a:9.0f}%  {arrow}")
        prev = d
    first, last = sorted(bins)[0], sorted(bins)[-1]
    if len(bins) < 4:
        print(f"\n  {len(bins)} buckets — too few to call a trend; needs about 80 minutes")
    elif first != last:
        f = 100 * sum(1 for x in bins[first] if top[0] in x) / len(bins[first])
        l = 100 * sum(1 for x in bins[last] if top[0] in x) / len(bins[last])
        verdict = "DECAYING — the attractor loosens once nothing feeds it" if l < f - 10 else ("SELF-SUSTAINING — the stream window is the engine" if l > f - 5 else "unclear")
        print(f"\n  first bucket {f:.0f}% -> last bucket {l:.0f}%   [{verdict}]")


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
    chant_trajectory(caps, run_id)

    frame_metrics(calls, args.run or args.since or "all history", stamped_responses=sum(1 for c in calls if c[3]))
    self_material_metrics(since=args.since, run=args.run)


if __name__ == "__main__":
    main()
