#!/bin/bash
# Night watch for mind mode (Sep 6): no model tokens — bash polls, appends to
# docs/night-watch-sep06.md; a Sonnet agent curates the doc once at the end.
# Usage: debug/night_watch_mind.sh [cycles=6] [interval_s=1200]
CYCLES=${1:-6}; INTERVAL=${2:-1200}
REPO=/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy
DOC=$REPO/docs/night-watch-sep06.md
cd "$REPO" || exit 1
source .venv/bin/activate
[ -f "$DOC" ] || echo "# Night watch — Sep 6 2026 (mind mode, bash poller; read-only)" > "$DOC"
for i in $(seq 1 "$CYCLES"); do
  sleep "$INTERVAL"
  {
    echo; echo "## $(date +%H:%M) — cycle $i/$CYCLES"
    python debug/mind_watch.py $((INTERVAL / 60)) 2>&1 | head -14
    if pgrep -f "python machine" | grep -qv "$$"; then echo "process: up"; else
      if [ -f STOP ]; then echo "process: DOWN (STOP file present — not restarting)"; else echo "process: DOWN — starting once"; ./start_impostor.sh >/dev/null 2>&1; fi
    fi
    LOG=$(ls -t event_log/*-event-log.json | head -1)
    python3 - "$LOG" <<'PY'
import sys, json, itertools
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for r in rows[-500:]:
    if r.get('type') == 'llm_api_call' and r.get('prompt_type') == 'caption':
        cue = (r.get('prompt') or '').split('\n')[-1]
        if any(k in cue for k in ('comes back', 'since anyone', 'has changed for', 'been awake', 'keep coming back', 'Something settles', 'Someone is here', 'just moved', ', then ')):
            print('CUE ', r['iso_timestamp'][11:19], '|', cue[:240])
    if r.get('action') == 'echo_spoken_not_stored':
        print('GATE', r['iso_timestamp'][11:19], r.get('reason'), '|', (r.get('caption_preview') or '')[:80])
    if 'Traceback' in json.dumps(r)[:2000] and 'ConnectionReset' not in json.dumps(r)[:2000]:
        print('ERR ', r['iso_timestamp'][11:19], json.dumps(r)[:200])
try:
    d = json.load(open('event_log/mind_thread.json'))
    th = [e for e in d['thread'] if e['kind'] in ('look', 'think', 'reflection', 'wake')][-60:]
    subs = [e.get('subject') or '-' for e in th]
    runs = [len(list(g)) for k, g in itertools.groupby(subs) if k != '-']
    print(f"subject runs (last 60 turns): mean {sum(runs)/max(1,len(runs)):.1f}, max {max(runs) if runs else 0}")
except Exception as e:
    print('thread read failed', e)
PY
    tmux capture-pane -p -t impostor-system:0 -S -300 2>/dev/null | grep -E "MIND\] recall" | tail -4
  } >> "$DOC" 2>&1
done
echo "## $(date +%H:%M) — watch ended after $CYCLES cycles" >> "$DOC"
