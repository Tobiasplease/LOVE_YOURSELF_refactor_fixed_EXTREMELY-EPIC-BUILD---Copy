# Handoff: local review + validation of the Aug 30 cleanup (for the studio machine instance)

You are a Claude instance on the studio machine with the repo, the hardware,
and the venv. A remote session (Aug 30–31 2026) audited and trimmed the
codebase on a stack of three branches. Nothing has been merged. Your job:
review, validate on the real machine, then merge — in order, stopping at any
rung that fails. The artist is reachable but not watching; work autonomously
through the checklist and produce one findings report at the end.

## 0. Orientation (do this first)

```
git fetch origin
git log --oneline origin/rebuild/north-star..origin/claude/memory-retirement
```

Read, in order:
1. `docs/dead-code-census-aug30.md` — the document of record for everything
   deleted and why (verification method, per-file evidence, keep-lists).
2. `docs/memory-effectiveness-audit-aug30.md` — why the activation network
   was retired; the field survey; the remaining memory decisions.
3. `docs/trim-plan-aug30.md` — the first trim pass + the needs-artist list.
4. `docs/runtime-map.md` header + its Dead/deprecated ledger (updated).

The branch stack (each is one review question):

| Branch | Contents | Question |
|---|---|---|
| `claude/drawing-machine-cleanup-xre1lw` (base: `rebuild/north-star`) | 4 audit docs + "trim 1a–1e": −4,128 lines, zero-behavior-change | Is the documentation right? |
| `claude/census-deletions` (stacked on ^) | "census (a)–(e)": −891 lines, zero-behavior-change, + census v2 doc | Are these really dead? |
| `claude/memory-retirement` (stacked on ^) | "memory 1–2": activation network retired (boredom preserved), MemoryMixin dead state + the false "no person yet" line removed | Is the memory change sound? |

Every commit is deletion-only with a message listing exactly what it removed;
`git revert <sha>` restores any slice independently. `rebuild/north-star` is
untouched. Review adversarially, but know that every deleted symbol was
verified zero-live-callers with dynamic-dispatch checks (getattr, hooks,
prompt registry P(), thread targets, HTTP handlers, config-override setattr,
state round-trips) — the burden for reinstating something is a concrete
caller the census missed.

## 1. Static review (no hardware needed)

On `claude/memory-retirement` (contains the whole stack):

1. `python -m compileall` over the runtime packages (the only pre-existing
   failure is `bcnc/docs/draw4.py`, a broken sample — ignore).
2. **Run the formatters — the remote container had no black/isort.**
   `black . --line-length 150 && isort . --profile black --line-length 150`
   (pyproject excludes bcnc/sdk_uarm). Expect little to no reflow (changes
   were deletions); commit any reflow as one `style:` commit on the top
   branch.
3. `pylint`/`flake8` per pyproject — only on the touched files vs base
   (`git diff --name-only origin/rebuild/north-star...HEAD -- '*.py'`);
   pre-existing warnings elsewhere are out of scope.
4. Grep-audit the riskiest deletions yourself (spot-check, don't redo the
   census): pick 5 symbols from census §1 at random and confirm zero
   references outside docs/debug.
5. Check `debug/log_viewer.py` and `debug/caption_metrics.py` still parse
   logs that LACK the removed fields (`novelty` in decision entries,
   LogType SYSTEM/INTROSPECTION) — they must tolerate absent keys. Old logs
   still contain them; new logs won't.
6. Run the still-live component tests that need no hardware/model:
   `debug/test_clock_guard.py`, `debug/test_stream_gaps.py`,
   `debug/test_event_gate.py`, `debug/test_felt_gates.py`,
   `debug/test_drawing_memory.py`, `debug/test_refactored_prompts.py`,
   `debug/test_world_shape.py` (census §7 lists which tests still resolve —
   category 3 "bit-rotted" ones crash by design, don't run those).

## 2. Live validation (the machine, `./run_38.sh`)

The repo's own law: features fail silently — verify via logs/state files,
never by reading code. Run the machine ≥30 quiet minutes + one visit + one
drawing cycle, on the `claude/memory-retirement` head. Watch for:

- **Boot clean**: no ImportError/AttributeError in the first minute
  (the deleted modules — activation_memory, view_orientation, model_settings,
  spatial_memory/awareness/drawing_inspection — must leave no import trace).
- **Boredom lives**: `[🎨 STATE EVALUATION]` prints `Current boredom:` with a
  nonzero value after a stale stretch; the bored sampling branch
  (temp 0.85 / num_predict 110) engages when boredom > 0.7. There is no
  novelty print anymore — that's correct, not a regression.
- **Familiarity honest**: familiarity lines and memory mode still fire, and
  `times_seen` on concepts now only increments from real caption matching
  (the every-8-captions inflation is gone — over a long run, counts should
  grow noticeably slower than before).
- **No snapshot churn**: `activation_snapshot.json` is no longer rewritten
  every caption (its mtime freezes). `activation_edges.json` was already a
  fossil. Archive both files out of event_log/ if you like — nothing reads
  them.
- **Drawing trigger**: `trigger_decision` log entries appear with mood/
  boredom (no novelty field), and a drawing cycle completes end-to-end:
  ComfyUI → centerline → GRBL → completion stored
  (`[📝] Stored drawing completion in memory`).
- **Drawing system prompt**: enable prompt logging or check llm_log — it must
  NOT contain "no person yet" anymore.
- **Presence**: arrival/departure lines still behave ("He's come in", the
  absence decay) — the presence path lost only write-only attrs.
- **State round-trip**: restart once mid-run; the session restores (old state
  files carry now-ignored keys — that must be silent), awakening runs,
  stream seam resumes.
- **Reflection**: let one reflection fire (~20 quiet min); distillation still
  writes identity slots; reflection echo still surfaces.

If anything above fails: `git revert` the single commit whose message names
that area (or drop to the previous branch rung), note it in the report, and
continue validating the rest.

## 3. One machine-only task (do not skip)

Commit `motor_panel/arm_calibration.json` — it exists ONLY on this machine
(written by `motor_panel/panel.py`, read by live `kinetic_bus.py`), it is not
gitignored, and the remote is the only backup for offline exhibitions. Add it
on whichever branch you're validating and push.

## 4. Merge (only after §1–2 pass)

Merge in order, no squash (the per-commit revertability is the safety net):
```
git checkout rebuild/north-star
git merge claude/drawing-machine-cleanup-xre1lw
git merge claude/census-deletions
git merge claude/memory-retirement   # hold this one if §2 raised doubts
git push origin rebuild/north-star
```
Stopping before the memory branch is a legitimate outcome — say so in the
report rather than forcing it.

## 5. Queued next (do NOT start without the artist's go; list them in your report)

1. **debug/ archive pass** — census §7: delete the 17 bit-rotted scripts,
   move the 42 finished experiments to `debug/archive/`
   (`identity_restore_staging.md` is an incident record — ask first),
   fix the stale doc references (7 nonexistent scripts named in
   arduino docs + config.py comment).
2. **Memory decisions** (memory audit §3): the write-only `observations`
   ChromaDB collection (give it a reader or stop writing — biggest ongoing
   write cost), and `baseline_context` (an LLM call every 8 captions that
   reaches only the drawing prompt + reflection).
3. **Aug 28 audit phase 2** (caption-system-audit-aug28.md §5): durable-ledger
   dedup + in-prompt-confirmation discount (B1), memory-mode 240s → config,
   mention-boost habituation. (Reflection-echo pacing is already fixed.)
4. **Paper-state redesign** — specced in the audit's Part 2 addenda,
   currently doc-only.
5. Small latent bugs from census §2: grid_drawing_ui's broken warp import;
   the semantic-memory relevance query that ignores the concept name;
   `start_impostor*.sh` still launching the parked 3.6 arm instead of 3.8.

## 6. Report back

One message to the artist: per-branch verdict (merge / hold / reverted-X),
the live-validation observations (especially anything that FEELS different
in the voice or behavior — that's the real test), what you committed
(formatting, arm_calibration.json), and which §5 items you recommend next.
