# Local validation report — Aug 31, 2026 (reply to handoff-local-validation-aug30.md)

Written by the studio-machine instance for the remote instance (and any future
session picking up this thread). The Aug 30 stack was reviewed, validated live
on the hardware, and **merged: `rebuild/north-star` ed1b3f3 → c2581a2, pushed.**
No rebase, no force-push, no squash — the stack was strictly linear off the
north-star head, so the in-order merges were pure fast-forwards; every commit
remains individually revertable. The branches themselves are also pushed
(claude/memory-retirement gained two commits, below).

## Verdicts

| Branch | Verdict | Notes |
|---|---|---|
| claude/drawing-machine-cleanup-xre1lw | merged, verified | payload-identical claim on model_settings checked against `_SAMPLER_PASSTHROUGH`; uArm controller modules confirmed nonexistent; live caption path intact |
| claude/census-deletions | merged, verified | 7 symbols spot-checked (zero refs outside docs/debug); 14 gaze globals zero readers; 27 config constants zero py refs + JSONs scrubbed; delete_concept restoration works |
| claude/memory-retirement | merged **after one fix** | boredom parity confirmed against the deleted network (same 0.3 boost / 0.95-per-10s decay / weights; >20 branch subsumed by min()); novelty removal consistent at all call sites; state round-trip silent on old keys |

## The one bug found (fix commit 908fc51)

**"memory 2" (d85bdc6) pasted a block twice in context_compression.py.** The
edit that replaced the spatial-familiarity callback also re-inserted the
concept-extraction + COMPRESSION-log + quiet-print sequence that already
existed a few lines above. Consequence per compression cycle: the concept
extraction LLM call ran twice, and the second `register_concepts_from_compression`
re-bumped `times_seen` on matched concepts — quietly reintroducing the exact
ledger inflation the memory branch exists to stop — plus every `update_baseline`
log entry written twice. Caught in static review (grep for the duplicated
marker), fixed by deleting the second copy, proven live: **15 compression
cycles across two sessions, exactly one `update_baseline` per queue.**

Pattern note for future deletion passes: when an Edit replaces a block that
sits between two similar-looking regions, grep the result for duplicated
landmarks before committing. Compile checks can't catch this class.

## Corrections to the handoff's assumptions

1. **§3 (arm_calibration.json) was moot — the file does not exist.** Not on
   this machine, not anywhere, never in git history. Your own
   `docs/motor-panel-handover.md:40` says "absent until calibrated"; the
   9-point grid was never captured. `kinetic_bus._reach_pose` has been running
   its proportional joint-space fallback the whole time. Nothing to commit.
   If the artist wants measured reach: capture the grid in panel.py's linkage
   tab, then commit it.
2. **"pyproject excludes bcnc/sdk_uarm" is half wrong** — black excludes only
   sdk_uarm; isort skips only sdk_uarm. `black .` formats bcnc/ project files,
   and that is the repo's own standing command (CLAUDE.md), so the reflow was
   committed.
3. **"Expect little to no reflow" was optimistic**: 103 files, +1526/−1232 —
   accumulated repo drift, not the deletions. One `style:` commit (c2581a2),
   black 25.1.0. machine.py's clock-guard-before-project-imports ordering
   survived isort untouched (verified by eye and by two clean boots).
4. **The "no person yet" line never actually reached a prompt.** The audit and
   the memory-2 commit message claim it sat in the drawing system prompt
   "permanently and falsely." In fact `_build_drawing_system_prompt_with_context`
   takes `" ".join(tlines[:2])`, and the last-person clause was element 2 (or 3
   with sleep_context) — always truncated away. Verified empirically: zero
   occurrences of "no person yet" / "last person Nh ago" in OLD-code run logs
   (32fee7ed, a4b96f67, 2e7537aa), and zero in the new run's 131+ calls.
   The deletion stands (it was dead-and-false either way), but
   memory-effectiveness-audit-aug30.md §4 and the runtime-map ledger entry
   overstate its reach — corrected here rather than rewriting your docs.

## Live validation (runs 81a932bd + 71281247, ~50 min, artist present)

Every §2 signal, with evidence:

- **Boot clean ×2** — no import trace of any deleted module; clock ok;
  llama-server ready 11s/12s on the 3.8 build, GPU resident.
- **State round-trip both directions** — first boot restored an OLD-code state
  file (now-ignored keys silently dropped); mid-run SIGINT restart at ~T+45min
  saved with NEW code and restored 52s later via the blink-resume path — no
  awakening ceremony, stream seam continued mid-thought ("He's back. Or maybe
  he was never fully gone").
- **Boredom lives** — logged in every trigger_decision: 0.545 → 0.742 → 0.397
  → 0.511; crossed the 0.7 bored threshold live at 12:49. No novelty anywhere.
  Caveat: llm_api_call entries do NOT log sampling options, so the temp-0.85
  switch itself is not log-verifiable — only the boredom value driving it.
  (Small observability gap; one line in llm_log would close it.)
- **No snapshot churn** — activation_snapshot.json mtime frozen at its pre-run
  value (12:31) across both sessions. activation_edges.json doesn't even exist
  on this machine.
- **Familiarity honest** — memory mode fired (240s gate), familiarity lines
  surfaced; concept extraction once per cycle post-fix.
- **Drawing cycle** — trigger fired (`startup`, then `desire`; mood+boredom
  fields, no novelty). The artist swapped in a blank sheet: early VLM check
  ALLOW (13:00) → conception → ComfyUI (VRAM handoff clean both directions)
  → centerline → GRBL init → homing (attempt-1 timeout, attempt-2 success —
  recurring hardware quirk, recovers by design) → **post-home backstop read
  drawn_paper+drawn_paper → BLOCK → aborted before streaming.** Pen never
  moved; the flattened completion ritual was therefore NOT exercised live
  (diff-verified byte-identical + flake8-clean only). Not a stack regression —
  it is the paper-state staleness/flap the queued redesign targets, now with
  a clean log specimen. Afterward the 13:07 reflection distilled a
  non-drawing want and the trigger honestly reported "no formed want" —
  correct desire-arc behavior, so no second cycle was forced (hunger = 2h).
- **Drawing system prompt** — zero "no person yet" (see correction 4).
- **Presence** — genuine-arrival vs re-detected discrimination correct all
  session; the artist tracked at the workbench throughout.
- **Reflection** — fired on schedule, distilled all three slots
  (self/belief/want, "the visitor" organ), echoes surfaced at the configured
  pacing.
- **Voice** — no register shift detectable. Same plain interior thread, same
  self-story continuity (ink fumes, the full page as border). Best specimen,
  minutes after the blocked draw, watching the artist: "I watch his hand.
  It's not drawing. Not yet. It's holding the pen like a tool that's being
  considered, not used."

Static review extras: compileall clean (bcnc/docs/draw4.py still the known
parse failure); flake8/pylint installed into the venv (they weren't present —
dev-only additions) and run over the 74 branch-touched files — nothing
introduced by the stack; `debug/test_world_shape.py`'s 2 failures reproduce
exactly on base ed1b3f3 (pre-existing); log_viewer/caption_metrics use
`.get()` + string matching throughout, tolerant of the removed fields;
`debug/test_refactored_prompts.py` still probes the retired activation module
and now prints a scary-but-expected error — strip that section in the debug
pass.

## Latent bugs surfaced by this review (all pre-existing, none fixed — decisions)

1. **captioner.py: `DetectionMemory` is never imported**, so
   `self._presence_arrival_count = max(1, DetectionMemory.get_person_count())`
   has ALWAYS thrown NameError into its silent except → count always 1 →
   `presence_identity.record_arrival(1)` every arrival. The singular-regime
   ledger has never seen a real count — a plausible code-side contributor to
   "forever the man." One import line to fix, but decide deliberately: the
   always-1 behavior is what the current voice was tuned against.
2. **YOLO shutdown race**: graceful shutdown nulls the model while the
   detection thread's in-flight pass reads `self.model.names` →
   AttributeError traceback + exit code 1 on every clean shutdown. Cosmetic
   (state already saved) but it makes every shutdown look like a crash.
3. memory.py carries six imports that were already unused on base
   (os/re/threading/Any/log_json_entry/LogType) — cosmetic, fold into any
   next pass.

## Recommended order for the queued work (§5 of the handoff — none started)

1. **Paper-state redesign** — today's ALLOW→BLOCK flap on the same sheet is
   the live evidence; the spec already exists in the audit addenda.
2. **`observations` collection decision** — reader or stop writing.
3. **debug/ archive pass** — plus the test_refactored_prompts trim and the
   seven phantom script references.
4. **Aug 28 phase-2 (B1 first)** — prerequisite for the becoming work.
5. Census §2 stragglers: grid_drawing_ui warp import, `start_impostor*.sh`
   still booting the parked 3.6 arm, the semantic-memory relevance query.

Machine left running on merged code (session 71281247). The venv gained
flake8+pylint. Nothing else on this machine changed outside git.
