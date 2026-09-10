# Sep 10 2026 — the uArm paper move, and the machine seeing what it made

Handoff for picking this up in a fresh session. Branch `rebuild/pre-mind`,
HEAD `0f09ba2`, all pushed.

## Where it stands right now

**The running process (PID 630152, started 21:47) does NOT have the last
commit.** It has everything up to `930373a`. The `$H` fix (`0f09ba2`) needs a
restart:

    pkill -INT -f machine.py    # tmux supervisor restarts it in ~5s

Do NOT use `./stop_machine.sh` for a restart — it touches the STOP file and
ends the supervisor loop. llama-server survives a restart (`stop_server()` is
only called around drawings), so there is no model reload.

## What was wrong, and what got built

### 1. The uArm paper move had silently stopped playing (`b0705b5`)

The Aug 30 retirement of the idle wanderer deleted `IdleMovementManager`'s
subprocess machinery, but the post-GRBL hook still probed `manager.process`.
The `AttributeError` raised before `app.play()` and was swallowed by the hook's
broad `except`, which printed one line and moved on. The startup play never
touched that module, which is why it kept working and hid the breakage — and
`debug/test_uarm_grbl_hook.py` has its own copy of the hook with no such probe,
so the test passed the whole time the live path was dead.

Also re-recorded the move for the new scale: `papermovenewest_20260910_193654.txt`,
a clean single grab/release (`ee,1` → `ee,0`); the March take opened with a
stray `ee,0`.

### 2. The machine never saw its own drawings (`0f29fbe`, `7e0cc5d`, `93d0ce6`)

The completion ritual homed, released the gaze from the drawing surface, and
let the uArm discard the sheet. Nothing in the post-draw path ever looked at
the table — the drawing was thrown away unseen.

**Ritual order now (artist's, Sep 10):**

    drawing finishes
    2a  pen UP                      (was welded to $H; had to come out first)
    2b  paper get-clear — SAME take as pre-drawing:
          arms   <- kinetic bus (paper_clear)
          gantry <- replayed on the ritual's own port
    2c  capture image               (waits for the slower half)
    2d  homing ($H)
    3-4 gaze unlock, completion memory
    4.5 uArm hook -> sheet discarded
    5-7 CNC state cleared, gantry re-acquired -> standard movement

New files: `drawing/finished_capture.py`, `grbl/paper_gantry.py`,
`debug/test_finished_capture.py`. Frames land in
`event_log/finished_drawings/finished_<stamp>_<i>.jpg`; last path filed on
`state_manager.last_finished_drawing_image`.

**Capture only, on purpose.** Nothing is asked of the model yet — the plan is
to look at a real capture and decide what post-processing it needs (deskew,
crop to the sheet) before the model sees it. That is the next decision.

### 3. Paper gaze re-tuned (`c485b29`)

80/65 → **90/70** via `debug/find_paper_gaze_angles.py`. The sheet used to sit
at the frame's bottom edge — fine for the gate's yes/no, but it cropped the
drawing. Pan 90 also matches the drawing-watch pan, so the camera no longer
swings sideways between watching the pen and judging the sheet.

Shared by: the paper gate (vlm + aruco), the finished-drawing capture, and the
`"paper"` chosen-glance. Drawing-watch tilt stays separate.

## Three gotchas worth not rediscovering

**The bus cannot move the gantry during the completion ritual.** A recorded
`paper` take's x/y track is dropped twice over: `_send_plan_raw` returns early
while `is_executing_cnc` is set (cleared at Step 5), and `gantry.alive` is
False because the port was released to the drawing pipeline (re-acquired at
Step 7). It fails *silently* — arms move, gantry doesn't. Re-recording the take
cannot fix it. Hence `grbl/paper_gantry.py`, which replays the same file's x/y
onto the serial link `grbl_utils` already holds open. Going through the bus
would mean acquiring the port, which resets GRBL and homes it — exactly what
the ritual defers until after the photograph.

**Keep exactly ONE `paper` take in the live bucket.** `paper_clear()` picks with
`random.choice` while the replay takes the first sorted, so with several takes
the arms and the gantry play different recordings. The replay warns if it sees
more than one. Currently: `session_paper_a.json` (195 moves, 24.5s).

**The pen-up burst leaves 6 unread `ok`s.** Step 2a fires 5× `M3 S.. ; PEN UP`
plus a `G4` dwell, all `wait_ok=False`. Anything that streams with `wait_ok=True`
after it will consume those instead of its own replies and lose backpressure.
This bit twice in one evening:

- the gantry replay outran GRBL's RX buffer → `error: Invalid gcode ID:24`
  against a coordinate that was never the problem. Fixed with
  `reset_input_buffer()` before streaming (`930373a`).
- `$H` had ALWAYS been sent with the 5s `DEFAULT_CMD_TIMEOUT` while homing takes
  ~10s. It only worked because it ate a stale `ok` and returned instantly.
  Draining the buffer exposed it: `$H` timed out, threw out of the ritual, and
  took the gaze unlock, the completion memory and the uArm discard with it —
  the paper move "stopped playing" again for a completely different reason.
  Fixed with `DEFAULT_HOME_TIMEOUT` + its own try (`0f09ba2`).

## Verified working on the live rig

- uArm paper move: `[uArm] Post-GRBL play: papermovenewest_...` → `Movement completed successfully`
- capture timing (event log): `gcode_execution_complete` → `completion_ritual_start`
  → `finished_drawing_captured` → `completion_homing_complete`. Capture lands
  ~30s after the pen stops, ~20s BEFORE homing. Correct.
- gantry replay: `paper_gantry_replayed`, 18s, no error (after `930373a`)
- framing at 90/70: whole sheet corner-to-corner, faint pencil marks legible

## Open threads

1. **Restart to pick up `0f09ba2`.** Next drawing should show `[✅] Homing
   complete` then `[uArm] Post-GRBL play`. If homing fails you should now see
   `[❌] Completion homing failed … — continuing the ritual so the sheet still
   gets taken`, with the paper move going ahead anyway.
2. **Look at a clean capture and decide the post-processing.** First frames with
   the body actually clear are `finished_20260910_221152_0/1.jpg`. Deskew +
   crop-to-sheet is the expected need before any model call.
3. **`[WARN] gantry resume hook failed: No GRBL port found`** at Step 7 —
   `gantry_acquire()` can't open the port because `execute_gcode_file` still
   holds it (the caller closes it after). Structurally fails on every drawing,
   predates this session, but it is the "return to standard movement" end of the
   sequence. Worth chasing.
4. **`debug/find_paper_gaze_angles.py` needs a small fix** — it starts from a
   hardcoded `pan=80, tilt=50` instead of reading the live config, and its
   `Q`-to-save didn't fire. Offered, not done.
5. **Unrelated bug, untouched:** the completion memory at `grbl_utils.py` reads
   `DrawingState.get_drawing_info()` for the description, but Step 1 already
   called `end_drawing()`, which clears it. So every completion memory records
   the literal `"Completed drawing a drawing."`
   `state_manager.current_drawing_prompt` survives that point.

## Data loss to be aware of

Four earlier capture images were deleted by an over-broad cleanup glob of mine
(`finished_2026091*_*.jpg` matched real captures, not just test output). Not
recoverable. `debug/test_finished_capture.py` now writes to a temp dir so it
cannot happen again.
