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

---

## Later that night — the machine judges what it made

Two commits on top of the above: `06c2432` (sheet crop) and `6052c31` (the
review). `0f09ba2` was confirmed on the live rig first — the 22:37 drawing's
ritual homed for **22 seconds** where the old 5s `DEFAULT_CMD_TIMEOUT` would
have thrown, and the whole order landed as designed:

    22:37:20  gcode_execution_complete / completion_ritual_start
    22:37:38  paper_gantry_replayed        15.0s
    22:37:52  finished_drawing_captured    2 frames, view_cleared=True
    22:38:14  completion_homing_complete

### Why the raw capture was never going to work

The vision tower reads an image as **32x32-pixel cells** (`patch_size 16`,
`spatial_merge_size 2`, from `mmproj-F16.gguf`), and `--image-min-tokens 1024`
floors every image at ~1024 of them. A 1280x720 table view is *below* that
floor, so mtmd was already upscaling our frame. Of those 1024 tokens the sheet
got ~184 and the marks themselves ~13. The skew was never the problem; the
sampling was.

Each capture is now written twice — the full frame, and `_sheet.jpg` cropped to
the paper. Per-token ground sampling goes **30x30px -> 15x15px**.

**Not deskewed**, on the artist's call: the trapezoid costs the model far less
than the wasted tokens did.

**The crop unions detection with a nominal box.** The sheet drifts between
drawings (61px in x, 27px in y across the four Sep 10 captures), so a fixed box
either clips or wastes — but naive detection alone would have clipped 1 of those
4 frames, when a shadow ate the right half of `221152_1`. Detection handles the
drift; the nominal box makes clipping impossible. Re-derive it with
`debug/find_sheet_crop.py` when the rig moves.

### Sharpening yes, contrast no — with numbers

Unsharp 0.6 deepens the ink (darkest 2%: 66 -> 59) while the paper median holds
(164 -> 163). That invariant is the whole test: the paper stays the reference
point, so a drawing that came out faint still reads as faint against its own
sheet. Two corrections were measured and **rejected**:

- **percentile contrast stretch** — maps this frame's darkest ink to black, so a
  barely-there drawing arrives looking confident.
- **illumination flatten** — measured 66 -> **125**. Dense hatching drags down
  the very background it divides by, so dividing brightens precisely the
  passages worth seeing. It washes the drawing out to fix a shadow that was not
  the problem.

### The review (`drawing/finished_review.py`)

The Aug 5 note said the critique should return judging the paper, not the
ComfyUI image, and should only run once. Both hold: it reads the crop against
`current_drawing_prompt`, runs inside the ritual's existing completion thread
(which the uArm hook does not wait on, so the sheet is still taken on time), and
publishes to `drawing.last_reflection` so the completion memory records it
without a second pass. **An empty answer is never stored** — saved timeouts are
what discredited the old pass.

Prompts are in the registry (`review.frame`, `review.intent-wrap`,
`review.elicit`), so they are live-editable; the `drawing_review` pass is now
`migrated: True`. The ask is an elicitation, not a fence: nothing presumes a
drawing worth describing, because many are barely legible and that is the
machine's to know.

Live results on the two Sep 10 drawings:

- real intent (a low dense pool of hatching) vs a sheet where the ink landed
  high — *"Not there. The ink is all high up, clustered around a rounded shape.
  ... The dense pool of shadow I intended doesn't exist."*
- matching intent, same sheet — *"Mostly there... but the line is shaky and
  broken, there's a stray mark low on the page"*

It graduates rather than always reaching for "rough sketch".

Also fixed: the completion memory read `get_drawing_info()` after Step 1's
`end_drawing()` had cleared it, so every drawing was remembered as the literal
"Completed drawing a drawing." (open thread 5 above — now closed).

## Still open after this session

1. **`ensure_homed` does a phantom home on every attempt 1.** It sends `$H`
   with `wait_ok=False` then polls `get_status()` with no delay, so the first
   `?` returns the pre-homing `Idle` and homing is declared complete in ~0.7s.
   `G54` then can't get its `ok` inside `DEFAULT_CMD_TIMEOUT` and throws:

       22:33:11  homing_start
       22:33:11  homing_complete  duration=0.70
       22:33:16  homing_exception  Timeout on G54, response=[]  TimeoutError
       22:33:16  homing_retry_delay
       22:33:35  homing_complete  duration=12.48

   Self-healing via the retry, which is why it has been invisible — but it costs
   ~24s and a spurious soft reset on every drawing and every startup. Fix: don't
   accept `Idle` until the machine has been seen to leave it.
2. **Capture at 2560x1440.** The camera offers it over MJPG (enumerated Sep 10;
   YUYV tops out at 1080p). The sheet would be ~1260x520 real pixels instead of
   ~630x260 — 2x the linear detail into the same 1024 tokens. The blocker is
   that `machine.py:194` opens the shared cap once at 720p for the 30fps loop
   and the capture takes the ArUco thread's shared frame, so it needs a
   momentary re-config during the ritual (already a stop-the-world moment) plus
   `CAP_PROP_FOURCC` set to MJPG.
3. **Occlusion at the sheet's edges.** The wooden shoulder takes the
   bottom-right corner and the pen carriage the right side, in every frame. The
   drawing itself is clear, so this is not urgent.
4. Gantry resume hook at Step 7, and `find_paper_gaze_angles.py` — both
   unchanged from above.
