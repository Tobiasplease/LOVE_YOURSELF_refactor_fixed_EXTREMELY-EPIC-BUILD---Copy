# Motor Panel & Motor-System Consolidation — Session Handover

Written July 12 2026, on branch `rebuild/north-star`, after the panel-building
sessions (commits `098f552`, `fbaa498`, `90f9540`). This is the working
reference for the motor-consolidation thread so it can run in parallel with
the warp-calibration thread. Runtime perception/caption work is documented
separately in `docs/runtime-map.md` — the panel is **fully standalone** and
none of this touches the running build.

## 1. What this is and where it's going

`motor_panel/` is a standalone control surface for every servo and stepper in
the machine (uArm excluded), built as the foundation for replacing the "blind
disconnected idle movement" systems with **recorded, learned choreography**:

- Today: perform each subsystem on its own workspace, layer takes like a
  looper pedal, train markov chains over joint recordings, generate endless
  in-style movement.
- Next: sessions named per emotion state become **temperament bundles**; the
  runtime mood system picks which bundle the whole body lives inside
  (see §8 roadmap and the LLM→kinetic design discussion — mood read in
  `captioner/context_compression.py::_mood_read`, July 10).

Run it with machine.py STOPPED (serial ports are exclusive):

    python motor_panel/panel.py

Nothing auto-connects; unconnected devices simulate and log `[sim]` lines to
the console pane, so the whole panel works with zero hardware.

## 2. Architecture (files)

| File | Role |
|---|---|
| `motor_panel/devices.py` | UI-free actuator layer: `Channel`/`SerialDevice`, wire protocols, ordered writer queues, smoothing, inversion. **This becomes the runtime kinetic bus's actuator layer.** |
| `motor_panel/panel.py` | Tkinter view: device frames, GRBL frame (reader/writer/pipeline), workspaces, session UI |
| `motor_panel/arms_markov.py` | Channel-agnostic engine: `Recorder`, `train()`, `Generator`, `Player` (ease vs plan channel families) |
| `motor_panel/session.py` | `Track`/`Session`/`Transport` looper; group topology; legacy hand-dataset import |
| `motor_panel/motor_directions.json` | persisted per-motor direction flags (rev checkboxes) |
| `motor_panel/arm_calibration.json` | 9-point left-arm square calibration (created by the UI, absent until calibrated) |
| `grbl/warp_calibration.py` + `debug/warp_calibrate.py` | the measured warp map for the drawing arm (parallel thread — see §7) |

Design rules that were paid for in debugging blood; do not regress them:

1. **One writer thread per serial device, strict FIFO.** Thread-per-event +
   a lock serializes but does NOT order — that was the original both-arms
   flailing. Continuous targets coalesce latest-wins per key.
2. **The GRBL port has exactly one reader thread** which classifies lines:
   `<...>` status → position (instantly), everything else → response queue
   for the writer. The writer never reads. `?` is injected out-of-band
   (realtime char, **no newline** — `?\n` makes GRBL also answer a blank
   line with a stray `ok`, which desyncs every subsequent response).
3. **No modal G-code state, ever.** `G90` asserted at connect/unlock; jog and
   goto are computed absolute. A single out-of-order `G91` strands the
   controller in relative mode and every absolute target becomes a lunge.
4. **Homing is sacred**: `$H` gets a 60s quiet wait; GRBL doesn't drain
   serial during homing and anything sent meanwhile shreds its 127-byte RX
   buffer into permanent parse garbage.
5. **Live motion = `G0` rapids; playback/generation = `G1 F` from recorded
   timing.** Deployment/homing/node-jumps are fast because they're rapids;
   G1 at any F is the deliberate drawing gait. (This one letter was the
   answer to weeks of "why is dragging slow".)
6. **Paths are streamed, not endpoints**: drag waypoints go into a bounded
   path deque (decimate-under-pressure, never latest-wins) with byte-budget
   pipelining (~120 bytes in flight) so the planner has lookahead and fast
   gestures arrive as *shapes*.

## 3. Actuator inventory (wire truth)

| Device | Port @ baud | Protocol | Channels (lo-hi, neutral) |
|---|---|---|---|
| lunggaze | `/dev/arduino_lunggaze` @9600 | `PAN:{d}` `TILT:{d}` `LUNG:{d}` | pan 45-135/90, tilt 65-150/107, lung 60-110/85 |
| lefthand | `/dev/arduino_lefthand` @9600 | `HAND,f0,f1,f2,f3` (composite), `SERVO,4/5,{d}`, `MOOD,{e}`, `LEFT_ARM_ENABLE/DISABLE` | fingers 0-180/90; elbow+shoulder from config `LEFT_ARM_*_LIMITS` (70-110/90, exhibition-proven) |
| lightbulb | `/dev/arduino_lightbulb` @9600 | `B:{0-255}`, `F` flash | brightness |
| grbl CNC | `/dev/arduino_cnc` @115200 | pre-1.1 fork G-code: `G0/G1`, `M3 S{n}` pen (up 34 / down 52), `$H`, `$X`, `?` realtime | x/y clamped into the MEASURED reach polygon (`grbl/warp_calibration.py MEASURED_BOUNDARY`, walked July 20 — see §7) |

Left-arm firmware (**critical, likely still pending**): `arduino_src/` has
variants. The **"fixed"** variant has NO `SERVO` handler — the arm is driven
by an internal random wanderer (70-110°, 0.3-1° steps; `LEFT_ARM_MAX_RANGE=40`
is literally the user's observed "40 degree area"). Direct control requires
flashing **`hand_controller_clean.ino`** (SERVO handler, 5ms/° slew, no
wanderer). Identify what's flashed: `python debug/identify_hand_firmware.py`
(reads the boot banner: "Improved Left Arm Control" = wanderer variant,
"Direct Servo Control" = the good one).

Device-layer features: per-channel `smooth` easing (arm servos; time constant
= the "smoothing s" slider, a future per-emotion parameter), per-motor `rev`
inversion at the wire level (logical values — and therefore all takes/chains —
are unaffected by mounting flips), config-driven limits.

## 4. Workspaces

- **right arm — bed**: the MEASURED reach envelope drawn to scale (July 26:
  uniform mm-per-px, aspect-true — the walked polygon filled, the 0.5-unit
  clamp inset dashed, 20-unit grid, 0,0 marked). ✛ commanded target vs
  ● reported machine dot (10Hz status polls), 10s fading trail, drag = G0
  rapids. **Every target — drag, jog, playback, generation — projects into
  the polygon via `clamp_to_reach()`** (shared with the drawing pipeline;
  convexity means straight moves between clamped points stay inside), so
  the panel physically cannot command past a joint stop. Max-feed slider
  caps playback/generation only (rapids ignore F).
  **Pen (July 19): hold the right button to put the pen down** — it draws;
  pen-down drags switch to G1 at the tempo you're performing (pen-up drags
  stay rapids), and the trail/indicator turn ink-white while down.
- **left arm — linkage**: SQUARE pad (joint-space by default: x=shoulder,
  y=elbow, corners = extremes) with the stylized skeleton beside it. `mapping:`
  toggles to **calibrated** after the 9-point physical-square capture
  (Calibrate 9-pt → drive wrist to each prompted point → Set; bilinear between
  captured poses; persists `arm_calibration.json`). Range % sliders compress
  joint-space sensitivity; `S_SIGN`/`E_SIGN` in `LinkageView` flip skeleton
  drawing only (rev flags fix physical direction).
- **hand**: the full cursor paradigm from the original hand controller
  (July 19 rebuild — the old four-column drag pad was too limited). A free
  cursor over four finger home columns: each finger follows the cursor's
  height in proportion to horizontal proximity (gravity/wave field) and
  relaxes to the default curl outside it. Side-panel knobs = the original's
  proven parameters (sensitivity 3.0, wave strength 2.0, gravity width 0.4,
  default curl 90°, range ±60°, reverse vertical). Narrow gravity ≈ the old
  single-column feel; wide gravity sweeps the whole hand. Keys w/s e/d r/f
  t/g lock single fingers (2°/tick) while held, releasing back to the field.
  Control engages only while the pointer is over the canvas — leave and the
  hand holds its pose (playback/generation own the channels). Legacy dataset
  import lives in the side panel.
- **lung**: breathing strip — drag vertically, 12s scrolling waveform.

Layout (July 19): the window sizes itself to the screen; devices + GRBL live
in one left column, the session frame gets the rest, and the console/action
bar pack bottom-first so buttons can never be clipped. Workspace canvases
scale from screen size (1920×1080 → bed 520² true-square, others ~970×520);
each tab is canvas + side control panel. `debug/test_panel_layout.py`
verifies the whole panel fits the screen headlessly — run it after layout
changes.

## 5. The session looper (recording model)

- **Track** = channel subset (right arm x/y; pen; left arm elbow/shoulder;
  hand finger0-3; lung). Per-track: **arm** (record-enable: armed tracks
  capture during the next Record pass), **mute** (excluded everywhere),
  **group**.
- **Pen layer (July 19)**: the pen (M3 S) is its own track on a third
  channel family — **step channels**: recorded continuously like everything
  else but emitted ONLY on value change, never interpolated (a half-lowered
  pen drags) and never streamed (each M3 barriers the GRBL writer queue).
  The old "always pen-up during choreography" rule is now scoped: transport
  forces pen up unless the pen track is armed or holds an unmuted take.
  Default group solo; **group it with the right arm to train a joint chain
  that learns WHERE the pen draws** — generation then only lowers the pen in
  regions you demonstrated. `debug/test_pen_layer.py` proves the step
  semantics closed-loop (player/trainer/generator/transport).
- **Record pass** records exactly one loop length (15-60s); unarmed tracks
  with takes play back during it — that's layering. Countdown in status line.
- **Groups (A/B/solo)** are a *training-time* concept: tracks sharing a letter
  train into ONE joint chain (they move in relation — collision safety and
  correlation come from the choreography, provably: generation can only visit
  demonstrated states and transitions). Solo tracks get independent chains.
  **∿ Generate runs every chain simultaneously.** Regroup + retrain any time
  without re-recording.
- **Speed slider** (0.25-2×) retempos playback and generation.
- Sessions persist as `movement_recordings/arms/session_{name}.json`
  (`format 4.1_session_groups`); name them per emotion state.
- **Legacy hand datasets** (`movement_recordings/*.json`, 13 present, per
  emotion) import as hand-track takes via the hand tab (tiled to the loop;
  originals untouched).

Engine details (`arms_markov.py`): 20Hz sampling; states discretized per
channel (`DEFAULT_BINS`); first + second-order transitions with per-transition
timing; **ease channels** (servos — substep-interpolated) vs **plan channels**
(x/y — one command per transition, GRBL planner does the easing;
`PLAN_CHANNELS` constant).

## 6. Verified vs hardware-pending

Verified in simulation/closed-loop (all repeatable via debug scripts):
writer ordering+coalescing (500-event storm → 1 write, strict order), GRBL
pipelining (3+ in flight, non-motion barrier), path preservation (100/100
waypoints of a 1s circle, 13/12 sectors covered), alarm/homing discipline
(0 commands during simulated 7s home), grouped generation (3 chains
concurrently on correct channels), inter-arm correlation preservation
(0.27° mean error vs discretization bin 1-2°), legacy import (real file).

Hardware-validated by the user: right-arm dragging/trail/recording basics,
alarm flow, arm direction flags. **Pending on hardware**: clean-firmware
flash for the left arm; a full record→overdub→generate session with real
takes; warp calibration (in progress, parallel thread).

Filed issues:
- GRBL small/fast moves (rapid circles, corrections) still slower than the
  machine's evident capability — likely `$120/$121` acceleration (short
  segments live inside the accel ramp) or USB latency timer
  (`/sys/bus/usb-serial/devices/*/latency_timer`, 16→1). Deliberately parked.
- `MAX/RX_BUDGET` pipelining adds ~2 segments of finger-to-machine lag; a
  knob if it bothers performance feel.
- Panel + machine.py cannot run at once (ports); acceptable by design.

## 7. Parallel thread: measured warp calibration (drawing arm)

Status at handover: toolkit shipped + proven in simulation; **first hardware
run imminent**. The legacy quad+band-aids cannot represent the arm's curved
distortion field (perfect 4-corner calibration still bows 0.76mm in sim);
the measured 25-point TPS draws a true square (0.18mm bow, 0.5% side spread;
robust to 0.5mm click noise). Workflow: `debug/warp_calibrate.py`
`--run` → photograph → `--measure photo.jpg` (click 4 corners + 25 dots) →
`--square` ruler test. Output `grbl/warp_calibration.json` auto-routes ALL
drawing gcode (`warp_transform_line`) through the fitted map; delete the
file to revert. **Operational requirement: paper position on the bed must be
consistent (mark it) — the map is command→ink-on-bed.**

Interaction with the panel (July 26): the warp thread's measured envelope
now BOUNDS the panel. `ARMS_DUET_ZONE` is retired; `clamp_to_reach()` /
`reach_polygon()` (public API in `grbl/warp_calibration.py`, extracted from
the TPS `_clamp_command`) project every panel target into
`MEASURED_BOUNDARY` with the same 0.5-unit margin + 0.35 hysteresis shell
the drawing g-code uses. `debug/test_reach_clamp.py` proves inside-pass /
outside-project / hysteresis / convex-segment properties. Still open: the
deeper unification — expressing the bed workspace and duet takes in *paper
mm* through the TPS map itself, so "where the pen draws" and "where the arm
performs" share one physical frame (the left arm's calibrated square
already lives in physical coords).

## 8. Roadmap (consolidation thread)

1. **Flash clean left-arm firmware**, re-verify linkage feel; tune
   `LEFT_ARM_*_LIMITS` to true mechanical range by slider-creeping.
2. **First real temperament**: record a full session (arms grouped, hand from
   legacy import or performed, lung breathed) per emotion state; iterate on
   what generation feels like. This is artistic work only the user can do.
3. **Gaze as a parameter track** (design settled, unbuilt): do NOT record
   pan/tilt — record nudges to the behavior engine's parameters
   (`PHYSICS_PATTERNS` shape: mass/spring/damping/tremor/orbital + pause
   cadence). Needs a new track *type* (parameter vs motor) in session.py and
   a workspace with parameter sliders recorded over the loop.
4. **Lightbulb track** (trivial: brightness channel on the existing engine).
5. **Runtime kinetic bus**: lift `devices.py` + `Generator` into the running
   build behind the mood system — mood read picks the session bundle; the
   markov generators replace `organic_left_arm.py`, the firmware wanderer,
   `grbl/idle_movements.py`, and the hand interface's generation loop.
   Port arbitration (panel vs runtime) becomes moot once the runtime owns the
   devices and the panel becomes a "practice room" mode of the same stack.
6. **Retire superseded systems** once 5 lands (legibility directive: fully,
   with runtime-map updates): the 5400-line hand interface's generation path,
   organic_left_arm, idle_movements' Lissajous wanderer.

## 9. Debug/verification tools

| Script | What it proves |
|---|---|
| `debug/identify_hand_firmware.py` | which hand firmware is flashed (banner + SERVO echo) |
| `debug/test_panel_layout.py` | panel layout fits the screen (headless, no window shown) |
| `debug/test_pen_layer.py` | pen step-channel semantics: on-change-only through play/train/generate |
| `debug/test_reach_clamp.py` | reach clamp: inside-pass, outside-project, hysteresis, convex segments |
| `debug/test_face_tracking_stability.py` | gaze servo closed-loop stability (separate thread) |
| `debug/warp_calibrate.py` | the whole warp workflow (`--run/--measure/--square`) |
| `debug/test_warp_calibration.py` | warp method proof vs simulated 2-link arm |
| `debug/compare_yolo_models.py` | detector eval (separate thread) |

The panel itself is the best debug tool: every wire line (real or `[sim]`)
appears in the console pane. When something misbehaves, read the console
before the code — GRBL's error strings + the July 12 logs solved five
layered bugs (modal stranding, fork dialect, homing shred, `?\n` desync,
comma-format position parse) exactly that way.

## Addendum — warp campaign close-out (July 21 2026)

State: 37-point calibration live; landscape window 225×159mm at 0°
(square to the observer); command-space safety clamp (`_clamp_command`,
uniform 0.5-unit `_offset_polygon` margin — NOT centroid scaling, which
over-pads far vertices ~4x). Bottom edge draws straight; the bottom-right
window corner draws with a deliberate cut (~30mm) reflecting the map's
measured belief.

FILED MYSTERY — the bottom-right corner: statically reachable (pen hovers
correctly at command (77,-7.5), elbow teetering on its inert home switch),
but under continuous motion the ink lands short/droops. Manual-pin
experiment (asserting the static observation as a calibration pair) brought
the droop BACK — i.e. the auto-merged motion-context dots are the truer
witness for drawing. Static-vs-dynamic discrepancy at one pose ⇒ prime
suspect is DIRECTION-DEPENDENT BACKLASH (listed, unmodeled, in the old
WARP_TRANSFORM_README). Not chased further by agreement. If reopened:
probe by approaching the same corner command from 4 directions with dots.

Ruled out along the way, with receipts: mechanical reach (user probes),
kinematic conditioning (Jacobian uniform ~10mm/unit everywhere), grinding
via limit switches (hard limits off, switch electrically inert), clamp
artifacts (fixed twice: raw/projected zigzag, then centroid-inset
over-padding). Also learned: command units cost ~10mm paper each — size
all safety margins in command space accordingly.

### Corner mystery — SOLVED IN PRINCIPLE (July 21, late)

The -50mm window-shift experiment closed the case: the cut FOLLOWED the
window's corners into fresh command territory (both reversal corners at
once, including commands far outside the old "droop zone"). Not reach
(user-proven), not the calibration (immune to every change all night):
**BACKLASH AT DIRECTION REVERSALS.** Corners are where a motor reverses;
mid-edges never skip; static hovers approach one-directionally and land
true. The old WARP_TRANSFORM_README listed direction-dependent backlash
as known-unmodeled — it was the answer the whole time.

Next session, 10 minutes:
1. Measure: pen-down line +X 20 units, retrace -X back; repeat for Y;
   the endpoint mismatch IS the per-axis backlash in ink. (Both axes,
   both directions; $100=30 vs $101=60 steps/mm — expect asymmetry.)
2. Compensate in gcode generation: track per-axis direction; on reversal
   add the measured backlash to all subsequent coordinates until the next
   reversal (GRBL fork has no native comp). Apply in warp_calibration
   emission paths + drawing pipeline.
3. Then the full 225x159 window (and likely the pre-clamp corners too)
   draws with true 90-degree corners.
