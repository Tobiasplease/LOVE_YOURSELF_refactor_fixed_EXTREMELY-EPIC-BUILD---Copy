# Curtain Rig Simulator

A real-time simulation of a mechanically driven sheer curtain, made to test whether strings, pulleys, spreader rods, a weighted hem
and fans can pass for wind. The left window is the rig. The right window is a plain curtain in simulated real wind,
for comparison. Both get exactly the same gusts at the same moments.

## Open it

On a phone or any browser: the published artifact at https://claude.ai/artifact/DYka3avVnVRiHCB7Qo46jm (private to you unless shared).
On a phone it shows the rig alone on a lighter cloth grid, to stay real-time. Switch ROOM ▸ SHOW to add the reference.

## Run it locally

The page uses JavaScript modules, so it needs a tiny local web server (opening the file directly won't work):

```bash
cd labs/curtain-sim
python3 -m http.server 8000
```

Then open http://localhost:8000 in Chrome, Edge or Firefox. Nothing to install: three.js is in `vendor/`.

Drag to orbit, pinch or scroll to zoom. Space = play/pause, ← = rewind, H = clean view.

## Two kinds of time

- **Transport** (tape deck): pause, rewind, scrub the last 20 s, slow motion, 2×. This is about *watching*. Press PLAY after rewinding
  and the physics carries on from that moment.
- **Cue** (what the motors play): source (gusts / swell / fader), tempo, wave direction, flip, polarity, wave speed, hem lag.
  This is the part that exists in the real rig. Running the motors backwards does not run the fabric backwards: gravity and air only go one way.

## The signal chain

```
gust envelope ──▶ motor (speed + accel limits) ──▶ line ──▶ elastic ──▶ floating pulley ──▶ 2 legs ──▶ rods / hem chain ──▶ cloth
   (shared)                                                                                                       ▲
       └──────────────────────────────▶ turbulent wind field ──────────────────────────────▶ reference curtain ──┘
```

## Files

| File | What it is |
| --- | --- |
| `src/cloth.js` | The fabric: a grid of beads joined by links (stretch, shear, bend), air drag and lift per triangle, rods, hem chain |
| `src/layout.js` | Lays out the grid: columns × rows (top, belly, hem) of bridles, rods and ceiling pulleys |
| `src/cue.js` | The cue engine: turns gusts, swell or the fader into a pull for each line, delayed so the wave travels in the chosen direction |
| `src/metrics.js` | The tent meter: how far each attachment point runs ahead of the fabric 15 cm around it (lower = less pinched) |
| `src/tape.js` | The tape recorder behind rewind and scrub (positions stored compressed, at 30 frames a second) |
| `src/rig.js` | `Motor` (reels line in and out under speed limits) and `Bridle` (elastic line plus floating pulley that splits the pull evenly between two legs) |
| `src/wind.js` | Gust timeline (attack / hold / release envelopes, fixed seed so it repeats) and the turbulent reference wind |
| `src/stage.js` | The room, lightboxes, rails, camera |
| `src/views.js` | How the curtain is drawn (sheer shader: folds stack up and darken against the light) and how strings and pulleys are drawn |
| `src/ui.js`, `src/main.js` | Instrument panel (faders, cue sheet, line matrix) and the loop that ties everything together |
| `media/` | Rendered clips from each step |

`debug/curtain_sim_snapshot.mjs` (in the repo root) renders stills or video frames headless, for review.

## Status

- [x] Step 0–1: room, one curtain, one rig column (belly bridle with two spreader rods, hem bridle on the chain), real-wind reference
- [x] Grid of 3–8 columns × 2–3 rows (default 5 × 3 = 15 lines, 30 attachment points), each line coloured slack / taut / at motor limit
- [x] Transport (pause, rewind, scrub, speed) and cue engine (gusts, swell, fader; tempo, direction, flip, polarity)
- [x] Published as a phone-friendly artifact
- [ ] Direct-to-points vs rods side by side, pulley friction, 4-leg bridles
- [ ] Step 3: fans
- [ ] Step 4: keyframe timeline with tracks per line and easing
- [ ] Step 5: cinematic light (shadows, haze, grain, depth of field, day cycle)
- [ ] Step 6: export to motor commands, Blender bridge

## Findings so far (stop test: one sharp pull, then hold)

Numbers are from the sim at default resolution. Trust the rankings more than the centimetres: they shift with grid and substeps.

| Setup | Hem swing after motors stop | Settles in |
| --- | --- | --- |
| Voile, far pulleys (start) | 22 cm | 4.9 s |
| Gentler motor braking (1 m/s²) | 15 cm | 2.7 s |
| Stiff fabric (10×) | 16 cm | 1.7 s |
| Sway cancel (shaped move, tuned to the 1.8 s sway) | 10 cm | 3.3 s |
| Stiff fabric + sway cancel + gentle braking | 7 cm | 0.8 s |
| Heavier fabric (120 g/m²) | 26 cm | 4.2 s |
| Preload on all lines | 26 cm | 4.1 s |

- Stiffness and gentle, shaped stops are what tame the sway. Weight makes it worse. Preload doesn't help: the sway is an overshoot toward the pulley, where a line can't push.
- Pulleys close in (0.5–1 m) pinch more (tent 5.3 cm vs 4.7) and give less billow. Stiff fabric brings the pinch back to about what voile has with far pulleys, but sway is harder to stop: near-vertical lines hold height, not in-and-out swing.
- A mistuned sway cancel is worse than none. The sway rhythm changes with fabric and pulley geometry.
- Stiffener patches around attachments add little where there is already a rod.

## Known simplifications

- The fabric remembers its hanging pleats: its natural shape is the hung shape, not flat.
- Rods are glued in their channels (in reality the fabric can slide along them).
- No self-collision: fabric can pass through itself in big billows.
- Pulleys have no friction yet.
- Wrinkles smaller than about two grid cells (8–11 cm) can't form, so the sim underestimates pinching, more so for soft fabric.
- The curtain's glow is a shading trick, not real light transport. It doesn't cast shadows yet.
