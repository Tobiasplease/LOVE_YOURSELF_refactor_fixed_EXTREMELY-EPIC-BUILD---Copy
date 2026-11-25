# Curtain Rig Simulator

A real-time simulation of a mechanically driven sheer curtain, made to test whether strings, pulleys, spreader rods, a weighted hem
and fans can pass for wind. The left window is the rig. The right window is a plain curtain in simulated real wind,
for comparison. Both get exactly the same gusts at the same moments.

## Run it

The page uses JavaScript modules, so it needs a tiny local web server (opening the file directly won't work):

```bash
cd labs/curtain-sim
python3 -m http.server 8000
```

Then open http://localhost:8000 in Chrome, Edge or Firefox. Nothing to install: three.js is in `vendor/`.

Drag to orbit, scroll to zoom, and use **Front / Angle / Side** for preset views.

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
| `src/rig.js` | `Motor` (reels line in and out under speed limits) and `Bridle` (elastic line plus floating pulley that splits the pull evenly between two legs) |
| `src/wind.js` | Gust timeline (attack / hold / release envelopes, fixed seed so it repeats) and the turbulent reference wind |
| `src/stage.js` | The room, lightboxes, rails, camera |
| `src/views.js` | How the curtain is drawn (sheer shader: folds stack up and darken against the light) and how strings and pulleys are drawn |
| `src/ui.js`, `src/main.js` | Control panel, signal scope, and the loop that ties everything together |
| `media/` | Rendered clips from each step |

`debug/curtain_sim_snapshot.mjs` (in the repo root) renders stills or video frames headless, for review.

## Status

- [x] Step 0–1: room, one curtain, one rig column (belly bridle with two spreader rods, hem bridle on the chain), real-wind reference
- [ ] Step 2: direct-to-points vs rods side by side, pulley friction
- [ ] Step 3: fans
- [ ] Step 4: full grid of columns, sequencer timeline, travelling-wave helper
- [ ] Step 5: cinematic light (shadows, haze, grain, depth of field, day cycle)
- [ ] Step 6: export to motor commands, Blender bridge

## Known simplifications

- The fabric remembers its hanging pleats: its natural shape is the hung shape, not flat.
- Rods are glued in their channels (in reality the fabric can slide along them).
- No self-collision: fabric can pass through itself in big billows.
- Pulleys have no friction yet.
- The curtain's glow is a shading trick, not real light transport. It doesn't cast shadows yet.
