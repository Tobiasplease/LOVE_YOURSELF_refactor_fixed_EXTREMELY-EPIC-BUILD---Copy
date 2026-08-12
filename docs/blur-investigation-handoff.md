# ComfyUI blur — investigation handoff (Aug 12 2026)

State dump from the session that ran the Aug 10–12 blur work, for the
session now continuing it. Read this before re-testing anything.

## The measurement method (use it, don't re-invent it)

Every output PNG embeds its EXACT executed workflow (prompt, seed, trigger,
ControlNet params, LoRA, models) in a `tEXt` chunk — the outputs folder is a
~1900-sample natural experiment. Group by embedded params, not by dates.
Edge-sharpness metric used throughout: 97th percentile of gradient magnitude
over ink pixels (`np.percentile(g[g>10], 97)` on the grayscale); "blurry"
threshold < 80. Blur is BIMODAL (sharp vs defocused-photo register), so
judge rates over ≥10 drawings, never single outputs.

## Established facts (don't re-litigate without new evidence)

- **Twelve months of frozen params** (Jul 2025–Jul 20 2026: sketch trigger,
  cnet_end 1.0, impostor-32@1.0, 25 steps) ran at **5–27% blurry** —
  Jul 2025: 5%, Sep 2025: 13%, Oct 2025 + Feb 2026: 27%, Jun–Jul 20: 19%.
  Prompt content is the only variable inside those rows.
- **The July 26 "crisp pipeline" param change** (ink trigger + cnet_end 0.6,
  commit 6edb086) took it to **58%** (n=45, median sharpness 70 vs 135).
  The same-seed A/B that justified it (debug/test_crisp_ab.py) tested ONE
  seed and got the population effect backwards.
- **cnet_end reverted to 1.0** (config, Aug 10) → Aug 12 measured **3/12
  blurry (25%)** = back to baseline. TRIGGER_PROMPT still says "ink"
  (kept as the single remaining delta from the proven config — revert to
  `"impostor black and white sketch line art "` is the next 1-variable test).
- **VRAM/handover conclusively exonerated**: same-seed replay under opposite
  memory pressure → 71.7% bit-identical pixels, mean diff 1.2/255. Partial
  model loading changes speed, not pixels. Logs during blurry gens are
  healthy (bf16, no OOM, no lowvram, normal it/s).
- **Input frame conclusively exonerated** (Aug 12): 750 input→output pairs,
  input-frame sharpness vs output blur r = −0.035, flat across quartiles,
  same when params held fixed. Also `capture_mood_snapshot` writes exactly
  ONE frame (`snapshot_queue[-1]`) — ComfyUI never receives multi-frame
  input, so the video/multi path is not involved.
- **Prompt-side changes so far**: render call rewritten Aug 10 (positive
  craft language, no negation wall, no plotter-meta in emitted prompt —
  prompts.py `render_system`); Aug 12 added an image-quality clause
  ("crisp, high-contrast line art, every stroke perfectly sharp against a
  flat, untextured pure-white background") — **UNTESTED as of this writing**
  (machine barely ran since). Deliberately avoids photo lexicon
  ("lit"/"scan"/"paper") — artist flagged that risk.
- **Artist's position**: the LoRA dataset was heavily filtered to look like
  scans, NOT photos — so "defocus mode baked into LoRA training data" is
  CONTESTED. Origin of the defocus register is open (Flux-base behavior in
  this prompt neighborhood is a candidate). Artist perceives blur as worse
  than the measured baseline; measurements say current rate ≈ historical.

## Open levers, in order

1. Measure the quality-clause effect (≥10 drawings after next restart).
2. Trigger revert to sketch wording (restores the exact 686-sample config).
3. If both fail: the anchor phrase "on white paper", seed-space analysis,
   or LoRA retrain on flattened/rendered data.

## Coordination warnings

- Both sessions have UNCOMMITTED edits in `config/config.py`. This session
  added: `COMFY_CNET_END_PERCENT=1.0` (the revert — do not undo without
  data), `CENTERLINE_TONE_FILLS`, `CENTERLINE_ENGINE`, flat feed rates
  (`GRBL_DRAW_FEED_RATE`/`GRBL_TRAVERSAL_FEED_RATE`), pen-opt default off,
  `GRBL_SERIAL_RECOVERY_MAX`. Also edited: prompts.py (render/intent/
  stocktake), grbl_utils.py (serial revival, bounds-normalized warp),
  warp files, bcnc/svg_centerliner_v2.py + bcnc/dsv_hybrid.py.
- Do NOT drop test generations into the ComfyUI output root — image_monitor
  draws any new PNG there. Use the `tests/` subfolder (expected-prefix gate
  exists but don't rely on it).
- The full history with plots: memory file `project_drawing_arc.md`, and
  `docs/runtime-map.md` §"Physical execution fidelity".
