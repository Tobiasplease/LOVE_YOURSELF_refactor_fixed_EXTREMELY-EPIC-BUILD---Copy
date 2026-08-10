# Phase 0 — Open-vocabulary feasibility scan

**Date:** 2026-08-05 · **Model:** YOLO-World-S (`yolov8s-worldv2.pt`, 26 MB) · **Frames:** 30, sampled across runs `873a6770` / `ccd4cfbc` / `43b17d9b` (Aug 3–5) · **Device:** CPU only — 0.06 s per frame warm, ~4 s to recompile a 12-term vocabulary.

Vocabulary drawn from the machine's actual monologue logs (`live_captions.txt` noun-phrase frequency) plus the session-brief candidates. Two rounds: original terms, then appearance-rewrites of the failures. Hit rate = fraction of frames with at least one detection at that confidence. Crops were eyeballed for the key terms — a "hit" below means the box actually contained the right object, not just that the detector fired.

## Verdicts

### Text-findable — promote-ready vocabulary

| term | hit @0.15 | max conf | crop check |
|---|---|---|---|
| wooden mannequin torso | 93% | 0.98 | ✔ the small wooden figure on the workbench |
| pink shelf | 87% | 0.86 | ✔ the pink shelving unit |
| coffee mug | 77% | 0.80 | ✔ |
| desk lamp | 77% | 0.76 | ✔ |
| red foam finger | 73% | 0.73 | ✔ clean crop of the finger sign |
| mannequin head | 73% | 0.41 | ✔ heads on the shelf — real but low-conf; needs a ~0.10 threshold |
| wooden chair | 70% | 0.97 | ✔ |
| computer monitor | 57% | 0.45 | ✔ |
| laptop | 57% | 0.90 | ✔ |
| fan | 57% (as `fan`) / 87% (as `pedestal fan`) | 0.87 | ~ real fan found, but `pedestal fan` also fires on cable coils and motion smears |
| wire basket | 33% | 0.66 | ✔ |
| book | 17% | 0.58 | ✔ fires when a book is held — an event, not a fixture |

### Naming lessons (the point of Phase 0)

- `electric fan` 0% → `pedestal fan` 87%, same object. CLIP matches **appearance language**, not function language.
- `LED sign` / `neon sign` / `red neon sign` all 0% — but the LED sign is a separate object that was **not in the room** during these frames, so this is absence, not a naming verdict. Re-test when it's back. (Structural lesson: in the scan data, absence / bad name / undetectable all read as the same zero — only presence-knowledge distinguishes them.)
- `strange wooden structure` matched blurry shelving at 0.07 — judgement words fail exactly as predicted.
- `sculpted human head` at 0.26 was a **live person's head**. Any head/face-like term must be cross-checked against the person tracker before entering the registry.

### Needs YOLOE visual prompt (box once, track as "that thing")

- **robotic arm** — 0% under every name tried (`robotic arm`, `robot arm`, `mechanical arm`). Too assemblage-like.
- **face casts on the lower shelf** — `human face mask` 0%, head-terms hit real people instead.
- **black curtain** — 27% @0.05 only; real but too weak to rely on.
- **power drill / red power tool** — ≤10% @0.05.
- **wooden crate / plywood box** — weak and sloppy.
- **the hanging wooden figure** (upper left) — all `wooden mannequin torso` crops went to the *desk* figure; the hanging one was never cleanly isolated.

### Ghosts (fail by design — log the looking, feed it back later)

`the hole`, `the wound`, `hole in the ceiling`, `crack in the wall`, `tangle of cables`, `cables`, `strange wooden structure`.

### Untestable — object not in the room during these frames

`LED sign`, `neon sign`, `red neon sign`. Zero hits are expected and prove nothing about the terms. Re-scan when the sign returns.

## Operational notes for Phase 1

1. **CPU-only is viable.** 60 ms/frame warm; the 3090 stays untouched. VRAM open question: closed.
2. **Vocabulary compile is the only real cost** (~4 s CPU) — pay it on promotion events, exactly as planned.
3. **Settle-gating is load-bearing, not a nicety**: the 0.87 `pedestal fan` hit was a motion-smeared sliver. Confident garbage on moving frames is normal.
4. **Thresholds are per-term.** Strong fixtures live at 0.3+; mannequin heads live at 0.10–0.15. Store a per-term floor in the registry rather than one global conf.
5. **Person suppression**: drop any detection whose box substantially overlaps a person-tracker box.

Full numbers: `results.json` (round 1) and `round2/results.json`. Crops: `crops/`, `round2/crops/`. Visual report: `report.html`.
