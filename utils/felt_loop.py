"""Felt loop (Sep 5 2026): the felt state as a driver of MANNER, mechanically.

The compressor reads the machine's last thoughts every two minutes and answers
how it feels (its own words), how pleasant, how much energy. Those became a
line in the frame and a nudge on temperature — and measured across a night,
drained and stirred captions differed by nothing but a little length. These
helpers map arousal to cadence, budget and short-beat odds, and valence to
the kind of thought the quiet elicitation invites. No words are added; the
manner follows the state the way a tired person talks slower and shorter.
"""

from typing import Optional


def _read() -> Optional[dict]:
    try:
        from captioner.context_compression import context_compressor

        return context_compressor.get_last_mood_read()
    except Exception:
        return None


def _t(arousal: float) -> float:
    """arousal 0.1 (drained) → 0, 0.8 (charged) → 1, clamped."""
    return max(0.0, min(1.0, (float(arousal) - 0.1) / 0.7))


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def cadence_mult(read: Optional[dict] = None) -> float:
    """Multiplier on the quiet/rest caption interval. 1.0 when off or unknown."""
    try:
        from config.config import FELT_CADENCE_MULT_CHARGED, FELT_CADENCE_MULT_DRAINED, FELT_LOOP_ENABLED

        if not FELT_LOOP_ENABLED:
            return 1.0
        read = read if read is not None else _read()
        if not read:
            return 1.0
        return _lerp(FELT_CADENCE_MULT_DRAINED, FELT_CADENCE_MULT_CHARGED, _t(read.get("arousal", 0.35)))
    except Exception:
        return 1.0


def budget_scale(read: Optional[dict] = None) -> float:
    try:
        from config.config import FELT_BUDGET_SCALE_CHARGED, FELT_BUDGET_SCALE_DRAINED, FELT_LOOP_ENABLED

        if not FELT_LOOP_ENABLED:
            return 1.0
        read = read if read is not None else _read()
        if not read:
            return 1.0
        return _lerp(FELT_BUDGET_SCALE_DRAINED, FELT_BUDGET_SCALE_CHARGED, _t(read.get("arousal", 0.35)))
    except Exception:
        return 1.0


def short_beat_delta(read: Optional[dict] = None) -> float:
    try:
        from config.config import FELT_LOOP_ENABLED, FELT_SHORT_BEAT_DELTA_CHARGED, FELT_SHORT_BEAT_DELTA_DRAINED

        if not FELT_LOOP_ENABLED:
            return 0.0
        read = read if read is not None else _read()
        if not read:
            return 0.0
        return _lerp(FELT_SHORT_BEAT_DELTA_DRAINED, FELT_SHORT_BEAT_DELTA_CHARGED, _t(read.get("arousal", 0.35)))
    except Exception:
        return 0.0


def elicit_lean(read: Optional[dict] = None) -> Optional[str]:
    """Which quiet-elicitation kind the valence leans toward: 'feel' when
    unpleasant, 'want' when pleasant, None in the middle (rotation as before)."""
    try:
        from config.config import FELT_LOOP_ENABLED, FELT_VALENCE_LEAN

        if not FELT_LOOP_ENABLED:
            return None
        read = read if read is not None else _read()
        if not read:
            return None
        v = float(read.get("valence", 0.0))
        if v <= -FELT_VALENCE_LEAN:
            return "feel"
        if v >= FELT_VALENCE_LEAN:
            return "want"
        return None
    except Exception:
        return None
