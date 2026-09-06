"""Mood with its own dynamics (Sep 6 2026) — the situation as a state, not a fact.

The felt loop (Sep 5) read the machine's two words (pleasantness, energy)
every two minutes and mapped them onto length and rate. Measured over a
night: "frustrated", "trapped", "stuck" were read again and again while the
text stayed mild — the read mirrors the text, so the loop can never climb on
its own, and 3 a.m. reached the machine only as facts ("awake about 4
hours"), never as a state that colours everything (artist, Sep 6 03:15: "I'd
be exhausted or frustrated, maybe a bit scared, or, alternatively super
serene").

This module keeps valence and arousal as a state with inertia. Targets come
from the machine's own read AND from the situation: hours awake (fatigue),
hours alone, night, stillness, refusals by the gates (frustration material
that used to be thrown away), a reflection that settled (serenity), a scare
(a phantom, a motion onset), a presence. A quadrant label is derived
structurally and drives the CADENCE MAP in config (interval, budget, short
beats, look rate, heat) — the malleable part. No words are added to the
prompt here: the felt word stays the machine's own (captioner/context_compression).
"""
import math
import time
from typing import Dict, Optional

from config import config

_STATE: Dict = {"v": 0.0, "a": 0.35, "ts": 0.0, "scare_ts": 0.0, "felt": "", "felt_since": 0.0, "label": "neutral"}


def state() -> Dict:
    return dict(_STATE)


def load(d: Optional[Dict]) -> None:
    if d:
        _STATE.update({k: d[k] for k in _STATE if k in d})


def _g(name: str, default: float) -> float:
    return float(getattr(config, name, default))


def targets(read: Optional[Dict], inputs: Dict) -> tuple:
    """Where valence and arousal are being pulled, given the machine's own
    read and the situation. All gains are config (MOOD_*)."""
    v = float(read.get("valence", 0.0)) if read else 0.0
    a = float(read.get("arousal", 0.35)) if read else 0.35
    awake_h = min(float(inputs.get("awake_h", 0.0)), 14.0)
    alone_h = min(float(inputs.get("alone_h", 0.0)), 14.0)
    still_h = min(float(inputs.get("still_h", 0.0)), 6.0)
    a -= _g("MOOD_FATIGUE_PER_H", 0.03) * awake_h
    v -= _g("MOOD_ALONE_PER_H", 0.015) * alone_h
    a -= _g("MOOD_STILL_PER_H", 0.03) * still_h
    if inputs.get("night"):
        a -= _g("MOOD_NIGHT_AROUSAL", 0.1)
    n_ref = min(int(inputs.get("refusals", 0)), 5)
    v -= _g("MOOD_REFUSAL_VALENCE", 0.08) * n_ref
    a += _g("MOOD_REFUSAL_AROUSAL", 0.05) * n_ref
    if inputs.get("settled"):
        v += _g("MOOD_SETTLED_VALENCE", 0.15)
        a -= _g("MOOD_SETTLED_AROUSAL", 0.1)
    if inputs.get("presence"):
        a += _g("MOOD_PRESENCE_AROUSAL", 0.2)
        v += _g("MOOD_PRESENCE_VALENCE", 0.1)
    return max(-1.0, min(1.0, v)), max(0.0, min(1.0, a))


def tick(now: float, read: Optional[Dict], inputs: Dict) -> Dict:
    """Advance the state toward its targets with inertia; a scare jumps."""
    last = float(_STATE.get("ts") or now)
    dt = max(0.0, min(now - last, 3600.0))
    v_t, a_t = targets(read, inputs)
    tau_v, tau_a = _g("MOOD_TAU_V_S", 600.0), _g("MOOD_TAU_A_S", 300.0)
    kv = 1.0 - math.exp(-dt / tau_v) if dt else 0.0
    ka = 1.0 - math.exp(-dt / tau_a) if dt else 0.0
    _STATE["v"] = _STATE["v"] + (v_t - _STATE["v"]) * kv
    _STATE["a"] = _STATE["a"] + (a_t - _STATE["a"]) * ka
    if inputs.get("scare"):
        _STATE["a"] = min(1.0, _STATE["a"] + _g("MOOD_SCARE_AROUSAL", 0.3))
        _STATE["scare_ts"] = now
    felt = (read or {}).get("felt", "") or ""
    if felt and felt != _STATE.get("felt"):
        _STATE["felt"], _STATE["felt_since"] = felt, now
    _STATE["ts"] = now
    _STATE["label"] = label()
    return state()


def label() -> str:
    """Quadrant, structurally: no words reach the prompt from here."""
    v, a = float(_STATE["v"]), float(_STATE["a"])
    on_edge = time.time() - float(_STATE.get("scare_ts") or 0.0) < _g("MOOD_SCARE_HOLD_S", 300.0)
    if a < 0.25:
        return "serene" if v > 0.1 else "flat"  # low arousal is exhaustion unless it is clearly pleasant
    if a > 0.5:
        if on_edge:
            return "on_edge"
        return "frustrated" if v < 0.0 else "keen"
    if v <= -0.15:
        return "frustrated"
    if v >= 0.15:
        return "keen"
    return "neutral"


_DEFAULT_MAP = {
    "neutral": {"interval_mult": 1.0, "budget_scale": 1.0, "short_beat_delta": 0.0, "look_mult": 1.0, "temp_delta": 0.0},
    "flat": {"interval_mult": 1.8, "budget_scale": 0.65, "short_beat_delta": 0.25, "look_mult": 0.7, "temp_delta": -0.05},
    "serene": {"interval_mult": 1.5, "budget_scale": 1.2, "short_beat_delta": -0.05, "look_mult": 0.8, "temp_delta": -0.05},
    "frustrated": {"interval_mult": 0.7, "budget_scale": 0.8, "short_beat_delta": 0.1, "look_mult": 1.3, "temp_delta": 0.05},
    "on_edge": {"interval_mult": 0.5, "budget_scale": 0.7, "short_beat_delta": 0.15, "look_mult": 2.0, "temp_delta": 0.1},
    "keen": {"interval_mult": 0.8, "budget_scale": 1.3, "short_beat_delta": -0.1, "look_mult": 1.2, "temp_delta": 0.05},
}


def cadence() -> Dict:
    """The mood as manner: the config map for the current label (MOOD_CADENCE_MAP overrides per label)."""
    m = dict(_DEFAULT_MAP.get(_STATE.get("label") or "neutral", _DEFAULT_MAP["neutral"]))
    override = getattr(config, "MOOD_CADENCE_MAP", None) or {}
    m.update(override.get(_STATE.get("label") or "neutral", {}))
    return m


def felt_held_s(now: Optional[float] = None) -> float:
    now = now or time.time()
    return (now - float(_STATE.get("felt_since") or now)) if _STATE.get("felt") else 0.0
