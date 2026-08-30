#!/usr/bin/env python3
"""
Idle Movement Manager — gantry hand-off shims around drawing execution.

The Lissajous wanderer this module used to manage was RETIRED July 28
(superseded by the kinetic bus: recorded temperament, not blind wandering;
startup homing runs in machine.py's awakening). Its subprocess machinery
was deleted Aug 30 2026 — the spawn target grbl/run_idle_movements.py no
longer existed, so the start path was a permanently-failing Popen. What
remains is the live part: pause_for_drawing / resume_after_drawing fire
the kinetic-bus gantry hooks so the bus releases and re-acquires the
gantry around a drawing, and the legacy call sites keep their contract.
Gantry idle motion returns as recorded datasets in bus v2 (port
arbitration). Git history keeps the wanderer.
"""

from typing import Optional


class IdleMovementManager:
    """No-op shim for the retired wanderer; keeps legacy call sites safe."""

    def pause_for_drawing(self) -> bool:
        print("[INFO] No idle movements to pause")
        return True

    def resume_after_drawing(self) -> bool:
        return True  # nothing to resume — quiet success for the legacy call sites

    def stop(self):
        pass


# Global instance
_manager: Optional[IdleMovementManager] = None


def get_manager() -> IdleMovementManager:
    global _manager
    if _manager is None:
        _manager = IdleMovementManager()
    return _manager


def pause_for_drawing() -> bool:
    """Drawing needs the gantry: the kinetic bus releases it (the wanderer
    these calls used to pause is retired)."""
    try:
        from utils import hooks

        if hooks.on_gantry_pause:
            hooks.on_gantry_pause()
    except Exception as e:
        print(f"[WARN] gantry pause hook failed: {e}")
    return get_manager().pause_for_drawing()


def resume_after_drawing() -> bool:
    """Drawing done: the bus re-acquires the gantry (re-homes — the port
    open reset GRBL — which fires the tuck choreography)."""
    try:
        from utils import hooks

        if hooks.on_gantry_resume:
            hooks.on_gantry_resume()
    except Exception as e:
        print(f"[WARN] gantry resume hook failed: {e}")
    return get_manager().resume_after_drawing()


def stop_idle_movements():
    """Stop idle movements (no-op since the wanderer's retirement)."""
    get_manager().stop()
