"""Runtime monitor — the kinetic bus's window, opened BY machine.py.

The old hand controller opened a Tk window alongside the running build;
its replacement does the same (KINETIC_MONITOR_UI). Read-only truth about
the temperament engine, live: the NOW PLAYING banner with rotation
countdown, the pulled emotion, per-state dataset counts with the playing
one marked, the gaze vector, plus a ⚡ startle test button.

Runs in its own daemon thread with its own mainloop — the proven
hand-controller pattern; every Tk call stays inside that thread. The
full practice room (recording, editing, auditioning) remains the
standalone panel: run it with machine.py STOPPED.
"""

import threading
import tkinter as tk
from tkinter import ttk

from motor_panel.kinetic_bus import STATES


def start_runtime_monitor(bus) -> None:
    def run():
        try:
            root = tk.Tk()
            root.title("kinetic bus — runtime")
            root.geometry("560x460+40+40")
            now = tk.Label(root, text="…", font=("monospace", 12, "bold"), fg="#8a63d2", anchor="w", width=1)
            now.pack(fill="x", padx=8, pady=(8, 2))
            sub = ttk.Label(root, text="", font=("monospace", 9), width=1, anchor="w")
            sub.pack(fill="x", padx=8)
            tree = ttk.Treeview(root, show="tree", selectmode="none")
            tree.tag_configure("active", foreground="#8a63d2")
            tree.tag_configure("empty", foreground="#777")
            tree.pack(fill="both", expand=True, padx=8, pady=6)
            ttk.Button(root, text="⚡ startle (test)", command=bus.startle).pack(fill="x", padx=8, pady=(0, 8))

            marked = {"bundle": object(), "scan": 0.0}

            def rebuild_tree(buckets, active):
                tree.delete(*tree.get_children())
                for state in STATES:
                    fns = buckets.get(state, [])
                    label = f"{state} — no datasets" if not fns else f"{state} — {len(fns)} dataset(s)"
                    parent = tree.insert("", "end", text=label, open=bool(fns), tags=() if fns else ("empty",))
                    for fn in fns:
                        stem = fn[len("session_") : -len(".json")]
                        is_active = fn == active
                        tree.insert(parent, "end", text=("▶  " if is_active else "   ") + stem, tags=("active",) if is_active else ())

            def tick():
                try:
                    s = bus.status()
                    if s["bundle"]:
                        name = s["bundle"][len("session_") : -len(".json")]
                        mins, secs = divmod(int(s["rotate_in"] or 0), 60)
                        now.config(text=f"▶ PLAYING  {name}  ({s['state']})  —  next in {mins}:{secs:02d}")
                    elif s["running"]:
                        now.config(text=f"▶ {s['state'] or '…'} — no dataset assigned, body idle")
                    else:
                        now.config(text="bus stopped")
                    gx, gy = bus.get_gaze()
                    sub.config(text=f"mood: {s['emotion']}   chains: {s['chains']}   gaze: {gx:+.2f}, {gy:+.2f}")
                    import time as _t

                    if s["bundle"] != marked["bundle"] or _t.time() - marked["scan"] > 5.0:
                        rebuild_tree(bus.library.scan(), s["bundle"])
                        marked["bundle"] = s["bundle"]
                        marked["scan"] = _t.time()
                except Exception:
                    pass  # the machine must never die for its mirror
                root.after(500, tick)

            tick()
            root.mainloop()
        except Exception as e:
            print(f"[kinetic monitor] window failed: {e}")

    threading.Thread(target=run, daemon=True, name="kinetic-monitor").start()
