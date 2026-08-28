"""
caption_monitor.py
------------------
Live caption display window (tkinter). Launched automatically by machine.py,
or run standalone while machine.py runs:

    python debug/caption_monitor.py

Captions flow as one continuous text on a book-like page: a fixed-width
column centered on a grey ground, older text fading back, the newest
caption bright. The view stays pinned to the newest words.
"""

import os
import sys

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ImportError:
    tk = None

_LOG_DIR = os.getenv("MOOD_SNAPSHOT_FOLDER", os.path.join(os.path.dirname(__file__), "..", "event_log"))
LIVE_LOG = os.path.join(_LOG_DIR, "live_captions.txt")
MAX_CHARS = 12000  # rolling window of stream text kept in memory
POLL_MS = 50
SEPARATOR = "   "

BG = "#3a3a3a"
COLOR_OLD = "#9a9a9a"
COLOR_RECENT = "#d6d6d6"
COLOR_LATEST = "#ffffff"
RECENT_SPAN = 4  # captions (including the latest) shown brighter than the fading past

FONT_CANDIDATES = ["EB Garamond", "Georgia", "DejaVu Serif", "Times New Roman", "Times"]
FONT_SIZE = 17
PAGE_CHARS = 52  # page column width, in average-character units
PAGE_LINES = 24
WINDOW_GEOMETRY = "980x940"


def tail_lines(path, last_pos):
    """Read any new lines appended since last_pos. Returns (new_lines, new_pos)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], last_pos
    if size <= last_pos:
        # File was reset/recreated
        if size < last_pos:
            last_pos = 0
        return [], last_pos
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(last_pos)
            new = f.read()
        return [l.strip() for l in new.splitlines() if l.strip()], size
    except OSError:
        return [], last_pos


class CaptionWindow:
    def __init__(self, root):
        self.root = root
        root.title("machine")
        root.configure(bg=BG)
        root.geometry(WINDOW_GEOMETRY)

        families = set(tkfont.families(root))
        family = next((f for f in FONT_CANDIDATES if f in families), "TkTextFont")
        self.base_font = tkfont.Font(family=family, size=FONT_SIZE)
        self.bold_font = tkfont.Font(family=family, size=FONT_SIZE, weight="bold")

        self.text = tk.Text(
            root,
            width=PAGE_CHARS,
            height=PAGE_LINES,
            wrap="word",
            bg=BG,
            fg=COLOR_OLD,
            bd=0,
            highlightthickness=0,
            font=self.base_font,
            spacing2=8,
            padx=40,
            pady=32,
            cursor="arrow",
            insertwidth=0,
        )
        self.text.tag_configure("old", foreground=COLOR_OLD)
        self.text.tag_configure("recent", foreground=COLOR_RECENT)
        self.text.tag_configure("latest", foreground=COLOR_LATEST, font=self.bold_font)
        self.text.pack(expand=True)
        self.text.configure(state="disabled")

        self.history = []
        # If file exists, start from end (don't replay old history)
        self.last_pos = os.path.getsize(LIVE_LOG) if os.path.exists(LIVE_LOG) else 0

        self._set_content([("waiting for captions...", "old")])
        root.after(POLL_MS, self.poll)

    def poll(self):
        new_lines, self.last_pos = tail_lines(LIVE_LOG, self.last_pos)
        if new_lines:
            self.history.extend(new_lines)
            self._trim()
            self._render()
        self.root.after(POLL_MS, self.poll)

    def _trim(self):
        total = sum(len(h) for h in self.history)
        while len(self.history) > 1 and total > MAX_CHARS:
            total -= len(self.history.pop(0))

    def _render(self):
        n = len(self.history)
        chunks = []
        for i, entry in enumerate(self.history):
            if i:
                chunks.append((SEPARATOR, "old"))
            if i == n - 1:
                tag = "latest"
            elif i >= n - RECENT_SPAN:
                tag = "recent"
            else:
                tag = "old"
            chunks.append((entry, tag))
        self._set_content(chunks)

    def _set_content(self, chunks):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for content, tag in chunks:
            self.text.insert("end", content, tag)
        self.text.configure(state="disabled")
        self.text.see("end")


if __name__ == "__main__":
    if tk is None:
        print("tkinter is not available in this environment.")
        sys.exit(1)
    if not os.path.isdir(_LOG_DIR):
        print("event_log/ directory not found. Run from the project root.")
        sys.exit(1)

    root = tk.Tk()
    CaptionWindow(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
