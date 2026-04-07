"""
caption_monitor.py
------------------
Full-screen live caption display. Run in a separate terminal while machine.py runs.

Usage:
    python debug/caption_monitor.py
"""

import os
import sys
import time

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

LIVE_LOG = os.path.join(os.path.dirname(__file__), "..", "event_log", "live_captions.txt")
MAX_LINES = 16


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


def render(history, width):
    lines = []
    for i, line in enumerate(history):
        dim = i < len(history) - 1
        if dim:
            lines.append(f"[dim]{line}[/dim]")
        else:
            lines.append(f"[bold white]{line}[/bold white]")
    return "\n\n".join(lines)


def run_rich():
    console = Console()
    history = []
    last_pos = 0

    # If file exists, start from end (don't replay old history)
    if os.path.exists(LIVE_LOG):
        last_pos = os.path.getsize(LIVE_LOG)

    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            new_lines, last_pos = tail_lines(LIVE_LOG, last_pos)
            if new_lines:
                history.extend(new_lines)
                if len(history) > MAX_LINES:
                    history = history[-MAX_LINES:]

            if not history:
                content = Align.center("[dim]waiting for captions...[/dim]", vertical="middle")
            else:
                content = Panel(
                    render(history, console.width),
                    title="[bold cyan]machine[/bold cyan]",
                    border_style="dim",
                    padding=(1, 3),
                )
            live.update(content)
            time.sleep(0.25)


def run_plain():
    last_pos = 0
    if os.path.exists(LIVE_LOG):
        last_pos = os.path.getsize(LIVE_LOG)
    print("Caption monitor — Ctrl+C to exit\n")
    while True:
        new_lines, last_pos = tail_lines(LIVE_LOG, last_pos)
        for line in new_lines:
            print(line + "\n")
        time.sleep(0.3)


if __name__ == "__main__":
    if not os.path.isdir(os.path.dirname(LIVE_LOG)):
        print("event_log/ directory not found. Run from the project root.")
        sys.exit(1)

    try:
        if HAS_RICH:
            run_rich()
        else:
            run_plain()
    except KeyboardInterrupt:
        pass
