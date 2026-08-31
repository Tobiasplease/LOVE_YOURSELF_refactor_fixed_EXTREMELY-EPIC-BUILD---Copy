import re
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

Point = Tuple[float, float]


_MOVE_RE = re.compile(r"^(G0|G1|G00|G01)\b", re.IGNORECASE)
_X_RE = re.compile(r"X(-?\d+(?:\.\d+)?)")
_Y_RE = re.compile(r"Y(-?\d+(?:\.\d+)?)")


def _replace_coord(token_line: str, axis: str, value: float) -> str:
    pattern = _X_RE if axis.upper() == "X" else _Y_RE
    fmt = f"{axis}{value:.4f}"
    if pattern.search(token_line):
        return pattern.sub(fmt, token_line)
    # If axis not present, append at end (keep spacing)
    return token_line.strip() + f" {fmt}\n"


def transform_gcode_lines(lines: Sequence[str], map_xy: Callable[[float, float], Tuple[float, float]]) -> List[str]:
    """Transform XY of G0/G1 moves using map_xy(x, y) -> (x', y').
    Preserves other commands and comments.
    """
    out: List[str] = []
    last_x = 0.0
    last_y = 0.0
    for line in lines:
        orig = line
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("%"):
            out.append(orig)
            continue

        if not _MOVE_RE.match(s):
            out.append(orig)
            continue

        mx = _X_RE.search(s)
        my = _Y_RE.search(s)
        x = last_x if not mx else float(mx.group(1))
        y = last_y if not my else float(my.group(1))

        tx, ty = map_xy(x, y)
        new_line = orig
        if mx:
            new_line = _replace_coord(new_line, "X", tx)
        else:
            # no X present, append
            new_line = new_line.rstrip("\n") + f" X{tx:.4f}\n"
        if my:
            new_line = _replace_coord(new_line, "Y", ty)
        else:
            new_line = new_line.rstrip("\n") + f" Y{ty:.4f}\n"

        out.append(new_line)
        last_x, last_y = x, y
    return out


def transform_gcode_file(input_path: str | Path, output_path: str | Path, map_xy: Callable[[float, float], Tuple[float, float]]):
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    new_lines = transform_gcode_lines(lines, map_xy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


def parse_path_for_preview(lines: Sequence[str]) -> List[Point]:
    """Parse G0/G1 XY for preview polyline. Returns list of (x,y)."""
    pts: List[Point] = []
    last_x = 0.0
    last_y = 0.0
    for line in lines:
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("%"):
            continue
        if not _MOVE_RE.match(s):
            continue
        mx = _X_RE.search(s)
        my = _Y_RE.search(s)
        x = last_x if not mx else float(mx.group(1))
        y = last_y if not my else float(my.group(1))
        pts.append((x, y))
        last_x, last_y = x, y
    return pts
