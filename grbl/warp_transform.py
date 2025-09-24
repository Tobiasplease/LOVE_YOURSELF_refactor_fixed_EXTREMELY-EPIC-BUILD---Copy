import re

# ================================================================
# Optional PRE‑TRANSFORM (ideal domain) settings
# ------------------------------------------------
# These apply BEFORE the warp mapping so the correction is preserved.
# Defaults are identity/no‑op so baseline behavior is unchanged.
#
# Units: same units as incoming G-code XY (typically mm) but in the
#        IDEAL domain (before mapping). For human‑oriented "up/down/left/right"
#        adjustments tied to the actual robot frame, prefer the POST nudges
#        (NUDGE_*_MM) below, which operate in machine coordinates.
# Pivot: if None, uses the domain center (max_x/2, max_y/2).
# Clamp: if True, clamps pre‑transformed XY to [0..max_x]/[0..max_y].
#
# To revert to baseline: set SCALE=1.0 and OFFSETS=0.0 (or leave defaults).
# ================================================================
# Original (baseline) settings:
# PRE_SCALE_X = 1.0
# PRE_SCALE_Y = 1.0
# PRE_OFFSET_X = 0.0
# PRE_OFFSET_Y = 0.0

# Modified settings: 20% scale up, shift 10mm closer to robot body
# Previous modified: PRE_OFFSET_Y = -5.0
PRE_SCALE_X = 1.2
PRE_SCALE_Y = 1.2
PRE_OFFSET_X = 0.0
PRE_OFFSET_Y = -10.0
PRE_PIVOT_X = None  # e.g., 0.0 to use origin, or None for center
PRE_PIVOT_Y = None
PRE_CLAMP_TO_DOMAIN = False

def find_max_xy_from_lines(lines):
    max_x = float('-inf')
    max_y = float('-inf')

    for line in lines:
        parts = line.strip().split()
        for part in parts:
            if part.startswith('X'):
                try:
                    x_val = float(part[1:])
                    if x_val > max_x:
                        max_x = x_val
                except ValueError:
                    pass
            elif part.startswith('Y'):
                try:
                    y_val = float(part[1:])
                    if y_val > max_y:
                        max_y = y_val
                except ValueError:
                    pass

    return (
        None if max_x == float('-inf') else max_x,
        None if max_y == float('-inf') else max_y,
    )

# DET HÄR är funktionen som ska konvertera tillbaka skjuvningen
# Den mappar (x,y) from 40x40-rutan in i en skev rektangel som blir en kvadratish när man skriver ut den
def map_to_quad(x, y, x_max=40, y_max=40):
    """
    Map (x,y) from square [0, x_max]x[0, y_max] into the quadrilateral.
    """

    # Normalize to [0,1]
    u = x / x_max
    v = y / y_max

    # Vertically flip Y coordinate to prevent upside-down output
    v = 1.0 - v

    # Professor's latest calibrated values (2025-09-17) - corrected for leftward skew
    Ax, Ay = 1, 40   # vänster längst från robot (top-left) - shifted right 1mm
    Bx, By = 35, 2   # vänster närmast robot (bottom-left) - unchanged
    Cx, Cy = 70, 3   # höger närmast robot (bottom-right) - unchanged
    Dx, Dy = 26, 40  # höger längst från robot (top-right) - shifted right 1mm

    # Previous coordinate versions (for reference):
    # Original values:
    # Ax, Ay = 8, 18   # vänster närmast robot
    # Bx, By = 40, 0   # höger närmast robot
    # Cx, Cy = 33, 20  # höger längst från robot
    # Dx, Dy = 0, 40   # vänster längst från robot

    # Professor's original calibrated values (before translation):
    # Ax, Ay = 5, 4   # vänster närmast robot
    # Bx, By = 40, 3   # höger närmast robot
    # Cx, Cy = -3, 40  # höger längst från robot
    # Dx, Dy = -28, 40   # vänster längst från robot

    # Bilinear interpolation
    X = (1 - u) * (1 - v) * Ax + u * (1 - v) * Dx + (1 - u)* v * Bx + u * v * Cx
    Y = (1 - u) * (1 - v) * Ay + u * (1 - v) * Dy + (1 - u) * v * By + u * v * Cy

    return X, Y



def warp_transform_line(gcode_line, max_x, max_y):
    """Apply inverse warp transform to G-code coordinates"""
    # TEMPORARY SCALING FIX - easily reversible by setting to 1.0
    SCALE_FACTOR = 1.0  # Increase output size (set to 1.0 to disable)

    # POSITION OFFSET - easily toggleable positioning adjustment (legacy)
    # NOTE: Applied in machine/GRBL coordinates AFTER mapping and scaling.
    # Keep as-is for baseline compatibility.
    OFFSET_X = 0.0  # X offset in mm (historical note: positive intended as 'away from robot')
    OFFSET_Y = 0.0  # Y offset in mm (historical note: positive intended as 'toward robot body')

    # ------------------------------------------------------------
    # Optional POST‑TRANSFORM direction nudges in machine coords
    # Consistent with quadrilateral labels in map_to_quad:
    #   - 'vänster'/'höger' correspond to X‑ decreases/increases
    #   - 'längst från robot' (top) has larger Y; 'närmast robot' (bottom) has smaller Y
    # Therefore:
    #   RIGHT  => +X, LEFT  => -X
    #   UP     => +Y (farther from robot), DOWN => -Y (toward robot)
    # Defaults 0.0 keep baseline behavior unchanged.
    # ------------------------------------------------------------
    NUDGE_RIGHT_MM = 0.0
    NUDGE_LEFT_MM = 0.0
    NUDGE_UP_MM = 0.0
    NUDGE_DOWN_MM = 0.0

    x_match = re.search(r"X([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)
    y_match = re.search(r"Y([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)

    if x_match and y_match:
        # Extract original coordinates
        original_x = float(x_match.group(1))
        original_y = float(y_match.group(1))

        # ------------------------------------------------------------
        # PRE‑TRANSFORM (ideal domain): scale/offset BEFORE warp map
        # Baseline reference: originally we called
        #   map_to_quad(original_x, original_y, max_x, max_y)
        # Applying PRE_* with identity (scale=1, offsets=0) reproduces
        # the baseline exactly.
        # ------------------------------------------------------------
        cx = (max_x / 2.0) if PRE_PIVOT_X is None else float(PRE_PIVOT_X)
        cy = (max_y / 2.0) if PRE_PIVOT_Y is None else float(PRE_PIVOT_Y)

        x_pre = cx + (original_x - cx) * float(PRE_SCALE_X) + float(PRE_OFFSET_X)
        y_pre = cy + (original_y - cy) * float(PRE_SCALE_Y) + float(PRE_OFFSET_Y)

        if PRE_CLAMP_TO_DOMAIN:
            if max_x is not None:
                x_pre = min(max(x_pre, 0.0), float(max_x))
            if max_y is not None:
                y_pre = min(max(y_pre, 0.0), float(max_y))

        # Apply JBE's inverse warp transform using pre‑adjusted coords
        transformed_x, transformed_y = map_to_quad(x_pre, y_pre, max_x, max_y)

        # Apply scaling around center point for larger drawings
        if SCALE_FACTOR != 1.0:
            # Calculate center of quadrilateral (updated for current coordinates)
            center_x = (0 + 25 + 70 + 35) / 4  # average of corner x-coords: 32.5
            center_y = (40 + 40 + 3 + 2) / 4  # average of corner y-coords: 21.25

            # Scale around center point
            transformed_x = center_x + (transformed_x - center_x) * SCALE_FACTOR
            transformed_y = center_y + (transformed_y - center_y) * SCALE_FACTOR

        # Optional post‑warp directional nudges (machine space)
        transformed_x += float(NUDGE_RIGHT_MM) - float(NUDGE_LEFT_MM)
        transformed_y += float(NUDGE_UP_MM) - float(NUDGE_DOWN_MM)

        # Apply legacy position offset (after warp transform and scaling)
        transformed_x += OFFSET_X
        transformed_y += OFFSET_Y

        # Update G-code line with transformed coordinates
        gcode_line = re.sub(r"X[-+]?\d*\.?\d+", f"X{transformed_x:.4f}", gcode_line, flags=re.IGNORECASE)
        gcode_line = re.sub(r"Y[-+]?\d*\.?\d+", f"Y{transformed_y:.4f}", gcode_line, flags=re.IGNORECASE)

    return gcode_line


if __name__ == "__main__":
    gcode_file = "/Users/jbe/repos/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy/grbl/test_files/impostor-20250815_201047_00001__center_lined_servo_adjusted.gcode"
    try:
        with open(gcode_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"G-code file not found: {gcode_file}")
    lines = lines[3:]  # vpype junk

    # Find max X and Y values from all lines
    max_x, max_y = find_max_xy_from_lines(lines)
    if max_x is None:
        max_x = 40  # default
    if max_y is None:
        max_y = 40  # default

    for line in lines:
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("%"):
                continue

            try:
                if line.startswith(("G0", "G1", "G00", "G01")):
                    print(f"[ORG] {line}")
                    line = warp_transform_line(line, max_x, max_y)
                    print(f"[ADJ] {line}")
                    print("=====")

            except Exception as e:
                print(f"[ERROR] Failed to process line {line_num}: {line}. Error: {e}")
                continue
