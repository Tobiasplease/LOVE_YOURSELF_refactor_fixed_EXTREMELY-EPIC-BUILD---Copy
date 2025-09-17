import re

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

    # Professor's latest calibrated values (2025-09-17) - new rotation correction
    Ax, Ay = 0, 40   # vänster längst från robot (top-left)
    Bx, By = 35, 2   # vänster närmast robot (bottom-left)
    Cx, Cy = 70, 3   # höger närmast robot (bottom-right)
    Dx, Dy = 25, 40  # höger längst från robot (top-right)

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
    SCALE_FACTOR = 2.5  # Increase output size (set to 1.0 to disable)

    x_match = re.search(r"X([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)
    y_match = re.search(r"Y([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)

    if x_match and y_match:
        # Extract original coordinates
        original_x = float(x_match.group(1))
        original_y = float(y_match.group(1))

        # Apply JBE's inverse warp transform directly
        transformed_x, transformed_y = map_to_quad(original_x, original_y, max_x, max_y)

        # TEMPORARY SCALING - COMMENTED OUT: Professor's calibrated coordinates should fix size issues
        # if SCALE_FACTOR != 1.0:
        #     # Calculate center of quadrilateral (updated for new coordinates)
        #     center_x = (5 + 40 + (-3) + (-28)) / 4  # average of corner x-coords: 3.5
        #     center_y = (4 + 3 + 40 + 40) / 4  # average of corner y-coords: 21.75
        #
        #     # Scale around center point
        #     transformed_x = center_x + (transformed_x - center_x) * SCALE_FACTOR
        #     transformed_y = center_y + (transformed_y - center_y) * SCALE_FACTOR

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

    for line in lines:
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("%"):
                continue

            try:
                if line.startswith(("G0", "G1", "G00", "G01")):
                    print(f"[ORG] {line}")
                    line = warp_transform_line(line)
                    print(f"[ADJ] {line}")
                    print("=====")

            except Exception as e:
                print(f"[ERROR] Failed to process line {line_num}: {line}. Error: {e}")
                continue