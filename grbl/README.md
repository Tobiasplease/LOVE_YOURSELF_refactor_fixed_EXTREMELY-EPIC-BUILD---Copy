# GRBL Control Scripts

This directory contains scripts for controlling GRBL CNC controllers with pen/servo plotters.

## svg_to_grbl.py

A complete SVG to GRBL pipeline that converts SVG files to G-code and executes them on GRBL controllers with servo pen control.

### What it does:

1. **SVG Conversion**: Uses vpype/inkscape (via bcnc converters) to convert SVG to G-code
2. **Servo Integration**: Converts Z-axis movements to servo commands (M3 S30/S50) for pen up/down
3. **GRBL Initialization**: Homes the machine, sets coordinate systems, and configures feed rates
4. **Execution**: Sends G-code line-by-line with proper wait states between commands

### Basic Usage:

```bash
# Convert SVG and execute on GRBL
python svg_to_grbl.py drawing.svg

# Generate G-code only (no execution)
python svg_to_grbl.py drawing.svg --no-execute

# Specify output file
# python svg_to_grbl.py drawing.svg -o output.gcode
```

### Advanced Usage:

```bash
# Custom work origin and feed rate
python svg_to_grbl.py drawing.svg --origin-x 50 --origin-y -5 --feed-rate 2000

# Apply offset to drawing
python svg_to_grbl.py drawing.svg -x 10 -y 15

# Scale to fit dims
python svg_to_grbl.py drawing.svg --scale-to 500x500mm

# Use absolute positioning (send G90)
python svg_to_grbl.py drawing.svg --use-absolute

# Use temporary directory for files
# python svg_to_grbl.py drawing.svg --temp-dir /tmp --no-execute
```

### Command Line Options:

- `svg_file` - Input SVG file (required)
- `-o, --output` - Output G-code file path
- `-x, --offset-x` - X offset for drawing (default: 0.0)
- `-y, --offset-y` - Y offset for drawing (default: 0.0)
- `--origin-x` - Work origin X coordinate (default: 66.0)
- `--origin-y` - Work origin Y coordinate (default: -2.0)
- `--feed-rate` - Feed rate for movements (default: 3000)
- `--no-execute` - Generate G-code only, don't execute on GRBL
- `--temp-dir` - Directory for temporary files
- `--scale-to` - scale to fit dims e.g 500x500mm
- `--use-absolute` - send G90 and use absolute positioning

### Requirements:

1. **Hardware**: GRBL controller connected via serial/USB
2. **Software Dependencies**:
   - vpype (recommended) or inkscape for SVG conversion
   - Python serial library
3. **Servo Setup**: Servo connected to spindle control (M3 commands)

### Servo Commands:

- **Pen Up**: `M3 S30`
- **Pen Down**: `M3 S50`

### Process Flow:

1. **SVG Input** → vpype/inkscape conversion → **Raw G-code**
2. **Raw G-code** → servo conversion → **GRBL-ready G-code**
3. **GRBL Connection** → homing & setup → **G-code execution**

### Example Output:

```
[STEP 1] Converting SVG to G-code...
[INFO] Försöker med vpype...
[INFO] vpype G-code generering lyckades
[STEP 2] Applying servo control conversion...
[INFO] Optimerad G-kod sparad: drawing_servo.gcode
[SUCCESS] G-code generated: drawing_servo.gcode
[STEP 3] Executing on GRBL...
[INFO] Testing /dev/ttyUSB0...
[INFO] /dev/ttyUSB0 responds as GRBL: <Idle|MPos:0.000,0.000,0.000|FS:0,0>
[INFO] Running homing cycle ($H)...
[INFO] Homing complete
[SUCCESS] Drawing complete!
```

## Other Scripts:

- `setup_grbl.py` - Basic GRBL initialization and work origin setup
- `setup_grbl_grid.py` - Draws a test grid pattern for calibration
- `grbl_utils.py` - Shared GRBL communication functions

## Troubleshooting:

**No GRBL port found:**

- Check USB/serial connection
- Verify GRBL firmware is running
- Try different baud rates

**Drawing offset issues:**

- Adjust `--origin-x` and `--origin-y` values
- Use `--offset-x` and `--offset-y` for drawing positioning
- Run `setup_grbl_grid.py` for calibration
