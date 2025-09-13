# GRBL Warp Transform Configuration System

## Overview

This system implements JBE's Swedish mathematician's inverse kinematics algorithm to correct robot arm distortion during drawing operations. The robot arm's mechanical constraints cause coordinates to be distorted - what should be straight lines become curves due to the arm's geometry and variable step lengths at different positions.

## The Problem

### Mechanical Reality vs. G-code Expectations
The core issue is that GRBL G-code assumes linear coordinate movement, but robot arms move in arcs and have variable step sizes depending on position. As our mechanical engineer Andreas explained:

> **Andreas (Mechanical Engineer):**
> "Börja med att kolla om algoritmen tar hänsyn till/om det spelar roll att den inte stegar i mm. Varje heltal är ju ca 5-7 mm. [...] Orsaken till att den inte är exakt i mm är ju på grund av att stegen har olika längd på olika ställen"

**Translation:** "Start by checking if the algorithm takes into account/if it matters that it doesn't step in mm. Each integer is about 5-7 mm. [...] The reason it's not exact in mm is because the steps have different lengths in different places"

This reveals the fundamental calibration issue - the algorithm assumes 1 unit = 1mm, but the actual robot uses ~5-7mm per step with variable step lengths.

### Working Coordinate System
Andreas provided the proven A4 paper coordinates that work optimally:

> **Andreas:**
> "Förut så var det ju de här värdena som var optimalt för A4:
> - Vänster ned x66 y-2
> - Höger ned x111 y-1
> - Vänster upp x-2 y67
> - Höger upp x24 y67"

**Translation:** "Previously these were the optimal values for A4:
- Bottom-left: x66, y-2
- Bottom-right: x111, y-1
- Top-left: x-2, y67
- Top-right: x24, y67"

> "Och optimalt är ju typ att ett A4 ligger rakt ut i mitten av en raklång arm (med en lätt vinkel på pappret som jag tror kommer från att pennan inte sitter helt i centrum av handen, men jag gissar här varför)"

**Translation:** "And optimal is basically that an A4 lies straight out in the middle of a straight extended arm (with a slight angle on the paper which I think comes from the pen not sitting exactly in the center of the hand, but I'm guessing here why)"

## The Mathematical Solution

### JBE's Inverse Kinematics Algorithm
The Swedish mathematician Mikael provided crucial context about the algorithm's assumptions and limitations:

> **Mikael Laaksoharju (Mathematician):**
> "En annan liten detalj som kanske borde kommenterats för att visa att vi gjort en del grova antaganden: det finns en faktor 0.1 på rad 66 när vi testar trans(). Den representerar drevningen av axelmotorn men siffran är helt tagen ur luften. Från g-code kommer en koordinat i x-y-planet som översätts till en vinkel av maskinen genom drevningen. G-code tror att den flyttar x steg i x-led men den roterar istället vid axeln så att den ritar en cirkelbåge. För att hålla vinkeln inom 180 grader i simuleringen slängde vi helt enkelt in en faktor."

**Translation:** "Another small detail that perhaps should have been commented to show that we made some rough assumptions: there is a factor 0.1 on line 66 when we test trans(). It represents the gearing of the axis motor but the number is completely made up. From g-code comes a coordinate in the x-y plane that is translated to an angle by the machine through the gearing. G-code thinks it moves x steps in x-direction but instead rotates at the axis so it draws a circular arc. To keep the angle within 180 degrees in the simulation we simply threw in a factor."

> "Inversen är förhoppningsvis oberoende av ovanstående men det kan hjälpa vid debuggning att veta vad vi bara killgissat."

**Translation:** "The inverse is hopefully independent of the above but it can help when debugging to know what we just guessed at."

### GRBL Configuration Dependencies
Mikael also highlighted the critical GRBL firmware settings:

> "Förutom att korrigera de mått som finns på maskindelarna i filen vill ni nog se över konstanterna 100 och 101 i GRBL konfig för att justera förhållandet mellan x och y."

**Translation:** "In addition to correcting the measurements of the machine parts in the file, you probably want to review constants 100 and 101 in GRBL config to adjust the ratio between x and y."

These refer to GRBL parameters `$100` (X-axis steps per mm) and `$101` (Y-axis steps per mm), which control the fundamental coordinate scaling.

## Current System Architecture

### Two-Layer Correction System

1. **Python Layer** (`warp_transform.py`):
   - Applies mathematical coordinate transformation before sending to GRBL
   - Uses JBE's inverse kinematics algorithm
   - Configurable via `warp_config.json`

2. **GRBL Firmware Layer**:
   - Converts coordinates to motor steps using `$100`/`$101` parameters
   - Must be configured via serial commands (not Python code)

### Configuration Flow
```
Original coordinates → Python warp transform → GRBL firmware → Motor steps
                      (warp_config.json)        ($100/$101)
```

## Files in This System

### Core Files
- **`warp_transform.py`**: The main transformation algorithm (JBE's inverse kinematics)
- **`warp_config.json`**: Configuration parameters for robot arm dimensions and coordinate system
- **`warp_calibration_tool.py`**: Interactive calibration and testing tool

### Integration
- **`grbl_utils.py`**: Conditionally applies warp transform based on `GRBL_WARP_TRANSFORM` config setting
- **`config/config.py`**: Contains `GRBL_WARP_TRANSFORM = False` toggle (currently disabled)

## Current Calibration Issues

### Scale Mismatch
The original algorithm was designed for theoretical 100mm arm segments, but the actual robot arm is ~295-320mm. This causes drawings to appear roughly 3x too small.

### Step Size Mismatch
The algorithm assumes 1 unit = 1mm, but the actual system uses ~5-7mm per step. This compounds the scale problem by another 5-7x factor.

### Position Offset
Drawings appear "very far up" from expected origin coordinates due to calibration mismatch between theoretical and actual robot geometry.

## Calibration Process

### Step 1: GRBL Firmware Settings
Check and adjust via serial terminal:
```
$100  # X-axis steps per mm
$101  # Y-axis steps per mm
```

### Step 2: Robot Arm Measurements
Update `warp_config.json` with actual measurements:
- `biceps`: Upper arm length (~300mm)
- `underarm`: Lower arm length (~300mm)
- `tendon_biceps`: Tendon length from biceps to elbow
- `tendon_underarm`: Tendon length from elbow to attachment

### Step 3: Coordinate System Scaling
Adjust in `warp_config.json`:
- `scale_factor`: Compensate for step size difference (try 5-7x)
- `origin_offset_x`, `origin_offset_y`: Correct positioning

### Step 4: Test with Known Coordinates
Use Andreas's working A4 coordinates as reference points to validate calibration.

### Step 5: Interactive Tuning
```bash
cd grbl
python warp_calibration_tool.py
```

## Usage

### Enable/Disable
Set in `config/config.py`:
```python
GRBL_WARP_TRANSFORM = True   # Enable correction
GRBL_WARP_TRANSFORM = False  # Use raw coordinates
```

### Debug Mode
Enable in `warp_config.json`:
```json
"debug_settings": {
  "enable_debug_output": true,
  "log_transformations": true
}
```

## Technical Notes

### Assumptions and Limitations
- The 0.1 gear ratio factor is estimated and may need adjustment
- Algorithm assumes specific robot arm geometry
- Coordinate system assumes consistent step sizes (but reality has variable steps)

### Future Improvements
- Calibrate gear ratio factor based on actual motor specifications
- Account for variable step lengths at different arm positions
- Develop automated calibration using known good coordinates
- Create visual feedback system for calibration validation

## References

- JBE's original inverse kinematics implementation
- Andreas's mechanical engineering insights on step sizes and optimal positioning
- Mikael Laaksoharju's mathematical analysis of algorithm assumptions
- Working A4 coordinate system as calibration reference

---

**Note**: This system is currently disabled (`GRBL_WARP_TRANSFORM = False`) pending proper calibration with actual robot arm measurements and step size corrections.