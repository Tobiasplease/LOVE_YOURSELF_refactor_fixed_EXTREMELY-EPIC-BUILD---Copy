# Paper Detection Calibration

This directory stores calibration data for the paper detection safety system.

## Files

- `paper_reference.jpg` - Reference image of paper properly positioned for drawing
- `test_images/` - Test images captured during detection testing

## Setup Instructions

1. **Enable paper detection** in `config/config.py`:
   ```python
   ENABLE_PAPER_DETECTION = True
   ```

2. **Capture reference image**:
   ```bash
   python debug/test_paper_detection.py --capture
   ```

3. **Test detection**:
   ```bash
   python debug/test_paper_detection.py --test
   ```

## Configuration Options

In `config/config.py`:

- `ENABLE_PAPER_DETECTION` - Master toggle
- `PAPER_CHECK_METHOD` - "reference" or "direct"
- `PAPER_DETECTION_CONFIDENCE_THRESHOLD` - Minimum confidence (0.0-1.0)
- `PAPER_DETECTION_GAZE_PAN/TILT` - Servo angles for detection view

## How It Works

1. Before drawing execution, system looks down at drawing area
2. Captures image of current view
3. LLM analyzes image for paper presence
4. Drawing proceeds only if paper detected with sufficient confidence
5. If no paper detected, drawing is safely aborted

## Testing

Use the interactive testing tool:

```bash
python debug/test_paper_detection.py
```

Or specific tests:

```bash
# Show system status
python debug/test_paper_detection.py --status

# Run 10 detection tests
python debug/test_paper_detection.py --multiple 10
```