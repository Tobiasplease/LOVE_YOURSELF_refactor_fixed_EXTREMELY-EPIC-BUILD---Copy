# LOVE_YOURSELF - AI-Powered Interactive Mirror System

A sophisticated AI-driven interactive system that combines computer vision, mood analysis, and servo control to create an empathetic digital companion. The system uses webcam input to detect faces, analyze emotions, generate captions using LLaVA vision models, and can optionally control servo motors for physical interaction.

## Features

- **Real-time Face Detection**: Uses OpenCV DNN face detection for robust person detection
- **Object Detection**: YOLOv8-powered object recognition and tracking
- **Mood Analysis**: AI-driven emotion and mood evaluation with LLaVA integration
- **Caption Generation**: Automatic scene description and context understanding
- **Servo Control**: Optional physical servo motor control for interactive responses
- **Memory System**: Maintains contextual awareness and interaction history
- **Breathing Simulation**: Simulates natural breathing patterns for life-like behavior

## System Requirements

- **Python**: 3.11+ (recommended)
- **Operating System**: macOS, Linux, or Windows
- **Hardware**:
  - Webcam/Camera (required)
  - Arduino with servo motors (optional, for physical interaction)
  - CUDA-compatible GPU (recommended for faster AI processing)

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment with Python 3.10 or 3.11
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### 4. Configuration

Edit `config/config.py` to customize your setup:

- **Camera Settings**: Set `CAMERA_INDEX` for your webcam
- **Servo Control**: Set `USE_SERVO = True` and configure `SERIAL_PORT` if using Arduino
- **Detection Thresholds**: Adjust `CONFIDENCE_THRESHOLD` for face detection sensitivity
- **Mood Settings**: Configure mood evaluation intervals and snapshot storage

### 5. External Dependencies

The system requires these external model files:

- **Face Detection Models**:
  - `deploy.prototxt` (update path in `machine.py`)
  - `res10_300x300_ssd_iter_140000.caffemodel` (update path in `machine.py`)
- **YOLO Models**: `yolov8m.pt` and `yolov8n.pt` (included)

### 6. LLaVA API Setup

For mood analysis and captioning, ensure LLaVA is running locally:

```bash
# Install and run LLaVA (example using Ollama)
ollama pull llava
ollama serve
```

The system expects LLaVA to be accessible at `http://localhost:11434/api/generate`. All Ollama API calls are now handled through the `ollama.py` module.

## Usage

### Basic Operation

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main application
python machine.py
```

### Testing Components

```bash
# Test LLaVA caption generation
python test_llava_caption.py

# Test individual modules
python -m captioner.captioner
python -m mood.mood
python -m perception.object_detection
```

## Project Structure

```
LOVE_YOURSELF/
├── machine.py              # Main application entry point
├── requirements.txt        # Python dependencies
├── config/                 # Configuration settings
├── captioner/             # AI captioning and memory system
├── mood/                  # Mood analysis and emotional processing
├── perception/            # Computer vision and object detection
├── vision/                # Gaze tracking and visual processing
├── breathing/             # Breathing simulation
├── servo_control/         # Arduino servo control
├── mood_snapshots/        # Stored mood analysis images
└── Lint-arduinoserial/    # Arduino code for servo control
```

## Key Dependencies

- **Computer Vision**: OpenCV, NumPy
- **Machine Learning**: Ultralytics (YOLOv8), PyTorch, TorchVision
- **Communication**: Requests (API calls), PySerial (Arduino)
- **Image Processing**: Pillow, Matplotlib

## Configuration Options

### Camera Settings

- `CAMERA_INDEX`: Webcam device index (default: 0)
- `CONFIDENCE_THRESHOLD`: Face detection sensitivity (0.0-1.0)

### Servo Control

- `USE_SERVO`: Enable/disable physical servo control
- `SERIAL_PORT`: Arduino serial port (e.g., 'COM3', '/dev/ttyUSB0')
- `BAUD_RATE`: Serial communication speed (default: 9600)

### Mood Analysis

- `MOOD_EVALUATION_INTERVAL`: How often to analyze mood
- `MOOD_SNAPSHOT_FOLDER`: Where to store analysis images

## Troubleshooting

### Common Issues

1. **Camera not found**: Check `CAMERA_INDEX` in config
2. **Servo not responding**: Verify `SERIAL_PORT` and Arduino connection
3. **LLaVA errors**: Ensure LLaVA server is running on localhost:11434
4. **Import errors**: Verify virtual environment is activated and dependencies installed

### Performance Optimization

- Use CUDA-compatible GPU for faster AI processing
- Adjust YOLO model size (`yolov8n.pt` for speed, `yolov8m.pt` for accuracy)
- Reduce camera resolution for better performance
- Adjust detection intervals in config

## Development

### Virtual Environment Management

```bash
# Activate environment
source .venv/bin/activate

# Deactivate when done
deactivate

# Update dependencies
pip install --upgrade -r requirements.txt

# Add new dependencies
pip install new-package
pip freeze > requirements.txt
```

### Code Structure

The system follows a modular architecture:

- `machine.py`: Main loop and coordination
- Individual modules handle specific functionality
- Configuration centralized in `config/config.py`
- Threaded processing for real-time performance

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and support, please [add contact information or issue tracker].
