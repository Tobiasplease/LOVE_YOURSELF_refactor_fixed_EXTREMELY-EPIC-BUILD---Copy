# LOVE_YOURSELF - AI-Powered Interactive Mirror System

A sophisticated AI-driven interactive system that combines computer vision, mood analysis, and servo control to create an empathetic digital companion. The system uses webcam input to detect faces, analyze emotions, generate captions using ollama vision models, and can optionally control servo motors for physical interaction. It can also generate images based on mood by posting to an external comfyui server.

## Features

- **Real-time Face Detection**: Uses OpenCV DNN face detection for robust person detection
- **Object Detection**: YOLOv8-powered object recognition and tracking
- **Mood Analysis**: AI-driven emotion and mood evaluation via ollama hosted model
- **Caption Generation**: Automatic scene description and context understanding
- **Image Generation**: Creates art based on mood analysis using ComfyUI integration
- **Servo Control**: Optional physical servo motor control for interactive responses
- **Memory System**: Maintains contextual awareness and interaction history
- **Breathing Simulation**: Simulates natural breathing patterns for life-like behavior
- **Event Logging**: Comprehensive JSON-based logging of all system events

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
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000.caffemodel`
- **YOLO Models**: `yolov8m.pt` and `yolov8n.pt` (included)

### 6. External Service Setup

#### Ollama API Setup
For mood analysis and captioning, ensure ollama is running locally:

```bash
# Install and run LLaVA (using Ollama)
ollama pull llava:7b-v1.6-mistral-q5_1
ollama serve
```

The system expects an LLM model to be accessible at `http://localhost:11434/api/generate`. All Ollama API calls are handled through the `utils/ollama.py` module.

#### ComfyUI Setup (Optional)
For AI image generation based on mood:

```bash
# Install and run ComfyUI
# Follow ComfyUI installation instructions
# Default URL: http://localhost:8188/prompt
```

ComfyUI integration is handled through the `drawing/` module and uses workflow templates.

## Usage

### Basic Operation

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main application
python machine.py
```

### To Control Log Output Folder

export MOOD_SNAPSHOT_FOLDER=/Users/jbe/Dropbox/\_outputs/impostor_event_log && python machine.py

### Testing Components

```bash
# Test ollama caption generation
python debug/test_ollama_caption.py

# Test ComfyUI integration
python debug/test_comfy.py drawing/example_workflow.json

# Test impostor flow
python debug/test_impostor_flow.py
```

## Project Structure

```
LOVE_YOURSELF/
├── machine.py              # Main application entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Code formatting configuration
├── config/                # Configuration settings
├── captioner/             # AI captioning and memory system
├── mood/                  # Mood analysis and emotional processing
├── perception/            # Computer vision and object detection
├── vision/                # Gaze tracking and visual processing
├── breathing/             # Breathing simulation
├── drawing/               # ComfyUI integration for image generation
├── servo_control/         # Arduino servo control
├── event_logging/         # JSON event logging system
├── utils/                 # Utility modules (ollama, continuity)
├── debug/                 # Test scripts for components
├── models/                # AI model files (face detection, YOLO)
├── mood_snapshots/        # Stored mood analysis images and logs
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
3. **ollama errors**: Ensure ollama server is running on localhost:11434
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
- Event-driven JSON logging for all system activities

### Code Formatting

The project uses standardized formatting:

```bash
# Format code (150 character line length)
black . --line-length 150
isort . --profile black --line-length 150

# Lint code
pylint . --max-line-length=150
flake8 . --max-line-length=150
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and support, please [add contact information or issue tracker].
