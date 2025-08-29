# LOVE_YOURSELF - AI-Powered Interactive Mirror System

A sophisticated AI-driven interactive system that combines computer vision, mood analysis, and servo control to create an empathetic digital companion. The system uses webcam input to detect faces, analyze emotions, generate captions using ollama vision models, and can optionally control servo motors for physical interaction. It can also generate images based on mood by posting to an external comfyui server.

## Features

- **Real-time Face Detection**: Uses OpenCV DNN face detection for robust person detection
- **Object Detection**: YOLOv8-powered object recognition and tracking
- **Mood Analysis**: AI-driven emotion and mood evaluation via ollama hosted model
- **Caption Generation**: Automatic scene description and context understanding
- **Image Generation**: Creates art based on mood analysis using ComfyUI integration
- **Physical Control Systems**: Multi-modal physical interaction capabilities:
  - **Servo Control**: Arduino-based servo motor control for interactive responses
  - **Hand Control**: Emotional hand movement mapping and Arduino integration
  - **GRBL Integration**: CNC machine control for precise drawing/engraving
  - **bCNC Integration**: G-code processing and CNC workflow management
- **Advanced Processing**:
  - **Memory System**: Maintains contextual awareness and interaction history
  - **Breathing Simulation**: Simulates natural breathing patterns for life-like behavior
  - **Speech Recognition**: Whisper-based local speech processing (optional)
  - **Pattern Recognition**: Advanced motif and thematic analysis
  - **Spatial Memory**: Object tracking and spatial awareness
- **Event Logging**: Comprehensive JSON-based logging of all system events with run management

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

### 6. Model Dependencies

The system requires several model files to be present:

```bash
# Face detection models (place in models/ directory)
# These should be downloaded from OpenCV's repository
models/deploy.prototxt
models/res10_300x300_ssd_iter_140000.caffemodel

# YOLO models (included in repository)
models/yolov8m.pt
models/yolov8n.pt
```

### 7. External Service Setup

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

ComfyUI integration is handled through the `drawing/` module and uses workflow templates. Multiple template configurations are available for different deployment scenarios:

- `impostor-template-gpupeon.json`: GPU-optimized configuration
- `impostor-template-impostor-bot.json`: Standard bot configuration
- `impostor-template-impostor-bot-svg.json`: SVG output configuration

## Usage

### Basic Operation

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main application
python machine.py

# Run with configuration override
python machine.py --config_override config/debug_config.json
python machine.py --config_override config/production_config.json

# Run with debug mode for verbose output
python machine.py --debug

# Combine debug mode with config override
python machine.py --config_override config/debug_config.json --debug
```

### To Control Log Output Folder

export MOOD_SNAPSHOT_FOLDER=/Users/jbe/Dropbox/\_outputs/impostor_event_log && python machine.py

### Testing Components

```bash
# Test ollama caption generation
python debug/test_ollama_caption.py

# Test ComfyUI integration
python debug/test_comfy.py drawing/example_workflow.json

# Test complete impostor image generation pipeline
python debug/test_impostor_flow.py

# Test G-code processing pipeline
python debug/test_pipeline.py

# View event logs
python debug/log_viewer.py

# Reset memory system
python debug/force_memory_reset.py
```

## Project Structure

```
LOVE_YOURSELF/
├── machine.py              # Main application entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Code formatting configuration
├── config/                # Configuration settings and overrides
│   ├── config.py          # Main configuration file
│   ├── loader.py          # Configuration override system
│   ├── debug_config.json  # Debug mode settings
│   ├── production_config.json # Production settings
│   └── platform-specific/ # Platform-specific configurations
├── captioner/             # AI captioning and memory system
├── mood/                  # Mood analysis and emotional processing
├── perception/            # Computer vision and object detection
├── vision/                # Gaze tracking and visual processing
├── breathing/             # Breathing simulation for life-like behavior
├── drawing/               # ComfyUI integration for image generation
├── servo_control/         # Arduino servo motor control
├── hand_control/          # Emotional hand movement control
├── grbl/                  # GRBL CNC machine integration
├── bcnc/                  # bCNC G-code processing utilities
├── event_logging/         # JSON event logging and run management
├── utils/                 # Utility modules (ollama, continuity, pattern recognition)
├── labs/                  # Experimental features
│   └── warp-fix-lab/      # Drawing warp correction
├── debug/                 # Test scripts and debugging tools
├── models/                # AI model files (face detection, YOLO)
├── event_log/             # Stored event logs and captured images
├── movement_recordings/   # Recorded movement patterns
└── arduino_src/           # Arduino code for physical control
```

## Key Dependencies

- **Computer Vision**: OpenCV, NumPy, scikit-image
- **Machine Learning**: Ultralytics (YOLOv8), PyTorch, TorchVision, spaCy
- **Communication**: Requests (API calls), PySerial (Arduino)
- **Image Processing**: Pillow, Matplotlib
- **CNC/G-code Processing**: vpype, vpype-gcode
- **GUI Automation**: PyAutoGUI, PyGetWindow (for certain features)

## Configuration Options

### Configuration Override System

The system supports runtime configuration overrides via JSON files:

```bash
# Use debug configuration (faster intervals for development)
python machine.py --config_override config/debug_config.json

# Use production configuration (slower, optimized intervals)
python machine.py --config_override config/production_config.json
```

#### Available Override Configurations

**Debug Config** (`config/debug_config.json`):

- Faster processing intervals for development and testing
- `REASON_INTERVAL`: 30 seconds (vs 360 default)
- `DRAWING_INTERVAL`: 60 seconds (vs 600 default)

**Production Config** (`config/production_config.json`):

- Standard production intervals for optimal performance
- Uses default values from `config/config.py`

**Platform-Specific Configs:**

- `gpu-peon/`: Configurations for GPU-enabled systems
- `impostor-bot-win/`: Windows-specific settings
- `jbe-osx/`: macOS-specific configurations

#### Creating Custom Override Files

Any configuration value in `config/config.py` can be overridden by creating a JSON file:

```json
{
  "CAMERA_INDEX": 1,
  "CONFIDENCE_THRESHOLD": 0.7,
  "USE_SERVO": false,
  "MOOD_EVALUATION_INTERVAL": 15,
  "DRAWING_INTERVAL": 300
}
```

The system automatically handles type conversion and validates that override keys exist in the base configuration.

### Base Configuration Settings

### Camera Settings

- `CAMERA_INDEX`: Webcam device index (default: 0)
- `CONFIDENCE_THRESHOLD`: Face detection sensitivity (0.0-1.0)

### Physical Control Systems

**Servo Control:**

- `USE_SERVO`: Enable/disable physical servo control
- `SERIAL_PORT`: Arduino serial port (e.g., 'COM3', '/dev/ttyUSB0')
- `BAUD_RATE`: Serial communication speed (default: 9600)

**Hand Control:**

- Emotional hand movement mapping
- Arduino integration for expressive gestures
- Movement pattern recording and playback

**CNC Integration:**

- GRBL-based precise drawing and engraving
- bCNC G-code processing and workflow management
- SVG to G-code conversion with centerline processing

### Mood Analysis

- `MOOD_EVALUATION_INTERVAL`: How often to analyze mood (seconds)
- `REASON_INTERVAL`: How often to generate reflective thoughts (seconds)
- `DRAWING_INTERVAL`: How often to trigger image generation (seconds)
- `MOOD_SNAPSHOT_FOLDER`: Where to store analysis images

## Troubleshooting

### Common Issues

1. **Camera not found**: Check `CAMERA_INDEX` in config
2. **Physical control not responding**:
   - Verify `SERIAL_PORT` and Arduino connection
   - Check GRBL connection for CNC functionality
   - Ensure proper Arduino firmware is uploaded
3. **ollama errors**: Ensure ollama server is running on localhost:11434
4. **Import errors**: Verify virtual environment is activated and dependencies installed
5. **Model files missing**: Ensure OpenCV face detection models are in `models/` directory
6. **G-code processing errors**: Verify vpype and related tools are properly installed

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

## Available Loras

- flux/own/impostor/impostor-32-balanced-8-5k.safetensors
- flux/own/impostor/1l1n3_F1D_SGX.safetensors
- flux/own/impostor/impostor-32-balanced-15k.safetensors
- flux/own/impostor/impostor-32-balanced-16k.safetensors
- flux/own/impostor/impostor-32-balanced-simple.safetensors
- flux/own/impostor/impostor-32-content-v2-10k.safetensors
- flux/own/impostor/impostor-32-content-v2-16k.safetensors
- flux/own/impostor/impostor-32-style-7k.safetensors
- flux/own/impostor/impostor-32-style-16k.safetensors
- flux/own/impostor/impostor-64-balanced-v2-16k-no-trig.safetensors
- flux/own/impostor/Line_Art_FLUX_V1.safetensors
