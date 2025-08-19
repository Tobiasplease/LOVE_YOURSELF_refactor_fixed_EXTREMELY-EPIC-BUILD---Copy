# Standalone Hand Control Interface

This is a completely isolated version of the hand control system extracted from the main workspace.

## Features

- Complete hand control interface with Tkinter GUI
- Direct cursor→servo control with physics
- 5 emotional states with different movement characteristics  
- Recording, playback, and Markov chain generation
- Dataset management and cycling
- Keyboard finger control (W/S, E/D, R/F, T/G for fingers 1-4)
- Visual feedback canvas

## Files

- `hand_control_interface.py` - Main interface (exact copy of baseline)
- `hand_expression.py` - Local hand controller (extracted from servo_control)
- `launcher.py` - Simple launcher script
- `requirements.txt` - Minimal dependencies
- `movement_recordings/` - Directory for recorded movements

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Connect Arduino hand controller to COM3 (or update port in code)

3. Run the interface:
   ```bash
   python launcher.py
   ```

## Usage

- **Connect**: Click "Connect to Hand Controller" button
- **Mouse Control**: Move mouse over the canvas to control servos
- **Keyboard Control**: Use W/S, E/D, R/F, T/G to control individual fingers
- **Recording**: Press spacebar or click "Record Movement" (20s segments)
- **Emotions**: Switch between 5 emotional states for different movement styles
- **Generation**: Use "Generate (Markov)" for AI-driven movements

## Hardware

- Arduino with 4 servos on pins 8, 9, 10, 11
- Serial communication at 9600 baud
- Command format: "HAND,servo1,servo2,servo3,servo4\n"
- Servo range: 40-130° (mapped from 0-180° interface range)

This is the exact baseline version that was working perfectly in the main workspace, now completely standalone.
