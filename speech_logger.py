#!/usr/bin/env python3
"""
Whisper-based speech recognition with timestamping and logging.
Uses OpenAI's Whisper tiny multilingual model for local speech processing.
"""

import whisper
import pyaudio
import wave
import threading
import time
import json
import os
from datetime import datetime
from typing import Optional


class WhisperLogger:
    def __init__(self, log_file: str = "speech_log.json", chunk_duration: float = 5.0):
        """
        Initialize Whisper logger.

        Args:
            log_file: Path to JSON log file
            chunk_duration: Duration of audio chunks in seconds
        """
        self.log_file = log_file
        self.chunk_duration = chunk_duration
        self.model = whisper.load_model("small")  # tiny multilingual model
        self.is_recording = False
        self.audio_thread: Optional[threading.Thread] = None

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Ensure log file exists
        self._init_log_file()

    def _init_log_file(self):
        """Initialize the log file with empty array if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def _log_speech(self, text: str, timestamp: str, confidence: Optional[float] = None):
        """Log speech text with timestamp to JSON file."""
        entry = {"timestamp": timestamp, "text": text.strip(), "confidence": confidence, "model": "whisper-tiny"}

        # Read existing logs
        try:
            with open(self.log_file, "r") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []

        # Append new entry
        logs.append(entry)

        # Write back to file
        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"[{timestamp}] {text}")

    def _record_audio_chunk(self) -> bytes:
        """Record a chunk of audio."""
        stream = self.audio.open(format=self.format, channels=self.channels, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)

        frames = []
        chunk_frames = int(self.sample_rate * self.chunk_duration / self.chunk_size)

        for _ in range(chunk_frames):
            if not self.is_recording:
                break
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        return b"".join(frames)

    def _save_temp_audio(self, audio_data: bytes) -> str:
        """Save audio data to temporary WAV file."""
        temp_file = "temp_audio.wav"

        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)

        return temp_file

    def _process_audio_chunk(self, audio_data: bytes):
        """Process audio chunk with Whisper and log results."""
        try:
            # Save temporary audio file
            temp_file = self._save_temp_audio(audio_data)

            # Transcribe with Whisper
            result = self.model.transcribe(temp_file, language=None)  # Auto-detect language

            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # Log if text was detected
            text = str(result.get("text", "")).strip()
            if text:
                timestamp = datetime.now().isoformat()
                self._log_speech(text, timestamp)

        except Exception as e:
            print(f"Error processing audio: {e}")

    def _audio_loop(self):
        """Main audio recording and processing loop."""
        while self.is_recording:
            try:
                # Record audio chunk
                audio_data = self._record_audio_chunk()

                if audio_data and self.is_recording:
                    # Process in separate thread to avoid blocking
                    processing_thread = threading.Thread(target=self._process_audio_chunk, args=(audio_data,))
                    processing_thread.daemon = True
                    processing_thread.start()

            except Exception as e:
                print(f"Error in audio loop: {e}")
                time.sleep(1)

    def start_recording(self):
        """Start continuous speech recognition and logging."""
        if self.is_recording:
            print("Already recording...")
            return

        print("Starting Whisper speech recognition...")
        print(f"Logging to: {self.log_file}")
        print(f"Chunk duration: {self.chunk_duration}s")
        print("Press Ctrl+C to stop")

        self.is_recording = True
        self.audio_thread = threading.Thread(target=self._audio_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop_recording(self):
        """Stop speech recognition."""
        if not self.is_recording:
            return

        print("Stopping speech recognition...")
        self.is_recording = False

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_recording()
        if hasattr(self, "audio"):
            self.audio.terminate()


def main():
    """Main function to run Whisper logger."""
    import argparse

    parser = argparse.ArgumentParser(description="Whisper speech recognition with logging")
    parser.add_argument("--log-file", default="speech_log.json", help="Log file path")
    parser.add_argument("--chunk-duration", type=float, default=5.0, help="Audio chunk duration in seconds")

    args = parser.parse_args()

    logger = WhisperLogger(log_file=args.log_file, chunk_duration=args.chunk_duration)

    try:
        logger.start_recording()

        # Keep running until interrupted
        while logger.is_recording:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        logger.stop_recording()
        print("Speech recognition stopped")


if __name__ == "__main__":
    main()
