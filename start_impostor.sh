#!/bin/bash

# Start impostor system with tmux and auto-restart capability
SESSION_NAME="impostor-system"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Window 0: ComfyUI
tmux rename-window -t $SESSION_NAME:0 'ComfyUI'
tmux send-keys -t $SESSION_NAME:0 'cd /home/impostor/ComfyUI' C-m
tmux send-keys -t $SESSION_NAME:0 'while true; do echo "Starting ComfyUI..."; source .venv/bin/activate && python3.12 main.py; echo "ComfyUI crashed, restarting in 5 seconds..."; sleep 5; done' C-m

# Window 1: Main Machine
tmux new-window -t $SESSION_NAME -n 'Machine'
tmux send-keys -t $SESSION_NAME:1 'cd /home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy' C-m
tmux send-keys -t $SESSION_NAME:1 'while true; do echo "Starting Machine..."; source .venv/bin/activate && python3.12 machine.py; echo "Machine crashed, restarting in 5 seconds..."; sleep 5; done' C-m

# Window 2: Log Viewer
tmux new-window -t $SESSION_NAME -n 'LogViewer'
tmux send-keys -t $SESSION_NAME:2 'cd /home/impostor/impostor-log-viewer' C-m
tmux send-keys -t $SESSION_NAME:2 'while true; do echo "Starting Log Viewer..."; npm run start; echo "Log Viewer crashed, restarting in 5 seconds..."; sleep 5; done' C-m

# Attach to session
echo "Starting impostor system in tmux session: $SESSION_NAME"
echo "Use 'tmux attach -t $SESSION_NAME' to view the session"
echo "Use Ctrl-B + number (0,1,2) to switch between windows"
echo "Use 'tmux kill-session -t $SESSION_NAME' to stop all processes"

tmux attach -t $SESSION_NAME