#!/bin/bash

# Start impostor system with tmux grid layout and auto-restart capability
SESSION_NAME="impostor-system"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Create grid layout: split horizontally, then split right pane vertically
tmux split-window -h -t $SESSION_NAME:0
tmux split-window -v -t $SESSION_NAME:0.1

# Pane 0 (left): ComfyUI
tmux send-keys -t $SESSION_NAME:0.0 'cd /home/impostor/ComfyUI' C-m
tmux send-keys -t $SESSION_NAME:0.0 'while true; do echo "Starting ComfyUI..."; source .venv/bin/activate && python3.12 main.py; echo "ComfyUI crashed, restarting in 5 seconds..."; sleep 5; done' C-m

# Pane 1 (top right): Main Machine
tmux send-keys -t $SESSION_NAME:0.1 'cd /home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy' C-m
tmux send-keys -t $SESSION_NAME:0.1 'while true; do echo "Starting Machine..."; source .venv/bin/activate && python3.12 machine.py; echo "Machine crashed, restarting in 5 seconds..."; sleep 5; done' C-m

# Pane 2 (bottom right): Log Viewer
tmux send-keys -t $SESSION_NAME:0.2 'cd /home/impostor/impostor-log-viewer/webapp' C-m
tmux send-keys -t $SESSION_NAME:0.2 'while true; do echo "Starting Log Viewer..."; npm run start; echo "Log Viewer crashed, restarting in 5 seconds..."; sleep 5; done' C-m

# Attach to session
echo "Starting impostor system in tmux grid layout: $SESSION_NAME"
echo "Use 'tmux attach -t $SESSION_NAME' to view the session"
echo "Use Ctrl-B + arrow keys to navigate between panes"
echo "Use 'tmux kill-session -t $SESSION_NAME' to stop all processes"

tmux attach -t $SESSION_NAME


# tmux kill-session -t impostor-system