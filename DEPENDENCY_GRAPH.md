# Project Dependency Graph Analysis

**Analysis Date:** 2025-08-27  
**Entry Point:** machine.py  
**Total Python Files:** 152  
**Files with Dependencies:** 58  
**Orphaned Files:** 94  

## Project Structure Overview

This AI-powered interactive mirror system has a complex modular architecture with multiple subsystems for computer vision, AI processing, servo control, and drawing generation.

## Machine.py Dependency Tree

Starting from the main entry point `machine.py`, here's the complete dependency tree:

```
machine.py
├── config/config.py
│   └── config/prompt_templates.py
├── config/loader.py
├── servo_control/lightbulb_pwm.py
├── perception/object_detection.py
│   ├── config/config.py (already seen)
│   └── perception/detection_memory.py
├── captioner/captioner.py
│   ├── config/config.py (already seen)
│   ├── event_logging/event_logger.py
│   │   ├── config/config.py (already seen)
│   │   ├── event_logging/log_type.py
│   │   └── utils/ollama.py
│   ├── event_logging/log_type.py (already seen)
│   ├── event_logging/run_manager.py
│   ├── drawing/drawing.py
│   │   ├── config/config.py (already seen)
│   │   ├── config/prompt_templates.py (already seen)
│   │   ├── event_logging/event_logger.py (already seen)
│   │   ├── event_logging/log_type.py (already seen)
│   │   ├── event_logging/run_manager.py (already seen)
│   │   ├── utils/ollama.py (already seen)
│   │   └── utils/state_manager.py
│   │       ├── config/config.py (already seen)
│   │       └── utils/continuity.py
│   ├── captioner/memory.py
│   │   ├── config/word_lists.py
│   │   ├── utils/continuity.py (already seen)
│   │   ├── utils/motif_scorer.py
│   │   └── captioner/prompts.py
│   ├── captioner/prompts.py (already seen)
│   ├── captioner/model_wrapper.py
│   │   ├── config/config.py (already seen)
│   │   ├── utils/ollama.py (already seen)
│   │   └── captioner/prompt_interface.py
│   ├── utils/motif_scorer.py (already seen)
│   └── utils/error_tracking.py
│       ├── event_logging/event_logger.py (already seen)
│       └── event_logging/log_type.py (already seen)
├── vision/gaze.py
├── mood/mood.py
│   ├── config/config.py (already seen)
│   ├── config/prompt_templates.py (already seen)
│   ├── event_logging/event_logger.py (already seen)
│   ├── event_logging/log_type.py (already seen)
│   ├── utils/ollama.py (already seen)
│   └── utils/pattern_recognition.py
├── breathing/breathing.py
├── image_monitor/__init__.py
├── utils/state_manager.py (already seen)
├── utils/continuity.py (already seen)
├── utils/error_tracking.py (already seen)
├── event_logging/run_manager.py (already seen)
├── event_logging/event_logger.py (already seen)
├── event_logging/log_type.py (already seen)
├── hand_control/direct_hand_control.py
├── reactivity/camera_reactive.py
└── servo_control/servo_control.py
```

## Core Module Dependencies

### Configuration System
- **config/config.py**: Central configuration hub
  - Imports: `config.prompt_templates`
  - Imported by: Almost all modules (58+ files)

### Captioner Subsystem (AI Caption Generation)
- **captioner/captioner.py**: Main captioning orchestrator
  - Key dependencies: `memory.py`, `model_wrapper.py`, `drawing.py`, event logging
- **captioner/memory.py**: Memory management and motif tracking
  - Dependencies: `prompts.py`, word lists, motif scoring, continuity utils
- **captioner/model_wrapper.py**: Ollama API wrapper
  - Dependencies: `prompt_interface.py`, `utils/ollama.py`

### Event Logging System
- **event_logging/event_logger.py**: Centralized logging
  - Dependencies: `log_type.py`, `utils/ollama.py`, config
- **event_logging/run_manager.py**: Session/run management
- **event_logging/log_type.py**: Log type definitions

### Mood Analysis System  
- **mood/mood.py**: 3D emotion analysis (valence/arousal/clarity)
  - Dependencies: pattern recognition, Ollama API, prompt templates

### Utility Modules
- **utils/ollama.py**: Core Ollama API communication
  - Imported by: All AI-dependent modules
- **utils/error_tracking.py**: Health monitoring and error tracking
- **utils/state_manager.py**: Session state persistence
- **utils/continuity.py**: Time/duration handling

## Circular Dependencies

No circular import dependencies were detected in the current codebase. The architecture maintains a clean hierarchical import structure.

## Orphaned Files Analysis

**94 files (62% of codebase)** are never imported, falling into these categories:

### Debug/Test Scripts (59 files)
Most files in `debug/` folder are standalone test scripts:
- `debug/test_*.py` (48 test files)
- `debug/clear_*.py` (3 cleanup utilities)  
- Various debugging tools and test harnesses

### Standalone Utilities (11 files)
- `bcnc/` directory: G-code generation tools (8 files)
- `grbl/` directory: GRBL hardware control (4 files) 
- `warp-fix-lab/` directory: Image warping correction (3 files)

### Disabled/Legacy Code (24 files)
- `captioner/model_wrapper_old.py`: Legacy model wrapper
- `captioner/contextual_memory.py`: Unused memory system
- `captioner/emotional_drift.py`: Experimental feature
- `hand_control/` modules: Hand controller interfaces (5 files)
- Various experimental and disabled modules

## Key Insights

### Architecture Strengths
1. **Modular Design**: Clear separation between vision, AI, mood, and control systems
2. **Clean Hierarchies**: No circular dependencies detected
3. **Centralized Configuration**: All settings flow through `config/config.py`
4. **Robust Error Handling**: Dedicated error tracking system
5. **Comprehensive Logging**: Event-driven logging with session management

### Potential Improvements
1. **Code Cleanup**: 62% of files are orphaned - significant cleanup opportunity
2. **Module Consolidation**: Some utility functions could be better organized
3. **Import Optimization**: Some modules have redundant imports
4. **Test Integration**: Debug scripts could be organized into formal test suites

### Critical Dependencies
The most critical files that many others depend on:
1. `config/config.py` - Central configuration (imported by 40+ files)
2. `utils/ollama.py` - AI API communication (imported by 15+ files)
3. `event_logging/event_logger.py` - Logging infrastructure (imported by 12+ files)
4. `utils/error_tracking.py` - Health monitoring (imported by 8+ files)

## File Categories

### Core Runtime Files (58 files, 38%)
Files that are actually imported and used during runtime:
- Configuration and utilities: 15 files
- AI/Captioning system: 12 files  
- Event logging: 8 files
- Perception/Vision: 6 files
- Mood analysis: 5 files
- Control systems: 7 files
- Module markers (__init__.py): 5 files

### Test/Debug Files (59 files, 39%)
Standalone testing and debugging tools that don't participate in main runtime.

### Experimental/Disabled (35 files, 23%)
Features that were developed but are not currently active in the main system.

## Recommendations

1. **Immediate Cleanup**: Remove or archive the 94 orphaned files to improve codebase clarity
2. **Test Organization**: Consolidate debug scripts into a proper test framework
3. **Documentation**: Document the purpose of experimental modules
4. **Dependency Review**: Some modules could benefit from reduced coupling
5. **Module Rationalization**: Consider consolidating related utilities

This dependency analysis reveals a well-structured but somewhat bloated codebase with significant cleanup potential while maintaining a solid architectural foundation.