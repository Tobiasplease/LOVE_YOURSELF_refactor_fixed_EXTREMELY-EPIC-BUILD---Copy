# ComfyUI Test Script

This script tests the ComfyUI integration module by sending workflow JSON files to a ComfyUI service.

## Usage

### Basic Usage

```bash
python test_comfy.py drawing/example_workflow.json
```

### Custom API URL

```bash
python test_comfy.py drawing/example_workflow.json http://your-comfy-server:8188/prompt
```

## Requirements

1. **ComfyUI Service**: Make sure ComfyUI is running and accessible

   - Default URL: `http://localhost:8188/prompt`
   - ComfyUI should be running with API enabled

2. **Workflow File**: A valid ComfyUI workflow JSON file
   - An example workflow file (`drawing/example_workflow.json`) is provided
   - You can export workflows from ComfyUI's web interface

## What the Script Tests

**File-based Workflow**: Tests `ComfyUIController.queue_prompt()`

- Loads workflow from JSON file
- Validates file exists and contains valid JSON
- Posts to ComfyUI API

## Example Output

```
ComfyUI Workflow Test Script
==================================================
Testing ComfyUI workflow:
  Workflow file: example_workflow.json
  API URL: http://localhost:8188/prompt
--------------------------------------------------
✓ Workflow file 'example_workflow.json' is valid JSON
Attempting to queue workflow...
✓ Workflow queued successfully!

==================================================
Test Results: 1/1 tests passed
✓ All tests passed!
```

## Troubleshooting

- **Connection errors**: Check that ComfyUI is running and the API URL is correct
- **JSON errors**: Validate your workflow file with a JSON validator
- **Import errors**: Make sure the `drawing` module is properly installed/available
