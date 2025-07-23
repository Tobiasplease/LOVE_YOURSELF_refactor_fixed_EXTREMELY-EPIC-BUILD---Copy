#!/usr/bin/env python3
"""
Test script for ComfyUI integration.
Usage: python test_comfy.py <workflow.json> [api_url]
"""

import sys
import argparse
import json
from pathlib import Path
from drawing import ComfyUIController


def validate_workflow_file(filepath: str) -> bool:
    """Validate that the workflow file exists and is valid JSON."""
    if not Path(filepath).exists():
        print(f"Error: Workflow file '{filepath}' not found")
        return False

    try:
        with open(filepath, "r") as f:
            json.load(f)
        print(f"✓ Workflow file '{filepath}' is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in workflow file: {e}")
        return False


def test_comfy_workflow(workflow_file: str, api_url: str = "http://localhost:8188/prompt") -> bool:
    """Test the ComfyUI workflow with the given file."""
    print("Testing ComfyUI workflow:")
    print(f"  Workflow file: {workflow_file}")
    print(f"  API URL: {api_url}")
    print("-" * 50)

    # Validate workflow file first
    if not validate_workflow_file(workflow_file):
        return False

    # Initialize ComfyUI controller
    comfy = ComfyUIController(api_url=api_url, workflow_file=workflow_file)

    # Test the queue_prompt method
    print("Attempting to queue workflow...")
    success = comfy.queue_prompt()

    if success:
        print("✓ Workflow queued successfully!")
        return True
    else:
        print("✗ Failed to queue workflow")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test ComfyUI workflow integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_comfy.py workflow.json
  python test_comfy.py workflow.json http://localhost:8188/prompt
        """,
    )

    parser.add_argument("workflow_file", help="Path to the workflow JSON file")

    parser.add_argument("api_url", nargs="?", default="http://localhost:8188/prompt", help="ComfyUI API URL (default: http://localhost:8188/prompt)")

    args = parser.parse_args()

    # Check if workflow file exists
    if not Path(args.workflow_file).exists():
        print(f"Error: Workflow file '{args.workflow_file}' not found")
        sys.exit(1)

    print("ComfyUI Workflow Test Script")
    print("=" * 50)

    success_count = 0
    total_tests = 1

    # Test file-based workflow method
    if test_comfy_workflow(args.workflow_file, args.api_url):
        success_count += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_tests} tests passed")

    if success_count == total_tests:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
