from __future__ import annotations

import json
import os
from urllib import request
from urllib.error import URLError
from typing import Optional


class ComfyUIController:
    """
    Controller for interacting with ComfyUI API to queue prompts and workflows.
    """

    def __init__(self, api_url: str = "http://localhost:8188/prompt", workflow_file: Optional[str] = None) -> None:
        """
        Initialize the ComfyUI controller.

        Args:
            api_url: The ComfyUI API endpoint URL
            workflow_file: Path to the workflow JSON file
        """
        self.api_url = api_url
        self.workflow_file = workflow_file

    def set_workflow_file(self, workflow_file: str) -> None:
        """Set the workflow file path."""
        self.workflow_file = workflow_file

    def queue_prompt(self) -> bool:
        """Load workflow JSON and queue it to the API"""
        if not self.workflow_file:
            print("Error: No workflow file specified")
            return False

        try:
            if not os.path.exists(self.workflow_file):
                print(f"Warning: Workflow file {self.workflow_file} not found")
                return False

            with open(self.workflow_file, "r") as file:
                workflow_data = json.load(file)

            prompt_workflow = {"prompt": workflow_data}
            data = json.dumps(prompt_workflow).encode("utf-8")
            req = request.Request(self.api_url, data=data)
            req.add_header("Content-Type", "application/json")

            response = request.urlopen(req, timeout=5)
            print(f"Workflow queued successfully: {response.status}")
            return True

        except (FileNotFoundError, json.JSONDecodeError, URLError) as e:
            print(f"Error with workflow or API: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error queuing workflow: {e}")
            return False
