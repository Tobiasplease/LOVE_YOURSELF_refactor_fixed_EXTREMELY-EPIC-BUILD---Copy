from __future__ import annotations

import json
import os
from urllib import request
from urllib.error import URLError
from typing import Optional, Dict, Any
from dataclasses import dataclass
import random


@dataclass
class ImpostorConfig:
    """
    Configuration for impostor template parameters.
    """

    # Image input
    load_image_path: str = "Photo on 2025-06-26 at 16.06.jpg"

    # Text prompts
    primitive_string: str = "black and white sketch line art "
    override_prompt: str = (
        "The image shows a man sitting on a gray armchair in a living room. He is wearing a blue plaid shirt and black headphones. "
    )

    # Sampling parameters
    sampler: str = "deis"
    steps: int = 25
    scheduler: str = "beta"

    # ControlNet parameters
    cnet_strength: float = 0.45
    cnet_start_percent: float = 0.0

    # Flux parameters
    flux_guidance: float = 4.0

    # LoRA parameters
    lora_path: str = "flux/own/impostor/impostor-64-balanced-v2-16k-no-trig.safetensors"
    lora_strength: float = 1.0

    # Generation parameters
    noise_seed: Optional[int] = None
    latent_width: int = 1328
    latent_height: int = 752

    # Output parameters
    filename_prefix: str = "impostor-out"


class ComfyUIController:
    """
    Controller for interacting with ComfyUI API to queue prompts and workflows.
    """

    def __init__(
        self, api_url: str = "http://localhost:8188/prompt", workflow_file: Optional[str] = None, config: Optional[ImpostorConfig] = None
    ) -> None:
        """
        Initialize the ComfyUI controller.

        Args:
            api_url: The ComfyUI API endpoint URL
            workflow_file: Path to the workflow JSON file
            config: Configuration for impostor template parameters
        """
        self.api_url = api_url
        self.workflow_file = workflow_file
        self.config = config or ImpostorConfig()

    def set_workflow_file(self, workflow_file: str) -> None:
        """Set the workflow file path."""
        self.workflow_file = workflow_file

    def set_config(self, config: ImpostorConfig) -> None:
        """Set the configuration for impostor template parameters."""
        self.config = config

    def update_config(self, **kwargs) -> None:
        """Update specific configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter '{key}' ignored")

    def _apply_config_to_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration parameters to workflow data."""
        # Generate random seed if not specified
        if self.config.noise_seed is None:
            self.config.noise_seed = random.randint(0, 2**32 - 1)

        # Apply parameters to workflow nodes
        updates = {
            # LoadImage path (node 607)
            "607": {"inputs": {"image": self.config.load_image_path}},
            # PrimitiveString (node 616)
            "616": {"inputs": {"value": self.config.primitive_string}},
            # OVERRIDE prompt (node 723)
            "723": {"inputs": {"String": self.config.override_prompt}},
            # Sampler (node 293)
            "293": {"inputs": {"sampler_name": self.config.sampler}},
            # Scheduler and steps (node 294)
            "294": {"inputs": {"scheduler": self.config.scheduler, "steps": self.config.steps, "denoise": 1}},
            # Noise seed (node 295)
            "295": {"inputs": {"noise_seed": self.config.noise_seed}},
            # Flux guidance (node 300)
            "300": {"inputs": {"guidance": self.config.flux_guidance}},
            # ControlNet strength and start percent (node 711)
            "711": {"inputs": {"strength": self.config.cnet_strength, "start_percent": self.config.cnet_start_percent}},
            # LoRA path and strength (node 741)
            "741": {"inputs": {"lora_1": {"on": True, "lora": self.config.lora_path, "strength": self.config.lora_strength}}},
            # Latent dimensions (node 5)
            "5": {"inputs": {"width": self.config.latent_width, "height": self.config.latent_height, "batch_size": 1}},
            # Output filename prefix (node 30)
            "30": {"inputs": {"filename_prefix": self.config.filename_prefix}},
        }

        # Apply updates to workflow
        for node_id, node_updates in updates.items():
            if node_id in workflow_data:
                for key, value in node_updates.items():
                    if key == "inputs":
                        workflow_data[node_id]["inputs"].update(value)
                    else:
                        workflow_data[node_id][key] = value

        return workflow_data

    def queue_prompt(self) -> bool:
        """Load workflow JSON, apply configuration, and queue it to the API"""
        if not self.workflow_file:
            print("Error: No workflow file specified")
            return False

        try:
            if not os.path.exists(self.workflow_file):
                print(f"Warning: Workflow file {self.workflow_file} not found")
                return False

            with open(self.workflow_file, "r") as file:
                workflow_data = json.load(file)

            # Apply configuration to workflow
            workflow_data = self._apply_config_to_workflow(workflow_data)

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


def create_impostor_controller(api_url: str = "http://localhost:8188/prompt", **config_params) -> ComfyUIController:
    """
    Create a ComfyUIController configured for the impostor template.

    Args:
        api_url: The ComfyUI API endpoint URL
        **config_params: Configuration parameters for ImpostorConfig

    Returns:
        ComfyUIController instance configured for impostor template
    """
    config = ImpostorConfig(**config_params)
    controller = ComfyUIController(api_url=api_url, config=config)

    # Set default workflow file path
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_workflow = os.path.join(script_dir, "impostor-template.json")
    controller.set_workflow_file(default_workflow)

    return controller
