#!/usr/bin/env python3
"""
Example usage of the configurable impostor template system.
This demonstrates how to tweak various parameters in code.
"""

from drawing import ImpostorConfig, create_impostor_controller, ComfyUIController


def example_basic_usage():
    """Basic usage with default parameters."""
    print("=== Basic Usage ===")

    # Create controller with default configuration
    controller = create_impostor_controller()

    # Queue the prompt
    success = controller.queue_prompt()
    print(f"Queue successful: {success}")


def example_custom_config():
    """Example with custom configuration parameters."""
    print("\n=== Custom Configuration ===")

    # Create controller with custom parameters
    controller = create_impostor_controller(
        load_image_path="my_custom_image.jpg",
        primitive_string="watercolor painting style ",
        steps=30,
        sampler="euler",
        scheduler="karras",
        flux_guidance=3.5,
        cnet_strength=0.6,
        lora_strength=0.8,
        latent_width=1024,
        latent_height=1024,
        filename_prefix="custom-impostor",
    )

    # Queue the prompt
    success = controller.queue_prompt()
    print(f"Queue successful: {success}")


def example_config_object():
    """Example using ImpostorConfig object directly."""
    print("\n=== Using ImpostorConfig Object ===")

    # Create custom configuration
    config = ImpostorConfig(
        load_image_path="portrait.jpg",
        primitive_string="pencil sketch ",
        override_prompt="A detailed portrait of a person with expressive eyes",
        sampler="dpmpp_2m",
        steps=20,
        scheduler="karras",
        flux_guidance=4.5,
        cnet_strength=0.5,
        cnet_start_percent=0.1,
        lora_path="flux/own/impostor/impostor-64-balanced-v2-16k-no-trig.safetensors",
        lora_strength=1.2,
        noise_seed=42,
        latent_width=768,
        latent_height=1024,
        filename_prefix="portrait-impostor",
    )

    # Create controller with custom config
    controller = ComfyUIController(config=config)
    controller.set_workflow_file("drawing/impostor-template-gpupeon.json")

    # Queue the prompt
    success = controller.queue_prompt()
    print(f"Queue successful: {success}")


def example_dynamic_updates():
    """Example of updating configuration dynamically."""
    print("\n=== Dynamic Configuration Updates ===")

    # Create controller with default config
    controller = create_impostor_controller()

    # Update specific parameters
    controller.update_config(load_image_path="new_image.jpg", steps=35, flux_guidance=5.0, filename_prefix="dynamic-impostor")

    # Queue the prompt
    success = controller.queue_prompt()
    print(f"Queue successful: {success}")


def example_parameter_variations():
    """Example showing different parameter combinations."""
    print("\n=== Parameter Variations ===")

    # High quality, slow generation
    # high_quality = create_impostor_controller(steps=40, sampler="dpmpp_2m", scheduler="karras", cnet_strength=0.7, filename_prefix="hq-impostor")

    # # Fast generation
    # fast = create_impostor_controller(steps=15, sampler="euler", scheduler="simple", cnet_strength=0.3, filename_prefix="fast-impostor")

    # # Artistic style
    # artistic = create_impostor_controller(
    #     primitive_string="abstract expressionist painting ", flux_guidance=6.0, lora_strength=1.5, filename_prefix="artistic-impostor"
    # )

    print("Created three different configurations:")
    print("- High quality (40 steps, strong ControlNet)")
    print("- Fast generation (15 steps, weak ControlNet)")
    print("- Artistic style (abstract expressionist, strong LoRA)")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_config_object()
    example_dynamic_updates()
    example_parameter_variations()

    print("\n=== All Available Parameters ===")
    print("The following parameters can be configured:")
    print("- load_image_path: Path to input image")
    print("- primitive_string: Style prefix for prompts")
    print("- override_prompt: Custom prompt text")
    print("- sampler: Sampling method (deis, euler, dpmpp_2m, etc.)")
    print("- steps: Number of sampling steps")
    print("- scheduler: Noise schedule (beta, karras, simple, etc.)")
    print("- cnet_strength: ControlNet strength (0.0-1.0)")
    print("- cnet_start_percent: ControlNet start percentage (0.0-1.0)")
    print("- flux_guidance: Flux guidance scale")
    print("- lora_path: Path to LoRA model")
    print("- lora_strength: LoRA strength multiplier")
    print("- noise_seed: Random seed (None for random)")
    print("- latent_width: Generated image width")
    print("- latent_height: Generated image height")
    print("- filename_prefix: Output filename prefix")
