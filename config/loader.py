import json
import os
from typing import Dict, Any


def load_config_override(json_path: str) -> Dict[str, Any]:
    """
    Load configuration overrides from a JSON file.

    Args:
        json_path: Path to the JSON configuration file

    Returns:
        Dictionary containing configuration overrides

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file not found: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def apply_config_overrides(config_module, overrides: Dict[str, Any]) -> None:
    """
    Apply configuration overrides to a config module.

    Args:
        config_module: The config module to modify
        overrides: Dictionary of configuration overrides
    """
    for key, value in overrides.items():
        if hasattr(config_module, key):
            # Get the original type to maintain type consistency
            original_value = getattr(config_module, key)
            if original_value is not None:
                original_type = type(original_value)
                # Convert the override value to match the original type
                try:
                    if original_type == bool and isinstance(value, str):
                        # Handle string boolean representations
                        converted_value = value.lower() in ("true", "1", "yes", "on")
                    else:
                        converted_value = original_type(value)
                    setattr(config_module, key, converted_value)
                    print(f"[CONFIG] Override: {key} = {converted_value} (was {original_value})")
                except (ValueError, TypeError) as e:
                    print(f"[CONFIG] Warning: Could not convert {key}={value} to {original_type.__name__}: {e}")
                    # Use the value as-is if conversion fails
                    setattr(config_module, key, value)
                    print(f"[CONFIG] Override: {key} = {value} (was {original_value}, type conversion failed)")
            else:
                setattr(config_module, key, value)
                print(f"[CONFIG] Override: {key} = {value} (was None)")
        else:
            print(f"[CONFIG] Warning: Unknown configuration key '{key}' in override file")
