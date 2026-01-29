from __future__ import annotations

import json
from pathlib import Path

from .presets import PROFILES, get_profile
from .schema import DecisionProfile


def load_profile_from_dict(d: dict) -> DecisionProfile:
    """
    Load a DecisionProfile from a dictionary.
    
    Raises:
        ValueError: If the dictionary is invalid.
    """
    try:
        return DecisionProfile.model_validate(d)
    except Exception as e:
        raise ValueError(f"Invalid profile dictionary: {e}") from e


def load_profile_from_json(path: str) -> DecisionProfile:
    """
    Load a DecisionProfile from a JSON file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is invalid or the profile is invalid.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    
    try:
        content = path_obj.read_text(encoding="utf-8")
        data = json.loads(content)
        return load_profile_from_dict(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in profile file: {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading profile from {path}: {e}") from e


def get_profile_or_load(name_or_path: str) -> DecisionProfile:
    """
    Get a profile by name (if it's a built-in preset) or load from a file path.
    
    Args:
        name_or_path: Either a preset name (e.g., "balanced") or a file path to a JSON profile.
    
    Returns:
        DecisionProfile instance.
    
    Raises:
        ValueError: If name doesn't match a preset and path doesn't exist or is invalid.
    """
    # First, check if it's a built-in preset
    if name_or_path in PROFILES:
        return get_profile(name_or_path)
    
    # Check if it looks like a path and exists
    path_obj = Path(name_or_path)
    if path_obj.exists():
        return load_profile_from_json(name_or_path)
    
    # Neither preset nor valid path
    available = ", ".join(sorted(PROFILES.keys()))
    raise ValueError(
        f"Profile '{name_or_path}' not found. "
        f"Available presets: {available}. "
        f"If specifying a file path, ensure it exists."
    )

