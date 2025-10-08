"""Utilities for argument parsing with YAML defaults."""

import argparse
import yaml
import os


def load_defaults_from_yaml(yaml_path):
    """
    Load default argument values from a YAML file.

    Args:
        yaml_path: Path to YAML file containing default values

    Returns:
        Dictionary of default values
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        defaults = yaml.safe_load(f)

    return defaults


def create_parser_from_yaml(yaml_path, parser=None):
    """
    Create or configure an argument parser using defaults from a YAML file.

    Args:
        yaml_path: Path to YAML file containing default values
        parser: Existing ArgumentParser to configure (creates new one if None)

    Returns:
        ArgumentParser with defaults set from YAML file
    """
    if parser is None:
        parser = argparse.ArgumentParser()

    defaults = load_defaults_from_yaml(yaml_path)
    parser.set_defaults(**defaults)

    return parser
