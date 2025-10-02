"""
Kernel loader utilities for PyMetallic.

This module provides utilities for loading Metal shader source code from external files.
"""

import os
from pathlib import Path


def get_kernel_source(kernel_name: str) -> str:
    """
    Load Metal kernel source code from the kernels directory.

    Args:
        kernel_name: Name of the kernel file (without .metal extension)

    Returns:
        str: The Metal shader source code

    Raises:
        FileNotFoundError: If the kernel file doesn't exist
    """
    kernel_dir = Path(__file__).parent
    kernel_file = kernel_dir / f"{kernel_name}.metal"

    if not kernel_file.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_file}")

    with open(kernel_file, 'r') as f:
        return f.read()


# Pre-load commonly used kernels for convenience
HELPER_KERNELS = get_kernel_source("helper_kernels")
DEMO_KERNELS = get_kernel_source("demo_kernels")
IMAGE_PROCESSING_KERNELS = get_kernel_source("image_processing")
GAME_OF_LIFE_KERNELS = get_kernel_source("game_of_life")
FLUID_SIMULATION_KERNELS = get_kernel_source("fluid_simulation")
LINALG_KERNELS = get_kernel_source("linalg_kernels")
