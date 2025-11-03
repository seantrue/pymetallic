#!/usr/bin/env python3
"""
PyMetallic: Python bindings for Apple Metal using swift-cffi
Architecture inspired by PyOpenCL for familiar API design
"""

# Public API now re-exported from metallic.py
from .metallic import (
    Buffer,
    CommandBuffer,
    CommandQueue,
    ComputeCommandEncoder,
    BlitCommandEncoder,
    ComputePipelineState,
    Device,
    Function,
    Kernel,
    Library,
    MetalError,
    async_buffer_from_numpy,
    run_simple_compute_example,
    scalar_add,
    scalar_multiply,
)


def get_default_device() -> Device:
    """Get the default Metal device."""
    return Device.get_default_device()


__all__ = [
    "MetalError",
    "Device",
    "CommandQueue",
    "Buffer",
    "Library",
    "Function",
    "ComputePipelineState",
    "CommandBuffer",
    "ComputeCommandEncoder",
    "BlitCommandEncoder",
    "Kernel",
    "async_buffer_from_numpy",
    "get_default_device",
    "run_simple_compute_example",
    "scalar_add",
    "scalar_multiply",
]

if __name__ == "__main__":
    print("PyMetallic: Python bindings for Apple Metal")
    print(
        "Note: This requires the Swift bridge library (libpymetallic.dylib) to be compiled and available"
    )
    # Uncomment to run example (requires compiled bridge library)
    run_simple_compute_example()
