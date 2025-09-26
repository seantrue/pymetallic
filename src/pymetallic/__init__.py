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
    ComputePipelineState,
    Device,
    Function,
    Kernel,
    Library,
    MetalError,
    run_simple_compute_example,
)

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
    "Kernel",
    "run_simple_compute_example",
]

if __name__ == "__main__":
    print("PyMetallic: Python bindings for Apple Metal")
    print(
        "Note: This requires the Swift bridge library (libpymetallic.dylib) to be compiled and available"
    )
    # Uncomment to run example (requires compiled bridge library)
    run_simple_compute_example()
