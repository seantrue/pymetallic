#!/usr/bin/env python3
"""
Basic PyMetallic Compute Example

This example demonstrates the basic usage of PyMetallic for GPU compute operations.
It performs a simple vector addition operation on the GPU.
"""

import numpy as np

import pymetallic


def main():
    """Run a basic vector addition example on Metal GPU."""
    print("PyMetallic Basic Compute Example")
    print("=" * 40)

    # Get Metal device and create command queue
    device = pymetallic.Device.get_default_device()
    print(f"Using device: {device.name}")

    queue = pymetallic.CommandQueue(device)

    # Create test data
    size = 1024
    a = np.random.random(size).astype(np.float32)
    b = np.random.random(size).astype(np.float32)
    expected = a + b

    # Create Metal buffers
    buffer_a = pymetallic.Buffer.from_numpy(device, a)
    buffer_b = pymetallic.Buffer.from_numpy(device, b)
    buffer_result = pymetallic.Buffer(device, size * 4)  # 4 bytes per float32

    # Metal compute shader source
    shader_source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void vector_add(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
        result[index] = a[index] + b[index];
    }
    """

    # Compile shader and create compute pipeline
    library = pymetallic.Library(device, shader_source)
    function = library.make_function("vector_add")
    pipeline = pymetallic.ComputePipelineState(device, function)

    # Create command buffer and encoder
    command_buffer = queue.make_command_buffer()
    encoder = command_buffer.make_compute_command_encoder()

    # Set up compute pipeline and buffers
    encoder.set_compute_pipeline_state(pipeline)
    encoder.set_buffer(buffer_a, 0, 0)
    encoder.set_buffer(buffer_b, 0, 1)
    encoder.set_buffer(buffer_result, 0, 2)

    # Dispatch threads
    threads_per_threadgroup = 64
    (size + threads_per_threadgroup - 1) // threads_per_threadgroup
    encoder.dispatch_threads((size, 1, 1), (threads_per_threadgroup, 1, 1))
    encoder.end_encoding()

    # Execute and wait for completion
    command_buffer.commit()
    command_buffer.wait_until_completed()

    # Get results and verify
    result = buffer_result.to_numpy(np.float32, a.shape)

    # Check correctness
    max_error = np.max(np.abs(result - expected))
    print(f"Max error: {max_error:.10f}")

    if max_error < 1e-6:
        print("✅ Vector addition completed successfully!")
    else:
        print("❌ Vector addition failed - results don't match expected values")

    return max_error < 1e-6


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
