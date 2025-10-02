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
        return False

    # Test scalar operations
    print("\n" + "=" * 40)
    print("Testing Scalar Operations")
    print("=" * 40)

    # Test scalar add
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    scalar_add_value = 10.0
    expected_add = test_data + scalar_add_value

    buffer_test = pymetallic.Buffer.from_numpy(device, test_data.copy())
    pymetallic.scalar_add(device, buffer_test, scalar_add_value)
    result_add = buffer_test.to_numpy(np.float32, test_data.shape)

    add_error = np.max(np.abs(result_add - expected_add))
    print(f"Scalar Add Test: {test_data[:3]} + {scalar_add_value} = {result_add[:3]}")
    print(f"  Max error: {add_error:.10f}")

    if add_error < 1e-6:
        print("  ✅ Scalar add completed successfully!")
    else:
        print("  ❌ Scalar add failed - results don't match expected values")
        return False

    # Test scalar multiply
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    scalar_mult_value = 3.0
    expected_mult = test_data * scalar_mult_value

    buffer_test = pymetallic.Buffer.from_numpy(device, test_data.copy())
    pymetallic.scalar_multiply(device, buffer_test, scalar_mult_value)
    result_mult = buffer_test.to_numpy(np.float32, test_data.shape)

    mult_error = np.max(np.abs(result_mult - expected_mult))
    print(f"\nScalar Multiply Test: {test_data[:3]} * {scalar_mult_value} = {result_mult[:3]}")
    print(f"  Max error: {mult_error:.10f}")

    if mult_error < 1e-6:
        print("  ✅ Scalar multiply completed successfully!")
    else:
        print("  ❌ Scalar multiply failed - results don't match expected values")
        return False

    # Test with larger arrays
    print("\n" + "=" * 40)
    print("Testing Scalar Operations on Large Arrays")
    print("=" * 40)

    large_size = 10000
    large_data = np.random.random(large_size).astype(np.float32)

    # Test add
    buffer_large = pymetallic.Buffer.from_numpy(device, large_data.copy())
    pymetallic.scalar_add(device, buffer_large, 5.0)
    result_large_add = buffer_large.to_numpy(np.float32)
    expected_large_add = large_data + 5.0
    large_add_error = np.max(np.abs(result_large_add - expected_large_add))

    print(f"Large array ({large_size} elements) scalar add max error: {large_add_error:.10f}")

    # Test multiply
    buffer_large = pymetallic.Buffer.from_numpy(device, large_data.copy())
    pymetallic.scalar_multiply(device, buffer_large, 2.5)
    result_large_mult = buffer_large.to_numpy(np.float32)
    expected_large_mult = large_data * 2.5
    large_mult_error = np.max(np.abs(result_large_mult - expected_large_mult))

    print(f"Large array ({large_size} elements) scalar multiply max error: {large_mult_error:.10f}")

    if large_add_error < 1e-5 and large_mult_error < 1e-5:
        print("✅ All scalar operations on large arrays completed successfully!")
    else:
        print("❌ Scalar operations on large arrays failed")
        return False

    print("\n" + "=" * 40)
    print("🎉 All tests passed successfully!")
    print("=" * 40)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
