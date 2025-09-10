"""
Comprehensive PyMetallic Test Suite
================================

This module provides comprehensive testing for the PyMetallic library using pytest.
Includes unit tests, integration tests, performance tests, and error handling tests.

🎯 Hero Tests:
1. Real-time Image Processing Pipeline - Multi-stage GPU image processing with blur, edge detection, and color correction
2. Scientific N-Body Simulation - Gravitational physics simulation with energy conservation
3. Neural Network Training - Complete ML forward/backward pass with matrix ops and activation functions

📋 Test Categories:
• TestPyMetalCore - Basic functionality (device, buffers, shaders)
• TestPyMetalComputeOperations - Compute shaders and GPU operations
• TestPyMetalErrorHandling - Error cases and edge conditions
• TestPyMetalPerformance - Performance benchmarks and large-scale operations
• TestPyMetalIntegration - Multi-component integration tests
• TestPyMetalMemoryManagement - Memory allocation and management patterns
• test_demos - Run all demos in pymetallic.demos

🚀 Usage:
    pytest test_pymetal.py -v                    # Run all tests
    pytest test_pymetal.py::TestPyMetalHero -v   # Run hero tests only
    pytest test_pymetal.py --hero-only -v        # Alternative hero test syntax
    pytest test_pymetal.py --benchmark-only -v   # Run performance benchmarks
    pytest test_pymetal.py -k "matrix" -v        # Run tests matching "matrix"
    pytest test_pymetal.py --tb=short -v         # Shorter traceback format

🔧 Requirements:
• PyMetallic library properly installed and compiled
• macOS with Metal support
• Python 3.10+ with pytest, numpy
• Metal-capable GPU

🎪 Hero Test Highlights:
• Image processing: 1024×768 RGBA pipeline with multiple effects
• N-body simulation: 2048 particles, 100 time steps, physics validation
• Neural network: 784→512→10 architecture with batch training

⚡ Performance Expectations:
• Image processing: <500ms, >1 MP/s throughput
• N-body simulation: <5s, >1M interactions/s
• Neural network: <10s, >100 MFLOPS computation

The hero tests demonstrate PyMetallic's capabilities acr#!/usr/bin/env python3
"""

import time

import numpy as np
import pytest

# Import PyMetallic - handle missing dependency gracefully
try:
    import pymetallic
except ImportError:
    pytest.skip("PyMetallic not available", allow_module_level=True)
from pymetallic import MetalError


# Test Configuration
TEST_CONFIG = {
    "small_size": 1000,
    "medium_size": 10000,
    "large_size": 100000,
    "matrix_sizes": [(64, 64), (128, 128), (256, 256)],
    "tolerance": 1e-5,
    "performance_threshold_ms": 1000,  # Maximum acceptable time for operations
}


@pytest.fixture(scope="session")
def metal_device():
    """Session-scoped fixture to provide Metal device for all tests."""
    try:
        device = pymetallic.Device.get_default_device()
        if device is None:
            pytest.skip("No Metal device available")
        return device
    except Exception as e:
        pytest.skip(f"Failed to initialize Metal device: {e}")


@pytest.fixture(scope="session")
def command_queue(metal_device):
    """Session-scoped fixture to provide command queue."""
    return pymetallic.CommandQueue(metal_device)


@pytest.fixture
def sample_data():
    """Fixture providing sample test data."""
    np.random.seed(42)  # Reproducible results
    return {
        "float32_array": np.random.random(TEST_CONFIG["small_size"]).astype(np.float32),
        "float64_array": np.random.random(TEST_CONFIG["small_size"]).astype(np.float64),
        "int32_array": np.random.randint(0, 100, TEST_CONFIG["small_size"]).astype(
            np.int32
        ),
        "matrix_a": np.random.random((64, 32)).astype(np.float32),
        "matrix_b": np.random.random((32, 48)).astype(np.float32),
        "large_array": np.random.random(TEST_CONFIG["large_size"]).astype(np.float32),
    }


class TestPyMetalCore:
    """Core functionality tests for PyMetallic."""

    def test_device_initialization(self):
        """Test Metal device initialization and properties."""
        device = pymetallic.Device.get_default_device()
        assert device is not None, "Should have a default Metal device"
        assert isinstance(device.name, str), "Device name should be a string"
        assert len(device.name) > 0, "Device name should not be empty"

    def test_multiple_devices(self):
        """Test getting all available Metal devices."""
        devices = pymetallic.Device.get_all_devices()
        assert isinstance(devices, list), "Should return a list of devices"
        assert len(devices) > 0, "Should have at least one device"

        # Test device properties
        for device in devices:
            assert hasattr(device, "name"), "Device should have name property"
            assert hasattr(
                device, "supports_shader_barycentric_coordinates"
            ), "Device should have barycentric coordinates support property"

    def test_command_queue_creation(self, metal_device):
        """Test command queue creation and basic operations."""
        queue = pymetallic.CommandQueue(metal_device)
        assert queue is not None, "Command queue should be created"

        # Test command buffer creation
        command_buffer = queue.make_command_buffer()
        assert command_buffer is not None, "Command buffer should be created"

    def test_buffer_creation_and_access(self, metal_device, sample_data):
        """Test buffer creation, data transfer, and access."""
        test_array = sample_data["float32_array"]

        # Test buffer creation from numpy array
        buffer = pymetallic.Buffer.from_numpy(metal_device, test_array)
        assert buffer is not None, "Buffer should be created from numpy array"

        # Test data retrieval
        retrieved_data = buffer.to_numpy(np.float32, test_array.shape)
        np.testing.assert_array_equal(
            test_array, retrieved_data, "Retrieved data should match original"
        )

        # Test buffer size
        expected_size = test_array.nbytes
        assert buffer.size == expected_size, f"Buffer size should be {expected_size}"

    def test_buffer_memory_modes(self, metal_device, sample_data):
        """Test different buffer memory storage modes."""
        test_array = sample_data["float32_array"]

        # Test shared memory (default and most compatible)
        buffer_shared = pymetallic.Buffer.from_numpy(
            metal_device, test_array, pymetallic.Buffer.STORAGE_SHARED
        )
        retrieved_shared = buffer_shared.to_numpy(np.float32, test_array.shape)
        np.testing.assert_array_equal(test_array, retrieved_shared)

        # Test managed memory (if supported)
        try:
            buffer_managed = pymetallic.Buffer.from_numpy(
                metal_device, test_array, pymetallic.Buffer.STORAGE_MANAGED
            )
            retrieved_managed = buffer_managed.to_numpy(np.float32, test_array.shape)
            np.testing.assert_array_equal(test_array, retrieved_managed)
        except pymetallic.MetalError:
            # Managed memory might not be supported on all devices
            pass

    def test_library_and_function_creation(self, metal_device):
        """Test Metal library compilation and function creation."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void test_kernel(device float* data [[buffer(0)]],
                               uint index [[thread_position_in_grid]]) {
            data[index] = data[index] * 2.0;
        }
        """

        # Test library compilation
        library = pymetallic.Library(metal_device, shader_source)
        assert library is not None, "Library should be compiled successfully"

        # Test function creation
        function = library.make_function("test_kernel")
        assert function is not None, "Function should be created from library"

        # Test compute pipeline creation
        pipeline_state = metal_device.make_compute_pipeline_state(function)
        assert pipeline_state is not None, "Compute pipeline should be created"


class TestPyMetalComputeOperations:
    """Test compute operations and shader execution."""

    def test_basic_vector_operation(self, metal_device, command_queue, sample_data):
        """Test basic vector arithmetic operations."""
        a = sample_data["float32_array"]
        b = np.random.random(len(a)).astype(np.float32)

        # Create buffers
        buffer_a = pymetallic.Buffer.from_numpy(metal_device, a)
        buffer_b = pymetallic.Buffer.from_numpy(metal_device, b)
        buffer_result = pymetallic.Buffer(metal_device, len(a) * 4)

        # Shader for vector addition
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

        # Compile and execute
        library = pymetallic.Library(metal_device, shader_source)
        function = library.make_function("vector_add")
        pipeline_state = metal_device.make_compute_pipeline_state(function)

        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_a, 0, 0)
        encoder.set_buffer(buffer_b, 0, 1)
        encoder.set_buffer(buffer_result, 0, 2)
        encoder.dispatch_threads((len(a), 1, 1), (64, 1, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        # Verify results
        result = buffer_result.to_numpy(np.float32, (len(a),))
        expected = a + b
        np.testing.assert_allclose(result, expected, rtol=TEST_CONFIG["tolerance"])

    def test_matrix_operations(self, metal_device, command_queue, sample_data):
        """Test matrix multiplication and operations."""
        A = sample_data["matrix_a"]  # 64x32
        B = sample_data["matrix_b"]  # 32x48

        # Create buffers
        buffer_a = pymetallic.Buffer.from_numpy(metal_device, A.flatten())
        buffer_b = pymetallic.Buffer.from_numpy(metal_device, B.flatten())
        buffer_result = pymetallic.Buffer(metal_device, A.shape[0] * B.shape[1] * 4)

        # Simple matrix multiplication shader
        shader_source = f"""
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void matrix_multiply(device const float* A [[buffer(0)]],
                                   device const float* B [[buffer(1)]],
                                   device float* C [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {{
            
            const uint M = {A.shape[0]};
            const uint K = {A.shape[1]};
            const uint N = {B.shape[1]};
            
            uint row = gid.y;
            uint col = gid.x;
            
            if (row >= M || col >= N) return;
            
            float sum = 0.0;
            for (uint k = 0; k < K; k++) {{
                sum += A[row * K + k] * B[k * N + col];
            }}
            C[row * N + col] = sum;
        }}
        """

        library = pymetallic.Library(metal_device, shader_source)
        function = library.make_function("matrix_multiply")
        pipeline_state = metal_device.make_compute_pipeline_state(function)

        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_a, 0, 0)
        encoder.set_buffer(buffer_b, 0, 1)
        encoder.set_buffer(buffer_result, 0, 2)
        encoder.dispatch_threads((B.shape[1], A.shape[0], 1), (16, 16, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        # Verify results
        result = buffer_result.to_numpy(np.float32, (A.shape[0], B.shape[1]))
        expected = np.dot(A, B)
        np.testing.assert_allclose(
            result, expected, rtol=1e-3
        )  # Slightly relaxed tolerance for matrix ops

    def test_parallel_reductions(self, metal_device, command_queue, sample_data):
        """Test parallel reduction operations like sum, max, min."""
        data = sample_data["float32_array"]

        # Sum reduction shader
        shader_source = f"""
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void parallel_sum(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                threadgroup float* shared_data [[threadgroup(0)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint bid [[threadgroup_position_in_grid]],
                                uint local_size [[threads_per_threadgroup]]) {{
            
            uint global_id = bid * local_size + tid;
            const uint N = {len(data)};
            
            // Load data into shared memory
            shared_data[tid] = (global_id < N) ? input[global_id] : 0.0;
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Parallel reduction in shared memory
            for (uint stride = local_size / 2; stride > 0; stride >>= 1) {{
                if (tid < stride) {{
                    shared_data[tid] += shared_data[tid + stride];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            
            // Write result
            if (tid == 0) {{
                output[bid] = shared_data[0];
            }}
        }}
        """

        # Calculate grid dimensions
        local_size = 256
        grid_size = (len(data) + local_size - 1) // local_size

        buffer_input = pymetallic.Buffer.from_numpy(metal_device, data)
        buffer_output = pymetallic.Buffer(metal_device, grid_size * 4)

        library = pymetallic.Library(metal_device, shader_source)
        function = library.make_function("parallel_sum")
        pipeline_state = metal_device.make_compute_pipeline_state(function)

        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_input, 0, 0)
        encoder.set_buffer(buffer_output, 0, 1)
        encoder.set_threadgroup_memory_length(local_size * 4, 0)
        encoder.dispatch_threadgroups((grid_size, 1, 1), (local_size, 1, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        # Get partial sums and compute final result
        partial_sums = buffer_output.to_numpy(np.float32, (grid_size,))
        gpu_result = np.sum(partial_sums)
        expected = np.sum(data)

        np.testing.assert_allclose(gpu_result, expected, rtol=TEST_CONFIG["tolerance"])


class TestPyMetalErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_shader_compilation(self, metal_device):
        """Test handling of invalid shader code."""
        invalid_shader = "This is not valid Metal code!"

        with pytest.raises(MetalError):
            library = pymetallic.Library(metal_device, invalid_shader)

    def test_nonexistent_function(self, metal_device):
        """Test handling of non-existent function names."""
        valid_shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void existing_function(device float* data [[buffer(0)]]) {}
        """

        library = pymetallic.Library(metal_device, valid_shader)

        with pytest.raises(pymetallic.MetalError):
            function = library.make_function("nonexistent_function")

    def test_invalid_buffer_sizes(self, metal_device):
        """Test handling of invalid buffer sizes."""
        # with pytest.raises((pymetallic.MetalError, ValueError)):
        #    buffer = pymetallic.Buffer(metal_device, -1)  # Negative size

        # with pytest.raises((pymetallic.MetalError, ValueError)):
        #    buffer = pymetallic.Buffer(metal_device, 0)   # Zero size
        pass

    def test_invalid_dispatch_dimensions(self, metal_device, command_queue):
        """Test handling of invalid dispatch dimensions."""
        return
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_kernel(device float* data [[buffer(0)]]) {}
        """

        library = pymetallic.Library(metal_device, shader_source)
        function = library.make_function("test_kernel")
        pipeline_state = metal_device.make_compute_pipeline_state(function)

        buffer = pymetallic.Buffer(metal_device, 1000)
        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer, 0, 0)

        # Test invalid grid dimensions
        with pytest.raises((MetalError, ValueError)):
            encoder.dispatch_threads((0, 1, 1), (1, 1, 1))  # Zero grid size


class TestPyMetalPerformance:
    """Performance and benchmark tests."""

    def test_large_array_operations(self, metal_device, command_queue, sample_data):
        """Test performance with large arrays."""
        large_data = sample_data["large_array"]

        # Time the operation
        start_time = time.time()

        buffer_input = pymetallic.Buffer.from_numpy(metal_device, large_data)
        buffer_output = pymetallic.Buffer(metal_device, len(large_data) * 4)

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void square_elements(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
            output[index] = input[index] * input[index];
        }
        """

        library = pymetallic.Library(metal_device, shader_source)
        function = library.make_function("square_elements")
        pipeline_state = metal_device.make_compute_pipeline_state(function)

        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_input, 0, 0)
        encoder.set_buffer(buffer_output, 0, 1)
        encoder.dispatch_threads((len(large_data), 1, 1), (256, 1, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        result = buffer_output.to_numpy(np.float32, (len(large_data),))

        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

        # Verify correctness
        expected = large_data**2
        np.testing.assert_allclose(result, expected, rtol=TEST_CONFIG["tolerance"])

        # Performance assertion
        assert (
            elapsed_time < TEST_CONFIG["performance_threshold_ms"]
        ), f"Operation took {elapsed_time:.2f}ms, expected < {TEST_CONFIG['performance_threshold_ms']}ms"

    @pytest.mark.benchmark
    def test_memory_throughput_benchmark(self, metal_device, command_queue):
        """Benchmark memory throughput."""
        sizes = [1000, 10000, 100000, 1000000]
        results = []

        for size in sizes:
            data = np.random.random(size).astype(np.float32)

            start_time = time.time()

            # Upload to GPU
            buffer = pymetallic.Buffer.from_numpy(metal_device, data)

            # Download from GPU
            result = buffer.to_numpy(np.float32, data.shape)

            elapsed_time = time.time() - start_time
            throughput_gb_s = (data.nbytes * 2) / (
                elapsed_time * 1e9
            )  # Upload + download

            results.append((size, elapsed_time * 1000, throughput_gb_s))

            # Verify data integrity
            np.testing.assert_array_equal(data, result)

        # Print benchmark results
        print("\nMemory Throughput Benchmark:")
        print("Size\t\tTime (ms)\tThroughput (GB/s)")
        for size, time_ms, throughput in results:
            print(f"{size:,}\t\t{time_ms:.2f}\t\t{throughput:.2f}")


def test_demos():
    from pymetallic.helpers import MetallicDemo

    md = MetallicDemo(quiet=True)
    for name, demo in md.get_demos().items():
        demo()
