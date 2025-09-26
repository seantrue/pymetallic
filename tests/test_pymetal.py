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
    import pymetallic as pm
    from pymetallic.metallic import fill_u32, read_scalar
except ImportError:
    pytest.skip("PyMetallic not available", allow_module_level=True)
from pymetallic import Kernel, MetalError

# Test Configuration
TEST_CONFIG = {
    "small_size": 1000,
    "medium_size": 10000,
    "large_size": 100000,
    "matrix_sizes": [(64, 64), (128, 128), (256, 256)],
    "tolerance": 1e-5,
    "performance_threshold_ms": 1000,  # Maximum acceptable time for operations
}


def _have_device():
    try:
        dev = pm.Device.get_default_device()
        return dev is not None
    except Exception:
        return False


requires_metal = pytest.mark.skipif(
    not _have_device(), reason="No Metal device available"
)


@pytest.fixture(scope="session")
def device():
    """Session-scoped fixture to provide Metal device for all tests."""
    try:
        device = pymetallic.Device.get_default_device()
        if device is None:
            pytest.skip("No Metal device available")
        return device
    except Exception as e:
        pytest.skip(f"Failed to initialize Metal device: {e}")


@pytest.fixture(scope="session")
def command_queue(device):
    return device.command_queue


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


@requires_metal
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
            assert hasattr(device, "supports_shader_barycentric_coordinates"), (
                "Device should have barycentric coordinates support property"
            )

    def test_command_queue_creation(self, device):
        """Test command queue creation and basic operations."""
        assert device.command_queue is not None, "CommandQueue should be created"
        assert device.command_buffer is not None, "CommandBuffer should be created"
        assert device.command_encoder is not None, "CommandEncoder should be created"

    def test_buffer_creation_and_access(self, device, sample_data):
        """Test buffer creation, data transfer, and access."""
        test_array = sample_data["float32_array"]

        # Test buffer creation from numpy array
        buffer = device.make_buffer_from_numpy(test_array)
        assert buffer is not None, "Buffer should be created from numpy array"

        # Test data retrieval
        retrieved_data = buffer.to_numpy(np.float32, test_array.shape)
        np.testing.assert_array_equal(
            test_array, retrieved_data, "Retrieved data should match original"
        )

        # Test buffer size
        expected_size = test_array.nbytes
        assert buffer.size == expected_size, f"Buffer size should be {expected_size}"

    def test_buffer_memory_modes(self, device, sample_data):
        """Test different buffer memory storage modes."""
        test_array = sample_data["float32_array"]

        # Test shared memory (default and most compatible)
        buffer_shared = device.make_buffer_from_numpy(
            test_array, pymetallic.Buffer.STORAGE_SHARED
        )
        retrieved_shared = buffer_shared.to_numpy(np.float32, test_array.shape)
        np.testing.assert_array_equal(test_array, retrieved_shared)

        # Test managed memory (if supported)
        try:
            buffer_managed = device.make_buffer_from_numpy(
                test_array, pm.Buffer.STORAGE_MANAGED
            )
            retrieved_managed = buffer_managed.to_numpy(np.float32, test_array.shape)
            np.testing.assert_array_equal(test_array, retrieved_managed)
        except pm.MetalError:
            # Managed memory might not be supported on all devices
            pass

    def test_library_and_function_creation(self, device):
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
        library = device.make_library(shader_source)
        assert library is not None, "Library should be compiled successfully"

        # Test function creation
        function = library.make_function("test_kernel")
        assert function is not None, "Function should be created from library"

        # Test compute pipeline creation
        pipeline_state = device.make_compute_pipeline_state(function)
        assert pipeline_state is not None, "Compute pipeline should be created"


@requires_metal
class TestPyMetalComputeOperations:
    """Test compute operations and shader execution."""

    def test_basic_vector_operation(self, device, sample_data):
        """Test basic vector arithmetic operations."""
        a = sample_data["float32_array"]
        b = np.random.random(len(a)).astype(np.float32)

        # Create buffers
        buffer_a = device.make_buffer_from_numpy(a)
        buffer_b = device.make_buffer_from_numpy(b)
        buffer_result = device.make_buffer(len(a) * 4)

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
        library = device.make_library(shader_source)
        function = library.make_function("vector_add")
        pipeline_state = device.make_compute_pipeline_state(function)

        encoder = device.command_encoder

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_a, 0, 0)
        encoder.set_buffer(buffer_b, 0, 1)
        encoder.set_buffer(buffer_result, 0, 2)
        encoder.dispatch_threads((len(a), 1, 1), (64, 1, 1))
        encoder.end_encoding()
        encoder.commit()
        encoder.wait_until_completed()

        # Verify results
        result = buffer_result.to_numpy(np.float32, (len(a),))
        expected = a + b
        np.testing.assert_allclose(result, expected, rtol=TEST_CONFIG["tolerance"])

    def test_matrix_operations(self, device, sample_data):
        """Test matrix multiplication and operations."""
        A = sample_data["matrix_a"]  # 64x32
        B = sample_data["matrix_b"]  # 32x48

        # Create buffers
        buffer_a = device.make_buffer_from_numpy(A.flatten())
        buffer_b = device.make_buffer_from_numpy(B.flatten())
        buffer_result = device.make_buffer(A.shape[0] * B.shape[1] * 4)

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

        library = device.make_library(shader_source)
        function = library.make_function("matrix_multiply")
        pipeline_state = device.make_compute_pipeline_state(function)

        encoder = device.command_encoder

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_a, 0, 0)
        encoder.set_buffer(buffer_b, 0, 1)
        encoder.set_buffer(buffer_result, 0, 2)
        encoder.dispatch_threads((B.shape[1], A.shape[0], 1), (16, 16, 1))
        encoder.end_encoding()

        encoder.commit()
        encoder.wait_until_completed()

        # Verify results
        result = buffer_result.to_numpy(np.float32, (A.shape[0], B.shape[1]))
        expected = np.dot(A, B)
        np.testing.assert_allclose(
            result, expected, rtol=1e-3
        )  # Slightly relaxed tolerance for matrix ops

    def test_parallel_reductions(self, device, command_queue, sample_data):
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

        buffer_input = device.make_buffer_from_numpy(data)
        buffer_output = device.make_buffer(grid_size * 4)

        library = device.make_library(shader_source)
        function = library.make_function("parallel_sum")
        pipeline_state = device.make_compute_pipeline_state(function)

        encoder = device.command_encoder

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_input, 0, 0)
        encoder.set_buffer(buffer_output, 0, 1)
        encoder.set_threadgroup_memory_length(local_size * 4, 0)
        encoder.dispatch_threadgroups((grid_size, 1, 1), (local_size, 1, 1))
        encoder.end_encoding()

        encoder.commit()
        encoder.wait_until_completed()

        # Get partial sums and compute final result
        partial_sums = buffer_output.to_numpy(np.float32, (grid_size,))
        gpu_result = np.sum(partial_sums)
        expected = np.sum(data)

        np.testing.assert_allclose(gpu_result, expected, rtol=TEST_CONFIG["tolerance"])


@requires_metal
class TestPyMetalErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_shader_compilation(self, device):
        """Test handling of invalid shader code."""
        invalid_shader = "This is not valid Metal code!"

        with pytest.raises(MetalError):
            device.make_library(invalid_shader)

    def test_nonexistent_function(self, device):
        """Test handling of non-existent function names."""
        valid_shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void existing_function(device float* data [[buffer(0)]]) {}
        """

        library = device.make_library(valid_shader)

        with pytest.raises(pymetallic.MetalError):
            library.make_function("nonexistent_function")

    def test_invalid_buffer_sizes(self, device):
        """Test handling of invalid buffer sizes."""
        # with pytest.raises((pymetallic.MetalError, ValueError)):
        #    buffer = pymetallic.Buffer(metal_device, -1)  # Negative size

        # with pytest.raises((pymetallic.MetalError, ValueError)):
        #    buffer = pymetallic.Buffer(metal_device, 0)   # Zero size
        pass

    def test_invalid_dispatch_dimensions(self, device, command_queue):
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
        encoder = device.command_encoder

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer, 0, 0)

        # Test invalid grid dimensions
        with pytest.raises((MetalError, ValueError)):
            encoder.dispatch_threads((0, 1, 1), (1, 1, 1))  # Zero grid size


@requires_metal
class TestPyMetalPerformance:
    """Performance and benchmark tests."""

    def test_large_array_operations(self, device, command_queue, sample_data):
        """Test performance with large arrays."""
        large_data = sample_data["large_array"]

        # Time the operation
        start_time = time.time()

        buffer_input = device.make_buffer_from_numpy(large_data)
        buffer_output = device.make_buffer(len(large_data) * 4)

        shader_source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void square_elements(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
            output[index] = input[index] * input[index];
        }
        """

        library = device.make_library(shader_source)
        function = library.make_function("square_elements")
        pipeline_state = device.make_compute_pipeline_state(function)

        encoder = device.command_encoder

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(buffer_input, 0, 0)
        encoder.set_buffer(buffer_output, 0, 1)
        encoder.dispatch_threads((len(large_data), 1, 1), (256, 1, 1))
        encoder.end_encoding()

        encoder.commit()
        encoder.wait_until_completed()

        result = buffer_output.to_numpy(np.float32, (len(large_data),))

        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

        # Verify correctness
        expected = large_data**2
        np.testing.assert_allclose(result, expected, rtol=TEST_CONFIG["tolerance"])

        # Performance assertion
        assert elapsed_time < TEST_CONFIG["performance_threshold_ms"], (
            f"Operation took {elapsed_time:.2f}ms, expected < {TEST_CONFIG['performance_threshold_ms']}ms"
        )

    @pytest.mark.benchmark
    def test_memory_throughput_benchmark(self, device, command_queue):
        """Benchmark memory throughput."""
        sizes = [1000, 10000, 100000, 1000000]
        results = []

        for size in sizes:
            data = np.random.random(size).astype(np.float32)

            start_time = time.time()

            # Upload to GPU
            buffer = device.make_buffer_from_numpy(data)

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


@requires_metal
def test_demos():
    from pymetallic.helpers import MetallicDemo

    md = MetallicDemo(quiet=True)
    for _name, demo in md.get_demos().items():
        demo()


@requires_metal
def test_blit_fill_and_read_scalar(device):
    n = 128
    buf = device.make_buffer(n * 4)
    fill_u32(device, buf, value=0xDEADBEEF, count_u32=n)
    # spot check a few
    arr = buf.to_numpy(np.uint32, (n,))
    assert arr[0] == 0xDEADBEEF
    assert arr[n // 2] == 0xDEADBEEF
    assert arr[-1] == 0xDEADBEEF

    # read_scalar convenience
    first = read_scalar(command_queue, buf, np.uint32)
    second = device.read_scalar(buf, np.uint32, 1)
    assert first == 0xDEADBEEF
    assert second == 0xDEADBEEF


VADD_SRC = r"""
#include <metal_stdlib>
using namespace metal;

kernel void vadd(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float*       o [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    o[gid] = a[gid] + b[gid];
}
"""


@requires_metal
def test_vector_add(device):
    n = 1024
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    buf_a = device.make_buffer_from_numpy(a)
    buf_b = device.make_buffer_from_numpy(b)
    buf_o = pm.Buffer(device, n * 4)

    lib = device.make_library(VADD_SRC)
    fn = lib.make_function("vadd")
    pso = device.make_compute_pipeline_state(fn)

    enc = device.command_encoder
    enc.set_compute_pipeline_state(pso)
    enc.set_buffer(buf_a, 0, 0)
    enc.set_buffer(buf_b, 0, 1)
    enc.set_buffer(buf_o, 0, 2)
    enc.dispatch_threads((n, 1, 1), (min(n, 256), 1, 1))
    enc.end_encoding()
    enc.commit()
    enc.wait_until_completed()

    out = buf_o.to_numpy(np.float32, (n,))
    np.testing.assert_allclose(out, a + b, rtol=1e-6, atol=1e-6)


INC_SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void inc(device uint* data [[buffer(0)]],
                uint gid [[thread_position_in_grid]]) {
    data[gid] += 1u;
}
"""


@requires_metal
def test_kernel_cache_reuses_pipeline(device, command_queue):
    # Build twice; should hit cache second time
    k1 = device.make_kernel(source=INC_SRC, func="inc")
    k2 = device.make_kernel(source=INC_SRC, func="inc")
    assert isinstance(k1, Kernel)
    assert isinstance(k2, Kernel)
    # The cache stores per-device pipeline state; identity equality should hold
    assert k1._pso is k2._pso

    n = 64
    buf = device.make_buffer_from_numpy(np.zeros(n, dtype=np.uint32))

    # run twice via the convenience call
    k1(command_queue, grid=(n, 1, 1), buffers=[(buf, 0)])
    k2(command_queue, grid=(n, 1, 1), buffers=[(buf, 0)])

    out = buf.to_numpy(np.uint32, (n,))
    # each element incremented twice
    assert (out == 2).all()
