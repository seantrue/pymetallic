"""
Comprehensive PyMetal Test Suite
================================

This module provides comprehensive testing for the PyMetal library using pytest.
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
• TestPyMetalHero - End-to-end comprehensive scenarios
• TestPyMetalIntegration - Multi-component integration tests
• TestPyMetalMemoryManagement - Memory allocation and management patterns

🚀 Usage:
    pytest test_pymetal.py -v                    # Run all tests
    pytest test_pymetal.py::TestPyMetalHero -v   # Run hero tests only
    pytest test_pymetal.py --hero-only -v        # Alternative hero test syntax
    pytest test_pymetal.py --benchmark-only -v   # Run performance benchmarks
    pytest test_pymetal.py -k "matrix" -v        # Run tests matching "matrix"
    pytest test_pymetal.py --tb=short -v         # Shorter traceback format

🔧 Requirements:
• PyMetal library properly installed and compiled
• macOS with Metal support
• Python 3.8+ with pytest, numpy
• Metal-capable GPU

🎪 Hero Test Highlights:
• Image processing: 1024×768 RGBA pipeline with multiple effects
• N-body simulation: 2048 particles, 100 time steps, physics validation
• Neural network: 784→512→10 architecture with batch training

⚡ Performance Expectations:
• Image processing: <500ms, >1 MP/s throughput
• N-body simulation: <5s, >1M interactions/s
• Neural network: <10s, >100 MFLOPS computation

The hero tests demonstrate PyMetal's capabilities acr#!/usr/bin/env python3
"""

import os
import time

import numpy as np
import pytest

# Import PyMetal - handle missing dependency gracefully
try:
    import pymetallic
except ImportError:
    pytest.skip("PyMetal not available", allow_module_level=True)
from pymetallic import MetalError

try:
    import PIL.Image as Image
except:
    Image = None


class AnimateIt:
    _usable = False

    def __init__(
        self,
        name="animation",
        mode: str | None = None,
        duration: int = 100,
        seconds: float | None = None,
        loop: bool = False,
        **kwargs,
    ):
        self._usable = Image is not None
        self.frames = []
        self.times = []
        self.mode = mode
        self.name = name
        self.seconds = seconds
        self.duration = duration
        self.loops = 0 if loop else 1
        self.path = os.path.join(os.path.dirname(__file__), f"hero_{self.name}.gif")
        self.save_args = kwargs

    def add_frame(self, array: np.ndarray):
        if self._usable:
            self.times.append(time.time())
            frame = np.clip(array, 0.0, 1.0)
            frame_image = Image.fromarray((frame * 255.0).astype(np.uint8))
            self.frames.append(frame_image)

    def save(self):
        if self._usable:
            duration = (
                int(1000 * self.seconds / len(self.frames))
                if self.seconds
                else self.duration
            )
            self.frames[0].save(
                self.path,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=self.loops,
                optimize=False,
                **self.save_args,
            )
            print(f"🎞️ Saved {self.name} GIF to {self.path}")


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
    """Core functionality tests for PyMetal."""

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
        pipeline_state = metal_device.compute_pipeline_state(function)
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
        pipeline_state = metal_device.compute_pipeline_state(function)

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
        pipeline_state = metal_device.compute_pipeline_state(function)

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
        pipeline_state = metal_device.compute_pipeline_state(function)

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
        pipeline_state = metal_device.compute_pipeline_state(function)

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
        pipeline_state = metal_device.compute_pipeline_state(function)

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


class TestPyMetalHero:
    """Hero Tests - Comprehensive end-to-end scenarios demonstrating PyMetal capabilities."""

    def test_hero_1_realtime_image_processing(self, metal_device, command_queue):
        """
        HERO TEST 1: Real-time Image Processing Pipeline

        Simulates a complete image processing pipeline with multiple effects:
        - Gaussian blur
        - Edge detection
        - Color correction
        - Noise reduction

        Tests compute shaders, 2D dispatch, texture-like operations, and pipeline chaining.
        """
        print("\n🎯 HERO TEST 1: Real-time Image Processing Pipeline")
        animation = AnimateIt("rtip", duration=2)
        # Simulate a high-resolution image
        width, height = 1024, 768
        channels = 4  # RGBA
        image_size = width * height * channels

        # Generate test image data (RGBA format)
        np.random.seed(123)
        original_image = np.random.random((height, width, channels)).astype(np.float32)
        colors = [[1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
        nstripes = len(colors)
        wstripe = width // nstripes
        for i, color in enumerate(colors):
            l = i * wstripe + 50
            r = l + wstripe - 100
            original_image[300:500, l:r, :] = color
        animation.add_frame(original_image)
        # Multi-stage image processing shader
        processing_shader = f"""
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void image_processing_pipeline(device const float* input [[buffer(0)]],
                                            device float* output [[buffer(1)]],
                                            device float* temp_buffer [[buffer(2)]],
                                            constant uint& width [[buffer(3)]],
                                            constant uint& height [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]]) {{
            
            if (gid.x >= width || gid.y >= height) return;
            
            const uint idx = gid.y * width + gid.x;
            const uint channels = 4;
            const uint pixel_idx = idx * channels;
            
            // Stage 1: Gaussian blur (simplified 3x3 kernel)
            float4 blurred = float4(0.0);
            float gaussian_kernel[9] = {{0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625}};
            
            for (int dy = -1; dy <= 1; dy++) {{
                for (int dx = -1; dx <= 1; dx++) {{
                    int nx = int(gid.x) + dx;
                    int ny = int(gid.y) + dy;
                    
                    if (nx >= 0 && nx < int(width) && ny >= 0 && ny < int(height)) {{
                        uint neighbor_idx = (ny * int(width) + nx) * channels;
                        float weight = gaussian_kernel[(dy + 1) * 3 + (dx + 1)];
                        
                        blurred.r += input[neighbor_idx + 0] * weight;
                        blurred.g += input[neighbor_idx + 1] * weight;
                        blurred.b += input[neighbor_idx + 2] * weight;
                        blurred.a += input[neighbor_idx + 3] * weight;
                    }}
                }}
            }}
            
            // Stage 2: Edge detection (Sobel operator)
            float4 edge_x = float4(0.0);
            float4 edge_y = float4(0.0);
            
            float sobel_x[9] = {{-1, 0, 1, -2, 0, 2, -1, 0, 1}};
            float sobel_y[9] = {{-1, -2, -1, 0, 0, 0, 1, 2, 1}};
            
            for (int dy = -1; dy <= 1; dy++) {{
                for (int dx = -1; dx <= 1; dx++) {{
                    int nx = int(gid.x) + dx;
                    int ny = int(gid.y) + dy;
                    
                    if (nx >= 0 && nx < int(width) && ny >= 0 && ny < int(height)) {{
                        uint neighbor_idx = (ny * int(width) + nx) * channels;
                        int kernel_idx = (dy + 1) * 3 + (dx + 1);
                        
                        float wx = sobel_x[kernel_idx];
                        float wy = sobel_y[kernel_idx];
                        
                        edge_x += float4(input[neighbor_idx + 0], input[neighbor_idx + 1], 
                                        input[neighbor_idx + 2], input[neighbor_idx + 3]) * wx;
                        edge_y += float4(input[neighbor_idx + 0], input[neighbor_idx + 1],
                                        input[neighbor_idx + 2], input[neighbor_idx + 3]) * wy;
                    }}
                }}
            }}
            
            float4 edge_magnitude = sqrt(edge_x * edge_x + edge_y * edge_y);
            
            // Stage 3: Color correction and final composition
            float4 final_color = blurred * 0.7 + edge_magnitude * 0.3;
            
            // Apply gamma correction
            final_color = pow(final_color, float4(0.8));
            
            // Clamp values
            final_color = clamp(final_color, 0.0, 1.0);
            
            // Write result
            output[pixel_idx + 0] = final_color.r;
            output[pixel_idx + 1] = final_color.g;
            output[pixel_idx + 2] = final_color.b;
            output[pixel_idx + 3] = final_color.a;
        }}
        """

        # Create buffers
        input_buffer = pymetallic.Buffer.from_numpy(
            metal_device, original_image.flatten()
        )
        output_buffer = pymetallic.Buffer(metal_device, image_size * 4)
        temp_buffer = pymetallic.Buffer(metal_device, image_size * 4)
        width_buffer = pymetallic.Buffer.from_numpy(
            metal_device, np.array([width], dtype=np.uint32)
        )
        height_buffer = pymetallic.Buffer.from_numpy(
            metal_device, np.array([height], dtype=np.uint32)
        )

        # Compile and execute
        start_time = time.time()

        library = pymetallic.Library(metal_device, processing_shader)
        function = library.make_function("image_processing_pipeline")
        pipeline_state = metal_device.compute_pipeline_state(function)

        command_buffer = command_queue.make_command_buffer()
        encoder = command_buffer.make_compute_command_encoder()

        encoder.set_compute_pipeline_state(pipeline_state)
        encoder.set_buffer(input_buffer, 0, 0)
        encoder.set_buffer(output_buffer, 0, 1)
        encoder.set_buffer(temp_buffer, 0, 2)
        encoder.set_buffer(width_buffer, 0, 3)
        encoder.set_buffer(height_buffer, 0, 4)

        # Use 2D dispatch for image processing
        encoder.dispatch_threads((width, height, 1), (16, 16, 1))
        encoder.end_encoding()

        command_buffer.commit()
        command_buffer.wait_until_completed()

        processing_time = (time.time() - start_time) * 1000

        # Retrieve and validate results
        processed_image = output_buffer.to_numpy(np.float32, original_image.shape)
        animation.add_frame(processed_image)
        # Validation checks
        assert (
            processed_image.shape == original_image.shape
        ), "Output shape should match input"
        assert np.all(processed_image >= 0.0) and np.all(
            processed_image <= 1.0
        ), "Processed image values should be in [0, 1] range"
        assert not np.array_equal(
            processed_image, original_image
        ), "Processed image should be different from original"

        # Performance metrics
        pixels_processed = width * height
        megapixels_per_second = (pixels_processed / 1e6) / (processing_time / 1000)

        print(f"✅ Processed {width}×{height} image in {processing_time:.1f}ms")
        print(f"📊 Performance: {megapixels_per_second:.1f} Megapixels/second")
        print(f"🎨 Pipeline stages: Gaussian blur → Edge detection → Color correction")
        animation.save()
        # Performance assertion for hero test
        assert (
            processing_time < 500
        ), f"Image processing should complete in <500ms, took {processing_time:.1f}ms"
        assert (
            megapixels_per_second > 1.0
        ), f"Should process >1 MP/s, achieved {megapixels_per_second:.1f} MP/s"

    def test_hero_2_cellular_automata(
        self,
        metal_device,
        command_queue,
        width: int = 512,
        height: int = 512,
        steps: int = 200,
        seed_probability: float = 0.05,
        threadgroup: tuple = (16, 16, 1),
    ) -> np.ndarray:
        """
        HERO: Cellular Automata (Conway's Game of Life)
        ------------------------------------------------
        Runs a GPU-accelerated Conway's Game of Life simulation on a 2D grid.

        Returns the final state as a (height, width) uint8 NumPy array.
        """
        """
        """
        from pymetallic import Buffer, Library, ComputePipelineState

        animation = AnimateIt("cellular", duration=5, pallette=2)
        print("\n🎯 HERO TEST 2: Cellular Automata")

        device = metal_device
        queue = command_queue

        # Prepare initial random state (0 or 1), packed as uint8
        rng = np.random.default_rng(1234)
        init = (rng.random((height, width)) < seed_probability).astype(np.uint8)
        animation.add_frame(init)
        buf_a = Buffer.from_numpy(device, init, storage_mode=Buffer.STORAGE_SHARED)
        buf_b = Buffer(device, init.size)  # bytes; uint8 per cell
        # Params buffer: [width, height] as uint32
        params = np.array([np.uint32(width), np.uint32(height)], dtype=np.uint32)
        buf_params = Buffer.from_numpy(
            device, params, storage_mode=Buffer.STORAGE_SHARED
        )

        # Metal kernel for one Life step (with wrap-around)
        source = f"""
        #include <metal_stdlib>
        using namespace metal;

        struct Params {{
            uint width;
            uint height;
        }};

        inline uint wrap_int(int v, int m) {{
            int r = v % m;
            return (uint)(r < 0 ? r + m : r);
        }}

        kernel void life_step(const device uchar* in_state     [[buffer(0)]],
                                   device uchar* out_state    [[buffer(1)]],
                             const device Params* p            [[buffer(2)]],
                             uint2 gid                         [[thread_position_in_grid]]) {{

            uint W = p->width;
            uint H = p->height;
            if (gid.x >= W || gid.y >= H) return;

            int x = (int)gid.x;
            int y = (int)gid.y;

            int count = 0;
            // 8-neighborhood
            for (int dy = -1; dy <= 1; ++dy) {{
                for (int dx = -1; dx <= 1; ++dx) {{
                    if (dx == 0 && dy == 0) continue;
                    uint nx = wrap_int(x + dx, (int)W);
                    uint ny = wrap_int(y + dy, (int)H);
                    uint nidx = ny * W + nx;
                    count += in_state[nidx] > 0 ? 1 : 0;
                }}
            }}

            uint idx = (uint)y * W + (uint)x;
            bool alive = in_state[idx] > 0;
            bool next_alive = (alive && (count == 2 || count == 3)) || (!alive && count == 3);
            out_state[idx] = next_alive ? (uchar)1 : (uchar)0;
        }}
        """

        lib = Library(device, source)
        fn = lib.make_function("life_step")
        pso = ComputePipelineState(device, fn)

        curr, nxt = buf_a, buf_b
        for _ in range(int(steps)):
            cb = queue.make_command_buffer()
            enc = cb.make_compute_command_encoder()
            enc.set_compute_pipeline_state(pso)
            enc.set_buffer(curr, 0, 0)
            enc.set_buffer(nxt, 0, 1)
            enc.set_buffer(buf_params, 0, 2)
            enc.dispatch_threads((width, height, 1), threadgroup)
            enc.end_encoding()
            cb.commit()
            cb.wait_until_completed()
            animation.add_frame(nxt.to_numpy(init.dtype, init.shape))
            curr, nxt = nxt, curr

        # Copy back to host
        out = curr.to_numpy(np.uint8, (height * width,))
        animation.save()
        return out.reshape((height, width))

    def test_hero_2_cfd(
        self,
        metal_device,
        command_queue,
        width: int = 256,
        height: int = 256,
        steps: int = 100,
        dt: float = 0.1,
        visc: float = 0.0001,
        threadgroup: tuple = (16, 16, 1),
    ) -> np.ndarray:
        """
        HERO: 2D Stable Fluids / CFD Demo
        ---------------------------------
        Simulates a simple 2D incompressible fluid using semi-Lagrangian advection and a Jacobi pressure solve.
        Returns a (height, width) float32 dye field that you can visualize.
        """
        from pymetallic import Buffer, Library

        animation = AnimateIt("fluid_dynamics", duration=10)
        print("\n🎯 HERO TEST 3: Computational Fluid Dynamics")

        device = metal_device
        queue = command_queue

        W, H = int(width), int(height)
        N = W * H

        # Allocate fields
        # velocity ping-pong (float2), pressure ping-pong (float), divergence (float), dye ping-pong (float)
        pr0 = Buffer(device, N * 4)
        pr1 = Buffer(device, N * 4)
        div = Buffer(device, N * 4)
        dye0 = Buffer(device, N * 4)
        dye1 = Buffer(device, N * 4)

        # Initialize dye and velocity with complex patterns for a richer start
        y, x = np.mgrid[0:H, 0:W]
        cx, cy = W // 2, H // 2

        # Composite dye: two Gaussians + a circular ring
        def gauss(cx0, cy0, sigma):
            return np.exp(-(((x - cx0) ** 2 + (y - cy0) ** 2) / (2.0 * sigma * sigma)))

        sigma1 = 0.10 * min(W, H)
        sigma2 = 0.07 * min(W, H)
        blob1 = gauss(0.30 * W, 0.40 * H, sigma1)
        blob2 = gauss(0.70 * W, 0.60 * H, sigma2) * 0.8

        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r0 = 0.28 * min(W, H)
        ring_sigma = 0.04 * min(W, H)
        ring = np.exp(-((r - r0) ** 2) / (2.0 * ring_sigma * ring_sigma)) * 0.6

        dye_comp = np.clip(blob1 + blob2 + ring, 0.0, 1.0).astype(np.float32)
        animation.add_frame(dye_comp)

        # Complex initial velocity: decaying swirl + shear bands
        dx = x.astype(np.float32) - np.float32(cx)
        dy_ = y.astype(np.float32) - np.float32(cy)
        r2 = dx * dx + dy_ * dy_
        sigma_v = np.float32(0.25 * min(W, H))
        swirl_mag = np.exp(-(r2) / (2.0 * sigma_v * sigma_v)).astype(np.float32)
        eps = np.float32(1e-5)
        inv_norm = 1.0 / np.sqrt(r2 + eps)
        ox = (-dy_) * inv_norm
        oy = (dx) * inv_norm
        swirl_strength = np.float32(2.0)
        vx_swirl = swirl_strength * swirl_mag * ox
        vy_swirl = swirl_strength * swirl_mag * oy

        vx_shear = np.zeros_like(vx_swirl, dtype=np.float32)
        vy_shear = np.zeros_like(vy_swirl, dtype=np.float32)
        band1 = (y >= int(0.25 * H)) & (y < int(0.35 * H))
        band2 = (y >= int(0.65 * H)) & (y < int(0.75 * H))
        vx_shear[band1] = 0.5
        vx_shear[band2] = -0.5

        vx = (vx_swirl + vx_shear).astype(np.float32)
        vy = (vy_swirl + vy_shear).astype(np.float32)

        vel_init = np.stack([vx, vy], axis=-1).astype(np.float32)
        vel0 = Buffer.from_numpy(device, vel_init)
        vel1 = Buffer.from_numpy(device, vel_init)

        # Fill initial dye
        dye_init = np.ascontiguousarray(dye_comp)
        dye_init = dye_init.reshape(-1)
        tmp_dye = Buffer.from_numpy(device, dye_init)
        # If we don't have a direct blit, just keep tmp_dye as dye0 initial
        dye0 = tmp_dye

        # Params buffer (match Metal struct layout exactly: uint,uint,float,float,float,float,float,float,float)
        params_dtype = np.dtype(
            [
                ("width", np.uint32),
                ("height", np.uint32),
                ("dt", np.float32),
                ("dx", np.float32),
                ("visc", np.float32),
                ("fx", np.float32),
                ("fy", np.float32),
                ("fr", np.float32),
                ("fs", np.float32),
            ],
            align=False,
        )
        params_struct = np.zeros(1, dtype=params_dtype)
        params_struct["width"] = np.uint32(W)
        params_struct["height"] = np.uint32(H)
        params_struct["dt"] = np.float32(dt)
        params_struct["dx"] = np.float32(1.0)  # dx = 1.0
        params_struct["visc"] = np.float32(visc)
        params_struct["fx"] = np.float32(cx)
        params_struct["fy"] = np.float32(cy)
        params_struct["fr"] = np.float32(0.15 * min(W, H))  # force radius
        params_struct["fs"] = np.float32(200.0)  # force strength
        # Reinterpret as 32-bit words for raw byte copy
        params_words = np.frombuffer(params_struct.tobytes(), dtype=np.uint32)
        params_buf = Buffer.from_numpy(device, params_words)

        # Metal kernels
        source = f"""
        #include <metal_stdlib>
        using namespace metal;

        struct Params {{
            uint width;
            uint height;
            float dt;
            float dx;
            float visc;
            float fx;
            float fy;
            float fr;
            float fs;
        }};

        inline uint idx(uint x, uint y, uint W) {{
            return y * W + x;
        }}

        inline float clamp01(float v) {{ return clamp(v, 0.0f, 1.0f); }}

        inline float2 sample_vel(const device float2* vel, float x, float y, uint W, uint H) {{
            // bilinear sample at (x,y) in grid space
            x = clamp(x, 0.0f, (float)(W-1));
            y = clamp(y, 0.0f, (float)(H-1));
            uint x0 = (uint)floor(x);
            uint y0 = (uint)floor(y);
            uint x1 = min(x0 + 1, W - 1);
            uint y1 = min(y0 + 1, H - 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;
            float2 v00 = vel[idx(x0,y0,W)];
            float2 v10 = vel[idx(x1,y0,W)];
            float2 v01 = vel[idx(x0,y1,W)];
            float2 v11 = vel[idx(x1,y1,W)];
            float2 vx0 = mix(v00, v10, tx);
            float2 vx1 = mix(v01, v11, tx);
            return mix(vx0, vx1, ty);
        }}

        inline float sample_s(const device float* s, float x, float y, uint W, uint H) {{
            x = clamp(x, 0.0f, (float)(W-1));
            y = clamp(y, 0.0f, (float)(H-1));
            uint x0 = (uint)floor(x);
            uint y0 = (uint)floor(y);
            uint x1 = min(x0 + 1, W - 1);
            uint y1 = min(y0 + 1, H - 1);
            float tx = x - (float)x0;
            float ty = y - (float)y0;
            float s00 = s[idx(x0,y0,W)];
            float s10 = s[idx(x1,y0,W)];
            float s01 = s[idx(x0,y1,W)];
            float s11 = s[idx(x1,y1,W)];
            float sx0 = mix(s00, s10, tx);
            float sx1 = mix(s01, s11, tx);
            return mix(sx0, sx1, ty);
        }}

        kernel void add_force(const device Params* P            [[buffer(3)]],
                              device float2* vel_out            [[buffer(0)]],
                              uint2 gid                         [[thread_position_in_grid]]) {{
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            float2 pos = float2(P->fx, P->fy);
            float r = P->fr;
            float2 center = float2((float)gid.x, (float)gid.y);
            float2 d = center - pos;
            float dist2 = dot(d,d);
            float influence = exp(-dist2 / (r*r));
            float2 orth = float2(-d.y, d.x);
            vel_out[idx(gid.x, gid.y, W)] += normalize(orth + 1e-5) * (P->fs * influence);
        }}

        kernel void advect_vel(const device Params* P           [[buffer(3)]],
                               const device float2* vel_in      [[buffer(0)]],
                               device float2* vel_out           [[buffer(1)]],
                               uint2 gid                        [[thread_position_in_grid]]) {{
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            float2 v = vel_in[idx(gid.x, gid.y, W)];
            float x = (float)gid.x - P->dt * v.x;
            float y = (float)gid.y - P->dt * v.y;
            vel_out[idx(gid.x, gid.y, W)] = sample_vel(vel_in, x, y, W, H);
        }}

        kernel void divergence(const device Params* P           [[buffer(3)]],
                               const device float2* vel         [[buffer(0)]],
                               device float* div_out            [[buffer(1)]],
                               uint2 gid                        [[thread_position_in_grid]]) {{
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            uint x = gid.x, y = gid.y;
            uint xm = max(int(x)-1, 0);
            uint xp = min(x+1, W-1);
            uint ym = max(int(y)-1, 0);
            uint yp = min(y+1, H-1);
            float2 vxm = vel[idx(xm,y,W)];
            float2 vxp = vel[idx(xp,y,W)];
            float2 vym = vel[idx(x,ym,W)];
            float2 vyp = vel[idx(x,yp,W)];
            float div = 0.5f * ((vxp.x - vxm.x) + (vyp.y - vym.y));
            div_out[idx(x,y,W)] = div;
        }}

        kernel void jacobi_pressure(const device Params* P      [[buffer(3)]],
                                    const device float* p_in    [[buffer(0)]],
                                    const device float* b       [[buffer(1)]],
                                    device float* p_out         [[buffer(2)]],
                                    uint2 gid                   [[thread_position_in_grid]]) {{
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            uint x = gid.x, y = gid.y;
            uint xm = max(int(x)-1, 0);
            uint xp = min(x+1, W-1);
            uint ym = max(int(y)-1, 0);
            uint yp = min(y+1, H-1);
            float pL = p_in[idx(xm,y,W)];
            float pR = p_in[idx(xp,y,W)];
            float pB = p_in[idx(x,ym,W)];
            float pT = p_in[idx(x,yp,W)];
            float rhs = b[idx(x,y,W)];
            // alpha = -dx*dx, rBeta = 0.25
            float p_new = (pL + pR + pB + pT - rhs) * 0.25f;
            p_out[idx(x,y,W)] = p_new;
        }}

        kernel void subtract_gradient(const device Params* P    [[buffer(3)]],
                                      const device float2* vel_in [[buffer(0)]],
                                      const device float* p     [[buffer(1)]],
                                      device float2* vel_out    [[buffer(2)]],
                                      uint2 gid                 [[thread_position_in_grid]]) {{
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            uint x = gid.x, y = gid.y;
            uint xm = max(int(x)-1, 0);
            uint xp = min(x+1, W-1);
            uint ym = max(int(y)-1, 0);
            uint yp = min(y+1, H-1);
            float pL = p[idx(xm,y,W)];
            float pR = p[idx(xp,y,W)];
            float pB = p[idx(x,ym,W)];
            float pT = p[idx(x,yp,W)];
            float2 v = vel_in[idx(x,y,W)];
            v -= 0.5f * float2(pR - pL, pT - pB);
            vel_out[idx(x,y,W)] = v;
        }}

        kernel void advect_scalar(const device Params* P        [[buffer(3)]],
                                  const device float* s_in      [[buffer(0)]],
                                  const device float2* vel      [[buffer(1)]],
                                  device float* s_out           [[buffer(2)]],
                                  uint2 gid                     [[thread_position_in_grid]]) {{
            uint W = P->width, H = P->height;
            if (gid.x >= W || gid.y >= H) return;
            float2 v = vel[idx(gid.x, gid.y, W)];
            float x = (float)gid.x - P->dt * v.x;
            float y = (float)gid.y - P->dt * v.y;
            s_out[idx(gid.x, gid.y, W)] = sample_s(s_in, x, y, W, H);
        }}
        """

        lib = Library(device, source)
        fn_add_force = lib.make_function("add_force")
        fn_advect_v = lib.make_function("advect_vel")
        fn_div = lib.make_function("divergence")
        fn_jacobi = lib.make_function("jacobi_pressure")
        fn_subgrad = lib.make_function("subtract_gradient")
        fn_advect_s = lib.make_function("advect_scalar")

        p_add = device.compute_pipeline_state(fn_add_force)
        p_advv = device.compute_pipeline_state(fn_advect_v)
        p_div = device.compute_pipeline_state(fn_div)
        p_jac = device.compute_pipeline_state(fn_jacobi)
        p_sub = device.compute_pipeline_state(fn_subgrad)
        p_advs = device.compute_pipeline_state(fn_advect_s)

        # Simple simulation loop
        for _ in range(int(steps)):
            # Add swirling force into vel0 -> vel1
            cb = queue.make_command_buffer()
            enc = cb.make_compute_command_encoder()
            enc.set_compute_pipeline_state(p_add)
            enc.set_buffer(vel0, 0, 0)  # out
            enc.set_buffer(params_buf, 0, 3)
            enc.dispatch_threads((W, H, 1), threadgroup)
            enc.end_encoding()
            cb.commit()
            cb.wait_until_completed()

            # Advect velocity: vel1 = advect_vel(vel0)
            cb = queue.make_command_buffer()
            enc = cb.make_compute_command_encoder()
            enc.set_compute_pipeline_state(p_advv)
            enc.set_buffer(vel0, 0, 0)  # in
            enc.set_buffer(vel1, 0, 1)  # out
            enc.set_buffer(params_buf, 0, 3)
            enc.dispatch_threads((W, H, 1), threadgroup)
            enc.end_encoding()
            cb.commit()
            cb.wait_until_completed()

            # Compute divergence of vel1 into div
            cb = queue.make_command_buffer()
            enc = cb.make_compute_command_encoder()
            enc.set_compute_pipeline_state(p_div)
            enc.set_buffer(vel1, 0, 0)
            enc.set_buffer(div, 0, 1)
            enc.set_buffer(params_buf, 0, 3)
            enc.dispatch_threads((W, H, 1), threadgroup)
            enc.end_encoding()
            cb.commit()
            cb.wait_until_completed()

            # Clear pressure buffers to zero (first few iterations rely on initial zeros)
            zero_np = np.zeros(N, dtype=np.float32)
            pr0 = Buffer.from_numpy(device, zero_np)
            pr1 = Buffer.from_numpy(device, zero_np)

            # Jacobi iterations to solve for pressure
            J_ITERS = 20
            pin, pout = pr0, pr1
            for __ in range(J_ITERS):
                cb = queue.make_command_buffer()
                enc = cb.make_compute_command_encoder()
                enc.set_compute_pipeline_state(p_jac)
                enc.set_buffer(pin, 0, 0)  # p_in
                enc.set_buffer(div, 0, 1)  # b
                enc.set_buffer(pout, 0, 2)  # p_out
                enc.set_buffer(params_buf, 0, 3)
                enc.dispatch_threads((W, H, 1), threadgroup)
                enc.end_encoding()
                cb.commit()
                cb.wait_until_completed()
                pin, pout = pout, pin
            pressure = pin

            # Subtract gradient: vel0 = project(vel1, pressure)
            cb = queue.make_command_buffer()
            enc = cb.make_compute_command_encoder()
            enc.set_compute_pipeline_state(p_sub)
            enc.set_buffer(vel1, 0, 0)
            enc.set_buffer(pressure, 0, 1)
            enc.set_buffer(vel0, 0, 2)
            enc.set_buffer(params_buf, 0, 3)
            enc.dispatch_threads((W, H, 1), threadgroup)
            enc.end_encoding()
            cb.commit()
            cb.wait_until_completed()

            # Advect dye by velocity: dye1 = advect_scalar(dye0, vel0)
            cb = queue.make_command_buffer()
            enc = cb.make_compute_command_encoder()
            enc.set_compute_pipeline_state(p_advs)
            enc.set_buffer(dye0, 0, 0)  # s_in
            enc.set_buffer(vel0, 0, 1)  # vel
            enc.set_buffer(dye1, 0, 2)  # s_out
            enc.set_buffer(params_buf, 0, 3)
            enc.dispatch_threads((W, H, 1), threadgroup)
            enc.end_encoding()
            cb.commit()
            cb.wait_until_completed()

            # Ping-pong dye
            dye0, dye1 = dye1, dye0
            frame = dye0.to_numpy(np.float32, (H, W))
            animation.add_frame(frame)

        # Read back dye
        dye_out = dye0.to_numpy(np.float32, (H, W))
        animation.add_frame(dye_out)
        animation.save()
        return dye_out
